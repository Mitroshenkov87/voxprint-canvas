"""Forge-verified LoRA records are metadata: verification is fail-closed, the gate ignores it.

Owner rule 2026-11-16: no classifier runs on generated pictures with any LoRA
or none; ``review_or_skip`` is a deprecated alias of ``release_generated``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canvas.lora_verify import (
    ALGORITHM,
    SCHEMA_ID,
    file_sha256,
    output_check_decision,
    record_path_for,
    review_or_skip,
    sign_record,
    verify_lora,
)
from canvas.safety.gate import GenerationBlocked
from canvas.safety.image import RawImage, SafeImage

from tests.fakes import World, rgb


@pytest.fixture(autouse=True)
def _suite_home(tmp_path, monkeypatch):
    """Point the shared suite folder (and thus the signing key) at a temp dir."""
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "suite"))
    return tmp_path


def _weights(tmp_path: Path, name: str = "step-001600.safetensors", body: bytes = b"lora-weights-bytes") -> Path:
    path = tmp_path / name
    path.write_bytes(body)
    return path


def _write_record(weights: Path, **overrides: object) -> dict:
    """Write a valid signed record next to ``weights``; overrides replace fields."""
    record = {
        "schema": SCHEMA_ID,
        "record_id": "forge-test-0001",
        "lora_file": weights.name,
        "lora_sha256": file_sha256(weights),
        "project_id": "zhenya-belaya-20261114-2202",
        "dataset_image_count": 42,
        "age_gate_version": "0.1.0-rc.45",
        "verified": True,
        "result": "pass",
        "verified_at": "2026-11-15T00:30:00+03:00",
        "algorithm": ALGORITHM,
    }
    record.update(overrides)
    record["signature"] = sign_record(record)
    record_path_for(weights).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


class RecordingGate:
    """A stand-in gate that records any output-check call. None may happen."""

    def __init__(self) -> None:
        self.reviewed: list[str] = []

    def review_image(self, raw: RawImage, prompt: str) -> SafeImage:
        self.reviewed.append(prompt)
        raise AssertionError("no output check may run on a generated picture")


def _released_without_check(lora_files: list[Path]) -> None:
    """``review_or_skip`` releases through the gate's issue() and wipes, for any LoRA record state."""
    gate = RecordingGate()
    raw = rgb()
    safe = review_or_skip(gate, raw, "p", lora_files)
    assert type(safe) is SafeImage
    assert gate.reviewed == []
    assert raw.width == 0


def test_valid_record_is_verified_and_released_without_check(tmp_path):
    weights = _weights(tmp_path)
    _write_record(weights)
    result = verify_lora(weights)
    assert result.verified and result.reason == "verified"
    assert output_check_decision([weights]).skip is True

    gate = RecordingGate()
    raw = rgb()
    safe = review_or_skip(gate, raw, "photo of vxpctgpsk woman, portrait", [weights])
    assert type(safe) is SafeImage          # minted through the gate's issue()
    assert gate.reviewed == []              # no output check
    assert raw.width == 0                   # the raw buffer was wiped like review_image does


def test_tampered_file_is_unverified_and_still_released_without_check(tmp_path):
    weights = _weights(tmp_path)
    _write_record(weights)
    weights.write_bytes(b"different-weights-after-signing")  # hash no longer matches
    assert verify_lora(weights).reason == "hash-mismatch"
    assert output_check_decision([weights]).skip is False

    _released_without_check([weights])      # verification no longer changes the gate


def test_bad_signature_is_unverified(tmp_path):
    weights = _weights(tmp_path)
    record = _write_record(weights)
    record["signature"] = "00" * 64         # valid hex, wrong signature
    record_path_for(weights).write_text(json.dumps(record), encoding="utf-8")
    assert verify_lora(weights).reason == "bad-signature"
    assert output_check_decision([weights]).skip is False


def test_signed_field_edit_breaks_signature(tmp_path):
    weights = _weights(tmp_path)
    record = _write_record(weights)
    record["result"] = "pass"               # keep it a pass, but flip a signed field
    record["dataset_image_count"] = 999999  # not what was signed
    record_path_for(weights).write_text(json.dumps(record), encoding="utf-8")
    assert verify_lora(weights).reason == "bad-signature"


def test_missing_record_is_unverified_and_released_without_check(tmp_path):
    weights = _weights(tmp_path)               # no record written
    assert verify_lora(weights).reason == "record-missing"
    assert output_check_decision([weights]).skip is False

    _released_without_check([weights])


def test_unreadable_record_is_unverified(tmp_path):
    weights = _weights(tmp_path)
    record_path_for(weights).write_text("{ not json", encoding="utf-8")
    assert verify_lora(weights).reason == "record-unreadable"
    assert output_check_decision([weights]).skip is False


def test_not_a_pass_is_unverified(tmp_path):
    weights = _weights(tmp_path)
    _write_record(weights, verified=False, result="fail")
    assert verify_lora(weights).reason == "not-a-pass"
    assert output_check_decision([weights]).skip is False


def test_mixed_stack_is_unverified_and_released_without_check(tmp_path):
    good = _weights(tmp_path, "good.safetensors", b"good-bytes")
    _write_record(good)
    bad = _weights(tmp_path, "bad.safetensors", b"bad-bytes")  # no record
    assert verify_lora(good).verified is True
    assert verify_lora(bad).verified is False
    decision = output_check_decision([good, bad])
    assert decision.skip is False and decision.reason == "unverified-lora"

    _released_without_check([good, bad])


def test_no_lora_is_released_without_check(tmp_path):
    assert output_check_decision([]).skip is False
    _released_without_check([])


def test_prompt_check_still_refuses_with_a_verified_lora(tmp_path):
    """Verification changes nothing in the gate; the prompt filter is untouched."""
    weights = _weights(tmp_path)
    _write_record(weights)
    assert output_check_decision([weights]).skip is True   # verified, metadata only

    world = World()
    world.safety = "unsafe"
    world.categories = ("child-sexual-content",)           # restricted -> refusal_stage blocks
    gate = world.gate()
    with pytest.raises(GenerationBlocked):
        gate.review_prompt("anything")
