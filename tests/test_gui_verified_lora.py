"""The window's result release (worker session) and the worker boundary run no output check.

Owner rule 2026-11-16: Canvas screens the request (prompt and input images),
not the generated pixels. No LoRA, an unverified LoRA and a Forge-verified
LoRA all release the picture the same way, without a classifier.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from canvas.app.shell_gate import ContentCheck
from canvas.lora_verify import ALGORITHM, SCHEMA_ID, file_sha256, record_path_for, sign_record
from canvas.pipeline.worker import release_worker_image
from canvas.safety.gate import SafetyUnavailable
from canvas.safety.image import SafeImage
from canvas.worker_jobs import _lora_files

from tests.fakes import World, rgb

_IMAGE_CALLS = {"nsfw", "image-score", "faces", "ages", "fine"}


@pytest.fixture(autouse=True)
def _suite_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "suite"))
    monkeypatch.setattr("canvas.app.shell_gate.run_off_ui", lambda work: work())


def _signed(tmp_path: Path, name: str = "step-001600.safetensors") -> Path:
    weights = tmp_path / name
    weights.write_bytes(b"lora-weights-" + name.encode())
    record = {
        "schema": SCHEMA_ID, "record_id": f"forge-{name}", "lora_file": name,
        "lora_sha256": file_sha256(weights), "project_id": "p", "dataset_image_count": 3,
        "age_gate_version": "5", "verified": True, "result": "pass",
        "verified_at": "2026-11-15T00:30:00+03:00", "algorithm": ALGORITHM,
    }
    record["signature"] = sign_record(record)
    record_path_for(weights).write_text(json.dumps(record), encoding="utf-8")
    return weights


def _lora(tmp_path: Path, kind: str) -> tuple[Path, ...]:
    if kind == "none":
        return ()
    if kind == "verified":
        return (_signed(tmp_path),)
    plain = tmp_path / "plain.safetensors"
    plain.write_bytes(b"x")
    return (plain,)


def _blocking_world() -> World:
    """A world whose image checks would refuse any picture they saw."""
    world = World()
    world.nsfw = 0.99
    world.image_scores = {label: 1.0 for label in world.image_scores}
    return world


@pytest.mark.parametrize("lora", ["none", "unverified", "verified"])
def test_window_releases_without_output_check(tmp_path, lora):
    world = _blocking_world()
    check = ContentCheck(gate=world.gate())
    check.stage("item", rgb(4, 4))
    assert check.review_result("photo of a woman", "item", lora_files=_lora(tmp_path, lora)) is None
    assert type(check.kept.pop("item")) is SafeImage
    assert not _IMAGE_CALLS & set(world.calls)


def test_window_old_call_shape_releases_without_output_check():
    world = _blocking_world()
    check = ContentCheck(gate=world.gate())
    check.stage("item", rgb(4, 4))
    assert check.review_result("photo", "item") is None
    assert type(check.kept.pop("item")) is SafeImage
    assert world.calls == []


def test_window_refuses_a_missing_picture():
    check = ContentCheck(gate=World().gate())
    assert check.review_result("photo", "never-staged") is not None
    assert check.kept == {}


@pytest.mark.parametrize("lora", ["none", "unverified", "verified"])
def test_worker_boundary_releases_without_output_check(tmp_path, lora):
    world = _blocking_world()
    image = rgb(4, 4)
    payload = {"ok": True, "width": 4, "height": 4, "data_b64": base64.b64encode(image.to_bytes()).decode("ascii")}
    released = release_worker_image(world.gate(), dict(payload), "p", lora_files=_lora(tmp_path, lora))
    assert type(released) is SafeImage
    assert released.data == image.to_bytes()
    assert world.calls == []


def test_worker_boundary_keeps_nothing_from_a_failed_worker():
    with pytest.raises(SafetyUnavailable):
        release_worker_image(World().gate(), {"ok": False, "message": "crashed"}, "p")


def test_lora_files_from_the_selected_lora(tmp_path):
    from types import SimpleNamespace

    assert _lora_files(None) == ()
    assert _lora_files(SimpleNamespace(weights=tmp_path / "a.safetensors")) == (tmp_path / "a.safetensors",)
    assert _lora_files(SimpleNamespace(weights=str(tmp_path / "b.safetensors"))) == (tmp_path / "b.safetensors",)
    assert _lora_files(SimpleNamespace(weights="")) == ()
