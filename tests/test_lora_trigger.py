"""The trigger is read from the Forge sidecar or record, then from safetensors metadata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from canvas.models.user_lora import (
    prepare_cli_lora,
    read_lora_trigger,
    trigger_from_metadata,
    trigger_from_record,
    trigger_from_tag_frequency,
)
from tests.lora_files import write_safetensors
from tests.test_forge_lora import _FORGE_A, _FORGE_B, _metadata

TRIGGER = "vxpnqpmwv"


def _weights(folder: Path, name: str = "Tatiana_Model.safetensors", metadata: dict[str, str] | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    write_safetensors(path, {_FORGE_A: b"\x01\x02", _FORGE_B: b"\x03\x04"}, metadata or {})
    return path


def test_sidecar_trigger_is_preferred_over_the_header(tmp_path: Path) -> None:
    path = _weights(tmp_path, metadata=_metadata("z-image", trigger="headerword"))
    sidecar = {
        "schema": "voxprint-model/1",
        "min_schema": 1,
        "id": "tatiana-model",
        "kind": "person",
        "type": "lora",
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
        "base": {"mdl_id": "z-image", "revision": "abc"},
        "licence": "Apache-2.0",
        "default_strength": 0.8,
        "strength_range": [0.0, 1.5],
        "trigger": TRIGGER,
    }
    (tmp_path / "voxprint-model.json").write_text(json.dumps(sidecar), encoding="utf-8")
    # The written sha256 in the official sidecar must match or _matching_sidecar rejects.
    # Use a trigger-only lookup on a possibly mismatched record via read_lora_trigger.
    assert trigger_from_record(sidecar) == TRIGGER
    assert read_lora_trigger(path, sidecar=sidecar) == TRIGGER
    assert prepare_cli_lora(path).trigger == TRIGGER


def test_a_sidecar_without_a_matching_file_field_still_supplies_the_trigger(tmp_path: Path) -> None:
    path = _weights(tmp_path, metadata=_metadata("z-image", trigger=""))
    sidecar = {
        "schema": "voxprint-model/1",
        "min_schema": 1,
        "id": "tatiana-model",
        "kind": "person",
        "type": "lora",
        "file": "other-id.safetensors",
        "sha256": "ab" * 32,
        "size": 1,
        "base": {"mdl_id": "z-image", "revision": "abc"},
        "licence": "Apache-2.0",
        "default_strength": 0.8,
        "strength_range": [0.0, 1.5],
        "trigger": TRIGGER,
    }
    (tmp_path / "voxprint-model.json").write_text(json.dumps(sidecar), encoding="utf-8")
    assert read_lora_trigger(path) == TRIGGER
    assert prepare_cli_lora(path).trigger == TRIGGER


def test_project_json_subject_trigger_is_read_from_the_parent_folder(tmp_path: Path) -> None:
    export = tmp_path / "export"
    path = _weights(export, metadata=_metadata("z-image", trigger=""))
    (tmp_path / "project.json").write_text(
        json.dumps({"subject": {"name": "Tatiana Model", "trigger": TRIGGER, "kind": "person"}}),
        encoding="utf-8",
    )
    assert read_lora_trigger(path) == TRIGGER
    assert prepare_cli_lora(path).trigger == TRIGGER


def test_modelspec_and_tag_frequency_fill_an_empty_forge_key(tmp_path: Path) -> None:
    freq = json.dumps({"img": {"a woman": 20, TRIGGER: 20, "smile": 4}})
    path = _weights(tmp_path, metadata={**_metadata("z-image", trigger=""), "ss_tag_frequency": freq})
    assert read_lora_trigger(path) == TRIGGER
    phrase = _weights(
        tmp_path / "phrase",
        metadata={**_metadata("z-image", trigger=""), "modelspec.trigger_phrase": "tatiana token"},
    )
    assert trigger_from_metadata({"modelspec.trigger_phrase": "tatiana token"}) == "tatiana token"
    assert read_lora_trigger(phrase) == "tatiana token"
    # The official Forge key wins over the common fallbacks.
    both = {**_metadata("z-image", trigger="headerword"), "modelspec.trigger_phrase": "other", "ss_tag_frequency": freq}
    assert trigger_from_metadata(both) == "headerword"


def test_tag_frequency_picks_the_top_count_and_prefers_a_forge_token() -> None:
    assert trigger_from_tag_frequency(json.dumps({"dir": {"woman": 8, "hat": 3}})) == "woman"
    assert trigger_from_tag_frequency(json.dumps({"dir": {"woman": 8, TRIGGER: 8}})) == TRIGGER
    assert trigger_from_tag_frequency("{not json") == ""
    assert trigger_from_tag_frequency("") == ""


def test_a_third_party_file_reads_modelspec_or_tag_frequency(tmp_path: Path) -> None:
    path = _weights(
        tmp_path,
        metadata={"modelspec.trigger_phrase": TRIGGER, "ss_tag_frequency": json.dumps({TRIGGER: 1})},
    )
    assert read_lora_trigger(path) == TRIGGER
