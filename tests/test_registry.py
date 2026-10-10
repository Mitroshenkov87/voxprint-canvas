"""Registry rules: permissive licences, no forbidden models, matching diffusers pin."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from canvas.models.loader import ensure_weights, load_catalog_stack
from canvas.models.registry import RegistryError, load_catalog
from canvas.safety.gate import SafetyUnavailable


def _rewrite(tmp_path: Path, mutate) -> Path:
    raw = json.loads(Path("canvas/models/catalog.json").read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_catalog_loads_and_rejects_a_bad_pin_or_licence(tmp_path: Path):
    catalog = load_catalog()
    assert catalog.by_id("z-image-turbo-bf16-16gb").steps == 8
    assert catalog.by_id("z-image-turbo-bf16-16gb").quant == "bf16"
    ids = {spec.id for spec in catalog.models}
    assert "z-image-turbo-fp8" not in ids
    assert "z-image-turbo-fun-controlnet-union-fp8" not in ids
    with pytest.raises(RegistryError):
        load_catalog(_rewrite(tmp_path, lambda raw: raw.__setitem__("diffusers", "0.32.0")))
    with pytest.raises(RegistryError):
        load_catalog(_rewrite(tmp_path, lambda raw: raw["models"][0].__setitem__("licence", "GPL-3.0")))
    with pytest.raises(RegistryError):
        load_catalog(_rewrite(tmp_path, lambda raw: raw["models"][0].__setitem__("repo", "notai/nudenet")))
    with pytest.raises(RegistryError):
        load_catalog(_rewrite(tmp_path, lambda raw: raw["models"][0]["sources"].append({"kind": "http", "uri": "https://example.invalid"})))


def test_z_image_fp8_is_rejected_and_edit_fp8_keeps_a_pending_gate(tmp_path: Path):
    catalog = load_catalog()
    ids = {spec.id for spec in catalog.models}
    assert "qwen-image-2512-fp8" not in ids
    assert "qwen-image-2512-int4" not in ids
    assert catalog.by_id("qwen-image-edit-2511-fp8").quality_gate == "pending"

    def as_fp8(raw: dict) -> None:
        raw["models"][0]["quant"] = "fp8-e4m3"
        raw["models"][0]["quality_gate"] = "passed"

    with pytest.raises(RegistryError, match="BF16"):
        load_catalog(_rewrite(tmp_path, as_fp8))

    def drop_gate(raw: dict) -> None:
        for row in raw["models"]:
            if row["id"] == "qwen-image-edit-2511-fp8":
                row["quality_gate"] = ""

    with pytest.raises(RegistryError, match="quality gate"):
        load_catalog(_rewrite(tmp_path, drop_gate))


def test_unpinned_weights_fail_closed(tmp_path: Path):
    spec = load_catalog().by_id("qwen3guard-gen-0.6b")
    with pytest.raises(SafetyUnavailable) as caught:
        ensure_weights(spec, tmp_path)
    assert caught.value.code == "missing_model"
    with pytest.raises(SafetyUnavailable) as caught:
        load_catalog_stack(tmp_path)
    assert caught.value.code == "missing_model"


def test_hash_mismatch_and_a_present_file(tmp_path: Path):
    spec = load_catalog().by_id("qwen3guard-gen-0.6b")
    payload = b"weights"
    digest = hashlib.sha256(payload).hexdigest()
    pinned = replace(spec, files={"weights": digest})
    folder = tmp_path / spec.repo.replace("/", "--")
    folder.mkdir()
    target = folder / "weights"
    target.write_bytes(b"other")
    with pytest.raises(SafetyUnavailable) as caught:
        ensure_weights(pinned, tmp_path)
    assert caught.value.code == "hash_mismatch"
    target.write_bytes(payload)
    ensure_weights(pinned, tmp_path)
