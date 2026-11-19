"""FP8 Turbo pack: an install with the old BF16 repack or FP32 upstream blobs keeps using them."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from canvas.app import preflight as pf
from canvas.infra.paths import models_dir
from canvas.models import loader
from canvas.models.registry import load_catalog
from canvas.models.snapshots import fallback_ready, snapshot_ready, snapshot_status

TURBO = "z-image-turbo-bf16-16gb"
LEGACY = "z-image-turbo-bf16-16gb-bf16"
UPSTREAM = f"{TURBO}-upstream"


@pytest.fixture(autouse=True)
def _placeholder_manifest(monkeypatch, tmp_path_factory):
    # The bundled mdl.json pins would otherwise win over the fake blobs below.
    from canvas.models import mdl_registry

    monkeypatch.setattr(mdl_registry, "_BUNDLED", tmp_path_factory.mktemp("nobundle") / "mdl.json")


@pytest.fixture
def root(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    return models_dir()


@pytest.fixture
def queued(monkeypatch) -> list[str]:
    calls: list[str] = []

    def record(model_id: str) -> bool:
        calls.append(model_id)
        return True

    monkeypatch.setattr("canvas.models.snapshots.request_download", record)
    monkeypatch.setattr("canvas.models.loader.request_download", record)
    monkeypatch.setattr("canvas.app.preflight.request_download", record)
    monkeypatch.setattr(pf, "_adopt", lambda _spec: False)
    return calls


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pins(tag: str) -> tuple[dict[str, str], dict[str, bytes]]:
    payloads = {
        "model_index.json": json.dumps({"tag": tag}).encode(),
        "transformer/diffusion_pytorch_model.safetensors": tag.encode() * 512,
    }
    return {path: _digest(data) for path, data in payloads.items()}, payloads


def _write(root: Path, payloads: dict[str, bytes]) -> None:
    for data in payloads.values():
        path = root / "blobs" / _digest(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _spec():
    """The real Turbo row with small fake pins for the FP8 pack, the FP32 copy and the BF16 repack."""
    base = load_catalog().by_id(TURBO)
    fp8, fp8_data = _pins("fp8")
    fp32, fp32_data = _pins("fp32")
    bf16, bf16_data = _pins("bf16")
    spec = replace(base, files=fp8, fallback_files=fp32, legacy_files=((LEGACY, bf16),))
    return spec, {"fp8": fp8_data, "fp32": fp32_data, "bf16": bf16_data}


def _view_key(folder: Path) -> str:
    return folder.name


def test_the_fp8_primary_wins_when_present(root, queued):
    spec, data = _spec()
    for blobs in data.values():
        _write(root, blobs)
    folder = snapshot_ready(spec, root)
    assert folder is not None and TURBO in _view_key(folder)
    assert LEGACY not in _view_key(folder) and "upstream" not in _view_key(folder)
    assert snapshot_status(spec, root) == "ready"
    assert pf._missing((spec,), root, fallback=True) is None
    assert queued == []


def test_only_the_legacy_bf16_repack_is_used_without_a_download(root, queued):
    spec, data = _spec()
    _write(root, data["bf16"])
    folder = snapshot_ready(spec, root)
    assert folder is not None and LEGACY in _view_key(folder)
    assert (folder / "model_index.json").read_bytes() == data["bf16"]["model_index.json"]
    assert fallback_ready(spec, root) == folder
    assert snapshot_status(spec, root) == "ready"
    assert pf._missing((spec,), root, fallback=True) is None
    assert queued == []


def test_only_the_fp32_upstream_copy_is_used(root, queued):
    spec, data = _spec()
    _write(root, data["fp32"])
    folder = snapshot_ready(spec, root)
    assert folder is not None and UPSTREAM in _view_key(folder)
    assert (folder / "model_index.json").read_bytes() == data["fp32"]["model_index.json"]
    assert snapshot_status(spec, root) == "ready"
    assert pf._missing((spec,), root, fallback=True) is None
    assert queued == []


def test_upstream_is_tried_before_the_legacy_repack(root, queued):
    spec, data = _spec()
    _write(root, data["fp32"])
    _write(root, data["bf16"])
    folder = fallback_ready(spec, root)
    assert folder is not None and UPSTREAM in _view_key(folder)


def test_nothing_on_disk_is_not_ready(root, queued):
    spec, _data = _spec()
    assert snapshot_ready(spec, root) is None
    assert fallback_ready(spec, root) is None
    assert snapshot_status(spec, root) == "not_downloaded"
    assert pf._missing((spec,), root, fallback=True) == spec


def test_a_partial_legacy_repack_counts_as_incomplete(root, queued):
    spec, data = _spec()
    first = next(iter(data["bf16"].items()))
    _write(root, dict([first]))
    assert snapshot_ready(spec, root) is None
    assert snapshot_status(spec, root) == "incomplete"


def test_a_corrupt_legacy_blob_is_a_hash_mismatch(root, queued):
    spec, data = _spec()
    _write(root, data["bf16"])
    digest = spec.legacy_files[0][1]["model_index.json"]
    (root / "blobs" / digest).write_bytes(b"x" * len(data["bf16"]["model_index.json"]))
    assert snapshot_ready(spec, root) is None
    assert snapshot_status(spec, root) == "hash_mismatch"


def test_request_fp32_does_not_queue_an_unlisted_fallback(queued):
    spec = load_catalog().by_id(TURBO)
    assert spec.fallback_files and spec.fallback_listed is False
    assert loader.request_fp32(spec) is False
    assert queued == []
    assert loader.request_fp32(replace(spec, fallback_listed=True)) is True
    assert queued == [UPSTREAM]


def test_the_registry_parses_legacy_and_the_listed_flag():
    raw = json.loads(Path("canvas/models/catalog.json").read_text(encoding="utf-8"))
    row = next(item for item in raw["models"] if item["id"] == TURBO)
    spec = load_catalog().by_id(TURBO)
    assert spec.fallback_listed is False
    assert [key for key, _files in spec.legacy_files] == [LEGACY]
    pins = spec.legacy_files[0][1]
    assert pins == {path: meta["sha256"] for path, meta in row["legacy"][0]["files"].items()}
    assert all(len(value) == 64 for value in pins.values())
    others = [s for s in load_catalog().models if s.id != TURBO]
    assert all(s.fallback_listed and not s.legacy_files for s in others)


def test_one_16gb_turbo_row_covers_both_tiers():
    catalog = load_catalog()
    tiers = {spec.tier for spec in catalog.models if spec.text_base == "z-image-turbo"}
    assert tiers == {"16gb"}
    assert catalog.default_text_base == "z-image-turbo"
    assert "z-image-turbo" in catalog.text_bases
