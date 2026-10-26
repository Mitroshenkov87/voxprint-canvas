"""CI fetches weights through MDL blobs and a Canvas view, not a legacy copy."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from canvas.infra.paths import models_dir
from voxprint_common.mdl.catalog import PlannedFile

from canvas.models.mdl_fetch import (
    ENCODER_ID,
    ENCODER_PREFIXES,
    encoder_view,
    gate_specs,
    materialize,
    plan_blobs,
)
from canvas.models.registry import load_catalog


@pytest.fixture
def home(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    return tmp_path


def test_gate_plans_are_blob_destinations() -> None:
    specs = gate_specs()
    assert [spec.role for spec in specs] == [
        "prompt-guard", "zero-shot", "face-detector", "age", "nsfw", "age-fine",
    ]
    planned = plan_blobs(specs, Path("/models"))
    assert planned
    assert all(item.destination == Path("/models") / "blobs" / item.sha256 for item in planned)
    assert all("--" not in item.destination.parent.name for item in planned)


def test_encoder_plan_keeps_the_text_encoder_and_skips_the_transformer() -> None:
    spec = load_catalog().by_id(ENCODER_ID)
    planned = plan_blobs((spec,), Path("/models"), ENCODER_PREFIXES)
    assert planned
    assert all(item.name.startswith(ENCODER_PREFIXES) for item in planned)
    assert all(not item.name.startswith("transformer/") for item in planned)
    assert all(item.destination.parent.name == "blobs" for item in planned)


def test_materialize_hardlinks_a_view_and_does_not_create_a_legacy_folder(home, monkeypatch) -> None:
    del home
    spec = load_catalog().by_id("yunet-face")

    def fake(planned: list[PlannedFile]) -> None:
        for item in planned:
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            item.destination.write_bytes(b"onnx")

    monkeypatch.setattr("canvas.models.mdl_fetch._download", fake)
    (folder,) = materialize((spec,))
    root = models_dir()
    name = next(iter(spec.files))
    linked = folder / name
    blob = root / "blobs" / spec.files[name]
    assert linked.is_file()
    assert os.path.samefile(linked, blob)
    assert folder.parent == root / "views" / "canvas"
    assert not (root / spec.repo.replace("/", "--")).exists()


def test_encoder_view_reads_the_blobs_and_keeps_a_legacy_copy(home, monkeypatch) -> None:
    del home
    spec = load_catalog().by_id(ENCODER_ID)
    root = models_dir()
    legacy = root / spec.repo.replace("/", "--") / "text_encoder"
    legacy.mkdir(parents=True)
    (legacy / "config.json").write_bytes(b"plain copy")

    def fake(planned: list[PlannedFile]) -> None:
        assert planned
        assert all(item.name.startswith(ENCODER_PREFIXES) for item in planned)
        for item in planned:
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            item.destination.write_bytes(b"blob")

    monkeypatch.setattr("canvas.models.mdl_fetch._download", fake)
    (folder,) = materialize((spec,), prefixes=ENCODER_PREFIXES, key_suffix="-encoder")
    assert encoder_view(spec, root) == folder
    assert (folder / "text_encoder" / "config.json").read_bytes() == b"blob"
    assert (legacy / "config.json").read_bytes() == b"plain copy"
