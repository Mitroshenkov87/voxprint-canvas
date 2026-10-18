"""The loader and the MDL downloader resolve the same weights folder."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxprint_common import paths as common_paths
from voxprint_common.mdl import catalog as mdl_catalog

from canvas.infra import manifest
from canvas.infra.paths import models_dir, shared_dir
from canvas.models.provision import _relative_pack


def _download_root() -> Path:
    # plan_files falls back to this function when no models_root is passed.
    return mdl_catalog.models_dir()


@pytest.mark.parametrize("source", ["default", "env", "suite", "legacy"])
def test_loader_and_downloader_agree(monkeypatch, tmp_path: Path, source: str) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("VOXPRINT_HOME", str(home))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    chosen = tmp_path / "weights"
    if source == "env":
        monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(chosen))
    elif source == "suite":
        (home / "state").mkdir(parents=True)
        (home / "state" / "suite.json").write_text(json.dumps({"models_dir": str(chosen)}), encoding="utf-8")
    elif source == "legacy":
        (home / "state").mkdir(parents=True)
        (home / "state" / "models_dir.txt").write_text(str(chosen) + "\n", encoding="utf-8")
    expected = home / "models" if source == "default" else chosen
    assert models_dir() == expected
    assert _download_root() == expected
    assert common_paths.models_dir() == expected
    assert shared_dir() / "models" != expected


def test_a_model_folder_outside_shared_is_refcounted_and_released(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    folder = models_dir() / "Tongyi-MAI--Z-Image-Turbo"
    folder.mkdir(parents=True)
    (folder / "w.bin").write_bytes(b"x")
    blob = models_dir() / "blobs" / ("a" * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"y")
    path = _relative_pack(folder)
    assert Path(path).is_absolute()
    sha = "0" * 64
    manifest.register_component(shared_dir(), "model-test", kind="model", version="1", path=path, sha256=sha, app="canvas")
    with pytest.raises(manifest.ManifestError):
        manifest.register_component(
            shared_dir(), "model-root", kind="model", version="1", path=str(models_dir()), sha256=sha, app="canvas",
        )
    with pytest.raises(manifest.ManifestError):
        manifest.register_component(
            shared_dir(), "model-out", kind="model", version="1", path=str(tmp_path / "elsewhere"), sha256=sha,
            app="canvas",
        )
    manifest.unregister_app(shared_dir(), "canvas")
    assert not folder.exists()
    assert blob.is_file()


def test_release_deletes_a_view_and_keeps_a_nested_blob(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    view = models_dir() / "views" / "canvas" / "demo-0123456789ab"
    view.mkdir(parents=True)
    (view / "model_index.json").write_text("{}", encoding="utf-8")
    nested = models_dir() / "blobs" / "ab" / ("cd" * 32)
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"blob")
    sha = "2" * 64
    manifest.register_component(
        shared_dir(), "model-view", kind="model", version="1", path=view.resolve().as_posix(), sha256=sha, app="canvas",
    )
    manifest.register_component(
        shared_dir(), "model-blob", kind="model", version="1", path=nested.resolve().as_posix(), sha256=sha,
        app="canvas",
    )
    manifest.unregister_app(shared_dir(), "canvas")
    assert not view.exists()
    assert nested.read_bytes() == b"blob"


def test_a_moved_model_follows_its_folder(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    sha = "1" * 64
    manifest.register_component(
        shared_dir(), "model-moved", kind="model", version="1", path="models/Example--Pack", sha256=sha, app="canvas",
    )
    folder = models_dir() / "Example--Pack"
    folder.mkdir(parents=True)
    manifest.register_component(
        shared_dir(), "model-moved", kind="model", version="1", path=_relative_pack(folder), sha256=sha, app="canvas",
    )
    raw = json.loads((shared_dir() / "manifest.json").read_text(encoding="utf-8"))
    assert raw["components"]["model-moved"]["path"] == folder.resolve().as_posix()
