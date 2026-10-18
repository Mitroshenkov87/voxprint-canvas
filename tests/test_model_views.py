"""Views over the VOXPrint MDL blob store: hard links, staging, marker, no weight copies, blobs kept."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from canvas.infra.paths import models_dir
from canvas.models import views
from canvas.models.registry import load_catalog
from canvas.models.snapshots import SnapshotError, snapshot_ready, snapshot_status

WEIGHTS = b"w" * 4096
CONFIG = b'{"a": 1}'


def _blob(root: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    path = root / "blobs" / digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return digest


def _files(root: Path) -> dict[str, str]:
    return {
        "transformer/diffusion_pytorch_model.safetensors": _blob(root, WEIGHTS),
        "model_index.json": _blob(root, CONFIG),
    }


def _spec(files: dict[str, str]):
    spec = next(s for s in load_catalog().models if s.role == "generator")
    return replace(spec, files=files, fallback_files={})


@pytest.fixture
def home(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    return tmp_path


def test_a_view_is_hard_linked_in_the_repo_layout_with_a_marker(home):
    root = models_dir()
    files = _files(root)
    folder = views.ensure_view(root, "z", files)
    assert folder is not None and folder.parent == root / "views" / "canvas"
    weight = folder / "transformer" / "diffusion_pytorch_model.safetensors"
    assert os.path.samefile(weight, views.blob_path(root, files["transformer/diffusion_pytorch_model.safetensors"]))
    marker = json.loads((folder / views.MARKER).read_text(encoding="utf-8"))
    assert marker["files"] == files
    assert not list(folder.parent.glob("*.staging-*"))
    assert views.ensure_view(root, "z", files) == folder


def test_the_view_follows_voxprint_models_dir(home, monkeypatch):
    custom = home / "elsewhere"
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(custom))
    root = models_dir()
    assert root == custom
    files = _files(root)
    spec = _spec(files)
    folder = snapshot_ready(spec, root)
    assert folder is not None and custom.resolve() in folder.resolve().parents
    assert snapshot_status(spec, root) == "ready"


def test_a_missing_blob_is_not_ready_and_removes_a_stale_view(home):
    root = models_dir()
    files = _files(root)
    spec = _spec(files)
    folder = views.ensure_view(root, spec.id, files)
    assert folder is not None
    views.blob_path(root, files["model_index.json"]).unlink()
    assert snapshot_ready(spec, root) is None
    assert snapshot_status(spec, root) == "incomplete"
    assert not folder.exists()
    assert views.blob_path(root, files["transformer/diffusion_pytorch_model.safetensors"]).read_bytes() == WEIGHTS


def test_a_changed_pin_builds_a_new_view_and_a_bad_marker_is_rebuilt(home):
    root = models_dir()
    files = _files(root)
    first = views.ensure_view(root, "z", files)
    changed = dict(files, **{"model_index.json": _blob(root, b'{"a": 2}')})
    second = views.ensure_view(root, "z", changed)
    assert first is not None and second is not None and first != second
    (second / views.MARKER).write_text("{}", encoding="utf-8")
    rebuilt = views.ensure_view(root, "z", changed)
    assert rebuilt == second
    assert json.loads((second / views.MARKER).read_text(encoding="utf-8"))["files"] == changed


def test_a_blob_that_does_not_match_its_pin_is_never_served(home):
    root = models_dir()
    files = _files(root)
    views.blob_path(root, files["model_index.json"]).write_bytes(b"tampered")
    spec = _spec(files)
    assert snapshot_ready(spec, root) is None
    assert snapshot_status(spec, root) == "hash_mismatch"


def test_windows_never_copies_weights_when_the_link_fails(home, monkeypatch):
    root = models_dir()
    files = _files(root)

    def no_link(*_args, **_kwargs):
        raise OSError("cross-device link")

    monkeypatch.setattr(views.os, "link", no_link)
    monkeypatch.setattr(views.sys, "platform", "win32")
    with pytest.raises(views.ViewError):
        views.ensure_view(root, "z", files)
    assert not (root / "views" / "canvas").exists() or not any((root / "views" / "canvas").iterdir())
    with pytest.raises(SnapshotError):
        snapshot_ready(_spec(files), root)
    only_config = {"model_index.json": files["model_index.json"]}
    folder = views.ensure_view(root, "cfg", only_config)
    assert folder is not None and (folder / "model_index.json").read_bytes() == CONFIG


def test_linux_uses_a_symlink_when_the_hard_link_fails(home, monkeypatch):
    root = models_dir()
    files = _files(root)

    def no_link(*_args, **_kwargs):
        raise OSError("cross-device link")

    monkeypatch.setattr(views.os, "link", no_link)
    monkeypatch.setattr(views.sys, "platform", "linux")
    folder = views.ensure_view(root, "z", files)
    assert folder is not None
    weight = folder / "transformer" / "diffusion_pytorch_model.safetensors"
    assert weight.is_symlink()
    assert views.ensure_view(root, "z", files) == folder


def test_removing_views_keeps_every_blob(home):
    root = models_dir()
    files = _files(root)
    folder = views.ensure_view(root, "z", files)
    assert folder is not None
    views.remove_views(root)
    assert not folder.exists()
    for digest in files.values():
        assert views.blob_path(root, digest).is_file()
    assert views.blob_path(root, files["transformer/diffusion_pytorch_model.safetensors"]).read_bytes() == WEIGHTS


def test_a_folder_outside_the_views_root_is_never_removed(home):
    root = models_dir()
    files = _files(root)
    views._remove_view(root, root / "blobs")
    views._remove_view(root, root)
    assert all(views.blob_path(root, digest).is_file() for digest in files.values())


def test_a_pinned_name_cannot_escape_the_view(home):
    root = models_dir()
    digest = _blob(root, CONFIG)
    with pytest.raises(views.ViewError):
        views.ensure_view(root, "z", {"../escape.json": digest})


def test_the_legacy_repo_folder_still_works(home):
    root = models_dir()
    spec = _spec({"model_index.json": hashlib.sha256(CONFIG).hexdigest()})
    legacy = root / spec.repo.replace("/", "--")
    legacy.mkdir(parents=True)
    (legacy / "model_index.json").write_bytes(CONFIG)
    assert snapshot_ready(spec, root) == legacy


def test_unregister_models_removes_the_views_and_keeps_the_blobs(home):
    from canvas.app.main import main

    root = models_dir()
    files = _files(root)
    folder = views.ensure_view(root, "z", files)
    assert folder is not None
    assert main(["--unregister-models"]) == 0
    assert not folder.exists()
    assert all(views.blob_path(root, digest).is_file() for digest in files.values())
