"""Views over the VOXPrint MDL blob store: hard links, staging, marker, no weight copies, blobs kept."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from voxprint_common.gpu_probe import VramTier
from voxprint_common.mdl.catalog import load_manifest, plan_files

from canvas.app.i18n import text
from canvas.infra.paths import models_dir
from canvas.models import views
from canvas.models.mdl_layout import files_for
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


def _install(model_id: str, files: dict[str, str]) -> None:
    """Register an installed mdl.json whose file list is ``files``."""
    root = models_dir()
    rows = []
    for path, digest in sorted(files.items()):
        rows.append({
            "path": path,
            "size": views.blob_path(root, digest).stat().st_size,
            "sha256": digest,
            "mirrors": ["https://example.com/weights"],
        })
    app = Path(os.environ["VOXPRINT_HOME"]) / "canvas-app"
    app.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": 1,
        "app": "canvas",
        "models": [{
            "id": model_id,
            "version": "1",
            "vram_tier": "16gb",
            "files": rows,
        }],
    }
    (app / "mdl.json").write_text(json.dumps(document), encoding="utf-8")
    state = Path(os.environ["VOXPRINT_HOME"]) / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "installed-apps.json").write_text(
        json.dumps({"apps": [{"id": "canvas", "path": str(app)}]}) + "\n",
        encoding="utf-8",
    )


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


def test_a_missing_blob_is_not_downloaded_and_removes_a_stale_view(home):
    root = models_dir()
    files = _files(root)
    spec = _spec(files)
    _install(spec.id, files)
    folder = views.ensure_view(root, spec.id, files)
    assert folder is not None
    views.blob_path(root, files["model_index.json"]).unlink()
    assert snapshot_ready(spec, root) is None
    assert snapshot_status(spec, root) == "not_downloaded"
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
    with pytest.raises(views.ViewError) as failed:
        views.ensure_view(root, "z", files)
    weight_name = "diffusion_pytorch_model.safetensors"
    assert str(failed.value) == text("view.link_failed", name=weight_name)
    assert not (root / "views" / "canvas").exists() or not any((root / "views" / "canvas").iterdir())
    with pytest.raises(SnapshotError) as refused:
        snapshot_ready(_spec(files), root)
    assert str(refused.value) == text("view.link_failed", name=weight_name)
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


def test_the_legacy_repo_folder_is_ignored(home):
    root = models_dir()
    spec = _spec({"model_index.json": hashlib.sha256(CONFIG).hexdigest()})
    legacy = root / spec.repo.replace("/", "--")
    legacy.mkdir(parents=True)
    (legacy / "model_index.json").write_bytes(CONFIG)
    assert snapshot_ready(spec, root) is None
    assert snapshot_status(spec, root) == "not_downloaded"
    assert (legacy / "model_index.json").read_bytes() == CONFIG


def test_the_loader_returns_only_a_view_and_keeps_the_legacy_copy(home):
    from canvas.models.loader import _safety_folder
    from canvas.safety.gate import SafetyUnavailable

    root = models_dir()
    digest = hashlib.sha256(CONFIG).hexdigest()
    spec = _spec({"model_index.json": digest})
    legacy = root / spec.repo.replace("/", "--")
    legacy.mkdir(parents=True)
    (legacy / "model_index.json").write_bytes(b"plain copy")
    with pytest.raises(SafetyUnavailable) as missing:
        _safety_folder(spec, root)
    assert str(missing.value) == text("model.missing")
    assert (legacy / "model_index.json").read_bytes() == b"plain copy"
    files = {"model_index.json": _blob(root, CONFIG)}
    ready = _spec(files)
    _install(ready.id, files)
    copy = root / ready.repo.replace("/", "--")
    copy.mkdir(parents=True, exist_ok=True)
    (copy / "model_index.json").write_bytes(b"plain copy")
    folder = _safety_folder(ready, root)
    assert folder is not None and views.is_view(root, folder)
    assert folder != copy
    assert (folder / "model_index.json").read_bytes() == CONFIG
    assert (copy / "model_index.json").read_bytes() == b"plain copy"
    log = (Path(os.environ["VOXPRINT_HOME"]) / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert f"id={ready.id} kind=view" in log
    assert "kind=legacy" not in log


def test_the_default_models_dir_opens_the_mdl_view(home):
    root = models_dir()
    assert root == Path(os.environ["VOXPRINT_HOME"]) / "models"
    spec = _spec({"model_index.json": _blob(root, b"other")})
    files = {"scheduler/scheduler_config.json": _blob(root, CONFIG)}
    _install(spec.id, files)
    folder = snapshot_ready(spec, root)
    assert folder is not None and folder.parent == root / "views" / "canvas"
    assert (folder / "scheduler" / "scheduler_config.json").read_bytes() == CONFIG
    assert not (folder / "model_index.json").exists()
    mapped = files_for(spec.id, root)
    assert mapped is not None
    assert mapped["scheduler/scheduler_config.json"] == views.blob_path(root, files["scheduler/scheduler_config.json"])


def test_an_mdl_view_follows_voxprint_models_dir(home, monkeypatch):
    custom = home / "elsewhere"
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(custom))
    root = models_dir()
    assert root == custom
    spec = _spec({"model_index.json": _blob(root, b"other")})
    files = {"scheduler/scheduler_config.json": _blob(root, CONFIG)}
    _install(spec.id, files)
    folder = snapshot_ready(spec, root)
    assert folder is not None and custom.resolve() in folder.resolve().parents
    assert snapshot_status(spec, root) == "ready"


def test_a_blob_of_the_wrong_size_is_not_downloaded(home):
    root = models_dir()
    files = _files(root)
    spec = _spec(files)
    _install(spec.id, files)
    views.blob_path(root, files["model_index.json"]).write_bytes(CONFIG + b" ")
    assert files_for(spec.id, root) is None
    assert snapshot_status(spec, root) == "not_downloaded"
    assert snapshot_ready(spec, root) is None


def test_an_sha_change_rebuilds_the_view_and_keeps_both_blobs(home):
    root = models_dir()
    files = _files(root)
    spec = _spec(files)
    _install(spec.id, files)
    first = snapshot_ready(spec, root)
    changed = dict(files)
    changed["model_index.json"] = _blob(root, b'{"a": 2}')
    _install(spec.id, changed)
    second = snapshot_ready(spec, root)
    assert first is not None and second is not None and first != second
    assert not first.exists()
    marker = json.loads((second / views.MARKER).read_text(encoding="utf-8"))
    assert marker["files"]["model_index.json"] == changed["model_index.json"]
    assert views.blob_path(root, files["model_index.json"]).read_bytes() == CONFIG
    assert views.blob_path(root, changed["model_index.json"]).read_bytes() == b'{"a": 2}'


def test_files_for_matches_the_mdl_download_plan(home):
    root = models_dir()
    spec = _spec({})
    files = {"model_index.json": _blob(root, CONFIG)}
    _install(spec.id, files)
    manifest = load_manifest(Path(os.environ["VOXPRINT_HOME"]) / "canvas-app" / "mdl.json")
    planned = plan_files([manifest], VramTier.GB16, models_root=root, only=[spec.id])
    assert {item.name: item.destination for item in planned} == files_for(spec.id, root)


def test_a_nested_blob_survives_removing_the_view(home):
    root = models_dir()
    nested = root / "blobs" / "ab" / ("cd" * 32)
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"keep")
    files = _files(root)
    folder = views.ensure_view(root, "z", files)
    views._remove_view(root, root / "blobs" / "ab")
    views.remove_views(root)
    assert folder is not None and not folder.exists()
    assert nested.read_bytes() == b"keep"


def test_unregister_models_removes_the_views_and_keeps_the_blobs(home):
    from canvas.app.main import main

    root = models_dir()
    files = _files(root)
    folder = views.ensure_view(root, "z", files)
    assert folder is not None
    assert main(["--unregister-models"]) == 0
    assert not folder.exists()
    assert all(views.blob_path(root, digest).is_file() for digest in files.values())
