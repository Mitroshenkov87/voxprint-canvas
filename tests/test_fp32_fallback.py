"""BF16 load failure falls back to the author's FP32 -upstream entry. No GPU, no weights."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from canvas.infra.paths import models_dir
from canvas.models import loader as loader_mod
from canvas.models import runtime as runtime_mod
from canvas.models.loader import ModelNotReady, ZImageGenerator
from canvas.models.mdl_layout import files_for
from canvas.models.modes import Mode, Tier
from canvas.models.registry import load_catalog
from canvas.models.runtime import WeightsLoadError
from canvas.models.snapshots import fallback_ready
from canvas.safety.image import RawImage
from tests.fakes import World

BF16 = Path("/models/bf16")
FP32 = Path("/models/fp32")


def _screened():
    return World().gate().release_request("a red boat", Mode.TEXT.value, ())


def _generator(monkeypatch, fp32: Path | None, calls: list[Path], fail_bf16: bool = True):
    queued: list[str] = []
    monkeypatch.setattr(loader_mod, "snapshot_ready", lambda spec, root, cache=None: BF16)
    monkeypatch.setattr(loader_mod, "fallback_ready", lambda spec, root, cache=None: fp32)
    monkeypatch.setattr(loader_mod, "request_fp32", lambda spec: queued.append(spec.id) or True)

    def render(plan, folder, screened, width, height, tier, **kwargs):
        calls.append(folder)
        if folder == BF16 and fail_bf16:
            raise WeightsLoadError("broken")
        return RawImage(1, 1, b"\x00\x00\x00")

    monkeypatch.setattr(loader_mod, "render", render)
    return ZImageGenerator(load_catalog(), Tier.GB24), queued


def test_a_bf16_load_failure_uses_the_fp32_entry(monkeypatch):
    calls: list[Path] = []
    generator, queued = _generator(monkeypatch, FP32, calls)
    image = generator.generate(_screened(), 1024, 1024)
    assert image.width == 1
    assert calls == [BF16, FP32]
    assert queued == []


def test_a_missing_fp32_entry_is_queued_with_mdl_and_reports_not_ready(monkeypatch):
    calls: list[Path] = []
    generator, queued = _generator(monkeypatch, None, calls)
    with pytest.raises(ModelNotReady):
        generator.generate(_screened(), 1024, 1024)
    assert calls == [BF16]
    assert queued and all(not item.endswith("-upstream") for item in queued)


def test_a_working_bf16_never_touches_the_fallback(monkeypatch):
    calls: list[Path] = []
    generator, queued = _generator(monkeypatch, FP32, calls, fail_bf16=False)
    generator.generate(_screened(), 1024, 1024)
    assert calls == [BF16]
    assert queued == []


def test_out_of_memory_is_not_a_load_failure():
    class OutOfMemoryError(RuntimeError):
        pass

    def oom(*args, **kwargs):
        raise OutOfMemoryError("CUDA out of memory")

    def broken(*args, **kwargs):
        raise OSError("no file named diffusion_pytorch_model.safetensors")

    with pytest.raises(OutOfMemoryError):
        runtime_mod._load(oom, BF16)
    with pytest.raises(WeightsLoadError):
        runtime_mod._load(broken, BF16)


def test_fallback_ready_opens_the_upstream_blob_view(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("VOXPRINT_HOME", str(home))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    root = models_dir()
    spec = next(s for s in load_catalog().models if s.fallback_files)
    data = b'{"fp32": true}'
    digest = hashlib.sha256(data).hexdigest()
    blob = root / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(data)
    decoy = hashlib.sha256(b"decoy").hexdigest()
    (root / "blobs" / decoy).write_bytes(b"decoy")
    small = replace(spec, fallback_files={"model_index.json": decoy})
    _install_upstream(home, f"{spec.id}-upstream", {"transformer/config.json": digest}, len(data))
    folder = fallback_ready(small, root)
    assert folder is not None
    assert folder.parent == root / "views" / "canvas"
    assert folder.name.startswith(f"{spec.id}-upstream-")
    assert (folder / "transformer" / "config.json").read_bytes() == data
    assert files_for(f"{spec.id}-upstream", root) == {"transformer/config.json": blob}
    assert not (folder / "model_index.json").exists()


def test_a_missing_upstream_blob_keeps_the_legacy_folder(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("VOXPRINT_HOME", str(home))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    root = models_dir()
    spec = next(s for s in load_catalog().models if s.fallback_files)
    data = b"fp32"
    digest = hashlib.sha256(data).hexdigest()
    (root / "blobs").mkdir(parents=True)
    (root / "blobs" / digest).write_bytes(data)
    small = replace(spec, fallback_files={"model_index.json": digest})
    legacy = root / (small.repo.replace("/", "--") + "--upstream")
    legacy.mkdir(parents=True)
    (legacy / "model_index.json").write_bytes(data)
    missing = hashlib.sha256(b"gone").hexdigest()
    (root / "blobs" / missing).write_bytes(b"gone")
    _install_upstream(home, f"{spec.id}-upstream", {"transformer/config.json": missing}, 4)
    (root / "blobs" / missing).unlink()
    assert fallback_ready(small, root) == legacy


def test_fallback_ready_needs_every_fp32_file(tmp_path):
    spec = next(s for s in load_catalog().models if s.fallback_files)
    data = b"fp32"
    digest = hashlib.sha256(data).hexdigest()
    small = replace(spec, fallback_files={"transformer/config.json": digest, "model_index.json": digest})
    folder = tmp_path / (small.repo.replace("/", "--") + "--upstream")
    (folder / "transformer").mkdir(parents=True)
    (folder / "transformer" / "config.json").write_bytes(data)
    assert fallback_ready(small, tmp_path) is None
    (folder / "model_index.json").write_bytes(data)
    assert fallback_ready(small, tmp_path) == folder


def _install_upstream(home: Path, model_id: str, files: dict[str, str], size: int) -> None:
    app = home / "canvas-app"
    app.mkdir(parents=True)
    document = {
        "schema": 1,
        "app": "canvas",
        "models": [{
            "id": model_id,
            "version": "1",
            "vram_tier": "16gb",
            "optional": True,
            "files": [
                {"path": path, "size": size, "sha256": digest, "mirrors": ["https://example.com/weights"]}
                for path, digest in files.items()
            ],
        }],
    }
    (app / "mdl.json").write_text(json.dumps(document), encoding="utf-8")
    state = home / "state"
    state.mkdir(parents=True)
    (state / "installed-apps.json").write_text(
        json.dumps({"apps": [{"id": "canvas", "path": str(app)}]}) + "\n",
        encoding="utf-8",
    )


def test_request_fp32_queues_the_upstream_id(monkeypatch):
    seen: list[tuple[str, list[str]]] = []
    import voxprint_common.mdl as mdl

    monkeypatch.setattr(mdl, "enqueue", lambda app, models, manifests=None: seen.append((app, list(models))) or "j")
    spec = next(s for s in load_catalog().models if s.fallback_files)
    assert loader_mod.request_fp32(spec) is True
    assert seen == [("canvas", [f"{spec.id}-upstream"])]
