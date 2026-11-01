"""Startup does not re-hash unchanged packs or load models before they are needed."""
from __future__ import annotations

import dataclasses
import hashlib
import os
import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from canvas.app import main as app_main
from canvas.models import pin, provision
from canvas.models import verified
from canvas.models.modes import Tier
from canvas.models.registry import load_catalog
from canvas.models.snapshots import blob_path
from canvas.models.verified import VerifiedCache
from canvas.infra.paths import models_dir
from canvas.safety.gate import AgeGate
from tests.fakes import World


@pytest.fixture
def home(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr(verified, "_process", None)
    return tmp_path


def _pack(body: bytes):
    digest = hashlib.sha256(body).hexdigest()
    target = blob_path(models_dir(), digest)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    base = next(spec for spec in load_catalog().models if spec.role == "generator")
    return dataclasses.replace(base, id="startup-pack", files={"weights.bin": digest}, fallback_files={})


def _count_hashes(monkeypatch) -> list[Path]:
    read: list[Path] = []
    real = pin.verify_file

    def counting(path: Path, expected: str) -> None:
        read.append(path)
        real(path, expected)

    monkeypatch.setattr(verified, "verify_file", counting)
    monkeypatch.setattr("canvas.models.snapshots.verify_file", counting)
    return read


def test_a_relaunch_does_not_hash_an_unchanged_pack_again(home, monkeypatch):
    spec = _pack(b"weights" * 4096)
    monkeypatch.setattr(provision, "default_specs", lambda _catalog: (spec,))
    read = _count_hashes(monkeypatch)
    provision.install_defaults(load_catalog(), lambda _message: None)
    assert len(read) == 1
    # A new process starts with an empty in-memory cache and reads the stored record.
    monkeypatch.setattr(verified, "_process", None)
    provision.install_defaults(load_catalog(), lambda _message: None)
    assert len(read) == 1


def test_a_changed_pack_is_hashed_again(home, monkeypatch):
    spec = _pack(b"weights" * 4096)
    monkeypatch.setattr(provision, "default_specs", lambda _catalog: (spec,))
    read = _count_hashes(monkeypatch)
    provision.install_defaults(load_catalog(), lambda _message: None)
    blob = blob_path(models_dir(), spec.files["weights.bin"])
    stat = blob.stat()
    os.utime(blob, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    monkeypatch.setattr(verified, "_process", None)
    provision.install_defaults(load_catalog(), lambda _message: None)
    assert len(read) == 2


def test_a_failed_flush_keeps_the_records_for_the_next_one(tmp_path, monkeypatch):
    cache = VerifiedCache({}, False, tmp_path)
    body = tmp_path / "blob.bin"
    body.write_bytes(b"abc")
    assert cache.matches(body, hashlib.sha256(b"abc").hexdigest())

    def refuse(*_args) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(verified, "update_extra", refuse)
    cache.flush()
    assert cache.dirty is True
    written: list[dict] = []
    monkeypatch.setattr(verified, "update_extra", lambda _shared, _key, value: written.append(value))
    cache.flush()
    assert cache.dirty is False
    assert len(written) == 1 and len(written[0]) == 1


class _Splash:
    def set_stage(self, _message: str) -> None:
        pass

    def finish(self, _window) -> None:
        pass

    def close(self) -> None:
        pass


class _App:
    def processEvents(self) -> None:  # noqa: N802 - Qt name
        pass

    def exec(self) -> int:
        return 0


def test_the_window_opens_without_loading_models(home, monkeypatch):
    QApplication.instance() or QApplication([])
    warmed: list[str] = []
    monkeypatch.setattr(AgeGate, "warm", lambda self: warmed.append(threading.current_thread().name))
    monkeypatch.setattr(app_main, "load_catalog_stack", lambda *_args, **_kwargs: World().stack())
    before = {thread.name for thread in threading.enumerate()}
    assert app_main._launch(Tier.GB16, _Splash(), _App()) == 0  # type: ignore[arg-type]
    started = {thread.name for thread in threading.enumerate()} - before
    assert warmed == []
    assert not any(name.startswith(("canvas-prepare", "stage-warmup")) for name in started)
    for widget in QApplication.topLevelWidgets():
        widget.close()


def test_a_relaunch_reads_each_stack_weight_once(home, monkeypatch):
    from canvas.models import loader

    roles = ("prompt-guard", "zero-shot", "face-detector", "age", "nsfw", "age-fine")
    catalog = load_catalog()
    specs = []
    for spec in catalog.models:
        if spec.role not in roles:
            continue
        body = spec.id.encode() * 2048
        digest = hashlib.sha256(body).hexdigest()
        target = blob_path(models_dir(), digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        specs.append(dataclasses.replace(spec, files={"weights.bin": digest}, fallback_files={}))
    stack_catalog = dataclasses.replace(catalog, models=tuple(specs))
    read = _count_hashes(monkeypatch)
    # The counting reader _count_hashes installed, for the check before load too.
    monkeypatch.setattr(loader, "verify_file", verified.verify_file)
    for name in ("Qwen3GuardGen", "SiglipZeroShot", "YuNetFaces", "FairFaceAge", "NsfwViT", "FaceAgeRegressor"):
        monkeypatch.setattr(loader, name, lambda folder: object())
    loader.load_safety_stack(stack_catalog, models_dir())
    monkeypatch.setattr(verified, "_process", None)
    read.clear()
    loader.load_safety_stack(stack_catalog, models_dir())
    # The view lookup trusts the stored record; the pre-load check still reads every file.
    assert len(read) == len(roles)
