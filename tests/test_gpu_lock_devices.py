"""The GPU lock is one file per CUDA device. It is taken before any weights load."""
import json
import os
from pathlib import Path

import pytest

from canvas.app.main import main
from canvas.app.platform import GpuCheck, OsCheck
from canvas.infra import gpu_lock
from canvas.infra.gpu import GpuInfo
from canvas.infra.gpu_lock import (
    GpuLockTimeout,
    device_key,
    legacy_lock_path,
    lock_path,
    pinned,
    prepare_startup_lock,
    read_lock,
    release,
    try_acquire,
)
from canvas.models.modes import Tier


@pytest.fixture
def temp_locks(tmp_path: Path, monkeypatch) -> Path:
    """Lock files land in ``tmp_path``. The device comes from ``CUDA_VISIBLE_DEVICES``."""
    monkeypatch.delenv("VOXPRINT_GPU_LOCK", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    monkeypatch.setattr(gpu_lock, "_cuda_uuid", lambda: None)
    return tmp_path


def _held_by_another_process(path: Path, owner: str = "canvas") -> None:
    """A live holder that is not this process: the parent pid."""
    payload = {
        "owner": owner,
        "pid": os.getppid(),
        "job": "canvas-generate",
        "started": "2099-01-01T00:00:00+00:00",
        "eta": "2099-01-01T00:10:00+00:00",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_each_device_has_its_own_lock_file(temp_locks: Path, monkeypatch) -> None:
    assert lock_path() == temp_locks / "voxprint-gpu-0.lock"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert device_key() == "1"
    assert lock_path() == temp_locks / "voxprint-gpu-1.lock"
    monkeypatch.setattr(gpu_lock, "_cuda_uuid", lambda: "GPU-1234abcd-0000")
    assert lock_path() == temp_locks / "voxprint-gpu-GPU-1234abcd-0000.lock"
    assert lock_path("GPU-a/b:c") == temp_locks / "voxprint-gpu-GPU-a_b_c.lock"


def test_a_process_on_another_gpu_is_not_refused(temp_locks: Path) -> None:
    _held_by_another_process(lock_path("0"))
    assert try_acquire("canvas-generate", 60, lock_path("0")) is not None
    assert try_acquire("canvas-generate", 60, lock_path("1")) is None
    release(lock_path("1"))
    assert read_lock(lock_path("1")) is None
    assert read_lock(lock_path("0")) is not None


def test_a_foreign_app_on_the_machine_wide_lock_still_counts(temp_locks: Path) -> None:
    _held_by_another_process(legacy_lock_path(), owner="dubber")
    assert try_acquire("canvas-generate", 60, lock_path()) is not None
    assert prepare_startup_lock() == "busy"
    legacy_lock_path().unlink()
    _held_by_another_process(legacy_lock_path(), owner="canvas")
    assert try_acquire("canvas-generate", 60, lock_path()) is None
    release(lock_path())


def test_a_pinned_lock_survives_inner_releases(temp_locks: Path) -> None:
    with pinned("canvas-photoshoot") as path:
        assert try_acquire("canvas-generate", 60, path) is None
        release(path)
        assert read_lock(path) is not None
        # The Enlarge hand-off releases it for real, then takes it back.
        release(path, force=True)
        assert read_lock(path) is None
        assert try_acquire("canvas-generate", 60, path) is None
        assert read_lock(path) is not None
    assert read_lock(path) is None
    _held_by_another_process(path)
    with pytest.raises(GpuLockTimeout):
        with pinned("canvas-photoshoot"):
            pass


def _ready_cli(monkeypatch, tmp_path: Path, calls: list[str]) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("canvas.app.main.current_os_check", lambda: OsCheck(True, ""))
    monkeypatch.setattr("canvas.app.main.probe_gpu", lambda: GpuInfo("RTX 4090", 23.99, 20, 8, 9, "fake"))
    monkeypatch.setattr("canvas.app.main.check_gpu", lambda info: GpuCheck(True, "", Tier.GB24))
    monkeypatch.setattr("canvas.app.main.install_supporting", lambda *args, **kwargs: calls.append("install"))

    def stack(*args: object, **kwargs: object) -> object:
        calls.append("load")
        raise AssertionError("weights must not load")

    monkeypatch.setattr("canvas.app.main.load_catalog_stack", stack)


def test_generate_is_refused_before_any_weights_load_when_the_gpu_is_busy(
    temp_locks: Path, monkeypatch, capsys,
) -> None:
    calls: list[str] = []
    _ready_cli(monkeypatch, temp_locks, calls)
    _held_by_another_process(lock_path())
    code = main(["generate", "--prompt", "a red boat", "--out", str(temp_locks / "out.png"), "--json"])
    assert code == 1
    assert json.loads(capsys.readouterr().out) == {"ok": False, "error": "the GPU is busy", "code": "gpu_busy"}
    assert calls == []


def test_photoshoot_is_refused_before_any_weights_load_when_the_gpu_is_busy(
    temp_locks: Path, monkeypatch, capsys,
) -> None:
    calls: list[str] = []
    _ready_cli(monkeypatch, temp_locks, calls)
    _held_by_another_process(lock_path())
    code = main(["photoshoot", "--prompt", "a red boat", "--out", str(temp_locks / "series"), "--json"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "gpu_busy"
    assert calls == []


def test_generate_on_a_free_gpu_takes_its_own_lock(temp_locks: Path, monkeypatch, capsys) -> None:
    calls: list[str] = []
    _ready_cli(monkeypatch, temp_locks, calls)
    _held_by_another_process(lock_path("1"))
    seen: list[dict[str, object] | None] = []
    monkeypatch.setattr(
        "canvas.app.main.install_supporting",
        lambda *args, **kwargs: seen.append(read_lock(lock_path("0"))),
    )
    monkeypatch.setattr(
        "canvas.app.main.load_catalog_stack",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("stopped after the lock")),
    )
    code = main(["generate", "--prompt", "a red boat", "--out", str(temp_locks / "out.png"), "--json"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["error"] == "stopped after the lock"
    assert seen and seen[0] is not None and seen[0]["pid"] == os.getpid()
    assert read_lock(lock_path("0")) is None
