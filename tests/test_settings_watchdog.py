"""canvas.toml settings, the stall watchdog and the self-healing GPU lock. CPU only."""
import json
import os
import threading
import time
from pathlib import Path

import pytest

from canvas.infra import settings, watchdog
from canvas.infra.gpu_lock import _pid_alive, prepare_startup_lock, read_lock, try_acquire


@pytest.fixture(autouse=True)
def _fresh_settings():
    settings.reset()
    yield
    settings.reset()


@pytest.fixture
def logged(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(settings, "note_error", lines.append)
    return lines


def test_missing_file_gives_defaults(tmp_path, logged):
    assert settings.load(tmp_path / "none.toml") == settings.Settings()
    assert logged == []


def test_valid_file(tmp_path, logged):
    path = tmp_path / "canvas.toml"
    path.write_text('gpu = 1\nplacement = "ram64"\nkeep_warm_s = 300\nwriter = true\nsteps = 12\n', encoding="utf-8")
    assert settings.load(path) == settings.Settings(gpu=1, placement="ram64", keep_warm_s=300, writer=True, steps=12)
    assert logged == []


def test_unknown_and_invalid_keep_defaults_with_one_line_each(tmp_path, logged):
    path = tmp_path / "canvas.toml"
    path.write_text('colour = 1\ngpu = "x"\nsteps = 9999\nwriter = 1\nplacement = "ram8"\nkeep_warm_s = true\n', encoding="utf-8")
    assert settings.load(path) == settings.Settings()
    assert len(logged) == 6


def test_broken_toml_gives_defaults(tmp_path, logged):
    path = tmp_path / "canvas.toml"
    path.write_text("gpu = = 1", encoding="utf-8")
    assert settings.load(path) == settings.Settings()
    assert len(logged) == 1


def test_get_reads_once(tmp_path, monkeypatch):
    path = tmp_path / "canvas.toml"
    path.write_text("steps = 5\n", encoding="utf-8")
    monkeypatch.setenv("CANVAS_CONFIG", str(path))
    assert settings.get().steps == 5
    path.write_text("steps = 6\n", encoding="utf-8")
    assert settings.get().steps == 5


def test_placement_setting_is_used_unless_env_names_one(monkeypatch):
    from canvas.models import hot_tier
    from voxprint_common.ram_probe import RamTier

    monkeypatch.setattr(settings, "_cached", settings.Settings(placement="ram96"))
    monkeypatch.delenv("VOXPRINT_RAM_MODE", raising=False)
    assert hot_tier.active_tier() is RamTier.RAM96
    monkeypatch.setenv("VOXPRINT_RAM_MODE", "ram32")
    assert hot_tier.active_tier() is RamTier.RAM32


def test_gpu_setting_does_not_override_env(monkeypatch):
    from canvas.app.main import _apply_gpu_setting

    monkeypatch.setattr(settings, "_cached", settings.Settings(gpu=2))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _apply_gpu_setting()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    _apply_gpu_setting()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_watchdog_aborts_a_stalled_stage(monkeypatch):
    codes: list[int] = []
    seen: list[str] = []
    fired = threading.Event()
    monkeypatch.setattr(watchdog, "note_error", lambda line: None)
    monkeypatch.setattr(watchdog, "_terminate", lambda code: (codes.append(code), fired.set()))
    forget = watchdog.on_abort(seen.append)
    try:
        with watchdog.watch("sample", stall_s=0.05):
            assert fired.wait(2)
    finally:
        forget()
    assert codes == [watchdog.EXIT_STALLED]
    assert seen == ["sample"]


def test_watchdog_stays_quiet_with_progress(monkeypatch):
    codes: list[int] = []
    monkeypatch.setattr(watchdog, "_terminate", codes.append)
    with watchdog.watch("sample", stall_s=0.2) as progress:
        for _ in range(6):
            time.sleep(0.05)
            progress.kick()
    time.sleep(0.1)
    assert codes == []


def test_step_beat_marks_progress():
    from canvas.models.runtime import _with_step_beat

    class Pipe:
        def __call__(self, prompt=None, callback_on_step_end=None):
            return None

    class Mark:
        count = 0

        def kick(self):
            self.count += 1

    mark = Mark()
    kwargs = _with_step_beat(Pipe(), {"prompt": "x"}, mark)
    kwargs["callback_on_step_end"](None, 0, 1, {"latents": 1})
    assert mark.count == 1
    assert _with_step_beat(object(), {"prompt": "x"}, mark) == {"prompt": "x"}


def test_batch_summary_records_a_stall(tmp_path):
    from canvas.app.batch_cli import _write_stalled
    from canvas.pipeline.batch import BatchEntry

    entry = BatchEntry(name="a", prompt="", shot="s", negative="", seed=1, count=2)
    _write_stalled(tmp_path, [entry], "sample")
    body = json.loads((tmp_path / "batch.json").read_text(encoding="utf-8"))
    assert body["ok"] is False
    assert body["code"] == watchdog.CODE
    assert body["requested"] == 2


def _dead_pid() -> int:
    pid = 2
    while pid < 100000:
        if not _pid_alive(pid):
            return pid
        pid += 1
    raise RuntimeError("no unused pid")


def _write_lock(path: Path, pid: int) -> None:
    now = "2999-01-01T00:00:00+00:00"
    path.write_text(json.dumps({"owner": "canvas", "pid": pid, "job": "x", "started": now, "eta": now}), encoding="utf-8")


def test_stale_lock_of_a_dead_pid_is_taken_over(tmp_path, monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr("canvas.infra.gpu_lock.note_error", lines.append)
    path = tmp_path / "gpu.lock"
    _write_lock(path, _dead_pid())
    assert try_acquire("job", 60, path) is None
    assert read_lock(path)["pid"] == os.getpid()
    assert len(lines) == 1


def test_live_owner_keeps_the_lock(tmp_path):
    path = tmp_path / "gpu.lock"
    _write_lock(path, os.getppid())
    holder = try_acquire("job", 60, path)
    assert holder is not None and holder["pid"] == os.getppid()


def test_startup_drops_a_dead_owner_lock(tmp_path, monkeypatch):
    path = tmp_path / "gpu.lock"
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(path))
    monkeypatch.setattr("canvas.infra.gpu_lock.note_error", lambda line: None)
    _write_lock(path, _dead_pid())
    assert prepare_startup_lock() is None
    assert not path.exists()


def test_takeover_leaves_a_lock_that_changed(tmp_path):
    from canvas.infra.gpu_lock import _remove_stale

    path = tmp_path / "gpu.lock"
    _write_lock(path, os.getppid())
    assert _remove_stale(path, {"owner": "canvas", "pid": _dead_pid()}) is False
    assert path.exists()
