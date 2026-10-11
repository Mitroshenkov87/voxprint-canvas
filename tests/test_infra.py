"""VRAM budget, GPU probe, lock, thermal pause, and the shared model folder."""
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from canvas.core.vram_policy import budget_gb, halve_on_spill, plan
from canvas.infra.gpu import probe_gpu
from canvas.infra.gpu_lock import _pid_alive, gpu_job, is_stale, read_lock, release, try_acquire
from canvas.infra.models_users import read_users, register, unregister
from canvas.infra.paths import models_dir, shared_dir
from canvas.infra.thermal import FULL_SPEED_S, THROTTLE_HOLD_S, Guard
from canvas.models.modes import Tier, headroom_gb, select_tier


def test_headroom_and_tiers():
    assert headroom_gb(16) == 2.0
    assert headroom_gb(30) == pytest.approx(2.4)
    assert headroom_gb(40) == 3.0
    assert budget_gb(10, 16) == 8.0
    assert plan(10, 16, 3, 8).batch == 2
    assert halve_on_spill(5) == 2
    assert halve_on_spill(1) == 1
    assert select_tier(15.49) is None
    assert select_tier(15.5) is Tier.GB16
    assert select_tier(23.49) is Tier.GB16
    assert select_tier(23.5) is Tier.GB24
    assert select_tier(23.99) is Tier.GB24


def test_fake_probe(monkeypatch):
    monkeypatch.setenv("VOXPRINT_FAKE_VRAM", "RTX 4090,24,20,8.9")
    info = probe_gpu()
    assert info is not None
    assert info.name == "RTX 4090"
    assert info.total_gb == 24 and info.free_gb == 20
    assert info.supported is True
    assert info.source == "fake"
    monkeypatch.setenv("VOXPRINT_FAKE_VRAM", "older,16,10,8.6")
    older = probe_gpu()
    assert older is not None and older.supported is False


def _dead_pid() -> int:
    pid = 2
    while pid < 100000:
        if not _pid_alive(pid):
            return pid
        pid += 1
    raise RuntimeError("no unused pid")


def test_gpu_lock_stale_rules(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    live = {"pid": os.getpid(), "eta": (now - timedelta(hours=2)).isoformat()}
    assert is_stale(live, now) is False
    assert is_stale(live, now + timedelta(seconds=1)) is True
    future = {"pid": _dead_pid(), "eta": (now + timedelta(hours=1)).isoformat()}
    assert is_stale(future, now) is True
    path = tmp_path / "voxprint-gpu.lock"
    ahead = datetime.now(timezone.utc) + timedelta(hours=1)
    path.write_text(json.dumps({"owner": "other", "pid": os.getpid(), "job": "x", "started": ahead.isoformat(), "eta": ahead.isoformat()}), encoding="utf-8")
    assert try_acquire("canvas-generate", 60, path) is not None
    path.write_text(json.dumps({"owner": "other", "pid": _dead_pid(), "job": "x", "eta": now.isoformat()}), encoding="utf-8")
    assert try_acquire("canvas-generate", 60, path) is None
    assert read_lock(path)["owner"] == "canvas"
    release(path)
    assert not path.exists()
    with gpu_job("canvas-generate", path=path):
        with gpu_job("again", path=path):
            assert read_lock(path)["owner"] == "canvas"
    held = tmp_path / "busy.lock"
    held.write_text(json.dumps({"owner": "other", "pid": os.getpid(), "eta": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}), encoding="utf-8")
    with pytest.raises(TimeoutError):
        with gpu_job("canvas-generate", path=held, timeout=0):
            pass


def test_thermal_pause_and_resume():
    clock = {"t": 0.0}
    guard = Guard(now=lambda: clock["t"])
    for index in range(3):
        clock["t"] = float(index)
        guard.observe(90, False)
    assert guard.should_pause() is False
    for index in range(3):
        clock["t"] = FULL_SPEED_S + index
        guard.observe(83, False)
    assert guard.should_pause() is True
    assert guard.cool_enough() is False
    throttle_clock = {"t": 0.0}
    cool = Guard(now=lambda: throttle_clock["t"])
    throttle_clock["t"] = FULL_SPEED_S
    cool.observe(70, True)
    throttle_clock["t"] = FULL_SPEED_S + THROTTLE_HOLD_S + 1
    cool.observe(70, True)
    assert cool.should_pause() is True
    settled_clock = {"t": 0.0}
    settled = Guard(now=lambda: settled_clock["t"])
    for index in range(3):
        settled_clock["t"] = FULL_SPEED_S + index
        settled.observe(75, False)
    assert settled.cool_enough() is True


def test_model_folder_and_users(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", "relative/models")
    assert models_dir() == tmp_path / "shared" / "models"
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path / "weights"))
    assert models_dir() == tmp_path / "weights"
    shared = shared_dir()
    register(shared)
    assert read_users(shared)["canvas"] is True
    register(shared)
    assert unregister(shared) == []
    assert "canvas" not in read_users(shared)
