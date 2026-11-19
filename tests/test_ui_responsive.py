"""The window keeps handling events while a check runs, and a failed run lets go of its locks."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QPushButton

from canvas.app.i18n import text
from canvas.app.shell_run import RunStopped, ShellRunner
from canvas.core.formats import OutputFormat
from canvas.infra.gpu_lock import lock_path, prepare_startup_lock, read_lock, try_acquire
from canvas.infra.manifest import manifest_lock
from canvas.models.modes import Mode, Tier
from canvas.safety.gate import SafetyUnavailable, _wait
from canvas.ui.shell_layout import PictureJob
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import World
from tests.test_infra import _dead_pid


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    yield tmp_path


def test_the_event_loop_runs_while_a_check_blocks(app, home: Path, monkeypatch) -> None:
    """The prompt check runs in the worker process; the window keeps ticking until the reply."""
    import json

    from voxprint_common.worker import Handlers, WorkerHost

    from canvas.ipc.client import WorkerClient
    from canvas.ui.shell_layout import open_worker_layout
    from canvas.ui.worker_checks import WorkerChecks
    from tests.worker_handlers import FIXTURE_ENV
    from tests.worker_support import settle

    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    monkeypatch.setenv(FIXTURE_ENV, json.dumps({"fault": "sleep", "sleep": 0.6, "fault_ops": ["prompts"]}))
    host = WorkerHost(Handlers(run="tests.worker_handlers:run", check="tests.worker_handlers:check"),
                      state_dir=home / "worker")
    client = WorkerClient(host, tier=Tier.GB16.value)
    window = ShellWindow(
        open_worker_layout(home / "state", checks=WorkerChecks(client)), tier=Tier.GB16, enlarge=object(), worker=client,
    )
    try:
        client.start()
        settle(window, timeout=60)
        assert client.state == "ready"
        window.view.set_prompt_text("a red boat")
        window.show()
        ticks: list[float] = []
        timer = QTimer()
        timer.timeout.connect(lambda: ticks.append(time.monotonic()))
        timer.start(20)
        started = time.monotonic()
        window.findChild(QPushButton, "generate").click()
        clicked = time.monotonic()
        settle(window, timeout=60)
        timer.stop()
        assert clicked - started < 0.2
        assert len([tick for tick in ticks if tick > clicked]) >= 5
        window.close()
    finally:
        client.shutdown()


def test_a_timed_out_check_leaves_no_non_daemon_thread() -> None:
    def slow() -> None:
        time.sleep(0.4)

    with pytest.raises(SafetyUnavailable) as caught:
        _wait(slow, timeout=0.05)
    assert caught.value.code == "timeout"
    leaked = [thread for thread in threading.enumerate() if thread.name == "canvas-check" and not thread.daemon]
    assert leaked == []


def test_a_stale_gpu_lock_is_cleared(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "gpu.lock"
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(path))
    path.write_text(
        json.dumps({"owner": "canvas", "pid": _dead_pid(), "eta": "2000-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    assert prepare_startup_lock() is None
    assert not path.exists()


def test_a_live_lock_is_reported_and_kept(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "gpu.lock"
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(path))
    ahead = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    path.write_text(json.dumps({"owner": "other", "pid": os.getpid(), "eta": ahead}), encoding="utf-8")
    assert prepare_startup_lock() == "busy"
    assert path.is_file()
    assert text("startup.lock")


def test_a_failed_body_releases_the_manifest_lock(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    try:
        with manifest_lock(shared):
            raise RuntimeError("stop")
    except RuntimeError:
        pass
    held = threading.Event()

    def second() -> None:
        with manifest_lock(shared):
            held.set()

    thread = threading.Thread(target=second, daemon=True)
    thread.start()
    assert held.wait(1)
    thread.join(1)


def test_a_stopped_run_releases_the_gpu_lock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "gpu.lock"))

    class _Gen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.seed = 0
            self.models = None
            self.lora_path = ""
            self.lora_strength = 0.7

    class _Pipe:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def generate(self, *args: object, **kwargs: object) -> None:
            assert try_acquire("canvas-generate", 30, lock_path()) is None
            raise SafetyUnavailable("timeout")

    monkeypatch.setattr("canvas.app.shell_run.load_catalog", lambda: object())
    monkeypatch.setattr("canvas.app.shell_run.ZImageGenerator", _Gen)
    monkeypatch.setattr("canvas.app.shell_run.Pipeline", _Pipe)
    runner = ShellRunner(gate=World().gate(), tier=Tier.GB16, enlarge=object())
    job = PictureJob(
        "a red boat", "", Mode.TEXT, "1:1", "native", OutputFormat.AVIF, 1, 4, 1.0, "euler", 1,
    )
    with pytest.raises(RunStopped) as caught:
        runner(job, ())
    assert caught.value.unavailable is True
    assert read_lock(lock_path()) is None
