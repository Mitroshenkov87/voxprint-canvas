"""Staged progress on a cold Generate, and a clear sentence when the machine kills the worker."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton

from canvas.app.i18n import text
from canvas.ipc.client import _crash_code
from canvas.ipc.schema import Reply
from canvas.models.modes import Tier
from canvas.ui.shell_layout import job_from
from canvas.ui.shell_screen import ShellWindow, _Run
from tests.fakes import rgb
from tests.worker_support import inline_window, settle
from voxprint_common.ui.simple_image import Phase


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    yield tmp_path


def _window(home: Path, runner):
    return inline_window(home / "state", tier=Tier.GB16, enlarge=object(), runner=runner)


def test_stage_lines_update_the_busy_label_and_bar(app, home: Path, monkeypatch) -> None:
    from canvas.infra.stage_timer import log_detail

    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    seen: list[tuple[str, str, str]] = []
    original = ShellWindow._step_progress

    def record(self, run, step, total, label):
        seen.append((step, total, label))
        return original(self, run, step, total, label)

    monkeypatch.setattr(ShellWindow, "_step_progress", record)
    labels: list[str] = []
    monkeypatch.setattr(ShellWindow, "_stage_label", lambda self, run, label: labels.append(label))

    def runner(job, inputs):
        del job, inputs
        log_detail("pipeline_load", "start", "cuda")
        log_detail("sample", "step", "cuda", "step=2")
        return rgb(8, 8)

    window = _window(home, runner)
    window.view.set_prompt_text("a red bicycle")
    window.findChild(QPushButton, "generate").click()
    settle(window)
    assert ("Loading the model…",) and "Loading the model…" in labels
    assert seen and seen[0][2] == text("canvas.stage.step", current=2, total=8)


def test_the_worker_forwards_only_live_stages(app, home: Path, monkeypatch) -> None:
    """The worker route wraps the runner in the stage listener; the listener is removed after."""
    from canvas import worker_jobs
    from canvas.infra.stage_timer import _listeners, log_detail

    assert _listeners == []

    def runner(job, inputs):
        del job, inputs
        log_detail("sample", "step", "cuda", "step=1")
        return rgb(8, 8)

    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    window = _window(home, runner)
    window.view.set_prompt_text("a red bicycle")
    window.findChild(QPushButton, "generate").click()
    settle(window)
    assert _listeners == []
    window.close()


def test_a_sigkilled_worker_names_memory(app, home: Path, monkeypatch) -> None:
    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    window = _window(home, runner=lambda job, inputs: rgb(8, 8))
    layout = window.shell.layout
    window.view.set_prompt_text("a red bicycle")
    job = job_from(layout, dict(layout.live_settings()))
    run = _Run(job=job, total=1, base_seed=job.seed, layout_prompt="a red bicycle")
    window._run = run
    assert layout.generate()
    reply = Reply("generate", "refused", "engine_killed", "", True, {})
    window._copy_done(run, reply)
    assert layout.phase is Phase.UNAVAILABLE
    assert layout.reason == text("canvas.engine.killed")
    window.close()


def test_other_crashes_keep_the_generic_sentence(app, home: Path, monkeypatch) -> None:
    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    window = _window(home, runner=lambda job, inputs: rgb(8, 8))
    layout = window.shell.layout
    window.view.set_prompt_text("a red bicycle")
    job = job_from(layout, dict(layout.live_settings()))
    run = _Run(job=job, total=1, base_seed=job.seed, layout_prompt="a red bicycle")
    window._run = run
    assert layout.generate()
    reply = Reply("generate", "refused", "worker_crashed", "", True, {})
    window._copy_done(run, reply)
    assert layout.phase is Phase.UNAVAILABLE
    assert layout.reason == text("gate.unavailable")
    window.close()


def test_crash_code_marks_sigkill_as_engine_killed() -> None:
    class _Event:
        def __init__(self, exit_code):
            self.exit_code = exit_code

    assert _crash_code(_Event(-9)) == "engine_killed"
    assert _crash_code(_Event(137)) == "engine_killed"
    assert _crash_code(_Event(-11)) == "worker_crashed"
    assert _crash_code(_Event(None)) == "worker_crashed"
    assert _crash_code(_Event(1)) == "worker_crashed"


def test_shutdown_cancels_the_open_job_and_leaves_no_child(app, home: Path, monkeypatch) -> None:
    """A real spawned worker with a job in flight: shutdown ends everything exactly once."""
    from voxprint_common.worker import Handlers, WorkerHost

    from canvas.ipc.client import WorkerClient
    from tests.worker_handlers import FIXTURE_ENV

    monkeypatch.setenv(
        FIXTURE_ENV, json.dumps({"fault": "sleep", "sleep": 30.0, "fault_ops": ["generate"]}),
    )
    host = WorkerHost(
        Handlers(run="tests.worker_handlers:run", check="tests.worker_handlers:check"),
        state_dir=home / "worker",
    )
    client = WorkerClient(host, tier=Tier.GB16.value)
    outcomes: list[Reply] = []
    try:
        client.start()
        deadline = time.monotonic() + 90
        while client.state != "ready" and time.monotonic() < deadline:
            app.processEvents()
            client.adapter._bridge._on_timeout()
            time.sleep(0.02)
        assert client.state == "ready"
        client.submit("generate", {"job": {}, "layout_prompt": "x", "inputs": []}, on_done=outcomes.append)
        time.sleep(1.5)  # the child is inside the sleeping job
        client.shutdown(grace=1.0)
        assert client._closed
        client.shutdown()  # a second call does nothing
        assert len(outcomes) == 1
        assert not outcomes[0].ok
        assert host.pid is None  # no child process is left behind
    finally:
        client.shutdown()
