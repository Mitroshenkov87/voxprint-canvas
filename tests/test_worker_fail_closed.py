"""A worker that does not answer cleanly ends the job with no picture shown or saved.

The window runs against a real spawned worker (``tests.worker_handlers``),
which is told to crash, raise, hang, or answer with a broken reply. In every
case the screen ends unavailable, the history stays empty, the pictures
folder stays empty, and the private session folder holds no picture.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QCheckBox, QPlainTextEdit, QPushButton, QWidget
from voxprint_common.ui.simple_image import Phase
from voxprint_common.worker import Handlers, WorkerHost
from voxprint_common.worker.qt import install_crash_bar

from canvas.app.i18n import text
from canvas.ipc.client import WorkerClient
from canvas.models.modes import Tier
from canvas.ui.shell_layout import open_worker_layout
from canvas.ui.shell_screen import ShellWindow
from canvas.ui.worker_checks import WorkerChecks
from tests.test_photoshoot import IDEA, _payload
from tests.worker_handlers import FIXTURE_ENV
from tests.worker_support import settle

PICTURE_SUFFIXES = {".avif", ".jxl", ".webp", ".png", ".jpg"}
HANDLERS = Handlers(run="tests.worker_handlers:run", check="tests.worker_handlers:check")


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def open_window(app, monkeypatch, tmp_path: Path):
    made: list[WorkerClient] = []

    def build(fixture: dict[str, object], *, host_options: dict[str, float] | None = None,
              deadlines: dict[str, float] | None = None, heartbeat: float | None = 60.0) -> ShellWindow:
        monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
        monkeypatch.setenv(FIXTURE_ENV, json.dumps(fixture))
        monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
        monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
        host = WorkerHost(HANDLERS, state_dir=tmp_path / "worker", **(host_options or {}))
        client = WorkerClient(host, tier=Tier.GB16.value, deadlines=deadlines, heartbeat_timeout=heartbeat)
        made.append(client)
        window = ShellWindow(
            open_worker_layout(tmp_path / "state", checks=WorkerChecks(client)),
            tier=Tier.GB16, enlarge=object(), worker=client,
        )
        install_crash_bar(window, client.adapter, title=text("app.name"), body=text("gate.unavailable"))
        window.resize(1280, 800)
        window.show()
        return window

    yield build
    for client in made:
        client.shutdown()


def _generate(window: ShellWindow, prompt: str = "a red bicycle") -> None:
    window.view.set_prompt_text(prompt)
    window.findChild(QPushButton, "generate").click()


def _pictures(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in PICTURE_SUFFIXES]


def _assert_no_pixels(window: ShellWindow, phase: Phase = Phase.UNAVAILABLE) -> None:
    layout = window.shell.layout
    assert layout.phase is phase, (layout.phase, layout.reason)
    assert layout.history == []
    assert window._pictures == {}
    assert _pictures(window.shell.state_dir / "pictures") == []
    assert _pictures(window.worker.session_dir) == []


def _wait_for_job(window: ShellWindow, op: str, *, timeout: float = 60.0) -> None:
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        window.worker.adapter._bridge._on_timeout()
        if any(job.op == op for job in window.worker._jobs.values()) and window.worker.host.pid:
            time.sleep(1.0)  # the job is now inside the handler
            return
        time.sleep(0.02)
    raise AssertionError(f"no {op} job started")


def test_a_clean_worker_keeps_the_picture(open_window) -> None:
    window = open_window({})
    _generate(window)
    settle(window, timeout=90)
    layout = window.shell.layout
    assert layout.phase is Phase.READY, layout.reason
    assert len(layout.history) == 1
    kept = Path(window._pictures[layout.history[0].item_id])
    assert kept.is_file() and kept.parent == window.shell.state_dir / "pictures"
    assert _pictures(window.worker.session_dir) == []


def test_a_handler_that_raises_shows_unavailable(open_window) -> None:
    window = open_window({"fault": "raise", "fault_ops": ["generate"]})
    _generate(window)
    settle(window, timeout=90)
    _assert_no_pixels(window)
    assert window.shell.layout.reason == text("gate.unavailable")


@pytest.mark.parametrize("fault", ["malformed", "outside", "proto"])
def test_a_bad_reply_shows_unavailable(open_window, fault: str) -> None:
    window = open_window({"fault": fault, "fault_ops": ["generate"]})
    _generate(window)
    settle(window, timeout=90)
    _assert_no_pixels(window)


def test_a_bad_reply_to_the_prompt_check_shows_unavailable(open_window) -> None:
    window = open_window({"fault": "malformed", "fault_ops": ["prompts"]})
    _generate(window)
    settle(window, timeout=90)
    _assert_no_pixels(window)


def test_a_gate_that_cannot_load_shows_unavailable_and_draws_nothing(open_window) -> None:
    window = open_window({"gate": "broken"})
    _generate(window)
    settle(window, timeout=90)
    _assert_no_pixels(window)
    assert window.worker.state == "failed"


def test_a_worker_killed_mid_picture_shows_unavailable_and_the_window_stays(open_window) -> None:
    window = open_window({"fault": "sleep", "sleep": 60, "fault_ops": ["generate"]})
    _generate(window)
    _wait_for_job(window, "generate")
    window.worker.host.kill_worker()
    settle(window, timeout=60)
    _assert_no_pixels(window)
    assert window.isVisible()
    bar = window.findChild(QWidget, "InfoBar")
    assert bar is not None and bar.isVisible()


def test_a_reply_past_its_deadline_shows_unavailable(open_window) -> None:
    window = open_window(
        {"fault": "sleep", "sleep": 60, "fault_ops": ["generate"]},
        host_options={"cancel_timeout": 0.5},
        deadlines={"generate": 2.0},
    )
    _generate(window)
    settle(window, timeout=60)
    _assert_no_pixels(window)


def test_a_worker_that_stops_beating_shows_unavailable(open_window) -> None:
    window = open_window(
        {"fault": "sleep", "sleep": 60, "fault_ops": ["generate"]},
        host_options={"heartbeat_interval": 0.0, "heartbeat_timeout": 1.0, "cancel_timeout": 0.5},
        heartbeat=None,
    )
    _generate(window)
    settle(window, timeout=60)
    _assert_no_pixels(window)


def test_a_series_whose_worker_dies_keeps_nothing(open_window) -> None:
    window = open_window({"fault": "sleep", "sleep": 60, "fault_ops": ["series"], "writer": _payload(IDEA, 3)})
    window.findChild(QCheckBox, "photoshootMode").click()
    _generate(window, IDEA)
    _wait_for_job(window, "series")
    window.worker.host.kill_worker()
    settle(window, timeout=60)
    _assert_no_pixels(window)
    assert not (window.shell.state_dir / "pictures" / "series").exists() or not _pictures(
        window.shell.state_dir / "pictures" / "series"
    )


def test_an_improve_that_fails_keeps_the_words(open_window) -> None:
    window = open_window({"fault": "raise", "fault_ops": ["improve"]})
    window.view.set_prompt_text("a red bicycle")
    window.findChild(QPushButton, "improvePrompt").click()
    settle(window, timeout=90)
    assert window.findChild(QPlainTextEdit, "promptBox").toPlainText() == "a red bicycle"
    assert window.shell.layout.phase is Phase.BLOCKED
    assert window.shell.layout.reason == text("gate.unavailable")


def test_a_refused_output_shows_no_pixels(open_window) -> None:
    window = open_window({"world": {"nsfw": 0.9, "faces": [[0, 0, 1, 1]], "ages": [[1.0, 0.0]]}})
    _generate(window)
    settle(window, timeout=90)
    _assert_no_pixels(window, Phase.BLOCKED)
