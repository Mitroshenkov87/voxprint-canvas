"""The shared headless GUI smoke walk (``voxprint_common.testing.gui_smoke``) on the Canvas window.

The window is wired as ``_launch`` wires it: the real ``canvas.worker_jobs``
handlers in a spawned worker, the crash bar, and the worker-backed screen.
Every enabled button and menu action must do something, and killing the
worker must show the crash bar without closing the window.
"""
from __future__ import annotations

from voxprint_common.testing import gui_smoke  # sets the offscreen platform before Qt starts

import pytest

pytest.importorskip("pytestqt")

from pathlib import Path

from voxprint_common.worker.qt import install_crash_bar

from canvas.app.i18n import text
from canvas.ipc.client import WorkerClient, default_host
from canvas.models.modes import Tier
from canvas.ui.shell_layout import open_worker_layout
from canvas.ui.shell_screen import ShellWindow
from canvas.ui.worker_checks import WorkerChecks


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    # Controls that leave the app: the walk checks that they respond, not that they start other programs.
    monkeypatch.setattr("canvas.ui.shell_screen.subprocess.Popen", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("canvas.ui.shell_screen.QDesktopServices.openUrl", lambda *_args: True)
    worker = WorkerClient(default_host(), tier=Tier.GB16.value)
    shown = ShellWindow(
        open_worker_layout(tmp_path / "state", checks=WorkerChecks(worker)),
        tier=Tier.GB16,
        enlarge=object(),
        worker=worker,
        installer=lambda _progress: "cancelled",
    )
    install_crash_bar(shown, worker.adapter, title=text("app.name"), body=text("gate.unavailable"))
    qtbot.addWidget(shown)
    shown.resize(1280, 800)
    shown.show()
    qtbot.waitExposed(shown)
    yield shown
    worker.shutdown()


def test_every_control_of_the_window_responds(qtbot, window) -> None:
    window.view.set_prompt_text("a red bicycle")
    activated = gui_smoke.exercise(window, qtbot=qtbot)
    assert activated > 10
    assert window.isVisible()


def test_the_window_survives_a_worker_crash(qtbot, window) -> None:
    worker = window.worker
    worker.start()
    qtbot.waitUntil(lambda: worker.state in {"ready", "failed"} and worker.host.pid is not None, timeout=120_000)
    gui_smoke.assert_survives_worker_crash(qtbot, window, worker.host)
    assert window.shell.layout.history == []
