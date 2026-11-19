"""Manage models: the MDL child is reaped, never double-started, and stopped with the window."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from canvas.app.i18n import text
from canvas.models.modes import Tier
from canvas.ui import shell_screen
from tests.worker_support import inline_window, settle


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    yield tmp_path


class _FakeMdl:
    """Stand-in for the Popen'ed MDL GUI. Records signals instead of starting a program."""

    def __init__(self, code: int | None = None) -> None:
        self.pid = 424242
        self.code = code
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.code

    def wait(self, timeout: float | None = None) -> int:
        return 0 if self.code is None else self.code


def _window(home: Path, fake: _FakeMdl, monkeypatch, launches: list[int]) -> "shell_screen.ShellWindow":
    monkeypatch.setattr("canvas.ui.shell_screen.subprocess.Popen", lambda argv: launches.append(1) or fake)
    monkeypatch.setattr("canvas.infra.gpu.probe_gpu", lambda: object())
    return inline_window(home / "state", tier=Tier.GB16, enlarge=object())


def test_the_mdl_child_is_not_started_twice(app, home: Path, monkeypatch) -> None:
    fake = _FakeMdl()
    launches: list[int] = []
    window = _window(home, fake, monkeypatch, launches)
    window._on_command("models.manage")
    settle(window)
    window._on_command("models.manage")
    settle(window)
    assert launches == [1]
    assert window._mdl_gui is fake
    assert window.view._note.text() == text("canvas.models.manage_running")
    window.close()


def test_a_finished_mdl_child_is_reaped_and_failure_is_noted(app, home: Path, monkeypatch) -> None:
    fake = _FakeMdl(code=2)
    launches: list[int] = []
    window = _window(home, fake, monkeypatch, launches)
    window._on_command("models.manage")
    settle(window)
    window._reap_mdl()
    assert window._mdl_gui is None
    assert window.view._note.text() == text("canvas.models.manage_failed")
    # A reaped child does not block the next start.
    fresh = _FakeMdl()
    monkeypatch.setattr("canvas.ui.shell_screen.subprocess.Popen", lambda argv: launches.append(1) or fresh)
    window._on_command("models.manage")
    settle(window)
    assert launches == [1, 1]
    assert window._mdl_gui is fresh
    window.close()


def test_window_close_stops_only_the_mdl_gui_this_window_started(app, home: Path, monkeypatch) -> None:
    fake = _FakeMdl()
    launches: list[int] = []
    window = _window(home, fake, monkeypatch, launches)
    window._on_command("models.manage")
    settle(window)
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(shell_screen.os, "kill", lambda pid, sig: signalled.append((pid, sig)))
    monkeypatch.setattr(shell_screen.time, "sleep", lambda _s: None)
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(shell_screen.time, "monotonic", lambda: clock.__setitem__("t", clock["t"] + 0.1) or clock["t"])
    window.close()
    assert window._mdl_gui is None
    # Only the GUI child this window started is signalled; the shared suite
    # download worker (any app's) is never touched.
    pids = {pid for pid, _sig in signalled}
    assert pids == {424242}
    # It is TERM'ed first and KILL'ed later, never the other way round.
    signals = [sig for _pid, sig in signalled]
    assert signals[0] == 15
    assert 9 in signals
    assert signals.index(9) > signals.index(15)


def test_no_worker_scan_remains() -> None:
    """The psutil create-time scan and worker killing are gone for good.

    ``voxprint_common.mdl worker`` is the shared suite download queue; Canvas
    must not kill it by any heuristic, and no other process may be signalled.
    """
    source = Path(shell_screen.__file__).read_text(encoding="utf-8")
    assert "_mdl_worker_pids" not in source
    assert "_signal_tree" not in source
    assert "psutil" not in source


def test_no_mdl_started_means_nothing_is_signalled(app, home: Path, monkeypatch) -> None:
    window = inline_window(home / "state", tier=Tier.GB16, enlarge=object())
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(shell_screen.os, "kill", lambda pid, sig: signalled.append((pid, sig)))
    window.close()
    assert signalled == []
