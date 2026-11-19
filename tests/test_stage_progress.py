"""Staged progress: worker stage lines and sampler steps reach the busy label (rc.45 finding 3)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QPushButton

from canvas.app.i18n import text
from canvas.infra.stage_timer import log_detail
from canvas.models.modes import Tier
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import rgb
from tests.worker_support import inline_window, settle


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    yield tmp_path


def _run_window(home: Path, monkeypatch, runner) -> tuple[list[str], list[tuple[int, int, str]], ShellWindow]:
    monkeypatch.setattr("canvas.ui.shell_screen.selectable", lambda: (object(),))
    labels: list[str] = []
    steps: list[tuple[int, int, str]] = []
    monkeypatch.setattr(ShellWindow, "_stage_label", lambda self, run, label: labels.append(label))
    monkeypatch.setattr(
        ShellWindow, "_step_progress", lambda self, run, step, total, label: steps.append((step, total, label)),
    )
    window = inline_window(home / "state", tier=Tier.GB16, enlarge=object(), runner=runner)
    window.view.set_prompt_text("a red bicycle")
    window.findChild(QPushButton, "generate").click()
    settle(window)
    return labels, steps, window


def test_stage_lines_reach_the_busy_label(app, home: Path, monkeypatch) -> None:
    def runner(job, inputs):
        del job, inputs
        log_detail("pipeline_load", "start", "none")
        return rgb(8, 8)

    labels, steps, window = _run_window(home, monkeypatch, runner)
    assert text("canvas.stage.load") in labels
    assert len(window.shell.layout.history) == 1
    window.close()


def test_sampler_steps_move_the_busy_bar(app, home: Path, monkeypatch) -> None:
    def runner(job, inputs):
        del job, inputs
        log_detail("sample", "step", "cuda", "step=3")
        log_detail("sample", "step", "cuda", "step=8")
        return rgb(8, 8)

    labels, steps, window = _run_window(home, monkeypatch, runner)
    assert (3, 8, text("canvas.stage.step", current=3, total=8)) in steps
    assert (8, 8, text("canvas.stage.step", current=8, total=8)) in steps
    window.close()


def test_notes_and_unrelated_stages_are_ignored(app, home: Path, monkeypatch) -> None:
    def runner(job, inputs):
        del job, inputs
        log_detail("hot", "start", "cpu", "pid=1 free=2 others=0")
        return rgb(8, 8)

    labels, steps, window = _run_window(home, monkeypatch, runner)
    assert text("canvas.stage.wait") in labels
    assert steps == []
    window.close()
