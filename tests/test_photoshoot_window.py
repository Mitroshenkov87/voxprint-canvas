"""Photoshoot is one option, a read-only shot list, and Cancel."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton, QSpinBox

from canvas.models.modes import Mode, Tier
from canvas.safety.gate import GenerationBlocked
from canvas.ui.shell_layout import open_layout
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import World, rgb
from tests.test_photoshoot import IDEA, _payload
from voxprint_common.i18n import set_language


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    set_language("en")
    yield tmp_path
    set_language("en")


def _window(home: Path, runner) -> ShellWindow:
    shell = open_layout(home / "state", gate=World().gate(), enlarge_installed=False)
    return ShellWindow(shell, tier=Tier.GB16, enlarge=object(), runner=runner, improver=lambda text: text)


def test_the_frame_count_starts_at_six(app, home: Path) -> None:
    window = _window(home, lambda job, inputs: rgb())
    window.show()
    app.processEvents()
    frames = window.findChild(QSpinBox, "photoshootFrames")
    label = window.findChild(QLabel, "photoshootFramesLabel")
    assert frames is not None and label is not None
    assert frames.value() == 6
    assert frames.minimum() == 1
    assert frames.maximum() == 50
    assert label.text() == "Frames"
    assert window.photoshoot.count == 6
    window.findChild(QPushButton, "photoshootMode").click()
    app.processEvents()
    assert frames.isVisible()
    assert label.isVisible()
    set_language("ru")
    russian = _window(home, lambda job, inputs: rgb())
    russian.show()
    app.processEvents()
    assert russian.findChild(QLabel, "photoshootFramesLabel").text() == "Кадров"
    set_language("en")


def test_progress_uses_the_chosen_frame_count(app, home: Path, monkeypatch) -> None:
    def runner(job, inputs):
        del job, inputs
        return rgb()

    def writer(prompt: str) -> str:
        body = json.loads(prompt)
        return _payload(IDEA, int(body["count"]))

    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    window = _window(home, runner)
    window.photoshoot.writer = writer
    window.view.set_prompt_text(IDEA)
    window.show()
    window.findChild(QPushButton, "photoshootMode").click()
    window.findChild(QSpinBox, "photoshootFrames").setValue(4)
    window.findChild(QPushButton, "generate").click()
    app.processEvents()
    lines = window.findChildren(QLabel, "photoshootLine")
    assert len(lines) == 4
    assert window.findChild(QLabel, "photoshootFrame").text() == "Shot 4 of 4"


def test_the_option_leaves_the_shell_create_button_in_place(app, home: Path) -> None:
    window = _window(home, lambda job, inputs: rgb())
    window.show()
    app.processEvents()
    mode = window.findChild(QPushButton, "photoshootMode")
    assert mode is not None and mode.text() == "Photoshoot"
    generate = window.findChild(QPushButton, "generate")
    assert generate is not None and generate.isVisible()
    mode.click()
    app.processEvents()
    assert mode.isChecked()
    assert generate.isVisible()
    assert window.findChild(QPushButton, "photoshootCreate") is None
    assert window.findChild(QPushButton, "photoshootWrite") is None
    assert window.findChild(QPushButton, "chip-photoshoot-count-6") is None
    assert window.findChild(QPushButton, "photoshootHandle") is None
    assert window.findChild(QPushButton, "photoshootMenu") is None


def test_generate_writes_a_read_only_series(app, home: Path, monkeypatch) -> None:
    prompts: list[str] = []

    def runner(job, inputs):
        del inputs
        prompts.append(job.prompt)
        return rgb()

    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    window = _window(home, runner)
    window.photoshoot.writer = lambda _prompt: _payload(IDEA)
    window.view.set_prompt_text(IDEA)
    window.show()
    window.findChild(QPushButton, "photoshootMode").click()
    window.findChild(QPushButton, "generate").click()
    app.processEvents()
    assert len(prompts) == 6
    assert all("50 mm" not in prompt for prompt in prompts)
    assert all(IDEA in prompt for prompt in prompts)
    lines = window.findChildren(QLabel, "photoshootLine")
    assert len(lines) == 6
    assert all(label.isVisible() for label in lines)
    assert window.findChild(QLineEdit, "photoshootCamera") is None
    assert window.findChild(QPushButton, "generate").isVisible()
    assert (home / "state" / "pictures" / "series").exists()
    assert window._series_folders
    assert all(path.endswith(".png") for path in window._pictures.values())
    recorded = json.loads((home / "state" / "pictures" / "series-map.json").read_text(encoding="utf-8"))
    assert set(recorded) == set(window._series_folders)
    again = _window(home, runner)
    assert again._series_folders == window._series_folders


def test_expert_does_not_add_a_camera_control(app, home: Path, monkeypatch) -> None:
    prompts: list[str] = []

    def runner(job, inputs):
        del inputs
        prompts.append(job.prompt)
        return rgb()

    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    window = _window(home, runner)
    window.photoshoot.writer = lambda _prompt: _payload(IDEA)
    window.view.set_prompt_text(IDEA)
    window.set_expert(True)
    window.show()
    window.findChild(QPushButton, "photoshootMode").click()
    window.findChild(QPushButton, "generate").click()
    app.processEvents()
    assert prompts and all("50 mm" not in prompt for prompt in prompts)
    assert window.findChild(QLineEdit, "photoshootCamera") is None
    assert window.findChild(QPushButton, "photoshootMenu") is None


def test_a_blocked_frame_is_labelled_and_is_not_the_hero(app, home: Path, monkeypatch) -> None:
    modes: list[Mode] = []

    def runner(job, inputs):
        del inputs
        modes.append(job.mode)
        return rgb()

    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    window = _window(home, runner)
    gate = window.shell.gate.age_gate
    real = gate.review_image

    def review_image(raw, prompt=""):
        if "three-quarter" in prompt:
            raise GenerationBlocked("image")
        return real(raw, prompt)

    gate.review_image = review_image
    window.photoshoot.writer = lambda _prompt: _payload(IDEA)
    window.view.set_prompt_text(IDEA)
    window.show()
    window.findChild(QPushButton, "photoshootMode").click()
    window.findChild(QPushButton, "generate").click()
    app.processEvents()
    labels = [label.text() for label in window.findChildren(QLabel, "photoshootLine")]
    assert any("Shot blocked" in text for text in labels)
    assert modes[0] is Mode.TEXT
    assert modes[1] is Mode.RESTYLE
    folder = next(path for path in (home / "state" / "pictures" / "series").iterdir() if path.is_dir())
    assert not (folder / "02.png").exists()
    assert (folder / "01.png").is_file()


def test_cancel_keeps_the_frames_already_finished(app, home: Path, monkeypatch) -> None:
    prompts: list[str] = []

    def runner(job, inputs):
        del inputs
        prompts.append(job.prompt)
        if len(prompts) == 1:
            window.photoshoot.cancel_series()
        return rgb()

    monkeypatch.setattr("canvas.ui.photoshoot_run.selectable", lambda: (object(),))
    window = _window(home, runner)
    window.photoshoot.writer = lambda _prompt: _payload(IDEA)
    window.view.set_prompt_text(IDEA)
    window.show()
    window.findChild(QPushButton, "photoshootMode").click()
    window.findChild(QPushButton, "generate").click()
    app.processEvents()
    assert len(prompts) == 1
    folder = next(path for path in (home / "state" / "pictures" / "series").iterdir() if path.is_dir())
    assert (folder / "01.png").is_file()
    assert not (folder / "02.png").exists()
    cancel = window.findChild(QPushButton, "photoshootCancel")
    assert cancel is not None and cancel.text() == "Cancel"
