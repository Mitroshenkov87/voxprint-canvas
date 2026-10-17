"""Photoshoot is a panel on the shared screen. Its Create button is the only primary."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton

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


def test_photoshoot_chips_replace_the_single_picture_count(app, home: Path) -> None:
    window = _window(home, lambda job, inputs: rgb())
    window.show()
    app.processEvents()
    mode = window.findChild(QPushButton, "photoshootMode")
    assert mode is not None and mode.text() == "Photoshoot"
    assert window.findChild(QPushButton, "generate").isVisible()
    mode.click()
    app.processEvents()
    assert window.findChild(QPushButton, "chip-photoshoot-count-6").isChecked()
    assert window.findChild(QPushButton, "chip-photoshoot-style-photo").isChecked()
    assert window.findChild(QPushButton, "chip-photoshoot-place-same").isChecked()
    assert not window.findChild(QPushButton, "chip-count-1").isVisible()
    assert not window.findChild(QPushButton, "chip-style-illustration").isVisible()
    assert window.findChild(QPushButton, "photoshootWrite") is None
    assert not window.findChild(QPushButton, "generate").isVisible()
    primary = window.findChild(QPushButton, "photoshootCreate")
    assert primary is not None and primary.isVisible() and primary.text() == "Generate"


def test_generate_writes_the_series_without_a_lens_in_simple_mode(app, home: Path, monkeypatch) -> None:
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
    window.findChild(QPushButton, "photoshootCreate").click()
    app.processEvents()
    assert len(prompts) == 6
    assert all("50 mm" not in prompt for prompt in prompts)
    assert all(IDEA in prompt for prompt in prompts)
    menus = window.findChildren(QPushButton, "photoshootMenu")
    handles = window.findChildren(QPushButton, "photoshootHandle")
    assert menus and all(menu.isVisible() for menu in menus)
    assert handles and all(handle.isVisible() for handle in handles)
    assert all(not camera.isVisible() for camera in window.findChildren(QLineEdit, "photoshootCamera"))
    assert not window.findChild(QPushButton, "generate").isVisible()
    assert window.findChild(QPushButton, "photoshootSheet").isEnabled()
    assert (home / "state" / "pictures" / "series").exists()
    assert window._series_folders
    assert all(path.endswith(".png") for path in window._pictures.values())


def test_expert_keeps_the_camera_on_the_line(app, home: Path, monkeypatch) -> None:
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
    window.findChild(QPushButton, "photoshootCreate").click()
    app.processEvents()
    assert prompts and all("50 mm" in prompt for prompt in prompts)
    cameras = window.findChildren(QLineEdit, "photoshootCamera")
    assert cameras and all(camera.isVisible() for camera in cameras)
    assert all(button.isVisible() for button in window.findChildren(QPushButton, "photoshootMenu"))
    assert not window.findChild(QPushButton, "generate").isVisible()


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
    window.findChild(QPushButton, "photoshootCreate").click()
    app.processEvents()
    cells = [cell.text() for cell in window.findChildren(QLabel, "photoshootCell")]
    assert "Shot blocked" in cells
    assert modes[0] is Mode.TEXT
    assert modes[1] is Mode.RESTYLE
    folder = next(path for path in (home / "state" / "pictures" / "series").iterdir())
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
    window.findChild(QPushButton, "photoshootCreate").click()
    app.processEvents()
    assert len(prompts) == 1
    folder = next(path for path in (home / "state" / "pictures" / "series").iterdir())
    assert (folder / "01.png").is_file()
    assert not (folder / "02.png").exists()
