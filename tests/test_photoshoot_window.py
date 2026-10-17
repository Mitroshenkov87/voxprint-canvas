"""Photoshoot is an extended Improve mode, not a second primary button."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton

from canvas.models.modes import Tier
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
    window.findChild(QPushButton, "generate").click()
    app.processEvents()
    assert len(prompts) == 6
    assert all("50 mm" not in prompt for prompt in prompts)
    assert all(IDEA in prompt for prompt in prompts)
    menus = window.findChildren(QPushButton, "photoshootMenu")
    assert menus and all(not menu.isVisible() for menu in menus)
    assert window.findChild(QPushButton, "photoshootSheet").isEnabled()
    assert (home / "state" / "pictures" / "series").exists()


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
    window.findChild(QPushButton, "generate").click()
    app.processEvents()
    assert prompts and all("50 mm" in prompt for prompt in prompts)
    cameras = window.findChildren(QLineEdit, "photoshootCamera")
    assert cameras and all(camera.isVisible() for camera in cameras)
    assert all(button.isVisible() for button in window.findChildren(QPushButton, "photoshootMenu"))
