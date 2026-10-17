"""The new window keeps the picture tools and does not offer instruction editing."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QPlainTextEdit, QPushButton, QSpinBox, QWidget

from canvas.app.i18n import text
from canvas.models.modes import Tier
from canvas.ui.shell_layout import open_layout
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import World
from voxprint_common.i18n import set_language
from voxprint_common.ui.simple_image import Phase


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


def _window(home: Path, *, runner=None, improver=None) -> ShellWindow:
    shell = open_layout(home / "state", World().gate(), enlarge_installed=False)
    return ShellWindow(shell, tier=Tier.GB16, enlarge=object(), runner=runner, improver=improver)


def test_simple_mode_hides_expert_words_and_keeps_picture_tools(app, home: Path) -> None:
    window = _window(home)
    window.show()
    app.processEvents()
    assert window.view.root.findChild(QPushButton, "shell.steps") is None
    setup = window.setup_wizard()
    assert window.findChild(QPushButton, "chip-change-restyle") is None
    assert setup.findChild(QPushButton, "setup-change-restyle") is not None
    assert setup.findChild(QPushButton, "setup-change-inpaint") is not None
    assert setup.findChild(QPushButton, "setup-change-pose") is not None
    assert setup.findChild(QPushButton, "setup-change-depth") is not None
    assert setup.findChild(QPushButton, "setup-change-edge") is not None
    assert setup.findChild(QPushButton, "setup-instruction-on") is None
    setup.findChild(QPushButton, "setup-change-inpaint").click()
    assert window.shell.layout.get("canvas.change") == "inpaint"
    assert window.findChild(QPushButton, "chip-instruction-on") is None
    fourk = window.findChild(QPushButton, "chip-output-4k")
    assert fourk is not None and not fourk.isEnabled()
    reason = window.findChild(QLabel, "reason-output-4k")
    assert reason is not None and reason.text()
    footer = window.findChild(QLabel, "footer")
    assert footer is not None and footer.text() == text("notice.responsibility")
    window.set_expert(True)
    app.processEvents()
    spin = window.findChild(QSpinBox, "shell.steps")
    assert spin is not None and spin.isVisible()
    spin.setValue(33)
    window.set_expert(False)
    app.processEvents()
    panel = window.findChild(QWidget, "expertPanel")
    assert panel is not None and not panel.isVisible()
    window.set_expert(True)
    app.processEvents()
    assert window.findChild(QSpinBox, "shell.steps").value() == 33


def test_improve_and_revert_stay_in_the_same_field(app, home: Path) -> None:
    del app
    window = _window(home, improver=lambda prompt: prompt + " in warm light")
    window.view.set_prompt_text("a red bicycle")
    improve = window.findChild(QPushButton, "improvePrompt")
    revert = window.findChild(QPushButton, "revertPrompt")
    assert improve is not None and revert is not None
    improve.click()
    box = window.findChild(QPlainTextEdit, "promptBox")
    assert "warm light" in box.toPlainText()
    revert.click()
    assert box.toPlainText() == "a red bicycle"


def test_generate_without_models_says_they_are_not_ready(app, home: Path) -> None:
    del app
    window = _window(home)
    window.view.set_prompt_text("a red bicycle")
    window.findChild(QPushButton, "generate").click()
    assert window.shell.layout.phase is Phase.UNAVAILABLE
    assert window.shell.layout.reason == text("canvas.models.missing")


def test_mask_and_picture_wizards_open(app, home: Path, tmp_path: Path) -> None:
    del app
    window = _window(home)
    picture = window.picture_wizard()
    assert picture.findChild(QPushButton, "pictureOpen") is not None
    path = tmp_path / "dot.png"
    from PIL import Image

    Image.new("RGB", (8, 8), (20, 30, 40)).save(path)
    picture.load_path(path)
    picture.findChild(QPushButton, "pictureUse").click()
    assert window._picture == path
    assert window.shell.layout.get("canvas.source") == "picture"
    mask = window.mask_wizard()
    board = mask.findChild(QWidget, "maskSurface")
    assert board is not None
    mask.findChild(QPushButton, "maskUse").click()
    assert window._mask is not None
    assert window.shell.layout.get("canvas.change") == "inpaint"
    addons = window.addons_wizard()
    assert addons.findChild(QLabel, "thirdPartyNotice").text() == text("notice.third_party")
    check = window.diagnostics_wizard()
    assert check.findChild(QPushButton, "checkRun") is not None
    assert check.findChild(QLabel, "checkReport").text() == ""


def test_the_new_screen_does_not_mention_temporary_editing() -> None:
    root = Path("canvas/ui")
    blob = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("shell_*.py"))
    blob += "\n" + (root / "wizards.py").read_text(encoding="utf-8")
    assert "edit.unavailable" not in blob
    assert "canvas.instruction" not in blob
    ids = json.loads(Path("canvas/mdl.json").read_text(encoding="utf-8"))
    folded = blob.casefold()
    for row in ids["models"]:
        assert row["id"].casefold() not in folded


def test_ui_sources_do_not_name_the_placeholder_models() -> None:
    blob = "\n".join(path.read_text(encoding="utf-8") for path in Path("canvas/ui").rglob("*.py")).casefold()
    catalog = json.loads(Path("canvas/mdl.json").read_text(encoding="utf-8"))
    for row in catalog["models"]:
        assert row["id"].casefold() not in blob
