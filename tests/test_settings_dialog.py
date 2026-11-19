"""Settings opens a real Settings dialog, and About stays its own item (rc.45 finding 1)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QLabel, QPushButton, QSpinBox

from canvas.models.modes import Tier
from canvas.ui.settings_dialog import SettingsDialog
from tests.worker_support import inline_window


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    yield tmp_path


def _window(home: Path):
    return inline_window(home / "state", tier=Tier.GB16, enlarge=object())


def test_settings_is_a_real_dialog_not_about(app, home: Path, monkeypatch) -> None:
    window = _window(home)
    executed: list[str] = []
    monkeypatch.setattr(SettingsDialog, "exec", lambda self: executed.append(self.objectName()) or 0)
    window._on_command("view.settings")
    assert executed == ["settingsDialog"]
    assert window.findChild(QDialog, "settingsDialog") is None or not window.findChild(QDialog, "settingsDialog").isVisible()


def test_the_settings_dialog_edits_the_stored_values(app, home: Path) -> None:
    window = _window(home)
    layout = window.shell.layout
    dialog = window.settings_dialog()
    assert dialog.findChild(QLabel, "settingsTitle") is not None
    assert dialog.findChild(QPushButton, "settingsDone") is not None
    # The picture file type writes the stored enum.
    combo = dialog._format
    avif = combo.findText("AVIF")
    combo.setCurrentIndex(avif)
    assert layout.get("canvas.format") == "avif"
    # The seed box is the plain opposite of the stored lock and keeps its meaning.
    seed = dialog._dice
    assert seed.isChecked()
    seed.setChecked(False)
    assert layout.seed_locked() is True
    seed.setChecked(True)
    assert layout.seed_locked() is False
    # The seed default is the locked seed the next run starts from.
    box = dialog._seed
    assert not box.isEnabled()
    seed.setChecked(False)
    assert box.isEnabled()
    box.setValue(1234)
    assert layout.base_seed() == 1234
    seed.setChecked(True)
    # The 18+ switch writes the stored adult setting, default off and unchanged in meaning.
    adult = dialog._adult
    assert not adult.isChecked()
    adult.setChecked(True)
    assert layout.adult_content() is True
    assert layout.get("image.adult_content") is True
    adult.setChecked(False)
    assert layout.adult_content() is False
    # Expert mode follows the window toggle.
    expert = dialog._expert
    assert not expert.isChecked()
    expert.setChecked(True)
    assert window._expert.enabled() is True
    expert.setChecked(False)
    assert window._expert.enabled() is False
    dialog.close()
    dialog.deleteLater()
    window.close()


def test_about_stays_its_own_help_menu_item(app, home: Path) -> None:
    window = _window(home)
    help_menu = window.help_menu
    names = [action.objectName() for action in help_menu.actions()]
    assert "actionAbout" in names
    about = next(action for action in help_menu.actions() if action.objectName() == "actionAbout")
    assert about.text().startswith("About")
    window.close()
