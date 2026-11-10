"""rc.41: the LoRA chip shows the alias (else the trigger) and inserts it; the prompt box is never rewritten."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QPlainTextEdit, QPushButton

from canvas.models.modes import Tier
from canvas.ui.shell_layout import open_layout
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import World
from tests.lora_files import write_lora
from voxprint_common.i18n import set_language


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def home(monkeypatch, tmp_path: Path):
    models = tmp_path / "models"
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(models))
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    set_language("en")
    yield tmp_path
    set_language("en")


def _window(home: Path, lora_id: str, selected: str):
    shell = open_layout(home / "state", gate=World().gate())
    window = ShellWindow(shell, tier=Tier.GB16, enlarge=object())
    window.view.set_prompt_text("a red boat")
    window._save_addons(selected, 0.7)
    window.show()
    return window


def test_the_chip_shows_and_inserts_the_alias(app, home: Path) -> None:
    del app
    models = home / "models"
    write_lora(models, "ada-1", trigger="zqxv7ada", metadata={"voxprint.subject.alias": "vox_ada"})
    window = _window(home, "ada-1", "ada-1")
    chip = window.findChild(QPushButton, "loraTrigger")
    box = window.findChild(QPlainTextEdit, "promptBox")
    assert chip is not None and chip.text() == "vox_ada"
    chip.click()
    assert box.toPlainText() == "vox_ada a red boat"
    assert "zqxv7ada" not in box.toPlainText()
    window.close()


def test_a_legacy_lora_chip_shows_and_inserts_the_trigger(app, home: Path) -> None:
    del app
    models = home / "models"
    write_lora(models, "old-1", trigger="kettleword")
    window = _window(home, "old-1", "old-1")
    chip = window.findChild(QPushButton, "loraTrigger")
    box = window.findChild(QPlainTextEdit, "promptBox")
    assert chip is not None and chip.text() == "kettleword"
    chip.click()
    assert box.toPlainText() == "kettleword a red boat"
    window.close()
