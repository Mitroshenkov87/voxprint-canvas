"""The LoRA library screen and the generate panel's pick and strength control."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QPushButton, QSlider

from canvas.app.i18n import text
from canvas.models.lora_library import import_lora, list_library
from canvas.models.modes import Tier
from canvas.ui import lora_panel
from canvas.ui.lora_panel import LibraryDialog
from canvas.ui.shell_layout import open_layout
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import World
from tests.voxlora_files import manifest, write_voxlora
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


def _entry(home: Path, **record):
    return import_lora(write_voxlora(home / "in", record=manifest(**record)), World().gate(), base_ready=lambda: True)


def test_the_library_lists_previews_and_removes(app, home: Path):
    entry = _entry(home)
    changed: list[int] = []
    dialog = LibraryDialog(None, World().gate(), lambda: changed.append(1))
    card = dialog.findChild(QLabel, "loraName")
    assert card is not None and card.text() == "Brass Lamp"
    assert dialog.findChild(QLabel, "loraPreview-0") is not None
    dialog.findChild(QPushButton, "loraRemove").click()
    assert list_library() == () and changed == [1]
    assert entry.id not in [item.id for item in list_library()]
    shiboken6.delete(dialog)


def test_a_refused_import_shows_the_catalog_line(app, home: Path):
    package = write_voxlora(home / "in", record=manifest(kind="vehicle"))
    dialog = LibraryDialog(None, World().gate(), base_ready=lambda: True)
    assert dialog.run_import(package) == "lora.kind"
    assert dialog.findChild(QLabel, "loraStatus").text() == text("lora.kind")
    assert list_library() == ()
    shiboken6.delete(dialog)


def test_the_pick_follows_the_library_and_the_strength_range(app, home: Path, monkeypatch):
    entry = _entry(home, scale_min=0.1, scale_max=1.0, scale_default=0.6)
    window = ShellWindow(open_layout(home / "state", gate=World().gate()), tier=Tier.GB16, enlarge=object())
    window.resize(1280, 800)
    window.show()
    app.processEvents()
    picker = window._lora_picker
    assert picker.isVisible()
    combo = window.findChild(QComboBox, "loraPick")
    assert [combo.itemText(index) for index in range(combo.count())] == [text("canvas.lora.none"), "Brass Lamp"]
    combo.setCurrentIndex(1)
    slider = window.findChild(QSlider, "loraStrength")
    assert (slider.minimum(), slider.maximum(), slider.value()) == (10, 100, 60)
    assert window.shell.layout.get("shell.addons") == entry.id
    assert window.lora_trigger() == "brsslmp"
    slider.setValue(90)
    assert window.shell.layout.get("shell.lora_weight") == pytest.approx(0.9)
    combo.setCurrentIndex(0)
    assert window.shell.layout.get("shell.addons") == ""
    window.close()


def test_the_window_import_adds_to_the_pick(app, home: Path, monkeypatch):
    monkeypatch.setattr(
        lora_panel, "import_lora", lambda path, gate, **options: import_lora(path, gate, base_ready=lambda: True, **options),
    )
    shown: list[str] = []
    monkeypatch.setattr(lora_panel.ImportedDialog, "exec", lambda self: shown.append("shown"))
    window = ShellWindow(open_layout(home / "state", gate=World().gate()), tier=Tier.GB16, enlarge=object())
    window.show()
    app.processEvents()
    assert not window._lora_picker.isVisible()
    result = window.import_lora(write_voxlora(home / "in"))
    assert not isinstance(result, str) and shown == ["shown"]
    assert window._lora_picker.combo.count() == 2
    window.close()


def test_an_import_from_add_ons_replaces_the_empty_line(app, home: Path):
    from canvas.models.user_lora import LoraIndex
    from canvas.ui.wizards import AddonsWizard

    entry = _entry(home)
    wizard = AddonsWizard(None, LoraIndex((), ()), "", 0.7, lambda _key, _strength: None, entry.as_lora)
    assert wizard.findChild(QLabel, "addonsEmpty") is not None
    wizard.findChild(QPushButton, "addonsImport").click()
    app.processEvents()
    assert wizard._empty is None
    shiboken6.delete(wizard)


def test_a_strength_step_does_not_redraw_the_window(app, home: Path, monkeypatch):
    entry = _entry(home)
    window = ShellWindow(open_layout(home / "state", gate=World().gate()), tier=Tier.GB16, enlarge=object())
    window.show()
    app.processEvents()
    window.findChild(QComboBox, "loraPick").setCurrentIndex(1)
    assert window.shell.layout.get("shell.addons") == entry.id
    calls: list[int] = []
    monkeypatch.setattr(window.view, "refresh", lambda: calls.append(1))
    window.findChild(QSlider, "loraStrength").setValue(50)
    assert calls == [] and window.shell.layout.get("shell.lora_weight") == pytest.approx(0.5)
    window.close()
