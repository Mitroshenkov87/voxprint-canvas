"""The suite GUI rule at 1280×800: one-line action row, no clipped labels, dark dialogs, anchored popups."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QWidget,
)

from canvas.models.lora_library import import_lora
from canvas.models.modes import Tier
from canvas.ui.lora_panel import ThirdPartyDialog, dark, file_dialog
from canvas.ui.shell_layout import open_layout
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import World
from tests.voxlora_files import write_voxlora
from voxprint_common.i18n import set_language


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(params=["ru", "en"])
def window(request, app, monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    set_language(request.param)
    entry = import_lora(write_voxlora(tmp_path / "in"), World().gate(), base_ready=lambda: True)
    w = ShellWindow(open_layout(tmp_path / "state", gate=World().gate()), tier=Tier.GB16, enlarge=object())
    w.resize(1280, 800)
    w.show()
    w._pick_lora(entry.id, 0.8)
    w.refresh_loras()
    w.view.refresh()
    for _ in range(6):
        app.processEvents()
    yield w
    w.close()
    shiboken6.delete(w)
    set_language("en")


def _shown(host: QWidget, kind):
    return [c for c in host.findChildren(kind) if c.isVisible()]


def test_the_action_row_is_one_line_at_1280(window):
    from PySide6.QtGui import QFontInfo

    from canvas.ui.fonts import FAMILY

    generate = window._action_row.findChild(QPushButton, "generate")
    assert QFontInfo(generate.font()).family() == FAMILY
    row = window._action_row
    buttons = _shown(row, QPushButton)
    centres = {b.mapTo(row, b.rect().center()).y() for b in buttons}
    sizes = [(b.objectName(), b.width(), b.minimumWidth(), b.sizeHint().width(), b.property("flowFloor"),
              b.fontMetrics().horizontalAdvance(b.text()), b.font().family()) for b in buttons]
    assert max(centres) - min(centres) <= 4, (row.width(), sizes)


def test_the_lora_row_is_one_line_and_flush_with_the_prompt(window):
    picker = window._lora_picker
    prompt = window.view._prompt
    left = prompt.mapToGlobal(prompt.rect().topLeft()).x()
    right = prompt.mapToGlobal(prompt.rect().topRight()).x()
    combo = picker.combo
    value = picker.value
    assert abs(combo.mapToGlobal(combo.rect().topLeft()).x() - left) <= 2
    assert abs(value.mapToGlobal(value.rect().topRight()).x() - right) <= 2
    rows = {w.mapToGlobal(w.rect().center()).y() for w in (combo, picker.slider, value)}
    assert max(rows) - min(rows) <= 2


def _clipped(widget) -> bool:
    label = widget.text().replace("&", "")
    if not label:
        return False
    room = widget.contentsRect().width()
    if isinstance(widget, QPushButton):
        room -= 8
    return widget.fontMetrics().horizontalAdvance(label) > room


def test_no_label_is_clipped(window):
    for host in (window._action_row, window._lora_picker):
        for widget in _shown(host, QPushButton) + _shown(host, QLabel):
            assert not _clipped(widget), widget.text()


def _is_dark(widget: QWidget) -> bool:
    widget.ensurePolished()
    return widget.palette().window().color().lightnessF() < 0.3


def test_every_lora_dialog_is_dark(app):
    set_language("ru")
    busy = QProgressDialog("…", "", 0, 0)
    refused = QMessageBox()
    for d in (busy, refused):
        dark(d)
    dialogs = [ThirdPartyDialog(None, "x"), file_dialog(None), busy, refused]
    try:
        for d in dialogs:
            assert _is_dark(d), d.objectName() or type(d).__name__
        more = dialogs[0].findChild(QPushButton, "loraThirdPartyMore")
        details = dialogs[0].findChild(QLabel, "loraThirdPartyDetails")
        assert details.isHidden()
        more.setChecked(True)
        assert not details.isHidden() and details.wordWrap()
    finally:
        for d in dialogs:
            shiboken6.delete(d)
        set_language("en")


def test_menus_open_from_their_buttons(window):
    row = window._action_row
    menus = [b for b in _shown(row, QPushButton) if b.menu() is not None]
    assert len(menus) >= 2
    assert window.view._overflow.menu() is not None
    combo = window._lora_picker.findChild(QComboBox, "loraPick")
    assert "combobox-popup: 0" in combo.styleSheet()



def test_the_bundled_font_is_registered(app):
    from canvas.ui.fonts import FAMILY, FONT_DIR, FONT_FILES, register_fonts

    assert register_fonts()
    assert all((FONT_DIR / name).is_file() for name in FONT_FILES)
    assert "Open Font License" in (FONT_DIR / "OFL.txt").read_text(encoding="utf-8")
    assert FAMILY == "Noto Sans"


def test_the_file_dialog_reads_in_russian(app, monkeypatch):
    from canvas.ui.fonts import qt_translations, translations_dir

    assert (translations_dir() / "qtbase_ru.qm").is_file()
    monkeypatch.setenv("LANG", "ru_RU.UTF-8")
    monkeypatch.setenv("LC_ALL", "ru_RU.UTF-8")
    set_language("ru")
    try:
        assert qt_translations() == "ru"
        dialog = file_dialog(None)
        labels = " ".join(w.text() for w in dialog.findChildren(QLabel)) + " ".join(
            b.text() for b in dialog.findChildren(QPushButton))
        assert "Look in" not in labels and "Отмена" in labels, labels
        shiboken6.delete(dialog)
    finally:
        set_language("en")
        qt_translations("en")
