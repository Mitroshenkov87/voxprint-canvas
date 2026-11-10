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


@pytest.fixture
def window(app, monkeypatch, tmp_path: Path):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    set_language("en")
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

    for button in _shown(window._action_row, QPushButton) + _shown(window._lora_picker, QPushButton):
        assert QFontInfo(button.font()).family() == FAMILY, button.objectName()
    row = window._action_row
    options = window._action_options
    buttons = _shown(options, QPushButton)
    centres = {b.mapTo(row, b.rect().center()).y() for b in buttons}
    sizes = [(b.objectName(), b.width(), b.minimumWidth(), b.sizeHint().width(), b.property("flowFloor"),
              b.fontMetrics().horizontalAdvance(b.text()), b.font().family()) for b in buttons]
    assert max(centres) - min(centres) <= 4, (row.width(), sizes)
    # The line is centred, and the large Generate sits centred under it.
    left = min(b.mapTo(row, b.rect().topLeft()).x() for b in buttons)
    right = max(b.mapTo(row, b.rect().topRight()).x() for b in buttons)
    assert abs((left + right) / 2 - row.rect().center().x()) <= 4, (left, right, row.width())
    generate = row.findChild(QPushButton, "generate")
    top = generate.mapTo(row, generate.rect().topLeft())
    assert top.y() > max(b.mapTo(row, b.rect().bottomLeft()).y() for b in buttons)
    assert abs(generate.mapTo(row, generate.rect().center()).x() - row.rect().center().x()) <= 2
    assert generate.height() > max(b.height() for b in buttons)


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


def _process(window, app, times: int = 6) -> None:
    for _ in range(times):
        app.processEvents()


def test_use_a_picture_sits_left_of_improve_across_modes(window, app):
    """Use a picture stays on the Improve prompt row, immediately to its left."""
    picture = window.findChild(QPushButton, "actionPicture")
    improve = window.findChild(QPushButton, "improvePrompt")
    assert picture is not None and improve is not None and picture.isVisible()
    assert picture.parentWidget() is not None
    assert window._action_options.isAncestorOf(picture)

    def geometry():
        _process(window, app)
        left = picture.mapTo(window, picture.rect().topLeft())
        right = picture.mapTo(window, picture.rect().topRight())
        other = improve.mapTo(window, improve.rect().topLeft())
        return left.x(), left.y(), picture.height(), improve.height(), other.x() - right.x(), other.y()

    positions = [geometry()]
    # A quality preset change must not move the button.
    for button in window.findChildren(QPushButton):
        if button.text() == "Best" and button.isVisible():
            button.click()
            break
    positions.append(geometry())
    window.set_expert(True)
    positions.append(geometry())
    window.set_expert(False)
    positions.append(geometry())
    for item in positions:
        x, y, height, improve_height, gap, improve_y = item
        assert height == improve_height == 36, item
        assert abs(y - improve_y) <= 2, item
        assert 0 <= gap <= 16, item  # immediate left, 8 px row spacing
    xs = {item[0] for item in positions}
    ys = {item[1] for item in positions}
    assert len(xs) == 1 and len(ys) == 1, positions


def test_the_lora_slider_row_stays_put_across_modes_and_values(window, app):
    """Slider, value label, and trigger chip keep their x and width in every mode."""
    picker = window._lora_picker
    chip = window._lora_chip

    def geometry():
        _process(window, app)
        def box(widget):
            pos = widget.mapTo(window, widget.rect().topLeft())
            return pos.x(), pos.y(), widget.width(), widget.height()
        return box(picker.slider), box(picker.value), box(chip)

    start = geometry()
    # min and max strength must not move the value column
    picker.slider.setValue(picker.slider.minimum())
    at_min = geometry()
    picker.slider.setValue(picker.slider.maximum())
    at_max = geometry()
    picker.slider.setValue(70)
    # A quality preset, Expert on, and Expert off.
    for button in window.findChildren(QPushButton):
        if button.text() == "Best" and button.isVisible():
            button.click()
            break
    preset = geometry()
    window.set_expert(True)
    expert = geometry()
    window.set_expert(False)
    simple = geometry()
    # Hide the trigger word the way a LoRA without one would: the reserved
    # column must still hold the slider and value in place.
    picker.show_chip(chip, "")
    hidden = geometry()
    picker.show_chip(chip, "brsslmp")
    restored = geometry()
    rows = (start, at_min, at_max, preset, expert, simple, hidden, restored)
    # Compare slider x/w and value x/w only (chip opacity may change, not its box).
    slider_boxes = {item[0] for item in rows}
    value_boxes = {item[1] for item in rows}
    chip_boxes = {item[2][:3] for item in rows}  # x, y, width
    assert len(slider_boxes) == 1, rows
    assert len(value_boxes) == 1, rows
    assert len(chip_boxes) == 1, rows


def _prompt_button_geometry(window):
    """x, y, width for every visible named button in the left prompt panel."""
    panel = window.view.root.findChild(QWidget, "promptPanel")
    assert panel is not None
    boxes = {}
    for button in panel.findChildren(QPushButton):
        if not button.isVisible() or not button.objectName():
            continue
        pos = button.mapTo(window, button.rect().topLeft())
        boxes[button.objectName()] = (pos.x(), pos.y(), button.width())
    return boxes


def test_button_rows_keep_geometry_when_expert_toggles(window, app):
    """Opening and closing Expert must not move or resize the prompt-panel buttons."""
    def cycle():
        _process(window, app)
        before = _prompt_button_geometry(window)
        window.set_expert(True)
        _process(window, app)
        opened = _prompt_button_geometry(window)
        window.set_expert(False)
        _process(window, app)
        closed = _prompt_button_geometry(window)
        assert before == opened, (before, opened)
        assert before == closed, (before, closed)
        return before

    off = cycle()
    assert off
    window.shell.layout.set_adult_content(True)
    window.view.refresh()
    _process(window, app)
    on = cycle()
    assert on


def test_the_lora_picker_survives_the_adult_switch_in_every_mode(window, app):
    """The LoRA pick stays visible and keeps the selected LoRA when 18+ toggles."""
    picker = window._lora_picker
    assert picker is not None
    selected = window.shell.layout.get("shell.addons")
    strength = window.shell.layout.get("shell.lora_weight")
    assert selected and float(strength) > 0

    def check(mode: str) -> None:
        _process(window, app)
        assert picker.isVisible() and picker.isEnabled(), mode
        assert picker.combo.isVisible() and picker.combo.isEnabled(), mode
        assert window.shell.layout.get("shell.addons") == selected, mode
        assert window.shell.layout.get("shell.lora_weight") == strength, mode

    # simple, a quality preset, Expert; 18+ off then on, then off again.
    check("simple-off")
    window.shell.layout.set_adult_content(True)
    window.view.refresh()
    check("simple-on")
    for button in window.findChildren(QPushButton):
        if button.text() == "Best" and button.isVisible():
            button.click()
            break
    check("preset-on")
    window.set_expert(True)
    check("expert-on")
    window.shell.layout.set_adult_content(False)
    window.view.refresh()
    check("expert-off")
    window.set_expert(False)
    check("simple-off-again")
    window.shell.layout.set_adult_content(True)
    window.view.refresh()
    window.shell.layout.set_adult_content(False)
    window.view.refresh()
    check("off-on-off")
    assert window.shell.layout.get("shell.addons") == selected
    assert window.shell.layout.get("shell.lora_weight") == strength
