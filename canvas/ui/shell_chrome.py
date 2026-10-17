"""Place the shared screen in the suite's dark theme.

Colours come from the shared tokens. This module does not invent a palette.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFontMetrics
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSplitter, QVBoxLayout, QWidget

from voxprint_common.i18n import tr
from voxprint_common.ui.presets import PresetMatch
from voxprint_common.ui.simple_image import Phase
from voxprint_common.ui.theme import (
    CARD_PLAIN,
    CONTROL_BG,
    CONTROL_BORDER,
    PRIMARY,
    PRIMARY_BORDER,
    ROOT_PLAIN,
    TEXT,
    TEXT_ON_PRIMARY,
    build_shell_style,
)

from canvas.app.i18n import text

PANEL_WIDTH = 320
SHOT_WIDTH = 1280
SHOT_HEIGHT = 800


def shell_style() -> str:
    """The shared shell stylesheet, with the window and labels kept on the dark ground."""
    return build_shell_style(False) + f"""
QMainWindow#shell, QWidget#shellColumn, QWidget#simpleRoot, QDialog {{
    background: {ROOT_PLAIN}; color: {TEXT};
}}
QLabel {{
    color: {TEXT}; background: transparent;
}}
QLabel#footer, QLabel#emptyHint, QLabel[presetLabel="true"], QLabel#maskNeed,
QLabel#thirdPartyNotice, QLabel#picturePreview {{
    color: {TEXT};
}}
QLabel#emptyHint {{ font-size: 18px; }}
QFrame#stage, QFrame#expertPanel, QScrollArea#historyStrip {{
    background: {CARD_PLAIN}; color: {TEXT};
}}
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QPlainTextEdit {{
    background: {CONTROL_BG}; color: {TEXT};
    border: 1px solid {CONTROL_BORDER};
}}
QPushButton[chip="true"]:disabled {{
    color: {TEXT}; background: {CONTROL_BG}; border-color: {CONTROL_BORDER};
}}
QPushButton#expertResetAll:disabled {{
    color: {TEXT}; background: {CONTROL_BG};
}}
QLabel#expertChip {{
    color: {TEXT_ON_PRIMARY}; background: {PRIMARY};
    border: 1px solid {PRIMARY_BORDER}; border-radius: 8px; padding: 2px 8px;
}}
"""


def mount(window):
    """Put the expert panel on the right, at the full height of the window.

    Args:
        window: The shell window. Its view is already built.

    Returns:
        The horizontal split. The picture column is the first child.
    """
    view = window.view
    panel = view._expert_panel
    panel.setMinimumWidth(0)
    panel.setMaximumWidth(PANEL_WIDTH)
    column = QWidget()
    column.setObjectName("shellColumn")
    layout = QVBoxLayout(column)
    layout.setContentsMargins(12, 8, 12, 8)
    layout.setSpacing(8)
    _toolbar(window, view._top)
    layout.addWidget(view.root, 1)
    footer = QLabel(text("notice.responsibility"))
    footer.setObjectName("footer")
    footer.setWordWrap(True)
    layout.addWidget(footer)
    split = QSplitter(Qt.Orientation.Horizontal)
    split.setObjectName("shellSplit")
    split.setChildrenCollapsible(False)
    split.addWidget(column)
    split.addWidget(panel)
    split.setStretchFactor(0, 1)
    split.setStretchFactor(1, 0)
    _menu(window, view._overflow.menu())
    _status(window)
    _tighten(window)
    scroll = view.root.findChild(QWidget, "expertScroll")
    if scroll is not None:
        scroll.setMinimumHeight(0)
    return split


def fit_panel(window, enabled):
    """Keep the expert panel at a fixed width, or give that width back.

    Args:
        window: The shell window.
        enabled: True when Expert mode is on.
    """
    panel = window.view._expert_panel
    panel.setMinimumWidth(PANEL_WIDTH if enabled else 0)
    panel.setMaximumWidth(PANEL_WIDTH)
    if not enabled:
        return
    _reap(window)
    _status(window)
    width = max(1, window.width() - PANEL_WIDTH)
    window._split.setSizes([width, PANEL_WIDTH])


def polish(window):
    """Apply the size menu, the 4K wording, and the upscale row after a refresh.

    Args:
        window: The shell window.
    """
    _size_menu(window)
    _output(window)
    _detail(window)
    _mask_button(window)
    _labels(window)
    _result(window)
    if window.view._expert_panel.isVisible():
        _reap(window)
        _status(window)


def _toolbar(window, row):
    picture = QPushButton(text("canvas.action.picture"))
    picture.setObjectName("actionPicture")
    picture.clicked.connect(window._open_picture)
    mask = QPushButton(text("canvas.action.mask"))
    mask.setObjectName("actionMask")
    mask.clicked.connect(window._open_mask)
    mask.hide()
    window._picture_button = picture
    window._mask_button = mask
    row.insertWidget(0, mask)
    row.insertWidget(0, picture)


def _menu(window, menu):
    specs = (
        ("actionSetup", "canvas.action.setup", window._open_setup),
        ("actionAddons", "canvas.action.addons", window._open_addons),
        ("actionCheck", "canvas.action.check", window._open_check),
    )
    existing = menu.actions()
    before = existing[0] if existing else None
    for name, key, slot in specs:
        action = QAction(text(key), menu)
        action.setObjectName(name)
        action.triggered.connect(slot)
        menu.insertAction(before, action)


def _status(window):
    panel_layout = window.view._panel_layout
    badge = getattr(window, "_expert_badge", None)
    if badge is None:
        row = QWidget()
        row.setObjectName("expertStatus")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 8)
        badge = QLabel()
        badge.setObjectName("expertChip")
        reset = QPushButton(text("canvas.expert.reset"))
        reset.setObjectName("expertResetAll")
        reset.clicked.connect(lambda: _reset(window))
        layout.addWidget(badge)
        layout.addWidget(reset, 1)
        window._expert_badge = badge
        window._expert_reset = reset
        panel_layout.insertWidget(0, row)
    custom = _custom(window)
    window._expert_badge.setText(text("canvas.expert.custom") if custom else text("canvas.expert.preset"))
    window._expert_reset.setEnabled(custom)


def _reset(window):
    for row in window.shell.layout.rows:
        window.shell.layout.reset_group(row.row_id)
    window.view.refresh()


def _custom(window):
    for row in window.shell.layout.rows:
        status = window.shell.layout.preset_status(row.row_id)
        if status is not None and status.kind is not PresetMatch.EXACT:
            return True
    return False


def _size_menu(window):
    root = window.view.root
    for button in root.findChildren(QPushButton):
        if button.objectName().startswith("chip-size-"):
            button.hide()
    combo = root.findChild(QComboBox, "sizeRatio")
    label = root.findChild(QLabel, "preset-size")
    if label is None:
        return
    if combo is None:
        combo = QComboBox()
        combo.setObjectName("sizeRatio")
        combo.setMinimumWidth(88)
        combo.setMaximumWidth(120)
        combo.currentIndexChanged.connect(lambda _index: _choose_ratio(window, combo))
        _insert_after_orientation(label, combo)
    row = next(item for item in window.shell.layout.rows if item.row_id == "size")
    selected = str(window.shell.layout.presets.get("size", row.default))
    current = next((choice for choice in row.choices if choice.value == selected), row.choices[0])
    combo.blockSignals(True)
    combo.clear()
    for choice in row.choices:
        if choice.group_key == current.group_key:
            combo.addItem(tr(choice.label_key), choice.value)
    index = combo.findData(selected)
    combo.setCurrentIndex(max(0, index))
    combo.blockSignals(False)


def _insert_after_orientation(label, combo):
    host = label.parentWidget().layout()
    if host is None:
        return
    for index in range(host.count()):
        line = host.itemAt(index).layout()
        if line is None:
            continue
        at = _label_index(line, label)
        if at < 0:
            continue
        insert = at + 1
        while insert < line.count():
            widget = line.itemAt(insert).widget()
            if widget is None or not str(widget.objectName()).startswith("orient-"):
                break
            insert += 1
        line.insertWidget(insert, combo)
        return


def _label_index(line, label):
    for index in range(line.count()):
        if line.itemAt(index).widget() is label:
            return index
    return -1


def _choose_ratio(window, combo):
    value = combo.currentData()
    if not isinstance(value, str) or not value:
        return
    if window.shell.layout.presets.get("size") == value:
        return
    window.shell.layout.choose("size", value)
    window.view.refresh()


def _output(window):
    button = window.view.root.findChild(QPushButton, "chip-output-4k")
    reason = window.view.root.findChild(QLabel, "reason-output-4k")
    if reason is not None:
        reason.hide()
    if button is None or button.isEnabled():
        return
    button.setText(text("canvas.output.4k.needs"))
    _reserve(button, 36)


def _detail(window):
    show = str(window.shell.layout.presets.get("output", "")) == "4k"
    root = window.view.root
    names = ("preset-detail", "chip-detail-light", "chip-detail-medium", "chip-detail-strong")
    for name in names:
        widget = root.findChild(QWidget, name)
        if widget is not None:
            widget.setVisible(show)


def _mask_button(window):
    button = getattr(window, "_mask_button", None)
    if button is not None:
        button.setVisible(window._picture is not None)


def _tighten(window):
    """Give the picture the height the prompt tools and chip gaps were using."""
    view = window.view
    rows = view._preset_rows
    rows.setSpacing(4)
    root = view.root.layout()
    if root is not None:
        root.setSpacing(6)
    prompt = view._prompt
    if prompt is not None:
        prompt.setFixedHeight(64)
    if getattr(window, "_prompt_row", False):
        return
    improve = getattr(view, "_improve", None)
    revert = getattr(view, "_revert", None)
    if improve is None or revert is None:
        return
    tools = _owner(view.root.layout(), improve)
    column = _owner(view.root.layout(), prompt)
    if tools is None or column is None or tools is column:
        return
    row = _owner_layout(view.root.layout(), column)
    if row is None:
        return
    tools.removeWidget(improve)
    tools.removeWidget(revert)
    _drop_empty(column)
    at = row.count()
    for index in range(row.count()):
        widget = row.itemAt(index).widget()
        if widget is not None and widget.objectName() == "generate":
            at = index
            break
    row.insertWidget(at, improve)
    row.insertWidget(at + 1, revert)
    window._prompt_row = True


def _result(window):
    bar = window.view.root.findChild(QWidget, "actionBar")
    if bar is None:
        return
    bar.setVisible(window.shell.layout.phase is not Phase.EMPTY)


def _labels(window):
    root = window.view.root
    for label in root.findChildren(QLabel):
        if not label.isVisible() or not label.property("presetLabel"):
            continue
        words = label.text()
        if not words:
            continue
        _reserve(label, 12)


def _reserve(widget, pad):
    words = widget.text()
    if not words:
        return
    width = QFontMetrics(widget.font()).horizontalAdvance(words) + pad
    if widget.minimumWidth() < width:
        widget.setMinimumWidth(width)


def _reap(window):
    """Drop expert fields that a second fill left behind.

    The shared form clears only the widgets it added directly. A second fill
    leaves the previous row's fields in place, at the default size, over the
    live values.
    """
    form = window.findChild(QWidget, "expertForm")
    if form is None:
        return
    keep = set()
    _collect(form.layout(), keep)
    for child in form.findChildren(QWidget):
        if child.parentWidget() is not form or child in keep:
            continue
        child.hide()
        child.setParent(None)
        child.deleteLater()


def _collect(layout, keep):
    if layout is None:
        return
    for index in range(layout.count()):
        item = layout.itemAt(index)
        widget = item.widget()
        if widget is not None:
            keep.add(widget)
            _collect(widget.layout(), keep)
        _collect(item.layout(), keep)


def _owner(layout, widget):
    if layout is None:
        return None
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item.widget() is widget:
            return layout
        found = _owner(item.layout(), widget)
        if found is not None:
            return found
    return None


def _owner_layout(layout, child):
    if layout is None:
        return None
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item.layout() is child:
            return layout
        found = _owner_layout(item.layout(), child)
        if found is not None:
            return found
    return None


def _drop_empty(layout):
    for index in range(layout.count() - 1, -1, -1):
        child = layout.itemAt(index).layout()
        if child is None:
            continue
        held = any(child.itemAt(at).widget() is not None for at in range(child.count()))
        if not held:
            layout.takeAt(index)
