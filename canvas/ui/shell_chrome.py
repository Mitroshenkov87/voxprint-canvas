"""Place the shared screen in the suite's dark theme.

Colours come from the shared tokens. This module does not invent a palette.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QSizePolicy, QSplitter, QVBoxLayout, QWidget

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
    TEXT_MUTED,
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
QDialog#setupWizard, QDialog#pictureWizard, QDialog#maskWizard,
QDialog#addonsWizard, QDialog#diagnosticsWizard {{
    background: {CARD_PLAIN}; border: 1px solid {CONTROL_BORDER};
}}
QLabel {{
    color: {TEXT}; background: transparent;
}}
QLabel#setupHint, QLabel[setupHint="true"], QLabel#maskHint {{
    color: {TEXT_MUTED};
}}
QLabel#setupTitle, QLabel#maskTitle {{
    font-size: 18px;
}}
QSlider::groove:horizontal {{
    height: 6px; background: {CONTROL_BG};
    border: 1px solid {CONTROL_BORDER}; border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 16px; margin: -6px 0; border-radius: 8px;
    background: {PRIMARY}; border: 1px solid {PRIMARY_BORDER};
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
    _result(window)
    _thumbs(window)
    if window.view._expert_panel.isVisible():
        _reap(window)
        _status(window)
    _labels(window)
    _fit_controls(window)


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
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(6)
        badge = QLabel()
        badge.setObjectName("expertChip")
        badge.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        reset = QPushButton(text("canvas.expert.reset"))
        reset.setObjectName("expertResetAll")
        reset.clicked.connect(lambda: _reset(window))
        layout.addWidget(badge, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(reset, 0, Qt.AlignmentFlag.AlignLeft)
        window._expert_badge = badge
        window._expert_reset = reset
        panel_layout.insertWidget(0, row)
    window._expert_reset.setText(text("canvas.expert.reset"))
    window._expert_badge.setText(_preset_name(window))
    window._expert_reset.setEnabled(_custom(window))


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


def _preset_name(window):
    """The quality preset's own name, or Custom after a hand edit."""
    if _custom(window):
        return text("canvas.expert.custom")
    for row in window.shell.layout.rows:
        if row.row_id != "quality":
            continue
        selected = str(window.shell.layout.presets.get("quality", row.default))
        current = next((choice for choice in row.choices if choice.value == selected), None)
        if current is not None:
            return tr(current.label_key)
    return text("canvas.expert.preset")


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
    ready = window.shell.layout.phase is not Phase.EMPTY
    bar.setVisible(ready)
    if not ready:
        return
    row = bar.layout()
    if row is None or bar.findChild(QPushButton, "useInput") is not None:
        return
    use = QPushButton(tr("simple.history.use_input"))
    use.setObjectName("useInput")
    use.clicked.connect(lambda: _history_action(window, "use_input"))
    delete = QPushButton(tr("simple.history.delete"))
    delete.setObjectName("deletePicture")
    delete.clicked.connect(lambda: _history_delete(window))
    at = _stretch_index(row)
    row.insertWidget(at, use)
    row.insertWidget(at + 1, delete)


def _history_action(window, action):
    item_id = window.shell.layout.selected
    if item_id:
        window.view._notify_history(action, item_id)


def _history_delete(window):
    item_id = window.shell.layout.selected
    if item_id:
        window.view._ask_delete(item_id)


def _stretch_index(row):
    for index in range(row.count()):
        if row.itemAt(index).spacerItem() is not None:
            return index
    return row.count()


def _thumbs(window):
    """Give each history item its own placeholder, so the strip is not one picture repeated."""
    buttons = sorted(
        (
            button
            for button in window.findChildren(QPushButton)
            if button.objectName().startswith("thumb-")
        ),
        key=lambda button: button.objectName(),
    )
    for index, button in enumerate(buttons):
        button.setIcon(QIcon(_thumb(index)))
        button.setIconSize(QSize(80, 80))


def _thumb(index):
    image = QImage(80, 80, QImage.Format.Format_RGB32)
    painter = QPainter(image)
    kind = index % 6
    hue = (index * 47) % 360
    _thumb_scene(painter, kind, hue)
    painter.end()
    return QPixmap.fromImage(image)


def _thumb_scene(painter, kind, hue):
    base = QColor.fromHsv(hue, 90, 70)
    accent = QColor.fromHsv((hue + 40) % 360, 140, 210)
    light = QColor.fromHsv((hue + 180) % 360, 80, 230)
    painter.fillRect(0, 0, 80, 80, base)
    if kind == 0:
        painter.setBrush(accent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(48, 10, 22, 22)
        painter.setBrush(QColor.fromHsv(hue, 60, 46))
        painter.drawPolygon(
            [QPoint(0, 62), QPoint(28, 34), QPoint(58, 70), QPoint(80, 48), QPoint(80, 80), QPoint(0, 80)]
        )
        return
    if kind == 1:
        painter.setBrush(QColor.fromHsv(hue, 40, 150))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(18, 8, 44, 44)
        painter.setBrush(accent)
        painter.drawRoundedRect(22, 48, 36, 28, 8, 8)
        return
    if kind == 2:
        painter.setPen(Qt.PenStyle.NoPen)
        for left, height, colour in ((6, 48, accent), (24, 62, light), (42, 36, accent), (58, 54, light)):
            painter.setBrush(colour)
            painter.drawRect(left, 80 - height, 14, height)
        return
    if kind == 3:
        painter.setBrush(QColor.fromHsv(hue, 50, 40))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(0, 52, 80, 28)
        painter.setBrush(accent)
        painter.drawEllipse(28, 22, 24, 30)
        return
    if kind == 4:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawEllipse(8, 14, 36, 36)
        painter.setBrush(light)
        painter.drawEllipse(32, 28, 36, 36)
        return
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor.fromHsv(hue, 30, 24))
    painter.drawRect(0, 0, 80, 80)
    painter.setBrush(light)
    for spot in ((12, 16), (30, 28), (58, 14), (46, 42), (18, 50), (66, 36)):
        painter.drawEllipse(spot[0], spot[1], 4, 4)
    painter.setBrush(accent)
    painter.drawEllipse(50, 48, 16, 16)


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
    widget.setMinimumWidth(width)


def _fit_controls(window):
    """Give every labelled button the width of its text, in this locale."""
    for button in window.findChildren(QPushButton):
        if not button.text() or button.objectName().startswith("thumb-"):
            continue
        _fit_text(button)
    badge = getattr(window, "_expert_badge", None)
    if badge is not None and badge.text():
        _reserve(badge, 20)


def _fit_text(button):
    words = button.text()
    font = QFont(button.font())
    font.setWeight(QFont.Weight.DemiBold)
    pad = 16
    if button.objectName() == "generate":
        font.setPixelSize(16)
        pad = 22
    elif button.property("chip") or button.objectName() in {"expertToggle", "useInput", "deletePicture"}:
        pad = 18
    elif _in_action_bar(button):
        pad = 18
    advance = QFontMetrics(font).horizontalAdvance(words)
    button.setMinimumWidth(advance + pad * 2)
    policy = button.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Policy.Minimum)
    button.setSizePolicy(policy)


def _in_action_bar(button):
    parent = button.parentWidget()
    return parent is not None and parent.objectName() == "actionBar"


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
