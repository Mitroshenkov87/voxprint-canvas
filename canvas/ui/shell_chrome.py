"""Place the shared screen in the suite's dark theme.

Colours come from the shared tokens. This module does not invent a palette.
"""
from __future__ import annotations

import logging

from typing import assert_never

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QAction, QColor, QFont, QFontInfo, QFontMetrics, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QLayout,
    QWidgetItem,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from voxprint_common.i18n import tr
from voxprint_common.ui.settings import SettingType
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
    build_shell_style,
)

from canvas.app.i18n import text
from canvas.ui.help_menu import add_help_menu
from canvas.ui.theme import BUTTON_RADIUS

PANEL_WIDTH = 320
SHOT_WIDTH = 1280
SHOT_HEIGHT = 800


from canvas.ui.prompt_row import arrange as arrange_prompt  # noqa: E402
from canvas.ui.action_row import sync as sync_actions  # noqa: E402
from canvas.ui.result_bar import wrap as wrap_result  # noqa: E402


def shell_style() -> str:
    """The shared shell stylesheet, with the window and labels kept on the dark ground."""
    from canvas.ui.fonts import register_fonts
    from canvas.ui.theme import UI_FONT

    register_fonts()
    family = ", ".join(f'"{name}"' for name in UI_FONT)
    return build_shell_style(False) + f"""
* {{ font-family: {family}, sans-serif; }}
QMainWindow#shell, QWidget#shellColumn, QWidget#simpleRoot, QDialog {{
    background: {ROOT_PLAIN}; color: {TEXT};
}}
QDialog#setupWizard, QDialog#pictureWizard, QDialog#maskWizard,
QDialog#addonsWizard, QDialog#diagnosticsWizard,
QDialog#enlargeConfirm, QDialog#enlargeProgress, QDialog#enlargeError {{
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
QPushButton, QPushButton[chip="true"], QPushButton#generate, QPushButton#overflow,
QPushButton#expertToggle, QToolButton, QSpinBox, QDoubleSpinBox, QComboBox,
QLineEdit, QPlainTextEdit {{
    border-radius: {BUTTON_RADIUS}px;
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
    layout.setContentsMargins(12, 4, 12, 4)
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
    window.help_menu = add_help_menu(view._overflow.menu())
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
    panel.setMaximumWidth(PANEL_WIDTH if enabled else 0)
    split = getattr(window, "_split", None)
    if split is None:
        return
    if not enabled:
        split.setSizes([max(1, window.width()), 0])
        _flow_heights(window, False)
        return
    _reap(window)
    _paint_panel(window)
    _widen_combos(window)
    _hide_row_resets(window)
    width = max(1, window.width() - PANEL_WIDTH)
    split.setSizes([width, PANEL_WIDTH])
    _flow_heights(window, True)


def polish(window):
    """Apply the size menu, the 4K wording, and the upscale row after a refresh.

    Args:
        window: The shell window.
    """
    _adopt_rows(window)
    _source_drop(window)
    arrange_prompt(window)
    _size_menu(window)
    _output(window)
    _detail(window)
    _mask_button(window)
    _result(window)
    _thumbs(window)
    if window.view._expert_panel.isVisible():
        _reap(window)
        _paint_panel(window)
        _widen_combos(window)
    _hide_row_resets(window)
    _labels(window)
    _fit_controls(window)
    _wrap_rows(window)
    _sync_lora_picker(window)
    _sync_lora_chip(window)
    sync_actions(window)


def _sync_lora_chip(window):
    """Show the selected trigger as a chip. Pressing it inserts the word."""
    button = getattr(window, "_lora_chip", None)
    if button is None:
        button = QPushButton()
        button.setObjectName("loraTrigger")
        button.setProperty("chip", True)
        button.clicked.connect(window.insert_trigger)
        window._lora_chip = button
    picker = getattr(window, "_lora_picker", None)
    if picker is not None and button.parentWidget() is not picker:
        # The trigger sits in the LoRA row, so the action row keeps one line.
        picker.set_chip(button)
    word = window.lora_trigger()
    button.setText(word)
    button.setToolTip(text("canvas.lora.trigger_tip", trigger=word) if word else "")
    if picker is not None:
        picker.show_chip(button, word)
    else:
        button.setVisible(False)


def _sync_lora_picker(window):
    """Put the LoRA pick and strength under the presets, once."""
    if getattr(window, "_lora_picker", None) is not None:
        return
    panel = window.view.root.findChild(QWidget, "promptPanel")
    column = None if panel is None else panel.layout()
    if column is None:
        return
    from canvas.ui.lora_panel import LoraPicker

    picker = LoraPicker(window._pick_lora)
    window._lora_picker = picker
    at = column.count()
    for index in range(column.count()):
        if column.itemAt(index).spacerItem() is not None:
            at = index
    column.insertWidget(at, picker)
    window.refresh_loras()


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
        ("actionLoraImport", "canvas.lora.import", window._open_lora_import),
        ("actionLoraLibrary", "canvas.lora.library", window._open_lora_library),
        ("actionCheck", "canvas.action.check", window._open_check),
    )
    existing = menu.actions()
    before = existing[0] if existing else None
    for name, key, slot in specs:
        action = QAction(text(key), menu)
        action.setObjectName(name)
        action.triggered.connect(slot)
        menu.insertAction(before, action)


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
    if isinstance(host, _Flow) and _label_index(host, label) >= 0:
        insert = _label_index(host, label) + 1
        while insert < host.count():
            widget = host.itemAt(insert).widget()
            if widget is None or not str(widget.objectName()).startswith("orient-"):
                break
            insert += 1
        host.insertWidget(insert, combo)
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
    chip = window.view.root.findChild(QPushButton, "chip-output-4k")
    reason = window.view.root.findChild(QLabel, "reason-output-4k")
    install = window.view.root.findChild(QPushButton, "unavailable-output-4k")
    if reason is not None:
        reason.hide()
    if install is not None:
        install.hide()
    if chip is None or _fourk_available(window):
        return
    chip.setText(text("canvas.output.4k.needs"))
    chip.setEnabled(True)
    _reserve(chip, 36)
    if getattr(window, "_fourk_wired", False):
        return
    chip.clicked.connect(lambda _checked=False: window._offer_enlarge())
    window._fourk_wired = True


def _fourk_available(window):
    state = window.shell.layout.state
    if state is None:
        return False
    group = state._presets.group("output")
    option = None if group is None else group.option("4k")
    return option is not None and option.available


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
        wrap_result(bar)
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
    wrap_result(bar)


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


def _paint_panel(window):
    """Copy live values into the expert fields.

    The shared screen looks for those fields under the picture column. This
    window keeps the panel on the splitter, so the copy happens here.
    """
    form = window.findChild(QWidget, "expertForm")
    if form is None:
        return
    layout = window.shell.layout
    specs = tuple(spec for spec in window.shell.settings.visible(expert=True) if spec.expert)
    for spec in specs:
        control = form.findChild(QWidget, spec.setting_id)
        if control is None:
            continue
        control.blockSignals(True)
        _write_expert(control, spec, layout.get(spec.setting_id))
        control.blockSignals(False)


def _write_expert(control, spec, value):
    kind = spec.type
    if kind is SettingType.BOOL:
        control.setChecked(value is True)
        return
    if kind is SettingType.ENUM:
        index = control.findData(value)
        if index >= 0:
            control.setCurrentIndex(index)
        return
    if kind is SettingType.INT:
        if isinstance(value, int) and not isinstance(value, bool):
            control.setValue(value)
        return
    if kind is SettingType.FLOAT:
        if isinstance(value, int | float) and not isinstance(value, bool):
            control.setValue(float(value))
        return
    if kind is SettingType.PATH:
        control.setText(value if isinstance(value, str) else "")
        return
    if kind is SettingType.TEXT:
        control.setPlainText(value if isinstance(value, str) else "")
        return
    assert_never(kind)


def _hide_row_resets(window):
    """Leave preset changes on the chip and the shared panel reset."""
    for button in window.findChildren(QPushButton):
        if button.objectName().startswith("reset-"):
            button.hide()


def _widen_combos(window):
    """Give each expert combo the width of its longest value."""
    form = window.findChild(QWidget, "expertForm")
    if form is None:
        return
    for combo in form.findChildren(QComboBox):
        metrics = QFontMetrics(combo.font())
        longest = 0
        for index in range(combo.count()):
            longest = max(longest, metrics.horizontalAdvance(combo.itemText(index)))
        combo.setMinimumWidth(longest + 48)
        label = _row_label(form.layout(), combo)
        if label is not None:
            label.setMaximumWidth(88)
        combo.resize(max(combo.minimumWidth(), combo.width()), max(combo.height(), 28))
        combo.updateGeometry()


def _row_label(layout, widget):
    if layout is None:
        return None
    found = None
    label = None
    for index in range(layout.count()):
        item = layout.itemAt(index)
        child = item.widget()
        if child is widget:
            found = layout
            break
        if isinstance(child, QLabel):
            label = child
        nested = _row_label(item.layout(), widget)
        if nested is not None:
            return nested
    if found is None:
        return None
    for index in range(found.count()):
        child = found.itemAt(index).widget()
        if isinstance(child, QLabel):
            return child
    return label


def _wrap_rows(window):
    """Let a chip row use a second line when the expert panel narrows it."""
    if not getattr(window, "_rows_wrapped", False):
        rows = window.view._preset_rows
        pending = []
        hosts = []
        while rows.count():
            item = rows.takeAt(0)
            line = item.layout()
            if line is None:
                widget = item.widget()
                if widget is not None:
                    pending.append(widget)
                continue
            widgets = []
            while line.count():
                taken = line.takeAt(0)
                widget = taken.widget()
                if widget is not None:
                    widgets.append(widget)
            host = QWidget()
            host.setObjectName("chipFlow")
            flow = _Flow()
            flow.setSpacing(8)
            host.setLayout(flow)
            for widget in widgets:
                flow.addWidget(widget)
            policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            policy.setHeightForWidth(True)
            host.setSizePolicy(policy)
            pending.append(host)
            hosts.append(host)
        for widget in pending:
            rows.addWidget(widget)
        window._flow_hosts = hosts
        window._rows_wrapped = True
        preset = window.view._preset_host
        policy = preset.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        preset.setSizePolicy(policy)
    shown = window.view._expert_panel.isVisible()
    _flow_heights(window, shown)


def _source_drop(window):
    """Hide the shared drop box. Canvas opens pictures from Picture setup.

    The Yitro source slot still holds the path; only the extra box above the
    prompt is hidden so the picture keeps its 300 px.
    """
    drop = window.view.root.findChild(QWidget, "sourceDrop")
    if drop is not None and not drop.isHidden():
        drop.hide()


def _adopt_rows(window):
    """Move each shared chip row (Yitro ``pillRow``) onto the Canvas flow.

    The Canvas flow keeps the style row on one line and lets the size menu sit
    after the orientation chips. Rows already on the Canvas flow are left alone.
    """
    hosts = list(getattr(window, "_flow_hosts", None) or [])
    for host in window.view.root.findChildren(QWidget, "pillRow"):
        old = host.layout()
        if old is None or isinstance(old, _Flow):
            if host not in hosts and isinstance(old, _Flow):
                hosts.append(host)
            continue
        spacing = old.spacing()
        widgets = []
        while old.count():
            taken = old.takeAt(0)
            widget = taken.widget() if taken is not None else None
            if widget is not None:
                widgets.append(widget)
        QWidget().setLayout(old)
        flow = _Flow()
        flow.setSpacing(spacing if spacing >= 0 else 8)
        host.setLayout(flow)
        for widget in widgets:
            flow.addWidget(widget)
        hosts.append(host)
    window._flow_hosts = [host for host in hosts if _alive(host)]


def _alive(widget):
    try:
        widget.objectName()
    except RuntimeError:
        return False
    return True


def _flow_heights(window, enabled):
    hosts = getattr(window, "_flow_hosts", None)
    if not hosts:
        return
    width = _preset_width(window, enabled)
    custom = getattr(window, "_flow_width", None)
    for host in hosts:
        flow = host.layout()
        if flow is None:
            continue
        own = custom(host, enabled) if custom is not None else None
        host.setMinimumHeight(flow.heightForWidth(own or width))
        host.updateGeometry()
    window.view._preset_host.updateGeometry()


def _preset_width(window, enabled):
    panel = PANEL_WIDTH + 6 if enabled else 0
    return max(160, window.width() - panel - 24 - 32 - 24)


class _Flow(QLayout):
    """A row of chips that wraps onto a second line."""

    def __init__(self):
        super().__init__()
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        # A narrow side panel lets every row wrap, with the row name on its own line.
        self.may_wrap = False
        self.label_above = False

    def addItem(self, item):  # noqa: N802
        self._items.append(item)

    def insertWidget(self, index, widget):  # noqa: N802
        """Put ``widget`` at ``index`` in the row."""
        self.addChildWidget(widget)
        self._items.insert(max(0, min(index, len(self._items))), QWidgetItem(widget))
        self.invalidate()

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._arrange(width, None)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect.width(), rect)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        width = 0
        height = 0
        for item in self._items:
            if _flow_hidden(item):
                continue
            item_width, item_height = _flow_size(item)
            width = max(width, item_width)
            height = max(height, item_height)
        margins = self.contentsMargins()
        return QSize(
            width + margins.left() + margins.right(),
            height + margins.top() + margins.bottom(),
        )

    def _arrange(self, width, rect):
        spacing = max(self.spacing(), 0)
        margins = self.contentsMargins()
        inner = max(0, width - margins.left() - margins.right())
        origin_x = margins.left() if rect is None else rect.x() + margins.left()
        origin_y = margins.top() if rect is None else rect.y() + margins.top()
        if self.may_wrap:
            for item in self._items:
                widget = item.widget()
                if widget is not None and widget.property("tightWidth"):
                    _scale_chip_font(widget, 100)
            one_row = False
            squeeze = {}
        else:
            items = self._items
            if self.label_above and items and _is_row_label(items[0]):
                items = items[1:]
            one_row = _fit_style_font(items, inner, spacing, warn=rect is not None)
            squeeze = _squeeze(items, inner, spacing)
        lines = []
        current = []
        x = 0
        line_height = 0
        for item in self._items:
            if _flow_hidden(item):
                continue
            item_width, item_height = _flow_size(item)
            item_width -= squeeze.get(id(item), 0)
            if self.label_above and current and _is_row_label(current[0][0]) and len(current) == 1:
                lines.append((current, line_height))
                current = []
                x = 0
                line_height = 0
            if x and x + item_width > inner and not one_row:
                lines.append((current, line_height))
                current = []
                x = 0
                line_height = 0
            current.append((item, item_width, item_height))
            x += item_width + spacing
            line_height = max(line_height, item_height)
        if current:
            lines.append((current, line_height))
        y = 0
        for index, (row, row_height) in enumerate(lines):
            x = 0
            for item, item_width, item_height in row:
                widget = item.widget()
                if widget is not None and widget.property("flowFill"):
                    # The main action takes the rest of its line.
                    item_width = max(item_width, inner - x)
                if rect is not None:
                    offset = (row_height - item_height) // 2
                    item.setGeometry(QRect(origin_x + x, origin_y + y + offset, item_width, item_height))
                x += item_width + spacing
            y += row_height
            if index + 1 < len(lines):
                y += spacing
        return margins.top() + y + margins.bottom()


def _is_row_label(item):
    widget = item.widget()
    return isinstance(widget, QLabel) and str(widget.objectName()).startswith("preset-")


def _row_fits(items, inner, spacing):
    shown = [item for item in items if not _flow_hidden(item)]
    total = 0
    for item in shown:
        widget = item.widget()
        tight = widget.property("tightWidth") if widget is not None else None
        total += int(tight) if tight else _flow_size(item)[0]
    return total + spacing * max(0, len(shown) - 1) <= inner


def _fit_style_font(items, inner, spacing, warn=True):
    """Shrink the style chip font, in steps down to 80%, so the row stays on one line.

    Returns True when the row holds style chips; such a row never wraps. When it
    still does not fit at the 80% floor, a warning is logged and it stays one row.
    """
    chips = [
        item.widget() for item in items
        if item.widget() is not None and item.widget().property("tightWidth")
    ]
    if not chips:
        return False
    for step in (100, 95, 90, 85, 80):
        for chip in chips:
            _scale_chip_font(chip, step)
        if _row_fits(items, inner, spacing):
            return True
    if warn:
        logging.getLogger(__name__).warning("style chip row does not fit in %d px at the 80%% font floor", inner)
    return True


def _scale_chip_font(chip, step):
    if chip.property("fontStep") == step:
        return
    base = chip.property("basePx")
    if base is None:
        base = QFontInfo(chip.font()).pixelSize() or 14
        chip.setProperty("basePx", base)
    chip.setProperty("fontStep", step)
    chip.setStyleSheet("" if step == 100 else f"font-size: {max(1, round(int(base) * step / 100))}px;")
    font = QFont(chip.font())
    font.setPixelSize(max(1, round(int(base) * step / 100)))
    font.setWeight(QFont.Weight.DemiBold)
    advance = QFontMetrics(font).horizontalAdvance(chip.text())
    chip.setProperty("tightWidth", advance + 12)
    chip.setProperty("naturalWidth", advance + 28)
    chip.setMinimumWidth(advance + 12)


def _squeeze(items, inner, spacing):
    """Trim style chip padding so the style row stays on one line.

    Returns how many pixels to take from each style chip. Only chips that set
    a ``tightWidth`` floor give up width; other items keep their full size.
    """
    shown = [item for item in items if not _flow_hidden(item)]
    total = sum(_flow_size(item)[0] for item in shown) + spacing * max(0, len(shown) - 1)
    over = total - inner
    tight = [item for item in shown if _floor(item) is not None]
    if over <= 0 or not tight:
        return {}
    room = {id(item): max(0, _flow_size(item)[0] - int(_floor(item))) for item in tight}
    if sum(room.values()) < over:
        return room
    share = {}
    left = over
    for index, item in enumerate(tight):
        want = -(-left // (len(tight) - index))
        take = min(room[id(item)], want)
        share[id(item)] = take
        left -= take
    return share


def _floor(item):
    """The narrowest width an item may be trimmed to: a style chip's, or a fill button's floor."""
    widget = item.widget()
    if widget is None:
        return None
    return widget.property("tightWidth") or widget.property("flowFloor") or None


def _flow_hidden(item):
    # isHidden, not isVisible: before the window is shown every chip reads as
    # not visible, and a row arranged then would leave its chips unplaced.
    widget = item.widget()
    return widget is not None and widget.isHidden()


def _flow_size(item):
    widget = item.widget()
    hint = item.sizeHint()
    width = hint.width()
    height = hint.height()
    if widget is not None:
        width = max(width, widget.minimumWidth())
        natural = widget.property("naturalWidth")
        if natural:
            width = max(width, int(natural))
        height = max(height, widget.minimumHeight(), widget.sizeHint().height())
    return max(width, 1), max(height, 1)


def _fit_text(button):
    words = button.text()
    font = QFont(button.font())
    font.setWeight(QFont.Weight.DemiBold)
    pad = 16
    if button.objectName() == "generate":
        font.setPixelSize(16)
        pad = 22
    elif button.property("chip"):
        # Padding plus the border, so a checked (weight 600) label is not cut.
        pad = 18
    elif button.objectName() in {"expertToggle", "useInput", "deletePicture"}:
        pad = 18
    elif _in_action_bar(button):
        pad = 18
    advance = QFontMetrics(font).horizontalAdvance(words)
    style_chip = str(button.objectName()).startswith("chip-style-") and not str(button.objectName()).endswith("-custom")
    if style_chip:
        # The style row must stay on one line at 1280px, so these chips may
        # give up padding (down to 6px a side) when the row is tight.
        button.setProperty("tightWidth", advance + 12)
        button.setProperty("naturalWidth", advance + pad * 2)
        button.setMinimumWidth(advance + 12)
    else:
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
