"""The left control panel: prompt, actions and every chip row; the picture fills the right."""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

COUNT_MIN = 1
COUNT_MAX = 99
_COUNT_CHIPS = ("chip-count-1", "chip-count-2", "chip-count-4", "chip-count-custom", "reset-count")

PANEL_SHARE = 0.43
PANEL_MIN = 540
# From this window width up the panel keeps room for a one-line action row.
WIDE_FROM = 1200
WIDE_MIN = 600
PANEL_MAX = 640
SCROLL_ROOM = 14


def panel_width(window_width: int) -> int:
    """The panel takes a share of the window, kept between the two limits."""
    floor = WIDE_MIN if window_width >= WIDE_FROM else PANEL_MIN
    return max(floor, min(PANEL_MAX, int(window_width * PANEL_SHARE)))


def arrange(window) -> None:
    """Rebuild the prompt area once, after the photoshoot controls exist."""
    hide_count_chips(window)
    ensure_count_field(window)
    if getattr(window, "_prompt_arranged", False):
        return
    view = window.view
    root = view.root
    parts = _parts(root)
    if view._prompt is None or parts is None:
        return
    outer = root.layout()
    row = getattr(view, "_prompt_row", None)
    if row is None or outer.indexOf(view._preset_host) < 0:
        return
    for widget in list(parts.values()) + [view._prompt]:
        if widget is not None:
            _detach(widget)
    _left_panel(window, outer, row, parts)
    window._prompt_arranged = True


def _parts(root):
    names = {
        "generate": "generate",
        "improve": "improvePrompt",
        "revert": "revertPrompt",
    }
    parts = {key: root.findChild(QPushButton, name) for key, name in names.items()}
    if parts["generate"] is None or parts["improve"] is None or parts["revert"] is None:
        return None
    parts["mode"] = root.findChild(QCheckBox, "photoshootMode")
    parts["frames_label"] = root.findChild(QLabel, "photoshootFramesLabel")
    parts["frames"] = root.findChild(QWidget, "photoshootFrames")
    return parts


def _detach(widget) -> None:
    parent = widget.parentWidget()
    layout = None if parent is None else parent.layout()
    owner = _owner(layout, widget) if layout is not None else None
    if owner is not None:
        owner.removeWidget(widget)


def _owner(layout, widget):
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item.widget() is widget:
            return layout
        child = item.layout()
        if child is not None:
            found = _owner(child, widget)
            if found is not None:
                return found
    return None


def _prompt_box(prompt, lines: int, cap: int) -> None:
    line = prompt.fontMetrics().lineSpacing()
    pad = 2 * prompt.frameWidth() + int(prompt.document().documentMargin() * 2) + 6
    prompt.setMinimumHeight(lines * line + pad)
    prompt.setMaximumHeight(cap * line + pad)
    prompt.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


def hide_count_chips(window) -> None:
    """The picture count is a typed field. The old chips stay off the row."""
    for name in _COUNT_CHIPS:
        widget = window.findChild(QWidget, name)
        if widget is not None:
            widget.hide()


def ensure_count_field(window) -> None:
    """A digits-only picture count, from 1 to 99, starting at the preset default."""
    if getattr(window, "_count_field", None) is not None:
        return
    label = window.findChild(QLabel, "preset-count")
    if label is None or label.parentWidget() is None or label.parentWidget().layout() is None:
        return
    current = window.shell.layout.batch_size()
    if current < COUNT_MIN or current > COUNT_MAX:
        current = COUNT_MIN
    field = QLineEdit()
    field.setObjectName("pictureCount")
    field.setValidator(QIntValidator(COUNT_MIN, COUNT_MAX, field))
    field.setMaxLength(len(str(COUNT_MAX)))
    field.setFixedWidth(72)
    field.setText(str(current))
    field.editingFinished.connect(lambda: _commit_count(window, field))
    layout = label.parentWidget().layout()
    layout.insertWidget(layout.indexOf(label) + 1, field)
    window._count_field = field
    window._count_last = current


def _commit_count(window, field: QLineEdit) -> None:
    raw = field.text().strip()
    if raw.isdigit():
        number = int(raw)
        if COUNT_MIN <= number <= COUNT_MAX:
            field.setText(str(number))
            window._count_last = number
            window.shell.layout.set_value("shell.count", number)
            hide_count_chips(window)
            return
        number = min(COUNT_MAX, max(COUNT_MIN, number))
        field.setText(str(number))
        window._count_last = number
        window.shell.layout.set_value("shell.count", number)
        hide_count_chips(window)
        return
    field.setText(str(window._count_last))


def _empty_row(outer, row) -> int:
    """Take the old row out of the root column. Leftover widgets stay hidden."""
    index = outer.indexOf(row)
    _drain(row)
    outer.removeItem(row)
    return index


def _drain(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.layout() is not None:
            _drain(item.layout())
        elif item.widget() is not None:
            item.widget().hide()


def _left_panel(window, outer, row, parts) -> None:
    """A left panel with prompt, actions and chips; the picture fills the right."""
    view = window.view
    prompt = view._prompt
    index = _empty_row(outer, row)
    panel = QWidget()
    panel.setObjectName("promptPanel")
    column = QVBoxLayout(panel)
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(8)
    _prompt_box(prompt, 5, 9)
    prompt.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    column.addWidget(prompt)
    from canvas.ui.action_row import build

    column.addWidget(build(window, parts))
    outer.removeWidget(view._preset_host)
    column.addSpacing(4)
    view._preset_rows.setSpacing(10)
    column.addWidget(view._preset_host)
    column.addStretch(1)
    for host in getattr(window, "_flow_hosts", None) or []:
        flow = host.layout()
        if hasattr(flow, "may_wrap"):
            # Every row may wrap at full chip size; the row name sits on its own line.
            flow.may_wrap = True
            flow.label_above = True
    scroll = QScrollArea()
    scroll.setObjectName("promptPanelScroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setWidget(panel)
    panel.setAutoFillBackground(False)
    scroll.viewport().setAutoFillBackground(False)
    scroll.setStyleSheet("QScrollArea#promptPanelScroll, QWidget#promptPanel { background: transparent; }")

    def fit() -> None:
        width = panel_width(window.width())
        if scroll.width() != width:
            scroll.setFixedWidth(width)
        panel.setFixedWidth(width - SCROLL_ROOM)

    window._flow_width = lambda _host, _enabled: panel_width(window.width()) - SCROLL_ROOM
    window._fit_prompt_panel = fit
    fit()
    window._prompt_panel_watch = _Resized(window, fit)
    window.installEventFilter(window._prompt_panel_watch)
    split = view._centre_split
    history = view._history_scroll
    outer.removeWidget(split)
    outer.removeWidget(history)
    right = QVBoxLayout()
    right.setContentsMargins(0, 0, 0, 0)
    right.setSpacing(6)
    right.addWidget(split, 1)
    right.addWidget(history)
    body = QHBoxLayout()
    body.setContentsMargins(0, 0, 0, 0)
    body.setSpacing(14)
    body.addWidget(scroll, 0)
    body.addLayout(right, 1)
    outer.insertLayout(index, body, 1)




class _Resized(QObject):
    """Refit the panel width and the chip row heights when the window resizes."""

    def __init__(self, window, fit) -> None:
        super().__init__(window)
        self._window = window
        self._fit = fit

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self._window and event.type() == QEvent.Type.Resize:
            from canvas.ui.shell_chrome import _flow_heights

            self._fit()
            _flow_heights(self._window, self._window.view._expert_panel.isVisible())
        return False
