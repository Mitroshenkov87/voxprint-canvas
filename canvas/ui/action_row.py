"""The action block under the prompt.

Improve with its back segment, Proportions and Image size share a centred
line that wraps when the panel is narrow. The large Generate button sits
centred under them, and the Photoshoot option is centred under Generate with
Frames beside it."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from voxprint_common.i18n import tr
from voxprint_common.ui.theme import CONTROL_BG, CONTROL_BORDER, TEXT

from canvas.app.i18n import text
from canvas.ui.theme import BUTTON_RADIUS

HEIGHT = 36
GENERATE_HEIGHT = 52
GENERATE_FONT = 20
GENERATE_MIN = 300
GENERATE_PAD = 32


def build(window, parts) -> QWidget:
    """Make the block and keep the shared size and output chips as hidden state holders."""
    from canvas.ui.shell_chrome import _Flow

    host = QWidget()
    host.setObjectName("actionRow")
    column = QVBoxLayout(host)
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(10)
    options = QWidget()
    options.setObjectName("actionOptions")
    flow = _Flow()
    flow.setSpacing(8)
    flow.centre = True
    options.setLayout(flow)
    policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    policy.setHeightForWidth(True)
    options.setSizePolicy(policy)
    flow.addWidget(_improve_split(window, parts["improve"], parts["revert"]))
    flow.addWidget(_proportions(window))
    flow.addWidget(_image_size(window))
    column.addWidget(options)
    generate = parts["generate"]
    generate.setMinimumSize(GENERATE_MIN, GENERATE_HEIGHT)
    generate.setFixedHeight(GENERATE_HEIGHT)
    generate.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
    # Over the shared theme: only the size changes, the colours stay.
    generate.setStyleSheet(
        f"QPushButton#generate {{ font-size: {GENERATE_FONT}px; padding: 8px {GENERATE_PAD}px; }}"
    )
    column.addWidget(generate, 0, Qt.AlignmentFlag.AlignHCenter)
    mode = parts["mode"]
    if mode is not None:
        column.addWidget(_photoshoot_line(mode, parts["frames_label"], parts["frames"]))
    window._action_row = host
    window._action_options = options
    return host


def _photoshoot_line(mode, label, frames) -> QWidget:
    """The Photoshoot checkbox centred on Generate, with Frames on its right."""
    line = QWidget()
    line.setObjectName("photoshootOption")
    row = QHBoxLayout(line)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    mode.setProperty("chip", False)
    mode.setFixedHeight(HEIGHT)
    side = QWidget()
    side.setObjectName("photoshootFramesBox")
    beside = QHBoxLayout(side)
    beside.setContentsMargins(0, 0, 0, 0)
    beside.setSpacing(8)
    for widget in (label, frames):
        if widget is not None:
            widget.setFixedHeight(HEIGHT if widget is frames else widget.sizeHint().height())
            beside.addWidget(widget)
    beside.addStretch(1)
    # Two equal sides keep the checkbox on the centre line whether Frames shows or not.
    balance = QWidget()
    for widget in (balance, side):
        widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    row.addWidget(balance, 1)
    row.addWidget(mode)
    row.addWidget(side, 1)
    return line


def _improve_split(window, improve, revert) -> QWidget:
    """Improve, with a back segment joined on its left once a rewrite can be undone."""
    box = QWidget()
    box.setObjectName("improveSplit")
    line = QHBoxLayout(box)
    line.setContentsMargins(0, 0, 0, 0)
    line.setSpacing(0)
    back = QPushButton()
    back.setObjectName("revertSegment")
    back.setAutoDefault(False)
    back.setText(text("canvas.revert.symbol"))
    back.setToolTip(tr("simple.revert"))
    back.setAccessibleName(tr("simple.revert"))
    back.setFixedSize(QSize(34, HEIGHT))
    back.hide()
    back.setStyleSheet(
        f"QPushButton#revertSegment {{ background: {CONTROL_BG}; color: {TEXT};"
        f" border: 1px solid {CONTROL_BORDER}; border-radius: {BUTTON_RADIUS}px;"
        " border-top-right-radius: 0; border-bottom-right-radius: 0; padding: 0; font-size: 16px; }"
    )
    improve.setProperty("chip", False)
    improve.setFixedHeight(HEIGHT)
    revert.hide()
    revert.setMaximumWidth(0)
    line.addWidget(back)
    line.addWidget(improve)
    prompt = window.view._prompt
    state = {"improved": None}

    def joined(on: bool) -> None:
        back.setVisible(on)
        improve.setStyleSheet(
            (
                f"QPushButton#improvePrompt {{ background: {CONTROL_BG}; color: {TEXT};"
                f" border: 1px solid {CONTROL_BORDER}; border-radius: {BUTTON_RADIUS}px;"
                " border-top-left-radius: 0; border-bottom-left-radius: 0; border-left-width: 0; }"
            )
            if on
            else ""
        )

    def after_improve() -> None:
        if revert.isEnabled():
            state["improved"] = prompt.toPlainText()
            joined(True)

    def go_back() -> None:
        state["improved"] = None
        revert.click()
        joined(False)

    def edited() -> None:
        if state["improved"] is None or prompt.toPlainText() == state["improved"]:
            return
        # A manual edit makes the earlier prompt stale.
        state["improved"] = None
        window.view._previous_prompt = None
        revert.setEnabled(False)
        joined(False)

    improve.clicked.connect(after_improve)
    back.clicked.connect(go_back)
    prompt.textChanged.connect(edited)
    window._revert_segment = back
    return box


def _menu_button(name: str) -> QPushButton:
    button = QPushButton()
    button.setObjectName(name)
    button.setAutoDefault(False)
    button.setFixedHeight(HEIGHT)
    button.setStyleSheet("QPushButton::menu-indicator { image: none; width: 0; }")
    return button


def _proportions(window) -> QPushButton:
    button = _menu_button("proportionsMenu")
    button.setToolTip(text("canvas.proportions"))
    menu = QMenu(button)
    menu.setObjectName("proportionsList")
    menu.aboutToShow.connect(lambda: _fill_proportions(window, menu))
    button.setMenu(menu)
    window._proportions_button = button
    return button


def _fill_proportions(window, menu) -> None:
    menu.clear()
    layout = window.shell.layout
    row = next(item for item in layout.rows if item.row_id == "size")
    chosen = str(layout.presets.get("size", row.default))
    group = QActionGroup(menu)
    seen = None
    for choice in row.choices:
        if choice.group_key != seen:
            seen = choice.group_key
            if menu.actions():
                menu.addSeparator()
            # A disabled item names the orientation; Fusion draws sections without text.
            header = menu.addAction(tr(seen))
            header.setEnabled(False)
            font = header.font()
            font.setBold(True)
            header.setFont(font)
        action = QAction(tr(choice.label_key), menu)
        action.setObjectName(f"proportion-{choice.value.replace(':', '-')}")
        action.setCheckable(True)
        action.setChecked(choice.value == chosen)
        action.triggered.connect(lambda _checked=False, value=choice.value: _choose_size(window, value))
        group.addAction(action)
        menu.addAction(action)


def _choose_size(window, value: str) -> None:
    if window.shell.layout.presets.get("size") == value:
        return
    window.shell.layout.choose("size", value)
    window.view.refresh()


def _image_size(window) -> QPushButton:
    button = _menu_button("imageSizeMenu")
    button.setToolTip(text("canvas.image_size"))
    menu = QMenu(button)
    menu.setObjectName("imageSizeList")
    menu.aboutToShow.connect(lambda: _fill_image_size(window, menu))
    button.setMenu(menu)
    window._image_size_button = button
    return button


def _fill_image_size(window, menu) -> None:
    menu.clear()
    group = QActionGroup(menu)
    for value in ("standard", "4k"):
        chip = window.view.root.findChild(QPushButton, f"chip-output-{value}")
        if chip is None:
            continue
        action = QAction(chip.text(), menu)
        action.setObjectName(f"imageSize-{value}")
        action.setCheckable(True)
        action.setChecked(chip.isChecked())
        action.setEnabled(chip.isEnabled())
        # The hidden chip keeps the existing wiring, including the Enlarge offer.
        action.triggered.connect(lambda _checked=False, target=chip: target.click())
        group.addAction(action)
        menu.addAction(action)


def sync(window) -> None:
    """Hide the old rows, refresh the two menu labels, Generate's width and the row height."""
    host = getattr(window, "_action_options", None)
    if host is None:
        return
    root = window.view.root
    improve = root.findChild(QPushButton, "improvePrompt")
    if improve is not None and improve.minimumHeight() != HEIGHT:
        # The shared stylesheet raises the minimum on polish, which pushes the
        # bottom border out of its 36 px split.
        improve.setFixedHeight(HEIGHT)
    label = root.findChild(QWidget, "preset-size")
    if label is not None and label.parentWidget() is not None:
        label.parentWidget().hide()
    output = root.findChild(QWidget, "preset-output")
    if output is not None and output.parentWidget() is not None:
        # Only the 4K detail chips remain in this row, so it shows with 4K alone.
        output.parentWidget().setVisible(str(window.shell.layout.presets.get("output", "")) == "4k")
    for name in ("preset-output", "chip-output-standard", "chip-output-4k", "chip-output-custom",
                 "reason-output-4k", "unavailable-output-4k"):
        widget = root.findChild(QWidget, name)
        if widget is not None:
            widget.hide()
    layout = window.shell.layout
    size = next(item for item in layout.rows if item.row_id == "size")
    chosen = str(layout.presets.get("size", size.default))
    choice = next((item for item in size.choices if item.value == chosen), size.choices[0])
    standard = root.findChild(QPushButton, "chip-output-standard")
    fourk = root.findChild(QPushButton, "chip-output-4k")
    current = fourk if fourk is not None and fourk.isChecked() else standard
    words = "" if current is None else current.text()
    menus = (
        (window._proportions_button, text("canvas.proportions"), tr(choice.label_key)),
        (window._image_size_button, text("canvas.image_size"), words),
    )
    width = window._flow_width(host, False) if hasattr(window, "_flow_width") else host.width()
    for compact in (False, True):
        for button, name, value in menus:
            if compact:
                button.setText(text("canvas.menu.compact", value=value))
            else:
                button.setText(text("canvas.menu.value", name=name, value=value))
            # The name stays in the tooltip and for screen readers when only the value shows.
            button.setToolTip(text("canvas.menu.value_tip", name=name, value=value))
            button.setAccessibleName(name)
            advance = button.fontMetrics().horizontalAdvance(button.text())
            button.setMinimumWidth(advance + 36)
        _allow_trim(host, compact)
        if one_line(host, width):
            break
    host.setMinimumHeight(host.layout().heightForWidth(width))
    generate = window._action_row.findChild(QPushButton, "generate")
    if generate is not None:
        generate.ensurePolished()
        # As with Improve, the shared stylesheet resets the height on polish.
        generate.setFixedHeight(GENERATE_HEIGHT)
        need = generate.fontMetrics().horizontalAdvance(generate.text()) + 2 * GENERATE_PAD + 4
        generate.setFixedWidth(min(width, max(GENERATE_MIN, need)))


TRIM_PAD = 12


def _allow_trim(host: QWidget, allowed: bool) -> None:
    """Let the row's buttons give up side padding, down to TRIM_PAD a side, before it wraps.

    With ``allowed`` False the floors are cleared, so the full labels are measured whole.
    """
    flow = host.layout()
    if flow is None:
        return
    for index in range(flow.count()):
        item = flow.itemAt(index)
        widget = item.widget() if item is not None else None
        if widget is None or widget.isHidden():
            continue
        if not allowed:
            widget.setProperty("flowFloor", None)
            continue
        buttons = [widget] if isinstance(widget, QPushButton) else widget.findChildren(QPushButton)
        floor = 0
        for button in buttons:
            if button.isHidden():
                continue
            if button.objectName() == "revertSegment":
                floor += button.width()
                continue
            need = button.fontMetrics().horizontalAdvance(button.text()) + 2 * TRIM_PAD
            button.setMinimumWidth(need)
            floor += need
        if floor:
            widget.setProperty("flowFloor", floor)


def one_line(host: QWidget, width: int) -> bool:
    """True when every shown item of the row fits on one line of ``width``."""
    flow = host.layout()
    if flow is None:
        return True
    total = 0
    count = 0
    for index in range(flow.count()):
        item = flow.itemAt(index)
        widget = item.widget() if item is not None else None
        if widget is None or widget.isHidden():
            continue
        floor = widget.property("flowFloor")
        total += int(floor) if floor else max(widget.minimumWidth(), widget.sizeHint().width())
        count += 1
    return total + flow.spacing() * max(0, count - 1) <= width
