"""The action row under the prompt: Improve with its back segment, Photoshoot,
Proportions, Image size and Generate. The row wraps when the panel is narrow."""
from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMenu,
    QPushButton,
    QSizePolicy,
    QWidget,
)
from voxprint_common.i18n import tr
from voxprint_common.ui.theme import CONTROL_BG, CONTROL_BORDER, TEXT

from canvas.app.i18n import text
from canvas.ui.theme import BUTTON_RADIUS

HEIGHT = 36
GENERATE_HEIGHT = 40
GENERATE_MIN = 180


def build(window, parts) -> QWidget:
    """Make the row and keep the shared size and output chips as hidden state holders."""
    from canvas.ui.shell_chrome import _Flow

    host = QWidget()
    host.setObjectName("actionRow")
    flow = _Flow()
    flow.setSpacing(8)
    host.setLayout(flow)
    policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    policy.setHeightForWidth(True)
    host.setSizePolicy(policy)
    flow.addWidget(_improve_split(window, parts["improve"], parts["revert"]))
    mode = parts["mode"]
    if mode is not None:
        mode.setProperty("chip", False)
        mode.setFixedHeight(HEIGHT)
        flow.addWidget(mode)
        for key in ("frames_label", "frames"):
            if parts[key] is not None:
                parts[key].setFixedHeight(HEIGHT if key == "frames" else parts[key].sizeHint().height())
                flow.addWidget(parts[key])
    flow.addWidget(_proportions(window))
    flow.addWidget(_image_size(window))
    generate = parts["generate"]
    generate.setMinimumSize(GENERATE_MIN, GENERATE_HEIGHT)
    generate.setProperty("flowFill", True)
    flow.addWidget(generate)
    window._action_row = host
    return host


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
    """Hide the old rows, refresh the two menu labels and the row height."""
    host = getattr(window, "_action_row", None)
    if host is None:
        return
    root = window.view.root
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
    window._proportions_button.setText(
        text("canvas.menu.value", name=text("canvas.proportions"), value=tr(choice.label_key))
    )
    standard = root.findChild(QPushButton, "chip-output-standard")
    fourk = root.findChild(QPushButton, "chip-output-4k")
    current = fourk if fourk is not None and fourk.isChecked() else standard
    words = "" if current is None else current.text()
    window._image_size_button.setText(text("canvas.menu.value", name=text("canvas.image_size"), value=words))
    for button in (window._proportions_button, window._image_size_button):
        advance = button.fontMetrics().horizontalAdvance(button.text())
        button.setMinimumWidth(advance + 36)
    width = window._flow_width(host, False) if hasattr(window, "_flow_width") else host.width()
    host.setMinimumHeight(host.layout().heightForWidth(width))
