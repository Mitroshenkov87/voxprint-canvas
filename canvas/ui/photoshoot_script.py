"""The Photoshoot script: one row per shot, in a separate non-modal window.

The model is headless: it keeps the ordered shot rows, the LoRA/trigger
details of each row, and the link from a row to its kept frame. The window is
a plain non-modal dialog the bar opens when a series starts writing; it is
never docked into the image stage. Closing the window hides it and leaves the
model alone, so the Script button can bring the same window back.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from voxprint_common.i18n import tr
from voxprint_common.ui.theme import (
    CARD_PLAIN,
    CONTROL_BG,
    CONTROL_BORDER,
    GREEN,
    PRIMARY,
    RED,
    ROOT_PLAIN,
    TEXT,
    TEXT_FAINT,
    TEXT_MUTED,
)

from canvas.models.user_lora import UserLora, chip_word

if TYPE_CHECKING:
    from pathlib import Path

    from canvas.ui.shell_screen import ShellWindow


WINDOW_WIDTH = 480
WINDOW_HEIGHT = 560
CARD_RADIUS = 8
BADGE_SIZE = 24

# Row cards on the Canvas dark ground. The selected card takes the yellow primary border;
# the list itself is flat so only the cards read as objects.
STYLE = f"""
QDialog#photoshootScriptWindow {{ background: {ROOT_PLAIN}; }}
QLabel#photoshootScriptHeading {{ font-size: 18px; font-weight: 600; color: {TEXT}; }}
QListWidget#photoshootScriptList {{ background: transparent; border: none; outline: none; }}
QListWidget#photoshootScriptList::item, QListWidget#photoshootScriptList::item:selected,
QListWidget#photoshootScriptList::item:hover {{ background: transparent; border: none; }}
QWidget#photoshootScriptRow {{
    background: {CARD_PLAIN}; border: 1px solid {CONTROL_BORDER}; border-radius: {CARD_RADIUS}px;
}}
QWidget#photoshootScriptRow[selected="true"] {{ background: {CONTROL_BG}; border: 1px solid {PRIMARY}; }}
QLabel#photoshootScriptNumber {{
    background: {CONTROL_BG}; border: 1px solid {CONTROL_BORDER}; border-radius: {BADGE_SIZE // 2}px;
    color: {TEXT_MUTED}; font-size: 12px; font-weight: 600;
}}
QWidget#photoshootScriptRow[selected="true"] QLabel#photoshootScriptNumber {{
    background: {PRIMARY}; border-color: {PRIMARY}; color: #1c1604;
}}
QLabel#photoshootScriptPrompt {{ color: {TEXT}; font-size: 14px; }}
QLabel#photoshootScriptLora {{
    background: {CONTROL_BG}; border: 1px solid {CONTROL_BORDER}; border-radius: 4px;
    color: {TEXT}; font-size: 12px; padding: 1px 8px;
}}
QLabel#photoshootScriptTrigger {{ color: {TEXT_FAINT}; font-size: 12px; }}
QLabel#photoshootScriptStatus {{ color: {TEXT_FAINT}; font-size: 12px; }}
QLabel#photoshootScriptStatus[state="kept"] {{ color: {GREEN}; }}
QLabel#photoshootScriptStatus[state="blocked"] {{ color: {RED}; }}
QLabel#photoshootScriptStatus[state="drawing"] {{ color: {PRIMARY}; }}
"""


@dataclass(frozen=True)
class ScriptLora:
    """One LoRA applied to a script row, exactly as the series used it.

    ``alias`` is the chip word (the LoRA alias, or its trigger when the LoRA
    has no alias). ``strength`` is the strength the series drew with.
    ``trigger`` is the trigger word the alias was rewritten to. ``injected``
    is True when the written line leads with that trigger.
    """

    alias: str
    strength: float
    trigger: str
    injected: bool


def series_loras(lora: UserLora | None, strength: float | None) -> tuple[ScriptLora, ...]:
    """The per-row LoRA list for one series, from the bound window LoRA and strength.

    The trigger is what ``rewrite_alias`` / the line assembly inject: the
    series line leads with it whenever it is not empty, so that is the
    injected flag. No LoRA gives an empty tuple and the row shows none.
    """
    if lora is None:
        return ()
    trigger = lora.trigger.strip()
    used = lora.default_strength if strength is None else float(strength)
    return (ScriptLora(chip_word(lora), used, trigger, bool(trigger)),)


@dataclass
class ScriptItem:
    """One shot of the script, as soon as its text has been written."""

    index: int
    prompt: str
    loras: list[ScriptLora] = field(default_factory=list)
    image_id: str = ""
    image_path: str = ""
    status: str = "waiting"


class PhotoshootScriptModel(QObject):
    """The ordered script rows and the two-way selection, without any widgets.

    Selection flows both ways through the same signals: a click on a row calls
    :meth:`select_item`, and a selection in the main window calls
    :meth:`select_image`, which highlights the row whose kept frame it is.
    """

    item_added = Signal(int)  # index of the appended row
    item_updated = Signal(int)  # index of the changed row
    item_selected = Signal(int)  # clicked/linked row; -1 clears the selection
    image_selected = Signal(str)  # an image/history id chosen in the main window
    window_closed = Signal()
    count_changed = Signal(int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._items: list[ScriptItem] = []
        self.selected_index = -1

    @property
    def count(self) -> int:
        """How many shot rows the script holds."""
        return len(self._items)

    def items(self) -> tuple[ScriptItem, ...]:
        """Every row, in shot order."""
        return tuple(self._items)

    def item(self, index: int) -> ScriptItem | None:
        """One row, or None outside the list."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def begin(self) -> None:
        """Drop every row. A new scenario is about to arrive."""
        self._items.clear()
        self.selected_index = -1
        self.count_changed.emit(0)

    def append_shot(self, prompt: str, loras: Iterable[ScriptLora] = ()) -> int:
        """Add one row as soon as its shot text is written.

        Args:
            prompt: The line this shot is drawn from.
            loras: The LoRAs the series applied, already resolved.

        Returns:
            The new row index.
        """
        index = len(self._items)
        self._items.append(ScriptItem(index, prompt, list(loras)))
        self.item_added.emit(index)
        self.count_changed.emit(len(self._items))
        return index

    def update_shot(self, index: int, *, prompt: str | None = None, status: str | None = None) -> None:
        """Refresh one row when its text or status changes."""
        item = self.item(index)
        if item is None:
            return
        if prompt is not None:
            item.prompt = prompt
        if status is not None:
            item.status = status
        self.item_updated.emit(index)

    def record_image(self, index: int, image_id: str, image_path: "str | Path" = "") -> None:
        """Link a row to the kept frame it produced, once the series is in the history."""
        item = self.item(index)
        if item is None or not image_id:
            return
        item.image_id = image_id
        item.image_path = str(image_path)
        self.item_updated.emit(index)

    def select_item(self, index: int) -> None:
        """A script row was clicked. Pass -1 to clear the selection."""
        if index != -1 and self.item(index) is None:
            return
        self.selected_index = index
        self.item_selected.emit(index)

    def select_image(self, image_id: str) -> None:
        """The main window selected an image; highlight its row when the script owns it."""
        self.image_selected.emit(image_id)
        found = next((item.index for item in self._items if item.image_id and item.image_id == image_id), None)
        self.select_item(-1 if found is None else found)

    def notify_window_closed(self) -> None:
        """The pop-up was closed. The script itself stays."""
        self.window_closed.emit()


class PhotoshootScriptRow(QWidget):
    """One script card: shot number, the prompt as the lead, then LoRA chip, trigger and status."""

    def __init__(self, item: ScriptItem) -> None:
        super().__init__()
        self.setObjectName("photoshootScriptRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("selected", False)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(12)
        self.number = QLabel(str(item.index + 1))
        self.number.setObjectName("photoshootScriptNumber")
        self.number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.number.setFixedSize(BADGE_SIZE, BADGE_SIZE)
        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        self.prompt = QLabel("")
        self.prompt.setObjectName("photoshootScriptPrompt")
        self.prompt.setWordWrap(True)
        self.lora = QLabel("")
        self.lora.setObjectName("photoshootScriptLora")
        self.trigger = QLabel("")
        self.trigger.setObjectName("photoshootScriptTrigger")
        self.status = QLabel("")
        self.status.setObjectName("photoshootScriptStatus")
        # Chip, trigger and status share one quiet footer, so a row without a LoRA keeps no gap.
        self.footer = QWidget()
        footer = QHBoxLayout(self.footer)
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(8)
        footer.addWidget(self.lora)
        footer.addWidget(self.trigger)
        footer.addStretch(1)
        footer.addWidget(self.status)
        column.addWidget(self.prompt)
        column.addWidget(self.footer)
        outer.addWidget(self.number, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(column, 1)
        self.show_item(item)

    def set_selected(self, selected: bool) -> None:
        """Draw the selected-card state; the row never takes focus from the list."""
        if bool(self.property("selected")) == selected:
            return
        self.setProperty("selected", selected)
        for widget in (self, self.number):
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def show_item(self, item: ScriptItem) -> None:
        """Refresh every part of the row from its model item."""
        self.prompt.setText(item.prompt)
        self._show_loras(item)
        self.status.setText(_status_text(item.status))
        self.status.setProperty("state", item.status)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.status.setVisible(bool(item.status))
        self.footer.setVisible(self.lora.isVisibleTo(self.footer) or self.trigger.isVisibleTo(self.footer)
                               or self.status.isVisibleTo(self.footer))

    def _show_loras(self, item: ScriptItem) -> None:
        if not item.loras:
            self.lora.hide()
            self.trigger.hide()
            return
        words = " · ".join(
            tr("photoshoot.script.lora", alias=lora.alias, strength=f"{lora.strength:g}") for lora in item.loras
        )
        self.lora.setText(words)
        self.lora.show()
        triggers = [lora.trigger for lora in item.loras if lora.trigger]
        if not triggers:
            self.trigger.hide()
        elif any(lora.injected for lora in item.loras):
            self.trigger.setText(tr("photoshoot.script.trigger", trigger=", ".join(triggers)))
            self.trigger.show()
        else:
            self.trigger.setText(tr("photoshoot.script.trigger.missed", trigger=", ".join(triggers)))
            self.trigger.show()


class PhotoshootScriptWindow(QDialog):
    """The non-modal script pop-up. Closing it hides the window, never the script."""

    def __init__(self, window: ShellWindow, model: PhotoshootScriptModel) -> None:
        """Build the hidden window; :meth:`open_window` shows it.

        Args:
            window: The main Canvas window. The pop-up is its child but a
                separate non-modal window, never docked into the stage.
            model: The script rows to show.
        """
        super().__init__(window)
        self.setObjectName("photoshootScriptWindow")
        self.setModal(False)
        self.setWindowTitle(tr("photoshoot.script.title"))
        self._model = model
        self.setStyleSheet(STYLE)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setMinimumSize(380, 320)
        column = QVBoxLayout(self)
        column.setContentsMargins(16, 16, 16, 16)
        column.setSpacing(12)
        heading = QLabel(tr("photoshoot.script.title"))
        heading.setObjectName("photoshootScriptHeading")
        column.addWidget(heading)
        self._list = QListWidget()
        self._list.setObjectName("photoshootScriptList")
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setSpacing(4)
        self._list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        column.addWidget(self._list)
        model.item_added.connect(self._on_item_added)
        model.item_updated.connect(self._on_item_updated)
        model.item_selected.connect(self._on_item_selected)
        model.count_changed.connect(self._on_count_changed)
        self._list.currentRowChanged.connect(self._on_row_changed)

    def open_window(self) -> None:
        """Show the pop-up above the main window, or bring it back after a close."""
        self.show()
        self.raise_()
        self.activateWindow()

    def done(self, result: int) -> None:
        """Any close path (X, Esc) hides the window and tells the model."""
        super().done(result)
        self._model.notify_window_closed()

    def _on_item_added(self, index: int) -> None:
        item = self._model.item(index)
        if item is None:
            return
        row = PhotoshootScriptRow(item)
        entry = QListWidgetItem(self._list)
        self._list.blockSignals(True)
        self._list.addItem(entry)
        self._list.setItemWidget(entry, row)
        self._list.blockSignals(False)
        self._fit(index)

    def _on_item_updated(self, index: int) -> None:
        row = self._row(index)
        item = self._model.item(index)
        if row is None or item is None:
            return
        row.show_item(item)
        self._fit(index)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Re-wrap the cards when the window width changes."""
        super().resizeEvent(event)
        for index in range(self._list.count()):
            self._fit(index)

    def _fit(self, index: int) -> None:
        """Give one card the height its wrapped prompt needs at the current list width."""
        row = self._row(index)
        entry = self._list.item(index)
        if row is None or entry is None:
            return
        width = max(1, self._list.viewport().width() - 2 * self._list.spacing())
        layout = row.layout()
        height = layout.totalHeightForWidth(width) if layout is not None else 0
        entry.setSizeHint(QSize(width, max(row.sizeHint().height(), height)))

    def _paint_selection(self) -> None:
        current = self._list.currentRow()
        for index in range(self._list.count()):
            row = self._row(index)
            if row is not None:
                row.set_selected(index == current)

    def _on_item_selected(self, index: int) -> None:
        if self._list.currentRow() != index:
            self._list.blockSignals(True)
            self._list.setCurrentRow(index)
            self._list.blockSignals(False)
        self._paint_selection()

    def _on_count_changed(self, count: int) -> None:
        if count == 0:
            self._list.clear()

    def _on_row_changed(self, row: int) -> None:
        self._model.select_item(row)
        self._paint_selection()

    def _row(self, index: int) -> PhotoshootScriptRow | None:
        entry = self._list.item(index)
        widget = None if entry is None else self._list.itemWidget(entry)
        return widget if isinstance(widget, PhotoshootScriptRow) else None


def _status_text(status: str) -> str:
    if status == "drawing":
        return tr("canvas.stage.draw")
    if status == "blocked":
        return tr("photoshoot.frame.blocked")
    if status == "kept":
        return tr("photoshoot.frame.kept")
    if status == "waiting":
        return tr("photoshoot.frame.waiting")
    return ""
