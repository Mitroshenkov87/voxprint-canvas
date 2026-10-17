"""Photoshoot option on the shared simple screen.

The option sits beside Improve prompt. The shell Create button stays the
primary action and runs the series while the option is on. The shot list is
read-only. Cancel stops after the frame that already finished.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QBoxLayout, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from voxprint_common.i18n import tr
from voxprint_common.ui.simple_image import Phase
from voxprint_common.ui.theme import CARD_PLAIN, TEXT

from canvas.pipeline.photoshoot import Shot, assemble_line

if TYPE_CHECKING:
    from canvas.ui.shell_screen import ShellWindow


@dataclass
class ShotRow:
    """One read-only shot line."""

    plain: str
    status: str
    label: QLabel


class PhotoshootBar:
    """The mode option, the read-only shot list, and Cancel."""

    def __init__(self, window: ShellWindow) -> None:
        """Add the option and the list to an open shell window.

        Args:
            window: A :class:`canvas.ui.shell_screen.ShellWindow`.
        """
        self.window = window
        self.writer: Callable[[str], str] | None = None
        self.enabled = False
        self.count = 6
        self.place = "same"
        self.seed_mode = "series"
        self.strength = "medium"
        self._stop = False
        self.rows: list[ShotRow] = []
        self.subject = ""
        self.folder: Path | None = None
        self._showing = False
        self._mount_mode()
        self.board = self._mount_board()
        self.sync()

    def line_for(self, index: int) -> str:
        """The read-only line for shot ``index``.

        Args:
            index: Row index in the current list.
        """
        return self.rows[index].plain

    def show_scenario(self, subject: str, shots: tuple[Shot, ...]) -> None:
        """Fill the read-only list before the first frame.

        Args:
            subject: Verbatim subject block.
            shots: Accepted shots.
        """
        self.subject = subject
        self._clear_rows()
        for shot in shots:
            self._add_row(assemble_line(subject, shot, include_camera=False))
        self.frame.setText("")
        self._showing = True
        self.sync()

    def mark(self, index: int, status: str) -> None:
        """Update one row and the progress line.

        Args:
            index: Row index.
            status: ``drawing``, ``kept``, or ``blocked``.
        """
        if index < 0 or index >= len(self.rows):
            return
        row = self.rows[index]
        row.status = status
        note = _note(status)
        row.label.setText(row.plain if not note else tr("photoshoot.row.status", line=row.plain, note=note))
        self.frame.setText(tr("photoshoot.progress.frame", current=index + 1, total=len(self.rows)))
        self.sync()

    def sync(self) -> None:
        """Show the shot list only after a scenario has been accepted."""
        busy = self.window.shell.layout.phase is Phase.BUSY
        self.mode.setEnabled(not busy)
        self.board.setVisible(self._showing)
        self.cancel.setVisible(busy and self._showing)

    def cancel_series(self) -> None:
        """Stop after the frame that already finished. Finished frames stay."""
        self._stop = True
        self.window.shell.layout.cancel()

    def _mount_mode(self) -> None:
        improve = self.window.view.root.findChild(QPushButton, "improvePrompt")
        self.mode = QPushButton(tr("photoshoot.button"))
        self.mode.setObjectName("photoshootMode")
        self.mode.setCheckable(True)
        self.mode.setAutoDefault(False)
        self.mode.setProperty("chip", True)
        self.mode.setToolTip(tr("photoshoot.tooltip"))
        self.mode.clicked.connect(self._toggle)
        if improve is not None:
            layout = _layout_containing(improve)
            if layout is not None:
                layout.insertWidget(layout.indexOf(improve) + 1, self.mode)

    def _toggle(self) -> None:
        self.enabled = self.mode.isChecked()
        if not self.enabled:
            self._showing = False
        self.sync()

    def _mount_board(self) -> QWidget:
        board = QWidget()
        board.setObjectName("photoshootBoard")
        board.setMinimumHeight(188)
        board.setStyleSheet(
            f"QWidget#photoshootBoard {{ background: {CARD_PLAIN}; }}"
            f"QLabel {{ color: {TEXT}; background: transparent; }}"
            f"QScrollArea, QWidget#photoshootLines {{ background: {CARD_PLAIN}; border: none; }}"
        )
        outer = QVBoxLayout(board)
        outer.setContentsMargins(8, 8, 8, 8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumHeight(96)
        scroll.setObjectName("photoshootList")
        _fill(scroll.viewport(), CARD_PLAIN)
        self._list_host = QWidget()
        self._list_host.setObjectName("photoshootLines")
        _fill(self._list_host, CARD_PLAIN)
        column = QVBoxLayout(self._list_host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addStretch(1)
        scroll.setWidget(self._list_host)
        outer.addWidget(scroll, 1)
        footer = QHBoxLayout()
        self.frame = QLabel("")
        self.frame.setObjectName("photoshootFrame")
        self.cancel = QPushButton(tr("simple.cancel"))
        self.cancel.setObjectName("photoshootCancel")
        self.cancel.setAutoDefault(False)
        self.cancel.clicked.connect(self.cancel_series)
        footer.addWidget(self.frame, 1)
        footer.addWidget(self.cancel)
        outer.addLayout(footer)
        _column(self.window.view._stage).insertWidget(0, board)
        board.hide()
        return board

    def _add_row(self, plain: str) -> None:
        label = QLabel(plain)
        label.setObjectName("photoshootLine")
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {TEXT}; background: transparent;")
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout = _column(self._list_host)
        layout.insertWidget(max(0, layout.count() - 1), label)
        self.rows.append(ShotRow(plain, "waiting", label))

    def _clear_rows(self) -> None:
        for row in self.rows:
            row.label.setParent(None)
            row.label.deleteLater()
        self.rows.clear()


def _fill(widget: QWidget, colour: str) -> None:
    widget.setAutoFillBackground(True)
    palette = widget.palette()
    painted = QColor(colour)
    palette.setColor(QPalette.ColorRole.Window, painted)
    palette.setColor(QPalette.ColorRole.Base, painted)
    widget.setPalette(palette)
    widget.setStyleSheet(f"background: {colour};")


def _column(widget: QWidget) -> QBoxLayout:
    layout = widget.layout()
    if not isinstance(layout, QBoxLayout):
        raise RuntimeError("the photoshoot row has no column")
    return layout


def _note(status: str) -> str:
    if status == "drawing":
        return tr("canvas.stage.draw")
    if status == "blocked":
        return tr("photoshoot.frame.blocked")
    if status == "kept" or status == "waiting":
        return ""
    return ""


def _layout_containing(widget: QWidget):
    parent = widget.parentWidget()
    if parent is None or parent.layout() is None:
        return None
    return _find_layout(parent.layout(), widget)


def _find_layout(layout, widget: QWidget):
    for index in range(layout.count()):
        item = layout.itemAt(index)
        if item is None:
            continue
        if item.widget() is widget:
            return layout
        child = item.layout()
        if child is not None:
            found = _find_layout(child, widget)
            if found is not None:
                return found
    return None
