"""Photoshoot controls on the shared simple screen.

The option sits beside Improve prompt. While it is on, the shell Create button
is hidden and the panel's Create button is the only primary action. Camera
detail stays in Expert. The drag handle and the one menu stay on every row.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QCursor, QDesktopServices
from PySide6.QtWidgets import (
    QBoxLayout,
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from voxprint_common.i18n import tr
from voxprint_common.ui.simple_image import Phase
from voxprint_common.ui.theme import (
    CARD_PLAIN,
    CONTROL_BG,
    CONTROL_BORDER,
    PRIMARY,
    PRIMARY_BORDER,
    PRIMARY_DISABLED_BG,
    PRIMARY_DISABLED_TEXT,
    TEXT,
    TEXT_ON_PRIMARY,
)

from canvas.pipeline.photoshoot import Shot, assemble_line

if TYPE_CHECKING:
    from canvas.ui.shell_screen import ShellWindow


@dataclass
class ShotRow:
    """One visible shot. Expert can edit it until it starts."""

    shot: Shot
    plain: str
    started: bool
    status: str
    editing: bool
    widget: QWidget
    label: QLabel
    editor: QLineEdit
    camera: QLineEdit
    handle: QPushButton
    menu: QPushButton


class PhotoshootBar:
    """The mode chip, the everyday rows, and the series board."""

    def __init__(self, window: ShellWindow) -> None:
        """Add the option and the rows to an open shell window.

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
        self._host = self._mount_rows()
        self.board = self._mount_board()
        self.sync()

    def line_for(self, index: int) -> str:
        """The line for shot ``index``, including the camera only in Expert.

        Args:
            index: Row index in the current list.
        """
        row = self.rows[index]
        text = row.editor.text().strip() if self._expert() or row.editing else row.plain
        camera = row.camera.text().strip()
        if self._expert() and camera:
            return f"{text}, {camera}" if text else camera
        return text

    def show_scenario(self, subject: str, shots: tuple[Shot, ...]) -> None:
        """Fill the read-only list before the first frame.

        Args:
            subject: Verbatim subject block.
            shots: Accepted shots.
        """
        self.subject = subject
        self._clear_rows()
        self._fill_grid(len(shots))
        for shot in shots:
            self._add_row(shot, assemble_line(subject, shot, include_camera=False))
        self.frame.setText("")
        self._showing = True
        self.sheet.setEnabled(False)
        self.zip_button.setEnabled(False)
        self.folder_button.setEnabled(False)
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
        if status == "drawing":
            row.started = True
        note = _status_text(status)
        row.label.setText(row.plain if not note else tr("photoshoot.row.status", line=row.plain, note=note))
        if index < len(self._cells):
            self._cells[index].setText(_cell_text(status))
        if status == "kept":
            self.sheet.setEnabled(True)
            self.zip_button.setEnabled(True)
            self.folder_button.setEnabled(self.folder is not None)
        current = index + 1
        total = len(self.rows)
        self.frame.setText(tr("photoshoot.progress.frame", current=current, total=total))
        self.sync()

    def move_row(self, source: int, target: int) -> None:
        """Reorder two shots that have not started.

        Args:
            source: Dragged row.
            target: Row under the pointer.
        """
        if source == target or source < 0 or target < 0:
            return
        if source >= len(self.rows) or target >= len(self.rows):
            return
        if self.rows[source].started or self.rows[target].started:
            return
        row = self.rows.pop(source)
        self.rows.insert(target, row)
        self._repack()

    def sync(self) -> None:
        """Show the Photoshoot rows only while the option is selected."""
        self._host.setVisible(self.enabled)
        self._set_row_visible("count", not self.enabled)
        self._set_row_visible("style", not self.enabled)
        busy = self.window.shell.layout.phase is Phase.BUSY
        for button in self._mode_buttons:
            button.setEnabled(not busy)
        shell_generate = self.window.findChild(QPushButton, "generate")
        if shell_generate is not None:
            shell_generate.setVisible(not self.enabled)
        self.primary.setVisible(self.enabled)
        self.primary.setEnabled(self.enabled and not busy)
        self.board.setVisible(self._showing)
        stack = getattr(self.window.view, "_stack", None)
        if stack is not None:
            stack.setVisible(not self._showing)
        self.cancel.setVisible(self.window.shell.layout.phase is Phase.BUSY and self._showing)
        self.sync_expert(self._expert())

    def sync_expert(self, enabled: bool) -> None:
        """Show the camera field only in Expert. The handle and menu stay.

        Args:
            enabled: True when Expert mode is on.
        """
        self._expert_row.setVisible(enabled and self.enabled)
        for row in self.rows:
            row.handle.setVisible(True)
            row.menu.setVisible(True)
            row.camera.setVisible(enabled)
            editing = enabled or row.editing
            row.editor.setVisible(editing)
            row.label.setVisible(not editing)

    def _expert(self) -> bool:
        return bool(self.window._expert.enabled())

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
        self._mode_buttons = [self.mode]

    def _toggle(self) -> None:
        self.enabled = self.mode.isChecked()
        if self.enabled:
            self.window.shell.layout.choose("style", "photo")
        else:
            self._showing = False
        self.sync()

    def _mount_rows(self) -> QWidget:
        host = QWidget()
        host.setObjectName("photoshootRows")
        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        self._count_group = self._chip_row(
            column, "photoshoot.count", "count", (("6", "6"), ("12", "12"), ("20", "20")), "6", self._choose_count,
        )
        self._style_group = self._chip_row(
            column,
            "simple.style",
            "style",
            (("photo", "simple.style.photo"), ("cinematic", "simple.style.cinematic"), ("portrait", "simple.style.portrait")),
            "photo",
            self._choose_style,
        )
        self._place_group = self._chip_row(
            column,
            "photoshoot.place",
            "place",
            (("same", "photoshoot.place.same"), ("different", "photoshoot.place.different")),
            "same",
            self._choose_place,
        )
        self._expert_row = self._expert_chips(column)
        preset = self.window.view._preset_rows
        preset.addWidget(host)
        return host

    def _chip_row(self, column: QVBoxLayout, label_key: str, row_id: str, choices: tuple[tuple[str, str], ...], default: str, on_pick) -> QButtonGroup:
        line = QHBoxLayout()
        line.setSpacing(8)
        label = QLabel(tr(label_key))
        label.setObjectName(f"photoshoot-{row_id}")
        label.setProperty("presetLabel", True)
        label.setMinimumWidth(108)
        line.addWidget(label)
        group = QButtonGroup(self.window)
        group.setExclusive(True)
        for value, key in choices:
            button = QPushButton(tr(key))
            button.setObjectName(f"chip-photoshoot-{row_id}-{value}")
            button.setProperty("chip", True)
            button.setCheckable(True)
            button.setAutoDefault(False)
            button.setChecked(value == default)
            button.clicked.connect(lambda _checked=False, picked=value: on_pick(picked))
            group.addButton(button)
            line.addWidget(button)
            self._mode_buttons.append(button)
        line.addStretch(1)
        column.addLayout(line)
        return group

    def _expert_chips(self, column: QVBoxLayout) -> QWidget:
        host = QWidget()
        host.setObjectName("photoshootExpert")
        line = QHBoxLayout(host)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(8)
        line.addWidget(QLabel(tr("photoshoot.expert.seed")))
        seeds = QButtonGroup(host)
        seeds.setExclusive(True)
        for value, key in (
            ("series", "photoshoot.expert.seed.series"),
            ("locked", "photoshoot.expert.seed.locked"),
            ("random", "photoshoot.expert.seed.random"),
        ):
            button = self._expert_button(f"chip-photoshoot-seed-{value}", key, value == "series")
            button.clicked.connect(lambda _checked=False, picked=value: self._choose_seed(picked))
            seeds.addButton(button)
            line.addWidget(button)
        line.addWidget(QLabel(tr("photoshoot.expert.strength")))
        strengths = QButtonGroup(host)
        strengths.setExclusive(True)
        for value, key in (
            ("high", "photoshoot.expert.strength.high"),
            ("medium", "photoshoot.expert.strength.medium"),
            ("low", "photoshoot.expert.strength.low"),
        ):
            button = self._expert_button(f"chip-photoshoot-strength-{value}", key, value == "medium")
            button.clicked.connect(lambda _checked=False, picked=value: self._choose_strength(picked))
            strengths.addButton(button)
            line.addWidget(button)
        line.addStretch(1)
        column.addWidget(host)
        self.primary = QPushButton(tr("simple.generate"))
        self.primary.setObjectName("photoshootCreate")
        self.primary.setAutoDefault(False)
        self.primary.setMinimumHeight(44)
        self.primary.setStyleSheet(
            f"QPushButton#photoshootCreate {{ background: {PRIMARY}; color: {TEXT_ON_PRIMARY}; "
            f"border: 1px solid {PRIMARY_BORDER}; font-weight: 600; padding: 8px 22px; border-radius: 8px; }}"
            f"QPushButton#photoshootCreate:disabled {{ background: {PRIMARY_DISABLED_BG}; "
            f"color: {PRIMARY_DISABLED_TEXT}; }}"
        )
        self.primary.clicked.connect(self.window.view._generate)
        column.addWidget(self.primary)
        return host

    def _expert_button(self, name: str, key: str, checked: bool) -> QPushButton:
        button = QPushButton(tr(key))
        button.setObjectName(name)
        button.setProperty("chip", True)
        button.setCheckable(True)
        button.setChecked(checked)
        button.setAutoDefault(False)
        return button

    def _choose_count(self, value: str) -> None:
        self.count = int(value)

    def _choose_style(self, value: str) -> None:
        self.window.shell.layout.choose("style", value)

    def _choose_place(self, value: str) -> None:
        self.place = value

    def _choose_seed(self, value: str) -> None:
        self.seed_mode = value

    def _choose_strength(self, value: str) -> None:
        self.strength = value

    def _mount_board(self) -> QWidget:
        board = QWidget()
        board.setObjectName("photoshootBoard")
        board.setStyleSheet(
            f"QWidget#photoshootBoard {{ background: {CARD_PLAIN}; }}"
            f"QLabel {{ color: {TEXT}; }}"
            f"QLineEdit {{ background: {CONTROL_BG}; color: {TEXT}; border: 1px solid {CONTROL_BORDER}; }}"
        )
        outer = QVBoxLayout(board)
        outer.setContentsMargins(12, 12, 12, 12)
        body = QHBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("photoshootList")
        self._list_host = QWidget()
        column = QVBoxLayout(self._list_host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)
        column.addStretch(1)
        scroll.setWidget(self._list_host)
        body.addWidget(scroll, 1)
        self._grid_host = QWidget()
        self._grid_host.setObjectName("photoshootGrid")
        self._grid_layout = QGridLayout(self._grid_host)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(8)
        self._cells: list[QLabel] = []
        body.addWidget(self._grid_host, 1)
        outer.addLayout(body, 1)
        footer = QHBoxLayout()
        self.folder_button = self._action("photoshootFolder", "photoshoot.folder", self._open_folder)
        self.sheet = self._action("photoshootSheet", "photoshoot.export.sheet", self._open_sheet)
        self.zip_button = self._action("photoshootZip", "photoshoot.export.zip", self._open_zip)
        self.frame = QLabel("")
        self.frame.setObjectName("photoshootFrame")
        self.cancel = QPushButton(tr("simple.cancel"))
        self.cancel.setObjectName("photoshootCancel")
        self.cancel.setAutoDefault(False)
        self.cancel.clicked.connect(self.cancel_series)
        footer.addWidget(self.folder_button)
        footer.addWidget(self.sheet)
        footer.addWidget(self.zip_button)
        footer.addStretch(1)
        footer.addWidget(self.frame)
        footer.addWidget(self.cancel)
        outer.addLayout(footer)
        _column(self.window.view._stage).insertWidget(0, board)
        board.hide()
        return board

    def _fill_grid(self, count: int) -> None:
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._cells = []
        columns = 3 if count <= 6 else 4
        for index in range(count):
            cell = QLabel(tr("photoshoot.frame.waiting"))
            cell.setObjectName("photoshootCell")
            cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cell.setMinimumSize(88, 64)
            cell.setStyleSheet(f"background: {CONTROL_BG}; color: {TEXT}; border: 1px solid {CONTROL_BORDER};")
            self._grid_layout.addWidget(cell, index // columns, index % columns)
            self._cells.append(cell)

    def _action(self, name: str, key: str, slot) -> QPushButton:
        button = QPushButton(tr(key))
        button.setObjectName(name)
        button.setAutoDefault(False)
        button.setEnabled(False)
        button.clicked.connect(slot)
        return button

    def _add_row(self, shot: Shot, plain: str) -> None:
        widget = QWidget()
        line = QHBoxLayout(widget)
        line.setContentsMargins(0, 0, 0, 0)
        handle = QPushButton(tr("photoshoot.handle.glyph"))
        handle.setObjectName("photoshootHandle")
        handle.setToolTip(tr("photoshoot.handle"))
        handle.setAutoDefault(False)
        handle.setProperty("chip", True)
        handle.released.connect(lambda button=handle: self._drop(button))
        menu = QPushButton(tr("photoshoot.menu"))
        menu.setObjectName("photoshootMenu")
        menu.setAutoDefault(False)
        menu.clicked.connect(lambda _checked=False, button=menu: self._open_menu(button))
        label = QLabel(plain)
        label.setObjectName("photoshootLine")
        label.setWordWrap(True)
        editor = QLineEdit(plain)
        editor.setObjectName("photoshootLineEdit")
        camera = QLineEdit(shot.camera)
        camera.setObjectName("photoshootCamera")
        camera.setPlaceholderText(tr("photoshoot.expert.camera"))
        line.addWidget(handle)
        line.addWidget(menu)
        line.addWidget(label, 1)
        line.addWidget(editor, 1)
        line.addWidget(camera)
        layout = _column(self._list_host)
        layout.insertWidget(max(0, layout.count() - 1), widget)
        self.rows.append(ShotRow(shot, plain, False, "waiting", False, widget, label, editor, camera, handle, menu))
        self.sync_expert(self._expert())

    def _drop(self, handle: QPushButton) -> None:
        source = next((index for index, row in enumerate(self.rows) if row.handle is handle), -1)
        local = self._list_host.mapFromGlobal(QCursor.pos())
        target = source
        for index, row in enumerate(self.rows):
            if row.widget.geometry().contains(local):
                target = index
                break
        self.move_row(source, target)

    def _open_menu(self, button: QPushButton) -> None:
        index = next((item for item, row in enumerate(self.rows) if row.menu is button), -1)
        if index < 0 or self.rows[index].started:
            return
        menu = QMenu(button)
        menu.addAction(tr("photoshoot.shot.edit"), lambda: self._edit(index))
        menu.addAction(tr("photoshoot.shot.duplicate"), lambda: self._duplicate(index))
        menu.addAction(tr("simple.history.delete"), lambda: self._delete(index))
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

    def _edit(self, index: int) -> None:
        if index < 0 or index >= len(self.rows) or self.rows[index].started:
            return
        self.rows[index].editing = True
        self.sync_expert(self._expert())
        self.rows[index].editor.setFocus()

    def _duplicate(self, index: int) -> None:
        if len(self.rows) >= 20 or self.rows[index].started:
            return
        row = self.rows[index]
        shot = Shot(row.shot.scene, row.shot.pose, row.camera.text(), row.shot.place)
        self._add_row(shot, row.editor.text().strip() or row.plain)
        self.count = len(self.rows)

    def _delete(self, index: int) -> None:
        if self.rows[index].started or len(self.rows) <= 6:
            return
        row = self.rows.pop(index)
        row.widget.setParent(None)
        row.widget.deleteLater()
        self.count = len(self.rows)

    def _clear_rows(self) -> None:
        for row in self.rows:
            row.widget.setParent(None)
            row.widget.deleteLater()
        self.rows.clear()

    def _repack(self) -> None:
        layout = _column(self._list_host)
        while layout.count():
            layout.takeAt(0)
        for row in self.rows:
            layout.addWidget(row.widget)
        layout.addStretch(1)

    def _set_row_visible(self, row_id: str, visible: bool) -> None:
        root = self.window.view.root
        for widget in root.findChildren(QWidget):
            name = widget.objectName()
            if name == f"preset-{row_id}" or name.startswith(f"chip-{row_id}-") or name.startswith(f"reset-{row_id}"):
                widget.setVisible(visible)

    def cancel_series(self) -> None:
        """Stop after the frame that already finished. Finished frames stay."""
        self._stop = True
        self.window.shell.layout.cancel()

    def _open_folder(self) -> None:
        _open_path(self.folder)

    def _open_sheet(self) -> None:
        if self.folder is not None:
            _open_path(self.folder / "overview.png")

    def _open_zip(self) -> None:
        if self.folder is not None:
            _open_path(self.folder / "series.zip")


def _column(widget: QWidget) -> QBoxLayout:
    layout = widget.layout()
    if not isinstance(layout, QBoxLayout):
        raise RuntimeError("the photoshoot row has no column")
    return layout


def _cell_text(status: str) -> str:
    if status == "drawing":
        return tr("canvas.stage.draw")
    if status == "kept":
        return tr("photoshoot.frame.kept")
    if status == "blocked":
        return tr("photoshoot.frame.blocked")
    if status == "waiting":
        return tr("photoshoot.frame.waiting")
    return tr("photoshoot.frame.waiting")


def _status_text(status: str) -> str:
    if status == "drawing":
        return tr("canvas.stage.draw")
    if status == "kept":
        return ""
    if status == "blocked":
        return tr("photoshoot.frame.blocked")
    if status == "waiting":
        return ""
    return ""


def _open_path(path: object) -> None:
    if path is None:
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


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
