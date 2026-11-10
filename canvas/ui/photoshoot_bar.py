"""Photoshoot option on the shared simple screen.

The option sits centred under the Create button, which stays the
primary action and runs the series while the option is on. The shot script
lives in a separate non-modal window (see :mod:`canvas.ui.photoshoot_script`),
never in the image stage; a ``Script (N)`` button beside the option reopens it.
Cancel stops after the frame that already finished.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from voxprint_common.i18n import tr
from voxprint_common.ui.simple_image import Phase

from canvas.pipeline.series_text import DEFAULT_FRAMES, MAX_FRAMES
from canvas.ui.photoshoot_script import PhotoshootScriptModel, PhotoshootScriptWindow, series_loras

if TYPE_CHECKING:
    from canvas.models.user_lora import UserLora
    from canvas.pipeline.photoshoot import Shot
    from canvas.ui.shell_screen import ShellWindow


FRAMES_PAD = 12
# Room for the up/down stepper on the right, so the digits are not clipped.
FRAMES_STEPPER = 18


def fit_frames(spin: QSpinBox) -> None:
    """Give the frame count room for its widest value, the stepper, and some air.

    Args:
        spin: The Frames spin box.
    """
    spin.ensurePolished()
    widest = max(spin.fontMetrics().horizontalAdvance(str(value)) for value in (spin.minimum(), spin.maximum()))
    spin.setMinimumWidth(widest + 2 * FRAMES_PAD + FRAMES_STEPPER)


class PhotoshootBar:
    """The mode option, the Script (N) reopen button, the progress line, and Cancel."""

    def __init__(self, window: ShellWindow) -> None:
        """Add the option and its controls to an open shell window.

        Args:
            window: A :class:`canvas.ui.shell_screen.ShellWindow`.
        """
        self.window = window
        self.writer: Callable[[str], str] | None = None
        self.enabled = False
        self.place = "same"
        self.seed_mode = "random"
        self.strength = "medium"
        self._stop = False
        self.subject = ""
        self.folder: Path | None = None
        self._showing = False
        self._lora: UserLora | None = None
        self._lora_strength: float | None = None
        self.script = PhotoshootScriptModel(window)
        self.script_window = PhotoshootScriptWindow(window, self.script)
        self._mount_mode()
        self.script.item_selected.connect(self._on_script_item_selected)
        self.script.count_changed.connect(self._on_script_count)
        self.script.window_closed.connect(self.sync)
        self.sync()

    @property
    def count(self) -> int:
        """How many frames this series should contain."""
        return int(self.frames.value())

    def line_for(self, index: int) -> str:
        """The read-only line for shot ``index``.

        Args:
            index: Row index in the current list.
        """
        item = self.script.item(index)
        return "" if item is None else item.prompt

    def begin_series(self, lora: UserLora | None, strength: float | None) -> None:
        """Remember the LoRA and strength this series draws with, for the script rows.

        Args:
            lora: The bound window LoRA, or None for a plain series.
            strength: The strength the series applies (already clamped).
        """
        self._lora = lora
        self._lora_strength = strength

    def show_scenario(self, subject: str, shots: tuple[Shot, ...], *, seed: int = 0) -> None:
        """Fill the script with the accepted shots before the first photograph.

        One row is appended per shot as its line is assembled, and the pop-up
        opens with the first content. The LoRA/trigger details come from the
        same bound LoRA and strength the series itself uses.

        Args:
            subject: Verbatim subject block.
            shots: Accepted shots.
            seed: Series seed. Each row keeps one item from a choice list.
        """
        from canvas.pipeline.photoshoot import assemble_line

        self.subject = subject
        self.script.begin()
        loras = series_loras(self._lora, self._lora_strength)
        for index, shot in enumerate(shots):
            line = assemble_line(subject, shot, include_camera=True, index=index, seed=seed, lora=self._lora)
            self.script.append_shot(line, loras)
        self.frame.setText("")
        self._showing = True
        self.script_window.open_window()
        self.sync()

    def mark(self, index: int, status: str) -> None:
        """Update one row's status and the progress line.

        Args:
            index: Row index.
            status: ``drawing``, ``kept``, or ``blocked``.
        """
        if self.script.item(index) is None:
            return
        self.script.update_shot(index, status=status)
        self.frame.setText(tr("photoshoot.progress.frame", current=index + 1, total=self.script.count))
        self.sync()

    def record_kept(self, item_id: str, kept: tuple[Path, ...]) -> None:
        """Link every kept frame to its script row, in kept order.

        Args:
            item_id: The grouped series history entry.
            kept: Kept frame paths, in draw order — the order the rows were
                marked ``kept`` in.
        """
        rows = [
            item.index for item in self.script.items() if item.status == "kept"
        ]
        for index, path in zip(rows, kept, strict=False):
            self.script.record_image(index, item_id, path)
        if self.window.shell.layout.selected == item_id:
            self.script.select_image(item_id)

    def sync(self) -> None:
        """Show the shot-area controls only after a scenario has been accepted."""
        busy = self.window.shell.layout.phase is Phase.BUSY
        self.mode.setEnabled(not busy)
        self.frames_label.setVisible(self.enabled)
        self.frames.setVisible(self.enabled)
        self.frames.setEnabled(self.enabled and not busy)
        running = busy and self._showing
        self.cancel.setVisible(running)
        self.frame.setVisible(running and bool(self.frame.text()))
        run_line = self.window.findChild(QWidget, "photoshootRun")
        if run_line is not None:
            run_line.setVisible(running)
        self.script_button.setVisible(self.script.count > 0 and not self.script_window.isVisible())

    def cancel_series(self) -> None:
        """Stop after the frame that already finished. Finished frames stay."""
        self._stop = True
        self.window.shell.layout.cancel()
        from canvas.ui.photoshoot_run import stop_series

        stop_series(self.window)

    def _mount_mode(self) -> None:
        improve = self.window.view.root.findChild(QPushButton, "improvePrompt")
        # A checkbox, so it reads as "this run will be a photoshoot", not as an action.
        self.mode = QCheckBox(tr("photoshoot.button"))
        self.mode.setObjectName("photoshootMode")
        self.mode.setToolTip(tr("photoshoot.tooltip"))
        self.mode.clicked.connect(self._toggle)
        self.frames_label = QLabel(tr("photoshoot.frames"))
        self.frames_label.setObjectName("photoshootFramesLabel")
        self.frames = QSpinBox()
        self.frames.setObjectName("photoshootFrames")
        self.frames.setRange(1, MAX_FRAMES)
        self.frames.setValue(DEFAULT_FRAMES)
        self.frames.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.frames.setCorrectionMode(QAbstractSpinBox.CorrectionMode.CorrectToNearestValue)
        self.frames.setAccelerated(True)
        # Keep a real right-hand stepper. A zero-width button, or NoButtons
        # with a stylesheet, collapsed the edit field to 1 px on Qt 6.12.
        self.frames.setStyleSheet(
            "QSpinBox#photoshootFrames::up-button, QSpinBox#photoshootFrames::down-button"
            f" {{ width: {FRAMES_STEPPER}px; }}"
        )
        self.frames.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fit_frames(self.frames)
        self.frames_label.setBuddy(self.frames)
        self.script_button = QPushButton(tr("photoshoot.script.button", count=0))
        self.script_button.setObjectName("photoshootScriptButton")
        self.script_button.setAutoDefault(False)
        self.script_button.setToolTip(tr("photoshoot.script.tooltip"))
        self.script_button.clicked.connect(self._open_script)
        self.frame = QLabel("")
        self.frame.setObjectName("photoshootFrame")
        self.cancel = QPushButton(tr("simple.cancel"))
        self.cancel.setObjectName("photoshootCancel")
        self.cancel.setAutoDefault(False)
        self.cancel.clicked.connect(self.cancel_series)
        if improve is not None:
            layout = _layout_containing(improve)
            if layout is not None:
                index = layout.indexOf(improve) + 1
                layout.insertWidget(index, self.mode)
                layout.insertWidget(index + 1, self.frames_label)
                layout.insertWidget(index + 2, self.frames)
                layout.insertWidget(index + 3, self.script_button)
                layout.insertWidget(index + 4, self.frame)
                layout.insertWidget(index + 5, self.cancel)

    def _toggle(self) -> None:
        self.enabled = self.mode.isChecked()
        if not self.enabled:
            self._showing = False
        self.sync()

    def _open_script(self) -> None:
        self.script_window.open_window()
        self.sync()

    def _on_script_count(self, count: int) -> None:
        self.script_button.setText(tr("photoshoot.script.button", count=count))
        self.sync()

    def _on_script_item_selected(self, index: int) -> None:
        """A script row was clicked: select its frame in the main window when it has one."""
        item = self.script.item(index) if index >= 0 else None
        if item is None or not item.image_id:
            return
        layout = self.window.shell.layout
        if layout.selected == item.image_id:
            return
        layout.select(item.image_id)
        self.window.view.refresh()


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
