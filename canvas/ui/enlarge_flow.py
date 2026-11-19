"""Confirm and install VOXPrint Enlarge from the 4K chip.

The download itself is the shared model queue. This module draws the question,
the progress line, and the plain error.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Literal

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from voxprint_common.i18n import tr
from voxprint_common.mdl.catalog import CatalogError
from voxprint_common.mdl.control import Controller, app_outcome
from voxprint_common.mdl.present import format_size
from voxprint_common.mdl.queue import load_queue

from canvas.app.i18n import text
from canvas.models.mdl_registry import enlarge_download_bytes, enlarge_manifest_path
from canvas.ui.shell_chrome import shell_style

EnlargeOutcome = Literal["ready", "failed", "cancelled"]
Installer = Callable[["EnlargeProgress"], EnlargeOutcome]

_WIDTH = 480


def enlarge_question() -> str:
    """The confirm sentence. A registry size is included only when one is known."""
    amount = enlarge_download_bytes()
    if amount is None:
        return text("canvas.enlarge.confirm")
    return text("canvas.enlarge.confirm.size", size=format_size(amount))


class EnlargeConfirm(QDialog):
    """Asks whether to install Enlarge before 4K can be selected."""

    def __init__(self, parent: QWidget | None, message: str) -> None:
        """Build the question.

        Args:
            parent: The shell window.
            message: The already translated question, with a size when known.
        """
        super().__init__(parent)
        self.setObjectName("enlargeConfirm")
        self.setWindowTitle(text("canvas.enlarge.title"))
        self.setStyleSheet(shell_style())
        self.setFixedWidth(_WIDTH)
        self.setModal(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(14)
        prompt = QLabel(message)
        prompt.setObjectName("enlargePrompt")
        prompt.setWordWrap(True)
        prompt.setFixedWidth(_WIDTH - 32)
        layout.addWidget(prompt)
        row = QHBoxLayout()
        row.setSpacing(8)
        install = QPushButton(tr("simple.output.install"))
        install.setObjectName("enlargeInstall")
        install.setAutoDefault(False)
        install.setProperty("primary", True)
        install.clicked.connect(self.accept)
        cancel = QPushButton(tr("simple.cancel"))
        cancel.setObjectName("enlargeCancel")
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.reject)
        _fit_button(install)
        _fit_button(cancel)
        row.addWidget(install)
        row.addWidget(cancel)
        row.addStretch(1)
        layout.addLayout(row)


class EnlargeProgress(QDialog):
    """The install progress. Cancel stops the shared queue."""

    def __init__(self, parent: QWidget | None) -> None:
        """Build the progress line and Cancel.

        Args:
            parent: The shell window.
        """
        super().__init__(parent)
        self.setObjectName("enlargeProgress")
        self.setWindowTitle(text("canvas.enlarge.title"))
        self.setStyleSheet(shell_style())
        self.setFixedWidth(_WIDTH)
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self._cancelled = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(14)
        self._label = QLabel(text("canvas.enlarge.progress"))
        self._label.setObjectName("enlargeProgressLabel")
        self._label.setWordWrap(True)
        self._label.setFixedWidth(_WIDTH - 32)
        layout.addWidget(self._label)
        self._bar = QProgressBar()
        self._bar.setObjectName("enlargeProgressBar")
        self._bar.setRange(0, 0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(10)
        layout.addWidget(self._bar)
        row = QHBoxLayout()
        self._cancel = QPushButton(tr("simple.cancel"))
        self._cancel.setObjectName("enlargeProgressCancel")
        self._cancel.setAutoDefault(False)
        self._cancel.clicked.connect(self._stop)
        _fit_button(self._cancel)
        row.addWidget(self._cancel)
        row.addStretch(1)
        layout.addLayout(row)

    def was_cancelled(self) -> bool:
        """True after Cancel."""
        return self._cancelled

    def report(self, done: int, total: int) -> None:
        """Show byte progress when the queue has a total.

        Args:
            done: Bytes already transferred.
            total: Bytes the queue expects. Zero keeps the bar busy.
        """
        if total <= 0:
            self._bar.setRange(0, 0)
            self._label.setText(text("canvas.enlarge.progress"))
            return
        self._bar.setRange(0, total)
        self._bar.setValue(min(max(0, done), total))
        self._label.setText(
            text(
                "canvas.enlarge.progress.bytes",
                done=format_size(max(0, done)),
                total=format_size(total),
            )
        )

    def _stop(self) -> None:
        self._cancelled = True
        self._cancel.setEnabled(False)


class EnlargeError(QDialog):
    """The plain failure. 4K stays unselected."""

    def __init__(self, parent: QWidget | None) -> None:
        """Build the error sentence.

        Args:
            parent: The shell window.
        """
        super().__init__(parent)
        self.setObjectName("enlargeError")
        self.setWindowTitle(text("canvas.enlarge.title"))
        self.setStyleSheet(shell_style())
        self.setFixedWidth(_WIDTH)
        self.setModal(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(14)
        message = QLabel(text("canvas.enlarge.failed"))
        message.setObjectName("enlargeErrorText")
        message.setWordWrap(True)
        message.setFixedWidth(_WIDTH - 32)
        layout.addWidget(message)
        close = QPushButton(text("app.close"))
        close.setObjectName("enlargeErrorClose")
        close.setAutoDefault(False)
        close.clicked.connect(self.accept)
        _fit_button(close)
        layout.addWidget(close, 0, Qt.AlignmentFlag.AlignLeft)


def run_shared_enlarge(progress: EnlargeProgress) -> EnlargeOutcome:
    """Install Enlarge through the shared model queue.

    Cancel on ``progress`` stops that queue. A missing manifest is a failure.

    Args:
        progress: The dialog whose Cancel flag and byte line this updates.
    """
    controller = Controller()
    try:
        job_id = controller.enqueue_app("enlarge", manifest=enlarge_manifest_path())
        controller.start(display=False)
    except (CatalogError, OSError, RuntimeError, ValueError):
        return "failed"
    app = QApplication.instance()
    started = time.monotonic()
    while True:
        if progress.was_cancelled():
            controller.cancel()
            return "cancelled"
        snap = controller.snapshot()
        progress.report(snap.bytes_done, snap.bytes_total)
        outcome = app_outcome("enlarge", snap)
        if outcome == "ready":
            return "ready"
        if outcome == "failed":
            return "failed"
        if outcome == "cancelled":
            return "cancelled"
        queued = _job_outcome(job_id)
        if queued is not None:
            return queued
        if time.monotonic() - started > 24 * 60 * 60:
            controller.cancel()
            return "failed"
        if isinstance(app, QApplication):
            app.processEvents()
        time.sleep(0.05)


def _job_outcome(job_id: str) -> EnlargeOutcome | None:
    try:
        jobs = load_queue()
    except CatalogError:
        return "failed"
    job = next((item for item in jobs if item.id == job_id), None)
    if job is None:
        return None
    if job.state == "done":
        return "ready"
    if job.state == "failed":
        return "failed"
    if job.state == "cancelled":
        return "cancelled"
    return None


def _fit_button(button: QPushButton) -> None:
    words = button.text()
    if not words:
        return
    font = QFont(button.font())
    if button.property("primary"):
        font.setWeight(QFont.Weight.DemiBold)
    advance = QFontMetrics(font).horizontalAdvance(words)
    button.setMinimumWidth(advance + 36)
