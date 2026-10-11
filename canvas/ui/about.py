"""About dialog. The version line is read from ``BUILD.json``."""
from __future__ import annotations

from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout, QWidget

from canvas.app.identity import version_label
from canvas.app.info import APP_NAME


class AboutDialog(QDialog):
    """Shows the app name and the release identity."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        self.version_line = version_label()
        layout = QVBoxLayout(self)
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        self.version = QLabel(self.version_line)
        self.version.setObjectName("about-version")
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        layout.addWidget(title)
        layout.addWidget(self.version)
        layout.addWidget(close)
