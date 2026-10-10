"""The two-mode window: text to image, and image to image."""
from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from canvas.app.info import APP_NAME, RESPONSIBILITY
from canvas.core.formats import DEFAULT_FORMAT, OutputFormat
from canvas.models.modes import Mode, Tier, mode_group, visible_modes
from canvas.ui.theme import build_style

_LABELS = {
    Mode.TEXT: "Text to image",
    Mode.TEXT_CREATIVE: "Creative",
    Mode.RESTYLE: "Restyle",
    Mode.INPAINT: "Inpaint",
    Mode.OUTPAINT: "Outpaint",
    Mode.POSE: "Pose",
    Mode.DEPTH: "Depth",
    Mode.EDGE: "Edge",
    Mode.INSTRUCTION: "Instruction edit",
}


class MainWindow(QWidget):
    """Prompt, mode and format. Generation is delegated to ``on_generate``."""

    def __init__(self, on_generate: Callable[[str, Mode, OutputFormat], str], tier: Tier, glass: bool = False) -> None:
        super().__init__()
        self.on_generate = on_generate
        self.tier = tier
        self.setObjectName("root")
        self.setWindowTitle(APP_NAME)
        self.setStyleSheet(build_style(glass))
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("Describe the picture")
        self.format_box = QComboBox()
        for fmt, label in ((OutputFormat.AVIF, "AVIF"), (OutputFormat.JXL, "JPEG XL"), (OutputFormat.WEBP, "WebP")):
            self.format_box.addItem(label, fmt)
        index = self.format_box.findData(DEFAULT_FORMAT)
        if index >= 0:
            self.format_box.setCurrentIndex(index)
        self.image_mode = QComboBox()
        for mode in visible_modes(tier):
            if mode_group(mode) == "image":
                self.image_mode.addItem(_LABELS[mode], mode)
        self.creative = QComboBox()
        self.creative.addItem(_LABELS[Mode.TEXT], Mode.TEXT)
        if Mode.TEXT_CREATIVE in visible_modes(tier):
            self.creative.addItem(_LABELS[Mode.TEXT_CREATIVE], Mode.TEXT_CREATIVE)
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.generate_button = QPushButton("Generate")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self._generate)
        tabs = QTabWidget()
        tabs.addTab(self._text_tab(), "Text to image")
        tabs.addTab(self._image_tab(), "Image to image")
        footer = QLabel(RESPONSIBILITY)
        footer.setObjectName("footer")
        footer.setWordWrap(True)
        layout = QVBoxLayout(self)
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        layout.addWidget(title)
        layout.addWidget(tabs)
        layout.addWidget(self.prompt)
        row = QHBoxLayout()
        row.addWidget(self.format_box)
        row.addWidget(self.generate_button)
        layout.addLayout(row)
        layout.addWidget(self.status)
        layout.addWidget(footer)

    def _text_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Quality"))
        layout.addWidget(self.creative)
        return page

    def _image_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Operation"))
        layout.addWidget(self.image_mode)
        return page

    def current_mode(self, tab_index: int) -> Mode:
        if tab_index == 0:
            return self.creative.currentData()
        return self.image_mode.currentData()

    def _generate(self) -> None:
        tabs = self.findChild(QTabWidget)
        index = 0 if tabs is None else tabs.currentIndex()
        mode = self.current_mode(index)
        fmt = self.format_box.currentData()
        try:
            self.status.setText(self.on_generate(self.prompt.toPlainText(), mode, fmt))
        except Exception as exc:
            self.status.setText(str(exc))
