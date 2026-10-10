"""The two-mode window: text to image, and image to image."""
from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from canvas.app.info import APP_NAME, ENLARGE_HINT, RESPONSIBILITY, THIRD_PARTY_NOTICE
from canvas.core.formats import DEFAULT_FORMAT, OutputFormat
from canvas.models.modes import Mode, Tier, mode_group, visible_modes
from canvas.models.sizes import UI_ASPECTS, closest_aspect, is_portrait, paired_aspect
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

    def __init__(self, on_generate: Callable[[str, Mode, OutputFormat], str], tier: Tier, glass: bool = False,
                 enlarge_ready: bool = False) -> None:
        super().__init__()
        self.on_generate = on_generate
        self.tier = tier
        self._sync = False
        self._aspect_touched = False
        self._input_size: tuple[int, int] | None = None
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
        self.aspect_box = QComboBox()
        self.aspect_box.setObjectName("aspect")
        for aspect in UI_ASPECTS:
            self.aspect_box.addItem(aspect, aspect)
        self.orientation = QPushButton("Landscape")
        self.orientation.setObjectName("orientation")
        self.orientation.setCheckable(True)
        self.aspect_box.currentIndexChanged.connect(self._aspect_changed)
        self.orientation.toggled.connect(self._orientation_changed)
        self.fourk = QCheckBox("Output size: 4K")
        self.fourk.setObjectName("fourk")
        self.fourk_hint = QLabel(ENLARGE_HINT)
        self.fourk_hint.setObjectName("fourk-hint")
        self.fourk_hint.setWordWrap(True)
        self.set_enlarge_ready(enlarge_ready)
        self.generate_button = QPushButton("Generate")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self._generate)
        tabs = QTabWidget()
        tabs.addTab(self._text_tab(), "Text to image")
        tabs.addTab(self._image_tab(), "Image to image")
        self.addons_box = QCheckBox("Third-party add-ons")
        self.addons_box.setChecked(False)
        self.addons_notice = QLabel(THIRD_PARTY_NOTICE)
        self.addons_notice.setObjectName("third-party-notice")
        self.addons_notice.setWordWrap(True)
        self.addons_notice.setVisible(False)
        self.addons_box.toggled.connect(self.addons_notice.setVisible)
        footer = QLabel(RESPONSIBILITY)
        footer.setObjectName("footer")
        footer.setWordWrap(True)
        layout = QVBoxLayout(self)
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        layout.addWidget(title)
        layout.addWidget(tabs)
        layout.addWidget(self.prompt)
        aspect_row = QHBoxLayout()
        aspect_row.addWidget(QLabel("Aspect"))
        aspect_row.addWidget(self.aspect_box)
        aspect_row.addWidget(self.orientation)
        layout.addLayout(aspect_row)
        layout.addWidget(self.fourk)
        layout.addWidget(self.fourk_hint)
        row = QHBoxLayout()
        row.addWidget(self.format_box)
        row.addWidget(self.generate_button)
        layout.addLayout(row)
        layout.addWidget(self.status)
        layout.addWidget(self.addons_box)
        layout.addWidget(self.addons_notice)
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

    def set_enlarge_ready(self, ready: bool) -> None:
        self.fourk.setEnabled(ready)
        self.fourk_hint.setVisible(not ready)
        self.fourk.setToolTip("" if ready else ENLARGE_HINT)
        if not ready:
            self.fourk.setChecked(False)

    def set_input_size(self, width: int, height: int) -> None:
        """Image-to-image starts from the input's aspect until the picker is used."""
        self._input_size = (width, height)
        if self._aspect_touched:
            return
        self._show_aspect(closest_aspect(width, height))

    def current_aspect(self) -> str:
        selected = str(self.aspect_box.currentData())
        if selected == "21:9" and self.orientation.isChecked():
            return "9:21"
        return selected

    def _show_aspect(self, aspect: str) -> None:
        self._sync = True
        shown = "21:9" if aspect == "9:21" else aspect
        index = self.aspect_box.findData(shown)
        if index >= 0:
            self.aspect_box.setCurrentIndex(index)
        portrait = aspect == "9:21" or is_portrait(shown)
        self.orientation.setChecked(portrait)
        self.orientation.setText("Portrait" if portrait else "Landscape")
        self._sync = False

    def _aspect_changed(self) -> None:
        if self._sync:
            return
        self._aspect_touched = True
        selected = str(self.aspect_box.currentData())
        self._sync = True
        portrait = is_portrait(selected)
        self.orientation.setChecked(portrait)
        self.orientation.setText("Portrait" if portrait else "Landscape")
        self._sync = False

    def _orientation_changed(self, checked: bool) -> None:
        if self._sync:
            return
        self._aspect_touched = True
        self.orientation.setText("Portrait" if checked else "Landscape")
        selected = str(self.aspect_box.currentData())
        if selected in {"1:1", "21:9"}:
            return
        target = paired_aspect(selected)
        want_portrait = is_portrait(target)
        if want_portrait != checked:
            return
        self._sync = True
        index = self.aspect_box.findData(target)
        if index >= 0:
            self.aspect_box.setCurrentIndex(index)
        self._sync = False

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
