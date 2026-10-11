"""The two-mode window: text to image, and image to image."""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, QRectF, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from canvas.app.info import APP_NAME, ENLARGE_HINT, ENLARGE_RELEASES, RESPONSIBILITY, THIRD_PARTY_NOTICE
from canvas.ui.about import AboutDialog
from canvas.ui.icons import load_approved
from canvas.core.formats import DEFAULT_FORMAT, OutputFormat
from canvas.models.modes import Mode, Tier, mode_group, visible_modes
from canvas.models.sizes import ASPECT_GROUPS, closest_aspect
from canvas.ui.theme import TEXT, build_style

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
        self.aspects = QButtonGroup(self)
        self.aspects.setExclusive(True)
        self._aspect_buttons: dict[str, QPushButton] = {}
        self.aspect_row = self._aspect_picker()
        self.aspects.idClicked.connect(self._aspect_picked)
        self.fourk = QCheckBox("Output size: 4K")
        self.fourk.setObjectName("fourk")
        self.fourk_hint = QLabel(ENLARGE_HINT)
        self.fourk_hint.setObjectName("fourk-hint")
        self.fourk_hint.setWordWrap(True)
        self.install_enlarge = QPushButton("Install Enlarge")
        self.install_enlarge.setObjectName("install-enlarge")
        self.install_enlarge.clicked.connect(self._install_enlarge)
        self.set_enlarge_ready(enlarge_ready)
        self.generate_button = QPushButton("Generate")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self._generate)
        self.action_icons = load_approved()
        self.about_button = QPushButton("About")
        self.about_button.setObjectName("about")
        self.about_button.clicked.connect(self._about)
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
        title_row = QHBoxLayout()
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.about_button)
        layout.addLayout(title_row)
        layout.addWidget(tabs)
        layout.addWidget(self.prompt)
        layout.addWidget(self.aspect_row)
        fourk_row = QHBoxLayout()
        fourk_row.addWidget(self.fourk)
        fourk_row.addWidget(self.install_enlarge)
        fourk_row.addStretch(1)
        layout.addLayout(fourk_row)
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
        """Enable 4K when Enlarge can run, and show the install hint otherwise."""
        self.fourk.setEnabled(ready)
        self.fourk_hint.setVisible(not ready)
        self.fourk.setToolTip("" if ready else ENLARGE_HINT)
        self.install_enlarge.setVisible(not ready)
        if not ready:
            self.fourk.setChecked(False)

    def _install_enlarge(self) -> None:
        QDesktopServices.openUrl(QUrl(ENLARGE_RELEASES))

    def set_input_size(self, width: int, height: int) -> None:
        """Image-to-image starts from the input's aspect until the picker is used."""
        self._input_size = (width, height)
        if self._aspect_touched:
            return
        self._select_aspect(closest_aspect(width, height))

    def current_aspect(self) -> str:
        """The selected aspect ratio, or ``1:1`` when none is selected."""
        button = self.aspects.checkedButton()
        if button is None:
            return "1:1"
        return str(button.property("ratio"))

    def _aspect_picker(self) -> QWidget:
        row = QWidget()
        row.setObjectName("aspect-picker")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for name, aspects in ASPECT_GROUPS:
            group = QWidget()
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(4)
            label = QLabel(name)
            label.setObjectName("aspect-group")
            buttons = QHBoxLayout()
            buttons.setSpacing(4)
            for aspect in aspects:
                button = QPushButton(aspect)
                button.setObjectName("aspect")
                button.setProperty("ratio", aspect)
                button.setIcon(_shape_icon(aspect))
                button.setCheckable(True)
                self.aspects.addButton(button)
                self._aspect_buttons[aspect] = button
                buttons.addWidget(button)
            group_layout.addWidget(label)
            group_layout.addLayout(buttons)
            layout.addWidget(group)
        layout.addStretch(1)
        self._aspect_buttons["1:1"].setChecked(True)
        return row

    def _select_aspect(self, aspect: str) -> None:
        button = self._aspect_buttons[aspect]
        self.aspects.blockSignals(True)
        button.setChecked(True)
        self.aspects.blockSignals(False)

    def _aspect_picked(self, _button_id: int) -> None:
        self._aspect_touched = True

    def current_mode(self, tab_index: int) -> Mode:
        """The mode selected on the text tab or the image tab."""
        if tab_index == 0:
            return self.creative.currentData()
        return self.image_mode.currentData()

    def _about(self) -> None:
        AboutDialog(self).exec()

    def confirm_download(self, notice: str) -> bool:
        """Ask before an optional pack download. The notice has the size and the estimate."""
        answer = QMessageBox.question(
            self,
            "Download",
            notice,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _generate(self) -> None:
        tabs = self.findChild(QTabWidget)
        index = 0 if tabs is None else tabs.currentIndex()
        mode = self.current_mode(index)
        fmt = self.format_box.currentData()
        try:
            self.status.setText(self.on_generate(self.prompt.toPlainText(), mode, fmt))
        except Exception as exc:
            self.status.setText(str(exc))


def _shape_icon(aspect: str) -> QIcon:
    """A small outline in the ratio's own proportions."""
    width_part, _, height_part = aspect.partition(":")
    ratio_w = int(width_part)
    ratio_h = int(height_part)
    long_side = 16
    if ratio_w >= ratio_h:
        shape_w = long_side
        shape_h = max(4, round(long_side * ratio_h / ratio_w))
    else:
        shape_h = long_side
        shape_w = max(4, round(long_side * ratio_w / ratio_h))
    canvas = 22
    pixmap = QPixmap(canvas, canvas)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(TEXT), 1.5))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    left = (canvas - shape_w) / 2
    top = (canvas - shape_h) / 2
    painter.drawRoundedRect(QRectF(left, top, shape_w, shape_h), 1.5, 1.5)
    painter.end()
    return QIcon(pixmap)
