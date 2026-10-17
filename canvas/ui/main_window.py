"""The two-mode window: text to image, and image to image."""
from __future__ import annotations

from typing import Callable, NamedTuple, assert_never

from PySide6.QtCore import Qt, QPointF, QRectF, QSize, QUrl
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
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

from canvas.app.i18n import text
from canvas.app.info import ENLARGE_NAME, ENLARGE_RELEASES
from canvas.models.features import instruction_edit_available
from canvas.ui.about import AboutDialog
from canvas.ui.icons import load_approved
from canvas.core.formats import DEFAULT_FORMAT, OutputFormat
from canvas.models.modes import Mode, Tier, mode_group, visible_modes
from canvas.models.sizes import ASPECT_GROUPS, closest_aspect
from canvas.ui.theme import TEXT, TEXT_DISABLED, build_style

class EnhanceReply(NamedTuple):
    """The prompt to show, and a note when the rewrite was not placed in the field."""

    text: str
    note: str


def _mode_label(mode: Mode) -> str:
    """The catalog label for one mode."""
    if mode is Mode.TEXT:
        key = "mode.text"
    elif mode is Mode.TEXT_CREATIVE:
        key = "mode.creative"
    elif mode is Mode.RESTYLE:
        key = "mode.restyle"
    elif mode is Mode.INPAINT:
        key = "mode.inpaint"
    elif mode is Mode.OUTPAINT:
        key = "mode.outpaint"
    elif mode is Mode.POSE:
        key = "mode.pose"
    elif mode is Mode.DEPTH:
        key = "mode.depth"
    elif mode is Mode.EDGE:
        key = "mode.edge"
    elif mode is Mode.INSTRUCTION:
        key = "mode.instruction"
    else:
        assert_never(mode)
    return text(key)


def _aspect_label(name: str) -> str:
    """The catalog label for one aspect group."""
    if name == "Landscape":
        return text("aspect.landscape")
    if name == "Square":
        return text("aspect.square")
    if name == "Portrait":
        return text("aspect.portrait")
    raise KeyError(name)


class MainWindow(QWidget):
    """Prompt, mode and format. Generation is delegated to ``on_generate``."""

    def __init__(self, on_generate: Callable[..., str], tier: Tier, glass: bool = False,
                 enlarge_ready: bool = False, on_enhance: Callable[[str], EnhanceReply] | None = None) -> None:
        super().__init__()
        self.on_generate = on_generate
        self.on_enhance = on_enhance
        self.tier = tier
        self._original: str | None = None
        self._busy = False
        self._aspect_touched = False
        self._input_size: tuple[int, int] | None = None
        self.setObjectName("root")
        self.setWindowTitle(text("app.name"))
        self.setStyleSheet(build_style(glass))
        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText(text("prompt.placeholder"))
        self.format_box = QComboBox()
        for fmt, key in (
            (OutputFormat.AVIF, "format.avif"),
            (OutputFormat.JXL, "format.jxl"),
            (OutputFormat.WEBP, "format.webp"),
        ):
            self.format_box.addItem(text(key), fmt)
        index = self.format_box.findData(DEFAULT_FORMAT)
        if index >= 0:
            self.format_box.setCurrentIndex(index)
        self.image_mode = QComboBox()
        for mode in visible_modes(tier):
            if mode_group(mode) == "image":
                self.image_mode.addItem(_mode_label(mode), mode)
        self._instruction_ready = instruction_edit_available()
        self._set_instruction_enabled(self._instruction_ready)
        self.instruction_note = QLabel(text("edit.unavailable"))
        self.instruction_note.setObjectName("instruction-note")
        self.instruction_note.setWordWrap(True)
        self.instruction_note.setVisible(not self._instruction_ready)
        self.creative = QComboBox()
        self.creative.addItem(_mode_label(Mode.TEXT), Mode.TEXT)
        if Mode.TEXT_CREATIVE in visible_modes(tier):
            self.creative.addItem(_mode_label(Mode.TEXT_CREATIVE), Mode.TEXT_CREATIVE)
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.aspects = QButtonGroup(self)
        self.aspects.setExclusive(True)
        self._aspect_buttons: dict[str, QPushButton] = {}
        self.aspect_row = self._aspect_picker()
        self.aspects.idClicked.connect(self._aspect_picked)
        self.fourk = QCheckBox(text("checkbox.fourk"))
        self.fourk.setObjectName("fourk")
        self.fourk_hint = QLabel(text("enlarge.hint", name=ENLARGE_NAME))
        self.fourk_hint.setObjectName("fourk-hint")
        self.fourk_hint.setWordWrap(True)
        self.install_enlarge = QPushButton(text("button.install_enlarge"))
        self.install_enlarge.setObjectName("install-enlarge")
        self.install_enlarge.clicked.connect(self._install_enlarge)
        self.set_enlarge_ready(enlarge_ready)
        self.generate_button = QPushButton(text("button.generate"))
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self._generate)
        self.enhance_button = QPushButton(text("button.enhance"))
        self.enhance_button.setObjectName("enhance")
        self.enhance_button.setToolTip(text("tooltip.enhance"))
        self.enhance_button.clicked.connect(self._enhance)
        self.enhance_button.setEnabled(on_enhance is not None)
        self.revert_button = QPushButton()
        self.revert_button.setObjectName("revert")
        self.revert_button.setIcon(_undo_icon())
        self.revert_button.setIconSize(QSize(18, 18))
        self.revert_button.setFixedSize(36, 36)
        self.revert_button.setToolTip(text("tooltip.revert"))
        self.revert_button.setAccessibleName(text("tooltip.revert"))
        self.revert_button.clicked.connect(self._revert)
        self.revert_button.setEnabled(False)
        self.action_icons = load_approved()
        self.about_button = QPushButton(text("button.about"))
        self.about_button.setObjectName("about")
        self.about_button.clicked.connect(self._about)
        tabs = QTabWidget()
        tabs.addTab(self._text_tab(), text("tab.text"))
        tabs.addTab(self._image_tab(), text("tab.image"))
        self.addons_box = QCheckBox(text("checkbox.addons"))
        self.addons_box.setChecked(False)
        self.addons_notice = QLabel(text("notice.third_party"))
        self.addons_notice.setObjectName("third-party-notice")
        self.addons_notice.setWordWrap(True)
        self.addons_notice.setVisible(False)
        self.addons_box.toggled.connect(self.addons_notice.setVisible)
        footer = QLabel(text("notice.responsibility"))
        footer.setObjectName("footer")
        footer.setWordWrap(True)
        layout = QVBoxLayout(self)
        title = QLabel(text("app.name"))
        title.setObjectName("title")
        title_row = QHBoxLayout()
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.about_button)
        layout.addLayout(title_row)
        layout.addWidget(tabs)
        prompt_row = QWidget()
        prompts = QHBoxLayout(prompt_row)
        prompts.setContentsMargins(0, 0, 0, 0)
        prompts.addWidget(self.prompt, 1)
        actions = QVBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addWidget(self.enhance_button)
        actions.addWidget(self.revert_button)
        actions.addStretch(1)
        prompts.addLayout(actions)
        layout.addWidget(prompt_row)
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
        if self.creative.count() > 1:
            layout.addWidget(self.creative)
        else:
            self.creative.setParent(page)
            self.creative.hide()
        return page

    def _image_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel(text("label.operation")))
        layout.addWidget(self.image_mode)
        layout.addWidget(self.instruction_note)
        return page

    def _set_instruction_enabled(self, ready: bool) -> None:
        """Leave instruction edit in the list. Grey it while the feature is unavailable."""
        index = self.image_mode.findData(Mode.INSTRUCTION)
        if index < 0:
            return
        item = self.image_mode.model().item(index)  # type: ignore[attr-defined]
        if item is None:
            return
        item.setEnabled(ready)
        if ready:
            item.setForeground(QBrush())
        else:
            item.setForeground(QBrush(QColor(TEXT_DISABLED)))

    def set_enlarge_ready(self, ready: bool) -> None:
        """Enable 4K when Enlarge can run, and show the install hint otherwise."""
        self.fourk.setEnabled(ready)
        self.fourk_hint.setVisible(not ready)
        self.fourk.setToolTip("" if ready else text("enlarge.hint", name=ENLARGE_NAME))
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
            label = QLabel(_aspect_label(name))
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
            text("dialog.download"),
            notice,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.enhance_button.setText(text("button.enhancing") if busy else text("button.enhance"))
        self.enhance_button.setEnabled((not busy) and self.on_enhance is not None)
        self.prompt.setReadOnly(busy)
        self.generate_button.setEnabled(not busy)
        self.revert_button.setEnabled((not busy) and self._original is not None)
        self.setCursor(Qt.CursorShape.WaitCursor if busy else Qt.CursorShape.ArrowCursor)

    def _replace_prompt(self, text: str) -> None:
        """Replace the field in one undo step, so Ctrl+Z restores the previous text."""
        cursor = self.prompt.textCursor()
        cursor.beginEditBlock()
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.insertText(text)
        cursor.endEditBlock()
        self.prompt.setTextCursor(cursor)

    def _enhance(self) -> None:
        if self.on_enhance is None or self._busy:
            return
        source = self.prompt.toPlainText()
        self._set_busy(True)
        self.status.setText(text("button.enhancing"))
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        reply: EnhanceReply | None = None
        try:
            reply = self.on_enhance(source)
        except Exception as exc:
            self.status.setText(str(exc))
        finally:
            self._set_busy(False)
        if reply is None:
            return
        if reply.text != source:
            self._original = source
            self._replace_prompt(reply.text)
            self.revert_button.setEnabled(True)
        self.status.setText(reply.note)

    def _revert(self) -> None:
        if self._original is None or self._busy:
            return
        original = self._original
        self._original = None
        self.revert_button.setEnabled(False)
        self._replace_prompt(original)
        self.status.setText("")

    def _generate(self) -> None:
        if self._busy:
            return
        tabs = self.findChild(QTabWidget)
        index = 0 if tabs is None else tabs.currentIndex()
        mode = self.current_mode(index)
        if mode is Mode.INSTRUCTION and not self._instruction_ready:
            self.status.setText(self.instruction_note.text())
            return
        fmt = self.format_box.currentData()
        prompt = self.prompt.toPlainText()
        try:
            if self._original is None:
                message = self.on_generate(prompt, mode, fmt)
            else:
                message = self.on_generate(prompt, mode, fmt, original=self._original)
            self.status.setText(message)
        except Exception as exc:
            self.status.setText(str(exc))


def _undo_icon() -> QIcon:
    """A counterclockwise arrow, the usual undo mark."""
    pixmap = QPixmap(22, 22)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(TEXT), 1.7)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.moveTo(16, 8)
    path.arcTo(QRectF(6, 5, 11, 11), 20, 250)
    painter.drawPath(path)
    tip = path.currentPosition()
    painter.setBrush(QColor(TEXT))
    head = QPainterPath()
    head.moveTo(tip)
    head.lineTo(tip + QPointF(-1.2, 4.2))
    head.lineTo(tip + QPointF(3.6, 2.2))
    head.closeSubpath()
    painter.drawPath(head)
    painter.end()
    return QIcon(pixmap)


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
