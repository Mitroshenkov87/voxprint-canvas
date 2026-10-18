"""Secondary screens for the tools that do not fit on the shared main screen."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QMouseEvent, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from canvas.app.i18n import text
from canvas.models.user_lora import LoraIndex, UserLora, clamp_strength
from canvas.safety.image import RawImage
from canvas.ui.shell_chrome import shell_style


_SETUP_ROWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("canvas.source", ("words", "picture")),
    ("canvas.variation", ("quick", "varied")),
    ("canvas.change", ("restyle", "inpaint", "outpaint", "pose", "depth", "edge")),
    ("canvas.format", ("avif", "jxl", "webp")),
)
_HINT_WIDTH = 336
_HINT_GAP = 12
_GROUP_HINT_WIDTH = 488


class SetupWizard(QDialog):
    """Start from, variation, change, and file type. They share the live settings."""

    def __init__(
        self,
        parent: QWidget | None,
        values: dict[str, str],
        on_choose: Callable[[str, str], None],
    ) -> None:
        """Build the picture-setup screen.

        Args:
            parent: The shell window.
            values: Current value of each setting id.
            on_choose: Writes one setting id and value into the shared state.
        """
        super().__init__(parent)
        self.setObjectName("setupWizard")
        self.setWindowTitle(text("canvas.setup.title"))
        self.setStyleSheet(shell_style())
        self.setFixedWidth(520)
        self._on_choose = on_choose
        self._groups: list[QButtonGroup] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        title = QLabel(text("canvas.setup.title"))
        title.setObjectName("setupTitle")
        layout.addWidget(title)
        for key, options in _SETUP_ROWS:
            layout.addWidget(self._group(key, options, values.get(key, "")))
        done = QPushButton(text("canvas.setup.done"))
        done.setObjectName("setupDone")
        _fit_button(done)
        done.clicked.connect(self.accept)
        layout.addWidget(done, 0, Qt.AlignmentFlag.AlignLeft)

    def _group(self, key: str, options: tuple[str, ...], current: str) -> QWidget:
        suffix = key.rsplit(".", 1)[-1]
        host = QWidget()
        host.setObjectName(f"setup-group-{suffix}")
        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        label = QLabel(text(key))
        label.setObjectName(f"setup-label-{suffix}")
        column.addWidget(label)
        group = QButtonGroup(self)
        group.setExclusive(True)
        self._groups.append(group)
        if key == "canvas.change":
            grid = QGridLayout()
            grid.setHorizontalSpacing(_HINT_GAP)
            grid.setVerticalSpacing(8)
            grid.setColumnMinimumWidth(1, _HINT_WIDTH)
            for index, option in enumerate(options):
                grid.addWidget(
                    self._option(group, key, suffix, option, current),
                    index,
                    0,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                )
                grid.addWidget(
                    self._hint(f"{key}.{option}.hint", f"setup-hint-{option}", _HINT_WIDTH),
                    index,
                    1,
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                )
            column.addLayout(grid)
            return host
        row = QHBoxLayout()
        row.setSpacing(8)
        for option in options:
            row.addWidget(self._option(group, key, suffix, option, current))
        row.addStretch(1)
        column.addLayout(row)
        column.addWidget(self._hint(f"{key}.help", f"setup-hint-{suffix}", _GROUP_HINT_WIDTH))
        return host

    def _option(
        self,
        group: QButtonGroup,
        key: str,
        suffix: str,
        option: str,
        current: str,
    ) -> QPushButton:
        button = QPushButton(text(f"{key}.{option}"))
        button.setObjectName(f"setup-{suffix}-{option}")
        button.setCheckable(True)
        button.setChecked(option == current)
        button.clicked.connect(lambda _checked=False, key=key, option=option: self._pick(key, option))
        group.addButton(button)
        _fit_button(button)
        return button

    def _hint(self, key: str, name: str, width: int) -> QLabel:
        hint = QLabel(text(key))
        hint.setObjectName(name)
        hint.setProperty("setupHint", True)
        hint.setWordWrap(True)
        hint.setFixedWidth(width)
        hint.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        return hint

    def _pick(self, key: str, option: str) -> None:
        self._on_choose(key, option)


class PictureWizard(QDialog):
    """Pick a picture for img2img, inpaint, or structure guidance."""

    def __init__(self, parent: QWidget | None, on_use: Callable[[Path], None]) -> None:
        """Build the picker.

        Args:
            parent: The shell window.
            on_use: Called with the chosen file when the person accepts it.
        """
        super().__init__(parent)
        self.setObjectName("pictureWizard")
        self.setWindowTitle(text("canvas.picture.title"))
        self.setStyleSheet(shell_style())
        self._on_use = on_use
        self._path: Path | None = None
        self._preview = QLabel(text("canvas.picture.empty"))
        self._preview.setObjectName("picturePreview")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(360, 240)
        open_button = QPushButton(text("canvas.picture.open"))
        open_button.setObjectName("pictureOpen")
        open_button.clicked.connect(self._open)
        _fit_button(open_button)
        use_button = QPushButton(text("canvas.picture.use"))
        use_button.setObjectName("pictureUse")
        use_button.clicked.connect(self._use)
        _fit_button(use_button)
        row = QHBoxLayout()
        row.addWidget(open_button)
        row.addWidget(use_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self._preview)
        layout.addLayout(row)

    def load_path(self, path: Path) -> None:
        """Show ``path`` in the preview without asking the file dialog.

        Args:
            path: An image file.
        """
        self._path = path
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._preview.setText(text("canvas.picture.empty"))
            return
        self._preview.setPixmap(pixmap.scaled(360, 240, Qt.AspectRatioMode.KeepAspectRatio))

    def _open(self) -> None:
        chosen, _filter = QFileDialog.getOpenFileName(self, "", "")
        if chosen:
            self.load_path(Path(chosen))

    def _use(self) -> None:
        if self._path is None:
            return
        self._on_use(self._path)
        self.accept()


class MaskWizard(QDialog):
    """Paint the area inpaint and extend should change."""

    def __init__(
        self,
        parent: QWidget | None,
        picture: RawImage | None,
        on_use: Callable[[RawImage], None],
    ) -> None:
        """Build the painter.

        Args:
            parent: The shell window.
            picture: The open picture, or None when the person has not chosen one.
            on_use: Called with a white-on-black mask.
        """
        super().__init__(parent)
        self.setObjectName("maskWizard")
        self.setWindowTitle(text("canvas.mask.title"))
        self.setStyleSheet(shell_style())
        self._on_use = on_use
        self._board: _MaskBoard | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        title = QLabel(text("canvas.mask.title"))
        title.setObjectName("maskTitle")
        layout.addWidget(title)
        if picture is None:
            note = QLabel(text("canvas.mask.need"))
            note.setObjectName("maskNeed")
            note.setWordWrap(True)
            layout.addWidget(note)
            return
        hint = QLabel(text("canvas.mask.hint"))
        hint.setObjectName("maskHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self._board = _MaskBoard(picture)
        size_label = QLabel(text("canvas.mask.size"))
        size_label.setObjectName("maskSizeLabel")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setObjectName("maskSize")
        slider.setRange(8, 64)
        slider.setValue(24)
        slider.valueChanged.connect(self._board.set_brush_size)
        self._board.set_brush_size(slider.value())
        size_row = QHBoxLayout()
        size_row.addWidget(size_label)
        size_row.addWidget(slider, 1)
        brush = QPushButton(text("canvas.mask.brush"))
        brush.setObjectName("maskBrush")
        brush.clicked.connect(lambda: self._board.set_eraser(False) if self._board is not None else None)
        eraser = QPushButton(text("canvas.mask.eraser"))
        eraser.setObjectName("maskEraser")
        eraser.clicked.connect(lambda: self._board.set_eraser(True) if self._board is not None else None)
        clear = QPushButton(text("canvas.mask.clear"))
        clear.setObjectName("maskClear")
        clear.clicked.connect(self._board.clear_mask)
        use_button = QPushButton(text("canvas.mask.use"))
        use_button.setObjectName("maskUse")
        use_button.clicked.connect(self._use)
        row = QHBoxLayout()
        for button in (brush, eraser, clear, use_button):
            _fit_button(button)
            row.addWidget(button)
        row.addStretch(1)
        layout.addWidget(self._board, 1)
        layout.addLayout(size_row)
        layout.addLayout(row)

    def _use(self) -> None:
        if self._board is None:
            return
        self._on_use(self._board.mask_image())
        self.accept()


class AddonsWizard(QDialog):
    """The discovered LoRAs and a strength. They share the expert values."""

    def __init__(
        self,
        parent: QWidget | None,
        index: LoraIndex,
        selected: str,
        strength: float,
        on_save: Callable[[str, float], None],
    ) -> None:
        """Build the add-ons screen.

        Args:
            parent: The shell window.
            index: Listed LoRAs and the folders that were skipped.
            selected: The current expert folder, or empty.
            strength: The current expert strength.
            on_save: Writes the folder and the strength into the shared state.
        """
        super().__init__(parent)
        self.setObjectName("addonsWizard")
        self.setWindowTitle(text("canvas.addons.title"))
        self.setStyleSheet(shell_style())
        self._on_save = on_save
        self._items: dict[QRadioButton, UserLora | None] = {}
        self._loading = True
        notice = QLabel(text("notice.third_party"))
        notice.setObjectName("thirdPartyNotice")
        notice.setWordWrap(True)
        host = QWidget()
        rows = QVBoxLayout(host)
        group = QButtonGroup(self)
        group.setExclusive(True)
        none = QRadioButton(text("canvas.addons.none"))
        none.setObjectName("addonsNone")
        none.setChecked(True)
        group.addButton(none)
        self._items[none] = None
        none.toggled.connect(lambda checked: self._on_toggle(checked, None))
        rows.addWidget(none)
        if not index.available:
            empty = QLabel(text("canvas.addons.empty"))
            empty.setObjectName("addonsEmpty")
            empty.setWordWrap(True)
            rows.addWidget(empty)
        chosen: UserLora | None = None
        for item in index.available:
            button = QRadioButton(_lora_label(item))
            button.setObjectName(f"loraOption-{item.id}")
            group.addButton(button)
            self._items[button] = item
            button.toggled.connect(lambda checked, lora=item: self._on_toggle(checked, lora))
            if _matches(item, selected):
                button.setChecked(True)
                chosen = item
            rows.addWidget(button)
        for skipped in index.skipped:
            label = QLabel(text("canvas.addons.skipped", name=skipped.name, reason=text(skipped.reason_key)))
            label.setObjectName("addonsSkipped")
            label.setWordWrap(True)
            rows.addWidget(label)
        scroll = QScrollArea()
        scroll.setObjectName("addonsList")
        scroll.setWidgetResizable(True)
        scroll.setWidget(host)
        self._strength = QDoubleSpinBox()
        self._strength.setObjectName("addonsStrength")
        self._strength.setDecimals(2)
        self._strength.setSingleStep(0.05)
        self._apply_strength(chosen, strength if chosen is not None else None)
        if chosen is None:
            self._strength.setValue(float(strength))
        self._loading = False
        save = QPushButton(text("canvas.addons.save"))
        save.setObjectName("addonsSave")
        save.clicked.connect(self._save)
        _fit_button(save)
        layout = QVBoxLayout(self)
        layout.addWidget(notice)
        layout.addWidget(scroll)
        layout.addWidget(QLabel(text("canvas.addons.strength")))
        layout.addWidget(self._strength)
        layout.addWidget(save)
        self.setMinimumWidth(420)

    def _on_toggle(self, checked: bool, lora: UserLora | None) -> None:
        if not checked or self._loading:
            return
        self._apply_strength(lora, None)

    def _apply_strength(self, lora: UserLora | None, saved: float | None) -> None:
        if lora is None:
            self._strength.setRange(0.0, 2.0)
            return
        low, high = lora.strength_range
        self._strength.setRange(low, high)
        value = lora.default_strength if saved is None else saved
        self._strength.setValue(min(max(float(value), low), high))

    def _save(self) -> None:
        selected = self._selected()
        strength = float(self._strength.value())
        if selected is None:
            self._on_save("", strength)
        else:
            self._on_save(str(selected.folder), clamp_strength(selected, strength))
        self.accept()

    def _selected(self) -> UserLora | None:
        for button, lora in self._items.items():
            if button.isChecked():
                return lora
        return None


class DiagnosticsWizard(QDialog):
    """Check this PC, including a full file check and the release check."""

    def __init__(
        self,
        parent: QWidget | None,
        on_check: Callable[[bool], str],
        on_update: Callable[[], str],
    ) -> None:
        """Build the check screen. Nothing runs until a button is pressed.

        Args:
            parent: The shell window.
            on_check: Runs the check. True re-reads every file.
            on_update: Looks for a newer release and returns one sentence.
        """
        super().__init__(parent)
        self.setObjectName("diagnosticsWizard")
        self.setWindowTitle(text("canvas.check.title"))
        self.setStyleSheet(shell_style())
        self._on_check = on_check
        self._on_update = on_update
        self._deep = QCheckBox(text("canvas.check.deep"))
        self._deep.setObjectName("checkDeep")
        run = QPushButton(text("canvas.check.run"))
        run.setObjectName("checkRun")
        run.clicked.connect(self._run)
        _fit_button(run)
        update = QPushButton(text("canvas.check.update"))
        update.setObjectName("checkUpdate")
        update.clicked.connect(self._update)
        _fit_button(update)
        self._report = QLabel("")
        self._report.setObjectName("checkReport")
        self._report.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self._deep)
        row = QHBoxLayout()
        row.addWidget(run)
        row.addWidget(update)
        layout.addLayout(row)
        layout.addWidget(self._report)

    def show_report(self, report: str) -> None:
        """Replace the result sentence.

        Args:
            report: Already translated text.
        """
        self._report.setText(report)

    def _run(self) -> None:
        self.show_report(self._on_check(self._deep.isChecked()))

    def _update(self) -> None:
        self.show_report(self._on_update())


class _MaskBoard(QWidget):
    """The picture with a painted overlay. White pixels are the mask."""

    def __init__(self, picture: RawImage) -> None:
        super().__init__()
        self.setObjectName("maskSurface")
        self.setMinimumSize(480, 320)
        self.setMouseTracking(True)
        data = picture.to_bytes()
        self._photo = QImage(data, picture.width, picture.height, picture.width * 3, QImage.Format.Format_RGB888).copy()
        self._mask = QImage(picture.width, picture.height, QImage.Format.Format_ARGB32)
        self._mask.fill(Qt.GlobalColor.transparent)
        self._eraser = False
        self._drawing = False
        self._diameter = 24

    def set_brush_size(self, diameter: int) -> None:
        """Set the brush diameter in screen pixels.

        Args:
            diameter: Width of the stroke on the board.
        """
        self._diameter = max(2, int(diameter))

    def set_eraser(self, enabled: bool) -> None:
        """Paint with the eraser when ``enabled`` is true.

        Args:
            enabled: True clears painted pixels.
        """
        self._eraser = enabled

    def clear_mask(self) -> None:
        """Remove every painted pixel."""
        self._mask.fill(Qt.GlobalColor.transparent)
        self.update()

    def mask_image(self) -> RawImage:
        """Return a white-on-black RGB mask the same size as the picture."""
        width = self._mask.width()
        height = self._mask.height()
        buffer = bytearray(width * height * 3)
        for y in range(height):
            for x in range(width):
                if self._mask.pixelColor(x, y).alpha() > 10:
                    index = (y * width + x) * 3
                    buffer[index:index + 3] = b"\xff\xff\xff"
        return RawImage(width, height, bytes(buffer))

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        target = self.rect()
        painter.drawImage(target, self._photo)
        painter.drawImage(target, self._mask)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._drawing = True
        self._stamp(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drawing:
            self._stamp(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        del event
        self._drawing = False

    def _stamp(self, event: QMouseEvent) -> None:
        if self.width() < 1 or self.height() < 1:
            return
        x = int(event.position().x() * self._mask.width() / self.width())
        y = int(event.position().y() * self._mask.height() / self.height())
        painter = QPainter(self._mask)
        if self._eraser:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        scale = self._mask.width() / self.width()
        radius = max(1, int(self._diameter * scale / 2))
        painter.setBrush(QColor(255, 255, 255, 255))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
        painter.end()
        self.update()


def _lora_label(lora: UserLora) -> str:
    name = lora.trigger or lora.id
    if lora.adult_content:
        return f"{name} — {text('canvas.addons.adult_label')}"
    return name


def _matches(lora: UserLora, selected: str) -> bool:
    if not selected:
        return False
    if selected == lora.id:
        return True
    try:
        return Path(selected).resolve() == lora.folder.resolve()
    except OSError:
        return False


def _fit_button(button: QPushButton) -> None:
    words = button.text()
    if not words:
        return
    font = QFont(button.font())
    if button.isCheckable():
        font.setWeight(QFont.Weight.DemiBold)
    advance = QFontMetrics(font).horizontalAdvance(words)
    button.setMinimumWidth(advance + 36)
    button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
