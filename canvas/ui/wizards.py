"""Secondary screens for the tools that do not fit on the shared main screen."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from canvas.app.i18n import text
from canvas.safety.image import RawImage


_SETUP_ROWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("canvas.source", ("words", "picture")),
    ("canvas.variation", ("quick", "varied")),
    ("canvas.change", ("restyle", "inpaint", "outpaint", "pose", "depth", "edge")),
    ("canvas.format", ("avif", "jxl", "webp")),
)


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
        self._on_choose = on_choose
        self._groups: list[QButtonGroup] = []
        layout = QVBoxLayout(self)
        for key, options in _SETUP_ROWS:
            label = QLabel(text(key))
            label.setObjectName(f"setup-label-{key.rsplit('.', 1)[-1]}")
            layout.addWidget(label)
            row = QHBoxLayout()
            group = QButtonGroup(self)
            group.setExclusive(True)
            self._groups.append(group)
            current = values.get(key, "")
            for option in options:
                button = QPushButton(text(f"{key}.{option}"))
                button.setObjectName(f"setup-{key.rsplit('.', 1)[-1]}-{option}")
                button.setCheckable(True)
                button.setChecked(option == current)
                button.clicked.connect(lambda _checked=False, key=key, option=option: self._pick(key, option))
                group.addButton(button)
                row.addWidget(button)
            row.addStretch(1)
            layout.addLayout(row)
        done = QPushButton(text("canvas.setup.done"))
        done.setObjectName("setupDone")
        done.clicked.connect(self.accept)
        layout.addWidget(done)

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
        self._on_use = on_use
        self._path: Path | None = None
        self._preview = QLabel(text("canvas.picture.empty"))
        self._preview.setObjectName("picturePreview")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(360, 240)
        open_button = QPushButton(text("canvas.picture.open"))
        open_button.setObjectName("pictureOpen")
        open_button.clicked.connect(self._open)
        use_button = QPushButton(text("canvas.picture.use"))
        use_button.setObjectName("pictureUse")
        use_button.clicked.connect(self._use)
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
        self._on_use = on_use
        self._board: _MaskBoard | None = None
        layout = QVBoxLayout(self)
        if picture is None:
            note = QLabel(text("canvas.mask.need"))
            note.setObjectName("maskNeed")
            note.setWordWrap(True)
            layout.addWidget(note)
            return
        self._board = _MaskBoard(picture)
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
        row.addWidget(brush)
        row.addWidget(eraser)
        row.addWidget(clear)
        row.addWidget(use_button)
        layout.addWidget(self._board, 1)
        layout.addLayout(row)

    def _use(self) -> None:
        if self._board is None:
            return
        self._on_use(self._board.mask_image())
        self.accept()


class AddonsWizard(QDialog):
    """The third-party notice, the folder, and the strength. They share the expert values."""

    def __init__(
        self,
        parent: QWidget | None,
        path: str,
        strength: float,
        on_save: Callable[[str, float], None],
    ) -> None:
        """Build the add-ons screen.

        Args:
            parent: The shell window.
            path: The current add-ons folder.
            strength: The current strength.
            on_save: Writes the folder and the strength into the shared state.
        """
        super().__init__(parent)
        self.setObjectName("addonsWizard")
        self.setWindowTitle(text("canvas.addons.title"))
        self._on_save = on_save
        notice = QLabel(text("notice.third_party"))
        notice.setObjectName("thirdPartyNotice")
        notice.setWordWrap(True)
        self._path = QLineEdit(path)
        self._path.setObjectName("addonsPath")
        self._strength = QDoubleSpinBox()
        self._strength.setObjectName("addonsStrength")
        self._strength.setRange(0.0, 2.0)
        self._strength.setSingleStep(0.1)
        self._strength.setValue(float(strength))
        save = QPushButton(text("canvas.addons.save"))
        save.setObjectName("addonsSave")
        save.clicked.connect(self._save)
        layout = QVBoxLayout(self)
        layout.addWidget(notice)
        layout.addWidget(QLabel(text("canvas.addons.path")))
        layout.addWidget(self._path)
        layout.addWidget(QLabel(text("canvas.addons.strength")))
        layout.addWidget(self._strength)
        layout.addWidget(save)

    def _save(self) -> None:
        self._on_save(self._path.text(), float(self._strength.value()))
        self.accept()


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
        self._on_check = on_check
        self._on_update = on_update
        self._deep = QCheckBox(text("canvas.check.deep"))
        self._deep.setObjectName("checkDeep")
        run = QPushButton(text("canvas.check.run"))
        run.setObjectName("checkRun")
        run.clicked.connect(self._run)
        update = QPushButton(text("canvas.check.update"))
        update.setObjectName("checkUpdate")
        update.clicked.connect(self._update)
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
        painter.setBrush(QColor(255, 255, 255, 255))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(x - 6, y - 6, 12, 12)
        painter.end()
        self.update()
