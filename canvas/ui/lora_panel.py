"""LoRA import, the LoRA library, and the pick and strength control.

Import takes a Model Forge ``.voxlora`` package or a ``.safetensors`` with its
``voxprint-model.json``. Every refusal is shown with its catalog line.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QEventLoop, QSize, Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from voxprint_common.ui.theme import CONTROL_BORDER

from canvas.app.i18n import text
from canvas.models.lora_library import (
    THIRD_PARTY_KIND,
    LibraryEntry,
    Manifest,
    import_lora,
    list_library,
    remove_entry,
    set_trigger,
)
from canvas.models.user_lora import LoraRejected, UserLora, discover
from canvas.ui.theme import BUTTON_RADIUS

THUMB = 72
PICK_ICON = 28
SLIDER_STEPS = 100


def import_with_dialog(parent: QWidget, gate: object, path: Path | None = None, **options) -> LibraryEntry | str | None:
    """Ask for a file, import it, and return the entry, a refusal code, or None when cancelled.

    Args:
        parent: The window that owns the dialogs.
        gate: The app's check stack.
        path: The file to import. A file dialog asks when omitted.
        options: Passed to :func:`canvas.models.lora_library.import_lora`.
    """
    if path is None:
        chosen, _filter = QFileDialog.getOpenFileName(
            parent, text("canvas.lora.import"), "", text("canvas.lora.import_filter"),
        )
        if not chosen:
            return None
        path = Path(chosen)
    progress = options.pop("progress", None)
    confirm = options.pop("confirm", None)

    def ask(manifest: Manifest) -> bool:
        return ThirdPartyDialog(parent, manifest.name).exec() == QDialog.DialogCode.Accepted

    def work(confirm_here: Callable[[Manifest], bool], report: Callable[[str], None]) -> LibraryEntry | None:
        return import_lora(Path(path), gate, confirm=confirm or confirm_here, progress=report, **options)

    try:
        return run_in_background(parent, work, ask, progress)
    except LoraRejected as exc:
        return exc.code


def run_in_background(
    parent: QWidget | None,
    work: Callable[[Callable[[Manifest], bool], Callable[[str], None]], LibraryEntry | None],
    ask: Callable[[Manifest], bool],
    progress: Callable[[str], None] | None = None,
) -> LibraryEntry | None:
    """Run an import on a worker thread while the window keeps handling events.

    A busy indicator is shown. A question for the user is asked on the UI thread.
    """
    state: dict[str, object] = {"done": False, "result": None, "error": None, "ask": None}
    lines: list[str] = []
    answered = threading.Event()
    reply = {"ok": False}

    def confirm_from_worker(manifest: Manifest) -> bool:
        state["ask"] = manifest
        answered.wait()
        return reply["ok"]

    def run() -> None:
        try:
            state["result"] = work(confirm_from_worker, lines.append)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the UI thread
            state["error"] = exc
        finally:
            state["done"] = True

    busy = QProgressDialog(text("canvas.lora.checking"), "", 0, 0, parent)
    busy.setObjectName("loraBusy")
    busy.setWindowTitle(text("canvas.lora.import"))
    busy.setCancelButton(None)
    busy.setWindowModality(Qt.WindowModality.WindowModal)
    busy.setMinimumDuration(200)
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(30)

    def tick() -> None:
        while lines:
            line = lines.pop(0)
            busy.setLabelText(line)
            if progress is not None:
                progress(line)
        manifest = state["ask"]
        if manifest is not None:
            state["ask"] = None
            busy.hide()
            reply["ok"] = bool(ask(manifest))  # type: ignore[arg-type]
            answered.set()
        if state["done"]:
            loop.quit()

    timer.timeout.connect(tick)
    worker = threading.Thread(target=run, name="canvas-lora-import", daemon=True)
    worker.start()
    timer.start()
    loop.exec()
    timer.stop()
    tick()
    busy.close()
    busy.deleteLater()
    worker.join()
    error = state["error"]
    if isinstance(error, BaseException):
        raise error
    return state["result"]  # type: ignore[return-value]


class ThirdPartyDialog(QDialog):
    """Asked before a LoRA made outside Model Forge is added to the library."""

    def __init__(self, parent: QWidget | None, name: str) -> None:
        super().__init__(parent)
        from canvas.ui.shell_chrome import shell_style

        self.setObjectName("loraThirdParty")
        self.setWindowTitle(text("canvas.lora.import"))
        self.setStyleSheet(shell_style())
        title = QLabel(text("canvas.lora.third_party.title", name=name))
        title.setObjectName("loraThirdPartyTitle")
        title.setWordWrap(True)
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        body = QLabel(text("canvas.lora.third_party.body"))
        body.setObjectName("loraThirdPartyBody")
        body.setWordWrap(True)
        cancel = QPushButton(text("canvas.lora.cancel"))
        cancel.setObjectName("loraThirdPartyCancel")
        cancel.clicked.connect(self.reject)
        go = QPushButton(text("canvas.lora.continue"))
        go.setObjectName("loraThirdPartyContinue")
        go.setDefault(True)
        go.clicked.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(body)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(go)
        layout.addLayout(buttons)
        self.setMinimumWidth(520)


def _kind_text(kind: str) -> str:
    return text(f"canvas.lora.kind.{kind}")


def _thumb(path: Path, size: int) -> QPixmap:
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        return pixmap
    return pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


def _preview_strip(previews: tuple[Path, ...], size: int = THUMB) -> QWidget:
    strip = QWidget()
    strip.setObjectName("loraPreviews")
    row = QHBoxLayout(strip)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    if not previews:
        empty = QLabel(text("canvas.lora.no_preview"))
        empty.setObjectName("loraPreviewPlaceholder")
        empty.setFixedSize(size, size)
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty.setWordWrap(True)
        empty.setStyleSheet(f"border: 1px dashed {CONTROL_BORDER}; border-radius: {BUTTON_RADIUS}px; font-size: 11px;")
        row.addWidget(empty)
    for index, path in enumerate(previews):
        label = QLabel()
        label.setObjectName(f"loraPreview-{index}")
        label.setFixedSize(size, size)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setPixmap(_thumb(path, size))
        row.addWidget(label)
    row.addStretch(1)
    return strip


def _details(entry: LibraryEntry) -> QLabel:
    lines = [text("canvas.lora.kind_line", kind=_kind_text(entry.manifest.kind))]
    if entry.manifest.trigger:
        lines.append(text("canvas.lora.trigger_line", trigger=entry.manifest.trigger))
    label = QLabel("\n".join(lines))
    label.setObjectName("loraDetails")
    label.setWordWrap(True)
    return label


class ImportedDialog(QDialog):
    """Shown after an import: the name, the trigger and the previews."""

    def __init__(self, parent: QWidget | None, entry: LibraryEntry) -> None:
        super().__init__(parent)
        from canvas.ui.shell_chrome import shell_style

        self.setObjectName("loraImported")
        self.setWindowTitle(text("canvas.lora.import"))
        self.setStyleSheet(shell_style())
        title = QLabel(text("canvas.lora.imported", name=entry.manifest.name))
        title.setObjectName("loraImportedTitle")
        title.setWordWrap(True)
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        done = QPushButton(text("canvas.lora.done"))
        done.setObjectName("loraImportedDone")
        done.clicked.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(_details(entry))
        layout.addWidget(_preview_strip(entry.previews, 120))
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(done)
        layout.addLayout(buttons)
        self.setMinimumWidth(560)


class LibraryDialog(QDialog):
    """The LoRA library: import, previews and remove."""

    def __init__(self, parent: QWidget | None, gate: object, on_change: Callable[[], None] | None = None,
                 root: Path | None = None, **options) -> None:
        super().__init__(parent)
        from canvas.ui.shell_chrome import shell_style

        self.setObjectName("loraLibrary")
        self.setWindowTitle(text("canvas.lora.library"))
        self.setStyleSheet(shell_style())
        self._gate = gate
        self._root = root
        self._options = options
        self._on_change = on_change or (lambda: None)
        self._import = QPushButton(text("canvas.lora.import"))
        self._import.setObjectName("loraImport")
        self._import.clicked.connect(lambda: self.run_import())
        self._status = QLabel("")
        self._status.setObjectName("loraStatus")
        self._status.setWordWrap(True)
        self._host = QWidget()
        self._rows = QVBoxLayout(self._host)
        self._rows.setContentsMargins(0, 0, 0, 0)
        self._rows.setSpacing(10)
        scroll = QScrollArea()
        scroll.setObjectName("loraLibraryList")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self._host)
        top = QHBoxLayout()
        top.addWidget(self._import)
        top.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self._status)
        layout.addWidget(scroll, 1)
        self.setMinimumSize(640, 480)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the list from the library folder."""
        while self._rows.count():
            item = self._rows.takeAt(0)
            widget = None if item is None else item.widget()
            if widget is not None:
                widget.deleteLater()
        entries = list_library(self._root)
        if not entries:
            empty = QLabel(text("canvas.lora.library_empty"))
            empty.setObjectName("loraLibraryEmpty")
            empty.setWordWrap(True)
            self._rows.addWidget(empty)
        for entry in entries:
            self._rows.addWidget(self._row(entry))
        self._rows.addStretch(1)

    def run_import(self, path: Path | None = None) -> LibraryEntry | str | None:
        """Import one file and show the result here."""
        self._status.setText(text("canvas.lora.checking"))
        result = import_with_dialog(self, self._gate, path, root=self._root,
                                    progress=self._status.setText, **self._options)
        if result is None:
            self._status.setText("")
        elif isinstance(result, str):
            self._status.setText(text(result))
        else:
            self._status.setText(text("canvas.lora.imported", name=result.manifest.name))
            self.refresh()
            self._on_change()
        return result

    def _row(self, entry: LibraryEntry) -> QWidget:
        card = QFrame()
        card.setObjectName(f"loraEntry-{entry.id}")
        card.setStyleSheet(f"QFrame#loraEntry-{entry.id} {{ border: 1px solid {CONTROL_BORDER}; border-radius: {BUTTON_RADIUS}px; }}")
        column = QVBoxLayout(card)
        head = QHBoxLayout()
        name = QLabel(entry.manifest.name)
        name.setObjectName("loraName")
        name.setStyleSheet("font-weight: 600; border: none;")
        remove = QPushButton(text("canvas.lora.remove"))
        remove.setObjectName("loraRemove")
        remove.clicked.connect(lambda _checked=False, entry_id=entry.id: self._remove(entry_id))
        head.addWidget(name)
        head.addStretch(1)
        head.addWidget(remove)
        column.addLayout(head)
        details = _details(entry)
        details.setStyleSheet("border: none;")
        column.addWidget(details)
        if entry.manifest.kind == THIRD_PARTY_KIND:
            edit_row = QHBoxLayout()
            caption = QLabel(text("canvas.lora.trigger_edit"))
            caption.setStyleSheet("border: none;")
            edit = QLineEdit(entry.manifest.trigger)
            edit.setObjectName("loraTriggerEdit")
            edit.setPlaceholderText(text("canvas.lora.trigger_placeholder"))
            edit.editingFinished.connect(
                lambda entry_id=entry.id, field=edit: self._set_trigger(entry_id, field.text())
            )
            edit_row.addWidget(caption)
            edit_row.addWidget(edit, 1)
            column.addLayout(edit_row)
        column.addWidget(_preview_strip(entry.previews))
        return card

    def _set_trigger(self, entry_id: str, trigger: str) -> None:
        set_trigger(entry_id, trigger, self._root)
        self._on_change()

    def _remove(self, entry_id: str) -> None:
        remove_entry(entry_id, self._root)
        self.refresh()
        self._on_change()


def library_loras(root: Path | None = None) -> list[UserLora]:
    """Imported LoRAs, by name."""
    items: list[UserLora] = []
    for entry in list_library(root):
        try:
            items.append(entry.as_lora())
        except LoraRejected:
            continue
    return items


def available_loras(root: Path | None = None) -> list[UserLora]:
    """Imported LoRAs first, then the listed add-on folders."""
    return [*library_loras(root), *discover().available]


class LoraPicker(QWidget):
    """A LoRA pick with thumbnails and a strength slider that follows the LoRA's range."""

    def __init__(self, on_change: Callable[[str, float], None], root: Path | None = None) -> None:
        super().__init__()
        self.setObjectName("loraPicker")
        self._on_change = on_change
        self._root = root
        self._items: list[UserLora | None] = []
        self._loading = False
        label = QLabel(text("canvas.lora.pick"))
        label.setObjectName("loraPickLabel")
        self.combo = QComboBox()
        self.combo.setObjectName("loraPick")
        self.combo.setIconSize(QSize(PICK_ICON, PICK_ICON))
        self.combo.setMinimumHeight(36)
        self.combo.currentIndexChanged.connect(self._picked)
        strength = QLabel(text("canvas.lora.strength"))
        strength.setObjectName("loraStrengthLabel")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setObjectName("loraStrength")
        self.slider.valueChanged.connect(self._moved)
        self.value = QLabel("")
        self.value.setObjectName("loraStrengthValue")
        self.value.setMinimumWidth(36)
        top = QVBoxLayout(self)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        top.addWidget(label)
        top.addWidget(self.combo)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(strength)
        row.addWidget(self.slider, 1)
        row.addWidget(self.value)
        self._strength_row = QWidget()
        self._strength_row.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._strength_row)

    def reload(self, selected: str, strength: float) -> None:
        """Fill the pick and set the slider from the shared state."""
        self._loading = True
        self.combo.clear()
        self._items = [None]
        self.combo.addItem(text("canvas.lora.none"))
        chosen = 0
        for lora in available_loras(self._root):
            icon = QIcon(_thumb(lora.previews[0], PICK_ICON * 2)) if lora.previews else QIcon()
            self.combo.addItem(icon, lora.name or lora.trigger or lora.id)
            self._items.append(lora)
            if selected and (selected == lora.id or selected == str(lora.folder)):
                chosen = len(self._items) - 1
        self.combo.setCurrentIndex(chosen)
        self._apply(self._items[chosen], strength)
        self._loading = False
        self.setVisible(len(self._items) > 1)

    def current(self) -> UserLora | None:
        index = self.combo.currentIndex()
        return self._items[index] if 0 <= index < len(self._items) else None

    def strength(self) -> float:
        return self.slider.value() / SLIDER_STEPS

    def _apply(self, lora: UserLora | None, strength: float | None) -> None:
        self._strength_row.setVisible(lora is not None)
        if lora is None:
            return
        low, high = lora.strength_range
        low, high = max(0.0, low), min(2.0, high)
        self.slider.setRange(round(low * SLIDER_STEPS), round(high * SLIDER_STEPS))
        value = lora.default_strength if strength is None else strength
        self.slider.setValue(round(min(max(float(value), low), high) * SLIDER_STEPS))
        self.value.setText(f"{self.strength():.2f}")

    def _picked(self, _index: int) -> None:
        if self._loading:
            return
        lora = self.current()
        self._apply(lora, None)
        self._emit()

    def _moved(self, _value: int) -> None:
        self.value.setText(f"{self.strength():.2f}")
        if not self._loading:
            self._emit()

    def _emit(self) -> None:
        lora = self.current()
        if lora is None:
            self._on_change("", self.strength())
            return
        key = lora.id if lora.previews or lora.name else str(lora.folder)
        self._on_change(key, self.strength())
