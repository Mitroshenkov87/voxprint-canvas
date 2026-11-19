"""The Settings screen: every persisted knob in one place.

The dialog edits the same stored settings as the simple panel and the Expert
panel, so a change here is a change there. The 18+ switch keeps its meaning
exactly: it is the stored ``canvas.adult_content`` value and only changes the
prompt policy.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from voxprint_common.i18n import tr

from canvas.app.i18n import text
from canvas.core.seeds import FIELD_MAX
from canvas.infra.paths import projects_dir
from canvas.ui.shell_chrome import shell_style

if TYPE_CHECKING:
    from canvas.ui.shell_screen import ShellWindow

_FORMATS = (("avif", "canvas.format.avif"), ("jxl", "canvas.format.jxl"), ("webp", "canvas.format.webp"))


class _PathLabel(QLabel):
    """A path that elides in the middle when the row is narrow.

    The full path stays on the tooltip, so the label never wraps the dialog
    taller and nothing is lost.
    """

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path
        self.setObjectName("settingsProjectsFolder")
        self.setProperty("setupHint", True)
        self.setToolTip(path)
        self._elide()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        metrics = self.fontMetrics()
        self.setText(metrics.elidedText(self._path, Qt.TextElideMode.ElideMiddle, max(0, self.width())))


class SettingsDialog(QDialog):
    """Pictures folder, file type, prompt metadata, seed default, Expert mode, and 18+."""

    # Two naming passes existed for these controls while the screen was built;
    # both names resolve to the same widget.
    _NAME_ALIASES = {
        "settingsSeedDice": "settingsRandomSeed",
        "settingsOpenFolder": "settingsFolderOpen",
    }

    def findChild(self, cls, name="", options=Qt.FindChildOption.FindChildrenRecursively):  # noqa: N802 - Qt override
        found = super().findChild(cls, name, options)
        if found is not None or not isinstance(name, str):
            return found
        alias = self._NAME_ALIASES.get(name)
        if alias:
            return super().findChild(cls, alias, options)
        return None

    def __init__(self, parent: ShellWindow) -> None:
        """Build the screen around the window's stored settings.

        Args:
            parent: The shell window. Its layout and Expert mode are read and
                written; the shared controls stay in step through the same
                stored values.
        """
        super().__init__(parent)
        self._window = parent
        self.setObjectName("settingsDialog")
        self.setWindowTitle(text("canvas.settings.title"))
        self.setStyleSheet(shell_style())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        title = QLabel(text("canvas.settings.title"))
        title.setObjectName("settingsTitle")
        title.setProperty("dialogTitle", True)
        layout.addWidget(title)

        layout.addWidget(self._section("canvas.settings.pictures"))
        folder_row = QWidget()
        folder_line = QHBoxLayout(folder_row)
        folder_line.setContentsMargins(0, 0, 0, 0)
        folder_line.setSpacing(8)
        shown = _PathLabel(str(projects_dir()))
        open_button = QPushButton(text("canvas.settings.open_folder"))
        open_button.setObjectName("settingsFolderOpen")
        open_button.clicked.connect(self._open_folder)
        folder_line.addWidget(shown, 1)
        folder_line.addWidget(open_button)
        layout.addWidget(folder_row)

        type_row = QWidget()
        type_line = QHBoxLayout(type_row)
        type_line.setContentsMargins(0, 0, 0, 0)
        type_line.setSpacing(8)
        type_label = QLabel(text("canvas.format"))
        type_label.setObjectName("settingsFormatLabel")
        type_label.setProperty("setupHint", True)
        self._format = QComboBox()
        self._format.setObjectName("settingsFormat")
        for value, key in _FORMATS:
            self._format.addItem(text(key), value)
        self._format.currentIndexChanged.connect(self._choose_format)
        type_line.addWidget(type_label)
        type_line.addWidget(self._format, 1)
        layout.addWidget(type_row)

        self._metadata = QCheckBox(text("canvas.save_metadata"))
        self._metadata.setObjectName("settingsMetadata")
        self._metadata.setToolTip(text("canvas.save_metadata.help"))
        self._metadata.toggled.connect(lambda on: self._set("canvas.save_metadata", bool(on)))
        layout.addWidget(self._metadata)

        layout.addWidget(self._section("canvas.settings.generation"))
        seed_row = QWidget()
        seed_line = QHBoxLayout(seed_row)
        seed_line.setContentsMargins(0, 0, 0, 0)
        seed_line.setSpacing(8)
        self._dice = QCheckBox(text("canvas.settings.random_seed"))
        self._dice.setObjectName("settingsSeedDice")
        self._dice.setToolTip(tr("canvas.seed.tip"))
        self._dice.toggled.connect(self._choose_dice)
        self._seed = QSpinBox()
        self._seed.setObjectName("settingsSeed")
        self._seed.setRange(0, FIELD_MAX)
        self._seed.setSingleStep(1)
        self._seed.valueChanged.connect(lambda value: self._set_seed(int(value)))
        seed_line.addWidget(self._dice)
        seed_line.addWidget(self._seed)
        seed_line.addStretch(1)
        layout.addWidget(seed_row)

        self._expert = QCheckBox(text("canvas.settings.expert"))
        self._expert.setObjectName("settingsExpert")
        self._expert.toggled.connect(lambda on: self._window.set_expert(bool(on)))
        layout.addWidget(self._expert)

        layout.addWidget(self._section("canvas.settings.content"))
        self._adult = QCheckBox(tr("canvas.adult.box"))
        self._adult.setObjectName("settingsAdult")
        self._adult.setToolTip(tr("canvas.adult.tip"))
        self._adult.toggled.connect(self._choose_adult)
        layout.addWidget(self._adult)

        done = QPushButton(text("canvas.settings.done"))
        done.setObjectName("settingsDone")
        done.setProperty("primary", True)
        done.setDefault(True)
        done.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(done)
        layout.addLayout(row)
        self._fill()

    def _section(self, key: str) -> QLabel:
        heading = QLabel(text(key))
        heading.setObjectName(f"settingsSection-{key.rsplit('.', 1)[-1]}")
        heading.setProperty("presetLabel", True)
        heading.setStyleSheet("font-weight: 600; padding-top: 6px;")
        return heading

    def _fill(self) -> None:
        """Copy every stored value into its control."""
        window = self._window
        layout = window.shell.layout
        self._format.blockSignals(True)
        index = self._format.findData(str(layout.get("canvas.format") or "webp"))
        self._format.setCurrentIndex(max(0, index))
        self._format.blockSignals(False)
        self._metadata.blockSignals(True)
        self._metadata.setChecked(layout.get("canvas.save_metadata") is True)
        self._metadata.blockSignals(False)
        locked = layout.seed_locked()
        self._dice.blockSignals(True)
        self._dice.setChecked(not locked)
        self._dice.blockSignals(False)
        self._seed.blockSignals(True)
        self._seed.setEnabled(locked)
        self._seed.setValue(int(layout.seed))
        self._seed.blockSignals(False)
        self._expert.blockSignals(True)
        self._expert.setChecked(window._expert.enabled())
        self._expert.blockSignals(False)
        self._adult.blockSignals(True)
        self._adult.setChecked(layout.adult_content())
        self._adult.blockSignals(False)

    def _set(self, key: str, value: object) -> None:
        self._window.shell.layout.set_value(key, value)

    def _set_seed(self, value: int) -> None:
        self._window.shell.layout.seed = value

    def _choose_format(self, _index: int) -> None:
        value = self._format.currentData()
        if isinstance(value, str) and value:
            self._set("canvas.format", value)

    def _choose_adult(self, on: bool) -> None:
        layout = self._window.shell.layout
        if layout.adult_content() is not bool(on):
            layout.set_adult_content(bool(on))

    def _choose_dice(self, on: bool) -> None:
        layout = self._window.shell.layout
        layout.set_seed_locked(not bool(on))
        self._seed.setEnabled(not bool(on))

    def _open_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(projects_dir())))
