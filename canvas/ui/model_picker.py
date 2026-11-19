"""The Generate-model picker: installed generate-group models in two tier sections.

The compact section lists the 16 GB-class builds, the full section the 24 GB+
ones, each row carrying a size badge. The section that fits the detected card
starts expanded with a row preselected; the other sits collapsed under a
«Show models for other cards» disclosure. The list comes from
:mod:`canvas.models.generate_picker`, so this widget holds no model ids.
"""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from voxprint_common.ui.theme import CONTROL_BG, CONTROL_BORDER, TEXT_FAINT, TEXT_MUTED

from canvas.app.i18n import text
from canvas.models.generate_picker import FULL_TIER, GenerateChoice, split
from canvas.models.modes import Tier
from canvas.ui.theme import BUTTON_RADIUS

ROW_HEIGHT = 36
SECTION_IDS = ("compact", "full")
BADGE_WIDTH = 64
# Quiet section titles, size badges and disclosure; the row buttons keep the shared look.
STYLE = f"""
QWidget#modelPicker QLabel[presetLabel="true"] {{ color: {TEXT_MUTED}; font-size: 12px; font-weight: 600; padding-top: 4px; }}
QWidget#modelPicker QLabel#modelBadge {{
    background: {CONTROL_BG}; border: 1px solid {CONTROL_BORDER}; border-radius: {BUTTON_RADIUS}px;
    color: {TEXT_MUTED}; font-size: 12px; padding: 0 8px;
}}
QWidget#modelPicker QToolButton#modelPickerOthers {{ color: {TEXT_FAINT}; font-size: 12px; padding: 6px 2px; border: none; }}
QWidget#modelPicker QToolButton#modelPickerOthers:hover {{ color: {TEXT_MUTED}; }}
"""


class ModelPicker(QWidget):
    """The generate-model pick, as two tier sections with size badges."""

    def __init__(self, on_change: Callable[[str], None]) -> None:
        super().__init__()
        self.setObjectName("modelPicker")
        self.setToolTip(text("canvas.model.pick_tip"))
        self.setStyleSheet(STYLE)
        self._on_change = on_change
        self._loading = False
        self._choices: tuple[GenerateChoice, ...] = ()
        self._tier: Tier | None = None
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._column = QVBoxLayout(self)
        self._column.setContentsMargins(0, 0, 0, 0)
        self._column.setSpacing(6)
        self._headers: dict[str, QLabel] = {}
        self._bodies: dict[str, QWidget] = {}
        for section in SECTION_IDS:
            header = QLabel(text(f"canvas.model.section.{section}"))
            header.setObjectName(f"modelSection-{section}")
            header.setProperty("presetLabel", True)
            body = QWidget()
            body.setObjectName(f"modelRows-{section}")
            rows = QVBoxLayout(body)
            rows.setContentsMargins(0, 0, 0, 0)
            rows.setSpacing(4)
            self._column.addWidget(header)
            self._column.addWidget(body)
            self._headers[section] = header
            self._bodies[section] = body
        self._others = QToolButton()
        self._others.setObjectName("modelPickerOthers")
        self._others.setCheckable(True)
        self._others.setAutoRaise(True)
        self._others.setCursor(Qt.CursorShape.PointingHandCursor)
        self._others.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._others.setArrowType(Qt.ArrowType.RightArrow)
        self._others.setText(text("canvas.model.show_others"))
        self._others.toggled.connect(self._toggled_others)
        self._column.addWidget(self._others)
        self.hide()

    def reload(self, choices_: tuple[GenerateChoice, ...], selected: str, tier: Tier | None) -> None:
        """Rebuild the sections and select ``selected``.

        The section holding the selection starts expanded; the other is
        collapsed under the «Show models for other cards» toggle. With nothing
        installed the picker hides itself.
        """
        self._loading = True
        self._choices = choices_
        self._tier = tier
        self._clear_rows()
        compact, full = split(choices_)
        for section, rows in zip(SECTION_IDS, (compact, full), strict=True):
            layout = self._bodies[section].layout()
            if layout is None:
                continue
            for choice in rows:
                layout.addWidget(self._row(choice))
        listed = bool(choices_)
        self.setVisible(listed)
        chosen = self._button(selected) or self._button(_first_id(compact, full))
        if chosen is not None:
            chosen.setChecked(True)
        self._sync_sections(tier)
        self._loading = False

    def selected(self) -> str:
        """The id of the checked row, empty when none."""
        button = self._group.checkedButton()
        return "" if button is None else str(button.property("model_id") or "")

    def section_expanded(self, section: str) -> bool:
        """True when ``section`` («compact» or «full») rows are shown."""
        return self._bodies[section].isVisibleTo(self)

    def others_shown(self) -> bool:
        """True when the collapsed section is revealed through the toggle."""
        return self._others.isChecked()

    def _clear_rows(self) -> None:
        for body in self._bodies.values():
            layout = body.layout()
            if layout is None:
                continue
            while layout.count():
                item = layout.takeAt(0)
                widget = None if item is None else item.widget()
                if widget is not None:
                    button = widget.findChild(QPushButton, "modelPick")
                    if button is not None:
                        self._group.removeButton(button)
                    widget.deleteLater()

    def _row(self, choice: GenerateChoice) -> QWidget:
        row = QWidget()
        row.setObjectName(f"modelRow-{choice.id}")
        row.setToolTip(choice.tip)
        row.setFixedHeight(ROW_HEIGHT)
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(8)
        button = QPushButton(choice.text)
        button.setObjectName("modelPick")
        button.setCheckable(True)
        button.setProperty("model_id", choice.id)
        button.setAccessibleName(choice.text)
        button.setToolTip(choice.tip)
        button.setFixedHeight(ROW_HEIGHT)
        policy = button.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        button.setSizePolicy(policy)
        button.toggled.connect(lambda checked, model_id=choice.id: self._picked(checked, model_id))
        badge_key = "canvas.model.badge.full" if choice.tier == FULL_TIER else "canvas.model.badge.compact"
        badge = QLabel(text(badge_key))
        badge.setObjectName("modelBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setToolTip(choice.tip)
        badge.setFixedSize(BADGE_WIDTH, ROW_HEIGHT)
        self._group.addButton(button)
        line.addWidget(button, 1)
        line.addWidget(badge)
        return row

    def _button(self, model_id: str) -> QPushButton | None:
        if not model_id:
            return None
        for button in self._group.buttons():
            if isinstance(button, QPushButton) and str(button.property("model_id") or "") == model_id:
                return button
        return None

    def _picked(self, checked: bool, model_id: str) -> None:
        if self._loading or not checked:
            return
        self._sync_sections(self._tier)
        self._on_change(model_id)

    def _section_of(self, model_id: str) -> str:
        for choice in self._choices:
            if choice.id == model_id:
                return "full" if choice.tier == FULL_TIER else "compact"
        return "compact"

    def _sync_sections(self, tier: Tier | None = None) -> None:
        """Expand the section holding the selection; collapse the other."""
        selected = self.selected()
        if selected:
            chosen_section = self._section_of(selected)
        else:
            chosen_section = "full" if tier is Tier.GB24 else "compact"
        for section in SECTION_IDS:
            expanded = section == chosen_section
            self._headers[section].setVisible(expanded)
            self._bodies[section].setVisible(expanded)
        self._others.blockSignals(True)
        self._others.setChecked(False)
        self._others.setText(text("canvas.model.show_others"))
        self._others.setArrowType(Qt.ArrowType.RightArrow)
        self._others.setVisible(bool(self._choices))
        self._others.blockSignals(False)

    def _toggled_others(self, shown: bool) -> None:
        selected = self.selected()
        chosen_section = self._section_of(selected) if selected else "compact"
        other = "full" if chosen_section == "compact" else "compact"
        self._headers[other].setVisible(shown)
        self._bodies[other].setVisible(shown)
        self._others.setText(text("canvas.model.hide_others" if shown else "canvas.model.show_others"))
        self._others.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)


def _first_id(compact: tuple[GenerateChoice, ...], full: tuple[GenerateChoice, ...]) -> str:
    rows = compact or full
    return rows[0].id if rows else ""
