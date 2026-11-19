"""Approved suite icons. Pending rows in ``ICONS.md`` are not loaded.

Colour comes from a palette token. There is no theme-icon fallback.
"""
from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from canvas.ui.theme import TEXT

ICON_DIR = Path(__file__).resolve().parent
_HEX = re.compile(r"#[0-9A-Fa-f]{3,8}")


def approved_names(spec: str) -> frozenset[str]:
    """Icon stems whose ``ICONS.md`` status is approved. Pending rows are omitted."""
    names: set[str] = set()
    for line in spec.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        file_name, status = cells[0], cells[1].casefold()
        if not file_name.endswith(".svg") or status != "approved":
            continue
        names.add(file_name[:-4])
    return frozenset(names)


def suite_icon(name: str, colour: str = TEXT) -> QIcon:
    """One approved icon, stroked in ``colour`` (a palette token)."""
    spec = (ICON_DIR / "ICONS.md").read_text(encoding="utf-8")
    if name not in approved_names(spec):
        raise ValueError(f"{name} is not an approved icon")
    raw = (ICON_DIR / f"{name}.svg").read_text(encoding="utf-8")
    if _HEX.search(raw):
        raise ValueError(f"{name} must be recoloured from the palette")
    coloured = raw.replace("currentColor", colour)
    renderer = QSvgRenderer(QByteArray(coloured.encode("utf-8")))
    if not renderer.isValid():
        raise ValueError(f"{name} is not a valid icon")
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


def load_approved(colour: str = TEXT) -> dict[str, QIcon]:
    """Every approved icon. Pending names are not included."""
    spec = (ICON_DIR / "ICONS.md").read_text(encoding="utf-8")
    return {name: suite_icon(name, colour) for name in sorted(approved_names(spec))}
