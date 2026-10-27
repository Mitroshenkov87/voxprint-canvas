"""The bundled UI font.

The app draws with the bundled Noto Sans (SIL OFL 1.1, see fonts/OFL.txt) on every
platform, so text metrics match the reviewed renders.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication

FAMILY = "Noto Sans"
FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FILES = ("NotoSans-Regular.ttf", "NotoSans-SemiBold.ttf", "NotoSans-Bold.ttf")

_loaded: list[int] = []


def register_fonts() -> bool:
    """Add the bundled font files once. True when the family is available."""
    if not _loaded:
        for name in FONT_FILES:
            font_id = QFontDatabase.addApplicationFont(str(FONT_DIR / name))
            if font_id >= 0:
                _loaded.append(font_id)
    if FAMILY not in QFontDatabase.families():
        return False
    app = QGuiApplication.instance()
    if isinstance(app, QGuiApplication) and app.font().family() != FAMILY:
        # Widgets with their own stylesheet fall back to the application font, so make it the bundled one.
        font = QFont(FAMILY)
        old = app.font()
        if old.pixelSize() > 0:
            font.setPixelSize(old.pixelSize())
        elif old.pointSizeF() > 0:
            font.setPointSizeF(old.pointSizeF())
        app.setFont(font)
    return True

