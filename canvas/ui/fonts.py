"""The bundled UI font and Qt's own translations.

The app draws with the bundled Noto Sans (SIL OFL 1.1, see fonts/OFL.txt) on every
platform, so text metrics match the reviewed renders. Qt's own dialogs (the file
dialog) read in the app language through qtbase_<lang>.qm from PySide6.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEvent, QLibraryInfo, QTranslator
from PySide6.QtGui import QFontDatabase
from voxprint_common.i18n import get_language

FAMILY = "Noto Sans"
FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FILES = ("NotoSans-Regular.ttf", "NotoSans-SemiBold.ttf", "NotoSans-Bold.ttf")

_loaded: list[int] = []
_translator: dict[str, QTranslator] = {}


def register_fonts() -> bool:
    """Add the bundled font files once. True when the family is available."""
    if not _loaded:
        for name in FONT_FILES:
            font_id = QFontDatabase.addApplicationFont(str(FONT_DIR / name))
            if font_id >= 0:
                _loaded.append(font_id)
    return FAMILY in QFontDatabase.families()


def translations_dir() -> Path:
    return Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath))


def qt_translations(lang: str | None = None) -> str:
    """Install qtbase_<lang>.qm for the app language. Returns the language installed, or ''."""
    app = QCoreApplication.instance()
    if app is None:
        return ""
    lang = (lang or get_language() or "en").split("-")[0].split("_")[0]
    if lang in _translator:
        return lang
    for old in _translator.values():
        app.removeTranslator(old)
    _translator.clear()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.LanguageChange)
    if lang == "en":
        return "en"
    translator = QTranslator(app)
    if not translator.load(f"qtbase_{lang}", str(translations_dir())):
        return ""
    app.installTranslator(translator)
    _translator[lang] = translator
    # Deliver the language change now, so labels set on a new dialog are not reset by it later.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.LanguageChange)
    return lang
