"""Help menu with the installed guide PDF."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMenu

from canvas.app.i18n import text

GUIDES = (
    ("menu.help.guide_en", "VOXPrint-AI-Canvas-Guide-en.pdf"),
)


def docs_dir(app_dir: Path | None = None) -> Path:
    """``<app dir>/docs``, where Setup puts the guide PDFs."""
    root = app_dir if app_dir is not None else Path(__file__).resolve().parents[2]
    return root / "docs"


def add_help_menu(parent: QMenu, app_dir: Path | None = None) -> QMenu:
    """Add a Help submenu to the window menu. A guide that is not installed is shown disabled."""
    menu = parent.addMenu(text("menu.help"))
    menu.setObjectName("menuHelp")
    folder = docs_dir(app_dir)
    for key, name in GUIDES:
        path = folder / name
        action = menu.addAction(text(key))
        action.setEnabled(path.is_file())
        action.triggered.connect(lambda _checked=False, target=path: open_guide(target))
    return menu


def open_guide(path: Path) -> bool:
    """Open one guide in the system PDF viewer."""
    return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))))
