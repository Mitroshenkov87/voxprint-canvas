"""Help menu opens the installed guide PDFs."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication, QMenu

from canvas.app.i18n import text
from canvas.ui import help_menu
from canvas.ui.help_menu import add_help_menu


def test_help_menu_lists_both_guides_and_opens_the_installed_one(tmp_path: Path, monkeypatch) -> None:
    QApplication.instance() or QApplication([])
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "VOXPrint-AI-Canvas-Guide-en.pdf").write_bytes(b"%PDF-1.7\n")
    opened: list[str] = []
    monkeypatch.setattr(help_menu.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True)
    parent = QMenu()
    menu = add_help_menu(parent, tmp_path)
    assert menu.menuAction() in parent.actions()
    assert menu.title() == text("menu.help")
    actions = menu.actions()
    assert [action.text() for action in actions] == [text("menu.help.guide_en"), text("menu.help.guide_ru")]
    assert actions[0].isEnabled() and not actions[1].isEnabled()
    actions[0].trigger()
    assert opened == [str(docs / "VOXPrint-AI-Canvas-Guide-en.pdf")]
