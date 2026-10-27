"""Help menu opens the installed guide PDF."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication, QMenu

from canvas.app.i18n import text
from canvas.ui import help_menu
from canvas.ui.help_menu import add_help_menu


def test_help_menu_lists_the_guide_and_opens_it(tmp_path: Path, monkeypatch) -> None:
    QApplication.instance() or QApplication([])
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "VOXPrint-AI-Canvas-Guide-en.pdf").write_bytes(b"%PDF-1.7\n")
    opened: list[Path] = []
    monkeypatch.setattr(help_menu.QDesktopServices, "openUrl", lambda url: opened.append(Path(url.toLocalFile())) or True)
    parent = QMenu()
    menu = add_help_menu(parent, tmp_path)
    assert menu.menuAction() in parent.actions()
    assert menu.title() == text("menu.help")
    actions = menu.actions()
    assert [action.text() for action in actions] == [text("menu.help.guide_en")]
    assert actions[0].isEnabled()
    actions[0].trigger()
    assert opened == [docs / "VOXPrint-AI-Canvas-Guide-en.pdf"]
    (docs / "VOXPrint-AI-Canvas-Guide-en.pdf").unlink()
    other = QMenu()
    assert not add_help_menu(other, tmp_path).actions()[0].isEnabled()
