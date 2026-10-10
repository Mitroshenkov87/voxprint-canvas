"""Approved icons only. A floppy drawing and the theme save icon are banned."""
import os
import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from canvas.ui.icons import approved_names, load_approved, suite_icon
from canvas.ui.theme import TEXT

ROOT = Path(__file__).resolve().parents[1]
BANNED = ("document-save", "floppy", "SP_DialogSaveButton")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_pending_rows_are_not_approved():
    spec = "\n".join([
        "| File | Status | Use |",
        "| save.svg | approved | Save |",
        "| later.svg | PENDING | not yet |",
    ])
    assert approved_names(spec) == frozenset({"save"})


def test_approved_icons_recolour_from_the_palette_and_render(app):
    del app
    icons = load_approved()
    assert set(icons) == {"download", "export", "save"}
    for name, icon in icons.items():
        assert not icon.isNull()
        raw = (ROOT / "canvas" / "ui" / "icons" / f"{name}.svg").read_text(encoding="utf-8")
        assert "currentColor" in raw
        assert re.search(r"#[0-9A-Fa-f]{3,8}", raw) is None
        assert TEXT not in raw
    with pytest.raises(ValueError):
        suite_icon("later")


def test_no_floppy_or_theme_save_fallback():
    offenders: list[str] = []
    roots = [ROOT / "canvas", ROOT / "installer"]
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".py", ".svg", ".iss", ".qss"}:
                continue
            text = path.read_text(encoding="utf-8").casefold()
            for word in BANNED:
                if word.casefold() in text:
                    offenders.append(f"{path.relative_to(ROOT)}:{word}")
    assert offenders == []
