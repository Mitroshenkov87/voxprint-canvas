"""The UI font and Qt's translations reach the installer and the Linux archive."""
from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_package_ships_the_ui_font():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    shipped = data["tool"]["setuptools"]["package-data"]["canvas.ui"]
    assert "fonts/*.ttf" in shipped and "fonts/OFL.txt" in shipped
    assert "Noto Sans" in (ROOT / "NOTICE").read_text(encoding="utf-8")


def test_both_bundles_carry_the_canvas_tree():
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "cp -a canvas " in release
    assert "Copy-Item -Path canvas -Destination $stage -Recurse" in release


def test_no_installer_drops_qt_translations():
    for script in (ROOT / "installer" / "windows" / "install.ps1", ROOT / "installer" / "linux" / "install.sh"):
        assert "translations" not in script.read_text(encoding="utf-8").lower()
