"""Offscreen shots of the simple screen and the secondary tools.

Weights are not loaded. The content gate is the real gate with the in-memory
stand-in classifiers. Nothing is generated.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication

from canvas.models.modes import Tier
from canvas.ui.shell_layout import open_layout
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import World, rgb
from voxprint_common.i18n import set_language


def main() -> int:
    """Write EN and RU shots of the main screen and the secondary screens."""
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/cursor/artifacts/shell-screens")
    destination.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="canvas-shell-shots-"))
    os.environ["VOXPRINT_HOME"] = str(home)
    app = QApplication.instance() or QApplication([])
    picture = destination / "sample.png"
    Image.new("RGB", (64, 48), (70, 110, 160)).save(picture)
    for language in ("en", "ru"):
        os.environ["LANG"] = "ru_RU.UTF-8" if language == "ru" else "en_US.UTF-8"
        os.environ["LC_ALL"] = os.environ["LANG"]
        set_language(language)
        for expert, name in ((False, "simple"), (True, "expert")):
            window = _window(home / language / name, Tier.GB16)
            window.set_expert(expert)
            _save(app, window, destination / f"{name}-{language}.png")
            window.close()
        window = _window(home / language / "tools", Tier.GB16)
        _save_dialog(app, window.setup_wizard(), destination / f"setup-{language}.png")
        _save_dialog(app, window.picture_wizard(), destination / f"picture-{language}.png", load=picture)
        _save_dialog(app, window.mask_wizard(), destination / f"mask-need-{language}.png")
        window.use_picture(picture)
        _save_dialog(app, window.mask_wizard(), destination / f"mask-{language}.png")
        _save_dialog(app, window.addons_wizard(), destination / f"addons-{language}.png")
        _save_dialog(app, window.diagnostics_wizard(), destination / f"check-{language}.png")
        window.close()
    return 0


def _window(state: Path, tier: Tier) -> ShellWindow:
    shell = open_layout(state, World().gate(), enlarge_installed=False)
    return ShellWindow(shell, tier=tier, enlarge=object(), runner=lambda *_args: rgb(), improver=lambda prompt: prompt)


def _save(app: QApplication, window: ShellWindow, path: Path) -> None:
    window.show()
    app.processEvents()
    window.grab().save(str(path))


def _save_dialog(app: QApplication, dialog, path: Path, load: Path | None = None) -> None:
    if load is not None and hasattr(dialog, "load_path"):
        dialog.load_path(load)
    dialog.resize(720, 520)
    dialog.show()
    app.processEvents()
    dialog.grab().save(str(path))
    dialog.close()


if __name__ == "__main__":
    raise SystemExit(main())
