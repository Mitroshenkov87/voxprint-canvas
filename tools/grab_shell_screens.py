"""Offscreen shots of the simple screen and the secondary tools.

Weights are not loaded. The content gate is the real gate with the in-memory
stand-in classifiers. Nothing is generated.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image, ImageDraw
from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QApplication, QPushButton

from canvas.models.modes import Tier
from canvas.ui.shell_layout import open_layout
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import World, rgb
from voxprint_common.i18n import set_language
from voxprint_common.ui.simple_image import Phase

SHOT = QSize(1280, 800)


def main() -> int:
    """Write EN and RU shots of the main screen and the secondary screens."""
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/cursor/artifacts/shell-screens")
    destination.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="canvas-shell-shots-"))
    os.environ["VOXPRINT_HOME"] = str(home)
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication([])
    picture = destination / "sample.png"
    _sample(picture)
    for language in ("en", "ru"):
        os.environ["LANG"] = "ru_RU.UTF-8" if language == "ru" else "en_US.UTF-8"
        os.environ["LC_ALL"] = os.environ["LANG"]
        set_language(language)
        for expert, name in ((False, "simple"), (True, "expert")):
            window = _window(home / language / name, Tier.GB16)
            window.set_expert(expert)
            _check(window, f"{name}-{language}")
            _save(app, window, destination / f"{name}-{language}.png")
            window.close()
        window = _window(home / language / "result", Tier.GB16)
        window.view._select("pic-01")
        _check(window, f"simple-result-{language}")
        _save(app, window, destination / f"simple-result-{language}.png")
        window.close()
        window = _window(home / language / "custom", Tier.GB16)
        window.set_expert(True)
        window.shell.layout.set_value("shell.steps", 33)
        window.view.refresh()
        _check(window, f"expert-custom-{language}")
        _save(app, window, destination / f"expert-custom-{language}.png")
        window.close()
        window = _window(home / language / "tools", Tier.GB16)
        _save_dialog(app, window, window.setup_wizard(), destination / f"setup-{language}.png")
        _save_dialog(app, window, window.picture_wizard(), destination / f"picture-{language}.png", load=picture)
        _save_dialog(app, window, window.mask_wizard(), destination / f"mask-need-{language}.png")
        window.use_picture(picture)
        _save_dialog(app, window, window.mask_wizard(), destination / f"mask-{language}.png")
        _save_dialog(app, window, window.addons_wizard(), destination / f"addons-{language}.png")
        _save_dialog(app, window, window.diagnostics_wizard(), destination / f"check-{language}.png")
        window.close()
    return 0


def _window(state: Path, tier: Tier) -> ShellWindow:
    state.mkdir(parents=True, exist_ok=True)
    _history(state / "image-history.json")
    shell = open_layout(state, gate=World().gate(), enlarge_installed=False)
    return ShellWindow(shell, tier=tier, enlarge=object(), runner=lambda *_args: rgb(), improver=lambda prompt: prompt)


def _history(path: Path) -> None:
    items = [
        {"id": f"pic-{index:02d}", "prompt": f"picture {index}", "presets": [], "seed": index}
        for index in range(1, 51)
    ]
    path.write_text(json.dumps({"schema": 1, "items": items}) + "\n", encoding="utf-8")


def _sample(path: Path) -> None:
    image = Image.new("RGB", (480, 320), (36, 64, 102))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 190, 480, 320), fill=(48, 92, 58))
    draw.polygon([(30, 200), (150, 70), (270, 200)], fill=(70, 78, 96))
    draw.ellipse((320, 36, 410, 126), fill=(240, 196, 90))
    image.save(path)


def _save(app: QApplication, window: ShellWindow, path: Path) -> None:
    window.setFixedSize(SHOT)
    window.show()
    app.processEvents()
    if window.shell.layout.phase is not Phase.EMPTY:
        window.view._rescale()
        app.processEvents()
    shot = window.grab()
    if shot.size() != SHOT:
        raise SystemExit(f"{path.name} is {shot.width()}x{shot.height()}")
    shot.save(str(path))


def _save_dialog(app: QApplication, window: ShellWindow, dialog, path: Path, load: Path | None = None) -> None:
    if load is not None and hasattr(dialog, "load_path"):
        dialog.load_path(load)
    window.setFixedSize(SHOT)
    window.show()
    dialog.adjustSize()
    dialog.show()
    app.processEvents()
    _check(dialog, path.name)
    base = window.grab().toImage()
    popup = dialog.grab()
    if base.size() != SHOT:
        raise SystemExit(f"{path.name} plate is {base.width()}x{base.height()}")
    if popup.height() > SHOT.height() - 24 or popup.width() > SHOT.width() - 24:
        raise SystemExit(f"{path.name} dialog is {popup.width()}x{popup.height()}")
    painter = QPainter(base)
    painter.drawPixmap(
        (SHOT.width() - popup.width()) // 2,
        (SHOT.height() - popup.height()) // 2,
        popup,
    )
    painter.end()
    base.save(str(path))
    dialog.close()


def _check(root, name: str) -> None:
    problems: list[str] = []
    for button in root.findChildren(QPushButton):
        if not button.isVisible() or not button.text() or button.objectName().startswith("thumb-"):
            continue
        gap_l, gap_r = _gaps(button.grab().toImage())
        if gap_l < 4 or gap_r < 4:
            problems.append(f"{button.objectName()} {button.text()!r} gaps {gap_l},{gap_r} width {button.width()}")
    if problems:
        raise SystemExit(name + " clipped: " + "; ".join(problems))


def _gaps(image) -> tuple[int, int]:
    width = image.width()
    height = image.height()
    if width < 8 or height < 8:
        return 0, 0
    fill = _pixel(image, width // 2, min(6, height - 1))
    left = width
    right = -1
    top = height // 3
    bottom = max(top + 1, (height * 2) // 3)
    for y in range(top, bottom):
        for x in range(2, width - 2):
            if _distance(_pixel(image, x, y), fill) < 48:
                continue
            left = min(left, x)
            right = max(right, x)
    if right < 0:
        return width, width
    return left, width - 1 - right


def _pixel(image, x: int, y: int) -> tuple[int, int, int]:
    colour = QColor(image.pixel(x, y))
    return colour.red(), colour.green(), colour.blue()


def _distance(one: tuple[int, int, int], other: tuple[int, int, int]) -> int:
    return abs(one[0] - other[0]) + abs(one[1] - other[1]) + abs(one[2] - other[2])


if __name__ == "__main__":
    raise SystemExit(main())
