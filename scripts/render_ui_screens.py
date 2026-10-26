"""Render each main Canvas screen offscreen in EN and RU at 1920×1080.

Usage::

    python scripts/render_ui_screens.py docs/screenshots/ui-polish/after
    python scripts/render_ui_screens.py /tmp/hidpi --scale 2

Weights are not loaded and nothing is generated: the window gets the in-memory
gate and a runner that returns a flat picture. Clipped or overflowing text is
printed as a warning, not raised, so a "before" render of a broken screen still
completes. PNGs are palette-quantized to keep the folder small.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("destination", type=Path)
    parser.add_argument("--scale", type=float, default=1.0, help="devicePixelRatio, e.g. 2 for HiDPI")
    parser.add_argument("--lang", choices=("en", "ru"), action="append", help="default: both")
    return parser.parse_args()


ARGS = _arguments() if __name__ == "__main__" else None
if ARGS is not None and ARGS.scale != 1.0:
    os.environ["QT_SCALE_FACTOR"] = str(ARGS.scale)

from PIL import Image  # noqa: E402
from PySide6.QtCore import QSize  # noqa: E402
from PySide6.QtGui import QPainter  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QApplication,
    QComboBox,
    QLabel,
    QProgressBar,
    QWidget,
)

from canvas.models.modes import Tier  # noqa: E402
from tools.grab_shell_screens import _sample, _window, prepare  # noqa: E402
from voxprint_common.i18n import set_language  # noqa: E402
from voxprint_common.ui.simple_image import Phase  # noqa: E402

SIZE = QSize(1920, 1080)
MAIN = ("simple", "expert", "simple-result", "expert-custom")


def main() -> int:
    """Write every screen for each language and print the overflow warnings."""
    assert ARGS is not None
    destination: Path = ARGS.destination
    destination.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    home = Path(tempfile.mkdtemp(prefix="canvas-ui-render-"))
    picture = home / "sample.png"
    _sample(picture)
    warnings: list[str] = []
    for language in ARGS.lang or ("en", "ru"):
        os.environ["LANG"] = os.environ["LC_ALL"] = "ru_RU.UTF-8" if language == "ru" else "en_US.UTF-8"
        set_language(language)
        for name in MAIN:
            window = _open(home / language / name, name)
            _save(app, window, None, destination / f"{name}-{language}.png", warnings)
            window.close()
        window = _open(home / language / "tools", "simple")
        dialogs = (
            ("setup", window.setup_wizard),
            ("picture", window.picture_wizard),
            ("mask-need", window.mask_wizard),
        )
        for name, build in dialogs:
            dialog = build()
            if name == "picture":
                dialog.load_path(picture)
            _save(app, window, dialog, destination / f"{name}-{language}.png", warnings)
        window.use_picture(picture)
        for name, build in (
            ("mask", window.mask_wizard),
            ("addons", window.addons_wizard),
            ("check", window.diagnostics_wizard),
            ("enlarge-confirm", window.enlarge_confirm),
            ("enlarge-progress", window.enlarge_progress),
        ):
            dialog = build()
            if name == "enlarge-progress":
                bar = dialog.findChild(QProgressBar, "enlargeProgressBar")
                if bar is not None:
                    bar.setRange(0, 100)
                    bar.setValue(40)
            _save(app, window, dialog, destination / f"{name}-{language}.png", warnings)
        window.close()
    for line in warnings:
        print("WARN", line)
    total = sum(p.stat().st_size for p in destination.glob("*.png"))
    print(f"{len(list(destination.glob('*.png')))} PNGs, {total / 1e6:.2f} MB in {destination}")
    return 0


def _open(root: Path, name: str):
    os.environ["VOXPRINT_HOME"] = str(root)
    window = _window(root / "state", Tier.GB16)
    prepare(window, name)
    return window


def _save(app: QApplication, window, dialog: QWidget | None, path: Path, warnings: list[str]) -> None:
    window.setFixedSize(SIZE)
    window.show()
    app.processEvents()
    if window.shell.layout.phase is not Phase.EMPTY:
        window.view._rescale()
    if dialog is not None:
        dialog.adjustSize()
        dialog.show()
    for _ in range(3):
        app.processEvents()
    warnings.extend(f"{path.name}: {line}" for line in overflow(dialog or window))
    base = window.grab().toImage()
    if dialog is not None:
        popup = dialog.grab()
        ratio = popup.devicePixelRatio()
        painter = QPainter(base)
        painter.drawPixmap(
            int((SIZE.width() - popup.width() / ratio) // 2),
            int((SIZE.height() - popup.height() / ratio) // 2),
            popup,
        )
        painter.end()
        dialog.close()
    raw = path.with_suffix(".raw.png")
    base.save(str(raw))
    Image.open(raw).convert("RGB").quantize(colors=128, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    raw.unlink()


def overflow(root: QWidget) -> list[str]:
    """Return one line per visible label or button whose text does not fit.

    Args:
        root: The window or dialog to inspect.
    """
    problems: list[str] = []
    for widget in [root, *root.findChildren(QWidget)]:
        if not widget.isVisible():
            continue
        if isinstance(widget, QLabel):
            words = widget.text()
            if not words or widget.wordWrap() or widget.pixmap() is not None and not widget.pixmap().isNull():
                continue
            if "<" in words:
                continue
            need = max(widget.fontMetrics().horizontalAdvance(line) for line in words.splitlines() or [""])
            room = widget.contentsRect().width() - 2 * widget.margin() - widget.indent() * (widget.indent() > 0)
            if need > room + 1:
                problems.append(f"label {widget.objectName() or '?'} {words[:40]!r} needs {need} has {room}")
        elif isinstance(widget, QAbstractButton) and widget.text():
            parent = widget.parentWidget()
            if parent is not None and widget.geometry().bottom() > parent.rect().bottom():
                problems.append(f"button {widget.objectName() or '?'} {widget.text()[:40]!r} is cut at the bottom")
            need = widget.fontMetrics().horizontalAdvance(widget.text().replace("&", ""))
            room = widget.width() - 16 - (widget.iconSize().width() + 4 if not widget.icon().isNull() else 0)
            if widget.inherits("QCheckBox") or widget.inherits("QRadioButton"):
                room = widget.width() - 24
            if need > room:
                problems.append(f"button {widget.objectName() or '?'} {widget.text()[:40]!r} needs {need} has {room}")
        elif isinstance(widget, QComboBox) and widget.currentText():
            need = widget.fontMetrics().horizontalAdvance(widget.currentText())
            room = widget.width() - 36
            if need > room:
                words = widget.currentText()[:40]
                problems.append(f"combo {widget.objectName() or '?'} {words!r} needs {need} has {room}")
        if widget is not root and widget.parentWidget() is root:
            corner = widget.mapTo(root, widget.rect().bottomRight())
            if corner.x() > root.width() or corner.y() > root.height():
                name = widget.objectName() or type(widget).__name__
                problems.append(f"widget {name} spills out of {root.objectName()}")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
