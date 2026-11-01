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
from typing import cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image, ImageDraw
from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLabel, QProgressBar, QPushButton, QSpinBox, QWidget

from canvas.models.mdl_registry import Model
from canvas.models.modes import Tier
from canvas.ui import photoshoot_run
from canvas.ui.shell_layout import open_layout
from canvas.ui.shell_screen import ShellWindow
from tests.fakes import World, rgb
from voxprint_common.i18n import set_language, tr
from voxprint_common.ui.simple_image import Phase

SHOT = QSize(1280, 800)


def main() -> int:
    """Write shots of the main screen and the secondary screens."""
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/cursor/artifacts/shell-screens")
    destination.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="canvas-shell-shots-"))
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication([])
    picture = destination / "sample.png"
    _sample(picture)
    for language in ("en",):
        os.environ["LANG"] = "en_US.UTF-8"
        os.environ["LC_ALL"] = os.environ["LANG"]
        set_language(language)
        for name in ("simple", "expert", "simple-result", "expert-custom"):
            window = _shot(home / language / name, name, Tier.GB16)
            _save(app, window, destination / f"{name}-{language}.png", name)
            window.close()
        window = _shot(home / language / "tools", "simple", Tier.GB16)
        _save_dialog(app, window, window.setup_wizard(), destination / f"setup-{language}.png", "setup")
        _save_dialog(app, window, window.picture_wizard(), destination / f"picture-{language}.png", "picture", load=picture)
        _save_dialog(app, window, window.mask_wizard(), destination / f"mask-need-{language}.png", "mask-need")
        window.use_picture(picture)
        _save_dialog(app, window, window.mask_wizard(), destination / f"mask-{language}.png", "mask")
        _save_dialog(app, window, window.addons_wizard(), destination / f"addons-{language}.png", "addons")
        _save_dialog(app, window, window.diagnostics_wizard(), destination / f"check-{language}.png", "check")
        _save_dialog(
            app, window, window.enlarge_confirm(), destination / f"enlarge-confirm-{language}.png", "enlarge-confirm"
        )
        progress = window.enlarge_progress()
        bar = progress.findChild(QProgressBar, "enlargeProgressBar")
        if bar is not None:
            bar.setRange(0, 100)
            bar.setValue(40)
        _save_dialog(app, window, progress, destination / f"enlarge-progress-{language}.png", "enlarge-progress")
        window.close()
    return 0


def prepare(window: ShellWindow, name: str) -> None:
    """Apply the mode and values the named shot has to show.

    Args:
        window: A shell window. Each shot keeps its own home.
        name: Shot name without the language suffix.
    """
    if name == "expert-custom":
        window.shell.layout.set_value("shell.steps", 33)
    window.set_expert(name.startswith("expert"))
    if name in {"simple-result", "expert-custom"}:
        window.view._select("pic-01")


def _shot(root: Path, name: str, tier: Tier) -> ShellWindow:
    os.environ["VOXPRINT_HOME"] = str(root)
    return _open(root / "state", tier, name)


def _open(state: Path, tier: Tier, name: str) -> ShellWindow:
    window = _window(state, tier)
    prepare(window, name)
    return window


def _window(state: Path, tier: Tier, history_count: int = 50) -> ShellWindow:
    state.mkdir(parents=True, exist_ok=True)
    _history(state / "image-history.json", history_count)
    shell = open_layout(state, gate=World().gate(), enlarge_installed=False)
    return ShellWindow(shell, tier=tier, enlarge=object(), runner=lambda *_args: rgb(), improver=lambda prompt: prompt)


def _history(path: Path, count: int = 50) -> None:
    items = [
        {"id": f"pic-{index:02d}", "prompt": f"picture {index}", "presets": [], "seed": index}
        for index in range(1, count + 1)
    ]
    path.write_text(json.dumps({"schema": 1, "items": items}) + "\n", encoding="utf-8")


def _sample(path: Path) -> None:
    image = Image.new("RGB", (480, 320), (36, 64, 102))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 190, 480, 320), fill=(48, 92, 58))
    draw.polygon([(30, 200), (150, 70), (270, 200)], fill=(70, 78, 96))
    draw.ellipse((320, 36, 410, 126), fill=(240, 196, 90))
    image.save(path)


def _save(app: QApplication, window: ShellWindow, path: Path, name: str) -> None:
    window.setFixedSize(SHOT)
    window.show()
    # Two passes: the expert form settles on a queued call after the first.
    app.processEvents()
    app.processEvents()
    if window.shell.layout.phase is not Phase.EMPTY:
        window.view._rescale()
        app.processEvents()
    _expect(window, name)
    _check(window, path.name)
    shot = window.grab()
    if shot.size() != SHOT:
        raise SystemExit(f"{path.name} is {shot.width()}x{shot.height()}")
    shot.save(str(path))


def _save_dialog(
    app: QApplication,
    window: ShellWindow,
    dialog,
    path: Path,
    name: str,
    load: Path | None = None,
) -> None:
    if load is not None and hasattr(dialog, "load_path"):
        dialog.load_path(load)
    window.setFixedSize(SHOT)
    window.show()
    dialog.adjustSize()
    dialog.show()
    app.processEvents()
    _expect(window, name)
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


def _expect(window: ShellWindow, name: str) -> None:
    """Fail the shot when the mode, the edit, or a wrapped row is wrong."""
    panel = window.findChild(QWidget, "expertPanel")
    toggle = window.findChild(QPushButton, "expertToggle")
    if panel is None or toggle is None:
        raise SystemExit(f"{name} is missing the expert controls")
    expert = name.startswith("expert")
    if panel.isVisible() != expert or toggle.isChecked() != expert:
        raise SystemExit(f"{name} expert visible={panel.isVisible()} checked={toggle.isChecked()}")
    _expect_chips(window, name)
    if name in {"simple-result", "expert-custom"}:
        result = window.findChild(QLabel, "result")
        pixmap = None if result is None else result.pixmap()
        if pixmap is None or pixmap.isNull():
            raise SystemExit(f"{name} result is empty")
    if not expert:
        return
    _expect_combo(window, name)
    if name != "expert-custom":
        return
    spins = [item for item in window.findChildren(QSpinBox, "shell.steps") if item.isVisible()]
    if len(spins) != 1 or spins[0].value() != 33:
        seen = [(item.value(), item.isVisible()) for item in window.findChildren(QSpinBox, "shell.steps")]
        raise SystemExit(f"{name} steps {seen}")
    chip = window.findChild(QPushButton, "chip-quality-balanced")
    words = "" if chip is None else chip.text()
    if "changes" not in words:
        raise SystemExit(f"{name} chip {words!r}")
    if window.findChild(QPushButton, "expertResetAll") is not None:
        raise SystemExit(f"{name} still has the canvas reset")
    if window.findChild(QLabel, "expertChip") is not None:
        raise SystemExit(f"{name} still has the canvas badge")
    row_reset = window.findChild(QPushButton, "reset-quality")
    if row_reset is not None and row_reset.isVisible():
        raise SystemExit(f"{name} shows the row reset")
    resets = [item for item in window.findChildren(QPushButton, "expertReset") if item.isVisible()]
    if len(resets) != 1:
        raise SystemExit(f"{name} shared resets {len(resets)}")


def _expect_chips(window: ShellWindow, name: str) -> None:
    groups: dict[int, list[QPushButton]] = {}
    for button in window.findChildren(QPushButton):
        if not button.isVisible() or not button.property("chip"):
            continue
        groups.setdefault(id(button.parentWidget()), []).append(button)
    for buttons in groups.values():
        for index, left in enumerate(buttons):
            box = left.geometry()
            for right in buttons[index + 1 :]:
                if box.intersects(right.geometry()):
                    raise SystemExit(f"{name} overlap {left.objectName()} {right.objectName()}")


def _expect_combo(window: ShellWindow, name: str) -> None:
    combo = window.findChild(QComboBox, "canvas.output_size")
    if combo is None or not combo.isVisible():
        raise SystemExit(f"{name} is missing the size combo")
    fitted = combo.property("fittedText")
    if fitted != combo.currentText() or "…" in str(fitted):
        raise SystemExit(
            f"{name} combo {combo.currentText()!r} fitted {fitted!r} width {combo.width()}"
        )


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


def photoshoot_shots(destination: Path) -> None:
    """Write one plain smoke grab at 1280×800.

    Args:
        destination: Directory for ``photoshoot-en.png``.
    """
    destination.mkdir(parents=True, exist_ok=True)
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication([])
    os.environ["LANG"] = "en_US.UTF-8"
    os.environ["LC_ALL"] = "en_US.UTF-8"
    set_language("en")
    idea = "A yellow kettle with a blue handle"
    home = Path(tempfile.mkdtemp(prefix="photoshoot-shot-en-"))
    os.environ["VOXPRINT_HOME"] = str(home)
    window = _photoshoot_window(home, idea)
    _run_smoke(app, window, idea, destination / "photoshoot-en.png")
    window.close()


def _photoshoot_window(state: Path, idea: str) -> ShellWindow:
    state.mkdir(parents=True, exist_ok=True)
    shell = open_layout(state, gate=World().gate(), enlarge_installed=False)
    window = ShellWindow(shell, tier=Tier.GB16, enlarge=object(), runner=lambda *_args: rgb(), improver=lambda prompt: prompt)
    window.view.set_prompt_text(idea)
    return window


def _ready_models() -> tuple[Model, ...]:
    return cast(tuple[Model, ...], (object(),))


def _mode(window: ShellWindow) -> QCheckBox:
    mode = window.findChild(QCheckBox, "photoshootMode")
    if not isinstance(mode, QCheckBox):
        raise SystemExit("missing photoshootMode")
    return mode


def _button(window: ShellWindow, name: str) -> QPushButton:
    button = window.findChild(QPushButton, name)
    if not isinstance(button, QPushButton):
        raise SystemExit(f"missing {name}")
    return button


def _run_smoke(app: QApplication, window: ShellWindow, idea: str, path: Path) -> None:
    photoshoot_run.selectable = _ready_models
    calls = {"n": 0}

    def runner(job, inputs):
        del job, inputs
        calls["n"] += 1
        if calls["n"] == 2:
            _save_smoke(app, window, path)
        return rgb()

    window._runner = runner
    window.photoshoot.writer = lambda _prompt: _scenario(idea)
    window.setFixedSize(SHOT)
    window.show()
    _mode(window).click()
    app.processEvents()
    _button(window, "generate").click()
    app.processEvents()
    if calls["n"] < 2 or not path.is_file():
        raise SystemExit(f"{path.name} was not saved during the series")


def _save_smoke(app: QApplication, window: ShellWindow, path: Path) -> bool:
    if path.is_file():
        return True
    app.processEvents()
    mode = _mode(window)
    if not mode.isChecked() or mode.text() != tr("photoshoot.button"):
        raise SystemExit(f"{path.name} is missing the Photoshoot option")
    cancel = _button(window, "photoshootCancel")
    if not cancel.isVisible() or cancel.text() != tr("simple.cancel"):
        raise SystemExit(f"{path.name} is missing Cancel")
    lines = [item for item in window.findChildren(QLabel, "photoshootLine") if item.isVisible() and item.text().strip()]
    if not lines:
        raise SystemExit(f"{path.name} is missing the shot list")
    generate = _button(window, "generate")
    if not generate.isVisible():
        raise SystemExit(f"{path.name} hides the shell Create button")
    _check(window, path.name)
    shot = window.grab()
    if shot.size() != SHOT:
        raise SystemExit(f"{path.name} is {shot.width()}x{shot.height()}")
    shot.save(str(path))
    return True


def _scenario(idea: str) -> str:
    shots = []
    scenes = ["close view", "three-quarter view", "from above", "from the side", "by the window", "among cups"]
    for scene in scenes:
        shots.append({"scene": scene, "pose": "still", "camera": "50 mm", "place": "kitchen table"})
    body = {"subject_block": idea, "descriptors": [idea], "place": "kitchen table", "shots": shots}
    return json.dumps(body)


if __name__ == "__main__":
    raise SystemExit(main())
