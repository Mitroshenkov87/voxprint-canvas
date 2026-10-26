"""Offscreen captures for the user guide.

Extends ``tools/grab_shell_screens.py``. English and Russian shots are written
to ``docs/guide/screens/<lang>/<id>.png``. The two Setup message boxes are
drawn from the installer strings. Weights are not loaded.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton, QRadioButton

from canvas.app.i18n import text
from canvas.models.modes import Tier
from canvas.pipeline.photoshoot import Shot
from canvas.safety.image import RawImage
from canvas.ui import photoshoot_run
from canvas.ui.about import AboutDialog
from canvas.ui.shell_screen import ShellWindow
from canvas.ui.splash import show_splash
from tests.lora_files import write_lora
from tools.grab_shell_screens import SHOT, _sample, _save, _save_dialog, _window, prepare
from voxprint_common.i18n import set_language, tr
from voxprint_common.ui.simple_image import Phase
from voxprint_common.mdl.present import ManageRow, ManageView, WindowView
from voxprint_common.ui.mdl_window import grab_view

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "guide" / "screens"
PROMPTS = {
    "en": {
        "first": "A red bicycle leaning against a brick wall on a quiet street, morning sun, soft shadows",
        "kettle": "A yellow kettle with a blue handle",
        "lora": "a lighthouse on a rocky coast at dusk",
    },
    "ru": {
        "first": "Красный велосипед у кирпичной стены на тихой улице, утреннее солнце, мягкие тени",
        "kettle": "Жёлтый чайник с синей ручкой",
        "lora": "маяк на скалистом берегу в сумерках",
    },
}
PWSH = {
    "en": {
        "ask": "PowerShell 7 is not installed. Install PowerShell 7 (Microsoft) now? Recommended.",
        "fallback": (
            "Setup will continue with Windows PowerShell 5.1. Please install PowerShell 7 when possible: "
            "https://aka.ms/powershell-release?tag=stable"
        ),
        "yes": "Yes",
        "no": "No",
        "ok": "OK",
    },
    "ru": {
        "ask": "PowerShell 7 не установлен. Установить PowerShell 7 (Microsoft) сейчас? Рекомендуется.",
        "fallback": (
            "Установка продолжится с Windows PowerShell 5.1. По возможности установите PowerShell 7: "
            "https://aka.ms/powershell-release?tag=stable"
        ),
        "yes": "Да",
        "no": "Нет",
        "ok": "ОК",
    },
}
LIMIT = 400_000


def main() -> int:
    """Capture every guide id that this machine can show."""
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    captured, mocked, dropped = capture_guide(destination)
    print("captured:", ", ".join(captured))
    print("mocked:", ", ".join(mocked))
    print("dropped:", ", ".join(dropped) if dropped else "(none)")
    return 0


def capture_guide(destination: Path) -> tuple[list[str], list[str], list[str]]:
    """Write one PNG per id and language. Return captured, mocked and dropped ids."""
    destination.mkdir(parents=True, exist_ok=True)
    existing = QApplication.instance()
    app = existing if isinstance(existing, QApplication) else QApplication([])
    home = Path(tempfile.mkdtemp(prefix="canvas-guide-"))
    os.environ["VOXPRINT_HOME"] = str(home)
    landscape = home / "landscape.png"
    interior = home / "interior.png"
    _sample(landscape)
    _interior(interior)
    captured: list[str] = []
    for language in ("en", "ru"):
        _language(language)
        folder = destination / language
        folder.mkdir(parents=True, exist_ok=True)
        _splash(app, folder / "splash.png")
        _windows(app, home, language, folder, landscape, interior)
        captured.append(language)
    mocked = _messages(destination)
    dropped = _models(app, destination)
    _shrink(destination)
    return captured, mocked, dropped


def _language(language: str) -> None:
    os.environ["LANG"] = "ru_RU.UTF-8" if language == "ru" else "en_US.UTF-8"
    os.environ["LC_ALL"] = os.environ["LANG"]
    os.environ["LC_MESSAGES"] = os.environ["LANG"]
    set_language(language)


def _windows(
    app: QApplication,
    home: Path,
    language: str,
    folder: Path,
    landscape: Path,
    interior: Path,
) -> None:
    prompts = PROMPTS[language]
    state = home / language
    window = _open(state)
    _save(app, window, folder / "main-window-simple.png", "simple")
    _menu(app, window, window.view._overflow.menu(), window.view._overflow, folder / "more-menu.png")
    _prompt(window, prompts["first"])
    _save(app, window, folder / "first-image-prompt.png", "simple")
    _progress(app, window, folder / "first-image-progress.png")
    window.shell.layout.cancel()
    prepare(window, "simple-result")
    _prompt(window, prompts["first"])
    _save(app, window, folder / "first-image-result.png", "simple-result")
    menu = window.view.history_menu("pic-01")
    thumb = window.findChild(QPushButton, "thumb-pic-01")
    anchor = window.view._overflow if thumb is None else thumb
    _menu(app, window, menu, anchor, folder / "history-item-menu.png")
    _landscape(app, window, folder / "size-ratio-menu.png")
    window.close()

    expert = _open(state / "expert")
    prepare(expert, "expert")
    _save(app, expert, folder / "expert-panel.png", "expert")
    expert.close()

    custom = _open(state / "custom")
    prepare(custom, "expert-custom")
    _save(app, custom, folder / "expert-custom.png", "expert-custom")
    custom.close()

    detail = _open(state / "detail")
    detail._select_4k()
    app.processEvents()
    _save(app, detail, folder / "output-4k-detail.png", "simple")
    _save_dialog(app, detail, detail.setup_wizard(), folder / "picture-setup.png", "setup")
    _save_dialog(app, detail, detail.picture_wizard(), folder / "use-picture.png", "picture", load=landscape)
    _mask(app, detail, interior, folder / "mask-editor.png")
    _save_dialog(app, detail, detail.enlarge_confirm(), folder / "enlarge-confirm.png", "enlarge-confirm")
    progress = detail.enlarge_progress()
    progress.report(40, 100)
    _save_dialog(app, detail, progress, folder / "enlarge-progress.png", "enlarge-progress")
    detail.close()

    _photoshoot(app, state / "series", prompts["kettle"], folder)
    _addons(app, state / "lora", prompts["lora"], folder)
    _check(app, state / "check", folder)
    _about(app, state / "about", folder)


def _open(state: Path) -> ShellWindow:
    os.environ["VOXPRINT_HOME"] = str(state)
    window = _window(state / "state", Tier.GB16, history_count=10)
    prepare(window, "simple")
    return window


def _prompt(window: ShellWindow, words: str) -> None:
    window.view.set_prompt_text(words)


def _progress(app: QApplication, window: ShellWindow, path: Path) -> None:
    layout = window.shell.layout
    if not layout.generate():
        raise SystemExit(f"{path.name} did not start")
    layout.progress(1, 1, text("canvas.stage.draw"), eta_seconds=45)
    window.view.refresh()
    app.processEvents()
    _grab(app, window, path)


def _landscape(app: QApplication, window: ShellWindow, path: Path) -> None:
    window.shell.layout.choose("size", "16:9")
    window.view.refresh()
    app.processEvents()
    button = window.findChild(QPushButton, "proportionsMenu")
    if button is None or button.menu() is None:
        raise SystemExit("missing proportions menu")
    window.setFixedSize(SHOT)
    window.show()
    app.processEvents()
    menu = button.menu()
    menu.aboutToShow.emit()
    menu.adjustSize()
    base = window.grab().toImage()
    _paint_popup(base, window, button, menu.grab())
    base.save(str(path))


def _mask(app: QApplication, window: ShellWindow, interior: Path, path: Path) -> None:
    window.use_picture(interior)
    dialog = window.mask_wizard()
    board = dialog._board
    if board is None:
        raise SystemExit("mask editor has no picture")
    painter = QPainter(board._mask)
    painter.setBrush(QColor(255, 255, 255, 170))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(150, 150, 180, 120)
    painter.end()
    board.update()
    _save_dialog(app, window, dialog, path, "mask")


def _photoshoot(app: QApplication, state: Path, idea: str, folder: Path) -> None:
    os.environ["VOXPRINT_HOME"] = str(state)
    window = _window(state / "state", Tier.GB16, history_count=4)
    prepare(window, "simple")
    _prompt(window, idea)
    window.setFixedSize(SHOT)
    window.show()
    app.processEvents()
    mode = window.findChild(QCheckBox, "photoshootMode")
    if mode is None:
        raise SystemExit("missing Photoshoot")
    mode.click()
    app.processEvents()
    _grab(app, window, folder / "photoshoot-on.png")
    shots = tuple(
        Shot(scene, "still", "50 mm", "kitchen table")
        for scene in ("close view", "three-quarter view", "from above", "from the side", "by the window", "among cups")
    )
    if not window.shell.layout.generate():
        raise SystemExit("photoshoot progress did not start")
    window.photoshoot.show_scenario(idea, shots)
    window.photoshoot.mark(1, "drawing")
    window.view.refresh()
    window.photoshoot.sync()
    app.processEvents()
    _grab(app, window, folder / "photoshoot-running.png")
    window.shell.layout.cancel()
    window.photoshoot._showing = False
    window.photoshoot.sync()
    _finish_series(app, window, idea, folder / "photoshoot-done.png")
    window.close()


def _finish_series(app: QApplication, window: ShellWindow, idea: str, path: Path) -> None:
    photoshoot_run.selectable = _ready
    window.photoshoot.writer = lambda _prompt: _scenario(idea)
    window.photoshoot.enabled = True
    mode = window.findChild(QCheckBox, "photoshootMode")
    if mode is not None:
        mode.setChecked(True)

    def runner(job, inputs):
        del job, inputs
        return _kettle()

    window._runner = runner
    _prompt(window, idea)
    window.photoshoot._showing = False
    generate = window.findChild(QPushButton, "generate")
    if generate is None:
        raise SystemExit("missing Generate")
    generate.click()
    app.processEvents()
    window.photoshoot._showing = False
    window.photoshoot.sync()
    window.view.refresh()
    app.processEvents()
    _grab(app, window, path)


def _addons(app: QApplication, state: Path, prompt: str, folder: Path) -> None:
    models = state / "models"
    os.environ["VOXPRINT_MODELS_DIR"] = str(models)
    os.environ["VOXPRINT_HOME"] = str(state)
    write_lora(models, "inkwash", kind="style", trigger="inkwash", default_strength=0.8)
    window = _window(state / "state", Tier.GB16, history_count=4)
    prepare(window, "simple")
    dialog = window.addons_wizard()
    option = dialog.findChild(QRadioButton, "loraOption-inkwash")
    if option is None:
        raise SystemExit("the inkwash add-on is not listed")
    option.click()
    app.processEvents()
    _save_dialog(app, window, dialog, folder / "addons-dialog.png", "addons")
    window._save_addons(str(models / "user-models" / "lora" / "inkwash"), 0.8)
    _prompt(window, prompt)
    window.view.refresh()
    app.processEvents()
    _grab(app, window, folder / "lora-trigger-chip.png")
    window.close()


def _check(app: QApplication, state: Path, folder: Path) -> None:
    os.environ["VOXPRINT_HOME"] = str(state)
    window = _window(state / "state", Tier.GB16, history_count=4)
    prepare(window, "simple")
    dialog = window.diagnostics_wizard()
    report = "\n".join(
        (
            text("canvas.check.ready"),
            f"{text('canvas.check.os')}: ok",
            f"{text('canvas.check.gpu')}: ok",
            f"{text('canvas.check.encoders')}: AVIF, JPEG XL, WebP",
        )
    )
    dialog.show_report(report)
    _save_dialog(app, window, dialog, folder / "check-this-pc.png", "check")
    window.close()


def _about(app: QApplication, state: Path, folder: Path) -> None:
    os.environ["VOXPRINT_HOME"] = str(state)
    window = _window(state / "state", Tier.GB16, history_count=4)
    prepare(window, "simple")
    _save_dialog(app, window, AboutDialog(window), folder / "about-dialog.png", "about")
    window.close()


def _splash(app: QApplication, path: Path) -> None:
    splash = show_splash()
    splash.set_stage(text("splash.gpu"))
    splash.repaint()
    app.processEvents()
    splash.grab().save(str(path))
    splash.close()


def _models(app: QApplication, destination: Path) -> list[str]:
    del app
    try:
        for language in ("en", "ru"):
            _language(language)
            path = destination / language / "manage-models.png"
            grab_view(_manage_view(), path, (980, 640))
            if not path.is_file() or path.stat().st_size < 1000:
                raise RuntimeError("manage-models grab was empty")
    except Exception as exc:
        print(f"manage-models dropped: {exc}")
        for language in ("en", "ru"):
            path = destination / language / "manage-models.png"
            if path.is_file():
                path.unlink()
        return ["manage-models"]
    return []


def _manage_view() -> WindowView:
    installed = ManageRow(
        "z-image-turbo",
        "Z-Image-Turbo",
        "28 GB",
        "ok",
        tr("mdl.manage.status.ok"),
        "remove",
        tr("mdl.manage.remove"),
    )
    available = ManageRow(
        "controlnet-lite",
        "ControlNet lite",
        "2 GB",
        "missing",
        tr("mdl.manage.status.missing"),
        "add",
        tr("mdl.manage.add"),
    )
    manage = ManageView(
        (installed,),
        (available,),
        tr("mdl.manage.disk", app="Canvas", used="28 GB", free="40 GB", path="models"),
        tr("mdl.manage.empty_installed", app="Canvas"),
        tr("mdl.manage.empty_available"),
    )
    return WindowView(
        mode="manage",
        focus_app="canvas",
        colour="#3d6b8a",
        motif="dot",
        heading=tr("mdl.heading.manage"),
        subtitle="Canvas",
        groups=(),
        overall=1.0,
        overall_caption="",
        overall_detail="",
        status="",
        error_what_title="",
        error_what="",
        error_do_title="",
        error_do="",
        technical="",
        show_error=False,
        show_retry=False,
        paused=False,
        show_pause=False,
        show_cancel=False,
        show_open=False,
        open_label="",
        open_app="",
        all_set=True,
        manage_label=tr("mdl.manage.open"),
        back_label=tr("mdl.manage.back"),
        pause_label="",
        cancel_label="",
        retry_label="",
        close_label=tr("mdl.close"),
        details_label="",
        details_hide_label="",
        manage=manage,
        footer="",
    )


def _messages(destination: Path) -> list[str]:
    for language in ("en", "ru"):
        folder = destination / language
        folder.mkdir(parents=True, exist_ok=True)
        words = PWSH[language]
        _message_box(folder / "setup-pwsh-offer.png", words["ask"], (words["yes"], words["no"]), question=True)
        _message_box(folder / "setup-pwsh-fallback.png", words["fallback"], (words["ok"],), question=False)
    return ["setup-pwsh-offer", "setup-pwsh-fallback"]


def _message_box(path: Path, message: str, buttons: tuple[str, ...], *, question: bool) -> None:
    width = 760
    font = _font(15)
    title = _font(13)
    lines = _lines(message, font, width - 124)
    height = 96 + 22 * len(lines) + 48
    image = Image.new("RGB", (width, height), (243, 243, 243))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, 31), fill=(255, 255, 255))
    draw.line((0, 31, width, 31), fill=(210, 210, 210))
    draw.text((16, 8), "VOXPrint AI Canvas", fill=(28, 36, 48), font=title)
    mark = (28, 28, 28) if question else (20, 96, 168)
    draw.ellipse((28, 64, 68, 104), outline=mark, width=3)
    glyph = "?" if question else "i"
    draw.text((42, 72), glyph, fill=mark, font=font)
    top = 58
    for line in lines:
        draw.text((88, top), line, fill=(28, 36, 48), font=font)
        top += 22
    x = width - 24
    for label in reversed(buttons):
        box = 96
        x -= box + 8
        draw.rounded_rectangle((x, height - 56, x + box, height - 24), radius=4, fill=(255, 255, 255), outline=(180, 180, 180))
        tw = draw.textlength(label, font=font)
        draw.text((x + (box - tw) / 2, height - 50), label, fill=(28, 36, 48), font=font)
    image.save(path)


def _lines(message: str, font: ImageFont.ImageFont, limit: int) -> list[str]:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    words: list[str] = []
    for word in message.split():
        if draw.textlength(word, font=font) <= limit:
            words.append(word)
            continue
        chunk = ""
        for char in word:
            trial = chunk + char
            if chunk and draw.textlength(trial, font=font) > limit:
                words.append(chunk)
                chunk = char
            else:
                chunk = trial
        if chunk:
            words.append(chunk)
    lines: list[str] = []
    line = ""
    for word in words:
        trial = word if not line else f"{line} {word}"
        if line and draw.textlength(trial, font=font) > limit:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _interior(path: Path) -> None:
    image = Image.new("RGB", (640, 420), (232, 224, 208))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 250, 640, 420), fill=(176, 148, 112))
    draw.rectangle((40, 40, 220, 250), fill=(186, 206, 214))
    draw.rectangle((250, 180, 430, 300), fill=(92, 122, 96))
    draw.rectangle((270, 140, 410, 190), fill=(214, 196, 168))
    image.save(path)


def _kettle() -> RawImage:
    image = Image.new("RGB", (320, 240), (236, 232, 224))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((90, 70, 210, 190), radius=24, fill=(232, 196, 48))
    draw.rectangle((200, 100, 250, 130), fill=(48, 108, 176))
    return RawImage(image.width, image.height, image.convert("RGB").tobytes())


def _scenario(idea: str) -> str:
    scenes = ["close view", "three-quarter view", "from above", "from the side", "by the window", "among cups"]
    shots = [{"scene": scene, "pose": "still", "camera": "50 mm", "place": "kitchen table"} for scene in scenes]
    return json.dumps({"subject_block": idea, "descriptors": [idea], "place": "kitchen table", "shots": shots})


def _ready() -> tuple[object, ...]:
    return (object(),)


def _grab(app: QApplication, window: ShellWindow, path: Path) -> None:
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


def _menu(app: QApplication, window: ShellWindow, menu, anchor: QPushButton, path: Path) -> None:
    window.setFixedSize(SHOT)
    window.show()
    app.processEvents()
    menu.popup(anchor.mapToGlobal(anchor.rect().bottomLeft()))
    app.processEvents()
    base = window.grab().toImage()
    _paint_popup(base, window, anchor, menu.grab())
    menu.close()
    base.save(str(path))


def _paint_popup(base, window: ShellWindow, anchor, popup) -> None:
    """Draw a popup under its anchor. Offscreen popups do not land in the window grab."""
    local = anchor.mapTo(window, QPoint(0, anchor.height()))
    x = local.x()
    y = local.y()
    if y + popup.height() > SHOT.height() - 8:
        y = max(8, local.y() - anchor.height() - popup.height())
    if x + popup.width() > SHOT.width() - 8:
        x = SHOT.width() - popup.width() - 8
    painter = QPainter(base)
    painter.drawPixmap(max(8, x), max(8, y), popup)
    painter.end()


def _shrink(root: Path) -> None:
    for path in sorted(root.rglob("*.png")):
        image = Image.open(path)
        image.save(path, format="PNG", optimize=True, compress_level=9)
        if path.stat().st_size <= LIMIT:
            continue
        reduced = image.convert("RGB").quantize(colors=128, method=Image.Quantize.MEDIANCUT)
        reduced.save(path, format="PNG", optimize=True)
        if path.stat().st_size > LIMIT:
            raise SystemExit(f"{path} is {path.stat().st_size} bytes")


if __name__ == "__main__":
    raise SystemExit(main())
