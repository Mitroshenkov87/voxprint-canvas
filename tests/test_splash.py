"""The startup splash matches the frozen 960×540 spec."""
import importlib
import time
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QWidget

from canvas.app.info import APP_NAME
from canvas.ui.splash import (
    BAR_H,
    FADE_MS,
    HOLD_MS,
    HIDPI_UPSCALER,
    LOGICAL_H,
    LOGICAL_W,
    MARGIN,
    PADDING,
    RADIUS,
    SHOW_BUDGET_MS,
    FUTURE_IMPORT,
    SHARED_RUNTIME_SCHEMA,
    Splash,
    show_splash,
)
from canvas.ui.theme import SPLASH, TEXT, UI_FONT

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_shared_splash_adapter_names_voxprint_common():
    assert FUTURE_IMPORT == "voxprint_common.ui.splash"
    assert SHARED_RUNTIME_SCHEMA == 1
    shared = importlib.import_module(FUTURE_IMPORT)
    assert hasattr(shared, "Splash")
    splash = (ROOT / "canvas" / "ui" / "splash.py").read_text(encoding="utf-8")
    assert "voxprint_engine" not in splash
    main = (ROOT / "canvas" / "app" / "main.py").read_text(encoding="utf-8")
    assert "from canvas.ui.splash import Splash, show_splash" in main


def test_startup_shows_the_splash_before_downloads_or_the_gpu_probe():
    body = (ROOT / "canvas" / "app" / "main.py").read_text(encoding="utf-8")
    shown = body.index("splash = show_splash()")
    assert shown < body.index("probe_gpu(")
    assert shown < body.index("install_defaults(")
    assert "time.sleep" not in (ROOT / "canvas" / "ui" / "splash.py").read_text(encoding="utf-8")


def test_approved_art_and_lanczos_hidpi_asset():
    approved = Image.open(ROOT / "assets" / "splash.jpg")
    hidpi = Image.open(ROOT / "canvas" / "ui" / "art" / "splash@2x.jpg")
    packaged = (ROOT / "canvas" / "ui" / "art" / "splash.jpg").read_bytes()
    assert approved.size == (1280, 720)
    assert packaged == (ROOT / "assets" / "splash.jpg").read_bytes()
    assert hidpi.size == (LOGICAL_W * 2, LOGICAL_H * 2)
    assert HIDPI_UPSCALER == "lanczos"


def test_splash_is_a_centred_borderless_card(app):
    del app
    splash = Splash()
    splash.show()
    QApplication.processEvents()
    assert (LOGICAL_W, LOGICAL_H) == (960, 540)
    assert splash.card_size == (LOGICAL_W, LOGICAL_H)
    assert splash.width() == LOGICAL_W + 2 * MARGIN
    assert splash.height() == LOGICAL_H + 2 * MARGIN
    assert RADIUS == 12
    flags = splash.windowFlags()
    assert flags & Qt.WindowType.FramelessWindowHint
    screen = QApplication.primaryScreen()
    assert screen is not None
    center = splash.frameGeometry().center()
    assert abs(center.x() - screen.availableGeometry().center().x()) <= 1
    assert abs(center.y() - screen.availableGeometry().center().y()) <= 1
    image = splash.grab().toImage()
    assert image.pixelColor(0, 0).alpha() < 16
    cut = image.pixelColor(MARGIN + 2, MARGIN + 2)
    inside = image.pixelColor(MARGIN + 40, MARGIN + 8)
    assert cut.alpha() < inside.alpha()
    assert inside.alpha() > 200
    outside = image.pixelColor(MARGIN - 6, MARGIN + LOGICAL_H // 2)
    assert outside.alpha() > 8
    splash.close()


def test_left_column_type_and_progress_bar(app):
    del app
    splash = Splash()
    assert splash.version_line.endswith('"Yotzer"')
    assert APP_NAME == "VOXPrint AI Canvas"
    assert splash.name_font.pixelSize() == 40
    assert splash.name_rect().height() >= 80
    assert splash.version_rect().top() >= splash.name_rect().bottom()
    assert splash.name_font.weight() == QFont.Weight.DemiBold
    assert splash.version_font.pixelSize() == 16
    assert splash.version_font.weight() == QFont.Weight.Normal
    assert splash.stage_font.pixelSize() == 14
    assert list(splash.name_font.families())[:2] == list(UI_FONT)[:2]
    left = LOGICAL_W // 3
    assert splash.text_width == left - 2 * PADDING
    bar = splash.bar_rect()
    assert bar.x() == PADDING
    assert bar.y() == LOGICAL_H - PADDING - BAR_H
    assert bar.width() == left - 2 * PADDING
    assert bar.height() == BAR_H == 4
    assert splash.off_white == TEXT
    assert splash.version_alpha == pytest.approx(0.70)
    assert splash.track_alpha == pytest.approx(0.20)
    assert splash.fill_color == SPLASH
    assert splash.progress is None
    splash.set_stage("Opening")
    assert splash.stage == "Opening"
    splash.set_progress(0.4)
    assert splash.progress == pytest.approx(0.4)
    assert splash.shimmering is False
    splash.close()


def test_show_returns_immediately_and_fade_waits_then_lasts_200ms(app):
    del app
    started = time.perf_counter()
    splash = show_splash()
    assert time.perf_counter() - started < SHOW_BUDGET_MS / 1000
    assert splash.isVisible()
    host = QWidget()
    splash._shown_at = time.monotonic()
    began = time.perf_counter()
    splash.finish(host)
    assert time.perf_counter() - began < 0.05
    assert splash.isVisible()
    assert splash.windowOpacity() == pytest.approx(1.0)
    assert splash.hold_ms() > 400
    splash._shown_at = time.monotonic() - 2
    splash._finishing = False
    splash.finish(host)
    QApplication.processEvents()
    assert splash.fade_ms == FADE_MS == 200
    assert HOLD_MS == 600
    assert SHOW_BUDGET_MS == 150
    splash.close()
    host.close()
