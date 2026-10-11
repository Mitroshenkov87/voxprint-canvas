"""Startup splash.

The shared widget will be published by ``voxprint-common`` (Suite Lead) as
``voxprint_common.ui.splash``. That package is not installed, so this module
keeps the same public names and implements the local splash for SHARED-RUNTIME
schema 1. ``canvas.app.main`` imports ``Splash`` and ``show_splash`` from here;
that import is the one-line swap when the shared widget arrives.

The 2× asset is a Lanczos upscale of ``assets/splash.jpg``. The Enlarge CLI
was not available to render it.
"""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

from canvas.app.i18n import text
from canvas.app.identity import splash_line
from canvas.ui.theme import SPLASH, TEXT, UI_FONT

LOGICAL_W = 960
LOGICAL_H = 540
RADIUS = 12
PADDING = 48
BAR_H = 4
MARGIN = 32
HOLD_MS = 600
FADE_MS = 200
SHOW_BUDGET_MS = 150
# Placeholder. Replace the file with an Enlarge upscale and set this to "enlarge".
HIDPI_UPSCALER = "lanczos"
# One-line swap once package voxprint-common publishes the shared widget.
FUTURE_IMPORT = "voxprint_common.ui.splash"
SHARED_RUNTIME_SCHEMA = 1

_ART = Path(__file__).resolve().parent / "art" / "splash@2x.jpg"
_NAME_PX = 40
_VERSION_PX = 16
_STAGE_PX = 14
_VERSION_ALPHA = 0.70
_TRACK_ALPHA = 0.20


def _font(px: int, weight: QFont.Weight) -> QFont:
    font = QFont()
    font.setFamilies(list(UI_FONT))
    font.setPixelSize(px)
    font.setWeight(weight)
    return font


class Splash(QWidget):
    """Borderless startup card. Art fills the 960×540 logical window."""

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(LOGICAL_W + 2 * MARGIN, LOGICAL_H + 2 * MARGIN)
        self.setWindowTitle(text("app.name"))
        self.version_line = splash_line()
        self.stage = ""
        self.progress: float | None = None
        self.name_font = _font(_NAME_PX, QFont.Weight.DemiBold)
        self.version_font = _font(_VERSION_PX, QFont.Weight.Normal)
        self.stage_font = _font(_STAGE_PX, QFont.Weight.Normal)
        self.off_white = TEXT
        self.version_alpha = _VERSION_ALPHA
        self.track_alpha = _TRACK_ALPHA
        self.fill_color = SPLASH
        self.fade_ms = FADE_MS
        self._pixmap = QPixmap(str(_ART)) if _ART.is_file() else QPixmap()
        if not self._pixmap.isNull():
            self._pixmap.setDevicePixelRatio(2)
        self._phase = 0.0
        self._shown_at: float | None = None
        self._finishing = False
        self._host: QWidget | None = None
        self._fade: QPropertyAnimation | None = None
        self._clock = QTimer(self)
        self._clock.setInterval(32)
        self._clock.timeout.connect(self._advance)

    @property
    def card_size(self) -> tuple[int, int]:
        """Logical art size, in pixels."""
        return (LOGICAL_W, LOGICAL_H)

    @property
    def text_width(self) -> int:
        """Width of the type column inside the left third."""
        return LOGICAL_W // 3 - 2 * PADDING

    @property
    def shimmering(self) -> bool:
        """True until the first determinate progress value."""
        return self.progress is None

    def bar_rect(self) -> QRect:
        """Progress track in logical card coordinates."""
        return QRect(PADDING, LOGICAL_H - PADDING - BAR_H, self.text_width, BAR_H)

    def name_rect(self) -> QRect:
        """App-name block. 40 px type wraps inside the padded left third."""
        bounds = QFontMetrics(self.name_font).boundingRect(
            0, 0, self.text_width, 240,
            int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
            text("app.name"),
        )
        return QRect(PADDING, PADDING, self.text_width, max(bounds.height(), _NAME_PX))

    def version_rect(self) -> QRect:
        """Release line, directly under the app name."""
        return QRect(PADDING, self.name_rect().bottom() + 8, self.text_width, 24)

    def set_stage(self, text: str) -> None:
        """Replace the status line under the release identity."""
        self.stage = text
        self.update()

    def set_progress(self, value: float | None) -> None:
        """Fill the bar. ``None`` keeps the indeterminate shimmer."""
        if value is None:
            self.progress = None
            if self.isVisible() and not self._clock.isActive():
                self._clock.start()
        else:
            self.progress = min(1.0, max(0.0, float(value)))
            self._clock.stop()
        self.update()

    def hold_ms(self) -> int:
        """Milliseconds still required before the fade may start."""
        started = self._shown_at if self._shown_at is not None else time.monotonic()
        remaining = HOLD_MS - int((time.monotonic() - started) * 1000)
        return max(0, remaining)

    def finish(self, widget: QWidget) -> None:
        """Fade out once the card has been visible for at least 600 ms.

        Returns immediately. The main window is not delayed.
        """
        self._host = widget
        if self._finishing:
            return
        self._finishing = True
        QTimer.singleShot(self.hold_ms(), self._start_fade)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt name
        """Centre the card and start the shimmer clock."""
        self._place()
        if self._shown_at is None:
            self._shown_at = time.monotonic()
        if self.progress is None:
            self._clock.start()
        super().showEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        """Paint the shadow, the art, the left-column type, and the bar."""
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        card = QRect(MARGIN, MARGIN, LOGICAL_W, LOGICAL_H)
        self._paint_shadow(painter, card)
        path = QPainterPath()
        path.addRoundedRect(QRectF(card), RADIUS, RADIUS)
        painter.setClipPath(path)
        self._paint_art(painter, card)
        painter.translate(MARGIN, MARGIN)
        self._paint_type(painter)
        self._paint_bar(painter)

    def _place(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        frame = self.frameGeometry()
        frame.moveCenter(screen.availableGeometry().center())
        self.move(frame.topLeft())

    def _advance(self) -> None:
        self._phase = (self._phase + 0.045) % 1.0
        self.update()

    def _start_fade(self) -> None:
        self._clock.stop()
        fade = QPropertyAnimation(self, b"windowOpacity", self)
        fade.setDuration(FADE_MS)
        fade.setStartValue(self.windowOpacity())
        fade.setEndValue(0.0)
        fade.setEasingCurve(QEasingCurve.Type.Linear)
        fade.finished.connect(self._after_fade)
        self._fade = fade
        fade.start()

    def _after_fade(self) -> None:
        host = self._host
        self.close()
        if host is not None:
            host.raise_()
            host.activateWindow()

    def _paint_shadow(self, painter: QPainter, card: QRect) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        for spread, alpha in ((20, 28), (14, 40), (8, 54), (3, 70)):
            painter.setBrush(QColor(0, 0, 0, alpha))
            painter.drawRoundedRect(card.adjusted(-spread, -spread + 3, spread, spread + 6), RADIUS + spread, RADIUS + spread)

    def _paint_art(self, painter: QPainter, card: QRect) -> None:
        if self._pixmap.isNull():
            painter.fillRect(card, QColor("#17171c"))
            return
        painter.drawPixmap(card, self._pixmap)

    def _paint_type(self, painter: QPainter) -> None:
        name = self.name_rect()
        version = self.version_rect()
        stage = QRect(PADDING, version.bottom() + 8, self.text_width, 48)
        painter.setFont(self.name_font)
        painter.setPen(QColor(TEXT))
        painter.drawText(
            name,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            text("app.name"),
        )
        painter.setFont(self.version_font)
        muted = QColor(TEXT)
        muted.setAlphaF(self.version_alpha)
        painter.setPen(muted)
        painter.drawText(version, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.version_line)
        painter.setFont(self.stage_font)
        painter.setPen(QColor(TEXT))
        painter.drawText(
            stage,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            self.stage,
        )

    def _paint_bar(self, painter: QPainter) -> None:
        bar = self.bar_rect()
        track = QColor(TEXT)
        track.setAlphaF(self.track_alpha)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track)
        radius = BAR_H / 2
        painter.drawRoundedRect(QRectF(bar), radius, radius)
        fill = QColor(SPLASH)
        painter.setBrush(fill)
        if self.progress is None:
            span = max(28, bar.width() * 2 // 5)
            travel = bar.width() + span
            x = bar.x() - span + int(self._phase * travel)
            painter.setClipRect(bar)
            painter.drawRoundedRect(QRectF(x, bar.y(), span, BAR_H), radius, radius)
            return
        filled = QRectF(bar.x(), bar.y(), bar.width() * self.progress, BAR_H)
        painter.drawRoundedRect(filled, radius, radius)


def show_splash() -> Splash:
    """Show the splash and return it. Does not wait for the main window."""
    splash = Splash()
    splash.show()
    app = QApplication.instance()
    if app is not None:
        app.processEvents()
    return splash
