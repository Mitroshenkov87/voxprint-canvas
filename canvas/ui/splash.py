"""Start-up splash. The version line is the release identity from ``BUILD.json``."""
from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from canvas.app.identity import version_label
from canvas.app.info import APP_NAME
from canvas.ui.theme import ROOT_PLAIN, SPLASH, TEXT, TEXT_MUTED

SIDE = 360


class Splash(QSplashScreen):
    def __init__(self) -> None:
        self.version_line = version_label()
        pixmap = QPixmap(SIDE, SIDE)
        pixmap.fill(QColor(ROOT_PLAIN))
        super().__init__(pixmap)

    def drawContents(self, painter: QPainter) -> None:  # noqa: N802 - Qt name
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(self.font())
        font.setPixelSize(22)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(TEXT))
        painter.drawText(QRect(24, 120, SIDE - 48, 36), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, APP_NAME)
        font.setPixelSize(13)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(TEXT_MUTED))
        painter.drawText(
            QRect(24, 168, SIDE - 48, 28),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.version_line,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(SPLASH))
        painter.drawRect(QRect(24, 210, SIDE - 48, 4))


def show_splash() -> Splash:
    splash = Splash()
    splash.show()
    QApplication.processEvents()
    return splash
