"""Open a picture file as a QPixmap, including formats Qt has no plugin for (AVIF)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QImage, QPixmap


def load_pixmap(path: str | Path | None) -> QPixmap:
    """The picture at ``path``. Qt reads it first; Pillow decodes what Qt cannot. Null when neither can."""
    if not path:
        return QPixmap()
    pixmap = QPixmap(str(path))
    if not pixmap.isNull():
        return pixmap
    try:
        from PIL import Image

        with Image.open(str(path)) as opened:
            rgba = opened.convert("RGBA")
    except (OSError, ValueError, ImportError):
        return QPixmap()
    data = rgba.tobytes("raw", "RGBA")
    image = QImage(data, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(image)
