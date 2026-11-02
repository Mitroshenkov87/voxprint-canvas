"""Open a picture file as a QPixmap, including formats Qt has no plugin for (AVIF)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QImage, QPixmap

from canvas.safety.image import RawImage  # the pixel container the mask painter already uses


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


def load_rgb(path: Path) -> RawImage:
    """Read one RGB image for the picture picker and the mask painter (Pillow decode only).

    Args:
        path: PNG, JPEG, WebP, or AVIF file.

    Raises:
        OSError: The file cannot be read as an image.
    """
    from PIL import Image

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            return RawImage(rgb.width, rgb.height, rgb.tobytes())
    except (OSError, ValueError) as exc:
        raise OSError(str(exc)) from exc
