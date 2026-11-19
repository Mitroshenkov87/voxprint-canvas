"""Plain-language looks. Each one is a suffix and a negative prompt.

The shared style row stores the negative. The suffix is applied when the
picture is composed, in the same prompt field.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Look:
    """One look the style chips can select."""

    style_id: str
    suffix: str
    negative: str


LOOKS: tuple[Look, ...] = (
    Look("photo", "photorealistic photograph", ""),
    Look("cinematic", "cinematic lighting, film still", "flat lighting"),
    Look("portrait", "portrait photograph, shallow depth of field", "wide scene"),
    Look("illustration", "illustration", "photograph"),
    Look("anime", "anime drawing", "photorealistic"),
    Look("3d", "3D render", "flat drawing"),
    Look("painting", "painting", "line sketch"),
    Look("sketch", "pencil sketch", "full color"),
)


def look_for(style_id: str) -> Look:
    """Return the look named ``style_id``.

    Args:
        style_id: Chip id, such as ``photo`` or ``3d``.

    Raises:
        KeyError: The id is not one of the shipped looks.
    """
    for look in LOOKS:
        if look.style_id == style_id:
            return look
    raise KeyError(style_id)


def style_label(style_id: str) -> str:
    """Catalog key for one look. The shared catalog already has these keys.

    Args:
        style_id: Chip id.
    """
    return f"simple.style.{style_id}"
