"""Roles for an image the user supplies. Every role is screened the same way."""
from __future__ import annotations

from enum import Enum
from typing import assert_never

from canvas.safety.image import RawImage


class ImageRole(Enum):
    """What an uploaded image is for. Adding a role does not skip the checks."""

    INIT = "init"
    MASK = "mask"
    REFERENCE = "reference"
    STYLE = "style"
    POSE = "pose"
    DEPTH = "depth"
    EDGE = "edge"


class PendingInput:
    """An uploaded image that has not been screened."""

    __slots__ = ("role", "image")

    def __init__(self, role: ImageRole, image: RawImage) -> None:
        if not isinstance(role, ImageRole):
            raise TypeError("unknown image role")
        if not isinstance(image, RawImage):
            raise TypeError("an input image has not been staged")
        self.role = role
        self.image = image


def role_name(role: ImageRole) -> str:
    """Return the stored name. A new role fails here until it is handled."""
    if role is ImageRole.INIT:
        return role.value
    if role is ImageRole.MASK:
        return role.value
    if role is ImageRole.REFERENCE:
        return role.value
    if role is ImageRole.STYLE:
        return role.value
    if role is ImageRole.POSE:
        return role.value
    if role is ImageRole.DEPTH:
        return role.value
    if role is ImageRole.EDGE:
        return role.value
    assert_never(role)
