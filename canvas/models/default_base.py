"""The shared suite default mapped onto a Canvas text-base family."""
from __future__ import annotations

from voxprint_common.suite_settings import model_default

# Common's model.default ids (`z-image`, `moss`) onto Canvas catalog text bases.
COMMON_TO_TEXT_BASE = {
    "z-image": "z-image-turbo",
    "moss": "moss-image-sft",
}

_FALLBACK = "z-image-turbo"


def selected_text_base() -> str:
    """Return the text-base family for the shared ``model.default``.

    A missing or unknown stored choice follows common and becomes Z-Image.
    This is the setting users who never picked a model explicitly follow.
    """
    return COMMON_TO_TEXT_BASE.get(model_default(), _FALLBACK)
