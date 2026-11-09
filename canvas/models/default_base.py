"""The shared suite default mapped onto a Canvas text-base family."""
from __future__ import annotations

from voxprint_common.suite_settings import model_default

# Common's model.default ids onto Canvas catalog text bases.
# ``z-image-turbo`` is the default generator. Full ``z-image`` and ``moss``
# are optional downloads; they are not aliases of Turbo.
COMMON_TO_TEXT_BASE = {
    "z-image-turbo": "z-image-turbo",
    "z-image": "z-image",
    "moss": "moss-image-sft",
}

_FALLBACK = "z-image-turbo"


def selected_text_base() -> str:
    """Return the text-base family for the shared ``model.default``.

    A missing or unknown stored choice follows common and becomes Z-Image-Turbo.
    An explicit ``z-image`` pick stays the optional full model, not Turbo.
    """
    return COMMON_TO_TEXT_BASE.get(model_default(), _FALLBACK)
