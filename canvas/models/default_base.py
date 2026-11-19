"""The shared suite default mapped onto a Canvas text-base family."""
from __future__ import annotations

from voxprint_common.suite_settings import model_default, read

# Common's model.default ids onto Canvas catalog text bases.
# ``z-image-turbo`` is the default generator. Full ``z-image`` stays an
# optional download; it is not an alias of Turbo. A leftover ``moss`` value
# is read as Turbo (MOSS is no longer a Canvas generator).
COMMON_TO_TEXT_BASE = {
    "z-image-turbo": "z-image-turbo",
    "z-image": "z-image",
}

_FALLBACK = "z-image-turbo"
_RETIRED = frozenset({"moss", "moss-image", "moss-image-sft", "moss-image-base", "moss5b"})


def selected_text_base() -> str:
    """Return the text-base family for the shared ``model.default``.

    A missing, unknown, or retired (MOSS) stored choice becomes Z-Image-Turbo.
    An explicit ``z-image`` pick stays the optional full model, not Turbo.
    """
    raw = read().get("model.default")
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in _RETIRED:
            return _FALLBACK
        if key in COMMON_TO_TEXT_BASE:
            return COMMON_TO_TEXT_BASE[key]
    return COMMON_TO_TEXT_BASE.get(model_default(), _FALLBACK)
