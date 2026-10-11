"""Short UI strings chosen from the system locale.

English is the fallback. Russian is used when the locale starts with ``ru``.
"""
from __future__ import annotations

import locale
import os

INSTRUCTION_EDIT_UNAVAILABLE_EN = (
    "Editing is temporarily unavailable: we are preparing a model for 16 GB+ graphics cards"
)
INSTRUCTION_EDIT_UNAVAILABLE_RU = (
    "Редактирование временно недоступно: готовим модель для видеокарт от 16 ГБ"
)

_STRINGS = {
    "instruction_edit_unavailable": {
        "en": INSTRUCTION_EDIT_UNAVAILABLE_EN,
        "ru": INSTRUCTION_EDIT_UNAVAILABLE_RU,
    },
}


def normalize(language: str) -> str:
    """``ru`` for a Russian locale, otherwise ``en``."""
    token = language.strip().casefold().replace("_", "-")
    if token.startswith("ru"):
        return "ru"
    return "en"


def system_language() -> str:
    """The process locale, from ``LC_ALL``, ``LC_MESSAGES``, ``LANG``, then the C locale."""
    for key in ("LC_ALL", "LC_MESSAGES", "LANG"):
        raw = os.environ.get(key, "").strip()
        name = raw.split(".", 1)[0]
        if name and name not in {"C", "POSIX"}:
            return name
    current = locale.getlocale()[0] or ""
    return current or "en"


def text(key: str, language: str | None = None) -> str:
    """One string for ``key`` in the requested language, or the system locale."""
    row = _STRINGS[key]
    lang = normalize(system_language() if language is None else language)
    return row[lang]
