"""UI strings. Every user-visible line lives in ``messages.json``.

Shipped locales are listed in that file. ``text`` follows the process locale:
Russian when it starts with ``ru``, otherwise English. Pass ``language="en"``
for a stable English line, including the CLI ``--json`` feature error.
"""
from __future__ import annotations

import json
import locale
import os
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from PySide6.QtCore import QLocale

_CATALOG_PATH = Path(__file__).with_name("messages.json")


@lru_cache(maxsize=1)
def _catalog() -> dict[str, object]:
    raw = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("messages.json must be an object")
    return raw


def shipped_locales() -> tuple[str, ...]:
    """Locales Canvas ships. A key without one of these is a broken catalog."""
    locales = _catalog().get("locales")
    if not isinstance(locales, list) or not locales:
        raise ValueError("messages.json has no locales")
    return tuple(str(item) for item in locales)


def strings() -> Mapping[str, Mapping[str, str]]:
    """Catalog entries keyed by id, then by locale."""
    found = _catalog().get("strings")
    if not isinstance(found, dict):
        raise ValueError("messages.json has no strings")
    return found


def normalize(language: str | None) -> str:
    """Map a locale tag onto a shipped language. Anything else is English."""
    if not language:
        return "en"
    token = language.strip().replace("-", "_").lower()
    if token in {"c", "posix"}:
        return "en"
    if token.startswith("ru"):
        return "ru"
    return "en"


def system_language() -> str:
    """Russian when the process locale starts with ru, otherwise English."""
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(name, "").strip()
        if value and value not in {"C", "POSIX"}:
            return normalize(value)
    qt_name = QLocale.system().name()
    if qt_name:
        return normalize(qt_name)
    try:
        current = locale.getlocale()[0] or ""
    except (ValueError, TypeError):
        current = ""
    return normalize(current)


def plural_form(count: int, language: str | None = None) -> str:
    """``one``, ``few``, or ``many`` for a count in the active language."""
    lang = normalize(language) if language is not None else system_language()
    number = abs(int(count))
    if lang != "ru":
        return "one" if number == 1 else "many"
    tail = number % 100
    if 11 <= tail <= 14:
        return "many"
    last = number % 10
    if last == 1:
        return "one"
    if 2 <= last <= 4:
        return "few"
    return "many"


def text(key: str, language: str | None = None, **values: object) -> str:
    """One catalog line. ``language`` forces a locale; otherwise the process locale is used."""
    lang = system_language() if language is None else normalize(language)
    entry = strings().get(key)
    if not isinstance(entry, Mapping) or lang not in entry:
        raise KeyError(key if lang == "en" else f"{key}.{lang}")
    line = str(entry[lang])
    if values:
        return line.format(**values)
    return line
