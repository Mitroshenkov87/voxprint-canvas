"""UI strings. Every user-visible line lives in ``messages.json``.

The UI is English. Prompts may still be written in any language the
enhancer and the checks read; only the window text is English.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping

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


def plural_form(count: int) -> str:
    """``one`` or ``many`` for a count."""
    return "one" if abs(int(count)) == 1 else "many"


def text(key: str, language: str | None = None, **values: object) -> str:
    """One catalog line. The UI is English; ``language`` is kept for callers that pin it."""
    del language
    entry = strings().get(key)
    if not isinstance(entry, Mapping) or "en" not in entry:
        raise KeyError(key)
    line = str(entry["en"])
    if values:
        return line.format(**values)
    return line
