"""Generic error notes for the suite log."""
from __future__ import annotations

import json
import logging
import os
import traceback

from voxprint_common.paths import logs_dir

_LOG = logging.getLogger("voxprint.crash")
if not any(isinstance(handler, logging.NullHandler) for handler in _LOG.handlers):
    _LOG.addHandler(logging.NullHandler())


def note_error(code: str) -> None:
    """Append a reason code to the suite log. The code is the only text written."""
    if not code:
        return
    _LOG.error("%s", code)
    try:
        folder = logs_dir()
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "voxprint.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{code}\n")
    except OSError:
        return


def note_crash(code: str, exc: BaseException) -> None:
    """Append one crash line: a code, the exception type, and the traceback.

    The line is written whether or not debug logging is on. It carries no picture
    text and no request text, only the code, the type name, and the traceback.
    """
    label = code or "internal"
    type_name = type(exc).__name__
    flat = " ".join(_stack_only(exc).split())
    _LOG.error("%s %s", label, type_name)
    try:
        folder = logs_dir()
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "voxprint.log").open("a", encoding="utf-8") as handle:
            handle.write(f"crash {label} {type_name} {flat}\n")
    except OSError:
        return


def _stack_only(exc: BaseException) -> str:
    """The traceback frames and type names of an error chain, without the error text."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            current = None
        else:
            current = current.__context__
    parts: list[str] = []
    for item in reversed(chain):
        parts.append("Traceback (most recent call last):")
        parts.extend(traceback.format_tb(item.__traceback__))
        parts.append(type(item).__name__)
    return "\n".join(parts)


def debug_enabled() -> bool:
    """True when ``VOXPRINT_DEBUG=1`` is set."""
    return os.environ.get("VOXPRINT_DEBUG", "").strip() == "1"


def note_debug(code: str, detail: str) -> None:
    """Append one debug entry when debug logging is on. The caller masks numbers in ``detail``."""
    if not code or not debug_enabled():
        return
    _LOG.debug("%s %s", code, detail)
    try:
        folder = logs_dir()
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "voxprint.log").open("a", encoding="utf-8") as handle:
            handle.write(f"debug {code} {json.dumps(detail, ensure_ascii=False)}\n")
    except OSError:
        return
