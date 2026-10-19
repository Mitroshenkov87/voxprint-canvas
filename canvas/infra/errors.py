"""Generic error notes for the suite log."""
from __future__ import annotations

import json
import logging

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


def note_debug(code: str, detail: str) -> None:
    """Append one debug entry: a code and a short text with numbers masked by the caller."""
    if not code:
        return
    _LOG.debug("%s %s", code, detail)
    try:
        folder = logs_dir()
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "voxprint.log").open("a", encoding="utf-8") as handle:
            handle.write(f"debug {code} {json.dumps(detail, ensure_ascii=False)}\n")
    except OSError:
        return
