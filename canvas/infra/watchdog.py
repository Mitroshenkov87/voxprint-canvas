"""Stall watchdog for a render stage.

A stage that makes no progress for ``STALL_S`` seconds is a hang: the process is
alive, the card is idle and nothing arrives. Python cannot interrupt a native
call, so the watchdog ends the process. It logs the reason, runs the registered
abort hooks (the batch writes its summary there), and exits with ``EXIT_STALLED``.
The picture of a stalled stage is never saved, because the process never gets
to the save.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from canvas.infra.errors import note_error

STALL_S = 900.0
EXIT_STALLED = 3
CODE = "stage_timeout"

_hooks: list[Callable[[str], None]] = []


class Progress:
    """Marks life signs for one watched stage."""

    def __init__(self) -> None:
        self._last = time.monotonic()

    def kick(self) -> None:
        """Record progress, for example one finished sampler step."""
        self._last = time.monotonic()

    def idle(self) -> float:
        """Seconds since the last sign of progress."""
        return time.monotonic() - self._last


def on_abort(hook: Callable[[str], None]) -> Callable[[], None]:
    """Run ``hook(stage)`` before the process exits. Returns a function that removes it."""
    _hooks.append(hook)

    def remove() -> None:
        if hook in _hooks:
            _hooks.remove(hook)

    return remove


def _terminate(code: int) -> None:
    os._exit(code)


def abort(stage: str) -> None:
    """Fail closed: log, run the hooks, and end the process."""
    note_error(f"{CODE} {stage}")
    try:
        print(f"error: stage {stage} made no progress and was stopped ({CODE})", file=sys.stderr, flush=True)
    except (AttributeError, OSError, ValueError):
        pass
    for hook in list(_hooks):
        try:
            hook(stage)
        except Exception:
            continue
    _terminate(EXIT_STALLED)


@contextmanager
def watch(stage: str, stall_s: float | None = None) -> Iterator[Progress]:
    """Watch one stage. The block must call ``kick`` on the yielded object, or finish, within ``stall_s``."""
    limit = STALL_S if stall_s is None else stall_s
    progress = Progress()
    done = threading.Event()

    def guard() -> None:
        while not done.wait(min(5.0, max(0.01, limit / 4))):
            if progress.idle() > limit:
                abort(stage)
                return

    thread = threading.Thread(target=guard, name=f"watch-{stage}", daemon=True)
    thread.start()
    try:
        yield progress
    finally:
        done.set()
        thread.join(timeout=1)
