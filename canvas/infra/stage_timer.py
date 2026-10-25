"""Per-stage wall-clock timings for one run. Stage names and seconds only.

No prompt text, picture data or check verdicts are stored, so the summary may go
to the ``--json`` report and to the generic log.
"""
from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class StageTimes:
    """Thread-safe totals per stage name."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._started = time.perf_counter()
        self._tags: dict[str, str] = {}

    def tag(self, key: str, value: str) -> None:
        """Record one short code (for example ``tier=gb16``) for the log line."""
        if not _NAME.match(key) or not _NAME.match(value):
            raise ValueError("tags are short lowercase codes")
        with self._lock:
            self._tags[key] = value

    def note_mib(self, key: str, mib: int) -> None:
        """Keep the larger MiB reading. The value is ``mib`` plus the number."""
        if not _NAME.match(key) or mib <= 0:
            raise ValueError("tags are short lowercase codes")
        with self._lock:
            current = self._tags.get(key, "")
            previous = int(current[3:]) if current.startswith("mib") and current[3:].isdigit() else 0
            best = max(previous, mib)
            value = f"mib{best}"
        self.tag(key, value)

    def add(self, stage: str, seconds: float) -> None:
        """Add one sample. ``stage`` must be a short lowercase code."""
        if not _NAME.match(stage):
            raise ValueError("stage names are short lowercase codes")
        with self._lock:
            self._totals[stage] = self._totals.get(stage, 0.0) + max(0.0, float(seconds))
            self._counts[stage] = self._counts.get(stage, 0) + 1

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Time the block under ``stage``, including when it raises."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - started)

    def summary(self) -> dict[str, object]:
        """``{"wall_s": .., "stages": {name: {"count", "total_s", "mean_s"}}}``."""
        with self._lock:
            stages = {
                name: {
                    "count": self._counts[name],
                    "total_s": round(total, 3),
                    "mean_s": round(total / self._counts[name], 3),
                }
                for name, total in sorted(self._totals.items())
            }
        return {"wall_s": round(time.perf_counter() - self._started, 3), "stages": stages}

    def log_line(self) -> str:
        """One generic-log line: ``timing wall=.. name=count/total ...``."""
        data = self.summary()
        stages = data["stages"]
        assert isinstance(stages, dict)
        parts = [f"wall={data['wall_s']}"]
        parts += [f"{name}={row['count']}x{row['total_s']}" for name, row in stages.items()]
        with self._lock:
            parts += [f"{key}={value}" for key, value in sorted(self._tags.items())]
        return "timing " + " ".join(parts)


_current: StageTimes | None = None


def set_current(times: StageTimes | None) -> StageTimes | None:
    """Make ``times`` the run-wide collector for library stages. Returns the previous one."""
    global _current
    previous, _current = _current, times
    return previous


@contextmanager
def timed(stage: str) -> Iterator[None]:
    """Time a library stage into the run-wide collector, when one is set."""
    times = _current
    if times is None:
        yield
        return
    with times.measure(stage):
        yield


def note_stage(stage: str, seconds: float) -> None:
    """Add one sample to the run-wide collector, when one is set."""
    times = _current
    if times is not None:
        times.add(stage, seconds)


def note_tag(key: str, value: str) -> None:
    """Tag the run-wide collector, when one is set."""
    times = _current
    if times is not None:
        times.tag(key, value)


def note_peak_mib(mib: int) -> None:
    """Remember the higher MiB peak on the timing line as ``peak_vram=mibN``."""
    times = _current
    if times is None or isinstance(mib, bool) or not isinstance(mib, int) or mib <= 0:
        return
    times.note_mib("peak_vram", mib)
