"""Cooling pauses after a long job.

Full speed for 2.5 hours. After that, pause while the 5-minute median GPU
temperature is at least 83 C, or while a thermal throttle has lasted more than
60 seconds. Resume at 75 C or below once the throttle has cleared.

The shape follows Voxprint AI Audiobook Builder ``core/gpu_thermal.py`` and the
suite rule (83 C, not the older 87 C threshold).
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

FULL_SPEED_S = 2.5 * 3600
HOT_C = 83.0
COOL_C = 75.0
WINDOW_S = 5 * 60
THROTTLE_HOLD_S = 60.0
MIN_SAMPLES = 3


class Guard:
    """Decides whether the next batch should wait."""

    def __init__(self, now: Callable[[], float] | None = None) -> None:
        self._now = now or time.monotonic
        self.t0 = self._now()
        self.samples: list[tuple[float, float]] = []
        self._throttle_since: float | None = None

    def elapsed(self) -> float:
        return self._now() - self.t0

    def observe(self, temp_c: float | None, throttling: bool) -> None:
        moment = self._now()
        if temp_c is not None:
            self.samples.append((moment, float(temp_c)))
            self.samples = [item for item in self.samples if item[0] >= moment - 15 * 60]
        if throttling:
            if self._throttle_since is None:
                self._throttle_since = moment
        else:
            self._throttle_since = None

    def median_c(self) -> float | None:
        moment = self._now()
        window = [temp for stamp, temp in self.samples if moment - WINDOW_S <= stamp <= moment]
        if len(window) < MIN_SAMPLES:
            return None
        return float(statistics.median(window))

    def throttle_seconds(self) -> float:
        if self._throttle_since is None:
            return 0.0
        return max(0.0, self._now() - self._throttle_since)

    def should_pause(self) -> bool:
        if self.elapsed() < FULL_SPEED_S:
            return False
        median = self.median_c()
        if median is not None and median >= HOT_C:
            return True
        return self.throttle_seconds() > THROTTLE_HOLD_S

    def cool_enough(self) -> bool:
        throttling = self._throttle_since is not None
        median = self.median_c()
        if median is None:
            return not throttling
        return median <= COOL_C and not throttling
