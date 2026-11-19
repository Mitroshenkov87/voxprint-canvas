"""Picture seeds: one random draw per picture, unless a seed is locked or given.

Every route (the window, ``generate``, ``photoshoot``, ``batch`` and
``render_batch``) uses these helpers, so they all follow one rule:

* No seed given and no lock: every picture draws its own fresh seed.
* A seed given or locked: that seed repeats, and a series or a run of copies
  counts up from it (``base + index``).
"""

from __future__ import annotations

import secrets

SEED_MIN = 0
SEED_MAX = 2**32 - 1
# The Expert seed field is a 32-bit signed Qt spin box. It cannot show more.
FIELD_MAX = 2**31 - 1

SEED_MODES = ("random", "locked")


def fresh_seed() -> int:
    """A new random seed in ``0 .. 2**32 - 1``, drawn from the operating system."""
    return secrets.randbelow(SEED_MAX + 1)


def offset_seed(base: int, index: int) -> int:
    """``base + index``, wrapped into the seed range."""
    return (int(base) + int(index)) % (SEED_MAX + 1)


def pick_seed(base: int | None, index: int = 0) -> int:
    """The seed for picture ``index``: counted up from ``base``, or fresh when there is no base."""
    if base is None:
        return fresh_seed()
    return offset_seed(base, index)


def as_seed(value: object) -> int | None:
    """``value`` as a whole seed in range, or None when it is not one."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if SEED_MIN <= value <= SEED_MAX else None


def field_value(seed: int) -> int:
    """What the Expert spin box can show for ``seed``."""
    return min(int(seed), FIELD_MAX)


class SeedPlan:
    """Maps the seed a series plans for each shot onto the seed the shot really uses.

    ``mode`` is ``random`` (a fresh seed per shot, the default when no seed was
    given), ``series`` (the planned seed, ``base + index``) or ``locked`` (the
    base on every shot). The plan remembers each answer, so the sampler and
    the final draw of one shot agree, and the saved seed can be read back.
    """

    def __init__(self, mode: str, base: int) -> None:
        if mode not in ("random", "series", "locked"):
            raise ValueError(f"unknown seed mode {mode!r}")
        self.mode = mode
        self.base = int(base)
        self.used: dict[int, int] = {}

    def seed_for(self, planned: int) -> int:
        """The seed for the shot the series planned as ``planned``."""
        found = self.used.get(planned)
        if found is None:
            if self.mode == "random":
                found = fresh_seed()
            elif self.mode == "locked":
                found = self.base
            else:
                found = int(planned) % (SEED_MAX + 1)
            self.used[planned] = found
        return found

    def used_seed(self, planned: int) -> int:
        """The seed already used for ``planned`` (the planned one when none was drawn yet)."""
        return self.used.get(planned, int(planned))

    def wrap(self, draw):
        """``draw`` with every seed passed through the plan. Hooks on ``draw`` are kept."""

        def drawn(line, seed, hero, strength):
            return draw(line, self.seed_for(seed), hero, strength)

        sample = getattr(draw, "sample", None)
        if callable(sample):
            def sampled(line, seed, hero, strength):
                return sample(line, self.seed_for(seed), hero, strength)

            drawn.sample = sampled
        for name in ("prepare", "lora"):
            if hasattr(draw, name):
                setattr(drawn, name, getattr(draw, name))
        drawn.used_seed = self.used_seed
        return drawn
