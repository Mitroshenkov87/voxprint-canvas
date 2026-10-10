"""How much video memory a batch may plan for.

Adapted from Voxprint AI Audiobook Builder ``core/vram_policy.py``
(Apache-2.0, Copyright Aleksandr Mitroshenkov).

    reserve = max(2 GB, 8 % of total)   clamped to 3 GB for this app
    budget  = max(0, free_now - reserve)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from canvas.models.modes import headroom_gb


@dataclass(frozen=True)
class VramPlan:
    batch: int
    budget_gb: float
    free_gb: float
    total_gb: float
    reserve_gb: float


def budget_gb(free_gb: float, total_gb: float) -> float:
    """GB that may still be taken: free memory minus the headroom."""
    free = max(0.0, float(free_gb))
    return max(0.0, free - headroom_gb(total_gb))


def plan(free_gb: float, total_gb: float, per_item_gb: float, max_batch: int) -> VramPlan:
    """Batch size from the memory that is free now. At least one item when the caller asks for a batch."""
    free = max(0.0, float(free_gb))
    total = max(0.0, float(total_gb))
    reserve = headroom_gb(total)
    room = budget_gb(free, total)
    if per_item_gb > 0:
        count = int(math.floor(room / per_item_gb + 1e-9))
    else:
        count = int(max_batch)
    batch = max(1, min(int(max_batch), count)) if max_batch >= 1 else 1
    return VramPlan(batch, room, free, total, reserve)


def halve_on_spill(batch: int) -> int:
    """The suite rule: halve the batch when work spills into system memory. Never below one."""
    return max(1, int(batch) // 2)
