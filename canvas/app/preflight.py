"""Fail fast before a heavy model load.

Adoption runs first. Then every picture model and every gate model must already
have a complete view. Free disk, installed RAM, and this check's own time
budget are read without hashing weight bytes. One log line records the result.
"""
from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from voxprint_common.ram_probe import total_gib

from canvas.infra.stage_timer import log_detail
from canvas.models.adopt import adopt_required, gate_specs, required_specs, view_complete
from canvas.models.registry import Catalog, ModelSpec, load_catalog
from canvas.models.snapshots import request_download
from canvas.safety.gate import SafetyUnavailable

# Working space for a photoshoot's frames. The model bytes are already in the blob store.
MIN_FREE_DISK_BYTES = 1 * 1024**3
# A 16 GB RAM stick reports about 14.9 GiB. Below 12 GiB there is no room to start.
MIN_RAM_BYTES = 12 * 1024**3
BUDGET_S = 5.0

Reason = Literal["ok", "view", "gate", "disk", "ram", "budget"]


@dataclass(frozen=True)
class PreflightResult:
    """The one preflight outcome. ``reason`` is ``ok`` when the heavy load may start."""

    ok: bool
    reason: Reason


class PreflightFailed(SafetyUnavailable):
    """Disk, RAM, or the time budget stopped the run before any model was loaded."""

    def __init__(self, reason: str) -> None:
        super().__init__("preflight")
        self.reason = reason

    def __str__(self) -> str:
        return f"preflight fail reason={self.reason}"


def preflight(
    root: Path,
    catalog: Catalog | None = None,
    *,
    budget_s: float = BUDGET_S,
    disk_free: Callable[[Path], int] | None = None,
    ram_bytes: Callable[[], int] | None = None,
    clock: Callable[[], float] | None = None,
) -> PreflightResult:
    """Adopt, then refuse a heavy load that is missing a view, disk, RAM, or time."""
    chosen = catalog or load_catalog()
    adopt_required(root, chosen)
    now = clock or time.monotonic
    started = now()

    def expired() -> bool:
        return now() - started >= budget_s

    if expired():
        return _finish(False, "budget")
    picture = _missing(_pictures(chosen), root)
    if expired():
        return _finish(False, "budget")
    if picture is not None:
        request_download(picture)
        return _finish(False, "view")
    gates = gate_specs(chosen)
    if not gates:
        return _finish(False, "gate")
    gate = _missing(gates, root)
    if expired():
        return _finish(False, "budget")
    if gate is not None:
        request_download(gate)
        return _finish(False, "gate")
    free = _disk(root) if disk_free is None else disk_free(root)
    if expired():
        return _finish(False, "budget")
    if free < MIN_FREE_DISK_BYTES:
        return _finish(False, "disk")
    installed = _ram() if ram_bytes is None else ram_bytes()
    if expired():
        return _finish(False, "budget")
    if installed < MIN_RAM_BYTES:
        return _finish(False, "ram")
    return _finish(True, "ok")


def _pictures(catalog: Catalog) -> tuple[ModelSpec, ...]:
    gates = {spec.id for spec in gate_specs(catalog)}
    return tuple(spec for spec in required_specs(catalog) if spec.id not in gates)


def _missing(specs: tuple[ModelSpec, ...], root: Path) -> str | None:
    for spec in specs:
        if not view_complete(spec, root):
            return spec.id
    return None


def _disk(root: Path) -> int:
    target = root if root.exists() else root.parent
    try:
        return int(shutil.disk_usage(target).free)
    except OSError:
        return 0


def _ram() -> int:
    try:
        return int(total_gib() * (1024**3))
    except (OSError, TypeError, ValueError):
        return 0


def _finish(ok: bool, reason: Reason) -> PreflightResult:
    status = "ok" if ok else "fail"
    log_detail("preflight", "result", "cpu", f"preflight {status} reason={reason}")
    return PreflightResult(ok, reason)
