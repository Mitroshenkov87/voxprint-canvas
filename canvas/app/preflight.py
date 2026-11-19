"""Fail fast before a heavy model load.

Every pinned picture model and every gate model must already have a complete view.
The scenario writer counts only for a run that uses it (Photoshoot, batch, enhance).
A generator whose own view is incomplete passes with its complete FP32
``<id>-upstream`` view, the same fallback the loader opens.
An unpinned row stays in the required list and is not fetched until its hashes exist.
A legacy folder is not fetched again: the installer adopts it, and a failed
adopt stays a missing view. Free disk, installed RAM, and this check's own
time budget are read without hashing weight bytes. One log line records the result.
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
from canvas.models.registry import Catalog, ModelSpec, load_catalog
from canvas.models.required import WRITER_ID, gate_specs, required_specs, view_complete
from canvas.models.snapshots import SnapshotError, fallback_ready, request_download, weight_dir
from canvas.models.verified import flush_process_cache, process_cache
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
    writer: bool = False,
    budget_s: float = BUDGET_S,
    disk_free: Callable[[Path], int] | None = None,
    ram_bytes: Callable[[], int] | None = None,
    clock: Callable[[], float] | None = None,
) -> PreflightResult:
    """Refuse a heavy load that is missing a view, disk, RAM, or time.

    ``writer`` is True for a run that opens the scenario writer. A plain prompt
    run neither requires nor queues it.
    """
    chosen = catalog or load_catalog()
    now = clock or time.monotonic
    started = now()

    def expired() -> bool:
        return now() - started >= budget_s

    if expired():
        return _finish(False, "budget")
    picture = _missing(_pictures(chosen, writer=writer), root, fallback=True)
    if expired():
        return _finish(False, "budget")
    if picture is not None:
        _enqueue(picture, root)
        return _finish(False, "view")
    gates = gate_specs(chosen)
    if not gates:
        return _finish(False, "gate")
    gate = _missing(gates, root)
    if expired():
        return _finish(False, "budget")
    if gate is not None:
        _enqueue(gate, root)
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


def _pictures(catalog: Catalog, *, writer: bool = False) -> tuple[ModelSpec, ...]:
    gates = {spec.id for spec in gate_specs(catalog)}
    return tuple(
        spec for spec in required_specs(catalog)
        if spec.id not in gates and (writer or spec.id != WRITER_ID)
    )


def _enqueue(spec: ModelSpec, root: Path) -> None:
    """Queue a fetch only when no legacy folder is already on disk.

    A legacy directory means adopt did not finish that model. Downloading it
    would write a second copy beside the folder the installer left in place.
    """
    if weight_dir(root, spec.repo).is_dir():
        return
    request_download(spec.id)


def _missing(specs: tuple[ModelSpec, ...], root: Path, *, fallback: bool = False) -> ModelSpec | None:
    """The first pinned row whose view is incomplete. Unpinned rows are not fetched.

    With ``fallback`` a row that has an FP32 ``<id>-upstream`` entry also passes
    when that view is complete. Gate rows are checked without it.
    """
    for spec in specs:
        if not spec.pinned():
            continue
        if view_complete(spec, root):
            continue
        if fallback and _fallback_complete(spec, root):
            continue
        if _adopt(spec) and view_complete(spec, root):
            continue
        return spec
    return None


def _adopt(spec: ModelSpec) -> bool:
    """Let VOXPrint MDL (voxprint-common 0.7.3) adopt a complete copy already on disk.

    ``repair_store`` links another app's identical copy into the shared store and
    rebuilds this model's view. It never downloads or deletes. False when it is
    unavailable or the bytes are not on this machine; the row then stays missing.
    """
    try:
        from voxprint_common.mdl.repair_store import repair_store

        from canvas.models.mdl_registry import manifest_path
    except ImportError:
        return False
    try:
        report = repair_store([manifest_path()], only=(spec.id,), include_groups=())
    except Exception:  # noqa: BLE001 - a failed adopt only leaves the row missing
        return False
    if report.adopted or report.repaired_views:
        log_detail("preflight", "adopt", "cpu", f"id={spec.id} adopted={report.adopted} views={report.repaired_views}")
    return bool(report.ready)


def _fallback_complete(spec: ModelSpec, root: Path) -> bool:
    """True when the loader's FP32 fallback or a legacy repack view opens with every file matching its pin.

    This is ``fallback_ready``, the call the loader makes. A blob whose recorded
    size and time match its hash is not read again.
    """
    if not spec.fallback_files and not spec.legacy_files:
        return False
    try:
        return fallback_ready(spec, root, process_cache()) is not None
    except (OSError, SnapshotError):
        return False
    finally:
        flush_process_cache()


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
