"""Keep heavy weights in system RAM and copy one model to VRAM per stage.

``voxprint-common`` 0.3.1 has no ``model_park``. Canvas uses 0.5.1, which parks
a state dict in pageable or pinned RAM and copies it with a dtype-preserving
``.to()``. The RAM copy stays for the session. A stage drops only the VRAM copy.

The suite tier comes from ``voxprint_common.ram_probe``. ``ram96`` keeps every
model. ``ram64`` keeps what fits, in order: transformer, text encoder, VAE,
writer, then the checks. ``ram32`` reads each model from disk when a stage
needs it.
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, assert_never

from voxprint_common.model_park import ModelPark, reset_process_park
from voxprint_common.ram_probe import RamTier, ram_tier, tier_for_gib

from canvas.infra import cuda_health, ram_ledger, settings
from canvas.infra.stage_timer import log_detail, note_tag

# Highest priority first. A smaller machine fills this list until the budget is gone.
_ORDER = (
    "transformer",
    "text_encoder",
    "vae",
    "writer",
    "prompt_check",
    "gate_siglip",
    "gate_fairface",
    "gate_regressor",
    "gate_nsfw",
    "controlnet",
)
_MAIN = frozenset({"transformer", "text_encoder", "vae", "writer"})
_GIB = 1024**3
_RESERVE = 12 * _GIB
# Free RAM left after a park. A second Canvas job on another card shares the RAM.
_FREE_FLOOR = 8 * _GIB

# RAM the picture models take once parked, from the Z-Image-Turbo bf16 parks on the 4090 test machine.
_FOOTPRINT = {
    "text_encoder": 8_044_936_704,
    "transformer": 12_309_817_472,
    "vae": 167_639_366,
}
# What a job uses besides its parked copies: loads from disk, the checks, CUDA host memory.
# A solo ram96 job peaked at 57 GB with about 28 GB parked.
_JOB_RESERVE = 32 * _GIB
_TIER_RANK = {RamTier.RAM32: 0, RamTier.RAM64: 1, RamTier.RAM96: 2}
# A job that does not fit next to another job's plan waits this long, then reads from disk.
_WAIT_LIMIT_S = 300.0
_WAIT_STEP_S = 2.0
_wait_listener: Callable[[str, float], None] | None = None

_cache: dict[str, "_Resident"] = {}
_bundles: dict[str, Any] = {}
_committed = 0
_tier_noted = False
_planned: set[str] = set()
_job: RamTier | None = None
_job_budget: int | None = None
_job_depth = 0
_ssd_logged: set[str] = set()


@dataclass
class _Resident:
    """One model whose RAM tensors are held for the session."""

    name: str
    module: Any
    parked: Any
    state: dict[str, Any]
    nbytes: int
    declared: int
    hooked: bool


def active_tier() -> RamTier:
    """The suite tier for this process.

    ``VOXPRINT_RAM_MODE`` wins. Otherwise the Canvas RAM probe is mapped with
    ``tier_for_gib``, which is the same cutoff ``ram_tier`` uses. A failed
    probe falls through to ``ram_tier``.
    """
    named = _named(os.environ.get("VOXPRINT_RAM_MODE", "").strip().lower())
    if named is not None:
        return named
    chosen = _named(settings.get().placement or "")
    if chosen is not None:
        return chosen
    total = probed_bytes()
    if total > 0:
        return tier_for_gib(total / _GIB)
    return ram_tier()


def keeps_main_tier() -> bool:
    """True when the picture models stay in RAM between runs."""
    tier = job_tier()
    match tier:
        case RamTier.RAM32:
            return False
        case RamTier.RAM64 | RamTier.RAM96:
            return True
        case _ as unreachable:
            assert_never(unreachable)


def note_tier() -> RamTier:
    """Log the tier once per process."""
    global _tier_noted
    tier = job_tier()
    if not _tier_noted:
        _tier_noted = True
        note_tag("ram", tier.value)
        log_detail("hot", "tier", "cpu", f"tier={tier.value}")
    return tier


def job_tier() -> RamTier:
    """The tier this job runs at: the suite tier, lowered when free RAM was short at the start."""
    wanted = active_tier()
    if _job is None:
        return plan_job()
    return _job if _TIER_RANK[_job] < _TIER_RANK[wanted] else wanted


def begin_job() -> RamTier:
    """Plan a new job, unless a ``job_scope`` already planned the one this is part of."""
    if _job_depth > 0 and _job is not None:
        return job_tier()
    return plan_job(wait=True)


@contextmanager
def job_scope() -> Iterator[RamTier]:
    """One RAM plan for everything inside, such as every entry of a batch."""
    global _job_depth
    tier = plan_job(wait=True) if _job_depth == 0 else job_tier()
    _job_depth += 1
    try:
        yield tier
    finally:
        _job_depth -= 1


def set_wait_listener(listener: Callable[[str, float], None] | None) -> Callable[[str, float], None] | None:
    """Call ``listener(stage, seconds_left)`` while a job waits for RAM. Returns the previous one.

    The stage is ``waiting_gpu``. A listener may raise to stop waiting, for
    example when the job was cancelled; the error reaches the caller of
    :func:`begin_job`.
    """
    global _wait_listener
    previous = _wait_listener
    _wait_listener = listener
    return previous


def plan_job(*, wait: bool = False) -> RamTier:
    """Choose the tier for the whole job from free RAM now.

    The picture models not parked yet, plus ``_JOB_RESERVE`` and ``_FREE_FLOOR``,
    must fit in free RAM minus what other Canvas jobs still expect to take
    (``ram_ledger``). Otherwise the job keeps what fits, highest priority first
    (``ram64``), or reads every model from disk (``ram32``). Unknown free RAM
    keeps the suite tier.

    Reading the ledger and publishing this plan happen under one
    cross-process lock. With ``wait``, a job for which even ``ram32`` does not
    fit next to the other jobs waits up to ``_wait_limit()`` seconds for them
    to finish, then runs at ``ram32``.
    """
    deadline: float | None = None
    while True:
        with ram_ledger.plan_lock():
            free = free_bytes()
            others = ram_ledger.others_pending()
            starved = _starved(free, others)
            if not (wait and starved) or (deadline is not None and _clock() >= deadline):
                return _commit_plan(free, others, waited=deadline is not None)
        now = _clock()
        if deadline is None:
            deadline = now + _wait_limit()
            log_detail("hot", "wait", "cpu", f"pid={os.getpid()} free={free} others={others}")
        left = max(0.0, deadline - now)
        if _wait_listener is not None:
            _wait_listener("waiting_gpu", left)
        _sleep(min(_WAIT_STEP_S, left) if left > 0 else 0.0)


def _starved(free: int, others: int) -> bool:
    """True when another job's plan leaves no room for this job's own working set."""
    if free <= 0 or others <= 0:
        return False
    return free - others - _JOB_RESERVE - _FREE_FLOOR < 0


def _commit_plan(free: int, others: int, *, waited: bool) -> RamTier:
    global _job, _job_budget
    wanted = active_tier()
    pending = sum(size for name, size in _FOOTPRINT.items() if name not in _cache)
    tier = wanted
    budget: int | None = None
    if free > 0 and wanted is not RamTier.RAM32:
        room = free - others - _JOB_RESERVE - _FREE_FLOOR
        if room < pending:
            first = 0 if "transformer" in _cache else _FOOTPRINT["transformer"]
            tier = RamTier.RAM64 if room >= first else RamTier.RAM32
            budget = _committed + max(0, room)
    _job, _job_budget = tier, budget
    keep = 0 if tier is RamTier.RAM32 else min(pending, budget if budget is not None else pending)
    ram_ledger.publish(ram_ledger.process_bytes(os.getpid()) + keep + _JOB_RESERVE)
    log_detail(
        "hot", "job", "cpu",
        f"pid={os.getpid()} tier={tier.value} wanted={wanted.value} free={free} others={others} need={pending}"
        + (" waited=1" if waited else ""),
    )
    return tier


def _wait_limit() -> float:
    """Seconds a job waits for RAM held by other jobs. ``VOXPRINT_RAM_WAIT_S`` overrides."""
    raw = os.environ.get("VOXPRINT_RAM_WAIT_S", "").strip()
    try:
        return max(0.0, float(raw)) if raw else _WAIT_LIMIT_S
    except ValueError:
        return _WAIT_LIMIT_S


def _clock() -> float:
    return time.monotonic()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def probed_bytes() -> int:
    """Installed RAM. Tests replace ``system_ram_bytes`` and this follows."""
    from canvas.models.runtime import system_ram_bytes

    return system_ram_bytes()


def cached(name: str) -> Any | None:
    """The module kept for ``name``, or None when this tier did not keep it."""
    resident = _cache.get(name)
    if resident is None:
        return None
    return resident.module


def bundle(name: str) -> Any | None:
    """An extra object stored with ``name``, such as a tokenizer."""
    return _bundles.get(name)


def remember_bundle(name: str, value: Any) -> None:
    """Keep ``value`` next to a parked model."""
    if name in _cache:
        _bundles[name] = value


def retained(name: str) -> bool:
    """True when ``name`` has a live RAM copy."""
    return name in _cache


def remember(name: str, module: Any, *, size_bytes: int | None = None) -> bool:
    """Park ``module`` when this tier has room. The dtype is left as it is.

    Returns True when the RAM copy will outlive this call. A model that does
    not fit is left with the caller and logged as disk.
    """
    if name in _cache:
        return True
    measured = _module_bytes(module)
    declared = measured if size_bytes is None else max(0, int(size_bytes))
    if not _seat(name, declared):
        _log_plan(name, "disk", declared)
        return False
    if not _fits_now(declared):
        _log_plan(name, "disk", declared)
        log_detail("hot", "skip", "cpu", f"model={_token(name)} reason=free_ram bytes={declared}")
        return False
    _refuse_meta(module)
    started = time.perf_counter()
    parked = None
    try:
        state = _cpu_state(module)
        parked = _park_state(name, state, declared)
        view = _cpu_view(parked, state)
    except (RuntimeError, MemoryError, OSError) as exc:
        if parked is not None:
            _release_quietly(parked)
        _log_plan(name, "disk", declared)
        log_detail("hot", "skip", "cpu", f"model={_token(name)} reason={type(exc).__name__}")
        return False
    hooked = _hooked(module)
    if not hooked:
        _assign(module, view)
    resident = _Resident(name, module, parked, view, measured, declared, hooked)
    _cache[name] = resident
    global _committed
    _committed += declared
    _log_plan(name, "ram", declared)
    _log_move("park", name, "cpu", measured, started, _pin_note(view))
    return True


def open_module(name: str, loader: Callable[[], Any], *, size_bytes: int | None = None) -> Any:
    """Return ``name``, calling ``loader`` only when the RAM copy is absent."""
    found = cached(name)
    if found is not None:
        return found
    module = loader()
    remember(name, module, size_bytes=size_bytes)
    return module


def activate(name: str, device: str, *, blocking: bool = False) -> bool:
    """Copy the RAM weights onto ``device``. The RAM tensors stay put.

    ``blocking`` waits for the copy. The VAE uses it. A non-blocking copy of a
    pinned CPU tensor stays in flight, and the next ``.cpu()`` on the decoder
    output was where that fault surfaced.
    """
    resident = _cache.get(name)
    if resident is None or resident.hooked or cuda_health.lost():
        return False
    kind = "cuda" if "cuda" in str(device) else "cpu"
    if kind == "cpu":
        deactivate(name)
        return True
    started = time.perf_counter()
    moved = resident.parked.to(device, non_blocking=not blocking)
    _assign(resident.module, moved if isinstance(moved, dict) else {})
    # The park dict can omit a buffer. Anything still on CPU or meta is moved
    # here so a later forward cannot keep running on the host.
    _force_device(resident.module, device, blocking=blocking)
    _log_move("unpark", name, kind, resident.nbytes, started)
    return True


def deactivate(name: str) -> None:
    """Drop the VRAM copy and point the module at the RAM tensors again.

    Nothing moves once the CUDA context is lost: dropping those tensors would free them.
    """
    resident = _cache.get(name)
    if resident is None or resident.hooked or cuda_health.lost():
        return
    started = time.perf_counter()
    _assign(resident.module, _cpu_view(resident.parked, resident.state))
    _empty_cuda()
    _log_move("park", name, "cpu", resident.nbytes, started)


def park_loaded(*, force: bool = False) -> tuple[str, ...]:
    """Drop VRAM copies. RAM copies stay.

    ``force`` parks every cached model. Otherwise only models whose tensors
    are on CUDA are parked, so a stage boundary does not log a second park
    for a model that is already in RAM.
    """
    if force:
        names = tuple(_cache)
    else:
        names = tuple(name for name, _size in cuda_residents())
    for name in names:
        deactivate(name)
    return names


def cuda_residents() -> tuple[tuple[str, int], ...]:
    """Cached models that still have a tensor on CUDA, with their byte sizes."""
    found: list[tuple[str, int]] = []
    for name, resident in _cache.items():
        if _module_on_cuda(resident.module):
            found.append((name, resident.nbytes))
    return tuple(found)


def note_ssd(name: str, module: Any, started: float) -> None:
    """Log one session's SSD read. A later open of ``name`` does not log again."""
    if name in _ssd_logged:
        return
    _ssd_logged.add(name)
    elapsed = max(0, int((time.perf_counter() - started) * 1000))
    log_detail(
        "load", "ssd", "cpu",
        f"model={_token(name)} ms={elapsed} bytes={_module_bytes(module)}",
    )


def forget(*names: str) -> None:
    """Drop these RAM copies. Other parked models are pinned again."""
    global _committed
    if not names:
        return
    for name in names:
        resident = _cache.pop(name, None)
        _bundles.pop(name, None)
        _planned.discard(name)
        if resident is None:
            continue
        _committed = max(0, _committed - resident.declared)
        if resident.parked is not None:
            _release_quietly(resident.parked)
    _rebuild()


def reset() -> None:
    """Drop every RAM copy. Tests call this between cases."""
    global _committed, _tier_noted, _job, _job_budget, _job_depth
    for resident in _cache.values():
        if resident.parked is not None:
            _release_quietly(resident.parked)
    _cache.clear()
    _bundles.clear()
    _planned.clear()
    _ssd_logged.clear()
    _committed = 0
    _tier_noted = False
    _job = None
    _job_budget = None
    _job_depth = 0
    reset_process_park()


def plan(sizes: dict[str, int]) -> dict[str, str]:
    """``ram`` or ``disk`` for each name, in priority order, for the current tier."""
    tier = job_tier()
    match tier:
        case RamTier.RAM32:
            return {name: "disk" for name in sizes}
        case RamTier.RAM96:
            return {name: "ram" for name in sizes}
        case RamTier.RAM64:
            return _fill(sizes, _keep_budget())
        case _ as unreachable:
            assert_never(unreachable)


def _fill(sizes: dict[str, int], budget: int) -> dict[str, str]:
    decision = {name: "disk" for name in sizes}
    used = 0
    for name in sorted(sizes, key=lambda item: (_rank(item), item)):
        need = max(0, int(sizes[name]))
        if used + need <= budget:
            decision[name] = "ram"
            used += need
    return decision


def _seat(name: str, size: int) -> bool:
    tier = job_tier()
    match tier:
        case RamTier.RAM32:
            return False
        case RamTier.RAM96:
            return True
        case RamTier.RAM64:
            higher = sum(
                resident.declared
                for resident in _cache.values()
                if _rank(resident.name) < _rank(name)
            )
            if name in _MAIN:
                return higher + size <= _keep_budget()
            return _committed + size <= _keep_budget()
        case _ as unreachable:
            assert_never(unreachable)


def _keep_budget() -> int:
    total = probed_bytes()
    if total <= 0:
        total = int(ram_tier_total())
    budget = max(0, total - _RESERVE)
    return budget if _job_budget is None else min(budget, _job_budget)


def ram_tier_total() -> float:
    """Installed bytes when the Canvas probe is empty, else that probe."""
    total = probed_bytes()
    if total > 0:
        return float(total)
    from voxprint_common.ram_probe import total_gib

    return total_gib() * _GIB


def _park_state(name: str, state: dict[str, Any], declared: int) -> Any:
    park = _park()
    role = "main" if name in _MAIN else "aux"
    snapshot = dict(state)

    def load() -> dict[str, Any]:
        return snapshot

    return park.get(name, load, size_bytes=max(declared, 1), role=role)


def _park() -> ModelPark:
    tier = job_tier()
    park = ModelPark.for_process(tier, app="voxprint-canvas")
    if park.tier is not tier:
        reset_process_park()
        park = ModelPark.for_process(tier, app="voxprint-canvas")
    return park


def _rebuild() -> None:
    """Pin the models that are still cached after a forget."""
    survivors = list(_cache.values())
    reset_process_park()
    for resident in survivors:
        try:
            resident.parked = _park_state(resident.name, resident.state, resident.declared)
        except (RuntimeError, MemoryError, OSError) as exc:
            global _committed
            _cache.pop(resident.name, None)
            _committed = max(0, _committed - resident.declared)
            log_detail("hot", "skip", "cpu", f"model={_token(resident.name)} reason={type(exc).__name__}")
            if not resident.hooked:
                _assign(resident.module, resident.state)
            continue
        if not resident.hooked:
            _assign(resident.module, _cpu_view(resident.parked, resident.state))


def free_bytes() -> int:
    """Available system RAM right now. 0 when it cannot be read."""
    try:
        import psutil

        return max(0, int(psutil.virtual_memory().available))
    except (ImportError, OSError, AttributeError, ValueError):
        return 0


def _fits_now(size: int) -> bool:
    """True when parking ``size`` bytes still leaves ``_FREE_FLOOR`` free. Unknown free RAM fits.

    RAM that other Canvas jobs still expect to take counts as used.
    """
    free = free_bytes()
    if free <= 0:
        return True
    return free - ram_ledger.others_pending() - max(0, int(size)) >= _FREE_FLOOR


def _release_quietly(parked: Any) -> None:
    release = getattr(parked, "release", None)
    if not callable(release):
        return
    try:
        release()
    except (RuntimeError, MemoryError, OSError):
        return


def _cpu_view(parked: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        moved = parked.to("cpu", non_blocking=False)
    except (RuntimeError, MemoryError, OSError):
        return fallback
    if isinstance(moved, dict) and moved:
        return moved
    return fallback


def _cpu_state(module: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key, tensor in _tensors(module):
        state[key] = _cpu_copy(tensor)
    return state


def _tensors(module: Any) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    for kind, getter in (("p", "named_parameters"), ("b", "named_buffers")):
        method = getattr(module, getter, None)
        if not callable(method):
            continue
        try:
            pairs = list(method())
        except (TypeError, RuntimeError):
            continue
        for name, tensor in pairs:
            found.append((f"{kind}:{name}", tensor))
    return found


def _refuse_meta(module: Any) -> None:
    """Park only a materialized module. A meta tensor has no storage to copy."""
    for _key, tensor in _tensors(module):
        if _tensor_kind(tensor) == "meta" or _tensor_kind(getattr(tensor, "data", None)) == "meta":
            raise RuntimeError("refusing to park a meta tensor")


def _force_device(module: Any, device: str, *, blocking: bool) -> None:
    """Move any parameter or buffer that did not land on ``device``."""
    kind = "cuda" if "cuda" in str(device) else "cpu"
    for _key, tensor in _tensors(module):
        if _tensor_kind(tensor) == kind or _tensor_kind(getattr(tensor, "data", None)) == kind:
            continue
        data = getattr(tensor, "data", None)
        target = data if data is not None else tensor
        mover = getattr(target, "to", None)
        if not callable(mover):
            continue
        try:
            moved = mover(device, non_blocking=not blocking)
        except TypeError:
            try:
                moved = mover(device)
            except (TypeError, RuntimeError, AttributeError):
                continue
        except (RuntimeError, AttributeError):
            continue
        if data is None:
            continue
        try:
            tensor.data = moved
        except (AttributeError, TypeError, RuntimeError):
            continue


def _tensor_kind(tensor: Any) -> str:
    if tensor is None:
        return ""
    device = getattr(tensor, "device", None)
    if device is None:
        return ""
    kind = getattr(device, "type", None)
    if isinstance(kind, str):
        return kind
    text = str(device)
    if "cuda" in text:
        return "cuda"
    if "meta" in text:
        return "meta"
    if "cpu" in text:
        return "cpu"
    return text


def _assign(module: Any, state: dict[str, Any]) -> None:
    if not state:
        return
    for kind, getter in (("p", "named_parameters"), ("b", "named_buffers")):
        method = getattr(module, getter, None)
        if not callable(method):
            continue
        try:
            pairs = list(method())
        except (TypeError, RuntimeError):
            continue
        for name, tensor in pairs:
            replacement = state.get(f"{kind}:{name}")
            if replacement is None:
                continue
            try:
                tensor.data = replacement
            except (AttributeError, TypeError, RuntimeError):
                continue


def _cpu_copy(tensor: Any) -> Any:
    """A pageable CPU copy of ``tensor``.

    Pinning is left to the park, which keeps every process inside the shared
    pin budget. Pinning a CUDA tensor here skipped that budget: with two jobs
    the host allocation failed and the CUDA context could not be used again.
    """
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        try:
            tensor = detach()
        except (TypeError, RuntimeError):
            pass
    mover = getattr(tensor, "to", None)
    if not callable(mover):
        return tensor
    try:
        return mover("cpu")
    except (TypeError, RuntimeError, AttributeError):
        return tensor


def _pin_note(state: dict[str, Any]) -> str:
    """``pinned=1`` when every parked tensor is pinned."""
    if not state:
        return "pinned=0"
    pinned = 0
    for tensor in state.values():
        probe = getattr(tensor, "is_pinned", None)
        if callable(probe):
            try:
                if probe():
                    pinned += 1
                    continue
            except (TypeError, RuntimeError):
                continue
        if getattr(tensor, "pinned", False):
            pinned += 1
    return "pinned=1" if pinned == len(state) else "pinned=0"


def _module_bytes(module: Any) -> int:
    total = 0
    for _key, tensor in _tensors(module):
        total += _nbytes(tensor)
    return total


def _nbytes(tensor: Any) -> int:
    value = getattr(tensor, "nbytes", 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _hooked(module: Any) -> bool:
    return getattr(module, "_hf_hook", None) is not None


def _module_on_cuda(module: Any) -> bool:
    for _key, tensor in _tensors(module):
        if _tensor_on_cuda(tensor) or _tensor_on_cuda(getattr(tensor, "data", None)):
            return True
    return False


def _tensor_on_cuda(tensor: Any) -> bool:
    if tensor is None:
        return False
    device = getattr(tensor, "device", None)
    if device is None:
        return False
    kind = getattr(device, "type", None)
    text = kind if isinstance(kind, str) else str(device)
    return "cuda" in text


def _rank(name: str) -> int:
    try:
        return _ORDER.index(name)
    except ValueError:
        return len(_ORDER)


def _named(value: str) -> RamTier | None:
    match value:
        case "ram32":
            return RamTier.RAM32
        case "ram64":
            return RamTier.RAM64
        case "ram96":
            return RamTier.RAM96
        case _:
            return None


def _log_plan(name: str, where: str, size: int) -> None:
    if name in _planned:
        return
    _planned.add(name)
    safe = _token(name)
    place = where if where in {"ram", "disk"} else "disk"
    log_detail("hot", "plan", "cpu", f"model={safe} where={place} bytes={max(0, int(size))}")


def _log_move(kind: str, name: str, device: str, nbytes: int, started: float, extra: str = "") -> None:
    elapsed = max(0, int((time.perf_counter() - started) * 1000))
    suffix = f"ms={elapsed} bytes={max(0, int(nbytes))}"
    if extra:
        suffix = f"{suffix} {extra}"
    log_detail(kind, _token(name), device, suffix)


def _token(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name.lower())
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "model"
    return cleaned[:32]


def _empty_cuda() -> None:
    if cuda_health.lost():
        return
    try:
        import torch
    except ImportError:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    try:
        if cuda.is_available():
            cuda.empty_cache()
    except (TypeError, RuntimeError, AttributeError) as exc:
        if cuda_health.is_cuda_fault(exc):
            cuda_health.mark_lost(exc)
        return
