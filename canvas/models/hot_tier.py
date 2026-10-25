"""Keep heavy weights in system RAM and copy one model to VRAM per stage.

``voxprint-common`` 0.3.1 has no ``model_park``. Canvas uses 0.4.2, which parks
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
from dataclasses import dataclass
from typing import Any, Callable, assert_never

from voxprint_common.model_park import ModelPark, reset_process_park
from voxprint_common.ram_probe import RamTier, ram_tier, tier_for_gib

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

_cache: dict[str, "_Resident"] = {}
_bundles: dict[str, Any] = {}
_committed = 0
_tier_noted = False
_planned: set[str] = set()
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
    total = probed_bytes()
    if total > 0:
        return tier_for_gib(total / _GIB)
    return ram_tier()


def keeps_main_tier() -> bool:
    """True when the picture models stay in RAM between runs."""
    tier = active_tier()
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
    tier = active_tier()
    if not _tier_noted:
        _tier_noted = True
        note_tag("ram", tier.value)
        log_detail("hot", "tier", "cpu", f"tier={tier.value}")
    return tier


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
    started = time.perf_counter()
    state = _cpu_state(module)
    parked = _park_state(name, state, declared)
    view = _cpu_view(parked, state)
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
    if resident is None or resident.hooked:
        return False
    kind = "cuda" if "cuda" in str(device) else "cpu"
    if kind == "cpu":
        deactivate(name)
        return True
    started = time.perf_counter()
    moved = resident.parked.to(device, non_blocking=not blocking)
    _assign(resident.module, moved if isinstance(moved, dict) else {})
    _log_move("unpark", name, kind, resident.nbytes, started)
    return True


def deactivate(name: str) -> None:
    """Drop the VRAM copy and point the module at the RAM tensors again."""
    resident = _cache.get(name)
    if resident is None or resident.hooked:
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
            resident.parked.release()
    _rebuild()


def reset() -> None:
    """Drop every RAM copy. Tests call this between cases."""
    global _committed, _tier_noted
    for resident in _cache.values():
        if resident.parked is not None:
            resident.parked.release()
    _cache.clear()
    _bundles.clear()
    _planned.clear()
    _ssd_logged.clear()
    _committed = 0
    _tier_noted = False
    reset_process_park()


def plan(sizes: dict[str, int]) -> dict[str, str]:
    """``ram`` or ``disk`` for each name, in priority order, for the current tier."""
    tier = active_tier()
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
    tier = active_tier()
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
    return max(0, total - _RESERVE)


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
    tier = active_tier()
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
        resident.parked = _park_state(resident.name, resident.state, resident.declared)
        if not resident.hooked:
            _assign(resident.module, _cpu_view(resident.parked, resident.state))


def _cpu_view(parked: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    moved = parked.to("cpu", non_blocking=False)
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
    """A CPU copy of ``tensor``.

    A CUDA tensor is copied once into pinned host memory. ``.to("cpu")`` lands
    in pageable memory, and the park then pins that copy, so the first park of
    the transformer was a slow transfer plus a second full copy.
    """
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        try:
            tensor = detach()
        except (TypeError, RuntimeError):
            pass
    if _tensor_on_cuda(tensor):
        pinned = _pinned_from_cuda(tensor)
        if pinned is not None:
            return pinned
    mover = getattr(tensor, "to", None)
    if not callable(mover):
        return tensor
    try:
        return mover("cpu")
    except (TypeError, RuntimeError, AttributeError):
        return tensor


def _pinned_from_cuda(tensor: Any) -> Any | None:
    """One pinned CPU copy of a CUDA tensor. None when torch cannot pin it.

    Torch is optional. The import stays inside this function so the park loads
    when the package is absent. This is not a circular import.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        pinned = torch.empty_like(tensor, device="cpu", pin_memory=True)
        pinned.copy_(tensor)
    except (TypeError, RuntimeError, AttributeError, ValueError):
        return None
    return pinned


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
    except (TypeError, RuntimeError, AttributeError):
        return
