"""VRAM lines for voxprint.log. Off unless ``VOXPRINT_VRAM_TRACE=1``."""
from __future__ import annotations

import os
from typing import Any

from voxprint_common.paths import logs_dir


def vram_trace_enabled() -> bool:
    """True when this process should write a VRAM line around each load and stage."""
    return os.environ.get("VOXPRINT_VRAM_TRACE", "").strip() == "1"


_holders: list[tuple[str, object]] = []


def hold_model(name: str, module: object) -> None:
    """Remember a model so the trace can name it while it is alive."""
    drop_model(name)
    _holders.append((name, module))


def drop_model(name: str) -> None:
    """Forget a model after it is unloaded."""
    global _holders
    _holders = [item for item in _holders if item[0] != name]


def note_vram(event: str, when: str, extra: object | None = None) -> None:
    """Append one VRAM line: allocations, driver memory, and CUDA-resident models.

    ``event`` and ``when`` are short codes (``pipeline_load``, ``before``). No
    prompt text is written.
    """
    if not vram_trace_enabled():
        return
    if not event or not when:
        return
    allocated, reserved, free, total = _cuda_bytes()
    models = _resident(extra)
    line = (
        f"vram {event} {when} allocated={allocated} reserved={reserved} "
        f"free={free} total={total} models={models}\n"
    )
    try:
        folder = logs_dir()
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "voxprint.log").open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        return


def _cuda_bytes() -> tuple[int, int, int, int]:
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return (0, 0, 0, 0)
    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    free, total = torch.cuda.mem_get_info()
    return allocated, reserved, int(free), int(total)


def _resident(extra: object | None) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    for name, module in _holders:
        _collect(name, module, parts, seen)
    if extra is not None:
        _collect("pipe", extra, parts, seen)
    return ",".join(parts) if parts else "none"


def _collect(name: str, module: object, parts: list[str], seen: set[int]) -> None:
    if module is None or id(module) in seen:
        return
    if _on_cuda(module):
        seen.add(id(module))
        parts.append(f"{_code(name)}:{_nbytes(module)}")
        return
    for child_name, child in _children(module):
        _collect(f"{name}.{child_name}", child, parts, seen)


def _children(module: object) -> list[tuple[str, object]]:
    found: list[tuple[str, object]] = []
    for name in ("text_encoder", "text_encoder_2", "transformer", "vae", "unet"):
        child = getattr(module, name, None)
        if child is not None and child is not module:
            found.append((name, child))
    components = getattr(module, "components", None)
    if isinstance(components, dict):
        for name, child in components.items():
            if child is not None and child is not module:
                found.append((str(name), child))
    return found


def _on_cuda(module: object) -> bool:
    device = getattr(module, "device", None)
    if device is not None and "cuda" in str(getattr(device, "type", device)):
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            return True
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return False
    try:
        for param in parameters():
            device = getattr(param, "device", None)
            if device is not None and getattr(device, "type", None) == "cuda":
                return True
    except (TypeError, RuntimeError):
        return False
    return False


def _nbytes(module: object) -> int:
    total = 0
    for getter in ("parameters", "buffers"):
        items = getattr(module, getter, None)
        if not callable(items):
            continue
        try:
            for tensor in items():
                total += int(getattr(tensor, "nbytes", 0) or 0)
        except (TypeError, RuntimeError):
            continue
    return total


def _code(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._" else "_" for char in name)
    return cleaned[:80] or "model"


def _torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch
