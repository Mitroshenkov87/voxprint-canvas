"""Stop a CUDA machine that left the writer or the text encoder on the CPU.

A check model is not covered here. Its CPU fallback stays where it is.
"""
from __future__ import annotations

import gc
from typing import Any

from canvas.app.i18n import text
from canvas.infra.errors import note_error
from canvas.infra.phase_vram import fold_peak


class DeviceRefused(Exception):
    """The writer or the text encoder did not stay on the GPU."""

    code = "device.cpu"

    def __init__(self) -> None:
        super().__init__(text("error.device"))

    def __str__(self) -> str:
        return text("error.device")


def require_cuda(module: Any, reason: str, torch: Any | None = None) -> None:
    """Stop when CUDA is up and a parameter or buffer is not on it.

    CPU and meta both refuse the load. A module with no parameters and no
    buffers is left alone. The unit-test doubles are built that way and never
    claim a device.
    """
    if not _cuda_up(torch):
        return
    tensors = _resident_tensors(module)
    if not tensors:
        return
    for tensor in tensors:
        if _device_type(tensor) != "cuda":
            note_error(reason)
            raise DeviceRefused()


def require_writer_materialized(module: Any) -> None:
    """Refuse a writer weight that is still meta or fp32.

    ``low_cpu_mem_usage`` leaves parameters on the meta device. Parking one of
    those copies raises ``Cannot copy out of meta tensor``. An fp32 load of
    this 4B model is about four times the bf16 file. Either one stops the load
    before ``model_park`` sees it.
    """
    for tensor in _parameters_only(module):
        if _is_meta(tensor):
            note_error("writer.meta")
            raise DeviceRefused()
        if _is_fp32(tensor):
            note_error("writer.fp32")
            raise DeviceRefused()
    for tensor in _buffers_only(module):
        if _is_meta(tensor):
            note_error("writer.meta")
            raise DeviceRefused()


def weights_on_cuda(module: Any) -> bool:
    """True when every parameter and buffer already lives on CUDA.

    A module with no parameters and no buffers is not on CUDA. Callers then
    use ``to``, which is how a test double records the move.
    """
    tensors = _resident_tensors(module)
    if not tensors:
        return False
    return all(_device_type(tensor) == "cuda" for tensor in tensors)


def pretrained_attempts(dtype: object, device: str) -> tuple[dict[str, Any], ...]:
    """``from_pretrained`` keyword sets, first match wins.

    CUDA uses an explicit device map so safetensors can map the file onto the
    card. ``device_map="auto"`` is not one of them: that map left the weights
    on the CPU. ``low_cpu_mem_usage`` keeps the loader from copying the whole
    file into a second buffer. A loader that rejects ``device_map`` is retried
    without it.
    """
    if device != "cuda" and device != "cpu":
        raise ValueError(f"unknown device {device}")
    common: dict[str, Any] = {"local_files_only": True, "low_cpu_mem_usage": True}
    ordered: list[dict[str, Any]] = []
    if device == "cuda":
        cuda_map = {"": "cuda"}
        ordered.append({**common, "torch_dtype": dtype, "device_map": cuda_map})
        ordered.append({**common, "dtype": dtype, "device_map": cuda_map})
    ordered.append({**common, "torch_dtype": dtype, "device_map": None})
    ordered.append({**common, "dtype": dtype, "device_map": None})
    ordered.append({**common, "torch_dtype": dtype})
    ordered.append({**common, "dtype": dtype})
    return tuple(ordered)


def reclaim_cuda(torch: Any) -> None:
    """Drop a freed reservation and forget the allocator's historical peak.

    A full-module copy can leave a second weight-sized block reserved after
    the live tensors are the only ones still referenced. The 14 GB check reads
    that peak, so it is cleared once the reserve is released.
    """
    gc.collect()
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    try:
        if not bool(cuda.is_available()):
            return
    except (TypeError, RuntimeError):
        return
    empty = getattr(cuda, "empty_cache", None)
    if callable(empty):
        try:
            empty()
        except (TypeError, RuntimeError):
            pass
    reset = getattr(cuda, "reset_peak_memory_stats", None)
    if callable(reset):
        fold_peak()
        try:
            reset()
        except (TypeError, RuntimeError):
            pass


def place_on_cuda(module: Any) -> Any:
    """Move parameters and buffers onto CUDA one tensor at a time.

    Copying the whole module with ``to`` keeps the old storage alive until the
    copy finishes, so a model that would fit is asked for twice the memory.
    A module with no parameters and no buffers uses ``to`` so a test double
    still records it.
    """
    tensors = _resident_tensors(module)
    if not tensors:
        mover = getattr(module, "to", None)
        if callable(mover):
            return mover("cuda")
        return module
    seen: set[int] = set()
    for tensor in tensors:
        data = getattr(tensor, "data", None)
        target = data if data is not None else tensor
        pointer = getattr(target, "data_ptr", None)
        key = 0
        if callable(pointer):
            try:
                key = int(pointer())
            except (TypeError, ValueError, RuntimeError):
                key = 0
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        if _device_type(target) == "cuda":
            continue
        mover = getattr(target, "to", None)
        if not callable(mover):
            continue
        moved = mover("cuda")
        if data is not None:
            try:
                tensor.data = moved
            except (AttributeError, TypeError, RuntimeError):
                continue
    return module


def _is_meta(tensor: Any) -> bool:
    if _device_type(tensor) == "meta":
        return True
    data = getattr(tensor, "data", None)
    return data is not None and data is not tensor and _device_type(data) == "meta"


def _is_fp32(tensor: Any) -> bool:
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        return False
    text = str(dtype).lower().replace("torch.", "")
    return text in {"fp32", "float32"} or text.endswith(".float32")


def _parameters_only(module: Any) -> list[Any]:
    method = getattr(module, "parameters", None)
    if not callable(method):
        return []
    try:
        return list(method())
    except (TypeError, RuntimeError):
        return []


def _buffers_only(module: Any) -> list[Any]:
    method = getattr(module, "buffers", None)
    if not callable(method):
        return []
    try:
        return list(method())
    except (TypeError, RuntimeError):
        return []


def _resident_tensors(module: Any) -> list[Any]:
    """Parameters and buffers. A missing accessor means that group is empty."""
    found: list[Any] = []
    for name in ("parameters", "buffers"):
        method = getattr(module, name, None)
        if not callable(method):
            continue
        try:
            found.extend(list(method()))
        except (TypeError, RuntimeError):
            continue
    return found


def _device_type(value: Any) -> str:
    device = getattr(value, "device", None)
    if device is None:
        data = getattr(value, "data", None)
        if data is not None and data is not value:
            return _device_type(data)
        return ""
    kind = getattr(device, "type", None)
    if isinstance(kind, str):
        return kind
    text_device = str(device)
    if "cuda" in text_device:
        return "cuda"
    if "cpu" in text_device:
        return "cpu"
    return text_device


def _cuda_up(torch: Any | None) -> bool:
    if torch is None:
        # torch is an optional extra. Importing it here keeps Canvas importable
        # in unit tests when the package is absent. This is not a circular import.
        try:
            import torch as imported
        except ImportError:
            return False
        torch = imported
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except (TypeError, RuntimeError):
        return False
