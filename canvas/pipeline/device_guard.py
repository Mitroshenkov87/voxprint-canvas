"""Stop a CUDA machine that left the writer or the text encoder on the CPU.

A check model is not covered here. Its CPU fallback stays where it is.
"""
from __future__ import annotations

from typing import Any

from canvas.app.i18n import text
from canvas.infra.errors import note_error


class DeviceRefused(Exception):
    """The writer or the text encoder did not stay on the GPU."""

    code = "device.cpu"

    def __init__(self) -> None:
        super().__init__(text("error.device"))

    def __str__(self) -> str:
        return text("error.device")


def require_cuda(module: Any, reason: str, torch: Any | None = None) -> None:
    """Stop when CUDA is up and a parameter is not on it.

    A module with no parameters is left alone. The unit-test doubles are built
    that way and never claim a device.
    """
    if not _cuda_up(torch):
        return
    parameters = _parameters(module)
    if not parameters:
        return
    for param in parameters:
        if _device_type(param) != "cuda":
            note_error(reason)
            raise DeviceRefused()


def weights_on_cuda(module: Any) -> bool:
    """True when every parameter already lives on CUDA.

    A module with no parameters is not on CUDA. Callers then use ``to``, which
    is how a test double records the move.
    """
    parameters = _parameters(module)
    if not parameters:
        return False
    return all(_device_type(param) == "cuda" for param in parameters)


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


def place_on_cuda(module: Any) -> Any:
    """Move parameters onto CUDA one tensor at a time.

    Copying the whole module with ``to`` keeps the old storage alive until the
    copy finishes, so a model that would fit is asked for twice the memory.
    A module with no parameters uses ``to`` so a test double still records it.
    """
    parameters = _parameters(module)
    if not parameters:
        mover = getattr(module, "to", None)
        if callable(mover):
            return mover("cuda")
        return module
    seen: set[int] = set()
    for param in parameters:
        data = getattr(param, "data", None)
        target = data if data is not None else param
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
        mover = getattr(target, "to", None)
        if not callable(mover):
            continue
        moved = mover("cuda")
        if data is not None:
            try:
                param.data = moved
            except (AttributeError, TypeError, RuntimeError):
                continue
    return module


def _parameters(module: Any) -> list[Any]:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return []
    try:
        return list(parameters())
    except (TypeError, RuntimeError):
        return []


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
