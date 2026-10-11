"""Plan B residency: group offload of transformer blocks, then a sequential fallback.

The text encoder, the generator, and the VAE do not share the GPU. Diffusers
group offload keeps one transformer block group on the device and the rest in
pinned CPU memory. ``use_stream`` overlaps the copy with compute.
``record_stream`` lets the next copy wait for that work. A pipeline that
cannot take the hook uses ``enable_sequential_cpu_offload`` instead.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from canvas.app.i18n import text


class OffloadUnavailable(Exception):
    """The pipeline has neither group offload nor sequential CPU offload."""

    def __init__(self) -> None:
        super().__init__(text("error.offload"))

    def __str__(self) -> str:
        return text("error.offload")


@dataclass(frozen=True)
class OffloadInstall:
    """Which residency hook was installed on one pipeline."""

    kind: str
    use_stream: bool
    record_stream: bool
    pinned: bool
    vae_tiling: bool
    vae_slicing: bool


def install_low_vram_offload(pipe: Any) -> OffloadInstall:
    """Install the 16 GB hooks. Group offload first, sequential CPU offload otherwise.

    VAE tiling and slicing are turned on when the pipeline offers them, so a
    2048 decode does not hold a full frame.
    """
    group = getattr(pipe, "enable_group_offload", None)
    if callable(group):
        try:
            sent = _call_group_offload(group)
        except (TypeError, RuntimeError, AttributeError, ValueError):
            sent = None
        if sent is not None:
            return OffloadInstall(
                "group-offload",
                bool(sent.get("use_stream")),
                bool(sent.get("record_stream")),
                _pin_transformer_blocks(pipe),
                _call_optional(pipe, "enable_vae_tiling"),
                _call_optional(pipe, "enable_vae_slicing"),
            )
    sequential = getattr(pipe, "enable_sequential_cpu_offload", None)
    if not callable(sequential):
        raise OffloadUnavailable()
    sequential()
    return OffloadInstall(
        "sequential-cpu-offload",
        False,
        False,
        False,
        _call_optional(pipe, "enable_vae_tiling"),
        _call_optional(pipe, "enable_vae_slicing"),
    )


def _call_group_offload(fn: Any) -> dict[str, Any]:
    """Pass the stream and pin settings the installed diffusers accepts.

    ``use_stream`` is required. A build that cannot take it is not group offload.
    """
    wanted: dict[str, Any] = {
        "onload_device": _device("cuda"),
        "offload_device": _device("cpu"),
        "offload_type": "block_level",
        "num_blocks_per_group": 1,
        "non_blocking": True,
        "use_stream": True,
        "record_stream": True,
    }
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        params = signature.parameters
        var_keyword = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in params.values())
        if not var_keyword:
            if "use_stream" not in params:
                raise TypeError("group offload has no stream")
            wanted = {key: value for key, value in wanted.items() if key in params}
    fn(**wanted)
    return wanted


def _pin_transformer_blocks(pipe: Any) -> bool:
    """Pin transformer block weights in CPU memory so the H2D copy does not page them."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return False
    blocks = getattr(transformer, "transformer_blocks", None)
    modules: list[Any]
    if isinstance(blocks, (list, tuple)) and blocks:
        modules = list(blocks)
    else:
        modules = [transformer]
    pinned = False
    for module in modules:
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            continue
        for param in parameters():
            data = getattr(param, "data", param)
            pin = getattr(data, "pin_memory", None)
            if not callable(pin):
                continue
            try:
                pinned_data = pin()
            except (RuntimeError, TypeError):
                continue
            if hasattr(param, "data") and pinned_data is not None:
                param.data = pinned_data
            pinned = True
    return pinned


def _call_optional(pipe: Any, name: str) -> bool:
    fn = getattr(pipe, name, None)
    if not callable(fn):
        return False
    fn()
    return True


def _device(name: str) -> Any:
    # torch is an optional extra. A missing install still records the device name.
    try:
        import torch
    except ImportError:
        return name
    return torch.device(name)
