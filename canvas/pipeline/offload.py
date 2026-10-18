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

    Exactly one residency strategy ends up on the pipeline. Diffusers applies
    group offload component by component, so a failure can leave some
    components hooked; those hooks are removed before anything else is tried,
    and sequential CPU offload is never called while group offload is active.
    When pre-pinning every weight fails, group offload is tried once more with
    pinning per transfer. VAE tiling and slicing are turned on when the
    pipeline offers them, so a 2048 decode does not hold a full frame.
    """
    group = getattr(pipe, "enable_group_offload", None)
    failure: BaseException | None = None
    if callable(group):
        for low_ram in (False, True):
            try:
                sent = _call_group_offload(group, low_ram)
            except (TypeError, RuntimeError, AttributeError, ValueError) as exc:
                failure = exc
                if not _clear_group_offload(pipe):
                    raise OffloadUnavailable() from exc
                if low_ram or not isinstance(exc, RuntimeError):
                    break
                continue
            return OffloadInstall(
                "group-offload",
                bool(sent.get("use_stream")),
                bool(sent.get("record_stream")),
                False if low_ram else _pin_transformer_blocks(pipe),
                _call_optional(pipe, "enable_vae_tiling"),
                _call_optional(pipe, "enable_vae_slicing"),
            )
    sequential = getattr(pipe, "enable_sequential_cpu_offload", None)
    if not callable(sequential):
        if failure is not None:
            raise OffloadUnavailable() from failure
        raise OffloadUnavailable()
    if _group_offload_active(pipe):
        raise OffloadUnavailable() from failure
    sequential()
    return OffloadInstall(
        "sequential-cpu-offload",
        False,
        False,
        False,
        _call_optional(pipe, "enable_vae_tiling"),
        _call_optional(pipe, "enable_vae_slicing"),
    )


_GROUP_HOOKS = ("lazy_prefetch_group_offloading", "layer_execution_tracker", "group_offloading")


def _group_offload_active(pipe: Any) -> bool:
    """True when any pipeline component still carries a diffusers group offload hook."""
    check = getattr(pipe, "_maybe_raise_error_if_group_offload_active", None)
    if callable(check):
        return bool(check(raise_error=False))
    return any(_module_hooks(module) for module in _components(pipe))


def _clear_group_offload(pipe: Any) -> bool:
    """Remove group offload hooks a partial install left behind. True when none remain."""
    for component in _components(pipe):
        for registry in _module_hooks(component):
            for name in _GROUP_HOOKS:
                if registry.get_hook(name) is not None:
                    registry.remove_hook(name, recurse=False)
    return not _group_offload_active(pipe)


def _components(pipe: Any) -> list[Any]:
    components = getattr(pipe, "components", None)
    if not isinstance(components, dict):
        return []
    return [item for item in components.values() if callable(getattr(item, "modules", None))]


def _module_hooks(component: Any) -> list[Any]:
    """Hook registries on the component's modules that hold a group offload hook."""
    found = []
    for module in component.modules():
        registry = getattr(module, "_diffusers_hook", None)
        get_hook = getattr(registry, "get_hook", None)
        if not callable(get_hook):
            continue
        if any(get_hook(name) is not None for name in _GROUP_HOOKS):
            found.append(registry)
    return found


def _call_group_offload(fn: Any, low_ram: bool = False) -> dict[str, Any]:
    """Pass the stream and pin settings the installed diffusers accepts.

    ``use_stream`` is required. A build that cannot take it is not group offload.
    ``low_ram`` pins each group when it moves instead of pinning every weight up front.
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
    if low_ram:
        wanted["low_cpu_mem_usage"] = True
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
            if low_ram and "low_cpu_mem_usage" not in params:
                raise TypeError("group offload cannot pin per transfer")
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
