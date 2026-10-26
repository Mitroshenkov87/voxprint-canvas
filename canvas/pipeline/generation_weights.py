"""Resident weights for autoregressive generation.

Diffusers' layerwise FP8 hook upcasts every linear on every token. On a 4B
model that is a ~4 GB cast per token, and the GPU waits on it. A model whose
FP8 size is at most 5 GB fits the 16 GB rule as BF16 (the 14 GB peak, one
heavy model), so those weights stay BF16 on the GPU. A larger model uses a
real FP8 matmul. The layerwise hook is not installed.
"""
from __future__ import annotations

from typing import Any, Literal

from canvas.infra.stage_guard import PEAK_LIMIT_BYTES

# Qwen3-4B is about 4 GB in FP8 and about 8 GB in BF16. BF16 is the generation
# dtype when the FP8 size is at most this and the BF16 weights fit the peak.
FP8_BF16_LIMIT_BYTES = 5 * 1024**3
GenerationDtype = Literal["bf16", "fp8"]


def generation_dtype(bf16_bytes: int) -> GenerationDtype:
    """``bf16`` when that footprint fits, otherwise ``fp8``.

    ``bf16_bytes`` is the weight size at two bytes per parameter. Zero means
    the module has no measurable weights and stays BF16.
    """
    size = max(0, int(bf16_bytes))
    if size <= 0:
        return "bf16"
    fp8_bytes = size // 2
    if fp8_bytes <= FP8_BF16_LIMIT_BYTES and size <= PEAK_LIMIT_BYTES:
        return "bf16"
    return "fp8"


def module_bf16_bytes(module: Any) -> int:
    """Parameter bytes as BF16, whatever dtype is stored now."""
    total = 0
    for param in _parameters(module):
        count = _numel(param)
        if count > 0:
            total += count * 2
            continue
        total += _nbytes(param)
    return total


def prepare_generation(module: Any, torch: Any) -> tuple[GenerationDtype, bool]:
    """Drop offload hooks and keep one resident BF16 copy.

    FP8 is not installed. The per-row torchao path is missing from the pinned
    build, and keeping the original tensor next to a cast copy put a second
    set of weights on the card. The bool is always False.
    """
    del torch
    drop_offload_hooks(module)
    _freeze(module)
    return "bf16", False


def drop_offload_hooks(module: Any) -> None:
    """Remove accelerate and layerwise hooks so a forward does not fetch weights.

    An accelerate hook moves a module per call. A layerwise hook upcasts the
    stored FP8 weight on every token. Both leave the GPU waiting.
    """
    modules = getattr(module, "modules", None)
    targets = list(modules()) if callable(modules) else [module]
    for sub in targets:
        _drop_hf_hook(sub)
        _drop_layerwise(sub)


def install_fp8_matmul(module: Any, torch: Any) -> bool:
    """Store linear weights as FP8 and multiply with ``_scaled_mm``.

    False when this torch has no FP8 matmul, or when the module has no linear
    layers. A failed layer is restored, so a False result leaves the module as
    it was.
    """
    storage = getattr(torch, "float8_e4m3fn", None)
    scaled = getattr(torch, "_scaled_mm", None)
    nn = getattr(torch, "nn", None)
    linear_type = getattr(nn, "Linear", None) if nn is not None else None
    compute = getattr(torch, "bfloat16", None)
    if storage is None or compute is None or not callable(scaled) or linear_type is None:
        return False
    finfo = getattr(torch, "finfo", None)
    if not callable(finfo):
        return False
    try:
        fp8_max = float(finfo(storage).max)
    except (TypeError, ValueError, RuntimeError):
        return False
    if fp8_max <= 0:
        return False
    modules = getattr(module, "modules", None)
    if not callable(modules):
        return False
    linears = [sub for sub in list(modules()) if type(sub) is linear_type]
    if not linears:
        return False
    done: list[Any] = []
    try:
        for sub in linears:
            _convert_linear(sub, torch, storage, compute, scaled, fp8_max)
            done.append(sub)
    except (RuntimeError, TypeError, AttributeError, ValueError):
        for sub in done:
            restore = getattr(sub, "_canvas_fp8_restore", None)
            if callable(restore):
                restore()
        return False
    return True


def dtype_label(module: Any, fallback: str) -> str:
    """Short dtype name from the first parameter, or ``fallback``."""
    for param in _parameters(module):
        label = _dtype_text(getattr(param, "dtype", None))
        if label:
            return label
    if fallback in {"bf16", "fp8", "fp16", "fp32"}:
        return fallback
    return "bf16"


def kv_cache_label(model: Any) -> str:
    """``on`` when generation keeps the KV cache. A config that refuses it is ``off``."""
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        try:
            config.use_cache = True
        except (AttributeError, TypeError):
            pass
        if getattr(config, "use_cache", True) is False:
            return "off"
    gen = getattr(model, "generation_config", None)
    if gen is not None and hasattr(gen, "use_cache"):
        try:
            gen.use_cache = True
        except (AttributeError, TypeError):
            pass
        if getattr(gen, "use_cache", True) is False:
            return "off"
    return "on"


def _convert_linear(
    module: Any, torch: Any, storage: Any, compute: Any, scaled: Any, fp8_max: float,
) -> None:
    weight = getattr(module, "weight", None)
    if weight is None or not hasattr(weight, "data"):
        raise TypeError("linear has no weight")
    original = weight.data
    original_forward = module.forward
    compute_weight = original if _dtype_text(getattr(original, "dtype", None)) == "bf16" else original.to(compute)
    scale = _weight_scale(compute_weight, torch, fp8_max)
    stored = (compute_weight / scale).to(storage)
    weight.data = stored

    def restore() -> None:
        weight.data = original
        module.forward = original_forward

    def forward(x: Any, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return _scaled_linear(x, weight, scale, module, torch, storage, compute, scaled, fp8_max)

    module.forward = forward
    module._canvas_fp8_restore = restore


def _scaled_linear(
    x: Any, weight: Any, scale_w: Any, module: Any, torch: Any, storage: Any, compute: Any, scaled: Any,
    fp8_max: float,
) -> Any:
    leading = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1]).to(compute)
    amax = flat.detach().abs().amax().clamp(min=1e-12)
    scale_a = (amax / fp8_max).to(torch.float32)
    left = (flat / scale_a).to(storage)
    right = weight.data.t()
    out = scaled(left, right, scale_a=scale_a.reshape(1), scale_b=scale_w.reshape(1), out_dtype=compute)
    bias = getattr(module, "bias", None)
    if bias is not None:
        out = out + bias.to(compute)
    return out.reshape(*leading, out.shape[-1])


def _weight_scale(weight: Any, torch: Any, fp8_max: float) -> Any:
    amax = weight.detach().abs().amax().clamp(min=1e-12)
    return (amax / fp8_max).to(torch.float32)


def _materialize_bf16(module: Any, torch: Any) -> None:
    """One BF16 copy of any FP8 parameter. Later tokens read that copy."""
    target = getattr(torch, "bfloat16", None)
    if target is None:
        return
    for param in _parameters(module):
        label = _dtype_text(getattr(param, "dtype", None))
        if label != "fp8":
            continue
        data = getattr(param, "data", None)
        caster = getattr(data, "to", None)
        if not callable(caster):
            continue
        try:
            param.data = caster(dtype=target)
        except TypeError:
            try:
                param.data = caster(target)
            except (TypeError, RuntimeError, AttributeError):
                continue
        except (RuntimeError, AttributeError):
            continue


def _freeze(module: Any) -> None:
    for param in _parameters(module):
        setter = getattr(param, "requires_grad_", None)
        if not callable(setter):
            continue
        try:
            setter(False)
        except (RuntimeError, TypeError, AttributeError):
            continue


def _drop_hf_hook(module: Any) -> None:
    hook = getattr(module, "_hf_hook", None)
    if hook is None:
        return
    detach = getattr(hook, "detach_hook", None)
    if callable(detach):
        try:
            detach(module)
        except (RuntimeError, AttributeError, TypeError):
            pass
    try:
        delattr(module, "_hf_hook")
    except (AttributeError, TypeError):
        return


def _drop_layerwise(module: Any) -> None:
    registry = getattr(module, "_diffusers_hook", None)
    if registry is None:
        return
    hooks = getattr(registry, "hooks", None)
    if not isinstance(hooks, dict) or "layerwise_casting" not in hooks:
        return
    remove = getattr(registry, "remove_hook", None)
    if callable(remove):
        try:
            remove("layerwise_casting")
            return
        except (RuntimeError, AttributeError, TypeError, KeyError):
            pass
    hooks.pop("layerwise_casting", None)
    order = getattr(registry, "_hook_order", None)
    if isinstance(order, list) and "layerwise_casting" in order:
        order.remove("layerwise_casting")


def _parameters(module: Any) -> list[Any]:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return []
    try:
        return list(parameters())
    except (TypeError, RuntimeError):
        return []


def _numel(param: Any) -> int:
    fn = getattr(param, "numel", None)
    if not callable(fn):
        return 0
    try:
        return max(0, int(fn()))
    except (TypeError, ValueError):
        return 0


def _nbytes(param: Any) -> int:
    value = getattr(param, "nbytes", 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _dtype_text(dtype: Any) -> str:
    if dtype is None:
        return ""
    text = str(dtype).lower()
    if "float8" in text or "fp8" in text:
        return "fp8"
    if "bfloat16" in text or text == "bf16":
        return "bf16"
    if "float16" in text or text == "fp16":
        return "fp16"
    if "float32" in text or text == "fp32":
        return "fp32"
    return ""
