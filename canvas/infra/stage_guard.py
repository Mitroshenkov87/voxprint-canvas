"""16 GB stage rule: one heavy model, FP8 or a logged fallback, and a clear card between stages.

The check models keep their own weights. Callers unload them before a heavy stage
and load them again afterwards. A stage that would spill does not run.
"""
from __future__ import annotations

import gc
import math
from collections.abc import Callable, Sequence
from typing import Any, Literal, assert_never

from canvas.infra.errors import note_error
from canvas.infra.vram_trace import drop_model, held_modules, hold_model, on_cuda, park_module, resident_summary
from canvas.models.modes import Tier

# Qwen3-4B is about 8 GB in BF16 and about 4 GB in FP8. The cook budget is the
# FP8 transformer (about 6.15 GB) plus the VAE (about 0.4 GB). The card stops at 14 GB.
PEAK_LIMIT_BYTES = 14 * 1024**3
NEAR_ZERO_BYTES = 512 * 1024**2
ENCODER_BF16_BYTES = 8 * 1024**3
ENCODER_FP8_BYTES = 4 * 1024**3
ENCODER_HEADROOM_BYTES = 1024**3
COOK_BUDGET_BYTES = int(6.15 * 1024**3) + int(0.4 * 1024**3)
QUALITY_COSINE = 0.99

HEAVY_STAGES = frozenset({"text_encoder", "cook", "writer", "enhancer", "captioner"})
HEAVY_MODELS = frozenset({"text_encoder", "transformer", "writer", "enhancer", "captioner"})
Storage = Literal["fp8", "bf16", "cpu"]

_releasers: list[Callable[[], None]] = []
_releasing = False


class StageBlocked(Exception):
    """A 16 GB stage would spill or share the card. ``code`` is the reason."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def register_release(fn: Callable[[], None]) -> None:
    """Remember a callback that drops a heavy model before the next stage."""
    if fn not in _releasers:
        _releasers.append(fn)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine of two equal-length vectors. 0 when a side is empty or the lengths differ."""
    if len(left) != len(right) or not left:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += float(left_value) * float(right_value)
        left_norm += float(left_value) * float(left_value)
        right_norm += float(right_value) * float(right_value)
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def choose_encoder_storage(
    tier: Tier, fp8_ready: bool, score: float | None, free_bytes: int | None,
) -> Storage:
    """Pick FP8, BF16, or CPU for the text encoder. A 16 GB fallback is logged."""
    if tier is Tier.GB24:
        if free_bytes is None:
            return "cpu"
        return "bf16"
    if tier is not Tier.GB16:
        assert_never(tier)
    if not fp8_ready:
        note_error("encoder.fp8_missing")
        return _weights_or_cpu(free_bytes, ENCODER_BF16_BYTES)
    if score is None or score < QUALITY_COSINE:
        note_error("encoder.fp8_quality")
        return _weights_or_cpu(free_bytes, ENCODER_BF16_BYTES)
    if free_bytes is None or free_bytes < ENCODER_FP8_BYTES + ENCODER_HEADROOM_BYTES:
        note_error("encoder.cpu")
        return "cpu"
    return "fp8"


def begin_stage(name: str, tier: Tier) -> None:
    """Open a heavy stage. On 16 GB the card is clear and under the cook budget."""
    _known(name)
    if tier is not Tier.GB16:
        return
    code = _block_code(name, allocated_bytes(), _other_heavy(name))
    if code is None:
        return
    _log(code, name)
    release_residents()
    raise StageBlocked(code)


def finish_stage(name: str, tier: Tier) -> None:
    """Close a heavy stage. On 16 GB the card is near zero and the peak stayed at or under 14 GB."""
    _known(name)
    if tier is not Tier.GB16:
        return
    gc.collect()
    _empty_cache()
    code = _block_code(name, allocated_bytes(), _other_heavy(""))
    if code is None:
        return
    _log(code, name)
    release_residents()
    raise StageBlocked(code)


def admit(name: str, module: object, tier: Tier) -> None:
    """Remember one resident module. A second heavy model on 16 GB stops the stage."""
    if tier is Tier.GB16 and name in HEAVY_MODELS:
        others = _other_heavy(name)
        if others:
            _log("vram.overlap", name)
            release_residents()
            raise StageBlocked("vram.overlap")
    hold_model(name, module)
    if tier is Tier.GB16 and name in HEAVY_MODELS and on_cuda(module):
        guard_peak(name, tier)


def guard_peak(stage: str, tier: Tier) -> None:
    """Stop a 16 GB stage whose allocation or peak passed 14 GB."""
    if tier is not Tier.GB16 or not _over_peak():
        return
    _log("vram.peak", stage)
    release_residents()
    raise StageBlocked("vram.peak")


def release_residents() -> None:
    """Move every remembered model off the GPU and run the registered unloaders."""
    global _releasing
    if _releasing:
        _park_holders()
        return
    _releasing = True
    try:
        for fn in list(_releasers):
            fn()
        _park_holders()
        gc.collect()
        _empty_cache()
    finally:
        _releasing = False


def heavy_names_on_cuda() -> tuple[str, ...]:
    """Heavy models whose parameters are on the GPU, in holder order."""
    return tuple(_other_heavy(""))


def allocated_bytes() -> int:
    """CUDA memory allocated in this process. 0 when CUDA is absent."""
    torch = _torch()
    if torch is None or not bool(torch.cuda.is_available()):
        return 0
    fn = getattr(torch.cuda, "memory_allocated", None)
    if not callable(fn):
        return 0
    try:
        return int(fn())
    except (TypeError, RuntimeError):
        return 0


def peak_bytes() -> int:
    """Peak CUDA allocation in this process. The current allocation when the peak is unavailable."""
    torch = _torch()
    if torch is None or not bool(torch.cuda.is_available()):
        return 0
    fn = getattr(torch.cuda, "max_memory_allocated", None)
    if not callable(fn):
        return allocated_bytes()
    try:
        return int(fn())
    except (TypeError, RuntimeError):
        return allocated_bytes()


def _weights_or_cpu(free_bytes: int | None, need: int) -> Literal["bf16", "cpu"]:
    if free_bytes is None or free_bytes < need + ENCODER_HEADROOM_BYTES:
        note_error("encoder.cpu")
        return "cpu"
    return "bf16"


def _known(name: str) -> None:
    if name not in HEAVY_STAGES:
        raise ValueError(f"unknown stage {name}")


def _other_heavy(name: str) -> list[str]:
    found: list[str] = []
    for held, module in held_modules():
        if held in HEAVY_MODELS and held != name and on_cuda(module):
            found.append(held)
    return found


def _block_code(name: str, allocated: int, others: list[str]) -> str | None:
    if name == "cook" and allocated > COOK_BUDGET_BYTES:
        return "vram.cook"
    if _over_peak() or allocated > PEAK_LIMIT_BYTES:
        return "vram.peak"
    if others:
        return "vram.overlap"
    if allocated > NEAR_ZERO_BYTES:
        return "vram.residual"
    return None


def _over_peak() -> bool:
    return peak_bytes() > PEAK_LIMIT_BYTES or allocated_bytes() > PEAK_LIMIT_BYTES


def _log(code: str, stage: str) -> None:
    note_error(f"{code}:{stage}:{resident_summary()}:allocated={allocated_bytes()}")


def _park_holders() -> None:
    pairs = list(held_modules())
    for _name, module in pairs:
        park_module(module)
    for name, _module in pairs:
        drop_model(name)


def _empty_cache() -> None:
    torch = _torch()
    if torch is None or not bool(torch.cuda.is_available()):
        return
    fn = getattr(torch.cuda, "empty_cache", None)
    if callable(fn):
        fn()


def _torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch
