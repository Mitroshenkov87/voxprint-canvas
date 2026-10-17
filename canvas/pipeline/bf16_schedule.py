"""Sequential BF16 load for one 16 GB Z-Image job.

The text encoder, the transformer, and the VAE each take a turn on the GPU.
ControlNet lite is resident only while an inpaint or guided mode denoises.
Every allocation asks ``budget_gb(free_now, total)`` first. The text encoder
may run on CPU when that budget is short. Any other stage that does not fit
refuses the picture and leaves the GPU as it found it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import assert_never

from canvas.app.i18n import text
from canvas.core.vram_policy import budget_gb
from canvas.models.modes import Mode
from canvas.models.pins import TEXT_ENCODER
from canvas.models.plan import CallPlan
from canvas.pipeline.vram_probe import PeakVramProbe

# Qwen3-4B BF16 weights. The 6B transformer is about 12.3 GB in system RAM.
# Plan B keeps one block group on the GPU (GROUP_BLOCK_GB) and streams the rest.
ENCODER_GB = 8.0
TRANSFORMER_GB = 12.3
GROUP_BLOCK_GB = 1.6
GROUP_ACT_PER_MP_GB = 0.45
CONTROL_LITE_GB = 1.2
VAE_WEIGHT_GB = 0.4
DENOISE_PER_MP_GB = 0.12
VAE_FULL_PER_MP_GB = 0.45
VAE_TILED_GB = 0.35


class VramRefused(Exception):
    """A stage does not fit in the memory budget. ``str`` is the UI line."""

    def __init__(self) -> None:
        super().__init__(text("error.vram"))

    def __str__(self) -> str:
        return text("error.vram")


@dataclass(frozen=True)
class StageCost:
    """GB each stage asks for. Tests substitute a tighter table."""

    encoder_gb: float = ENCODER_GB
    transformer_gb: float = TRANSFORMER_GB
    control_lite_gb: float = CONTROL_LITE_GB
    vae_weight_gb: float = VAE_WEIGHT_GB
    denoise_per_mp_gb: float = DENOISE_PER_MP_GB
    vae_full_per_mp_gb: float = VAE_FULL_PER_MP_GB
    vae_tiled_gb: float = VAE_TILED_GB

    def denoise_gb(self, width: int, height: int) -> float:
        """Transformer weights plus activation cost for this frame."""
        return self.transformer_gb + self.denoise_per_mp_gb * _megapixels(width, height)

    def vae_full_gb(self, width: int, height: int) -> float:
        """VAE weights plus a full-frame decode."""
        return self.vae_weight_gb + self.vae_full_per_mp_gb * _megapixels(width, height)

    def vae_tiled_cost_gb(self) -> float:
        """VAE weights plus one tile. The frame size does not change this."""
        return self.vae_weight_gb + self.vae_tiled_gb


def group_offload_cost() -> StageCost:
    """Resident GPU cost for Plan B. Full BF16 weights stay in system RAM.

    The text encoder is loaded and freed before denoise. One transformer block
    group stays on the GPU. Activations scale with the frame.
    """
    return StageCost(transformer_gb=GROUP_BLOCK_GB, denoise_per_mp_gb=GROUP_ACT_PER_MP_GB)


@dataclass
class GpuLedger:
    """Resident GPU weights for one card. ``used_gb`` includes anything already there."""

    total_gb: float
    used_gb: float = 0.0
    held: dict[str, float] = field(default_factory=dict)
    peak_gb: float = 0.0
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.peak_gb = max(self.peak_gb, self.used_gb)
        self.history.append(self.used_gb)

    def free_gb(self) -> float:
        """GB not currently reserved on this card."""
        return max(0.0, self.total_gb - self.used_gb)

    def room_gb(self) -> float:
        """GB a new stage may take: free memory minus the suite headroom."""
        return budget_gb(self.free_gb(), self.total_gb)

    def take(self, name: str, cost_gb: float) -> bool:
        """Reserve ``cost_gb`` when the budget allows. A refusal changes nothing."""
        cost = float(cost_gb)
        if cost < 0:
            raise ValueError("a stage cost is negative")
        if name in self.held:
            raise ValueError("a stage is already on the GPU")
        if cost > self.room_gb() + 1e-9:
            return False
        self.held[name] = cost
        self.used_gb += cost
        self.peak_gb = max(self.peak_gb, self.used_gb)
        self.history.append(self.used_gb)
        return True

    def drop(self, name: str) -> None:
        """Free one named stage. A missing name changes nothing."""
        cost = self.held.pop(name, 0.0)
        self.used_gb = max(0.0, self.used_gb - cost)
        self.history.append(self.used_gb)

    def drop_all(self) -> None:
        """Free every stage this job reserved."""
        for name in list(self.held):
            self.drop(name)


class EmbeddingCache:
    """Prompt to encoder output. The seed is not part of the key."""

    def __init__(self) -> None:
        self._items: dict[str, str] = {}

    def get(self, prompt: str) -> str | None:
        """The cached embedding for this prompt, or None."""
        return self._items.get(prompt)

    def put(self, prompt: str, value: str) -> None:
        """Store an embedding. The key is the prompt text."""
        self._items[prompt] = value

    def __contains__(self, prompt: str) -> bool:
        return prompt in self._items

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class ScheduleTrace:
    """How one 16 GB job used the GPU."""

    device: str
    tiled: bool
    peak_gb: float
    control: bool


def needs_control(mode: Mode) -> bool:
    """Inpaint and the guided modes denoise with ControlNet lite. Text and restyle do not."""
    if mode is Mode.TEXT or mode is Mode.TEXT_CREATIVE or mode is Mode.RESTYLE or mode is Mode.INSTRUCTION:
        return False
    if mode is Mode.INPAINT or mode is Mode.OUTPAINT or mode is Mode.POSE or mode is Mode.DEPTH or mode is Mode.EDGE:
        return True
    assert_never(mode)


def uses_bf16_16gb(tier_value: str, plan: CallPlan) -> bool:
    """The sequential path is the 16 GB Z-Image BF16 plan, including its Qwen3-4B encoder."""
    return tier_value == "16gb" and plan.quant == "bf16" and plan.text_encoder == TEXT_ENCODER and plan.tier == "16gb"


def run_bf16_16gb(
    *,
    prompt: str,
    mode: Mode,
    width: int,
    height: int,
    ledger: GpuLedger,
    cache: EmbeddingCache,
    probe: PeakVramProbe | None = None,
    cost: StageCost | None = None,
    seed: int = 0,
) -> ScheduleTrace:
    """Load, run, and free each stage. ``seed`` is ignored so a re-roll reuses ``prompt``."""
    del seed
    table = cost or StageCost()
    control = needs_control(mode)
    start = len(ledger.history)
    try:
        device = _encode(prompt, ledger, cache, probe, table)
        _denoise(control, width, height, ledger, probe, table)
        tiled = _decode(width, height, ledger, probe, table)
    except VramRefused:
        ledger.drop_all()
        raise
    if ledger.held:
        ledger.drop_all()
    samples = ledger.history[start - 1 :] if ledger.history else [ledger.used_gb]
    return ScheduleTrace(device, tiled, max(samples), control)


def _megapixels(width: int, height: int) -> float:
    return max(0.0, float(width) * float(height)) / 1_000_000.0


def _mark(probe: PeakVramProbe | None, stage: str) -> None:
    if probe is not None:
        probe.mark(stage)


def _encode(
    prompt: str, ledger: GpuLedger, cache: EmbeddingCache, probe: PeakVramProbe | None, cost: StageCost,
) -> str:
    if cache.get(prompt) is not None:
        _mark(probe, "encode-cache-hit")
        return "cache"
    if ledger.take("encoder", cost.encoder_gb):
        _mark(probe, "encode-gpu-start")
        cache.put(prompt, "bf16")
        _mark(probe, "encode-gpu-end")
        ledger.drop("encoder")
        _mark(probe, "encode-free")
        return "gpu"
    _mark(probe, "encode-cpu-start")
    cache.put(prompt, "bf16")
    _mark(probe, "encode-cpu-end")
    return "cpu"


def _denoise(
    control: bool, width: int, height: int, ledger: GpuLedger, probe: PeakVramProbe | None, cost: StageCost,
) -> None:
    if control:
        if not ledger.take("control", cost.control_lite_gb):
            raise VramRefused()
        _mark(probe, "control-load")
    if not ledger.take("transformer", cost.denoise_gb(width, height)):
        if control and "control" in ledger.held:
            ledger.drop("control")
            _mark(probe, "control-free")
        raise VramRefused()
    _mark(probe, "denoise-start")
    _mark(probe, "denoise-end")
    ledger.drop("transformer")
    _mark(probe, "transformer-free")
    if control:
        ledger.drop("control")
        _mark(probe, "control-free")


def _decode(width: int, height: int, ledger: GpuLedger, probe: PeakVramProbe | None, cost: StageCost) -> bool:
    tiled = False
    if not ledger.take("vae", cost.vae_full_gb(width, height)):
        tiled = True
        if not ledger.take("vae", cost.vae_tiled_cost_gb()):
            raise VramRefused()
    _mark(probe, "decode-start")
    if tiled:
        _mark(probe, "decode-tiled")
    _mark(probe, "decode-end")
    ledger.drop("vae")
    _mark(probe, "vae-free")
    return tiled
