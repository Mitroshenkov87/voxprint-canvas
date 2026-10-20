"""Diffusers inference. The heavy libraries load only when a snapshot is on disk.

torch, diffusers and transformers are imported inside the functions that load
weights. That is not a circular import. Those packages are an optional extra so
the unit tests can import Canvas without them.
"""
from __future__ import annotations

import gc
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never

import psutil
from PIL import Image

from canvas.app.i18n import text
from canvas.infra.stage_timer import timed
from canvas.infra.vram import reset_peak_allocated
from canvas.models.modes import Tier
from canvas.models.pins import (
    PIPELINE_CONTROL,
    PIPELINE_CONTROL_INPAINT,
    PIPELINE_IMG2IMG,
    PIPELINE_INPAINT,
    PIPELINE_TEXT,
)
from canvas.models.plan import CallPlan
from canvas.models.user_lora import UserLora, apply_lora, unload_lora
from canvas.pipeline.offload import (
    OffloadUnavailable,
    install_low_vram_offload,
    install_model_offload,
    model_offload_fits,
)
from canvas.pipeline.schedule import note_image_seconds
from canvas.safety.image import RawImage, ScreenedFrame, ScreenedInput


class InferenceUnavailable(Exception):
    """The plan is known and this machine cannot run it."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        return text("gate.unavailable")


class WeightsLoadError(InferenceUnavailable):
    """The pipeline weights in one folder could not be opened (not an out-of-memory error)."""


def _load(loader: Any, *args: Any, **kwargs: Any) -> Any:
    """Open a pipeline. A broken or unreadable weight tree becomes ``WeightsLoadError``."""
    try:
        return loader(*args, **kwargs)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower():
            raise
        raise WeightsLoadError("the generator weights could not be loaded") from exc


_BASE_PIPELINES = frozenset({PIPELINE_TEXT, PIPELINE_IMG2IMG, PIPELINE_INPAINT})
_PARK_RAM_BYTES = 64 * 1024**3
_scope_depth = 0
_lock = threading.RLock()


@dataclass
class _Held:
    """One pipeline kept for the open run, placement included."""

    folder: str
    tier: Tier
    dtype: object
    control: str
    pipeline_name: str
    lora_key: tuple[str, float] | None
    pipe: Any
    controlnet: Any
    torch: Any


_slot: _Held | None = None


def system_ram_bytes() -> int:
    """Installed RAM. A failed read reports nothing, so the pipeline is not parked."""
    try:
        return int(psutil.virtual_memory().total)
    except (OSError, AttributeError, ValueError):
        return 0


def _park_between_runs() -> bool:
    """64 GB and above may keep the pipeline after the run. Smaller machines load it again."""
    return system_ram_bytes() >= _PARK_RAM_BYTES


def _should_retain() -> bool:
    return _scope_depth > 0 or _park_between_runs()


@contextmanager
def pipeline_scope() -> Iterator[None]:
    """Keep one pipeline, and its placement, until the outermost run ends."""
    global _scope_depth
    with _lock:
        _scope_depth += 1
    try:
        yield
    except Exception:
        release_pipeline()
        raise
    finally:
        with _lock:
            _scope_depth = max(0, _scope_depth - 1)
            if _scope_depth == 0 and not _park_between_runs():
                release_pipeline()


def release_pipeline() -> None:
    """Drop the kept pipeline. Safe when nothing is loaded and when torch is absent.

    Waits for a picture another thread is drawing, so no pipeline is dropped mid-call.
    """
    global _slot
    with _lock:
        held = _slot
        _slot = None
        if held is None:
            return
        _release_lora(held.pipe)
        _release_inference(held.torch, held.pipe, held.controlnet)


def reset_pipeline_cache() -> None:
    """Drop a pipeline and a scope left open by a test."""
    global _scope_depth
    with _lock:
        release_pipeline()
        _scope_depth = 0


def _weights_key(folder: Path, tier: Tier, dtype: object, control_file: Path | None) -> tuple[str, Tier, object, str]:
    control = "" if control_file is None else str(control_file.resolve())
    return (str(folder.resolve()), tier, dtype, control)


def _lora_key(lora: UserLora | None, lora_strength: float | None) -> tuple[str, float] | None:
    if lora is None:
        return None
    strength = lora.default_strength if lora_strength is None else float(lora_strength)
    return (str(lora.weights), strength)


def _pipeline_type(diffusers: Any, pipeline_name: str) -> Any:
    if pipeline_name == PIPELINE_TEXT:
        return diffusers.ZImagePipeline
    if pipeline_name == PIPELINE_IMG2IMG:
        return diffusers.ZImageImg2ImgPipeline
    if pipeline_name == PIPELINE_INPAINT:
        return diffusers.ZImageInpaintPipeline
    if pipeline_name == PIPELINE_CONTROL:
        return diffusers.ZImageControlNetPipeline
    if pipeline_name == PIPELINE_CONTROL_INPAINT:
        return diffusers.ZImageControlNetInpaintPipeline
    raise InferenceUnavailable(f"unknown pipeline {pipeline_name}")


def _retarget(diffusers: Any, held: _Held, pipeline_name: str) -> Any | None:
    """Share the loaded weights with another pipeline class. No second load from disk."""
    if held.pipeline_name not in _BASE_PIPELINES or pipeline_name not in _BASE_PIPELINES:
        return None
    convert = getattr(_pipeline_type(diffusers, pipeline_name), "from_pipe", None)
    if not callable(convert):
        return None
    try:
        return convert(held.pipe)
    except (OSError, ValueError, KeyError, RuntimeError, AttributeError, TypeError):
        return None


def _bind_lora(pipe: Any, lora: UserLora | None, lora_strength: float | None) -> None:
    if lora is None:
        return
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        raise InferenceUnavailable("the generator has no transformer")
    requested = lora.default_strength if lora_strength is None else lora_strength
    apply_lora(transformer, lora, requested)


def _open_pipeline(
    diffusers: Any, pipeline_name: str, folder: Path, dtype: object, control_file: Path | None,
) -> tuple[Any, Any]:
    """Load one pipeline from disk."""
    if pipeline_name == PIPELINE_TEXT:
        pipe = _load(diffusers.ZImagePipeline.from_pretrained, folder, torch_dtype=dtype, local_files_only=True)
        return pipe, None
    if pipeline_name == PIPELINE_IMG2IMG:
        pipe = _load(diffusers.ZImageImg2ImgPipeline.from_pretrained, folder, torch_dtype=dtype, local_files_only=True)
        return pipe, None
    if pipeline_name == PIPELINE_INPAINT:
        pipe = _load(diffusers.ZImageInpaintPipeline.from_pretrained, folder, torch_dtype=dtype, local_files_only=True)
        return pipe, None
    if pipeline_name == PIPELINE_CONTROL or pipeline_name == PIPELINE_CONTROL_INPAINT:
        if control_file is None:
            raise InferenceUnavailable("control weights are missing")
        controlnet = diffusers.ZImageControlNetModel.from_single_file(control_file, torch_dtype=dtype)
        kind = _pipeline_type(diffusers, pipeline_name)
        pipe = _load(kind.from_pretrained, folder, controlnet=controlnet, torch_dtype=dtype, local_files_only=True)
        return pipe, controlnet
    raise InferenceUnavailable(f"unknown pipeline {pipeline_name}")


def _pipeline_for(
    diffusers: Any,
    torch: Any,
    pipeline_name: str,
    folder: Path,
    dtype: object,
    tier: Tier,
    lora: UserLora | None,
    lora_strength: float | None,
    control_file: Path | None,
) -> Any:
    """Return the pipeline for this call, loading it only when the kept one cannot be used."""
    global _slot
    held = _slot
    key = _weights_key(folder, tier, dtype, control_file)
    if held is not None and (held.folder, held.tier, held.dtype, held.control) != key:
        release_pipeline()
        held = None
    if held is not None and held.pipeline_name != pipeline_name:
        converted = _retarget(diffusers, held, pipeline_name)
        if converted is None:
            release_pipeline()
            held = None
        else:
            held.pipe = converted
            held.pipeline_name = pipeline_name
    if held is None:
        with timed("pipeline_load"):
            pipe, controlnet = _open_pipeline(diffusers, pipeline_name, folder, dtype, control_file)
        held = _Held(
            folder=key[0], tier=tier, dtype=dtype, control=key[3], pipeline_name=pipeline_name,
            lora_key=None, pipe=pipe, controlnet=controlnet, torch=torch,
        )
        _slot = held
        with timed("pipeline_place"):
            _place_and_bind(pipe, tier, lora, lora_strength)
        held.lora_key = _lora_key(lora, lora_strength)
        return pipe
    lora_key = _lora_key(lora, lora_strength)
    if held.lora_key != lora_key:
        _release_lora(held.pipe)
        _bind_lora(held.pipe, lora, lora_strength)
        held.lora_key = lora_key
    _slot = held
    return held.pipe


def applied_strength(requested: float | None) -> float:
    """Img2img strength. None keeps the restyle default of 0.6.

    Args:
        requested: A series style-reference strength, or None for the restyle path.

    Raises:
        ValueError: The value is not a series strength.
    """
    # series imports pipeline_scope from this module.
    from canvas.pipeline.series import REFERENCE_MIN

    if requested is None:
        return 0.6
    if isinstance(requested, bool) or not isinstance(requested, (int, float)):
        raise ValueError("strength must be a number")
    value = float(requested)
    if value < REFERENCE_MIN or value > 1.0:
        raise ValueError("strength must be at least 0.75 and at most 1")
    return value


def render(plan: CallPlan, folder: Path, screened: ScreenedInput, width: int, height: int, tier: Tier, *,
           seed: int = 0, control_file: Path | None = None,
           strength: float | None = None, lora: UserLora | None = None,
           lora_strength: float | None = None) -> RawImage:
    """Run one pinned snapshot. ``folder`` is a local diffusers tree."""
    reset_peak_allocated()
    torch = _import_torch()
    diffusers = _import_diffusers()
    dtype = torch.bfloat16
    steps = plan.steps if plan.steps is not None else 8
    guidance = _call_guidance(plan)
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(seed)
    common = {
        "prompt": screened.prompt,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "generator": generator,
        "output_type": "pil",
    }
    common.update(guidance)
    pipeline_name = plan.pipeline_class
    failed = False
    with _lock:
        try:
            pipe = _pipeline_for(
                diffusers, torch, pipeline_name, folder, dtype, tier, lora, lora_strength, control_file,
            )
            if pipeline_name == PIPELINE_TEXT:
                result = _timed(pipe, **common)
            elif pipeline_name == PIPELINE_IMG2IMG:
                result = _timed(pipe, image=_pil(_frame(screened, "init")), strength=applied_strength(strength), **common)
            elif pipeline_name == PIPELINE_INPAINT:
                result = _timed(
                    pipe, image=_pil(_frame(screened, "init")), mask_image=_pil(_frame(screened, "mask")), **common,
                )
            elif pipeline_name == PIPELINE_CONTROL:
                role = _control_role(screened)
                result = _timed(
                    pipe, control_image=_pil(_frame(screened, role)), controlnet_conditioning_scale=0.75, **common,
                )
            elif pipeline_name == PIPELINE_CONTROL_INPAINT:
                result = _timed(
                    pipe,
                    image=_pil(_frame(screened, "init")),
                    mask_image=_pil(_frame(screened, "mask")),
                    control_image=_pil(_frame(screened, _control_role(screened))),
                    controlnet_conditioning_scale=0.75,
                    **common,
                )
            else:
                raise InferenceUnavailable(f"unknown pipeline {pipeline_name}")
            image = result.images[0].convert("RGB")
            return RawImage(image.width, image.height, image.tobytes())
        except Exception:
            failed = True
            raise
        finally:
            if failed or not _should_retain():
                release_pipeline()


def _timed(pipe: Any, **kwargs: Any) -> Any:
    """Run one pipeline call and remember the seconds for the JSON report."""
    started = time.perf_counter()
    result = pipe(**kwargs)
    note_image_seconds(time.perf_counter() - started)
    return result


def _call_guidance(plan: CallPlan) -> dict[str, float]:
    """The guidance argument for this plan."""
    kind = plan.guidance_kind
    if kind == "true_cfg_scale":
        value = 4.0 if plan.guidance_scale is None else plan.guidance_scale
        return {"true_cfg_scale": value}
    if kind == "" or kind == "guidance_scale":
        if plan.guidance_scale is None:
            return {}
        return {"guidance_scale": plan.guidance_scale}
    raise InferenceUnavailable("unknown guidance kind")


def _release_inference(torch: Any, pipe: Any, controlnet: Any) -> None:
    """Drop the generator, text encoder, and VAE before the caller continues."""
    if pipe is not None:
        for name in ("text_encoder", "text_encoder_2", "vae", "transformer", "unet", "controlnet"):
            if getattr(pipe, name, None) is not None:
                try:
                    setattr(pipe, name, None)
                except (AttributeError, TypeError):
                    pass
    del pipe, controlnet
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if ipc_collect is not None:
            ipc_collect()


def _place_and_bind(pipe: Any, tier: Tier, lora: UserLora | None, lora_strength: float | None) -> None:
    """Place the pipeline, then load the selected LoRA on its transformer."""
    _place(pipe, tier)
    if lora is None:
        return
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        raise InferenceUnavailable("the generator has no transformer")
    requested = lora.default_strength if lora_strength is None else lora_strength
    apply_lora(transformer, lora, requested)


def _release_lora(pipe: Any) -> None:
    """Remove adapters before the pipeline is dropped."""
    if pipe is None:
        return
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        unload_lora(transformer)


def _place(pipe: Any, tier: Tier) -> None:
    """16 GB streams transformer blocks from pinned RAM. 24 GB keeps the pipeline resident."""
    if tier is Tier.GB16:
        if model_offload_fits(pipe, _free_vram_bytes()):
            install_model_offload(pipe)
            return
        try:
            install_low_vram_offload(pipe)
        except OffloadUnavailable as exc:
            raise InferenceUnavailable("this pipeline cannot offload weights") from exc
        return
    if tier is Tier.GB24:
        pipe.to("cuda")
        return
    assert_never(tier)


def _free_vram_bytes() -> int | None:
    """Free CUDA memory now, or None without a CUDA device."""
    try:
        torch = _import_torch()
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def _pil(frame: ScreenedFrame) -> Image.Image:
    return Image.frombytes("RGB", (frame.width, frame.height), frame.to_bytes())


def _frame(screened: ScreenedInput, role: str) -> ScreenedFrame:
    for frame in screened.frames:
        if frame.role == role:
            return frame
    raise InferenceUnavailable(f"missing {role} image")


def _control_role(screened: ScreenedInput) -> str:
    for role in ("pose", "depth", "edge", "init"):
        if any(frame.role == role for frame in screened.frames):
            return role
    raise InferenceUnavailable("missing control image")


def _import_torch() -> Any:
    import torch
    return torch


def _import_diffusers() -> Any:
    import diffusers
    return diffusers
