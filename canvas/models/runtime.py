"""Diffusers inference. The heavy libraries load only when a snapshot is on disk.

torch, diffusers and transformers are imported inside the functions that load
weights. That is not a circular import. Those packages are an optional extra so
the unit tests can import Canvas without them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, assert_never

from PIL import Image

from canvas.models.edit import EditLoadError, load_edit_pipeline
from canvas.models.modes import Tier
from canvas.models.pins import (
    PIPELINE_CONTROL,
    PIPELINE_CONTROL_INPAINT,
    PIPELINE_EDIT,
    PIPELINE_IMG2IMG,
    PIPELINE_INPAINT,
    PIPELINE_TEXT,
)
from canvas.models.plan import CallPlan
from canvas.safety.image import RawImage, ScreenedFrame, ScreenedInput


class InferenceUnavailable(Exception):
    """The plan is known and this machine cannot run it."""

    def __init__(self, detail: str) -> None:
        super().__init__("Image generation is unavailable.")
        self.detail = detail


def render(plan: CallPlan, folder: Path, screened: ScreenedInput, width: int, height: int, tier: Tier, *,
           seed: int = 0, control_file: Path | None = None) -> RawImage:
    """Run one pinned snapshot. ``folder`` is a local diffusers tree."""
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
    if pipeline_name == PIPELINE_TEXT:
        pipe = diffusers.ZImagePipeline.from_pretrained(folder, torch_dtype=dtype, local_files_only=True)
        _place(pipe, tier)
        result = pipe(**common)
    elif pipeline_name == PIPELINE_IMG2IMG:
        pipe = diffusers.ZImageImg2ImgPipeline.from_pretrained(folder, torch_dtype=dtype, local_files_only=True)
        _place(pipe, tier)
        result = pipe(image=_pil(_frame(screened, "init")), strength=0.6, **common)
    elif pipeline_name == PIPELINE_INPAINT:
        pipe = diffusers.ZImageInpaintPipeline.from_pretrained(folder, torch_dtype=dtype, local_files_only=True)
        _place(pipe, tier)
        result = pipe(image=_pil(_frame(screened, "init")), mask_image=_pil(_frame(screened, "mask")), **common)
    elif pipeline_name == PIPELINE_CONTROL or pipeline_name == PIPELINE_CONTROL_INPAINT:
        if control_file is None:
            raise InferenceUnavailable("control weights are missing")
        controlnet = diffusers.ZImageControlNetModel.from_single_file(control_file, torch_dtype=dtype)
        if pipeline_name == PIPELINE_CONTROL:
            pipe = diffusers.ZImageControlNetPipeline.from_pretrained(
                folder, controlnet=controlnet, torch_dtype=dtype, local_files_only=True,
            )
            role = _control_role(screened)
            _place(pipe, tier)
            result = pipe(control_image=_pil(_frame(screened, role)), controlnet_conditioning_scale=0.75, **common)
        else:
            pipe = diffusers.ZImageControlNetInpaintPipeline.from_pretrained(
                folder, controlnet=controlnet, torch_dtype=dtype, local_files_only=True,
            )
            _place(pipe, tier)
            result = pipe(
                image=_pil(_frame(screened, "init")),
                mask_image=_pil(_frame(screened, "mask")),
                control_image=_pil(_frame(screened, _control_role(screened))),
                controlnet_conditioning_scale=0.75,
                **common,
            )
    elif pipeline_name == PIPELINE_EDIT:
        try:
            pipe = load_edit_pipeline(folder, dtype)
        except EditLoadError as exc:
            raise InferenceUnavailable(exc.detail) from exc
        result = pipe(image=_pil(_frame(screened, "init")), **common)
    else:
        raise InferenceUnavailable(f"unknown pipeline {pipeline_name}")
    image = result.images[0].convert("RGB")
    return RawImage(image.width, image.height, image.tobytes())


def _call_guidance(plan: CallPlan) -> dict[str, float]:
    """The guidance argument for this plan. Instruction edit uses true CFG."""
    kind = plan.guidance_kind
    if kind == "true_cfg_scale":
        value = 4.0 if plan.guidance_scale is None else plan.guidance_scale
        return {"true_cfg_scale": value}
    if kind == "" or kind == "guidance_scale":
        if plan.guidance_scale is None:
            return {}
        return {"guidance_scale": plan.guidance_scale}
    raise InferenceUnavailable("unknown guidance kind")


def _place(pipe: Any, tier: Tier) -> None:
    if tier is Tier.GB16:
        pipe.enable_model_cpu_offload()
        return
    if tier is Tier.GB24:
        pipe.to("cuda")
        return
    assert_never(tier)


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
