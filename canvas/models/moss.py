"""MOSS-Image text-to-image. The text encoder is freed before the pipeline loads.

The embed-call contract of ``pipeline_moss_image`` must be verified on a GPU
before release. Compact weights are pre-compressed, so this module does not
quantize at load time. A 16 GB run with one LoRA stays at or under 14 GB.

torch, diffusers and transformers are imported inside the functions that load
weights. The unit tests can import this module without them.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from canvas.models.modes import Tier
from canvas.models.plan import CallPlan
from canvas.models.runtime import InferenceUnavailable, _bind_lora, _image_from_result
from canvas.models.user_lora import UserLora
from canvas.safety.image import RawImage, ScreenedInput

_VENDORED = "pipeline_moss_image.py"


def render_moss(plan: CallPlan, folder: Path, screened: ScreenedInput, width: int, height: int, tier: Tier, *,
                seed: int = 0, control_file: Path | None = None,
                strength: float | None = None, lora: UserLora | None = None,
                lora_strength: float | None = None, negative: str = "") -> RawImage:
    """Run one local MOSS snapshot. The encoder is gone before the pipeline loads.

    ``control_file`` and ``strength`` match :func:`canvas.models.runtime.render`.
    Text-to-image does not use them. Weights are never quantized here.
    """
    del control_file, strength
    encoder = _load_encoder(folder)
    prompt_embeds, negative_embeds = _encode(encoder, screened.prompt, negative)
    _release(encoder)
    pipe = _load_pipe(folder)
    return _run(
        pipe, plan, prompt_embeds, negative_embeds, width, height, tier, seed, lora, lora_strength,
    )


def _load_encoder(folder: Path) -> tuple[Any, Any]:
    """Load only the tokenizer and the text encoder, in BF16, from this folder."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(folder / "tokenizer", local_files_only=True)
    encoder = AutoModel.from_pretrained(
        folder / "text_encoder",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    return tokenizer, encoder


def _encode(encoder: Any, prompt: str, negative: str) -> tuple[Any, Any]:
    """Prompt and negative embeddings. A test double may implement ``encode``."""
    if isinstance(encoder, tuple):
        tokenizer, model = encoder
        return _encode_text(tokenizer, model, prompt), _encode_text(tokenizer, model, negative)
    hook = getattr(encoder, "encode", None)
    if callable(hook):
        return hook(prompt, negative)
    raise InferenceUnavailable("the text encoder did not load")


def _encode_text(tokenizer: Any, model: Any, text: str) -> Any:
    """One text embedding. The hidden state is what the vendored pipeline must accept."""
    import torch

    tokens = tokenizer(
        text,
        padding="max_length",
        max_length=getattr(tokenizer, "model_max_length", None) or 256,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return model(**tokens).last_hidden_state


def _release(encoder: Any) -> None:
    """Delete the encoder, collect garbage, and drop the CUDA cache."""
    del encoder
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    torch.cuda.empty_cache()


def _load_pipe(folder: Path) -> Any:
    """Load the diffusion pipeline from the vendored file in ``folder``.

    The pipeline class is a local file. Remote custom code is never trusted.
    """
    vendored = folder / _VENDORED
    if not vendored.is_file():
        raise InferenceUnavailable(f"{_VENDORED} is not in the model folder")
    import torch
    from diffusers import DiffusionPipeline

    return DiffusionPipeline.from_pretrained(
        folder,
        custom_pipeline=str(vendored),
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )


def _run(pipe: Any, plan: CallPlan, prompt_embeds: Any, negative_embeds: Any, width: int, height: int,
         tier: Tier, seed: int, lora: UserLora | None, lora_strength: float | None) -> RawImage:
    """Apply the LoRA, sample, and free the pipeline on a 16 GB card."""
    _bind_lora(pipe, lora, lora_strength)
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    result = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        num_inference_steps=plan.steps,
        guidance_scale=plan.guidance_scale,
        generator=generator,
        width=width,
        height=height,
    )
    image = _image_from_result(pipe, result, torch)
    if tier is Tier.GB16:
        del pipe, result
        gc.collect()
        empty = getattr(getattr(torch, "cuda", None), "empty_cache", None)
        if callable(empty):
            empty()
    return image
