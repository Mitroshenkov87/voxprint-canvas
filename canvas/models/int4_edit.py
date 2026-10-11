"""Qwen-Image-Edit-2511 with a Nunchaku SVDQuant INT4 transformer.

The text encoder runs and is then unloaded. The transformer keeps one block
on the GPU. The VAE is tiled. The planning peak stays at or under 12 GB when
the VRAM cap is 16 GB.

torch and nunchaku are imported inside the functions that load weights. That
is not a circular import. Those packages are optional, so the unit tests can
import Canvas without them.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from canvas.models.edit import EditLoadError, load_edit_pipeline
from canvas.pipeline.schedule import NUNCHAKU_PEAK_GB, int4_edit_peak_gb

_TRANSFORMER_CLASSES = ("NunchakuQwenImageTransformer2DModel", "NunchakuQwenImageModel")


def load_int4_edit_pipeline(transformer_file: Path, base_folder: Path, dtype: Any) -> Any:
    """Open the edit pipeline with the INT4 transformer and the shared FP8 encoder."""
    if int4_edit_peak_gb(2048, 1152) > NUNCHAKU_PEAK_GB:
        raise EditLoadError("the INT4 edit plan exceeds the VRAM target")
    if not transformer_file.is_file():
        raise EditLoadError("the INT4 transformer is missing")
    transformer = _load_transformer(_import_nunchaku(), transformer_file, dtype)
    pipe = load_edit_pipeline(base_folder, dtype, transformer=transformer)
    enable_vae_tiling(pipe)
    install_sequential_text_encoder(pipe, _import_torch())
    return pipe


def enable_vae_tiling(pipe: Any) -> None:
    """Tile the VAE decode so a full frame is not resident at once."""
    tiling = getattr(pipe, "enable_vae_tiling", None)
    if not callable(tiling):
        raise EditLoadError("the edit model cannot tile the VAE")
    tiling()


def install_sequential_text_encoder(pipe: Any, torch: Any) -> None:
    """Encode, move the text encoder off the GPU, and drop it."""
    encode = getattr(pipe, "encode_prompt", None)
    if not callable(encode):
        raise EditLoadError("the edit model has no text encoder")

    def encode_then_unload(*args: Any, **kwargs: Any) -> Any:
        encoder = getattr(pipe, "text_encoder", None)
        _move(encoder, "cuda" if _cuda(torch) else "cpu")
        try:
            return encode(*args, **kwargs)
        finally:
            held = getattr(pipe, "text_encoder", None)
            _move(held, "cpu")
            pipe.text_encoder = None
            _empty_cache(torch)

    pipe.encode_prompt = encode_then_unload


def _load_transformer(module: Any, path: Path, dtype: Any) -> Any:
    cls = _transformer_class(module)
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if _accepts_zero_cond_t(cls):
        kwargs["zero_cond_t"] = True
    try:
        transformer = cls.from_pretrained(str(path), **kwargs)
    except (OSError, TypeError, ValueError) as exc:
        raise EditLoadError("the INT4 transformer could not be loaded") from exc
    offload = getattr(transformer, "set_offload", None)
    if not callable(offload):
        raise EditLoadError("the INT4 transformer cannot offload blocks")
    offload(True, use_pin_memory=True, num_blocks_on_gpu=1)
    return transformer


def _transformer_class(module: Any) -> Any:
    for name in _TRANSFORMER_CLASSES:
        cls = getattr(module, name, None)
        if cls is not None:
            return cls
    raise EditLoadError("the INT4 runtime has no Qwen-Image transformer")


def _accepts_zero_cond_t(cls: Any) -> bool:
    loader = getattr(cls, "from_pretrained", None)
    if loader is None:
        return False
    try:
        params = inspect.signature(loader).parameters
    except (TypeError, ValueError):
        return False
    if "zero_cond_t" in params:
        return True
    return any(item.kind is inspect.Parameter.VAR_KEYWORD for item in params.values())


def _move(module: Any, device: str) -> None:
    if module is None:
        return
    move = getattr(module, "to", None)
    if callable(move):
        move(device)


def _cuda(torch: Any) -> bool:
    cuda = getattr(torch, "cuda", None)
    available = getattr(cuda, "is_available", None)
    return bool(callable(available) and available())


def _empty_cache(torch: Any) -> None:
    if not _cuda(torch):
        return
    empty = getattr(torch.cuda, "empty_cache", None)
    if callable(empty):
        empty()


def _import_nunchaku() -> Any:
    import nunchaku
    return nunchaku


def _import_torch() -> Any:
    import torch
    return torch
