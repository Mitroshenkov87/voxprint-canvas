"""Qwen-Image-Edit-2511 FP8 weights for both VRAM tiers.

The mirror stores linear weights as float8_e4m3fn plus a per-tensor BF16 scale
(``voxprint-fp8-v1``). Embeddings, norms, and the layers the quantizer left
alone stay BF16. Compute dequantizes one linear at a time to BF16, so the
stored weights stay FP8 in system RAM. On 16 GB, group offload brings one
transformer block to the GPU and that block is upcast for the matmul. On
24 GB, one whole component is on the GPU. The VAE is tiled, and on 16 GB it
is sliced as well.

torch, diffusers, transformers, accelerate, and safetensors are imported inside
the functions that load weights. That is not a circular import. Those packages
are an optional extra so the unit tests can import Canvas without them.
"""
from __future__ import annotations

import json
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from canvas.pipeline.offload import OffloadUnavailable, install_low_vram_offload

_HEADER_MAX = 8_000_000
_FP8_DTYPE = "F8_E4M3"
_FORMAT = "voxprint-fp8-v1"
_SCALE_SUFFIX = ".scale_weight"

EDIT_ID = "qwen-image-edit-2511-fp8"
ENCODER_FILE = "qwen2.5-vl-7b-text-encoder-fp8.safetensors"
ENCODER_FP8_TENSORS = 197
TRANSFORMER_SHARDS = tuple(
    f"qwen-image-edit-2511-transformer-fp8-{index:05d}-of-00005.safetensors" for index in range(1, 6)
)


class EditLoadError(Exception):
    """The edit snapshot is on disk and the FP8 pack cannot be loaded."""

    def __init__(self, detail: str) -> None:
        super().__init__("Image generation is unavailable.")
        self.detail = detail


@dataclass(frozen=True)
class Fp8Plan:
    """Checkpoint tensor names mapped onto module keys, and how many are FP8."""

    assignments: dict[str, str]
    fp8_count: int


def vl_checkpoint_key(name: str) -> str:
    """Map a Qwen2.5-VL checkpoint name onto transformers 5.19.

    The published file uses ``model.layers`` and ``visual.*``. Transformers
    5.19 stores those as ``model.language_model.*`` and ``model.visual.*``.
    """
    if name.startswith("visual."):
        return "model." + name
    prefix = "model."
    if name.startswith(prefix) and not name.startswith(("model.language_model.", "model.visual.")):
        return "model.language_model." + name[len(prefix):]
    return name


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """The JSON header of a safetensors file. The tensor bytes are not read."""
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("safetensors header is missing")
        length = struct.unpack("<Q", raw)[0]
        if length < 2 or length > _HEADER_MAX:
            raise ValueError("safetensors header is missing")
        payload = handle.read(length)
    if len(payload) != length:
        raise ValueError("safetensors header is missing")
    meta = json.loads(payload)
    if not isinstance(meta, dict):
        raise ValueError("safetensors header is missing")
    return meta


def plan_fp8_assignment(
    headers: list[dict[str, Any]],
    module_keys: set[str],
    remap: Callable[[str], str] | None = None,
) -> Fp8Plan:
    """Match every checkpoint tensor to a module key.

    An FP8 tensor without its scale, a scale without an FP8 tensor, or a name
    the module does not have refuses the pack. Partial application is how the
    text encoder previously kept 196 of 197 FP8 tensors.
    """
    assignments: dict[str, str] = {}
    fp8_names: list[str] = []
    scales: set[str] = set()
    for header in headers:
        meta = header.get("__metadata__")
        if not isinstance(meta, dict) or meta.get("format") != _FORMAT:
            raise EditLoadError("the FP8 pack format is missing")
        for key, info in header.items():
            if key == "__metadata__":
                continue
            weight, is_scale = _scale_name(key)
            if is_scale:
                scales.add(weight)
                continue
            if weight in assignments:
                raise EditLoadError("a repeated tensor")
            mapped = weight if remap is None else remap(weight)
            if mapped not in module_keys:
                raise EditLoadError("a tensor did not match the model")
            assignments[weight] = mapped
            if isinstance(info, dict) and info.get("dtype") == _FP8_DTYPE:
                fp8_names.append(weight)
    fp8_set = set(fp8_names)
    if any(name not in scales for name in fp8_names):
        raise EditLoadError("an FP8 tensor has no scale")
    if any(name not in fp8_set for name in scales):
        raise EditLoadError("a scale has no FP8 tensor")
    return Fp8Plan(assignments, len(fp8_names))


def probe_edit_pack(folder: Path) -> str:
    """Empty when the FP8 headers match the published check, else a short code."""
    try:
        encoder = read_safetensors_header(folder / ENCODER_FILE)
        if _format_of(encoder) != _FORMAT:
            return "fp8_format"
        if _fp8_count(encoder) != ENCODER_FP8_TENSORS:
            return "fp8_encoder"
        total = 0
        for name in TRANSFORMER_SHARDS:
            header = read_safetensors_header(folder / name)
            if _format_of(header) != _FORMAT:
                return "fp8_format"
            total += _fp8_count(header)
        if total < 1:
            return "fp8_transformer"
    except (OSError, ValueError, json.JSONDecodeError):
        return "fp8_format"
    return ""


def install_fp8_files(
    module: Any,
    paths: list[Path],
    remap: Callable[[str], str] | None,
) -> int:
    """Load FP8 shards onto ``module`` and return how many FP8 tensors were installed."""
    torch = _import_torch()
    try:
        headers = [read_safetensors_header(path) for path in paths]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise EditLoadError("the FP8 pack could not be read") from exc
    module_keys = {name for name, _tensor in module.named_parameters()}
    module_keys.update(name for name, _tensor in module.named_buffers())
    planned = plan_fp8_assignment(headers, module_keys, remap)
    load_file = _import_load_file()
    seen: set[str] = set()
    for path in paths:
        state = load_file(str(path), device="cpu")
        for name, module_key in planned.assignments.items():
            if name not in state:
                continue
            if name in seen:
                raise EditLoadError("a repeated tensor")
            seen.add(name)
            tensor = state[name]
            scale = None
            if tensor.dtype == torch.float8_e4m3fn:
                scale = state.get(name + _SCALE_SUFFIX)
                if scale is None:
                    raise EditLoadError("an FP8 tensor has no scale")
            _assign(module, module_key, tensor, scale, torch)
        del state
    if len(seen) != len(planned.assignments):
        raise EditLoadError("a shard was missing a tensor")
    _reject_meta_parameters(module)
    _restore_meta_buffers(module, torch)
    return planned.fp8_count


def load_edit_pipeline(folder: Path, dtype: Any, *, low_vram: bool = False) -> Any:
    """Open the edit pipeline with FP8 weights.

    ``low_vram`` installs group offload (or sequential CPU offload) so the
    encoder, the transformer blocks, and the VAE take turns. Otherwise one
    component at a time is moved to the GPU, which is the 24 GB path.
    """
    torch = _import_torch()
    diffusers = _import_diffusers()
    transformers = _import_transformers()
    accelerate = _import_accelerate()
    try:
        transformer = _empty(
            accelerate, diffusers.QwenImageTransformer2DModel, folder / "transformer" / "config.json",
        )
        transformer_count = install_fp8_files(
            transformer, [folder / name for name in TRANSFORMER_SHARDS], remap=None,
        )
        if transformer_count < 1:
            raise EditLoadError("the FP8 transformer did not match")
        config = transformers.AutoConfig.from_pretrained(folder / "text_encoder", local_files_only=True)
        encoder = _empty(
            accelerate, transformers.Qwen2_5_VLForConditionalGeneration, config, dtype,
        )
        encoder_count = install_fp8_files(encoder, [folder / ENCODER_FILE], remap=vl_checkpoint_key)
        if encoder_count != ENCODER_FP8_TENSORS:
            raise EditLoadError("the FP8 text encoder did not match")
        pipe = diffusers.QwenImageEditPlusPipeline.from_pretrained(
            folder,
            transformer=transformer,
            text_encoder=encoder,
            torch_dtype=dtype,
            local_files_only=True,
        )
    except EditLoadError:
        raise
    except (OSError, ValueError) as exc:
        raise EditLoadError("the edit model could not be loaded") from exc
    if low_vram:
        _install_low_vram(pipe)
    else:
        pipe.enable_model_cpu_offload()
        tiling = getattr(pipe, "enable_vae_tiling", None)
        if callable(tiling):
            tiling()
    del torch
    return pipe


def _install_low_vram(pipe: Any) -> None:
    try:
        install_low_vram_offload(pipe)
    except OffloadUnavailable as exc:
        raise EditLoadError("the edit model could not be offloaded") from exc


def _empty(accelerate: Any, cls: Any, config: Any, dtype: Any = None) -> Any:
    """Build ``cls`` with parameters on the meta device.

    Diffusers 0.41 still has ``from_config``. Transformers 5.19 does not, so the
    text encoder goes through ``_from_config``. RoPE tables stay real CPU tensors
    because ``init_empty_weights`` moves parameters only.
    """
    loaded = json.loads(config.read_text(encoding="utf-8")) if isinstance(config, Path) else config
    with accelerate.init_empty_weights():
        builder = getattr(cls, "from_config", None)
        if builder is not None:
            return builder(loaded)
        return cls._from_config(loaded, dtype=dtype)


def _assign(module: Any, module_key: str, tensor: Any, scale: Any, torch: Any) -> None:
    parent_name, _, attr = module_key.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    if scale is not None:
        if not isinstance(parent, torch.nn.Linear) or attr != "weight":
            raise EditLoadError("an FP8 tensor is not a linear weight")
        parent.weight = torch.nn.Parameter(tensor.detach(), requires_grad=False)
        parent.register_buffer(
            "fp8_scale",
            scale.detach().to(dtype=torch.bfloat16).reshape(()),
            persistent=False,
        )
        parent.forward = _fp8_forward.__get__(parent, torch.nn.Linear)
        return
    current = getattr(parent, attr)
    if isinstance(current, torch.nn.Parameter):
        setattr(parent, attr, torch.nn.Parameter(tensor.detach(), requires_grad=False))
        return
    parent.register_buffer(attr, tensor.detach())


def _fp8_forward(self: Any, inputs: Any) -> Any:
    """Upcast this linear to BF16 for the matmul. The stored FP8 weight is left in place.

    Group offload has one block on the GPU when this runs, so the copy is that
    block and not the whole transformer. Optional gen extra, same reason as
    the other imports in this module.
    """
    import torch

    device = inputs.device
    weight = self.weight.to(device=device, dtype=torch.bfloat16) * self.fp8_scale.to(device=device, dtype=torch.bfloat16)
    bias = None if self.bias is None else self.bias.to(device=device, dtype=torch.bfloat16)
    return torch.nn.functional.linear(inputs.to(dtype=torch.bfloat16), weight, bias)


def _reject_meta_parameters(module: Any) -> None:
    missing = [name for name, tensor in module.named_parameters() if getattr(tensor, "is_meta", False)]
    if missing:
        raise EditLoadError(f"{len(missing)} parameters were not in the FP8 pack")


def _restore_meta_buffers(module: Any, torch: Any) -> None:
    for sub in module.modules():
        _restore_inv_freq(sub, torch)
        _restore_embed_rope(sub, torch)
    leftover = [name for name, buf in module.named_buffers() if getattr(buf, "is_meta", False)]
    if leftover:
        raise EditLoadError(f"{len(leftover)} buffers were not in the FP8 pack")
    plain = [
        name for name, tensor in _plain_tensors(module)
        if getattr(tensor, "is_meta", False)
    ]
    if plain:
        raise EditLoadError(f"{len(plain)} buffers were not in the FP8 pack")


def _restore_inv_freq(sub: Any, torch: Any) -> None:
    inv = getattr(sub, "inv_freq", None)
    if inv is None or not getattr(inv, "is_meta", False):
        return
    compute = getattr(sub, "compute_default_rope_parameters", None)
    if compute is None:
        compute = getattr(sub, "compute_axial_rope_parameters", None)
    config = getattr(sub, "config", None)
    if compute is None or config is None:
        return
    freqs, scale = compute(config)
    sub.inv_freq = torch.nn.Buffer(freqs, persistent=False)
    if hasattr(sub, "original_inv_freq"):
        sub.original_inv_freq = torch.nn.Buffer(freqs.detach().clone(), persistent=False)
    sub.attention_scaling = scale


def _restore_embed_rope(sub: Any, torch: Any) -> None:
    """Rebuild Qwen image RoPE tables when empty init left them on the meta device.

    Diffusers 0.41 stores ``pos_freqs`` and ``neg_freqs`` as plain tensors, 4096
    positions, built by ``rope_params``. They are not parameters, so a checkpoint
    never contains them.
    """
    pos = getattr(sub, "pos_freqs", None)
    if pos is None or not getattr(pos, "is_meta", False):
        return
    rope_params = getattr(sub, "rope_params", None)
    axes = getattr(sub, "axes_dim", None)
    theta = getattr(sub, "theta", None)
    if rope_params is None or axes is None or theta is None or len(axes) != 3:
        return
    pos_index = torch.arange(4096)
    neg_index = torch.arange(4096).flip(0) * -1 - 1
    sub.pos_freqs = torch.cat([rope_params(pos_index, axes[index], theta) for index in range(3)], dim=1)
    sub.neg_freqs = torch.cat([rope_params(neg_index, axes[index], theta) for index in range(3)], dim=1)


def _plain_tensors(module: Any):
    for name, sub in module.named_modules():
        prefix = f"{name}." if name else ""
        for key, value in vars(sub).items():
            if key.startswith("_"):
                continue
            is_meta = getattr(value, "is_meta", None)
            if is_meta is None:
                continue
            yield prefix + key, value


def _scale_name(key: str) -> tuple[str, bool]:
    if key.endswith(_SCALE_SUFFIX):
        return key[: -len(_SCALE_SUFFIX)], True
    return key, False


def _format_of(header: dict[str, Any]) -> str:
    meta = header.get("__metadata__")
    if not isinstance(meta, dict):
        return ""
    return str(meta.get("format") or "")


def _fp8_count(header: dict[str, Any]) -> int:
    total = 0
    for key, info in header.items():
        if key == "__metadata__" or not isinstance(info, dict):
            continue
        if info.get("dtype") == _FP8_DTYPE:
            total += 1
    return total


def _import_torch() -> Any:
    import torch
    return torch


def _import_diffusers() -> Any:
    import diffusers
    return diffusers


def _import_transformers() -> Any:
    import transformers
    return transformers


def _import_accelerate() -> Any:
    import accelerate
    return accelerate


def _import_load_file() -> Any:
    from safetensors.torch import load_file
    return load_file
