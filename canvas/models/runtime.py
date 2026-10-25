"""Diffusers inference. The heavy libraries load only when a snapshot is on disk.

torch, diffusers and transformers are imported inside the functions that load
weights. That is not a circular import. Those packages are an optional extra so
the unit tests can import Canvas without them.
"""
from __future__ import annotations

import gc
import inspect
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, assert_never

import psutil
from PIL import Image

from canvas.app.i18n import text
from canvas.infra.errors import note_error
from canvas.infra.stage_timer import note_peak_mib, note_stage, note_tag, timed
from canvas.infra.stage_guard import (
    PEAK_LIMIT_BYTES,
    StageBlocked,
    admit,
    allocated_bytes,
    begin_stage,
    choose_encoder_storage,
    cosine,
    finish_stage,
    guard_peak,
    register_release,
)
from canvas.infra.vram_trace import drop_model, hold_model, note_vram
from canvas.infra.vram import peak_allocated_bytes, reset_peak_allocated
from canvas.models import hot_tier as hot
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
from canvas.pipeline.bf16_schedule import VramRefused
from canvas.pipeline.device_guard import DeviceRefused, pretrained_attempts, require_cuda
from canvas.pipeline.offload import (
    OffloadUnavailable,
    cast_module_fp8,
    install_low_vram_offload,
    install_fp8_storage,
    install_model_offload,
    resident_fits,
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
    """``ram64`` and ``ram96`` keep the pipeline weights in RAM. ``ram32`` loads them again."""
    return hot.keeps_main_tier()


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
        if _park_between_runs():
            park_pipeline()
        else:
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
    tier: Tier | None = None
    with _lock:
        held = _slot
        _slot = None
        if held is None:
            return
        note_vram("pipeline_release", "before", held.pipe)
        _release_lora(held.pipe)
        _release_inference(held.torch, held.pipe, held.controlnet)
        for name in ("transformer", "vae", "text_encoder"):
            drop_model(name)
        note_vram("pipeline_release", "after")
        tier = held.tier
    hot.forget("transformer", "text_encoder", "vae", "controlnet")
    if tier is Tier.GB16:
        _leave("cook", tier)


def park_pipeline() -> None:
    """Drop VRAM copies. Tiers that keep main models leave the RAM copies in place."""
    for name in ("transformer", "text_encoder", "vae", "controlnet"):
        hot.deactivate(name)
    if not _park_between_runs():
        release_pipeline()


def reset_pipeline_cache() -> None:
    """Drop a pipeline and a scope left open by a test."""
    global _scope_depth
    with _lock:
        release_pipeline()
        hot.reset()
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
    """Share the loaded weights with another pipeline class. No second load from disk.

    ``from_pipe`` casts every shared module to float32. That moves FP8 storage
    and the layers that must stay in BF16, and the next matmul fails. A new
    pipeline object is built on the same modules and is not cast.
    """
    if held.pipeline_name not in _BASE_PIPELINES or pipeline_name not in _BASE_PIPELINES:
        return None
    shared = _share_pipeline(diffusers, held.pipe, pipeline_name)
    if shared is not None:
        return shared
    convert = getattr(_pipeline_type(diffusers, pipeline_name), "from_pipe", None)
    if not callable(convert):
        return None
    try:
        return _from_pipe_keep_dtype(convert, held.pipe)
    except (OSError, ValueError, KeyError, RuntimeError, AttributeError, TypeError):
        return None


def _share_pipeline(diffusers: Any, pipe: Any, pipeline_name: str) -> Any | None:
    """Another pipeline class on the modules ``pipe`` already holds."""
    kind = _pipeline_type(diffusers, pipeline_name)
    if not isinstance(kind, type):
        return None
    try:
        return kind(
            scheduler=pipe.scheduler,
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            transformer=pipe.transformer,
        )
    except (AttributeError, TypeError):
        return None


def _from_pipe_keep_dtype(convert: Any, pipe: Any) -> Any:
    """``from_pipe`` without the float32 cast it applies to shared modules."""
    modules = _dtype_modules(pipe)
    originals: list[tuple[Any, Any]] = []
    for module in modules:
        original = module.to

        def frozen(*args: Any, _module: Any = module, _original: Any = original, **kwargs: Any) -> Any:
            args = tuple(arg for arg in args if not _is_dtype(arg))
            kwargs = {key: value for key, value in kwargs.items() if key not in {"dtype", "torch_dtype"} and not _is_dtype(value)}
            if not args and not kwargs:
                return _module
            return _original(*args, **kwargs)

        module.to = frozen
        originals.append((module, original))
    try:
        return convert(pipe)
    finally:
        for module, original in originals:
            module.to = original


def _dtype_modules(pipe: Any) -> list[Any]:
    components = getattr(pipe, "components", None)
    if not isinstance(components, dict):
        return []
    return [module for module in components.values() if callable(getattr(module, "to", None))]


def _is_dtype(value: object) -> bool:
    return type(value).__name__ == "dtype"


_prompt_embeds: dict[str, Any] | None = None
_cosine_cache: dict[str, float | None] = {}
# The encoder prepare holds one loaded pair so the cosine probe does not open it again.
_borrowed_pair: tuple[Any, Any] | None = None


# A few neutral lines. FP8 stays when each embedding is within cosine 0.99 of BF16.
_PROBE_PROMPTS = (
    "a red ceramic cup on a wooden table",
    "a mountain lake in the morning",
    "a quiet city street after rain",
)


def prepare_series_embeds(
    folder: Path, tier: Tier, prompts: tuple[str, ...], *,
    lora: UserLora | None = None, lora_strength: float | None = None,
) -> None:
    """Encode every prompt with the text encoder alone, keep the embeddings on the CPU, and unload it.

    The transformer and the VAE stay on disk until cook. On a 16 GB card the encoder
    runs in FP8 when those embeddings stay close to BF16. ``lora`` binds to the
    transformer at cook.
    """
    del lora, lora_strength
    with _lock:
        _prepare_locked(folder, tier, prompts)


def _prepare_locked(folder: Path, tier: Tier, prompts: tuple[str, ...]) -> None:
    global _prompt_embeds, _borrowed_pair
    torch = _import_torch()
    hot.note_tier()
    _enter("text_encoder", tier)
    tokenizer: Any = None
    encoder: Any = None
    try:
        with timed("encoder"):
            found = hot.bundle("text_encoder")
            if isinstance(found, tuple) and len(found) == 3 and found[2] is tier:
                tokenizer, encoder = found[0], found[1]
            else:
                if found is not None:
                    hot.forget("text_encoder")
                with timed("load"):
                    tokenizer, encoder = _load_text_pair(folder, torch)
                    _borrowed_pair = (tokenizer, encoder)
                    try:
                        score = _encoder_cosine(folder, tier, torch)
                    finally:
                        _borrowed_pair = None
                    storage = choose_encoder_storage(tier, _fp8_ready(torch), score, _free_vram_bytes())
                    if _encoder_needs_reload(encoder, storage):
                        _release_text_encoder(encoder, tokenizer, torch)
                        tokenizer, encoder = _load_text_pair(folder, torch)
                _place_encoder(encoder, storage, torch, tier)
                if hot.remember("text_encoder", encoder):
                    hot.remember_bundle("text_encoder", (tokenizer, encoder, tier))
            device = "cuda" if _cuda_is_up(torch) else "cpu"
            hot.activate("text_encoder", device)
            if _cuda_is_up(torch):
                require_cuda(encoder, "encoder.device", torch)
            _keep("text_encoder", encoder, tier)
            note_vram("text_encoder", "before", encoder)
            encoded: dict[str, Any] = {}
            for prompt in prompts:
                encoded[prompt] = _to_cpu(_encode_prompt_text(tokenizer, encoder, prompt, torch))
            if _prompt_embeds is None:
                _prompt_embeds = encoded
            else:
                _prompt_embeds = {**_prompt_embeds, **encoded}
    finally:
        if hot.retained("text_encoder"):
            hot.deactivate("text_encoder")
            _borrowed_pair = None
        else:
            _release_text_encoder(encoder, tokenizer, torch)
        encoder = None
        tokenizer = None
        gc.collect()
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        note_vram("text_encoder", "after")
        _leave("text_encoder", tier)


def text_encoder_on_cuda(pipe: Any) -> bool:
    """True when a text encoder parameter still lives on the GPU."""
    for name in ("text_encoder", "text_encoder_2"):
        module = getattr(pipe, name, None)
        if module is not None and _module_on_cuda(module):
            return True
    return False


def _module_on_cuda(module: Any) -> bool:
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        try:
            for param in parameters():
                device = getattr(param, "device", None)
                if device is not None and getattr(device, "type", None) == "cuda":
                    return True
        except (TypeError, RuntimeError):
            return False
        return False
    device = getattr(module, "device", None)
    return device is not None and "cuda" in str(getattr(device, "type", device))


def _keep(name: str, module: Any, tier: Tier) -> None:
    try:
        admit(name, module, tier)
    except StageBlocked as exc:
        raise VramRefused() from exc


def _peak(stage: str, tier: Tier) -> None:
    try:
        guard_peak(stage, tier)
    except StageBlocked as exc:
        raise VramRefused() from exc


def _enter(name: str, tier: Tier) -> None:
    try:
        begin_stage(name, tier)
    except StageBlocked as exc:
        raise VramRefused() from exc


def _leave(name: str, tier: Tier) -> None:
    try:
        finish_stage(name, tier)
    except StageBlocked as exc:
        raise VramRefused() from exc


def _fp8_ready(torch: Any) -> bool:
    return getattr(torch, "float8_e4m3fn", None) is not None and _layerwise_casting() is not None


def _encoder_cosine(folder: Path, tier: Tier, torch: Any) -> float | None:
    """Minimum cosine of FP8 embeddings against a BF16 pass. None outside 16 GB.

    On CUDA the reference pass stays on the GPU and is unloaded before the FP8
    pass. A CPU reference is used only when CUDA is absent.
    """
    if tier is not Tier.GB16 or not _fp8_ready(torch):
        return None
    key = str(folder)
    if key in _cosine_cache:
        return _cosine_cache[key]
    reference_storage: Literal["cpu", "bf16"] = "bf16" if _cuda_is_up(torch) else "cpu"
    try:
        reference = _probe_vectors(folder, torch, reference_storage)
        candidate = _probe_vectors(folder, torch, "fp8")
    except DeviceRefused:
        raise
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError, WeightsLoadError):
        if _borrowed_pair is not None:
            _mark_encoder(_borrowed_pair[1], "_canvas_dirty")
        _cosine_cache[key] = 0.0
        return 0.0
    if len(reference) != len(candidate) or not reference:
        _cosine_cache[key] = 0.0
        return 0.0
    scores = [cosine(left, right) for left, right in zip(reference, candidate, strict=True)]
    score = min(scores) if scores else 0.0
    _cosine_cache[key] = score
    return score


def _probe_vectors(folder: Path, torch: Any, storage: Literal["cpu", "bf16", "fp8"]) -> list[list[float]]:
    borrowed = _borrowed_pair
    if borrowed is None:
        tokenizer, encoder = _load_text_pair(folder, torch)
    else:
        tokenizer, encoder = borrowed
    try:
        _apply_encoder_storage(encoder, storage, torch)
        vectors: list[list[float]] = []
        for prompt in _PROBE_PROMPTS:
            encoded = _to_cpu(_encode_prompt_text(tokenizer, encoder, prompt, torch))
            vectors.append(_vector(encoded))
        return vectors
    finally:
        if borrowed is None:
            _release_text_encoder(encoder, tokenizer, torch)


def _apply_encoder_storage(encoder: Any, storage: Literal["cpu", "bf16", "fp8"], torch: Any) -> None:
    """Move one already-loaded encoder. This does not open the weights again."""
    if storage == "cpu":
        _refuse_cpu_encoder(torch)
        encoder.to("cpu")
        return
    if storage == "bf16":
        if _cuda_is_up(torch):
            if not _module_on_cuda(encoder):
                encoder.to("cuda")
            require_cuda(encoder, "encoder.device", torch)
        else:
            encoder.to("cpu")
        return
    if storage == "fp8":
        with timed("cast"):
            if not cast_module_fp8(encoder, torch):
                raise RuntimeError("fp8 cast failed")
        _mark_encoder(encoder, "_canvas_fp8")
        if _cuda_is_up(torch):
            if not _module_on_cuda(encoder):
                with timed("place"):
                    encoder.to("cuda")
            require_cuda(encoder, "encoder.device", torch)
            return
        raise RuntimeError("fp8 cast failed")
    assert_never(storage)


def _load_text_pair(folder: Path, torch: Any) -> tuple[Any, Any]:
    """The tokenizer and the text encoder. The picture pipeline stays unloaded.

    On CUDA the encoder is mapped from the safetensors file onto the card.
    ``device_map="auto"`` is not used.
    """
    transformers = _import_transformers()
    tokenizer = _load(
        transformers.AutoTokenizer.from_pretrained, folder / "tokenizer", local_files_only=True,
    )
    device = "cuda" if _cuda_is_up(torch) else "cpu"
    encoder = _load_component(
        transformers.AutoModel.from_pretrained, folder / "text_encoder", torch.bfloat16, device,
    )
    return tokenizer, encoder


def _load_component(loader: Any, folder: Path, dtype: object, onto: str, **extra: Any) -> Any:
    """Open one weight folder. A rejected ``device_map`` is tried again without it."""
    attempts = pretrained_attempts(dtype, onto)
    error: TypeError | None = None
    last = len(attempts) - 1
    for index, kwargs in enumerate(attempts):
        try:
            return _load(loader, folder, **extra, **kwargs)
        except TypeError as exc:
            error = exc
        except WeightsLoadError:
            if index == last:
                raise
    if error is None:
        raise InferenceUnavailable("the generator weights could not be loaded")
    raise error


def _encoder_needs_reload(encoder: Any, storage: str) -> bool:
    """True when the probe cast the module and the series will not keep that cast."""
    if bool(getattr(encoder, "_canvas_dirty", False)):
        return True
    return bool(getattr(encoder, "_canvas_fp8", False)) and storage != "fp8"


def _mark_encoder(encoder: Any, name: str) -> None:
    try:
        setattr(encoder, name, True)
    except (AttributeError, TypeError):
        return


def _place_encoder(encoder: Any, storage: Literal["fp8", "bf16", "cpu"], torch: Any, tier: Tier) -> None:
    if storage == "fp8":
        already = bool(getattr(encoder, "_canvas_fp8", False))
        if already and _cuda_is_up(torch):
            _move_encoder_cuda(encoder, torch, tier)
            return
        if not already:
            with timed("cast"):
                cast_ok = bool(cast_module_fp8(encoder, torch))
            if cast_ok and _cuda_is_up(torch):
                _mark_encoder(encoder, "_canvas_fp8")
                _move_encoder_cuda(encoder, torch, tier)
                return
        note_error("encoder.fp8_missing")
        storage = "cpu" if _free_vram_bytes() is None else "bf16"
    if storage == "bf16":
        if _cuda_is_up(torch):
            _move_encoder_cuda(encoder, torch, tier)
            return
        _refuse_cpu_encoder(torch)
        encoder.to("cpu")
        return
    if storage == "cpu":
        _refuse_cpu_encoder(torch)
        encoder.to("cpu")
        return
    assert_never(storage)


def _move_encoder_cuda(encoder: Any, torch: Any, tier: Tier) -> None:
    """Move parameters that are still on the CPU. A module already on CUDA stays put."""
    if not _module_on_cuda(encoder):
        with timed("place"):
            encoder.to("cuda")
    _peak("text_encoder", tier)
    require_cuda(encoder, "encoder.device", torch)


def _refuse_cpu_encoder(torch: Any) -> None:
    """Stop when a CUDA machine would encode on the CPU."""
    if _cuda_is_up(torch):
        note_error("encoder.device")
        raise DeviceRefused()


def _cuda_is_up(torch: Any) -> bool:
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except (TypeError, RuntimeError):
        return False


def _encode_prompt_text(tokenizer: Any, encoder: Any, prompt: str, torch: Any) -> Any:
    """One Z-Image prompt embedding: chat template, then the penultimate hidden state."""
    messages = [{"role": "user", "content": prompt}]
    try:
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
    except TypeError:
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    if not isinstance(templated, str):
        raise InferenceUnavailable("the tokenizer did not render a prompt")
    text_inputs = tokenizer(
        [templated], padding="max_length", max_length=512, truncation=True, return_tensors="pt",
    )
    device = _module_device(encoder)
    input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device).bool()
    inference = getattr(torch, "inference_mode", None)
    context = inference() if callable(inference) else _null_context()
    with context:
        hidden = encoder(
            input_ids=input_ids, attention_mask=mask, output_hidden_states=True,
        ).hidden_states[-2]
    return [hidden[0][mask[0]]]


def _release_text_encoder(encoder: Any, tokenizer: Any, torch: Any) -> None:
    """Drop the text encoder, collect it, and return CUDA blocks to the driver."""
    drop_model("text_encoder")
    if encoder is not None:
        mover = getattr(encoder, "to", None)
        if callable(mover):
            try:
                mover("cpu")
            except (TypeError, RuntimeError, AttributeError):
                pass
    del encoder, tokenizer
    gc.collect()
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_cpu(value: Any) -> Any:
    return _moved(value, "cpu")


def _moved(value: Any, device: str) -> Any:
    if isinstance(value, list):
        return [_moved(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_moved(item, device) for item in value)
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            value = detach()
        except (TypeError, RuntimeError):
            pass
    mover = getattr(value, "to", None)
    if not callable(mover):
        return value
    try:
        return mover(device)
    except (TypeError, RuntimeError, AttributeError):
        return value


def _vector(value: Any) -> list[float]:
    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], (int, float)):
            return [float(item) for item in value]
        flat: list[float] = []
        for item in value:
            flat.extend(_vector(item))
        return flat
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            value = detach()
        except (TypeError, RuntimeError):
            pass
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        try:
            value = cpu()
        except (TypeError, RuntimeError):
            pass
    reshape = getattr(value, "reshape", None)
    if callable(reshape):
        try:
            value = reshape(-1)
        except (TypeError, RuntimeError):
            pass
    listed = getattr(value, "tolist", None)
    if callable(listed):
        try:
            return _vector(listed())
        except (TypeError, RuntimeError, ValueError):
            return []
    return []


def _module_device(encoder: Any) -> str:
    device = getattr(encoder, "device", None)
    if device is None:
        return "cpu"
    kind = getattr(device, "type", None)
    if isinstance(kind, str):
        return kind
    text = str(device)
    return "cuda" if "cuda" in text else "cpu"


@contextmanager
def _null_context() -> Iterator[None]:
    yield


def clear_series_embeds() -> None:
    """Drop cached prompt embeddings at the end of a series."""
    global _prompt_embeds, _borrowed_pair
    _prompt_embeds = None
    _borrowed_pair = None
    _cosine_cache.clear()


def _remember_components(pipe: Any, tier: Tier) -> None:
    for name in ("transformer", "vae", "text_encoder"):
        module = getattr(pipe, name, None)
        if module is None:
            continue
        if name == "vae":
            hold_model(name, module)
        else:
            _keep(name, module, tier)


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
    *, drop_text: bool,
) -> tuple[Any, Any]:
    """Load the transformer and the VAE. The text encoder stays out of this pipeline."""
    cooked = _open_cook(diffusers, pipeline_name, folder, dtype, control_file)
    if cooked is not None:
        return cooked
    pipe, controlnet = _open_full(diffusers, pipeline_name, folder, dtype, control_file)
    if drop_text:
        _detach_text_encoder(pipe)
    return pipe, controlnet


def _open_cook(
    diffusers: Any, pipeline_name: str, folder: Path, dtype: object, control_file: Path | None,
) -> tuple[Any, Any] | None:
    """Build a pipeline from the transformer, VAE, and scheduler subfolders."""
    transformer_cls = getattr(diffusers, "ZImageTransformer2DModel", None)
    vae_cls = getattr(diffusers, "AutoencoderKL", None)
    scheduler_cls = getattr(diffusers, "FlowMatchEulerDiscreteScheduler", None)
    if transformer_cls is None or vae_cls is None or scheduler_cls is None:
        return None
    if not all(callable(getattr(cls, "from_pretrained", None)) for cls in (transformer_cls, vae_cls, scheduler_cls)):
        return None
    kind = _pipeline_type(diffusers, pipeline_name)
    if not isinstance(kind, type):
        return None
    torch = _import_torch()
    # The transformer is cast to FP8 before it stays on the card. Mapping the
    # BF16 file straight onto CUDA would hold that larger copy during the sample.
    # The VAE stays in BF16 and is small enough to map onto the card here.
    vae_device = "cuda" if _cuda_is_up(torch) else "cpu"
    transformer = _load_component(transformer_cls.from_pretrained, folder / "transformer", dtype, "cpu")
    vae = _load_component(vae_cls.from_pretrained, folder / "vae", dtype, vae_device)
    scheduler = _load(scheduler_cls.from_pretrained, folder / "scheduler", local_files_only=True)
    controlnet = None
    try:
        if pipeline_name == PIPELINE_CONTROL or pipeline_name == PIPELINE_CONTROL_INPAINT:
            if control_file is None:
                raise InferenceUnavailable("control weights are missing")
            control_cls = getattr(diffusers, "ZImageControlNetModel", None)
            if control_cls is None or not callable(getattr(control_cls, "from_single_file", None)):
                return None
            controlnet = control_cls.from_single_file(control_file, torch_dtype=dtype)
            pipe = kind(
                scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None,
                transformer=transformer, controlnet=controlnet,
            )
        else:
            pipe = kind(
                scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer,
            )
    except TypeError:
        return None
    return pipe, controlnet


def _open_full(
    diffusers: Any, pipeline_name: str, folder: Path, dtype: object, control_file: Path | None,
) -> tuple[Any, Any]:
    """Load one pipeline from disk. Tests use this when the component classes are absent."""
    if pipeline_name == PIPELINE_TEXT:
        pipe = _load_component(diffusers.ZImagePipeline.from_pretrained, folder, dtype, "cpu")
        return pipe, None
    if pipeline_name == PIPELINE_IMG2IMG:
        pipe = _load_component(diffusers.ZImageImg2ImgPipeline.from_pretrained, folder, dtype, "cpu")
        return pipe, None
    if pipeline_name == PIPELINE_INPAINT:
        pipe = _load_component(diffusers.ZImageInpaintPipeline.from_pretrained, folder, dtype, "cpu")
        return pipe, None
    if pipeline_name == PIPELINE_CONTROL or pipeline_name == PIPELINE_CONTROL_INPAINT:
        if control_file is None:
            raise InferenceUnavailable("control weights are missing")
        controlnet = diffusers.ZImageControlNetModel.from_single_file(control_file, torch_dtype=dtype)
        kind = _pipeline_type(diffusers, pipeline_name)
        pipe = _load_component(kind.from_pretrained, folder, dtype, "cpu", controlnet=controlnet)
        return pipe, controlnet
    raise InferenceUnavailable(f"unknown pipeline {pipeline_name}")


def _detach_text_encoder(pipe: Any) -> None:
    """Move the text encoder to CPU and drop it from the cook pipeline."""
    components = getattr(pipe, "components", None)
    for name in ("text_encoder", "text_encoder_2"):
        module = getattr(pipe, name, None)
        if module is not None:
            mover = getattr(module, "to", None)
            if callable(mover):
                try:
                    mover("cpu")
                except (TypeError, RuntimeError, AttributeError):
                    pass
        try:
            setattr(pipe, name, None)
        except (AttributeError, TypeError):
            pass
        if isinstance(components, dict):
            components.pop(name, None)


def _ensure_prompt_embed(
    folder: Path, tier: Tier, prompt: str, lora: UserLora | None, lora_strength: float | None,
) -> None:
    """Encode one prompt before cook. A 16 GB card keeps the full pipeline unloaded."""
    if _prompt_embeds is not None and prompt in _prompt_embeds:
        return
    if not _cook_classes_ready() and not _transformers_importable():
        return
    try:
        prepare_series_embeds(folder, tier, (prompt,), lora=lora, lora_strength=lora_strength)
    except DeviceRefused:
        raise
    except (WeightsLoadError, ImportError, OSError, RuntimeError, ValueError, AttributeError, TypeError) as exc:
        if _cook_classes_ready():
            note_error("encoder.standalone")
            raise InferenceUnavailable("the text encoder could not be loaded on its own") from exc
        return


def _cook_classes_ready() -> bool:
    try:
        diffusers = _import_diffusers()
    except ImportError:
        return False
    transformer_cls = getattr(diffusers, "ZImageTransformer2DModel", None)
    return callable(getattr(transformer_cls, "from_pretrained", None))


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
        _enter("cook", tier)
        note_vram("pipeline_load", "before")
        try:
            with timed("pipeline_load"):
                with timed("load"):
                    pipe, controlnet = _open_pipeline(
                        diffusers, pipeline_name, folder, dtype, control_file,
                        drop_text=tier is Tier.GB16 or _prompt_embeds is not None,
                    )
            note_vram("pipeline_load", "after", pipe)
            held = _Held(
                folder=key[0], tier=tier, dtype=dtype, control=key[3], pipeline_name=pipeline_name,
                lora_key=None, pipe=pipe, controlnet=controlnet, torch=torch,
            )
            _slot = held
            with timed("pipeline_place"):
                with timed("place"):
                    _place_and_bind(pipe, tier, lora, lora_strength)
            _remember_components(pipe, tier)
            _shelter(pipe)
            _peak("cook", tier)
        except Exception:
            release_pipeline()
            raise
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


def _shelter(pipe: Any) -> None:
    """Park the cast picture models in RAM. A resident pipeline leaves VRAM until its stage."""
    if not hot.keeps_main_tier():
        return
    for name in ("transformer", "text_encoder", "vae", "controlnet"):
        module = getattr(pipe, name, None)
        if module is not None:
            hot.remember(name, module)


def render(plan: CallPlan, folder: Path, screened: ScreenedInput, width: int, height: int, tier: Tier, *,
           seed: int = 0, control_file: Path | None = None,
           strength: float | None = None, lora: UserLora | None = None,
           lora_strength: float | None = None, negative: str = "") -> RawImage:
    """Run one pinned snapshot. ``folder`` is a local diffusers tree."""
    hot.note_tier()
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
        "output_type": "latent",
    }
    embed = None if _prompt_embeds is None else _prompt_embeds.get(screened.prompt)
    if embed is None:
        _ensure_prompt_embed(folder, tier, screened.prompt, lora, lora_strength)
        embed = None if _prompt_embeds is None else _prompt_embeds.get(screened.prompt)
    if embed is not None:
        common["prompt"] = None
        common["prompt_embeds"] = _moved(embed, "cuda" if torch.cuda.is_available() else "cpu")
    common.update(guidance)
    common.update(_negative_argument(negative))
    pipeline_name = plan.pipeline_class
    failed = False
    with _lock:
        try:
            pipe = _pipeline_for(
                diffusers, torch, pipeline_name, folder, dtype, tier, lora, lora_strength, control_file,
            )
            call = _accepting(pipe, common)
            _disable_vae_upcast(pipe)
            resident = getattr(pipe, "_canvas_placement", "resident") == "resident"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if resident:
                hot.activate("transformer", device)
                if getattr(pipe, "controlnet", None) is not None:
                    hot.activate("controlnet", device)
            try:
                if pipeline_name == PIPELINE_TEXT:
                    result = _timed(pipe, **call)
                elif pipeline_name == PIPELINE_IMG2IMG:
                    result = _timed(
                        pipe, image=_pil(_frame(screened, "init")), strength=applied_strength(strength), **call,
                    )
                elif pipeline_name == PIPELINE_INPAINT:
                    result = _timed(
                        pipe, image=_pil(_frame(screened, "init")), mask_image=_pil(_frame(screened, "mask")), **call,
                    )
                elif pipeline_name == PIPELINE_CONTROL:
                    role = _control_role(screened)
                    result = _timed(
                        pipe, control_image=_pil(_frame(screened, role)), controlnet_conditioning_scale=0.75, **call,
                    )
                elif pipeline_name == PIPELINE_CONTROL_INPAINT:
                    result = _timed(
                        pipe,
                        image=_pil(_frame(screened, "init")),
                        mask_image=_pil(_frame(screened, "mask")),
                        control_image=_pil(_frame(screened, _control_role(screened))),
                        controlnet_conditioning_scale=0.75,
                        **call,
                    )
                else:
                    raise InferenceUnavailable(f"unknown pipeline {pipeline_name}")
            finally:
                if resident:
                    hot.deactivate("transformer")
                    hot.deactivate("controlnet")
            return _image_from_result(pipe, result, torch)
        except Exception:
            failed = True
            raise
        finally:
            _remember_vram_peak()
            if not _should_retain():
                release_pipeline()
            elif failed:
                park_pipeline()


def _timed(pipe: Any, **kwargs: Any) -> Any:
    """Run one pipeline call and remember the seconds for the JSON report."""
    started = time.perf_counter()
    result = pipe(**kwargs)
    seconds = time.perf_counter() - started
    note_image_seconds(seconds)
    note_stage("sample", seconds)
    return result


def _disable_vae_upcast(pipe: Any) -> None:
    """Keep the VAE in its loaded dtype. A missing config is left alone."""
    vae = getattr(pipe, "vae", None)
    config = getattr(vae, "config", None)
    if config is None or not hasattr(config, "force_upcast"):
        return
    try:
        config.force_upcast = False
    except (AttributeError, TypeError):
        return


def _image_from_result(pipe: Any, result: Any, torch: Any) -> RawImage:
    """Pixels from a pipeline result. A latent is decoded on CUDA in BF16."""
    first = _first_output(result)
    if isinstance(first, Image.Image):
        with timed("to_pil"):
            image = first.convert("RGB")
        return RawImage(image.width, image.height, image.tobytes())
    with timed("vae_decode"):
        decoded = _decode_latents(pipe, first, torch)
    with timed("to_pil"):
        image = _latents_to_pil(pipe, decoded)
    return RawImage(image.width, image.height, image.tobytes())


def _first_output(result: Any) -> Any:
    images = getattr(result, "images", result)
    if isinstance(images, Image.Image):
        return images
    if isinstance(images, (list, tuple)):
        return images[0]
    return images


def _decode_latents(pipe: Any, latents: Any, torch: Any) -> Any:
    """Decode one latent on the VAE's device and dtype. Tile when the card is near the 14 GB line.

    On CUDA the VAE is moved to BF16 first. The latents are cast to that same
    device and dtype after scaling, immediately before ``decode``. A VAE that
    stays in FP32 keeps the latents in FP32.
    """
    vae = getattr(pipe, "vae", None)
    if vae is None:
        raise InferenceUnavailable("the generator has no vae")
    if _needs_vae_tiling():
        tiling = getattr(vae, "enable_tiling", None)
        if callable(tiling):
            tiling()
    device = "cuda" if _cuda_is_up(torch) else "cpu"
    hot.activate("vae", device)
    try:
        vae = _vae_for_decode(pipe, vae, torch)
        config = getattr(vae, "config", None)
        scaling = getattr(config, "scaling_factor", 0.3611) if config is not None else 0.3611
        shift = getattr(config, "shift_factor", 0.1159) if config is not None else 0.1159
        try:
            scaled = (latents / scaling) + shift
        except TypeError:
            scaled = latents
        scaled = _latents_like_vae(vae, scaled)
        decoded = vae.decode(scaled)
    finally:
        hot.deactivate("vae")
    sample = getattr(decoded, "sample", None)
    return decoded if sample is None else sample


def _vae_for_decode(pipe: Any, vae: Any, torch: Any) -> Any:
    """Put the VAE on CUDA in BF16 when CUDA is up. Otherwise leave it where it is."""
    if not _cuda_is_up(torch):
        return vae
    dtype = getattr(torch, "bfloat16", None)
    mover = getattr(vae, "to", None)
    if not callable(mover):
        return vae
    try:
        moved = mover("cuda", dtype) if dtype is not None else mover("cuda")
    except TypeError:
        try:
            moved = mover("cuda")
        except (TypeError, RuntimeError, AttributeError):
            return vae
    except (RuntimeError, AttributeError):
        return vae
    if moved is None:
        return vae
    try:
        pipe.vae = moved
    except (AttributeError, TypeError):
        pass
    return moved


def _latents_like_vae(vae: Any, latents: Any) -> Any:
    """Cast latents onto the VAE parameter device and dtype. A fake with no parameters is unchanged."""
    device, dtype = _parameter_placement(vae)
    if device is None and dtype is None:
        return latents
    return _cast_latents(latents, device, dtype)


def _parameter_placement(module: Any) -> tuple[Any, Any]:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return None, None
    try:
        for param in parameters():
            device = getattr(param, "device", None)
            dtype = getattr(param, "dtype", None)
            if device is not None or dtype is not None:
                return device, dtype
    except (TypeError, RuntimeError):
        return None, None
    return None, None


def _cast_latents(value: Any, device: Any, dtype: Any) -> Any:
    if isinstance(value, list):
        return [_cast_latents(item, device, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_cast_latents(item, device, dtype) for item in value)
    mover = getattr(value, "to", None)
    if not callable(mover):
        return value
    try:
        return mover(device=device, dtype=dtype)
    except TypeError:
        try:
            return mover(device, dtype)
        except (TypeError, RuntimeError, AttributeError):
            return value
    except (RuntimeError, AttributeError):
        return value


def _needs_vae_tiling() -> bool:
    """Tile when another 2 GiB would pass 14 GiB, or when less than 3 GiB is free."""
    if allocated_bytes() + (2 * 1024**3) > PEAK_LIMIT_BYTES:
        return True
    free = _free_vram_bytes()
    return free is not None and free < 3 * 1024**3


def _latents_to_pil(pipe: Any, decoded: Any) -> Image.Image:
    processor = getattr(pipe, "image_processor", None)
    post = getattr(processor, "postprocess", None) if processor is not None else None
    if callable(post):
        images = post(decoded, output_type="pil")
        image = images[0] if isinstance(images, (list, tuple)) else images
        if isinstance(image, Image.Image):
            return image.convert("RGB")
    if isinstance(decoded, Image.Image):
        return decoded.convert("RGB")
    raise InferenceUnavailable("the vae did not return an image")


def _negative_argument(negative: str) -> dict[str, str]:
    """The negative text, kept aside until the pipeline call is known."""
    text = negative.strip()
    if not text:
        return {}
    return {"negative_prompt": text}


def _accepting(pipe: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    """Drop ``negative_prompt`` when this pipeline call does not take one."""
    if "negative_prompt" not in arguments:
        return arguments
    try:
        parameters = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return {key: value for key, value in arguments.items() if key != "negative_prompt"}
    names = set(parameters)
    if "negative_prompt" in names or any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return arguments
    return {key: value for key, value in arguments.items() if key != "negative_prompt"}


def _remember_vram_peak() -> None:
    """Write the CUDA allocation peak onto the timing line, in MiB."""
    peak = peak_allocated_bytes()
    if not peak or peak <= 0:
        return
    note_peak_mib(peak // (1024 * 1024))


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


def _mark_placement(pipe: Any, kind: str) -> None:
    """Remember whether this pipeline owns its CUDA copies or an offload hook does."""
    try:
        setattr(pipe, "_canvas_placement", kind)
    except (AttributeError, TypeError):
        return


def _place(pipe: Any, tier: Tier) -> None:
    """16 GB offloads unless the whole pipeline fits with room for activations.

    Keeping a tight pipeline resident fills a 16 GB card and spills into system
    memory. 24 GB keeps the pipeline resident.
    """
    if tier is Tier.GB16:
        torch = _import_torch()
        with timed("cast"):
            cast = install_fp8_storage(pipe, torch, _layerwise_casting())
        if "transformer" not in cast:
            note_error("weights.bf16")
        note_tag("tier", "gb16")
        note_tag("weights", "fp8" if cast else "bf16")
        free = _free_vram_bytes()
        room = max(0, PEAK_LIMIT_BYTES - allocated_bytes())
        budget = room if free is None else min(int(free), room)
        if resident_fits(pipe, budget):
            note_tag("placement", "resident")
            _mark_placement(pipe, "resident")
            pipe.to("cuda")
            _peak("cook", tier)
            return
        if model_offload_fits(pipe, budget):
            note_tag("placement", "model_offload")
            _mark_placement(pipe, "offload")
            install_model_offload(pipe)
            _peak("cook", tier)
            return
        try:
            install_low_vram_offload(pipe)
        except OffloadUnavailable as exc:
            raise InferenceUnavailable("this pipeline cannot offload weights") from exc
        note_tag("placement", "group_offload")
        _mark_placement(pipe, "offload")
        _peak("cook", tier)
        return
    if tier is Tier.GB24:
        note_tag("tier", "gb24")
        note_tag("weights", "bf16")
        note_tag("placement", "resident")
        _mark_placement(pipe, "resident")
        pipe.to("cuda")
        return
    assert_never(tier)


def _layerwise_casting() -> Any:
    """diffusers' layerwise casting hook, or None on a build without it."""
    try:
        from diffusers.hooks import apply_layerwise_casting
    except ImportError:
        return None
    return apply_layerwise_casting


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


def _import_transformers() -> Any:
    import transformers
    return transformers


def _transformers_importable() -> bool:
    try:
        _import_transformers()
    except ImportError:
        return False
    return True


register_release(release_pipeline)
