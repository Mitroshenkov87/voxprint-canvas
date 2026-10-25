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
    if tier is Tier.GB16:
        _leave("cook", tier)


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
    global _prompt_embeds
    torch = _import_torch()
    _enter("text_encoder", tier)
    tokenizer: Any = None
    encoder: Any = None
    try:
        score = _encoder_cosine(folder, tier, torch)
        storage = choose_encoder_storage(tier, _fp8_ready(torch), score, _free_vram_bytes())
        tokenizer, encoder = _load_text_pair(folder, torch)
        _place_encoder(encoder, storage, torch, tier)
        _keep("text_encoder", encoder, tier)
        note_vram("text_encoder", "before", encoder)
        encoded: dict[str, Any] = {}
        for prompt in prompts:
            encoded[prompt] = _to_cpu(_encode_prompt_text(tokenizer, encoder, prompt, torch))
        _prompt_embeds = encoded
    finally:
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
    """Minimum cosine of FP8 embeddings against a CPU BF16 pass. None outside 16 GB."""
    if tier is not Tier.GB16 or not _fp8_ready(torch):
        return None
    try:
        reference = _probe_vectors(folder, torch, "cpu")
        candidate = _probe_vectors(folder, torch, "fp8")
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError, WeightsLoadError):
        return 0.0
    if len(reference) != len(candidate) or not reference:
        return 0.0
    scores = [cosine(left, right) for left, right in zip(reference, candidate, strict=True)]
    return min(scores) if scores else 0.0


def _probe_vectors(folder: Path, torch: Any, storage: Literal["cpu", "fp8"]) -> list[list[float]]:
    tokenizer, encoder = _load_text_pair(folder, torch)
    try:
        if storage == "cpu":
            encoder.to("cpu")
        elif storage == "fp8":
            if not cast_module_fp8(encoder, torch):
                raise RuntimeError("fp8 cast failed")
            if torch.cuda.is_available():
                encoder.to("cuda")
        else:
            assert_never(storage)
        vectors: list[list[float]] = []
        for prompt in _PROBE_PROMPTS:
            encoded = _to_cpu(_encode_prompt_text(tokenizer, encoder, prompt, torch))
            vectors.append(_vector(encoded))
        return vectors
    finally:
        _release_text_encoder(encoder, tokenizer, torch)


def _load_text_pair(folder: Path, torch: Any) -> tuple[Any, Any]:
    """The tokenizer and the text encoder. The picture pipeline stays unloaded."""
    transformers = _import_transformers()
    tokenizer = _load(
        transformers.AutoTokenizer.from_pretrained, folder / "tokenizer", local_files_only=True,
    )
    try:
        encoder = _load(
            transformers.AutoModel.from_pretrained,
            folder / "text_encoder",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    except TypeError:
        encoder = _load(
            transformers.AutoModel.from_pretrained,
            folder / "text_encoder",
            dtype=torch.bfloat16,
            local_files_only=True,
        )
    return tokenizer, encoder


def _place_encoder(encoder: Any, storage: Literal["fp8", "bf16", "cpu"], torch: Any, tier: Tier) -> None:
    if storage == "fp8":
        if not cast_module_fp8(encoder, torch) or not torch.cuda.is_available():
            note_error("encoder.fp8_missing")
            storage = "cpu" if _free_vram_bytes() is None else "bf16"
        else:
            encoder.to("cuda")
            _peak("text_encoder", tier)
            return
    if storage == "bf16":
        if torch.cuda.is_available():
            encoder.to("cuda")
            _peak("text_encoder", tier)
            return
        encoder.to("cpu")
        return
    if storage == "cpu":
        encoder.to("cpu")
        return
    assert_never(storage)


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
    global _prompt_embeds
    _prompt_embeds = None


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
    transformer = _load(
        transformer_cls.from_pretrained, folder / "transformer", torch_dtype=dtype, local_files_only=True,
    )
    vae = _load(vae_cls.from_pretrained, folder / "vae", torch_dtype=dtype, local_files_only=True)
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
                _place_and_bind(pipe, tier, lora, lora_strength)
            _remember_components(pipe, tier)
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


def render(plan: CallPlan, folder: Path, screened: ScreenedInput, width: int, height: int, tier: Tier, *,
           seed: int = 0, control_file: Path | None = None,
           strength: float | None = None, lora: UserLora | None = None,
           lora_strength: float | None = None, negative: str = "") -> RawImage:
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
            image = result.images[0].convert("RGB")
            return RawImage(image.width, image.height, image.tobytes())
        except Exception:
            failed = True
            raise
        finally:
            _remember_vram_peak()
            if failed or not _should_retain():
                release_pipeline()


def _timed(pipe: Any, **kwargs: Any) -> Any:
    """Run one pipeline call and remember the seconds for the JSON report."""
    started = time.perf_counter()
    result = pipe(**kwargs)
    seconds = time.perf_counter() - started
    note_image_seconds(seconds)
    note_stage("sample", seconds)
    return result


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


def _place(pipe: Any, tier: Tier) -> None:
    """16 GB offloads unless the whole pipeline fits with room for activations.

    Keeping a tight pipeline resident fills a 16 GB card and spills into system
    memory. 24 GB keeps the pipeline resident.
    """
    if tier is Tier.GB16:
        torch = _import_torch()
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
            pipe.to("cuda")
            _peak("cook", tier)
            return
        if model_offload_fits(pipe, budget):
            note_tag("placement", "model_offload")
            install_model_offload(pipe)
            _peak("cook", tier)
            return
        try:
            install_low_vram_offload(pipe)
        except OffloadUnavailable as exc:
            raise InferenceUnavailable("this pipeline cannot offload weights") from exc
        note_tag("placement", "group_offload")
        _peak("cook", tier)
        return
    if tier is Tier.GB24:
        note_tag("tier", "gb24")
        note_tag("weights", "bf16")
        note_tag("placement", "resident")
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
