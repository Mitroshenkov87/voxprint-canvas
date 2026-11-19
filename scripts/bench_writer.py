"""Time the scenario writer inside the installed app environment.

The folder is the MDL view. Loading uses the app's own causal loader.
One stdout line per timed run. A warmup generate is not one of those lines.

``fp8_perrow_compile`` quantizes with
``Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())`` and
``torch.compile``. When torchao is not installed that variant prints a skip
line and exits 0.

Example::

    python scripts/bench_writer.py --variant app --attn sdpa --tokens 128 --runs 3
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal, assert_never

from canvas.infra.paths import models_dir
from canvas.models.modes import Tier, select_tier
from canvas.models.registry import load_catalog
from canvas.models.snapshots import snapshot_ready
from canvas.pipeline.device_guard import place_on_cuda, weights_on_cuda
from canvas.pipeline.enhance import (
    WRITER_ID,
    QwenRewriter,
    _causal_model,
    _generation_libs,
    writer_spec,
)
from canvas.pipeline.generation_weights import drop_offload_hooks, kv_cache_label

try:
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        PerRow,
        quantize_,
    )
except ImportError:
    try:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_
        from torchao.quantization.granularity import PerRow
    except ImportError:
        Float8DynamicActivationFloat8WeightConfig = None
        PerRow = None
        quantize_ = None

Variant = Literal["plain_bf16", "app", "fp8_perrow_compile"]
Attn = Literal["sdpa", "eager", "flash_attention_2"]
_VARIANTS: tuple[Variant, ...] = ("plain_bf16", "app", "fp8_perrow_compile")
_ATTN: tuple[Attn, ...] = ("sdpa", "eager", "flash_attention_2")
_USER = "Describe a sunlit street in one sentence."
_SYSTEM = "Reply with one sentence."


def main(argv: list[str] | None = None) -> int:
    """Load the writer view and print one bench line per timed run."""
    args = _parser().parse_args(argv)
    variant = _variant(args.variant)
    attn = _attn(args.attn)
    tokens = int(args.tokens)
    runs = int(args.runs)
    if tokens < 1 or runs < 1:
        _emit(variant=variant, attn=attn, error="tokens_and_runs_must_be_positive")
        return 2
    if variant == "fp8_perrow_compile" and not _torchao_ready():
        _emit(variant=variant, attn=attn, skipped="torchao_missing")
        return 0
    folder = _view()
    if folder is None:
        _emit(variant=variant, attn=attn, error="view_missing", id=WRITER_ID)
        return 2
    torch, transformers = _generation_libs()
    _install_attn(transformers, attn)
    started = time.perf_counter()
    try:
        model, tokenizer = _load(variant, folder, tokens, torch, transformers)
    except Exception as exc:
        _emit(variant=variant, attn=attn, error=_flat(exc))
        return 1
    load_s = time.perf_counter() - started
    try:
        encoded = _encode(tokenizer, model, torch)
        _generate(model, torch, encoded, tokens)
        for _run in range(runs):
            _timed_line(variant, attn, model, torch, encoded, tokens, load_s)
    except Exception as exc:
        _emit(variant=variant, attn=attn, error=_flat(exc))
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Time the installed scenario writer.")
    parser.add_argument("--variant", required=True, choices=_VARIANTS)
    parser.add_argument("--attn", required=True, choices=_ATTN)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    return parser


def _variant(value: str) -> Variant:
    if value == "plain_bf16" or value == "app" or value == "fp8_perrow_compile":
        return value
    raise ValueError(value)


def _attn(value: str) -> Attn:
    if value == "sdpa" or value == "eager" or value == "flash_attention_2":
        return value
    raise ValueError(value)


def _torchao_ready() -> bool:
    return (
        Float8DynamicActivationFloat8WeightConfig is not None
        and PerRow is not None
        and quantize_ is not None
    )


def _view() -> Path | None:
    spec = writer_spec(load_catalog())
    return snapshot_ready(spec, models_dir())


def _install_attn(transformers: Any, attn: Attn) -> None:
    """Pass ``attn_implementation`` through the app's ``from_pretrained`` attempts."""
    cls = transformers.AutoModelForCausalLM
    original = cls.from_pretrained

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs["attn_implementation"] = attn
        return original(*args, **kwargs)

    cls.from_pretrained = wrapped


def _load(
    variant: Variant, folder: Path, tokens: int, torch: Any, transformers: Any,
) -> tuple[Any, Any]:
    if variant == "plain_bf16":
        return _load_plain(folder, torch, transformers)
    if variant == "app":
        return _load_app(folder, tokens, torch)
    if variant == "fp8_perrow_compile":
        return _load_fp8(folder, torch, transformers)
    assert_never(variant)


def _load_plain(folder: Path, torch: Any, transformers: Any) -> tuple[Any, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = _causal_model(transformers, torch, folder, device)
    drop_offload_hooks(model)
    if device != "cpu" and not weights_on_cuda(model):
        model = place_on_cuda(model)
    return model, tokenizer


def _load_app(folder: Path, tokens: int, torch: Any) -> tuple[Any, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rewriter = QwenRewriter(
        folder,
        folder,
        max_new_tokens=tokens,
        device=device,
        tier=_tier(torch),
        stage="writer",
    )
    rewriter._load()
    if rewriter._model is None or rewriter._tokenizer is None:
        raise RuntimeError("writer_not_loaded")
    return rewriter._model, rewriter._tokenizer


def _load_fp8(folder: Path, torch: Any, transformers: Any) -> tuple[Any, Any]:
    if not _torchao_ready():
        raise RuntimeError("torchao_missing")
    model, tokenizer = _load_plain(folder, torch, transformers)
    config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    quantize_(model, config)
    compiled = torch.compile(model)
    return compiled, tokenizer


def _tier(torch: Any) -> Tier:
    if not torch.cuda.is_available():
        return Tier.GB16
    props = torch.cuda.get_device_properties(0)
    total = getattr(props, "total_memory", None)
    if total is None:
        total = getattr(props, "total_mem", 0)
    chosen = select_tier(float(total) / (1024**3))
    if chosen is None:
        return Tier.GB16
    return chosen


def _encode(tokenizer: Any, model: Any, torch: Any) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER},
    ]
    try:
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    if not isinstance(templated, str):
        raise RuntimeError("tokenizer_template")
    encoded = tokenizer(templated, return_tensors="pt")
    device = _param_device(model) or ("cuda" if torch.cuda.is_available() else "cpu")
    return {key: value.to(device) for key, value in encoded.items()}


def _timed_line(
    variant: Variant,
    attn: Attn,
    model: Any,
    torch: Any,
    encoded: dict[str, Any],
    tokens: int,
    load_s: float,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = _generate(model, torch, encoded, tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    count = _new_tokens(output, int(encoded["input_ids"].shape[-1]))
    rate = 0.0 if elapsed <= 0 else count / elapsed
    _emit(
        variant=variant,
        attn=attn,
        params=_param_counts(model),
        kv=_kv_device(output),
        load_s=f"{load_s:.3f}",
        tok_s=f"{rate:.2f}",
        peak_vram_mib=_peak_mib(torch),
    )


def _generate(model: Any, torch: Any, encoded: dict[str, Any], tokens: int) -> Any:
    use_cache = kv_cache_label(model) == "on"
    kwargs: dict[str, Any] = {
        "max_new_tokens": tokens,
        "min_new_tokens": tokens,
        "do_sample": False,
        "use_cache": use_cache,
        "return_dict_in_generate": True,
    }
    with torch.inference_mode():
        try:
            return model.generate(**encoded, **kwargs)
        except TypeError:
            kwargs.pop("min_new_tokens")
            return model.generate(**encoded, **kwargs)


def _new_tokens(output: Any, start: int) -> int:
    sequences = getattr(output, "sequences", output)
    shape = getattr(sequences, "shape", None)
    if shape is None:
        return 0
    try:
        return max(0, int(shape[-1]) - start)
    except (TypeError, ValueError, IndexError):
        return 0


def _param_counts(model: Any) -> str:
    counts: Counter[str] = Counter()
    flags: set[str] = set()
    parameters = getattr(model, "parameters", None)
    listed = list(parameters()) if callable(parameters) else []
    for param in listed:
        device = _device_type(param)
        dtype = _dtype_name(getattr(param, "dtype", None))
        counts[f"{device}_{dtype}"] += 1
        if device == "cpu" or device == "meta":
            flags.add(device)
    if not counts:
        body = "none"
    else:
        body = ",".join(f"{key}={counts[key]}" for key in sorted(counts))
    flag = ",".join(sorted(flags)) if flags else "none"
    return f"{body} flag={flag}"


def _kv_device(output: Any) -> str:
    past = getattr(output, "past_key_values", None)
    found: list[Any] = []
    _tensors(past, set(), found)
    if not found:
        return "missing"
    devices = sorted({_device_type(tensor) for tensor in found})
    dtypes = sorted({_dtype_name(getattr(tensor, "dtype", None)) for tensor in found})
    return f"{'+'.join(devices)}:{'+'.join(dtypes)}"


def _tensors(value: Any, seen: set[int], found: list[Any], depth: int = 0) -> None:
    if value is None or depth > 6:
        return
    key = id(value)
    if key in seen:
        return
    seen.add(key)
    if hasattr(value, "device") and hasattr(value, "dtype") and hasattr(value, "shape"):
        found.append(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _tensors(item, seen, found, depth + 1)
        return
    for name in ("key_cache", "value_cache", "layers", "keys", "values"):
        child = getattr(value, name, None)
        if child is not None and child is not value:
            _tensors(child, seen, found, depth + 1)


def _param_device(model: Any) -> str:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return ""
    for param in parameters():
        return _device_type(param)
    return ""


def _device_type(value: Any) -> str:
    device = getattr(value, "device", None)
    kind = getattr(device, "type", None)
    if kind in {"cuda", "cpu", "meta"}:
        return str(kind)
    return "none"


def _dtype_name(dtype: Any) -> str:
    if dtype is None:
        return "none"
    text = str(dtype).lower()
    if "float8" in text or "fp8" in text:
        return "fp8"
    if "bfloat16" in text or text == "bf16":
        return "bf16"
    if "float16" in text or text == "fp16":
        return "fp16"
    if "float32" in text or text == "fp32":
        return "fp32"
    return "other"


def _peak_mib(torch: Any) -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated() // (1024 * 1024))


def _emit(**fields: object) -> None:
    print("bench " + " ".join(f"{key}={fields[key]}" for key in fields), flush=True)


def _flat(exc: BaseException) -> str:
    text = f"{type(exc).__name__}:{exc}"
    return "_".join(text.split())[:200]


if __name__ == "__main__":
    sys.exit(main())
