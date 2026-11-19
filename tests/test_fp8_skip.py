"""FP8 storage keeps the model's precision-sensitive layers in compute dtype."""
from __future__ import annotations

from typing import Any

from canvas.pipeline.offload import FP8_DEFAULT_SKIP, fp8_skip_patterns, install_fp8_storage


class _Torch:
    float8_e4m3fn = "fp8"
    bfloat16 = "bf16"


class _Transformer:
    _skip_layerwise_casting_patterns = ["t_embedder", "cap_embedder"]


class _Encoder:
    pass


class _Pipe:
    def __init__(self) -> None:
        self.transformer = _Transformer()
        self.text_encoder = _Encoder()


def test_model_patterns_are_passed_to_the_cast() -> None:
    seen: dict[str, Any] = {}

    def cast(module: Any, storage_dtype: Any, compute_dtype: Any, skip_modules_pattern: Any) -> None:
        seen[type(module).__name__] = skip_modules_pattern

    assert install_fp8_storage(_Pipe(), _Torch, cast) == ("transformer", "text_encoder")
    assert "t_embedder" in seen["_Transformer"] and "cap_embedder" in seen["_Transformer"]
    assert seen["_Encoder"] == FP8_DEFAULT_SKIP


def test_defaults_kept_and_no_duplicates() -> None:
    class _M:
        _skip_layerwise_casting_patterns = "norm"

    assert fp8_skip_patterns(_M()) == FP8_DEFAULT_SKIP
