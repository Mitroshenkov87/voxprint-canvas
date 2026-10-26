"""Resident dtype for generation. A 4B model stays BF16. A larger one asks for FP8."""
from canvas.pipeline.generation_weights import generation_dtype

_GIB = 1024**3


def test_generation_stays_bf16_when_the_fp8_size_fits() -> None:
    assert generation_dtype(0) == "bf16"
    assert generation_dtype(8 * _GIB) == "bf16"
    assert generation_dtype(10 * _GIB) == "bf16"


def test_generation_uses_fp8_when_the_fp8_size_exceeds_five_gigabytes() -> None:
    assert generation_dtype(12 * _GIB) == "fp8"
    assert generation_dtype(15 * _GIB) == "fp8"
