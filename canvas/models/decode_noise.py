"""Fail-closed noise check for a decoded frame.

Root cause (rc45): the latent-to-pixel math matches stock diffusers, but the
VAE was force-cast with a bare ``to("cuda", bfloat16)`` at decode time and FP8
casting hooks were dropped without upcasting the stored weights first, so a
decode could run on down-cast or still-FP8 weights and write fine grain. This
assert trips on that grain before any picture is saved.
"""
from __future__ import annotations

import numpy
from PIL import Image

DECODE_NOISE_CODE = "inference.decode_noise"

# A white-noise frame keeps about 8/9 of its variance in the 3x3 blur residual;
# a real or smooth picture keeps well under half. The std gate keeps a flat
# frame (a different failure) out of this check.
_STD_MIN = 0.02
_HF_RATIO_MIN = 0.5


class DecodeNoiseError(Exception):
    """The decoder produced noise, so no picture was saved. ``code`` is a stable catalog key."""

    def __init__(self, code: str = DECODE_NOISE_CODE, **details: str) -> None:
        super().__init__(code)
        self.code = code
        self.details = details

    def __str__(self) -> str:
        from canvas.app.i18n import text

        return text(self.code, "en", **self.details)


def noise_metrics(frame: object) -> dict[str, float]:
    """Per-channel std and the high-frequency share of the variance of one frame.

    Accepts a PIL image or an ``H x W x C`` numpy array, uint8 or float in
    ``[0, 1]``. The high-frequency measure is the mean squared residual after a
    3x3 box blur, divided by the frame variance.
    """
    array = _as_float_hwc(frame)
    std = float(numpy.mean(array.std(axis=(0, 1))))
    residual = array - _box_blur3(array)
    variance = float(numpy.mean((array - array.mean(axis=(0, 1))) ** 2))
    hf_energy = float(numpy.mean(residual**2))
    ratio = hf_energy / variance if variance > 1e-12 else 0.0
    return {"std": std, "hf_energy": hf_energy, "hf_ratio": ratio}


def is_noise(metrics: dict[str, float]) -> bool:
    """True when the frame looks like pure grain: enough variation, nearly all of it high-frequency."""
    return metrics["std"] >= _STD_MIN and metrics["hf_ratio"] >= _HF_RATIO_MIN


def assert_not_noise(frame: object, *, code: str = DECODE_NOISE_CODE) -> dict[str, float]:
    """Raise :class:`DecodeNoiseError` when the decoded frame is pure grain. Returns the metrics."""
    metrics = noise_metrics(frame)
    if is_noise(metrics):
        raise DecodeNoiseError(
            code,
            std=f"{metrics['std']:.4f}",
            hf_ratio=f"{metrics['hf_ratio']:.3f}",
        )
    return metrics


def _as_float_hwc(frame: object) -> numpy.ndarray:
    """``H x W x C`` float32 in ``[0, 1]``. A PIL image or an array is accepted."""
    if isinstance(frame, Image.Image):
        array = numpy.asarray(frame.convert("RGB"), dtype=numpy.float32) / 255.0
    else:
        array = numpy.asarray(frame, dtype=numpy.float32)
        if array.size and float(array.max()) > 1.5:
            array = array / 255.0
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"a decoded frame must be HxW or HxWxC, got shape {array.shape}")
    return numpy.clip(array, 0.0, 1.0)


def _box_blur3(array: numpy.ndarray) -> numpy.ndarray:
    """3x3 box blur with edge replication. Numpy only, so the check runs anywhere."""
    padded = numpy.pad(array, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[1:-1, 1:-1]
        + padded[:-2, :-2]
        + padded[:-2, 2:]
        + padded[2:, :-2]
        + padded[2:, 2:]
    ) / 9.0
