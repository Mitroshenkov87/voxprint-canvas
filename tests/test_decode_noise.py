"""The decode-noise assert trips on pure grain and stays quiet on real pictures."""
from __future__ import annotations

import types
from pathlib import Path

import numpy
import pytest
from PIL import Image

from canvas.app.i18n import text
from canvas.models import runtime
from canvas.models.decode_noise import (
    DECODE_NOISE_CODE,
    DecodeNoiseError,
    assert_not_noise,
    noise_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


def _grain(width: int = 64, height: int = 64, seed: int = 0) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    return rng.normal(0.5, 0.28, (height, width, 3)).clip(0.0, 1.0).astype(numpy.float32)


def _blob(width: int = 64, height: int = 64) -> numpy.ndarray:
    y, x = numpy.mgrid[0:height, 0:width]
    blob = numpy.exp(-(((x - width / 2) ** 2 + (y - height / 2) ** 2) / (2 * 10.0**2))).astype(numpy.float32)
    return numpy.stack([blob, blob * 0.5, 1.0 - blob], axis=-1)


def test_white_grain_trips_the_assert_with_the_stable_code() -> None:
    with pytest.raises(DecodeNoiseError) as caught:
        assert_not_noise(_grain())
    assert caught.value.code == DECODE_NOISE_CODE


def test_a_smooth_ramp_does_not_trip() -> None:
    ramp = numpy.tile(numpy.linspace(0.0, 1.0, 64, dtype=numpy.float32)[:, None], (1, 64))
    metrics = assert_not_noise(numpy.stack([ramp] * 3, axis=-1))
    assert metrics["hf_ratio"] < 0.5


def test_a_low_frequency_picture_does_not_trip() -> None:
    metrics = assert_not_noise(_blob())
    assert metrics["std"] > 0.02
    assert metrics["hf_ratio"] < 0.5


def test_a_ui_screenshot_does_not_trip() -> None:
    shot = Image.open(ROOT / "docs" / "ui" / "screens" / "addons-en.png")
    metrics = assert_not_noise(shot)
    assert metrics["std"] > 0.02


def test_uint8_pil_and_batched_arrays_are_accepted() -> None:
    grain = (_grain() * 255).astype(numpy.uint8)
    with pytest.raises(DecodeNoiseError):
        assert_not_noise(grain)
    with pytest.raises(DecodeNoiseError):
        assert_not_noise(Image.fromarray(grain))
    with pytest.raises(DecodeNoiseError):
        assert_not_noise(grain[None, ...])


def test_noise_metrics_reports_std_and_high_frequency_share() -> None:
    metrics = noise_metrics(_grain())
    assert metrics["std"] > 0.2
    assert metrics["hf_ratio"] > 0.5
    clean = noise_metrics(_blob())
    assert clean["hf_ratio"] < 0.1


def test_the_production_decode_path_fails_closed_on_noise(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr(runtime, "_decode_latents", lambda _pipe, _latents, _torch: _grain()[None, ...])
    pipe = types.SimpleNamespace(vae=object())
    with pytest.raises(DecodeNoiseError) as caught:
        runtime._image_from_result(pipe, object(), _cpu_torch())
    assert caught.value.code == DECODE_NOISE_CODE
    assert str(caught.value) == text("inference.decode_noise", "en")
    assert not list(tmp_path.rglob("*.png"))


def test_the_production_decode_path_returns_a_clean_frame(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    clean = _blob()[None, ...]
    monkeypatch.setattr(runtime, "_decode_latents", lambda _pipe, _latents, _torch: clean)
    pipe = types.SimpleNamespace(vae=object())
    raw = runtime._image_from_result(pipe, object(), _cpu_torch())
    assert (raw.width, raw.height) == (64, 64)


def test_vae_for_decode_keeps_the_loaded_dtype(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "_cuda_is_up", lambda _torch: True)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class _Vae:
        def to(self, *args: object, **kwargs: object) -> _Vae:
            calls.append((args, kwargs))
            return self

    vae = _Vae()
    pipe = types.SimpleNamespace(vae=None)
    moved = runtime._vae_for_decode(pipe, vae, _cuda_torch())
    assert moved is vae
    assert pipe.vae is vae
    assert calls == [(("cuda",), {})]


def test_prepare_vae_moves_the_vae_without_a_dtype_cast(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "_cuda_is_up", lambda _torch: True)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class _Vae:
        def to(self, *args: object, **kwargs: object) -> _Vae:
            calls.append((args, kwargs))
            return self

        def parameters(self):
            return iter(())

    vae = _Vae()
    pipe = types.SimpleNamespace(vae=vae)
    assert runtime._prepare_vae(pipe, _cuda_torch()) is vae
    assert calls == [(("cuda",), {})]


def test_a_casting_hook_is_stripped_after_its_weights_are_upcast() -> None:
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(4, 4)
    module.to(dtype=torch.float8_e4m3fn)
    hook = types.SimpleNamespace(
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
        non_blocking=False,
        fn_ref=None,
    )
    registry = types.SimpleNamespace(
        hooks={"layerwise_casting": hook},
        _hook_order=["layerwise_casting"],
        _fn_refs=[hook],
    )
    module._diffusers_hook = registry
    removed, saw_non_blocking = runtime._strip_casting_hooks(module)
    assert removed == 1
    assert saw_non_blocking is False
    assert "layerwise_casting" not in registry.hooks
    assert next(module.parameters()).dtype == torch.bfloat16


def test_an_hf_casting_hook_is_cleared_after_its_weights_are_upcast() -> None:
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(4, 4)
    module.to(dtype=torch.float8_e4m3fn)
    module._hf_hook = types.SimpleNamespace(
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
        non_blocking=True,
    )
    removed, saw_non_blocking = runtime._strip_casting_hooks(module)
    assert removed == 1
    assert saw_non_blocking is True
    assert getattr(module, "_hf_hook", None) is None
    assert next(module.parameters()).dtype == torch.bfloat16


def _cpu_torch() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )


def _cuda_torch() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: None,
            empty_cache=lambda: None,
        ),
    )
