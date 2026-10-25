"""VAE decode keeps the latents on the VAE's device and dtype."""
from __future__ import annotations

import types

import pytest

from canvas.models import hot_tier, runtime


class _Latent:
    def __init__(self, device: object, dtype: object) -> None:
        self.device = device
        self.dtype = dtype

    def __truediv__(self, _other: object) -> _Latent:
        return self

    def __add__(self, _other: object) -> _Latent:
        return self

    def to(self, *, device: object = None, dtype: object = None) -> _Latent:
        self.device = device
        self.dtype = dtype
        return self


class _Param:
    def __init__(self, device: object, dtype: object) -> None:
        self.device = device
        self.dtype = dtype


class _Vae:
    def __init__(self, device: object, dtype: object) -> None:
        self.param = _Param(device, dtype)
        self.config = types.SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159)
        self.seen: tuple[object, object] | None = None

    def parameters(self):
        yield self.param

    def decode(self, latents: _Latent) -> types.SimpleNamespace:
        self.seen = (latents.device, latents.dtype)
        return types.SimpleNamespace(sample="decoded")


def _decode(device: object, vae_dtype: object, latent_dtype: object) -> tuple[object, object]:
    vae = _Vae(device, vae_dtype)
    pipe = types.SimpleNamespace(vae=vae)
    torch = types.SimpleNamespace(bfloat16="bf16", cuda=types.SimpleNamespace(is_available=lambda: False))
    runtime._decode_latents(pipe, _Latent(device, latent_dtype), torch)
    assert vae.seen is not None
    return vae.seen


def test_latents_match_the_vae_for_both_dtype_pairs() -> None:
    assert _decode("cpu", "fp32", "bf16") == ("cpu", "fp32")
    assert _decode("cpu", "bf16", "fp32") == ("cpu", "bf16")


def _tiny_vae(diffusers: object):
    vae_cls = diffusers.AutoencoderKL
    return vae_cls(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(32,),
        layers_per_block=1,
        act_fn="silu",
        latent_channels=4,
        norm_num_groups=32,
        sample_size=8,
        scaling_factor=0.3611,
        shift_factor=0.1159,
        force_upcast=False,
    )


def test_a_tiny_autoencoder_decodes_both_dtype_pairs_on_cpu(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    monkeypatch.setattr(runtime, "_cuda_is_up", lambda _torch: False)
    vae = _tiny_vae(diffusers).eval()
    original = vae.decode
    seen: dict[str, object] = {}

    def spy(latents: object, *args: object, **kwargs: object) -> object:
        param = next(vae.parameters())
        assert latents.dtype == param.dtype
        assert latents.device == param.device
        seen["dtype"] = latents.dtype
        seen["device"] = latents.device
        return original(latents, *args, **kwargs)

    vae.decode = spy
    pipe = types.SimpleNamespace(vae=vae)
    pairs = ((torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32))
    for vae_dtype, latent_dtype in pairs:
        vae.to(dtype=vae_dtype)
        latents = torch.zeros(1, 4, 8, 8, dtype=latent_dtype)
        decoded = runtime._decode_latents(pipe, latents, torch)
        sample = getattr(decoded, "sample", decoded)
        assert sample.dtype == torch.float32
        assert sample.device.type == "cpu"
        assert seen["dtype"] == vae_dtype
        assert seen["device"].type == "cpu"


def test_cuda_decode_tiles_and_returns_cpu_float32() -> None:
    flags: list[str] = []
    pixels = _Pixels()

    class _Vae:
        config = types.SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159)

        def enable_tiling(self) -> None:
            flags.append("tiling")

        def enable_slicing(self) -> None:
            flags.append("slicing")

        def parameters(self):
            yield types.SimpleNamespace(device="cuda", dtype="bf16")

        def to(self, *args: object, **kwargs: object) -> _Vae:
            del args, kwargs
            return self

        def decode(self, latents: object) -> types.SimpleNamespace:
            del latents
            return types.SimpleNamespace(sample=pixels)

    torch = types.SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: None,
            empty_cache=lambda: None,
        ),
    )
    hot_tier.reset()
    try:
        decoded = runtime._decode_latents(types.SimpleNamespace(vae=_Vae()), _Latent("cuda", "bf16"), torch)
    finally:
        hot_tier.reset()
    assert flags == ["tiling", "slicing"]
    assert decoded is pixels
    assert pixels.dtype == "fp32"
    assert pixels.device.type == "cpu"


def test_a_cuda_decode_error_logs_its_type_and_the_memory_summary(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    flags: list[str] = []

    class _Vae:
        config = types.SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159)

        def enable_tiling(self) -> None:
            flags.append("tiling")

        def enable_slicing(self) -> None:
            flags.append("slicing")

        def parameters(self):
            yield types.SimpleNamespace(device="cuda", dtype="bf16")

        def to(self, *args: object, **kwargs: object) -> _Vae:
            del args, kwargs
            return self

        def decode(self, latents: object) -> object:
            del latents
            raise RuntimeError("")

    torch = types.SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: None,
            empty_cache=lambda: None,
            memory_summary=lambda: "allocated 11 GiB",
        ),
    )
    hot_tier.reset()
    try:
        with pytest.raises(RuntimeError):
            runtime._decode_latents(types.SimpleNamespace(vae=_Vae()), _Latent("cuda", "bf16"), torch)
    finally:
        hot_tier.reset()
    assert flags == ["tiling", "slicing"]
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "stage vae_decode error" in log
    assert "type=RuntimeError" in log
    assert "summary=allocated 11 GiB" in log


class _Pixels:
    def __init__(self) -> None:
        self.device = types.SimpleNamespace(type="cuda")
        self.dtype = "bf16"

    def to(self, device: object = None, dtype: object = None) -> _Pixels:
        if device == "cpu":
            self.device = types.SimpleNamespace(type="cpu")
        if dtype is not None:
            self.dtype = dtype
        return self
