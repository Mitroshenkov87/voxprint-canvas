"""VAE decode keeps the latents on the VAE's device and dtype."""
from __future__ import annotations

import types

import pytest

from canvas.models import runtime


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
    monkeypatch.setattr(runtime, "_needs_vae_tiling", lambda: False)
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
        assert sample.dtype == vae_dtype
        assert sample.device.type == "cpu"
        assert seen["dtype"] == vae_dtype
        assert seen["device"].type == "cpu"
