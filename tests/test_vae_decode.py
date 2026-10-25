"""VAE decode keeps the latents on the VAE's device and dtype."""
from __future__ import annotations

import types

import numpy
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
    original = vae.tiled_decode
    seen: dict[str, object] = {}

    def spy(latents: object, *args: object, **kwargs: object) -> object:
        param = next(vae.parameters())
        assert latents.dtype == param.dtype
        assert latents.device == param.device
        seen["dtype"] = latents.dtype
        seen["device"] = latents.device
        return original(latents, *args, **kwargs)

    vae.tiled_decode = spy
    pipe = types.SimpleNamespace(vae=vae)
    pairs = ((torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32))
    for vae_dtype, latent_dtype in pairs:
        vae.to(dtype=vae_dtype)
        latents = torch.zeros(1, 4, 8, 8, dtype=latent_dtype)
        decoded = runtime._decode_latents(pipe, latents, torch)
        assert decoded.dtype == numpy.float32
        assert decoded.shape[0] == 1
        assert decoded.shape[-1] == 3
        assert float(decoded.min()) >= 0.0
        assert float(decoded.max()) <= 1.0
        assert seen["dtype"] == vae_dtype
        assert seen["device"].type == "cpu"


def test_cuda_decode_tiles_and_returns_cpu_float32(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    flags: list[str] = []
    calls: list[str] = []
    pixels = _Pixels(-1.0)

    class _Vae:
        config = types.SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159)
        tile_sample_min_size = 1024
        tile_latent_min_size = 128

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
            calls.append("decode")
            return types.SimpleNamespace(sample=pixels)

        def tiled_decode(self, latents: object, return_dict: bool = True) -> types.SimpleNamespace:
            del latents, return_dict
            calls.append("tiled")
            return types.SimpleNamespace(sample=pixels)

    vae = _Vae()
    torch = _cuda_torch()
    hot_tier.reset()
    try:
        decoded = runtime._decode_latents(types.SimpleNamespace(vae=vae), _Latent("cuda", "bf16"), torch)
    finally:
        hot_tier.reset()
    assert flags == ["tiling", "slicing"]
    assert calls == ["tiled"]
    assert vae.tile_sample_min_size == 256
    assert vae.tile_latent_min_size == 32
    assert decoded.dtype == numpy.float32
    assert float(decoded.flat[0]) == 0.0
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert f"vae tiling=1 slicing=1 id={id(vae)}" in log


def test_a_latent_tile_that_covers_the_frame_is_split() -> None:
    class _Vae:
        tile_sample_min_size = 256
        tile_latent_min_size = 128

    vae = _Vae()
    runtime._fit_vae_tiles(vae, types.SimpleNamespace(shape=(1, 16, 128, 128)))
    assert vae.tile_latent_min_size <= 32
    assert vae.tile_latent_min_size < 128
    assert vae.tile_sample_min_size < 256


def test_a_tiny_latent_keeps_its_tile() -> None:
    class _Vae:
        tile_sample_min_size = 8
        tile_latent_min_size = 8

    vae = _Vae()
    runtime._fit_vae_tiles(vae, types.SimpleNamespace(shape=(1, 4, 8, 8)))
    assert vae.tile_sample_min_size == 8
    assert vae.tile_latent_min_size == 8


def test_custom_postprocess_maps_the_vae_range_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    sample = torch.tensor([[[[-1.0, 0.0], [1.0, 3.0]]]], dtype=torch.float32)
    array = runtime._pixels_from_decoder(sample, torch, [None])
    assert isinstance(array, numpy.ndarray)
    assert array.dtype == numpy.float32
    assert array.shape == (1, 2, 2, 1)
    numpy.testing.assert_allclose(array.reshape(-1), [0.0, 0.5, 1.0, 1.0], atol=1e-6)


def test_a_postprocess_error_logs_the_tensor_and_the_cause(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))

    class _Vae:
        config = types.SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

        def enable_tiling(self) -> None:
            return None

        def enable_slicing(self) -> None:
            return None

        def parameters(self):
            yield types.SimpleNamespace(device="cuda", dtype="bf16")

        def to(self, *args: object, **kwargs: object) -> _Vae:
            del args, kwargs
            return self

        def tiled_decode(self, latents: object, return_dict: bool = True) -> types.SimpleNamespace:
            del latents, return_dict
            return types.SimpleNamespace(sample=_Broken())

    hot_tier.reset()
    try:
        with pytest.raises(RuntimeError):
            runtime._decode_latents(types.SimpleNamespace(vae=_Vae()), _Latent("cuda", "bf16"), _cuda_torch())
    finally:
        hot_tier.reset()
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "stage vae_decode error" in log
    assert "type=RuntimeError" in log
    assert "repr=RuntimeError('')" in log
    assert "cause=ValueError('cast')" in log
    assert "context=ValueError('cast')" in log
    assert "dtype=bf16" in log
    assert "device=cuda:0" in log
    assert "shape=(1, 3, 4, 4)" in log
    assert "contiguous=False" in log
    assert "summary=allocated 11 GiB" in log


def test_tiled_decode_uses_the_hot_tier_vae(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("VOXPRINT_RAM_MODE", "ram96")
    monkeypatch.setattr(hot_tier, "probed_bytes", lambda: 96 * 1024**3)
    calls: list[str] = []

    class _Vae:
        def __init__(self, label: str) -> None:
            self.label = label
            self.config = types.SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

        def parameters(self):
            yield types.SimpleNamespace(device="cpu", dtype="bf16", is_pinned=lambda: False)

        def named_parameters(self):
            return iter(())

        def named_buffers(self):
            return iter(())

        def enable_tiling(self) -> None:
            return None

        def enable_slicing(self) -> None:
            return None

        def to(self, *args: object, **kwargs: object) -> _Vae:
            del args, kwargs
            return self

        def tiled_decode(self, latents: object, return_dict: bool = True) -> types.SimpleNamespace:
            del latents, return_dict
            calls.append(self.label)
            return types.SimpleNamespace(sample="frame")

    cached = _Vae("cached")
    other = _Vae("pipe")
    hot_tier.reset()
    try:
        assert hot_tier.remember("vae", cached) is True
        runtime._decode_latents(types.SimpleNamespace(vae=other), _Latent("cpu", "bf16"), _cpu_torch())
    finally:
        hot_tier.reset()
    assert calls == ["cached"]
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert f"vae tiling=1 slicing=1 id={id(cached)}" in log


def test_a_layerwise_hook_is_stripped_and_forward_is_restored() -> None:
    def plain(module: object, *args: object, **kwargs: object) -> object:
        del module, args, kwargs
        return "plain"

    class _Ref:
        forward = staticmethod(plain)
        original_forward = None
        non_blocking = True

    class _Registry:
        def __init__(self) -> None:
            self.hooks = {"layerwise_casting": types.SimpleNamespace(fn_ref=_Ref(), non_blocking=True, storage_dtype="fp8")}
            self._hook_order = ["layerwise_casting"]
            self._fn_refs = [self.hooks["layerwise_casting"].fn_ref]

    class _Vae:
        def __init__(self) -> None:
            self._diffusers_hook = _Registry()
            self.forward = lambda *_args, **_kwargs: "hooked"

        def modules(self):
            yield self

    vae = _Vae()
    removed, saw_non_blocking = runtime._strip_casting_hooks(vae)
    assert removed == 1
    assert saw_non_blocking is True
    assert "layerwise_casting" not in vae._diffusers_hook.hooks
    assert vae.forward is plain
    assert runtime._casting_hook_count(vae) == 0


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

        def tiled_decode(self, latents: object, return_dict: bool = True) -> object:
            del latents, return_dict
            raise RuntimeError("")

    vae = _Vae()
    hot_tier.reset()
    try:
        with pytest.raises(RuntimeError):
            runtime._decode_latents(types.SimpleNamespace(vae=vae), _Latent("cuda", "bf16"), _cuda_torch())
    finally:
        hot_tier.reset()
    assert flags == ["tiling", "slicing"]
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert f"vae tiling=1 slicing=1 id={id(vae)}" in log
    assert "stage vae_decode error" in log
    assert "type=RuntimeError" in log
    assert "repr=RuntimeError('')" in log
    assert "cause=None" in log
    assert "summary=allocated 11 GiB" in log


def _cuda_torch() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: None,
            empty_cache=lambda: None,
            memory_summary=lambda: "allocated 11 GiB",
        ),
    )


def _cpu_torch() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )


class _Broken:
    """A decoder sample whose CPU copy fails the way ``pt_to_numpy`` did."""

    dtype = "bf16"
    device = "cuda:0"
    shape = (1, 3, 4, 4)

    def __truediv__(self, _other: object) -> _Broken:
        return self

    def __add__(self, _other: object) -> _Broken:
        return self

    def clamp(self, _lo: object, _hi: object) -> _Broken:
        return self

    def is_contiguous(self) -> bool:
        return False

    def to(self, _device: object, _dtype: object) -> _Broken:
        try:
            raise ValueError("cast")
        except ValueError as exc:
            raise RuntimeError("") from exc


class _Pixels:
    def __init__(self, value: float) -> None:
        self.value = value
        self.device = types.SimpleNamespace(type="cuda")
        self.dtype = "bf16"
        self.shape = (1, 3, 2, 2)

    def __truediv__(self, other: object) -> _Pixels:
        return _Pixels(self.value / float(other))

    def __add__(self, other: object) -> _Pixels:
        return _Pixels(self.value + float(other))

    def clamp(self, lo: object, hi: object) -> _Pixels:
        return _Pixels(min(float(hi), max(float(lo), self.value)))

    def to(self, device: object, dtype: object) -> _Pixels:
        moved = _Pixels(self.value)
        moved.device = types.SimpleNamespace(type=str(device))
        moved.dtype = dtype
        return moved

    def permute(self, *_axes: int) -> _Pixels:
        return self

    def contiguous(self) -> _Pixels:
        return self

    def is_contiguous(self) -> bool:
        return True

    def numpy(self) -> numpy.ndarray:
        return numpy.full((1, 2, 2, 3), self.value, dtype=numpy.float32)
