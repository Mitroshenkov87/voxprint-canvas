"""16 GB placement: whole-component offload when the transformer fits, block streaming otherwise."""
from __future__ import annotations

from canvas.models import runtime
from canvas.models.modes import Tier
from canvas.pipeline.offload import MODEL_OFFLOAD_HEADROOM_BYTES, model_offload_fits


GB = 1024**3


class _Torch:
    float8_e4m3fn = "fp8"
    bfloat16 = "bf16"


def _cast(module: _Module, storage_dtype: object, compute_dtype: object, **kwargs: object) -> None:
    assert storage_dtype == "fp8" and compute_dtype == "bf16"
    module.size = 1


class _Tensor:
    def __init__(self, count: int, size: int = 2) -> None:
        self.count = count
        self.size = size

    def numel(self) -> int:
        return self.count

    def element_size(self) -> int:
        return self.size


class _Module:
    def __init__(self, count: int) -> None:
        self.count = count
        self.size = 2

    def modules(self) -> list[object]:
        return [self]

    def parameters(self) -> list[_Tensor]:
        return [_Tensor(self.count, self.size)]

    def buffers(self) -> list[_Tensor]:
        return []


class _Pipe:
    def __init__(self, transformer_params: int) -> None:
        self.transformer = _Module(transformer_params)
        self.text_encoder = _Module(4 * GB)
        self.components = {"transformer": self.transformer, "text_encoder": self.text_encoder, "vae": _Module(10)}
        self.installed: list[str] = []

    def to(self, device: str) -> None:
        self.installed.append(device)

    def enable_model_cpu_offload(self) -> None:
        self.installed.append("model")

    def enable_group_offload(self, **_kwargs: object) -> None:
        self.installed.append("group")




def test_the_transformer_fits_a_16_gb_card() -> None:
    pipe = _Pipe(6 * GB)  # 12 GB at BF16
    assert model_offload_fits(pipe, 15 * GB)
    assert not model_offload_fits(pipe, 12 * GB + MODEL_OFFLOAD_HEADROOM_BYTES - 1)
    assert not model_offload_fits(pipe, None)


def _arm(monkeypatch, free_gb: float, cast: bool = True) -> None:
    monkeypatch.setattr(runtime, "_import_torch", lambda: _Torch())
    monkeypatch.setattr(runtime, "_layerwise_casting", lambda: _cast if cast else None)
    monkeypatch.setattr(runtime, "_free_vram_bytes", lambda: int(free_gb * GB))


def test_16_gb_stores_fp8_and_stays_resident(monkeypatch) -> None:
    pipe = _Pipe(6 * GB)  # 6 GB transformer + 4 GB encoder at FP8
    _arm(monkeypatch, 15.2)
    runtime._place(pipe, Tier.GB16)
    assert pipe.transformer.size == 1 and pipe.text_encoder.size == 1
    assert pipe.installed == ["cuda"]


def test_place_uses_model_offload_when_it_fits(monkeypatch) -> None:
    pipe = _Pipe(6 * GB)
    _arm(monkeypatch, 15, cast=False)
    runtime._place(pipe, Tier.GB16)
    assert pipe.installed == ["model"]


def test_place_streams_blocks_when_vram_is_short(monkeypatch) -> None:
    pipe = _Pipe(6 * GB)
    _arm(monkeypatch, 10, cast=False)
    monkeypatch.setattr(runtime, "install_low_vram_offload", lambda p: p.installed.append("group"))
    runtime._place(pipe, Tier.GB16)
    assert pipe.installed == ["group"]
