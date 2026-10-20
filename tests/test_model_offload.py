"""16 GB placement: whole-component offload when the transformer fits, block streaming otherwise."""
from __future__ import annotations

from canvas.models import runtime
from canvas.models.modes import Tier
from canvas.pipeline.offload import MODEL_OFFLOAD_HEADROOM_BYTES, model_offload_fits


class _Tensor:
    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count

    def element_size(self) -> int:
        return 2


class _Module:
    def __init__(self, count: int) -> None:
        self.count = count

    def modules(self) -> list[object]:
        return [self]

    def parameters(self) -> list[_Tensor]:
        return [_Tensor(self.count)]

    def buffers(self) -> list[_Tensor]:
        return []


class _Pipe:
    def __init__(self, transformer_params: int) -> None:
        self.components = {"transformer": _Module(transformer_params), "vae": _Module(10)}
        self.installed: list[str] = []

    def enable_model_cpu_offload(self) -> None:
        self.installed.append("model")

    def enable_group_offload(self, **_kwargs: object) -> None:
        self.installed.append("group")


GB = 1024**3


def test_the_transformer_fits_a_16_gb_card() -> None:
    pipe = _Pipe(6 * GB)  # 12 GB at BF16
    assert model_offload_fits(pipe, 15 * GB)
    assert not model_offload_fits(pipe, 12 * GB + MODEL_OFFLOAD_HEADROOM_BYTES - 1)
    assert not model_offload_fits(pipe, None)


def test_place_uses_model_offload_when_it_fits(monkeypatch) -> None:
    pipe = _Pipe(6 * GB)
    monkeypatch.setattr(runtime, "_free_vram_bytes", lambda: 15 * GB)
    runtime._place(pipe, Tier.GB16)
    assert pipe.installed == ["model"]


def test_place_streams_blocks_when_vram_is_short(monkeypatch) -> None:
    pipe = _Pipe(6 * GB)
    monkeypatch.setattr(runtime, "_free_vram_bytes", lambda: 10 * GB)
    monkeypatch.setattr(runtime, "install_low_vram_offload", lambda p: p.installed.append("group"))
    runtime._place(pipe, Tier.GB16)
    assert pipe.installed == ["group"]
