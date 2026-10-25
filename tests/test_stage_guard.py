"""16 GB stages keep one heavy model, and the card is near zero between them."""
from __future__ import annotations

import sys
import types

import pytest

from canvas.infra import stage_guard, vram_trace
from canvas.infra.stage_guard import (
    StageBlocked,
    admit,
    begin_stage,
    choose_encoder_storage,
    cosine,
    finish_stage,
    heavy_names_on_cuda,
)
from canvas.infra.vram_trace import drop_model, hold_model, note_vram
from canvas.models.modes import Tier

_GIB = 1024**3


class _Device:
    def __init__(self, kind: str) -> None:
        self.type = kind


class _Param:
    def __init__(self, device: _Device, dtype_name: str, nbytes: int) -> None:
        self.device = device
        self.dtype = types.SimpleNamespace(name=dtype_name)
        self.nbytes = nbytes


class _CudaModule:
    def __init__(self, dtype_name: str = "float8_e4m3fn", nbytes: int = 4096) -> None:
        self.device = _Device("cuda")
        self.param = _Param(self.device, dtype_name, nbytes)
        self.history: list[str] = []

    def parameters(self):
        yield self.param

    def buffers(self):
        return iter(())

    def to(self, device: str) -> _CudaModule:
        self.history.append(device)
        kind = "cpu" if device == "cpu" else device
        self.device = _Device(kind)
        self.param.device = self.device
        return self


class _Memory:
    def __init__(self) -> None:
        self.allocated = 0
        self.peak = 0

    def is_available(self) -> bool:
        return True

    def memory_allocated(self) -> int:
        return self.allocated

    def max_memory_allocated(self) -> int:
        return max(self.peak, self.allocated)

    def empty_cache(self) -> None:
        return None


def _arm(monkeypatch) -> tuple[_Memory, list[str]]:
    memory = _Memory()
    codes: list[str] = []
    torch = types.SimpleNamespace(cuda=memory)
    monkeypatch.setattr(stage_guard, "_torch", lambda: torch)
    monkeypatch.setattr(stage_guard, "note_error", codes.append)
    vram_trace._holders.clear()
    runtime = sys.modules.get("canvas.models.runtime")
    if runtime is not None:
        runtime.reset_pipeline_cache()
    return memory, codes


def test_cook_refuses_an_allocation_past_the_transformer_budget(monkeypatch) -> None:
    memory, codes = _arm(monkeypatch)
    memory.allocated = 21 * _GIB
    with pytest.raises(StageBlocked) as caught:
        begin_stage("cook", Tier.GB16)
    assert caught.value.code == "vram.cook"
    assert any(code.startswith("vram.cook:cook:") for code in codes)
    assert heavy_names_on_cuda() == ()


def test_a_stage_refuses_leftover_memory(monkeypatch) -> None:
    memory, codes = _arm(monkeypatch)
    memory.allocated = _GIB
    with pytest.raises(StageBlocked) as caught:
        begin_stage("text_encoder", Tier.GB16)
    assert caught.value.code == "vram.residual"
    assert any(code.startswith("vram.residual:text_encoder:") for code in codes)


def test_a_stage_refuses_a_peak_past_14gb(monkeypatch) -> None:
    memory, codes = _arm(monkeypatch)
    memory.peak = 15 * _GIB
    with pytest.raises(StageBlocked) as caught:
        begin_stage("cook", Tier.GB16)
    assert caught.value.code == "vram.peak"
    assert any(code.startswith("vram.peak:cook:") for code in codes)


def test_finish_refuses_memory_left_between_stages(monkeypatch) -> None:
    memory, codes = _arm(monkeypatch)
    memory.allocated = 2 * _GIB
    with pytest.raises(StageBlocked) as caught:
        finish_stage("writer", Tier.GB16)
    assert caught.value.code == "vram.residual"
    assert any(code.startswith("vram.residual:writer:") for code in codes)


def test_one_heavy_model_per_stage(monkeypatch) -> None:
    _arm(monkeypatch)
    writer = _CudaModule()
    captioner = _CudaModule()
    begin_stage("writer", Tier.GB16)
    admit("writer", writer, Tier.GB16)
    assert heavy_names_on_cuda() == ("writer",)
    with pytest.raises(StageBlocked) as caught:
        admit("captioner", captioner, Tier.GB16)
    assert caught.value.code == "vram.overlap"
    assert "cpu" in writer.history
    assert writer.device.type == "cpu"
    assert heavy_names_on_cuda() == ()
    admit("captioner", captioner, Tier.GB16)
    assert heavy_names_on_cuda() == ("captioner",)
    assert len(heavy_names_on_cuda()) == 1


def test_the_vae_can_sit_with_the_transformer(monkeypatch) -> None:
    _arm(monkeypatch)
    transformer = _CudaModule()
    vae = _CudaModule()
    begin_stage("cook", Tier.GB16)
    admit("transformer", transformer, Tier.GB16)
    admit("vae", vae, Tier.GB16)
    assert heavy_names_on_cuda() == ("transformer",)


def test_a_24gb_card_keeps_its_weights(monkeypatch) -> None:
    memory, codes = _arm(monkeypatch)
    memory.allocated = 21 * _GIB
    begin_stage("cook", Tier.GB24)
    finish_stage("cook", Tier.GB24)
    assert codes == []


def test_cosine_of_equal_and_orthogonal_vectors() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_encoder_storage_uses_fp8_when_the_cosine_holds(monkeypatch) -> None:
    codes: list[str] = []
    monkeypatch.setattr(stage_guard, "note_error", codes.append)
    free = 15 * _GIB
    assert choose_encoder_storage(Tier.GB16, True, 0.995, free) == "fp8"
    assert codes == []
    assert choose_encoder_storage(Tier.GB16, True, 0.5, free) == "bf16"
    assert codes == ["encoder.fp8_quality"]
    codes.clear()
    assert choose_encoder_storage(Tier.GB16, False, 0.995, free) == "bf16"
    assert codes == ["encoder.fp8_missing"]
    codes.clear()
    assert choose_encoder_storage(Tier.GB16, True, 0.995, 2 * _GIB) == "cpu"
    assert codes == ["encoder.cpu"]
    codes.clear()
    assert choose_encoder_storage(Tier.GB24, True, 0.5, free) == "bf16"
    assert choose_encoder_storage(Tier.GB24, True, None, None) == "cpu"
    assert codes == []


def test_vram_trace_logs_dtype_size_and_peak(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("VOXPRINT_VRAM_TRACE", "1")
    vram_trace._holders.clear()
    module = _CudaModule(dtype_name="float8_e4m3fn", nbytes=4096)
    hold_model("transformer", module)
    try:
        note_vram("pipeline_load", "before")
    finally:
        drop_model("transformer")
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "transformer:4096:float8_e4m3fn" in log
    assert "peak=" in log
