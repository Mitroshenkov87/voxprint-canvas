"""RAM copies stay for the session. A later open does not read them from disk.

The default CI job has no torch. These doubles have a dtype and a ``.to`` so the
park can pin them without that package.
"""
from __future__ import annotations

import sys
import types

from voxprint_common.paths import logs_dir

from canvas.models import hot_tier
from canvas.models.hot_tier import open_module, plan, reset


class _Weight:
    """One tensor. ``.to`` keeps the dtype and copies the values."""

    def __init__(self, dtype: str, values: list[int]) -> None:
        self.dtype = dtype
        self.values = list(values)
        self.nbytes = len(values) * 4
        self.data = self
        self.device = "cpu"

    def detach(self) -> _Weight:
        return self

    def to(self, device: object, non_blocking: bool = False) -> _Weight:
        del non_blocking
        copy = _Weight(self.dtype, self.values)
        copy.device = str(device)
        return copy

    def pin_memory(self) -> _Weight:
        pinned = self.to("cpu")
        pinned.pinned = True
        return pinned


class _Module:
    def __init__(self, dtype: str) -> None:
        self.weight = _Weight(dtype, [1, 2, 3, 4])

    def named_parameters(self):
        yield "weight", self.weight

    def named_buffers(self):
        return iter(())


def _tier(monkeypatch, name: str, total: int) -> None:
    monkeypatch.setenv("VOXPRINT_RAM_MODE", name)
    monkeypatch.setattr(hot_tier, "probed_bytes", lambda: total)
    reset()


def test_a_second_open_does_not_read_a_parked_model(monkeypatch) -> None:
    _tier(monkeypatch, "ram96", 96 * 1024**3)
    calls = {"n": 0}

    def load() -> _Module:
        calls["n"] += 1
        return _Module("bfloat16")

    first = open_module("transformer", load)
    second = open_module("transformer", load)
    assert calls["n"] == 1
    assert second is first
    hot_tier.deactivate("transformer")
    assert first.weight.data.dtype == "bfloat16"
    assert first.weight.data.values == [1, 2, 3, 4]


def test_parking_keeps_the_gate_dtype(monkeypatch) -> None:
    _tier(monkeypatch, "ram96", 96 * 1024**3)
    module = _Module("float32")
    assert hot_tier.remember("gate_nsfw", module) is True
    hot_tier.activate("gate_nsfw", "cpu")
    hot_tier.deactivate("gate_nsfw")
    assert module.weight.data.dtype == "float32"
    assert module.weight.data.values == [1, 2, 3, 4]


def test_ram32_reads_the_disk_again(monkeypatch) -> None:
    _tier(monkeypatch, "ram32", 32 * 1024**3)
    calls = {"n": 0}

    def load() -> _Module:
        calls["n"] += 1
        return _Module("float32")

    open_module("vae", load)
    open_module("vae", load)
    assert calls["n"] == 2
    assert hot_tier.cached("vae") is None


def test_ram64_keeps_earlier_models_and_logs_the_rest_as_disk(monkeypatch) -> None:
    _tier(monkeypatch, "ram64", 64 * 1024**3)
    gib = 1024**3
    decision = plan({
        "writer": 8 * gib,
        "transformer": 30 * gib,
        "prompt_check": 4 * gib,
        "vae": gib,
        "text_encoder": 20 * gib,
    })
    assert decision["transformer"] == "ram"
    assert decision["text_encoder"] == "ram"
    assert decision["vae"] == "ram"
    assert decision["writer"] == "disk"
    assert decision["prompt_check"] == "disk"


def test_a_cuda_weight_parks_into_pinned_memory(monkeypatch, tmp_path) -> None:
    """The first park copies a CUDA weight straight into pinned host memory."""
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    _tier(monkeypatch, "ram96", 96 * 1024**3)

    class _Pinned:
        def __init__(self) -> None:
            self.device = types.SimpleNamespace(type="cpu")
            self.dtype = "bfloat16"
            self.nbytes = 16
            self.data = self

        def is_pinned(self) -> bool:
            return True

        def detach(self) -> _Pinned:
            return self

        def to(self, device: object, non_blocking: bool = False) -> _Pinned:
            del device, non_blocking
            return self

        def pin_memory(self) -> _Pinned:
            return self

        def copy_(self, source: object) -> _Pinned:
            del source
            return self

    class _Torch:
        @staticmethod
        def empty_like(tensor: object, device: str = "cpu", pin_memory: bool = False) -> _Pinned:
            del tensor
            assert device == "cpu"
            assert pin_memory is True
            return _Pinned()

    class _CudaWeight:
        def __init__(self) -> None:
            self.device = types.SimpleNamespace(type="cuda")
            self.dtype = "bfloat16"
            self.nbytes = 16
            self.data = self

        def detach(self) -> _CudaWeight:
            return self

        def to(self, *args: object, **kwargs: object) -> _CudaWeight:
            del args, kwargs
            raise AssertionError("pageable copy")

    class _GpuModule:
        def __init__(self) -> None:
            self.weight = _CudaWeight()

        def named_parameters(self):
            yield "weight", self.weight

        def named_buffers(self):
            return iter(())

    monkeypatch.setitem(sys.modules, "torch", _Torch)
    try:
        assert hot_tier.remember("transformer", _GpuModule()) is True
    finally:
        hot_tier.reset()
    text = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "stage park transformer" in text
    assert "pinned=1" in text


def test_park_and_unpark_log_milliseconds_bytes_ram_and_vram(monkeypatch) -> None:
    _tier(monkeypatch, "ram96", 96 * 1024**3)
    log = logs_dir() / "voxprint.log"
    before = log.stat().st_size if log.is_file() else 0
    hot_tier.remember("writer", _Module("float32"))
    hot_tier.activate("writer", "cpu")
    hot_tier.deactivate("writer")
    text = log.read_bytes()[before:].decode("utf-8")
    assert "stage hot plan" in text
    assert "model=writer where=ram" in text
    assert "stage park writer" in text
    assert "vram_mib=" in text and "ram_mib=" in text
    assert "ms=" in text and "bytes=" in text
    assert "where=disk" not in text


def test_a_second_job_without_free_ram_keeps_the_model_on_its_card(monkeypatch) -> None:
    """Two jobs on two cards: the one that finds RAM taken does not park a copy."""
    _tier(monkeypatch, "ram96", 128 * 1024**3)
    monkeypatch.setattr(hot_tier, "free_bytes", lambda: 4 * 1024**3)
    module = _Module("float8_e4m3fn")
    original = module.weight
    assert hot_tier.remember("transformer", module, size_bytes=12 * 1024**3) is False
    assert hot_tier.cached("transformer") is None
    assert module.weight.data is original


def test_a_failed_park_leaves_the_model_usable(monkeypatch) -> None:
    """A host-memory error while parking falls back to no RAM copy instead of aborting."""
    _tier(monkeypatch, "ram96", 128 * 1024**3)

    def fail(*_args, **_kwargs):
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(hot_tier, "_park_state", fail)
    module = _Module("bfloat16")
    assert hot_tier.remember("transformer", module) is False
    assert hot_tier.retained("transformer") is False
    assert module.weight.data.values == [1, 2, 3, 4]


def test_a_failed_cpu_view_keeps_the_ram_tensors(monkeypatch) -> None:
    _tier(monkeypatch, "ram96", 128 * 1024**3)
    module = _Module("bfloat16")
    assert hot_tier.remember("vae", module) is True
    resident = hot_tier._cache["vae"]

    class _Broken:
        def to(self, *_args, **_kwargs):
            raise RuntimeError("AcceleratorError")

        def release(self) -> None:
            return None

    resident.parked = _Broken()
    hot_tier.deactivate("vae")
    assert module.weight.data.values == [1, 2, 3, 4]
