"""Fit decisions read driver free memory. The allocator peak is only logged.

rc.29 laptop: the writer was parked, but the peak from its load was still in
``max_memory_allocated`` and the series stopped with "does not fit".
"""
from __future__ import annotations

import types
from pathlib import Path

from canvas.infra import phase_vram, stage_guard, vram_trace
from canvas.infra.stage_guard import begin_stage, finish_stage, guard_peak, measure_free, note_image_fit
from canvas.models.modes import Tier
from canvas.pipeline import series
from tests.fakes import World
from tests.test_photoshoot import IDEA, _payload, _run

_GIB = 1024**3
_MIB = 1024**2


class _Cuda:
    """``torch.cuda`` double: allocator counters plus driver free and total."""

    def __init__(self, *, free: int, total: int = 16 * _GIB) -> None:
        self.allocated = 0
        self.peak = 0
        self.free = free
        self.total = total
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return True

    def memory_allocated(self) -> int:
        return self.allocated

    def max_memory_allocated(self) -> int:
        return max(self.peak, self.allocated)

    def reset_peak_memory_stats(self) -> None:
        self.calls.append("reset")
        self.peak = self.allocated

    def empty_cache(self) -> None:
        self.calls.append("empty")

    def mem_get_info(self) -> tuple[int, int]:
        self.calls.append("info")
        return self.free, self.total


def _arm(monkeypatch, cuda: _Cuda) -> list[str]:
    torch = types.SimpleNamespace(cuda=cuda)
    codes: list[str] = []
    monkeypatch.setattr(stage_guard, "_torch", lambda: torch)
    monkeypatch.setattr(stage_guard, "note_error", codes.append)
    monkeypatch.setattr(phase_vram, "_cuda", lambda: cuda)
    vram_trace._holders.clear()
    return codes


def _rc29_after_writer(monkeypatch) -> tuple[_Cuda, list[str]]:
    cuda = _Cuda(free=15634268160, total=17170956288)
    cuda.allocated = 34078720
    cuda.peak = 15746 * _MIB
    return cuda, _arm(monkeypatch, cuda)


def test_measure_free_empties_the_cache_and_resets_the_peak_before_reading(monkeypatch) -> None:
    cuda, _codes = _rc29_after_writer(monkeypatch)
    assert measure_free() == (15634268160, 17170956288)
    assert cuda.calls == ["empty", "reset", "info"]
    assert cuda.max_memory_allocated() == cuda.allocated


def test_measure_free_is_none_without_cuda(monkeypatch) -> None:
    cuda = _Cuda(free=0)
    cuda.is_available = lambda: False  # type: ignore[method-assign]
    _arm(monkeypatch, cuda)
    assert measure_free() is None
    assert cuda.calls == []


def test_the_writer_stage_closes_on_a_stale_peak(monkeypatch) -> None:
    _cuda, codes = _rc29_after_writer(monkeypatch)
    finish_stage("writer", Tier.GB16)
    assert codes == []


def test_each_stage_starts_with_a_fresh_peak(monkeypatch) -> None:
    cuda, codes = _rc29_after_writer(monkeypatch)
    begin_stage("text_encoder", Tier.GB16)
    assert "reset" in cuda.calls
    assert cuda.max_memory_allocated() < _GIB
    guard_peak("text_encoder", Tier.GB16)
    assert codes == []


def test_the_picture_fits_on_free_memory_whatever_the_old_peak(monkeypatch) -> None:
    cuda, _codes = _rc29_after_writer(monkeypatch)
    assert note_image_fit() is True
    assert cuda.calls[-1] == "info"


def test_the_picture_does_not_fit_when_free_memory_is_short(monkeypatch) -> None:
    cuda = _Cuda(free=4 * _GIB)
    _arm(monkeypatch, cuda)
    assert note_image_fit() is False


def test_a_live_allocation_past_14gb_still_stops_the_stage(monkeypatch) -> None:
    cuda = _Cuda(free=_GIB)
    codes = _arm(monkeypatch, cuda)
    cuda.allocated = 15 * _GIB
    try:
        guard_peak("cook", Tier.GB16)
    except stage_guard.StageBlocked as exc:
        assert exc.code == "vram.peak"
    else:
        raise AssertionError("a 15 GB live allocation was admitted")
    assert any(code.startswith("vram.peak:cook:") for code in codes)


def test_the_phase_line_keeps_the_peak_a_fit_check_reset(monkeypatch) -> None:
    cuda = _Cuda(free=15 * _GIB)
    _arm(monkeypatch, cuda)
    current = phase_vram.start_phase("prep")
    try:
        cuda.peak = 9 * _GIB
        measure_free()
        assert cuda.max_memory_allocated() == 0
    finally:
        assert current.finish() == 9 * _GIB


class _Writer:
    """``rewrite`` with a ``release`` method, like the real writer session."""

    def __init__(self, order: list[str], count: int) -> None:
        self.order = order
        self.count = count

    def __call__(self, _prompt: str) -> str:
        self.order.append("write")
        return _payload(IDEA, self.count)

    def release(self) -> None:
        self.order.append("release")


def test_the_writer_is_released_before_the_check_models_load(tmp_path: Path, monkeypatch) -> None:
    gate = World().gate()
    order: list[str] = []

    def load(name: str):
        def _load() -> None:
            order.append(f"load:{name}")

        return _load

    for name in ("nsfw", "zeroshot", "faces", "ages", "fine"):
        setattr(getattr(gate._stack, name), "load", load(name))
    real_release = series.release_cuda_cache

    def release_cache() -> bool:
        order.append("empty")
        return real_release()

    monkeypatch.setattr(series, "release_cuda_cache", release_cache)
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    result = _run(tmp_path / "out", age_gate=gate, rewrite=_Writer(order, 2), count=2)
    assert result.stopped == ""
    loads = [index for index, step in enumerate(order) if step.startswith("load:")]
    assert loads
    assert order.index("release") < order.index("empty") < min(loads)
    assert max(index for index, step in enumerate(order) if step == "write") < min(loads)


def test_check_models_without_a_loader_warm_after_the_writer_is_released(tmp_path: Path) -> None:
    gate = World().gate()
    order: list[str] = []
    real = gate.warm

    def warm() -> None:
        order.append("warm")
        real()

    gate.warm = warm  # type: ignore[method-assign]
    result = _run(tmp_path, age_gate=gate, rewrite=_Writer(order, 2), count=2)
    assert result.stopped == ""
    assert order.index("release") < order.index("warm")
