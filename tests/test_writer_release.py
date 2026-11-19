"""The writer loads in bf16 next to a side-thread check load, and leaves no VRAM behind.

transformers sets torch's process-wide default dtype while it builds a model.
These doubles model that with one shared default, so a check build that is
not serialized against the writer build makes the writer's layers fp32.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from canvas.models import hot_tier as hot
from canvas.pipeline import enhance
from canvas.pipeline.enhance import (
    WRITER_RESIDUAL_LIMIT_BYTES,
    QwenRewriter,
    _causal_model,
    _release_cuda,
    settle_bf16,
)
from canvas.pipeline.writer_decode import GraphDecode
from canvas.safety.adapters import SiglipZeroShot


class _Device:
    def __init__(self, kind: str) -> None:
        self.type = kind


class _Data:
    def __init__(self, dtype: str, kind: str) -> None:
        self.dtype = dtype
        self.device = _Device(kind)

    def to(self, target: str) -> _Data:
        if target in {"cpu", "cuda"}:
            return _Data(self.dtype, target)
        return _Data(target, self.device.type)


class _Param:
    def __init__(self, dtype: str, kind: str = "cuda") -> None:
        self.data = _Data(dtype, kind)

    @property
    def dtype(self) -> str:
        return self.data.dtype

    @property
    def device(self) -> _Device:
        return self.data.device


class _Module:
    def __init__(self, params: list[_Param], buffers: list[_Param] | None = None) -> None:
        self._params = params
        self._buffers = buffers or []

    def parameters(self):
        return iter(self._params)

    def buffers(self):
        return iter(self._buffers)

    def eval(self) -> _Module:
        return self


class _SharedDefault:
    """torch's default dtype: one value for the whole process."""

    def __init__(self) -> None:
        self.value = "float32"
        self.lock = threading.Lock()


def _building_loader(default: _SharedDefault, layers: int, entered: threading.Event | None = None,
                     resume: threading.Event | None = None, *, wake: threading.Event | None = None,
                     hold: threading.Event | None = None) -> Any:
    """A ``from_pretrained`` that sets the shared default and builds layers under it.

    ``entered``/``resume`` pause the build after its first layer. ``wake`` is
    set once this build's default is in place, and ``hold`` keeps that default
    until the other build is done.
    """

    def load(folder: Path, **kwargs: Any) -> _Module:
        del folder
        wanted = kwargs.get("dtype") or kwargs.get("torch_dtype") or "float32"
        previous = default.value
        default.value = str(wanted)
        if wake is not None:
            wake.set()
        if hold is not None:
            hold.wait(timeout=0.5)
        try:
            params = []
            for index in range(layers):
                if index == 1 and entered is not None:
                    entered.set()
                    if resume is not None:
                        resume.wait(timeout=0.2)
                params.append(_Param(default.value))
            return _Module(params)
        finally:
            default.value = previous

    return load


class _Torch:
    bfloat16 = "bfloat16"
    float32 = "float32"


def test_a_check_built_on_the_side_thread_cannot_flip_the_writer_to_fp32(tmp_path: Path, monkeypatch) -> None:
    """The check build waits for the writer build, so every writer layer is bf16."""
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setattr("canvas.models.runtime.system_ram_bytes", lambda: 16 * 1024**3)
    default = _SharedDefault()
    entered = threading.Event()
    resume = threading.Event()
    writer_done = threading.Event()
    writer_loader = _building_loader(default, 6, entered, resume)
    check_loader = _building_loader(default, 2, wake=resume, hold=writer_done)
    transformers = type("T", (), {
        "AutoModelForCausalLM": type("Auto", (), {"from_pretrained": staticmethod(writer_loader)}),
    })()
    # The lock alone has to keep the writer bf16. The cast backstop is checked separately.
    monkeypatch.setattr(enhance, "settle_bf16", lambda *args: 0)
    monkeypatch.setattr("canvas.safety.adapters.remember", lambda *args, **kwargs: False)
    monkeypatch.setattr("canvas.safety.adapters.note_ssd", lambda *args, **kwargs: None)
    gate = SiglipZeroShot(tmp_path)
    monkeypatch.setattr(gate, "_place", lambda model: model)

    def side() -> None:
        entered.wait(timeout=2)
        gate._take("gate_siglip", lambda: check_loader(tmp_path, dtype="float32"))

    worker = threading.Thread(target=side)
    worker.start()
    try:
        model = _causal_model(transformers, _Torch(), tmp_path, "cuda")
    finally:
        writer_done.set()
        resume.set()
        worker.join(timeout=5)
    assert {param.dtype for param in model.parameters()} == {"bfloat16"}
    assert {param.dtype for param in gate._model.parameters()} == {"float32"}
    hot.reset()


def test_without_the_lock_the_same_overlap_upcasts_the_writer(tmp_path: Path) -> None:
    """The double reproduces the laptop bug when the two builds are not serialized."""
    default = _SharedDefault()
    entered = threading.Event()
    resume = threading.Event()
    writer_done = threading.Event()
    writer_loader = _building_loader(default, 6, entered, resume)
    check_loader = _building_loader(default, 2, wake=resume, hold=writer_done)

    def side() -> None:
        entered.wait(timeout=2)
        check_loader(tmp_path, dtype="float32")

    worker = threading.Thread(target=side)
    worker.start()
    try:
        model = writer_loader(tmp_path, dtype="bfloat16")
    finally:
        writer_done.set()
        worker.join(timeout=5)
    dtypes = [param.dtype for param in model.parameters()]
    assert dtypes[0] == "bfloat16"
    assert "float32" in dtypes


def test_a_stray_fp32_parameter_is_cast_to_bf16_and_logged(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    model = _Module([_Param("bfloat16"), _Param("float32"), _Param("float32")])
    assert settle_bf16(model, _Torch(), "cuda") == 2
    assert {param.dtype for param in model.parameters()} == {"bfloat16"}
    assert {param.device.type for param in model.parameters()} == {"cuda"}
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "stage writer dtype device=cuda" in log
    assert "loaded=bf16 params=3 fp32=2 cast=2 now=bf16" in log
    assert "writer.fp32_cast" in log


def test_a_bf16_writer_is_logged_without_a_cast(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    model = _Module([_Param("bfloat16"), _Param("bfloat16")])
    assert settle_bf16(model, _Torch(), "cuda") == 0
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "fp32=0 cast=0" in log
    assert "writer.fp32_cast" not in log


def test_a_build_that_rejects_dtype_retries_with_torch_dtype(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    seen: list[dict[str, object]] = []

    def load(folder: Path, **kwargs: object) -> _Module:
        del folder
        seen.append(dict(kwargs))
        if "dtype" in kwargs:
            raise TypeError("unexpected keyword argument 'dtype'")
        return _Module([_Param("bfloat16")])

    transformers = type("T", (), {
        "AutoModelForCausalLM": type("Auto", (), {"from_pretrained": staticmethod(load)}),
    })()
    _causal_model(transformers, _Torch(), tmp_path, "cuda")
    assert seen == [
        {"local_files_only": True, "device_map": "cuda", "dtype": "bfloat16"},
        {"local_files_only": True, "device_map": "cuda", "torch_dtype": "bfloat16"},
    ]
    assert all("low_cpu_mem_usage" not in kwargs for kwargs in seen)


class _Layer:
    def __init__(self) -> None:
        self.keys: object | None = object()
        self.values: object | None = object()


class _StaticCache:
    def __init__(self) -> None:
        self.layers = [_Layer(), _Layer()]


def test_release_drops_the_graph_cache_buffers_and_model() -> None:
    model = _Module([_Param("bfloat16")])
    graph = GraphDecode(model, _Torch(), 64)
    cache = _StaticCache()
    layers = list(cache.layers)
    graph.graph = object()
    graph.cache = cache
    graph.static_input = object()
    graph.cache_position = object()
    graph.static_logits = object()
    graph.release()
    assert graph.graph is None
    assert graph.cache is None
    assert graph.static_input is None
    assert graph.cache_position is None
    assert graph.static_logits is None
    assert graph.model is None
    assert graph.stale()
    assert cache.layers == []
    assert all(layer.keys is None and layer.values is None for layer in layers)


class _Cuda:
    def __init__(self, allocated: int) -> None:
        self.allocated = allocated
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return True

    def synchronize(self) -> None:
        self.calls.append("synchronize")

    def empty_cache(self) -> None:
        self.calls.append("empty_cache")

    def ipc_collect(self) -> None:
        self.calls.append("ipc_collect")

    def reset_peak_memory_stats(self) -> None:
        return None

    def memory_allocated(self) -> int:
        return self.allocated

    def memory_reserved(self) -> int:
        return self.allocated


def test_release_logs_the_residual_and_flags_a_leak(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    clean = _Cuda(64 * 1024**2)
    assert _release_cuda(type("Torch", (), {"cuda": clean})(), "writer") == 64 * 1024**2
    assert clean.calls == ["synchronize", "empty_cache", "ipc_collect"]
    leak = _Cuda(WRITER_RESIDUAL_LIMIT_BYTES + 1)
    _release_cuda(type("Torch", (), {"cuda": leak})(), "writer")
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert f"allocated={64 * 1024**2} reserved={64 * 1024**2} over=0" in log
    assert "over=1" in log
    assert "writer.residual" in log


def test_release_without_cuda_is_quiet() -> None:
    assert _release_cuda(None, "writer") == 0
    off = type("Cuda", (), {"is_available": staticmethod(lambda: False)})()
    assert _release_cuda(type("Torch", (), {"cuda": off})(), "writer") == 0


def test_close_releases_the_graph_and_moves_every_cuda_tensor_off(tmp_path: Path, monkeypatch) -> None:
    """A kept writer is parked, and a tensor the park did not cover still leaves the card."""
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    weight = _Param("bfloat16")
    rotary = _Param("float32")
    model = _Module([weight], [rotary])
    model._cache = object()
    cuda = _Cuda(0)
    torch = type("Torch", (), {"cuda": cuda})()
    deactivated: list[str] = []

    def deactivate(name: str) -> None:
        deactivated.append(name)
        # The park reassigns the weights it holds. The buffer is not in its dict.
        weight.data = weight.data.to("cpu")

    monkeypatch.setattr(enhance, "_optional_torch", lambda: torch)
    monkeypatch.setattr(enhance.hot, "retained", lambda name: name == "writer")
    monkeypatch.setattr(enhance.hot, "deactivate", deactivate)
    rewriter = QwenRewriter(tmp_path, tmp_path)
    rewriter._model = model
    graph = GraphDecode(model, torch, 64)
    graph.graph = object()
    graph.cache = _StaticCache()
    rewriter._graph = graph
    rewriter.close()
    assert rewriter._graph is None
    assert graph.model is None and graph.cache is None
    assert deactivated == ["writer"]
    assert weight.device.type == "cpu"
    assert rotary.device.type == "cpu"
    assert model._cache is None
    assert rewriter._model is model
    assert "empty_cache" in cuda.calls
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "stage writer release device=cuda" in log
    assert "allocated=0 reserved=0 over=0" in log


def test_reopening_after_close_captures_a_fresh_graph(tmp_path: Path, monkeypatch) -> None:
    """``close`` drops the graph object, so the next load builds a new one on CUDA."""
    model = _Module([_Param("bfloat16")])
    built: list[GraphDecode] = []

    class _Graph(GraphDecode):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built.append(self)

        def capture(self) -> None:
            self.graph = object()

    monkeypatch.setattr(enhance, "GraphDecode", _Graph)
    monkeypatch.setattr(enhance, "_generation_libs", lambda: (_Torch(), None))
    rewriter = QwenRewriter(tmp_path, tmp_path, device="cuda")
    rewriter._model = model
    rewriter._capture_decode()
    first = rewriter._graph
    rewriter._graph = None
    rewriter._capture_decode(rebuild=True)
    assert len(built) == 2
    assert rewriter._graph is not first


@pytest.fixture(autouse=True)
def _clean_park():
    yield
    hot.reset()
