"""Greedy writer decode without a CUDA graph.

Capture failure stays on the eager argmax loop and is logged. A stand-in with
no ``forward`` leaves the reply to ``generate``. A captured graph has to emit
the same ids as eager greedy, and the decode clock waits for CUDA.
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from canvas.pipeline.device_guard import DeviceRefused
from canvas.pipeline.writer_decode import GraphDecode, eager_decode, try_decode


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> bool:
        del args
        return False


class _Row:
    def __init__(self, token: int) -> None:
        self.token = token
        self.shape = (1, 1)
        self.device = "cpu"

    def argmax(self, dim: int = -1, keepdim: bool = True) -> _Row:
        del dim, keepdim
        return _Row(self.token)


class _Logits:
    def __init__(self, token: int) -> None:
        self.token = token

    def __getitem__(self, key: object) -> _Row:
        del key
        return _Row(self.token)


class _Output:
    def __init__(self, token: int) -> None:
        self.logits = _Logits(token)


class _Ids:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)
        self.device = "cpu"


class _Cache:
    def reset(self) -> None:
        return None


class _Model:
    def __init__(self) -> None:
        self.config = object()
        self.shapes: list[tuple[int, ...]] = []
        self.step = 0

    def forward(self, **kwargs: Any) -> _Output:
        ids = kwargs["input_ids"]
        self.shapes.append(tuple(ids.shape))
        self.step += 1
        return _Output(self.step)

    def __call__(self, **kwargs: Any) -> _Output:
        return self.forward(**kwargs)


class _Torch:
    @staticmethod
    def inference_mode() -> _Null:
        return _Null()

    @staticmethod
    def arange(n: int, device: object = None) -> int:
        del device
        return n

    @staticmethod
    def cat(pieces: list[_Row], dim: int = -1) -> list[int]:
        del dim
        return [piece.token for piece in pieces]


def test_capture_without_a_config_stays_eager(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    graph = GraphDecode(types.SimpleNamespace(), torch, 32)
    graph.capture()
    assert graph.label == "eager"
    assert graph.reason == "no_config"
    assert graph.graph is None
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "graph=eager" in log
    assert "reason=no_config" in log


def test_a_failed_capture_is_logged_and_stays_eager(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))

    def boom(model: object, length: int) -> object:
        del model, length
        raise RuntimeError("capture failed")

    monkeypatch.setattr("canvas.pipeline.writer_decode._new_cache", boom)
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    graph = GraphDecode(types.SimpleNamespace(config=object()), torch, 32)
    graph.capture()
    assert graph.label == "eager"
    assert graph.reason == "runtimeerror"
    assert graph.graph is None
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "graph=eager reason=runtimeerror" in log


def test_eager_decode_prefills_then_steps_one_token(monkeypatch) -> None:
    fake = types.ModuleType("transformers")

    def StaticCache(config: object, max_cache_len: int) -> _Cache:
        assert config is not None
        assert max_cache_len == 7
        return _Cache()

    fake.StaticCache = StaticCache  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake)
    model = _Model()
    found = eager_decode(model, _Torch(), _Ids(4), 3)
    assert found is not None
    assert found.graph == "eager"
    assert found.count == 3
    assert found.prefill_ms >= 0
    assert found.ids == [1, 2, 3]
    assert model.shapes == [(1, 4), (1, 1), (1, 1)]


def test_a_model_without_forward_leaves_decode_to_generate() -> None:
    torch = _Torch()
    assert eager_decode(object(), torch, _Ids(2), 4) is None
    assert try_decode(object(), torch, _Ids(2), 4, None) is None


def test_cuda_refuses_a_writer_left_on_cpu_or_meta(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    ran = {"n": 0}

    def model(kind: str) -> Any:
        class _Weight:
            device = types.SimpleNamespace(type=kind)

        class _Model:
            def parameters(self):
                yield _Weight()

            def buffers(self):
                return iter(())

            def forward(self, **kwargs: object) -> object:
                del kwargs
                ran["n"] += 1
                raise AssertionError("decoded")

        return _Model()

    for kind in ("cpu", "meta"):
        with pytest.raises(DeviceRefused):
            try_decode(model(kind), torch, _Ids(2), 3, None)
    assert ran["n"] == 0


def test_captured_tokens_match_eager_and_the_clock_waits_for_sync(monkeypatch) -> None:
    fake = types.ModuleType("transformers")

    class StaticCache:
        def __init__(self, config: object, max_cache_len: int) -> None:
            assert config is not None
            assert max_cache_len >= 6

        def reset(self) -> None:
            return None

    fake.StaticCache = StaticCache  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake)
    syncs = {"n": 0}

    def synchronize() -> None:
        syncs["n"] += 1
        time.sleep(0.05)

    class _Token:
        def __init__(self, value: int) -> None:
            self.value = value
            self.shape = (1, 1)
            self.device = "cuda"

        def argmax(self, dim: int = -1, keepdim: bool = True) -> _Token:
            del dim, keepdim
            return _Token(self.value)

        def copy_(self, other: _Token) -> None:
            self.value = other.value

        def fill_(self, value: int) -> None:
            self.value = value

    class _Logits:
        def __init__(self, value: int) -> None:
            self.value = value
            self.shape = (1, 1, 4)

        def __getitem__(self, key: object) -> _Token:
            del key
            return _Token(self.value)

        def copy_(self, other: _Logits) -> None:
            self.value = other.value

    class _Output:
        def __init__(self, value: int) -> None:
            self.logits = _Logits(value)

    class _Prompt:
        def __init__(self, values: list[int]) -> None:
            self.values = list(values)
            self.shape = (1, len(values))
            self.device = "cuda"

    class _Model:
        def __init__(self) -> None:
            self.config = object()

        def forward(self, **kwargs: Any) -> _Output:
            ids = kwargs["input_ids"]
            last = ids.values[-1] if hasattr(ids, "values") else ids.value
            return _Output(last + 3)

        def __call__(self, **kwargs: Any) -> _Output:
            return self.forward(**kwargs)

    class _Stream:
        def wait_stream(self, other: object) -> None:
            del other

    class _Graph:
        def __init__(self) -> None:
            self.body: Any = None

        def bind_replay(self, fn: Any) -> None:
            self.body = fn

        def replay(self) -> None:
            self.body()

    class _GraphCM:
        def __init__(self, graph: _Graph) -> None:
            self.graph = graph

        def __enter__(self) -> _GraphCM:
            return self

        def __exit__(self, *args: object) -> bool:
            del args
            return False

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            synchronize()

        @staticmethod
        def Stream() -> _Stream:
            return _Stream()

        @staticmethod
        def current_stream() -> _Stream:
            return _Stream()

        @staticmethod
        def stream(stream: object) -> _Null:
            del stream
            return _Null()

        @staticmethod
        def CUDAGraph() -> _Graph:
            return _Graph()

        @staticmethod
        def graph(graph: _Graph) -> _GraphCM:
            return _GraphCM(graph)

    class _CaptureTorch:
        long = "long"
        cuda = _Cuda

        @staticmethod
        def inference_mode() -> _Null:
            return _Null()

        @staticmethod
        def no_grad() -> _Null:
            return _Null()

        @staticmethod
        def zeros(shape: tuple[int, ...], dtype: object = None, device: object = None) -> _Token:
            del shape, dtype, device
            return _Token(0)

        @staticmethod
        def empty_like(logits: _Logits) -> _Logits:
            return _Logits(logits.value)

        @staticmethod
        def arange(n: int, device: object = None) -> int:
            del device
            return n

        @staticmethod
        def cat(pieces: list[_Token], dim: int = -1) -> list[int]:
            del dim
            return [piece.value for piece in pieces]

    model = _Model()
    torch = _CaptureTorch()
    graph = GraphDecode(model, torch, 32)
    graph.capture()
    assert graph.label == "captured"
    before = syncs["n"]
    found = graph.run(_Prompt([4, 5]), 4)
    assert found is not None
    eager = eager_decode(model, torch, _Prompt([4, 5]), 4)
    assert eager is not None
    assert found.ids == eager.ids == [8, 11, 14, 17]
    assert found.count == 4
    assert eager.count == 4
    assert found.graph == "captured"
    assert syncs["n"] >= before + 2
    assert found.prefill_ms >= 40
    assert 0 < found.decode_tok_s < 500


def test_gpu_graph_matches_eager_greedy_on_the_same_prompt() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    class _Cache:
        def reset(self) -> None:
            return None

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(32, 8)
            self.proj = torch.nn.Linear(8, 32)
            self.config = object()

        def forward(self, input_ids: Any, past_key_values: Any = None, use_cache: bool = False,
                    cache_position: Any = None) -> Any:
            del past_key_values, use_cache, cache_position
            return types.SimpleNamespace(logits=self.proj(self.embed(input_ids)))

    torch.manual_seed(0)
    model = Tiny().cuda().eval()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("canvas.pipeline.writer_decode._new_cache", lambda model, length: _Cache())
    try:
        graph = GraphDecode(model, torch, 32)
        graph.capture()
        assert graph.label == "captured"
        prompt = torch.tensor([[1, 2, 3, 4]], device="cuda")
        found = graph.run(prompt, 5)
        eager = eager_decode(model, torch, prompt, 5)
    finally:
        monkeypatch.undo()
    assert found is not None and eager is not None
    assert torch.equal(found.ids, eager.ids)
    assert found.count == int(found.ids.shape[-1]) == 5
