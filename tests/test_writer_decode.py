"""Greedy writer decode without a CUDA graph.

Capture failure stays on the eager argmax loop and is logged. A stand-in with
no ``forward`` leaves the reply to ``generate``.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

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
