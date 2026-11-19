"""The scenario writer parks its VRAM copy when the stage ends.

The default CI job has no torch. The double below has a dtype and a ``.to`` so
the RAM park can pin it, and a device that follows ``.data`` the way a parameter
does. ``model_rewrite`` and ``QwenRewriter.rewrite`` are the real stage paths.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canvas.app.photoshoot_cli import model_rewrite
from canvas.infra import vram_trace
from canvas.models import hot_tier
from canvas.models.modes import Tier
from canvas.pipeline.enhance import QwenRewriter


class _Device:
    def __init__(self, kind: str) -> None:
        self.type = kind


class _Tensor:
    """One weight. ``device`` follows ``.data`` after the park assigns a copy."""

    def __init__(self, kind: str = "cpu") -> None:
        self._device = _Device(kind)
        self.dtype = "bfloat16"
        self.values = [1, 2, 3, 4]
        self.nbytes = 16
        self.shape = (1, 2)
        self.data = self
        self.pinned = False

    @property
    def device(self) -> _Device:
        current = self.data
        if current is self:
            return self._device
        return current.device

    def detach(self) -> _Tensor:
        return self

    def to(self, device: object, non_blocking: bool = False) -> _Tensor:
        del non_blocking
        kind = "cuda" if "cuda" in str(device) else "cpu"
        copy = _Tensor(kind)
        copy.pinned = self.pinned
        return copy

    def pin_memory(self) -> _Tensor:
        pinned = self.to("cpu")
        pinned.pinned = True
        return pinned


class _Output:
    def __getitem__(self, key: object) -> list[int]:
        del key
        return [1, 2, 3]


class _Tokenizer:
    def apply_chat_template(
        self, messages: object, tokenize: bool = False, add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        del messages, tokenize, add_generation_prompt, enable_thinking
        return "ready"

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, _Tensor]:
        del text, return_tensors
        return {"input_ids": _Tensor()}

    def decode(self, tokens: object, skip_special_tokens: bool = True) -> str:
        del tokens, skip_special_tokens
        return "1. wide shot"


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> bool:
        del args
        return False


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def empty_cache() -> None:
        return None


class _Torch:
    bfloat16 = "bf16"
    cuda = _Cuda

    @staticmethod
    def inference_mode() -> _Null:
        return _Null()


class _Transformers:
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> _Tokenizer:
            del args, kwargs
            return _Tokenizer()


def _writer_dir(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"], "tie_word_embeddings": True}),
        encoding="utf-8",
    )
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model.safetensors"}}),
        encoding="utf-8",
    )
    return folder


def _arm(monkeypatch, tmp_path: Path, *, fail: bool) -> tuple[_Tensor, list[str]]:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("VOXPRINT_RAM_MODE", "ram96")
    hot_tier.reset()
    vram_trace._holders.clear()
    folder = _writer_dir(tmp_path / "writer")
    seen: list[str] = []
    weight = _Tensor("cuda")

    class _Model:
        def named_parameters(self):
            yield "weight", weight

        def parameters(self):
            yield weight

        def named_buffers(self):
            return iter(())

        def buffers(self):
            return iter(())

        def generate(self, **kwargs: object) -> _Output:
            del kwargs
            seen.append(weight.device.type)
            if fail:
                raise RuntimeError("stopped")
            return _Output()

    monkeypatch.setattr("canvas.pipeline.enhance.snapshot_ready", lambda *_args, **_kwargs: folder)
    monkeypatch.setattr(
        "canvas.pipeline.enhance._generation_libs", lambda: (_Torch(), _Transformers()),
    )
    monkeypatch.setattr(
        "canvas.pipeline.enhance._causal_model", lambda *_args, **_kwargs: _Model(),
    )
    return weight, seen


def _log(tmp_path: Path) -> str:
    return (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")


def test_the_writer_stage_parks_its_vram_copy(monkeypatch, tmp_path: Path) -> None:
    weight, seen = _arm(monkeypatch, tmp_path, fail=False)
    try:
        rewrite = model_rewrite(Tier.GB16, False)
        reply = rewrite("a red cup on a table")
        rewrite.release()  # type: ignore[attr-defined]
    finally:
        hot_tier.reset()
        vram_trace._holders.clear()
    assert reply.startswith("1.")
    assert seen == ["cuda"]
    assert weight.device.type == "cpu"
    log = _log(tmp_path)
    assert "stage park writer" in log
    assert "stage unpark writer" in log
    assert "stage park residual" not in log


def test_unpark_puts_every_tensor_on_cuda_and_recaptures(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("VOXPRINT_RAM_MODE", "ram96")
    hot_tier.reset()
    vram_trace._holders.clear()
    folder = _writer_dir(tmp_path / "writer")
    weight = _Tensor("cuda")
    buffer = _Tensor("cuda")
    caches: list[int] = []

    class _Cache:
        def reset(self) -> None:
            return None

    def new_cache(model: object, length: int) -> _Cache:
        del model
        caches.append(length)
        return _Cache()

    class _Token:
        def __init__(self, value: int) -> None:
            self.value = value
            self.shape = (1, 1)
            self.device = "cuda"

        def argmax(self, dim: int = -1, keepdim: bool = True) -> _Token:
            del dim, keepdim
            return _Token(self.value)

        def copy_(self, other: object) -> None:
            self.value = int(getattr(other, "value", 0))

        def fill_(self, value: int) -> None:
            self.value = int(value)

        def to(self, device: object, non_blocking: bool = False) -> _Token:
            del device, non_blocking
            return self

    class _Logits:
        def __init__(self, value: int) -> None:
            self.value = value

        def __getitem__(self, key: object) -> _Token:
            del key
            return _Token(self.value)

        def copy_(self, other: _Logits) -> None:
            self.value = other.value

    class _Output:
        def __init__(self, value: int) -> None:
            self.logits = _Logits(value)

    class _Model:
        config = object()

        def named_parameters(self):
            yield "weight", weight

        def parameters(self):
            yield weight

        def named_buffers(self):
            yield "inv_freq", buffer

        def buffers(self):
            yield buffer

        def forward(self, **kwargs: object) -> _Output:
            ids = kwargs["input_ids"]
            last = ids.values[-1] if hasattr(ids, "values") else getattr(ids, "value", 1)
            assert weight.device.type == "cuda"
            assert buffer.device.type == "cuda"
            return _Output(int(last) + 1)

        def __call__(self, **kwargs: object) -> _Output:
            return self.forward(**kwargs)

    class _Stream:
        def wait_stream(self, other: object) -> None:
            del other

    class _Graph:
        def __init__(self) -> None:
            self.body = None

        def bind_replay(self, fn: object) -> None:
            self.body = fn

        def replay(self) -> None:
            assert self.body is not None
            self.body()

    class _CM:
        def __enter__(self) -> _CM:
            return self

        def __exit__(self, *args: object) -> bool:
            del args
            return False

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def Stream() -> _Stream:
            return _Stream()

        @staticmethod
        def current_stream() -> _Stream:
            return _Stream()

        @staticmethod
        def stream(stream: object) -> _CM:
            del stream
            return _CM()

        @staticmethod
        def CUDAGraph() -> _Graph:
            return _Graph()

        @staticmethod
        def graph(graph: _Graph) -> _CM:
            del graph
            return _CM()

    class _Torch:
        bfloat16 = "bf16"
        long = "long"
        cuda = _Cuda

        @staticmethod
        def inference_mode() -> _CM:
            return _CM()

        @staticmethod
        def no_grad() -> _CM:
            return _CM()

        @staticmethod
        def zeros(shape: object, dtype: object = None, device: object = None) -> _Token:
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

    monkeypatch.setattr("canvas.pipeline.enhance._generation_libs", lambda: (_Torch(), _Transformers()))
    monkeypatch.setattr("canvas.pipeline.enhance._causal_model", lambda *_args, **_kwargs: _Model())
    monkeypatch.setattr("canvas.pipeline.writer_decode._new_cache", new_cache)
    try:
        first = QwenRewriter(folder, folder, tier=Tier.GB16, max_new_tokens=4)
        assert first.complete("ping", system="sys").startswith("1.")
        first.close()
        assert weight.device.type == "cpu"
        assert buffer.device.type == "cpu"
        second = QwenRewriter(folder, folder, tier=Tier.GB16, max_new_tokens=4)
        assert second.complete("ping", system="sys").startswith("1.")
        second.close()
    finally:
        hot_tier.reset()
        vram_trace._holders.clear()
    assert weight.device.type == "cpu"
    assert buffer.device.type == "cpu"
    assert len(caches) >= 2
    log = _log(tmp_path)
    token_lines = [line for line in log.splitlines() if "stage writer tokens" in line]
    assert len(token_lines) >= 2
    assert "device=cuda" in token_lines[1]
    assert "graph=captured" in token_lines[1]


def test_a_writer_error_still_parks_the_vram_copy(monkeypatch, tmp_path: Path) -> None:
    weight, seen = _arm(monkeypatch, tmp_path, fail=True)
    folder = tmp_path / "writer"
    rewriter = QwenRewriter(folder, folder, tier=Tier.GB16, stage="writer")
    try:
        with pytest.raises(RuntimeError, match="stopped"):
            rewriter.rewrite("a red cup on a table")
    finally:
        hot_tier.reset()
        vram_trace._holders.clear()
    assert seen == ["cuda"]
    assert weight.device.type == "cpu"
    assert rewriter.closed is True
    log = _log(tmp_path)
    assert "stage park writer" in log
    assert "stage park residual" not in log
