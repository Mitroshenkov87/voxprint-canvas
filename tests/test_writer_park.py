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
    weight = _Tensor()

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
        reply = model_rewrite(Tier.GB16, False)("a red cup on a table")
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
