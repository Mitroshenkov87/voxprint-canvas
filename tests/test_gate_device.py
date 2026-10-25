"""Gate models: GPU placement when it fits, the same check again on the CPU after a device error."""
from __future__ import annotations

import math
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest

from canvas.safety import adapters
from canvas.safety.adapters import NsfwViT, gate_device
from canvas.safety.gate import AgeGate, SafetyStack
from canvas.safety.image import RawImage

GB = 1024**3


class OutOfMemoryError(RuntimeError):
    """Same type name as torch.cuda.OutOfMemoryError."""


class _T:
    def __init__(self, rows: list) -> None:
        self.rows = rows

    def __getitem__(self, index: int) -> object:
        item = self.rows[index]
        return _T(item) if isinstance(item, list) else item

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.rows),)

    def cpu(self) -> _T:
        return self


def _softmax(tensor: _T, dim: int = -1) -> _T:
    del dim
    out = []
    for row in tensor.rows:
        peak = max(row)
        exps = [math.exp(value - peak) for value in row]
        total = sum(exps)
        out.append([value / total for value in exps])
    return _T(out)


def _fake_torch(free: int, reserved: int = 0, cuda: bool = True) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.no_grad = nullcontext  # type: ignore[attr-defined]
    torch.softmax = _softmax  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: cuda,
        mem_get_info=lambda: (free, 16 * GB),
        memory_reserved=lambda: reserved,
        empty_cache=lambda: None,
    )
    return torch


class _Input:
    def __init__(self) -> None:
        self.device = "cpu"

    def to(self, device: str) -> _Input:
        moved = _Input()
        moved.device = device
        return moved


class _Model:
    def __init__(self, fail_on_cuda: bool = False) -> None:
        self.device = "cpu"
        self.fail_on_cuda = fail_on_cuda
        self.calls: list[str] = []
        self.config = types.SimpleNamespace(id2label={0: "normal", 1: "nsfw"})

    def to(self, device: str) -> _Model:
        self.device = device
        return self

    def parameters(self) -> list[object]:
        return []

    def buffers(self) -> list[object]:
        return []

    def __call__(self, pixel_values: _Input) -> object:
        assert pixel_values.device == self.device
        self.calls.append(self.device)
        if self.device == "cuda" and self.fail_on_cuda:
            raise OutOfMemoryError("CUDA out of memory")
        return types.SimpleNamespace(logits=_T([[0.25, 1.75]]))


def _nsfw(tmp_path: Path, model: _Model, device: str) -> NsfwViT:
    check = NsfwViT(tmp_path)
    check._model = model
    check._processor = lambda images, return_tensors: {"pixel_values": _Input()}
    check.device = device
    model.device = device
    return check


_PICTURE = RawImage(2, 2, bytes(12))


def test_own_weights_are_not_a_foreign_generator(monkeypatch) -> None:
    """A few hundred MiB of allocator slack is not the 12 GB picture reserve."""
    adapters._gate_cuda_bytes = 2 * GB
    try:
        monkeypatch.setitem(
            sys.modules, "torch", _fake_torch(free=13 * GB, reserved=2 * GB + 128 * 1024**2),
        )
        assert gate_device(2 * GB) == "cuda"
    finally:
        adapters._gate_cuda_bytes = 0


def test_device_choice_respects_the_generator_reserve(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free=15 * GB, reserved=0))
    assert gate_device(2 * GB) == "cuda"  # empty card: no generator reserve
    assert gate_device(3 * GB) == "cuda"
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free=5 * GB, reserved=11 * GB))
    assert gate_device(4 * GB) == "cpu"  # 1 GB of the reserve is still needed
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free=6 * GB, reserved=12 * GB))
    assert gate_device(4 * GB) == "cuda"
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free=40 * GB, cuda=False))
    assert gate_device(1) == "cpu"


def test_place_moves_to_cuda_and_a_failed_move_stays_on_cpu(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free=15 * GB))
    monkeypatch.setattr(adapters, "gate_device", lambda _need: "cuda")
    check = NsfwViT(tmp_path)
    model = check._place(_Model())
    assert check.device == "cuda" and model.device == "cuda"

    class _Stuck(_Model):
        def to(self, device: str) -> _Model:
            if device == "cuda":
                raise OutOfMemoryError("CUDA out of memory")
            return super().to(device)

    other = NsfwViT(tmp_path)
    model = other._place(_Stuck())
    assert other.device == "cpu" and model.device == "cpu"


def test_an_oom_runs_the_same_check_on_the_cpu(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free=15 * GB))
    reference = _nsfw(tmp_path, _Model(), "cpu").sexual_score(_PICTURE)
    model = _Model(fail_on_cuda=True)
    check = _nsfw(tmp_path, model, "cuda")
    assert check.sexual_score(_PICTURE) == reference
    assert model.calls == ["cuda", "cpu"]
    assert check.device == "cpu"


def test_cpu_and_gpu_paths_give_the_same_verdict(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free=15 * GB))
    cpu = _nsfw(tmp_path, _Model(), "cpu").sexual_score(_PICTURE)
    gpu_model = _Model()
    gpu = _nsfw(tmp_path, gpu_model, "cuda").sexual_score(_PICTURE)
    assert cpu == gpu
    assert gpu_model.calls == ["cuda"]


class _Held:
    def __init__(self, device: str) -> None:
        self.device = device
        self.released = False

    def release(self) -> None:
        self.released = True
        self.device = "cpu"


def test_cuda_weights_are_parked_when_the_device_flag_says_cpu() -> None:
    class _Tensor:
        def __init__(self) -> None:
            self.device = types.SimpleNamespace(type="cuda")

    class _Model:
        def parameters(self):
            yield _Tensor()

        def buffers(self):
            return iter(())

    leaked = _Held("cpu")
    leaked._model = _Model()
    gate = AgeGate(SafetyStack(leaked, _Held("cpu"), _Held("cpu"), _Held("cpu"), _Held("cpu"), _Held("cpu")))
    gate._load_state = "idle"
    gate.release_weights()
    assert leaked.released is True


def test_gpu_weights_are_dropped_and_cpu_weights_stay() -> None:
    gpu = _Held("cuda")
    cpu = _Held("cpu")
    gate = AgeGate(SafetyStack(gpu, cpu, cpu, cpu, cpu, cpu))
    gate._load_state = "ready"
    gate.release_weights()
    assert gpu.released is True
    assert cpu.released is False
    assert gate._load_state == "idle"

    staying = _Held("cpu")
    other = AgeGate(SafetyStack(staying, staying, staying, staying, staying, staying))
    other._load_state = "ready"
    other.release_weights()
    assert staying.released is False
    assert other._load_state == "ready"


def test_a_non_device_error_is_not_retried(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free=15 * GB))

    class _Broken(_Model):
        def __call__(self, pixel_values: _Input) -> object:
            self.calls.append(self.device)
            raise ValueError("bad input")

    model = _Broken()
    check = _nsfw(tmp_path, model, "cuda")
    with pytest.raises(ValueError):
        check.sexual_score(_PICTURE)
    assert model.calls == ["cuda"]


def test_weights_already_on_cuda_are_not_copied_again(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free=15 * GB))
    monkeypatch.setattr(adapters, "gate_device", lambda _need: "cuda")
    adapters._gate_cuda_bytes = 0

    class _Tensor:
        device = types.SimpleNamespace(type="cuda")

        def numel(self) -> int:
            return 4

        def element_size(self) -> int:
            return 2

    class _Ready:
        def __init__(self) -> None:
            self.copied: list[str] = []

        def to(self, device: str) -> _Ready:
            self.copied.append(device)
            return self

        def parameters(self):
            yield _Tensor()

        def buffers(self):
            return iter(())

    try:
        model = _Ready()
        check = NsfwViT(tmp_path)
        check._park_name = "gate_nsfw"
        check._place(model)
        check._place(model)
        assert model.copied == []
        assert check.device == "cuda"
        assert adapters._gate_cuda_bytes == 16
    finally:
        adapters._gate_cuda_bytes = 0


def test_a_retained_check_parks_the_pinned_copy(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(free=15 * GB))
    monkeypatch.setattr(adapters, "gate_device", lambda _need: "cuda")
    moves: list[str] = []
    monkeypatch.setattr(adapters, "retained", lambda name: name == "gate_nsfw")

    def activate(name: str, device: str, blocking: bool = False) -> bool:
        del blocking
        moves.append(f"unpark:{name}:{device}")
        return True

    monkeypatch.setattr(adapters, "activate", activate)
    monkeypatch.setattr(adapters, "deactivate", lambda name: moves.append(f"park:{name}"))
    adapters._gate_cuda_bytes = 0

    class _Tensor:
        def __init__(self) -> None:
            self.device = types.SimpleNamespace(type="cpu")

        def numel(self) -> int:
            return 4

        def element_size(self) -> int:
            return 2

    class _Ready:
        def __init__(self) -> None:
            self.copied: list[str] = []
            self.tensor = _Tensor()

        def to(self, device: str) -> _Ready:
            self.copied.append(device)
            return self

        def parameters(self):
            yield self.tensor

        def buffers(self):
            return iter(())

    try:
        model = _Ready()
        check = NsfwViT(tmp_path)
        check._park_name = "gate_nsfw"
        check._model = check._place(model)
        check.release()
        assert model.copied == []
        assert moves == ["unpark:gate_nsfw:cuda", "park:gate_nsfw"]
        assert check.device == "cpu"
        assert check._model is model
        assert adapters._gate_cuda_bytes == 0
    finally:
        adapters._gate_cuda_bytes = 0


def test_siglip_text_embeddings_are_reused_for_the_same_labels() -> None:
    calls: list[object] = []

    class _Embed:
        def norm(self, dim: int = -1, keepdim: bool = False) -> _Embed:
            del dim, keepdim
            return self

        def __truediv__(self, _other: object) -> _Embed:
            return self

        def to(self, _device: str) -> _Embed:
            return self

    class _Model:
        dtype = "bf16"

        def get_text_features(self, **kwargs: object) -> _Embed:
            calls.append(kwargs)
            return _Embed()

    class _Processor:
        def __call__(self, **_kwargs: object) -> dict[str, _Embed]:
            return {"input_ids": _Embed()}

    torch = types.SimpleNamespace(no_grad=nullcontext)
    model = _Model()
    adapters._siglip_text.clear()
    try:
        adapters._siglip_text_embeds(torch, model, _Processor(), ["minor", "adult"], "cuda")
        adapters._siglip_text_embeds(torch, model, _Processor(), ["minor", "adult"], "cuda")
        adapters._siglip_text_embeds(torch, model, _Processor(), ["minor", "adult"], "cpu")
        assert len(calls) == 2
    finally:
        adapters._siglip_text.clear()
