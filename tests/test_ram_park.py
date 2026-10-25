"""RAM copies stay for the session. A later series does not read them from disk."""
from __future__ import annotations

import torch
from torch import nn
from voxprint_common.paths import logs_dir

from canvas.models import hot_tier
from canvas.models.hot_tier import open_module, plan, reset


def _tier(monkeypatch, name: str, total: int) -> None:
    monkeypatch.setenv("VOXPRINT_RAM_MODE", name)
    monkeypatch.setattr(hot_tier, "probed_bytes", lambda: total)
    reset()


def _linear(dtype: torch.dtype) -> nn.Linear:
    module = nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    return module.to(dtype=dtype)


def _weight(module: nn.Linear) -> torch.Tensor:
    return module.weight.detach().cpu()


def test_a_second_open_does_not_read_a_parked_model(monkeypatch) -> None:
    _tier(monkeypatch, "ram96", 96 * 1024**3)
    calls = {"n": 0}
    original = _weight(_linear(torch.bfloat16))

    def load() -> nn.Linear:
        calls["n"] += 1
        module = _linear(torch.bfloat16)
        with torch.no_grad():
            module.weight.copy_(original.to(dtype=torch.bfloat16))
        return module

    first = open_module("transformer", load)
    second = open_module("transformer", load)
    assert calls["n"] == 1
    assert second is first
    assert _weight(first).dtype is torch.bfloat16
    hot_tier.deactivate("transformer")
    assert _weight(first).dtype is torch.bfloat16
    assert torch.equal(_weight(first), original.to(dtype=torch.bfloat16))


def test_parking_keeps_the_gate_dtype(monkeypatch) -> None:
    _tier(monkeypatch, "ram96", 96 * 1024**3)
    module = _linear(torch.float32)
    before = _weight(module).clone()
    assert hot_tier.remember("gate_nsfw", module) is True
    hot_tier.activate("gate_nsfw", "cpu")
    hot_tier.deactivate("gate_nsfw")
    assert _weight(module).dtype is torch.float32
    assert torch.equal(_weight(module), before)


def test_ram32_reads_the_disk_again(monkeypatch) -> None:
    _tier(monkeypatch, "ram32", 32 * 1024**3)
    calls = {"n": 0}

    def load() -> nn.Linear:
        calls["n"] += 1
        return _linear(torch.float32)

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


def test_park_and_unpark_log_milliseconds_bytes_ram_and_vram(monkeypatch) -> None:
    _tier(monkeypatch, "ram96", 96 * 1024**3)
    log = logs_dir() / "voxprint.log"
    before = log.stat().st_size if log.is_file() else 0
    module = _linear(torch.float32)
    hot_tier.remember("writer", module)
    hot_tier.activate("writer", "cpu")
    hot_tier.deactivate("writer")
    text = log.read_bytes()[before:].decode("utf-8")
    assert "stage hot plan" in text
    assert "model=writer where=ram" in text
    assert "stage park writer" in text
    assert "vram_mib=" in text and "ram_mib=" in text
    assert "ms=" in text and "bytes=" in text
    assert "where=disk" not in text
