"""A LoRA whose weights reach no layer fails as lora.invalid instead of drawing without it."""
from __future__ import annotations

from pathlib import Path

import pytest

from canvas.models import runtime
from canvas.models.user_lora import CODE_INVALID, LoraRejected, UserLora, apply_lora


class _Layer:
    def __init__(self) -> None:
        self.lora_A: dict[str, object] = {}


class _Transformer:
    def __init__(self, *, matches: bool) -> None:
        self.layer = _Layer()
        self.matches = matches
        self.weights: list[float] = []
        self.peft_config: dict[str, object] = {}

    def named_modules(self):
        yield "", self
        yield "layer", self.layer

    def load_lora_adapter(self, path: str, adapter_name: str) -> None:
        del path
        if self.matches:
            self.layer.lora_A[adapter_name] = object()
            self.peft_config[adapter_name] = object()

    def set_adapters(self, names, weights) -> None:
        del names
        self.weights = list(weights)

    def delete_adapters(self, names) -> None:
        for name in names:
            self.peft_config.pop(name, None)
            self.layer.lora_A.pop(name, None)


def _lora(tmp_path: Path) -> UserLora:
    return UserLora(
        id="shape", kind="lora", folder=tmp_path, weights=tmp_path / "a.safetensors", trigger="",
        default_strength=0.8, strength_range=(0.0, 1.5), adult_content=False,
    )


def test_a_lora_that_reaches_no_layer_is_invalid(tmp_path: Path) -> None:
    transformer = _Transformer(matches=False)
    with pytest.raises(LoraRejected) as caught:
        apply_lora(transformer, _lora(tmp_path), 0.8)
    assert caught.value.code == CODE_INVALID
    assert transformer.weights == []


def test_the_invalid_code_survives_the_runtime_wrapper(tmp_path: Path) -> None:
    with pytest.raises(LoraRejected) as caught:
        runtime._apply_lora_reported(_Transformer(matches=False), _lora(tmp_path), 0.8)
    assert caught.value.code == CODE_INVALID


def test_a_matching_lora_still_applies(tmp_path: Path) -> None:
    transformer = _Transformer(matches=True)
    assert apply_lora(transformer, _lora(tmp_path), 0.8) == 0.8
    assert transformer.weights == [0.8]


def test_the_stop_message_names_the_unreadable_lora() -> None:
    from canvas.app import photoshoot_cli

    assert photoshoot_cli._english_stop_message(CODE_INVALID) == "This LoRA file cannot be read."
