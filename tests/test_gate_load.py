"""Component load is separate from the inference bound."""
import threading
import time
from pathlib import Path

import pytest

from canvas.safety.classifiers import GuardVerdict
from canvas.safety.constants import CLASSIFIER_TIMEOUT_S, IMAGE_TIMEOUT_S, MODEL_LOAD_TIMEOUT_S
from canvas.safety.gate import AgeGate, SafetyStack, SafetyUnavailable
from tests.fakes import World


def test_model_load_bound_is_separate() -> None:
    assert MODEL_LOAD_TIMEOUT_S == 240.0
    assert CLASSIFIER_TIMEOUT_S == 20.0
    assert IMAGE_TIMEOUT_S == 30.0
    source = Path("canvas/safety/gate.py").read_text(encoding="utf-8")
    assert "timeout=MODEL_LOAD_TIMEOUT_S" in source
    assert "timeout=CLASSIFIER_TIMEOUT_S" in source
    assert "timeout=IMAGE_TIMEOUT_S" in source


def test_a_prompt_loads_the_guard_and_leaves_the_image_models() -> None:
    calls: list[str] = []

    class Part:
        def __init__(self, name: str) -> None:
            self.name = name

        def load(self) -> None:
            calls.append(self.name)

        def assess(self, text: str) -> GuardVerdict:
            del text
            return GuardVerdict("safe", ())

    stack = SafetyStack(
        Part("guard"), Part("zeroshot"), Part("faces"), Part("ages"), Part("nsfw"), Part("fine"),
    )
    AgeGate(stack).review_prompt("a red boat")
    assert calls == ["guard"]


def test_warm_loads_every_component_once() -> None:
    calls: list[str] = []

    class Part:
        def __init__(self, name: str) -> None:
            self.name = name

        def load(self) -> None:
            calls.append(self.name)

    stack = SafetyStack(
        Part("guard"), Part("zeroshot"), Part("faces"), Part("ages"), Part("nsfw"), Part("fine"),
    )
    gate = AgeGate(stack)
    gate.warm()
    gate.warm()
    assert calls == ["guard", "nsfw", "zeroshot", "faces", "ages", "fine"]


class _SlowGuard:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.loads = 0
        self.assessed: list[str] = []
        self.started = threading.Event()
        self.loaded = threading.Event()

    def load(self) -> None:
        self.loads += 1
        self.started.set()
        time.sleep(self.delay)
        self.loaded.set()

    def assess(self, text: str) -> GuardVerdict:
        assert self.loaded.is_set()
        self.assessed.append(text)
        return GuardVerdict("safe", ())


def _stack(guard: _SlowGuard) -> SafetyStack:
    stack = World().stack()
    object.__setattr__(stack, "guard", guard)
    return stack


def test_a_slow_load_then_a_fast_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("canvas.safety.gate.CLASSIFIER_TIMEOUT_S", 0.05)
    guard = _SlowGuard(0.2)
    gate = AgeGate(_stack(guard))
    gate.warm()
    gate.review_prompt("a red boat")
    assert guard.loads == 1
    assert guard.assessed == ["a red boat"]


def test_a_load_past_the_bound_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("canvas.safety.gate.MODEL_LOAD_TIMEOUT_S", 0.05)
    guard = _SlowGuard(0.4)
    gate = AgeGate(_stack(guard))
    with pytest.raises(SafetyUnavailable) as caught:
        gate.warm()
    assert caught.value.code == "timeout"
    assert guard.assessed == []
    started = time.monotonic()
    with pytest.raises(SafetyUnavailable) as again:
        gate.review_prompt("a red boat")
    assert again.value.code == "timeout"
    assert time.monotonic() - started < 0.2
    assert guard.assessed == []


def test_a_check_waits_for_the_load_then_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("canvas.safety.gate.CLASSIFIER_TIMEOUT_S", 0.05)
    guard = _SlowGuard(0.3)
    gate = AgeGate(_stack(guard))
    worker = threading.Thread(target=gate.warm)
    worker.start()
    assert guard.started.wait(1)
    gate.review_prompt("a red boat")
    worker.join(2)
    assert guard.loads == 1
    assert guard.assessed == ["a red boat"]
    assert guard.loaded.is_set()
