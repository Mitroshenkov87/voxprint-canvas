"""Cook draws every frame, then each frame is released before it is saved.

Owner rule 2026-11-16: no classifier runs on generated frames and no image
check model is loaded for a series. Only the prompt guard is warmed, and a
guard that cannot load stops the series before a frame is drawn.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from canvas.infra.stage_timer import StageTimes
from canvas.pipeline import series
from canvas.pipeline.series import SerialGate
from canvas.safety.gate import GenerationBlocked, SafetyUnavailable, SafetyError
from tests.fakes import World

_IMAGE_CALLS = {"nsfw", "image-score", "faces", "ages", "fine"}
_IMAGE_MODELS = ("nsfw", "zeroshot", "faces", "ages", "fine")


def _record_releases(monkeypatch, on_release) -> None:
    real = series.release_generated

    def release_generated(raw: object) -> object:
        on_release()
        return real(raw)  # type: ignore[arg-type]

    monkeypatch.setattr(series, "release_generated", release_generated)
from tests.test_photoshoot import IDEA, _frame, _payload, _run


def test_no_output_verdict_runs_on_generated_frames(tmp_path: Path) -> None:
    world = World()
    # Image checks that would refuse every frame they saw: none may run.
    world.nsfw = 0.99
    world.image_scores = {label: 1.0 for label in world.image_scores}
    gate = world.gate()

    def review_image(raw: object, prompt: str = "") -> object:
        raise GenerationBlocked("image")

    gate.review_image = review_image
    drawn: list[int] = []

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        drawn.append(seed)
        return _frame()

    result = _run(tmp_path, age_gate=gate, rewrite=lambda _p: _payload(IDEA), generate=generate)
    assert result.stopped == ""
    assert len(drawn) == 6
    assert [path.name for path in result.kept] == [f"0{index}.webp" for index in range(1, 7)]
    assert "guard" in world.calls
    assert not _IMAGE_CALLS & set(world.calls)


def test_saving_overlaps_the_next_render_and_follows_the_release(tmp_path: Path, monkeypatch) -> None:
    from canvas.pipeline import series

    order: list[str] = []
    real_store = series._store_frame

    def slow_store(folder: Path, index: int, released: object, container: object, metadata=None) -> Path:
        time.sleep(0.05)
        order.append(f"save{index}")
        return real_store(folder, index, released, container, metadata=metadata)  # type: ignore[arg-type]

    monkeypatch.setattr(series, "_store_frame", slow_store)
    gate = World().gate()
    _record_releases(monkeypatch, lambda: order.append(f"check{len([o for o in order if o.startswith('check')])}"))

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        order.append(f"draw{len([o for o in order if o.startswith('draw')])}")
        return _frame()

    times = StageTimes()
    result = series.run_series(
        idea=IDEA, count=6, style="photo", place_mode="same", seed=0, strength=None,
        include_camera=lambda: True, rewrite=lambda _p: _payload(IDEA), age_gate=gate,
        draw_frame=generate, cancelled=lambda: False, folder=tmp_path, timings=times,
    )
    assert len(result.kept) == 6
    assert order.index("draw5") < order.index("check0")
    # Every frame is released in one batch, and only then is anything written.
    for index in range(6):
        assert order.index(f"check{index}") < order.index(f"save{index}")
        assert order.index(f"check{index}") < order.index("save0")
    assert [item for item in order if item.startswith("save")] == [f"save{index}" for index in range(6)]
    stages = times.summary()["stages"]
    assert isinstance(stages, dict)
    assert {"scenario", "prompt_check", "render", "release", "save"} <= set(stages)
    assert "output_check" not in stages


def test_no_file_is_written_before_its_release(tmp_path: Path, monkeypatch) -> None:
    gate = World().gate()
    checked: set[int] = set()
    lock = threading.Lock()

    def note() -> None:
        time.sleep(0.02)
        with lock:
            checked.add(len(checked))

    _record_releases(monkeypatch, note)

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        # Any frame file present now must already have been released.
        on_disk = sorted(tmp_path.glob("0*.webp"))
        with lock:
            assert len(on_disk) <= len(checked)
        return _frame()

    result = _run(tmp_path, age_gate=gate, rewrite=lambda _p: _payload(IDEA), generate=generate)
    assert result.stopped == ""
    assert len(result.kept) == 6
    assert len(checked) == 6


def test_a_refusal_that_does_not_end_the_series_only_drops_the_shot(tmp_path: Path) -> None:
    gate = World().gate()
    real = gate.review_prompt

    def review_prompt(prompt: str) -> object:
        if "three-quarter" in prompt:
            raise SafetyError("shot")
        return real(prompt)

    gate.review_prompt = review_prompt
    result = _run(tmp_path, age_gate=gate, rewrite=lambda _p: _payload(IDEA))
    assert result.stopped == ""
    assert len(result.kept) == 5
    document = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert document["blocked"] == [1]


def test_serial_gate_passes_every_call_through_one_at_a_time() -> None:
    active = 0
    peak = 0
    calls: list[str] = []

    class Gate:
        def review_image(self, raw: object, prompt: str = "") -> str:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            time.sleep(0.01)
            calls.append(prompt)
            active -= 1
            return "ok"

    serial = SerialGate.wrap(Gate())  # type: ignore[arg-type]
    assert SerialGate.wrap(serial) is serial
    threads = [threading.Thread(target=serial.review_image, args=(None, str(n))) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(calls) == ["0", "1", "2", "3"]
    assert peak == 1


def test_gate_weights_are_parked_before_the_frame_is_saved(tmp_path: Path, monkeypatch) -> None:
    """The gate weights leave VRAM once, after the releases and before the frame files are written."""
    order: list[str] = []
    real_store = series._store_frame
    real_release = series.release_gate_weights

    def store(folder: Path, index: int, released: object, container: object, metadata=None) -> Path:
        order.append(f"save{index}")
        return real_store(folder, index, released, container, metadata=metadata)  # type: ignore[arg-type]

    def release(gate: object) -> None:
        order.append("park")
        real_release(gate)

    monkeypatch.setattr(series, "_store_frame", store)
    monkeypatch.setattr(series, "release_gate_weights", release)
    gate = World().gate()
    _record_releases(monkeypatch, lambda: order.append("check"))
    result = _run(tmp_path, age_gate=gate, rewrite=lambda _prompt: _payload(IDEA, 2), count=2)
    assert result.stopped == ""
    assert len(result.kept) == 2
    first = order.index("check")
    # The gate weights leave the card once, after the last release and before any save.
    assert order[first:] == ["check", "check", "park", "save0", "save1"]


def test_prep_warms_only_the_prompt_guard_before_a_frame(tmp_path: Path) -> None:
    gate = World().gate()
    order: list[str] = []
    real = gate.warm_prompt

    def warm_prompt() -> None:
        order.append("warm_prompt")
        real()

    def warm() -> None:
        raise AssertionError("a series must not load the image check models")

    gate.warm_prompt = warm_prompt  # type: ignore[method-assign]
    gate.warm = warm  # type: ignore[method-assign]

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        del line, seed, hero, strength
        order.append("draw")
        return _frame()

    result = _run(tmp_path, age_gate=gate, rewrite=lambda _prompt: _payload(IDEA, 2), generate=generate, count=2)
    assert result.stopped == ""
    assert order[0] == "warm_prompt"
    assert order.count("warm_prompt") == 1
    assert order.count("draw") == 2
    assert gate._loaded == {"guard"}


def test_a_prompt_guard_that_cannot_load_draws_no_frame(tmp_path: Path) -> None:
    gate = World().gate()

    def warm_prompt() -> None:
        raise SafetyUnavailable("classifier_error")

    gate.warm_prompt = warm_prompt  # type: ignore[method-assign]
    drawn: list[int] = []

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        del line, hero, strength
        drawn.append(seed)
        return _frame()

    result = _run(tmp_path, age_gate=gate, rewrite=lambda _prompt: _payload(IDEA, 2), generate=generate, count=2)
    assert drawn == []
    assert result.stopped == "error"
    assert result.code == "check.classifier_error"
    assert result.kept == ()
    assert list(tmp_path.glob("*.webp")) == []


def test_a_series_never_loads_the_picture_check_models(tmp_path: Path, monkeypatch) -> None:
    gate = World().gate()
    loaded: list[str] = []

    def load(name: str):
        def _load() -> None:
            loaded.append(name)
            raise SafetyUnavailable("classifier_error")  # would stop the series if it were ever called

        return _load

    for name in _IMAGE_MODELS:
        setattr(getattr(gate._stack, name), "load", load(name))
    setattr(gate._stack.guard, "load", lambda: loaded.append("guard"))

    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    result = _run(tmp_path / "out", age_gate=gate, rewrite=lambda _prompt: _payload(IDEA, 2), count=2)
    assert result.stopped == ""
    assert len(result.kept) == 2
    assert set(loaded) == {"guard"}
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "stage check_load preload" not in log
    assert "name=nsfw" not in log
    assert "stage frame prompt" in log
    assert "chars=" in log and "sha=" in log
    assert IDEA not in log


def test_a_prompt_guard_load_failure_saves_nothing(tmp_path: Path) -> None:
    gate = World().gate()

    def load() -> None:
        raise SafetyUnavailable("classifier_error")

    setattr(gate._stack.guard, "load", load)
    result = _run(tmp_path, age_gate=gate, rewrite=lambda _prompt: _payload(IDEA, 2), count=2)
    assert result.kept == ()
    assert result.stopped in {"error", "idea"}
    assert list(tmp_path.glob("*.png")) == []
    assert list(tmp_path.glob("*.avif")) == []
    assert list(tmp_path.glob("*.webp")) == []


def test_stage_times_report_names_and_seconds_only() -> None:
    times = StageTimes()
    with times.measure("render"):
        pass
    times.add("save", 0.5)
    summary = times.summary()
    stages = summary["stages"]
    assert isinstance(stages, dict)
    assert set(stages) == {"render", "save"}
    assert stages["save"] == {"count": 1, "total_s": 0.5, "mean_s": 0.5}
    assert times.log_line().startswith("timing wall=")
    try:
        times.add("a prompt with words", 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("free text is not a stage name")
