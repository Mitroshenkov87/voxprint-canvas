"""Cook draws every frame, then each output check runs before that frame is saved.

A picture verdict does not stop the remaining checks. It stops the remaining saves.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from canvas.infra.stage_timer import StageTimes
from canvas.pipeline import series
from canvas.pipeline.series import SerialGate
from canvas.safety.gate import GenerationBlocked, SafetyUnavailable, StylisedBlocked
from tests.fakes import World
from tests.test_photoshoot import IDEA, _frame, _payload, _run


def test_an_output_picture_verdict_stops_before_the_next_render(tmp_path: Path) -> None:
    gate = World().gate()
    real = gate.review_image
    reviewed: list[int] = []
    drawn: list[int] = []

    def review_image(raw: object, prompt: str = "") -> object:
        reviewed.append(len(reviewed))
        if "three-quarter" in prompt:
            raise GenerationBlocked("image")
        return real(raw, prompt)

    gate.review_image = review_image

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        drawn.append(seed)
        return _frame()

    result = _run(tmp_path, age_gate=gate, rewrite=lambda _p: _payload(IDEA), generate=generate)
    assert result.stopped == "scenario"
    assert result.code == "scenario.minor"
    assert len(drawn) == 6
    assert (tmp_path / "01.png").is_file()
    assert not (tmp_path / "02.png").exists()
    assert not (tmp_path / "03.png").exists()
    assert [path.name for path in result.kept] == ["01.png"]


def test_saving_overlaps_the_next_render_and_follows_the_check(tmp_path: Path, monkeypatch) -> None:
    from canvas.pipeline import series

    order: list[str] = []
    real_store = series._store_frame

    def slow_store(folder: Path, index: int, released: object, container: object) -> Path:
        time.sleep(0.05)
        order.append(f"save{index}")
        return real_store(folder, index, released, container)  # type: ignore[arg-type]

    monkeypatch.setattr(series, "_store_frame", slow_store)
    gate = World().gate()
    real = gate.review_image

    def review_image(raw: object, prompt: str = "") -> object:
        order.append(f"check{len([o for o in order if o.startswith('check')])}")
        return real(raw, prompt)

    gate.review_image = review_image

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
    for index in range(6):
        assert order.index(f"check{index}") < order.index(f"save{index}")
        if index:
            assert order.index(f"save{index - 1}") < order.index(f"check{index}")
    stages = times.summary()["stages"]
    assert isinstance(stages, dict)
    assert {"scenario", "prompt_check", "render", "output_check", "save"} <= set(stages)


def test_no_file_is_written_before_its_output_check(tmp_path: Path) -> None:
    gate = World().gate()
    real = gate.review_image
    checked: set[int] = set()
    lock = threading.Lock()

    def review_image(raw: object, prompt: str = "") -> object:
        time.sleep(0.02)
        with lock:
            checked.add(len(checked))
        return real(raw, prompt)

    gate.review_image = review_image

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        # Any frame file present now must already have been checked.
        on_disk = sorted(tmp_path.glob("0*.png"))
        with lock:
            assert len(on_disk) <= len(checked)
        return _frame()

    result = _run(tmp_path, age_gate=gate, rewrite=lambda _p: _payload(IDEA), generate=generate)
    assert result.stopped == ""
    assert len(result.kept) == 6
    assert len(checked) == 6


def test_a_stylised_prompt_verdict_still_only_drops_the_shot(tmp_path: Path) -> None:
    gate = World().gate()
    real = gate.review_prompt

    def review_prompt(prompt: str) -> object:
        if "three-quarter" in prompt:
            raise StylisedBlocked()
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
    """The output check drops its VRAM copy before the frame file is written."""
    order: list[str] = []
    real_store = series._store_frame
    real_release = series.release_gate_weights

    def store(folder: Path, index: int, released: object, container: object) -> Path:
        order.append(f"save{index}")
        return real_store(folder, index, released, container)  # type: ignore[arg-type]

    def release(gate: object) -> None:
        order.append("park")
        real_release(gate)

    monkeypatch.setattr(series, "_store_frame", store)
    monkeypatch.setattr(series, "release_gate_weights", release)
    gate = World().gate()
    real = gate.review_image

    def review_image(raw: object, prompt: str = "") -> object:
        order.append("check")
        return real(raw, prompt)

    gate.review_image = review_image
    result = _run(tmp_path, age_gate=gate, rewrite=lambda _prompt: _payload(IDEA, 2), count=2)
    assert result.stopped == ""
    assert len(result.kept) == 2
    first = order.index("check")
    assert order[first:] == ["check", "park", "save0", "check", "park", "save1"]


def test_prep_loads_the_picture_checks_before_a_frame(tmp_path: Path) -> None:
    gate = World().gate()
    order: list[str] = []
    real = gate.warm

    def warm() -> None:
        order.append("warm")
        real()

    gate.warm = warm  # type: ignore[method-assign]

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        del line, seed, hero, strength
        order.append("draw")
        return _frame()

    result = _run(tmp_path, age_gate=gate, rewrite=lambda _prompt: _payload(IDEA, 2), generate=generate, count=2)
    assert result.stopped == ""
    assert order[0] == "warm"
    assert order.count("warm") == 1
    assert order.count("draw") == 2


def test_a_picture_check_that_cannot_load_draws_no_frame(tmp_path: Path) -> None:
    gate = World().gate()

    def warm() -> None:
        raise SafetyUnavailable("classifier_error")

    gate.warm = warm  # type: ignore[method-assign]
    drawn: list[int] = []

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        del line, hero, strength
        drawn.append(seed)
        return _frame()

    result = _run(tmp_path, age_gate=gate, rewrite=lambda _prompt: _payload(IDEA, 2), generate=generate, count=2)
    assert drawn == []
    assert result.stopped == "error"
    assert result.code == "check.classifier_error"


def test_picture_checks_load_while_a_frame_is_drawn(tmp_path: Path, monkeypatch) -> None:
    gate = World().gate()
    drawing = threading.Event()
    started = threading.Event()
    order: list[str] = []

    def load(name: str) -> None:
        def _load() -> None:
            order.append(f"start:{name}")
            started.set()
            order.append("overlapped" if drawing.wait(2) else "missed")

        return _load

    for name in ("nsfw", "zeroshot", "faces", "ages", "fine"):
        setattr(getattr(gate._stack, name), "load", load(name))

    def generate(line: str, seed: int, hero: object | None, strength: float | None) -> object:
        del line, seed, hero, strength
        assert started.wait(2)
        drawing.set()
        order.append("draw")
        return _frame()

    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    result = _run(tmp_path / "out", age_gate=gate, rewrite=lambda _prompt: _payload(IDEA, 2), generate=generate, count=2)
    assert result.stopped == ""
    assert result.kept
    assert order[0].startswith("start:")
    assert "overlapped" in order
    assert "missed" not in order
    assert order.index("draw") > 0
    assert [name for name in ("nsfw", "zeroshot", "faces", "ages", "fine") if f"start:{name}" in order] == [
        "nsfw", "zeroshot", "faces", "ages", "fine",
    ]
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "stage check_load preload" in log
    assert "stage frame prompt" in log
    assert "chars=" in log and "sha=" in log
    assert IDEA not in log


def test_a_background_check_load_that_fails_saves_nothing(tmp_path: Path) -> None:
    gate = World().gate()

    def load() -> None:
        raise SafetyUnavailable("classifier_error")

    for name in ("nsfw", "zeroshot", "faces", "ages", "fine"):
        setattr(getattr(gate._stack, name), "load", load)
    result = _run(tmp_path, age_gate=gate, rewrite=lambda _prompt: _payload(IDEA, 2), count=2)
    assert result.stopped == "error"
    assert result.code == "check.classifier_error"
    assert result.kept == ()
    assert list(tmp_path.glob("*.png")) == []
    assert list(tmp_path.glob("*.avif")) == []


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
