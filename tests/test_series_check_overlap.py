"""The output check overlaps rendering on a CPU worker, and still gates every frame.

Each test holds one invariant: no frame reaches disk or the result unchecked,
a refused or failing check drops its frame, a worker failure never lets a frame
through, cancellation stops the worker, and no frame skips the check.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from canvas.pipeline import series
from canvas.safety.gate import GenerationBlocked
from canvas.safety.image import RawImage
from tests.fakes import World, rgb
from tests.test_photoshoot import IDEA, _payload

_WAIT_S = 5.0


def _frames(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(path.name for path in folder.iterdir() if path.suffix == ".png" and path.stem.isdigit())


class _Run:
    """A gate and a drawer that record which thread checked which frame."""

    def __init__(self, folder: Path, *, wait_for_check: bool = False) -> None:
        self.folder = folder
        self.gate = World().gate()
        self.real = self.gate.review_image
        self.checks: list[tuple[int, str]] = []
        self.draws: list[int] = []
        self.checked = threading.Condition()
        self.fail: dict[int, BaseException] = {}
        self.wait_for_check = wait_for_check
        self.cancel = threading.Event()
        self.cancel_after: int | None = None
        self.gate.review_image = self.review_image

    def review_image(self, raw: RawImage, prompt: str = "") -> object:
        index = raw.to_bytes()[0]
        # Nothing is on disk while a frame is still being checked.
        assert _frames(self.folder) == []
        try:
            if index in self.fail:
                raise self.fail[index]
            return self.real(raw, prompt)
        finally:
            with self.checked:
                self.checks.append((index, threading.current_thread().name))
                self.checked.notify_all()

    def draw(self, line: str, seed: int, hero: object | None, strength: float | None) -> RawImage:
        del line, hero, strength
        index = len(self.draws)
        if self.wait_for_check and index >= 1:
            # The next render waits until the previous frame was checked, so the overlap is certain.
            with self.checked:
                assert self.checked.wait_for(lambda: len(self.checks) >= index, timeout=_WAIT_S)
        self.draws.append(seed)
        if self.cancel_after is not None and index + 1 >= self.cancel_after:
            self.cancel.set()
        return rgb(2, 2, (index, 7, 7))

    def series(self, count: int = 4) -> series.SeriesResult:
        return series.run_series(
            idea=IDEA, count=count, style="photo", place_mode="same", seed=0, strength=None,
            include_camera=lambda: True, rewrite=lambda _prompt: _payload(IDEA, count),
            age_gate=self.gate, draw_frame=self.draw, cancelled=self.cancel.is_set, folder=self.folder,
        )

    def by_worker(self) -> list[int]:
        return [index for index, name in self.checks if name == "check-overlap"]


@pytest.fixture(autouse=True)
def unpaced(monkeypatch) -> None:
    """The worker takes every frame it is given, so each test knows who checks what."""
    monkeypatch.setattr(series, "_PACE_OVERLAP", False)


@pytest.fixture
def started(monkeypatch) -> list[series._CheckOverlap]:
    """Every overlap worker the series starts."""
    found: list[series._CheckOverlap] = []
    real = series._CheckOverlap.start

    def start(*args: object, **kwargs: object) -> series._CheckOverlap:
        job = real(*args, **kwargs)  # type: ignore[arg-type]
        found.append(job)
        return job

    monkeypatch.setattr(series._CheckOverlap, "start", start)
    return found


def test_the_check_of_one_frame_overlaps_the_next_render(tmp_path: Path, started) -> None:
    run = _Run(tmp_path / "s", wait_for_check=True)
    result = run.series()

    assert result.stopped == ""
    assert _frames(run.folder) == ["01.png", "02.png", "03.png", "04.png"]
    assert run.by_worker()[:3] == [0, 1, 2]
    assert sorted(index for index, _name in run.checks) == [0, 1, 2, 3]
    assert started and not started[0].alive


def test_every_frame_is_checked_exactly_once_on_either_path(tmp_path: Path, monkeypatch) -> None:
    for overlap in (True, False):
        monkeypatch.setattr(series, "_OVERLAP_CHECK", overlap)
        run = _Run(tmp_path / str(overlap))
        result = run.series(count=5)
        assert result.stopped == ""
        assert sorted(index for index, _name in run.checks) == [0, 1, 2, 3, 4]
        assert len(result.kept) == 5
        if not overlap:
            assert run.by_worker() == []


def test_a_written_frame_is_the_checked_output(tmp_path: Path) -> None:
    run = _Run(tmp_path / "s", wait_for_check=True)
    released: list[bytes] = []

    def review_image(raw: RawImage, prompt: str = "") -> object:
        safe = run.review_image(raw, prompt)
        released.append(safe.data)  # type: ignore[attr-defined]
        return safe

    run.gate.review_image = review_image
    result = run.series(count=3)
    assert [path.name for path in result.kept] == ["01.png", "02.png", "03.png"]
    for path in result.kept:
        width, height, data = series._read_png(path)
        assert (width, height) == (2, 2)
        assert data in released


def test_a_check_refused_on_the_worker_drops_the_frame(tmp_path: Path) -> None:
    run = _Run(tmp_path / "s", wait_for_check=True)
    run.fail = {1: GenerationBlocked("image")}
    result = run.series()

    assert 1 in run.by_worker()
    assert result.code == "scenario.minor"
    # The refused frame and every frame after it stay off disk.
    assert _frames(run.folder) == ["01.png"]
    assert [path.name for path in result.kept] == ["01.png"]


def test_a_check_error_on_the_worker_drops_only_that_frame(tmp_path: Path) -> None:
    run = _Run(tmp_path / "s", wait_for_check=True)
    run.fail = {2: RuntimeError("check crashed")}
    result = run.series()

    assert 2 in run.by_worker()
    assert _frames(run.folder) == ["01.png", "02.png", "04.png"]
    assert [path.name for path in result.kept] == ["01.png", "02.png", "04.png"]


def test_a_worker_crash_mid_frame_drops_the_frame_and_later_saves(tmp_path: Path, monkeypatch, started) -> None:
    run = _Run(tmp_path / "s", wait_for_check=True)
    real = series._released_frame

    def released_frame(gate: object, image: object, line: str) -> object:
        if threading.current_thread().name == "check-overlap" and image.to_bytes()[0] == 1:  # type: ignore[attr-defined]
            # Later renders stop waiting for a worker that is about to die.
            run.wait_for_check = False
            with run.checked:
                run.checks.append((1, "check-overlap"))
                run.checked.notify_all()
            raise MemoryError("worker died")
        return real(gate, image, line)  # type: ignore[arg-type]

    monkeypatch.setattr(series, "_released_frame", released_frame)
    result = run.series()

    # Frame 2 could not be checked. It and every later frame stay off disk.
    assert result.stopped == "error"
    assert result.code.startswith("check.")
    assert _frames(run.folder) == ["01.png"]
    assert [path.name for path in result.kept] == ["01.png"]
    assert not started[0].alive
    # The frames the dead worker never reached were checked by the batch pass.
    assert {2, 3} <= {index for index, name in run.checks if name != "check-overlap"}


def test_a_worker_that_fails_before_any_check_leaves_every_frame_to_the_batch(tmp_path: Path, monkeypatch) -> None:
    def broken(_gate: object) -> list[tuple[object, bool, object]]:
        raise RuntimeError("pin failed")

    monkeypatch.setattr(series, "_pin_cpu", broken)
    run = _Run(tmp_path / "s")
    result = run.series()

    assert result.stopped == ""
    assert run.by_worker() == []
    assert sorted(index for index, _name in run.checks) == [0, 1, 2, 3]
    assert _frames(run.folder) == ["01.png", "02.png", "03.png", "04.png"]


def test_cancellation_stops_the_worker(tmp_path: Path, started) -> None:
    run = _Run(tmp_path / "s")
    run.cancel_after = 2
    result = run.series(count=5)

    assert result.stopped == "cancelled"
    assert len(run.draws) == 2
    assert not started[0].alive
    # Frames on disk after a cancel were still checked first.
    checked = {index for index, _name in run.checks}
    for path in result.kept:
        assert int(path.stem) - 1 in checked


def test_the_worker_pins_every_check_to_the_cpu_and_restores_it(tmp_path: Path) -> None:
    run = _Run(tmp_path / "s", wait_for_check=True)
    stack = run.gate._stack
    seen: list[object] = []
    inner = run.review_image

    def review_image(raw: RawImage, prompt: str = "") -> object:
        if threading.current_thread().name == "check-overlap":
            seen.extend(getattr(getattr(stack, name), "_forced", None) for name in series._CHECK_NAMES)
        return inner(raw, prompt)

    run.gate.review_image = review_image
    run.series(count=3)
    assert seen and set(seen) == {"cpu"}
    for name in series._CHECK_NAMES:
        assert not hasattr(getattr(stack, name), "_forced")


def test_a_verdict_belongs_to_one_frame_object() -> None:
    job = series._CheckOverlap(World().gate(), series._CheckPreload(), lambda: False, series.StageTimes())
    item = series._Drawn(0, "line", 1, rgb())
    twin = series._Drawn(0, "line", 1, rgb())
    job._keep(item, None, RuntimeError("refused"))
    assert job.verdict(item) is not None
    assert job.verdict(twin) is None


def test_the_log_names_the_check_device_and_the_series_wall(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    run = _Run(tmp_path / "s", wait_for_check=True)
    run.series(count=3)
    log = (tmp_path / "home" / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "stage frame render " in log
    assert "stage frame check device=cpu " in log
    line = next(row for row in log.splitlines() if "stage series_check done" in row)
    assert "frames=3" in line and "overlap=" in line and "wall_s=" in line


def test_with_pacing_every_frame_is_still_checked_once(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(series, "_PACE_OVERLAP", True)
    for count in (1, 3, 5):
        run = _Run(tmp_path / str(count))
        result = run.series(count=count)
        assert result.stopped == ""
        assert sorted(index for index, _name in run.checks) == list(range(count))
        assert len(result.kept) == count


def test_pacing_passes_a_frame_on_when_the_gpu_is_nearly_done(monkeypatch) -> None:
    monkeypatch.setattr(series, "_PACE_OVERLAP", True)
    job = series._CheckOverlap(World().gate(), series._CheckPreload(), lambda: False, series.StageTimes())
    # Nothing measured yet: any GPU work ahead is worth one try.
    job.gpu_left(1.0)
    assert job.worth_starting()
    job.gpu_left(0.0)
    assert not job.worth_starting()
    # One CPU check took 2 s. One second of GPU work ahead does not cover the next.
    job.seconds, job._checked = 2.0, 1
    job.gpu_left(1.0)
    assert not job.worth_starting()
    job.gpu_left(10.0)
    assert job.worth_starting()
