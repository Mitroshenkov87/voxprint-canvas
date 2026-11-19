"""Frames are released in one batch after decode, with no classifier on generated pixels.

Owner rule 2026-11-16: the request is screened (prompt checks in prep); each
decoded frame then leaves through ``release_generated`` only. The release
step keeps its stop logic: a release that raises drops that frame, and one
that cannot finish keeps every later frame off disk.

Also: each shot's own place, pose and light reach that shot's draw call.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canvas.pipeline import series
from canvas.pipeline.photoshoot import planned_shots
from canvas.pipeline.series import run_series
from canvas.safety.gate import SafetyUnavailable
from canvas.safety.image import RawImage
from tests.fakes import World, rgb
from tests.test_photoshoot import IDEA, PERSON, _payload

_IMAGE_CALLS = {"nsfw", "image-score", "faces", "ages", "fine"}


def _frames(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(path.name for path in folder.iterdir() if path.suffix == ".webp" and path.stem.isdigit())


class _Watch:
    """Records the order of draws, releases, gate weight releases, statuses, and files on disk."""

    def __init__(self, folder: Path, monkeypatch: pytest.MonkeyPatch, fail: dict[int, Exception] | None = None) -> None:
        self.folder = folder
        self.fail = fail or {}
        self.events: list[tuple[str, int]] = []
        self.checks = 0
        self.world = World()
        # Image checks that would refuse every frame they saw: none may run.
        self.world.nsfw = 0.99
        self.gate = self.world.gate()
        real = series.release_generated
        real_release = getattr(self.gate, "release_weights", None)

        def release_generated(raw: RawImage) -> object:
            index = self.checks
            self.checks += 1
            # Nothing has been written, shown as kept, or returned while any frame is still unreleased.
            assert _frames(self.folder) == []
            assert not [event for event in self.events if event[0] == "kept"]
            self.events.append(("check", index))
            if index in self.fail:
                raw.wipe()
                raise self.fail[index]
            return real(raw)

        def release_weights() -> None:
            self.events.append(("release", -1))
            if callable(real_release):
                real_release()

        monkeypatch.setattr(series, "release_generated", release_generated)
        self.gate.release_weights = release_weights

    def status(self, index: int, status: str) -> None:
        self.events.append((status, index))

    def draw(self, line: str, seed: int, hero: object | None, strength: float | None) -> RawImage:
        del line, hero, strength
        self.events.append(("draw", seed))
        return rgb(4, 4, (seed % 250, 10, 10))


def _series(tmp_path: Path, watch: _Watch, count: int = 4):
    return run_series(
        idea=IDEA, count=count, style="photo", place_mode="same", seed=0, strength=None,
        include_camera=lambda: True, rewrite=lambda _prompt: _payload(IDEA, count),
        age_gate=watch.gate, draw_frame=watch.draw, cancelled=lambda: False, folder=watch.folder,
        on_status=watch.status,
    )


def test_every_frame_is_released_before_any_frame_is_saved_or_shown(tmp_path: Path, monkeypatch) -> None:
    watch = _Watch(tmp_path / "series", monkeypatch)
    result = _series(tmp_path, watch)

    assert result.stopped == ""
    assert watch.checks == 4
    assert _frames(watch.folder) == ["01.webp", "02.webp", "03.webp", "04.webp"]
    # The prompt guard ran; no image classifier ran on any frame.
    assert "guard" in watch.world.calls
    assert not _IMAGE_CALLS & set(watch.world.calls)
    kinds = [kind for kind, _index in watch.events]
    last_draw = max(index for index, kind in enumerate(kinds) if kind == "draw")
    checks = [index for index, kind in enumerate(kinds) if kind == "check"]
    kept = [index for index, kind in enumerate(kinds) if kind == "kept"]
    assert last_draw < checks[0]
    # The gate weights leave the card once, after the last release and before the first kept frame.
    between = kinds[checks[-1]:kept[0]]
    assert between.count("release") == 1
    assert "release" not in kinds[checks[0]:checks[-1]]
    assert [index for kind, index in watch.events if kind == "kept"] == [0, 1, 2, 3]
    assert [path.name for path in result.kept] == _frames(watch.folder)


def test_a_raising_release_drops_only_that_frame(tmp_path: Path, monkeypatch) -> None:
    watch = _Watch(tmp_path / "series", monkeypatch, fail={1: RuntimeError("release crashed")})
    result = _series(tmp_path, watch)

    assert watch.checks == 4
    assert _frames(watch.folder) == ["01.webp", "03.webp", "04.webp"]
    assert [path.name for path in result.kept] == ["01.webp", "03.webp", "04.webp"]
    assert ("blocked", 1) in watch.events
    assert ("kept", 1) not in watch.events
    document = json.loads((watch.folder / "series.json").read_text(encoding="utf-8"))
    assert 1 in document["blocked"]
    assert len(document["lines"]) == 3


def test_a_release_that_cannot_finish_keeps_every_later_frame_off_disk(tmp_path: Path, monkeypatch) -> None:
    watch = _Watch(tmp_path / "series", monkeypatch, fail={2: SafetyUnavailable("classifier_error")})
    result = _series(tmp_path, watch)

    # Later frames still go through the release step, and none of them are written.
    assert watch.checks == 4
    assert result.stopped == "error"
    assert result.code.startswith("check.")
    assert _frames(watch.folder) == ["01.webp", "02.webp"]
    assert [path.name for path in result.kept] == ["01.webp", "02.webp"]
    for index in (2, 3):
        assert ("kept", index) not in watch.events
        assert ("blocked", index) in watch.events


# Places and poses


def _lines(count: int) -> str:
    places = ("loft with tall windows", "rooftop at dusk", "quiet courtyard", "wooden terrace")
    poses = ("seated on a stool", "walking toward the lens", "leaning on a rail", "kneeling")
    return "\n".join(
        f"{index + 1}. view {index}, warm light | {poses[index]} | 50 mm, eye level | {places[index]}"
        for index in range(count)
    ), places, poses


def test_different_places_draw_every_shot_from_its_own_line(tmp_path: Path) -> None:
    reply, places, poses = _lines(4)
    seen: list[tuple[str, object | None, float | None]] = []

    def draw(line: str, seed: int, hero: object | None, strength: float | None) -> RawImage:
        seen.append((line, hero, strength))
        return rgb(4, 4, (seed % 250, 0, 0))

    result = run_series(
        idea=PERSON, count=4, style="photo", place_mode="different", seed=1, strength=None,
        include_camera=lambda: True, rewrite=lambda _prompt: reply, age_gate=World().gate(),
        draw_frame=draw, cancelled=lambda: False, folder=tmp_path / "series",
    )
    assert result.stopped == ""
    assert len(seen) == 4
    # No shot is redrawn from the first shot's pixels, so its own place and pose lead it.
    assert all(hero is None and strength is None for _line, hero, strength in seen)
    for index, (line, _hero, _strength) in enumerate(seen):
        assert places[index] in line
        assert poses[index] in line
        assert all(other not in line for other in places if other != places[index])
    assert len({line for line, _hero, _strength in seen}) == 4


def test_same_place_person_series_still_follows_its_first_shot(tmp_path: Path) -> None:
    seen: list[tuple[object | None, float | None]] = []

    def draw(line: str, seed: int, hero: object | None, strength: float | None) -> RawImage:
        del line
        seen.append((hero, strength))
        return rgb(4, 4, (seed % 250, 0, 0))

    run_series(
        idea=PERSON, count=3, style="photo", place_mode="same", seed=1, strength=None,
        include_camera=lambda: True, rewrite=lambda _prompt: _payload(PERSON, 3), age_gate=World().gate(),
        draw_frame=draw, cancelled=lambda: False, folder=tmp_path / "series",
    )
    assert seen[0][0] is None
    assert all(hero is not None for hero, _strength in seen[1:])


def test_planned_shots_change_place_light_and_pose_every_shot() -> None:
    shots = planned_shots(PERSON, 12, "different", place="", taken=())
    assert len(shots) == 12
    assert len({shot.place for shot in shots}) == 12
    assert len({shot.pose for shot in shots}) == 8
    lights = [shot.scene.split(", ", 1)[1] for shot in shots]
    assert all(left != right for left, right in zip(lights, lights[1:]))
    assert len(set(lights)) == 8
    long_run = planned_shots(PERSON, 99, "different", place="", taken=())
    assert len(long_run) == 99
    assert len({shot.scene for shot in long_run}) == 99
