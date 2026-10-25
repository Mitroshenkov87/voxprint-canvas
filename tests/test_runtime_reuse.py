"""One pipeline load for a run, and the paths that release it."""
from __future__ import annotations

import threading
import types
from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from canvas.models import runtime
from canvas.models.modes import Tier
from canvas.ui.shell_layout import open_layout
from canvas.ui.shell_screen import ShellWindow
from canvas.models.plan import PIPELINE_IMG2IMG, PIPELINE_TEXT, CallPlan
from canvas.models.snapshots import SnapshotError
from canvas.pipeline.enhance import QwenRewriter
from canvas.pipeline.series import REFERENCE_MIN, REFERENCE_STRENGTH, run_series
from canvas.safety.image import RawImage
from canvas.safety.inputs import ImageRole, PendingInput
from tests.fakes import World
from tests.test_photoshoot import IDEA, PERSON, _payload

_GB32 = 32 * 1024**3
_GB64 = 64 * 1024**3


class _Pipe:
    def __init__(self, name: str, modules: dict[str, object]) -> None:
        self.name = name
        self.modules = modules

    def __call__(self, **kwargs):
        image = Image.new("RGB", (kwargs["width"], kwargs["height"]), (40, 90, 120))
        return types.SimpleNamespace(images=[image])

    def to(self, device: str):
        self.device = device
        return self


def _install(monkeypatch, *, fail_on_call: int | None = None) -> tuple[list[int], list[Tier]]:
    loads = [0]
    places: list[Tier] = []
    calls = [0]

    class _CountingPipe(_Pipe):
        def __call__(self, **kwargs):
            calls[0] += 1
            if fail_on_call is not None and calls[0] == fail_on_call:
                raise RuntimeError("diffusion failed")
            return super().__call__(**kwargs)

    def text_load(*_args, **_kwargs):
        loads[0] += 1
        return _CountingPipe("text", {"weights": object()})

    def image_load(*_args, **_kwargs):
        loads[0] += 1
        return _CountingPipe("img2img", {"weights": object()})

    diffusers = types.SimpleNamespace(
        ZImagePipeline=types.SimpleNamespace(
            from_pretrained=text_load,
            from_pipe=lambda pipe: _CountingPipe("text", pipe.modules),
        ),
        ZImageImg2ImgPipeline=types.SimpleNamespace(
            from_pretrained=image_load,
            from_pipe=lambda pipe: _CountingPipe("img2img", pipe.modules),
        ),
    )

    class _Generator:
        def __init__(self, device=None):
            self.device = device

        def manual_seed(self, seed):
            self.seed = seed
            return self

    cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    torch = types.SimpleNamespace(bfloat16="bf16", Generator=_Generator, cuda=cuda)
    monkeypatch.setattr(runtime, "_import_torch", lambda: torch)
    monkeypatch.setattr(runtime, "_import_diffusers", lambda: diffusers)
    monkeypatch.setattr(runtime, "_place", lambda _pipe, tier: places.append(tier))
    monkeypatch.setattr(runtime, "reset_peak_allocated", lambda: None)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: _GB32)
    return loads, places


def _plan(pipeline: str) -> CallPlan:
    mode = "text" if pipeline == PIPELINE_TEXT else "restyle"
    return CallPlan("diffusers", pipeline, ("stub",), "bf16", 8, None, "none", "16gb", mode)


def _render(gate, folder: Path, line: str, hero: RawImage | None, tier: Tier = Tier.GB16) -> RawImage:
    if hero is None:
        screened = gate.release_request(line, "text", ())
        pipeline = PIPELINE_TEXT
        strength = None
    else:
        raw = RawImage(hero.width, hero.height, hero.to_bytes())
        screened = gate.release_request(line, "restyle", (PendingInput(ImageRole.INIT, raw),))
        pipeline = PIPELINE_IMG2IMG
        strength = None
    return runtime.render(
        _plan(pipeline), folder, screened, 32, 32, tier, seed=1, strength=strength,
    )


def test_series_strength_stays_at_least_the_reference_floor() -> None:
    assert REFERENCE_STRENGTH >= REFERENCE_MIN
    assert runtime.applied_strength(None) == 0.6
    assert runtime.applied_strength(REFERENCE_STRENGTH) == REFERENCE_STRENGTH
    assert runtime.applied_strength(REFERENCE_MIN) == REFERENCE_MIN
    with pytest.raises(ValueError):
        runtime.applied_strength(0.45)


def test_a_ten_frame_series_loads_the_pipeline_once(tmp_path: Path, monkeypatch) -> None:
    loads, places = _install(monkeypatch)
    gate = World().gate()

    def draw(line: str, seed: int, hero, strength: float):
        del seed, strength
        found = hero if isinstance(hero, RawImage) else None
        return _render(gate, tmp_path / "weights", line, found)

    result = run_series(
        idea=IDEA,
        count=10,
        style="photo",
        place_mode="same",
        seed=3,
        strength=None,
        include_camera=lambda: True,
        rewrite=lambda _prompt: _payload(IDEA, 10),
        age_gate=gate,
        draw_frame=draw,
        cancelled=lambda: False,
        folder=tmp_path / "series",
    )
    assert result.stopped == ""
    assert len(result.kept) == 10
    assert loads[0] == 1
    assert places == [Tier.GB16]


def test_a_person_series_loads_and_casts_once(tmp_path: Path, monkeypatch) -> None:
    real_place = runtime._place
    loads, _places = _install(monkeypatch)
    casts: list[str] = []

    def cast(pipe: object, torch: object, apply: object) -> tuple[str, ...]:
        del pipe, torch, apply
        casts.append("fp8")
        return ("transformer",)

    monkeypatch.setattr(runtime, "_place", real_place)
    monkeypatch.setattr(runtime, "install_fp8_storage", cast)
    monkeypatch.setattr(runtime, "_free_vram_bytes", lambda: 20 * 1024**3)
    monkeypatch.setattr(runtime, "resident_fits", lambda pipe, free: True)
    gate = World().gate()
    heroes: list[object] = []

    def draw(line: str, seed: int, hero, strength: float):
        del seed, strength
        heroes.append(hero)
        found = hero if isinstance(hero, RawImage) else None
        return _render(gate, tmp_path / "weights", line, found)

    result = run_series(
        idea=PERSON,
        count=6,
        style="photo",
        place_mode="same",
        seed=3,
        strength=None,
        include_camera=lambda: True,
        rewrite=lambda _prompt: _payload(PERSON),
        age_gate=gate,
        draw_frame=draw,
        cancelled=lambda: False,
        folder=tmp_path / "series",
    )
    assert result.stopped == ""
    assert len(result.kept) == 6
    assert heroes[0] is None
    assert all(isinstance(hero, RawImage) for hero in heroes[1:])
    assert loads[0] == 1
    assert casts == ["fp8"]


def test_a_run_releases_the_pipeline_below_64gb(tmp_path: Path, monkeypatch) -> None:
    loads, _places = _install(monkeypatch)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: _GB32)
    gate = World().gate()
    folder = tmp_path / "weights"
    with runtime.pipeline_scope():
        _render(gate, folder, IDEA, None)
    assert loads[0] == 1
    _render(gate, folder, IDEA, None)
    assert loads[0] == 2


def test_64gb_keeps_the_pipeline_between_runs(tmp_path: Path, monkeypatch) -> None:
    loads, _places = _install(monkeypatch)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: _GB64)
    gate = World().gate()
    folder = tmp_path / "weights"
    with runtime.pipeline_scope():
        _render(gate, folder, IDEA, None)
    _render(gate, folder, IDEA, None)
    assert loads[0] == 1


def test_release_pipeline_drops_a_parked_pipeline(tmp_path: Path, monkeypatch) -> None:
    loads, _places = _install(monkeypatch)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: _GB64)
    gate = World().gate()
    folder = tmp_path / "weights"
    with runtime.pipeline_scope():
        _render(gate, folder, IDEA, None)
    _render(gate, folder, IDEA, None)
    assert loads[0] == 1
    runtime.release_pipeline()
    _render(gate, folder, IDEA, None)
    assert loads[0] == 2


def test_a_tier_or_model_change_loads_again(tmp_path: Path, monkeypatch) -> None:
    loads, places = _install(monkeypatch)
    gate = World().gate()
    folder = tmp_path / "weights"
    other = tmp_path / "other"
    with runtime.pipeline_scope():
        _render(gate, folder, IDEA, None, Tier.GB16)
        _render(gate, folder, IDEA, None, Tier.GB24)
        _render(gate, other, IDEA, None, Tier.GB24)
    assert loads[0] == 3
    assert places == [Tier.GB16, Tier.GB24, Tier.GB24]


def test_an_error_releases_the_pipeline(tmp_path: Path, monkeypatch) -> None:
    loads, _places = _install(monkeypatch, fail_on_call=2)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: _GB64)
    gate = World().gate()
    folder = tmp_path / "weights"
    with runtime.pipeline_scope():
        _render(gate, folder, IDEA, None)
        with pytest.raises(RuntimeError):
            _render(gate, folder, IDEA, None)
        _render(gate, folder, IDEA, None)
    assert loads[0] == 2


def test_the_writer_load_releases_the_pipeline(tmp_path: Path, monkeypatch) -> None:
    loads, _places = _install(monkeypatch)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: _GB64)
    gate = World().gate()
    folder = tmp_path / "weights"
    with runtime.pipeline_scope():
        _render(gate, folder, IDEA, None)
    assert loads[0] == 1
    rewriter = QwenRewriter(tmp_path / "encoder", tmp_path / "tokenizer")
    with pytest.raises((FileNotFoundError, ImportError, SnapshotError)):
        rewriter._load()
    _render(gate, folder, IDEA, None)
    assert loads[0] == 2


def test_the_window_does_not_keep_the_pipeline_between_runs(tmp_path: Path, monkeypatch) -> None:
    QApplication.instance() or QApplication([])
    loads, _places = _install(monkeypatch)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: _GB32)
    gate = World().gate()
    shell = open_layout(tmp_path / "state", gate=gate)
    window = ShellWindow(shell, tier=Tier.GB16, enlarge=object())
    folder = tmp_path / "weights"
    _render(gate, folder, IDEA, None)
    assert runtime._slot is None
    _render(gate, folder, IDEA, None)
    assert loads[0] == 2
    assert runtime._slot is None
    window.close()


def test_a_release_from_another_thread_waits_for_the_picture(tmp_path: Path, monkeypatch) -> None:
    loads, _places = _install(monkeypatch)
    gate = World().gate()
    folder = tmp_path / "weights"
    started = threading.Event()
    go_on = threading.Event()
    seen: list[str] = []
    real_timed = runtime._timed

    def slow_timed(pipe, **kwargs):
        started.set()
        go_on.wait(5)
        seen.append("drawn")
        return real_timed(pipe, **kwargs)

    monkeypatch.setattr(runtime, "_timed", slow_timed)

    def other() -> None:
        started.wait(5)
        runtime.release_pipeline()
        seen.append("released")

    with runtime.pipeline_scope():
        helper = threading.Thread(target=other)
        helper.start()
        worker = threading.Thread(target=lambda: _render(gate, folder, IDEA, None))
        worker.start()
        started.wait(5)
        go_on.set()
        worker.join(5)
        helper.join(5)
    assert seen == ["drawn", "released"]
    assert loads[0] == 1
    assert runtime._slot is None
