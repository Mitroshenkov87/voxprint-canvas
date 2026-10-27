"""A series runs prep, render, then decode. Each model loads once per phase.

The series doubles record the order of every hook. The runtime doubles stand
in for torch and diffusers, so no weights are loaded and no GPU is used.
"""
from __future__ import annotations

import threading
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from PIL import Image

from canvas.app.shell_run import ShellRunner
from canvas.infra import phase_vram
from canvas.models import runtime
from canvas.models.modes import Tier
from canvas.models.plan import PIPELINE_TEXT, CallPlan
from canvas.pipeline import series
from canvas.pipeline.engine import run_sampler
from canvas.pipeline.series import REFERENCE_STRENGTH, run_series
from canvas.safety.gate import issue_input
from canvas.safety.image import RawImage
from tests.fakes import World, rgb
from tests.test_photoshoot import IDEA, PERSON, _payload

_GB = 1024**3


class _Cuda:
    """``torch.cuda`` counters for the phase checks."""

    def __init__(self, allocated: int = 0) -> None:
        self.allocated = allocated
        self.peak = 0
        self.resets = 0

    def is_available(self) -> bool:
        return True

    def memory_allocated(self) -> int:
        return self.allocated

    def memory_reserved(self) -> int:
        return self.allocated

    def max_memory_allocated(self) -> int:
        return self.peak

    def reset_peak_memory_stats(self) -> None:
        self.resets += 1
        self.peak = 0

    def synchronize(self) -> None:
        return None

    def empty_cache(self) -> None:
        return None


def _log(home: Path) -> str:
    path = home / "logs" / "voxprint.log"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _recorded(monkeypatch, events: list[object]) -> None:
    """Record the phase boundaries the series opens."""
    real_released = series.require_released
    real_render = series.render_phase
    real_decode = series.decode_phase

    def released(name: str) -> int:
        events.append(f"released:{name}")
        return real_released(name)

    @contextmanager
    def render_phase() -> Iterator[None]:
        events.append("render:start")
        with real_render():
            yield
        events.append("render:end")

    @contextmanager
    def decode_phase() -> Iterator[None]:
        events.append("decode:start")
        with real_decode():
            yield
        events.append("decode:end")

    monkeypatch.setattr(series, "require_released", released)
    monkeypatch.setattr(series, "render_phase", render_phase)
    monkeypatch.setattr(series, "decode_phase", decode_phase)


def _writer(events: list[object], payload: str):
    def rewrite(prompt: str) -> str:
        del prompt
        events.append("write")
        return payload

    rewrite.release = lambda: events.append("writer:release")  # type: ignore[attr-defined]
    return rewrite


def _gate(events: list[object]):
    gate = World().gate()
    review_prompt = gate.review_prompt
    review_image = gate.review_image

    def prompt_check(prompt: str) -> object:
        events.append("prompt_check")
        return review_prompt(prompt)

    def output_check(raw: object, prompt: str = "") -> object:
        events.append("output_check")
        return review_image(raw, prompt)

    gate.review_prompt = prompt_check
    gate.review_image = output_check
    return gate


def _draw(events: list[object], encoded: list[list[str]], *, keeps: bool = True):
    def draw(line: str, seed: int, hero: object | None, strength: float | None) -> RawImage:
        del line, strength
        events.append(("draw", seed, hero is not None))
        return rgb(4, 4, (seed % 200, 40, 60))

    def prepare(lines: list[str]) -> None:
        events.append("encode")
        encoded.append(list(lines))

    def sample(line: str, seed: int, hero: object | None, strength: float | None) -> bool:
        del line, strength
        events.append(("sample", seed, hero is not None))
        return keeps

    draw.prepare = prepare  # type: ignore[attr-defined]
    draw.sample = sample  # type: ignore[attr-defined]
    return draw


def _series(tmp_path: Path, idea: str, count: int, events: list[object], draw) -> series.SeriesResult:
    return run_series(
        idea=idea, count=count, style="photo", place_mode="same", seed=10, strength=None,
        include_camera=lambda: True, rewrite=_writer(events, _payload(idea, count)),
        age_gate=_gate(events), draw_frame=draw, cancelled=lambda: False, folder=tmp_path / "series",
    )


def _kind(event: object) -> object:
    return event[0] if isinstance(event, tuple) else event


def _first(events: list[object], name: str) -> int:
    return next(index for index, event in enumerate(events) if _kind(event) == name)


def _last(events: list[object], name: str) -> int:
    return max(index for index, event in enumerate(events) if _kind(event) == name)


def test_prep_render_decode_then_check_in_that_order(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    events: list[object] = []
    encoded: list[list[str]] = []
    _recorded(monkeypatch, events)
    result = _series(tmp_path, IDEA, 3, events, _draw(events, encoded))

    assert result.stopped == ""
    assert len(result.kept) == 3
    # Every line is encoded in one pass, before the picture pipeline.
    assert len(encoded) == 1 and len(encoded[0]) == 3
    assert _last(events, "write") < _first(events, "writer:release")
    assert _last(events, "prompt_check") < _first(events, "encode")
    assert _first(events, "writer:release") < _first(events, "encode")
    assert _first(events, "encode") < events.index("released:prep") < events.index("render:start")
    samples = [event for event in events if _kind(event) == "sample"]
    draws = [event for event in events if _kind(event) == "draw"]
    assert [seed for _kind_, seed, _hero in samples] == [10, 11, 12]
    assert [seed for _kind_, seed, _hero in draws] == [10, 11, 12]
    assert events.index("render:start") < _first(events, "sample")
    assert _last(events, "sample") < events.index("render:end") < events.index("decode:start")
    assert events.index("decode:start") < _first(events, "draw") <= _last(events, "draw") < events.index("decode:end")
    # Each frame is checked after decode, one by one.
    assert events.count("output_check") == 3
    assert events.index("decode:end") < _first(events, "output_check")
    log = _log(tmp_path / "home")
    for name in ("prep", "render", "decode"):
        assert f"stage phase {name} " in log
    assert "stage phase prep " in log and "over=0" in log
    assert log.count("stage frame render ") == 3
    assert log.count("stage frame decode ") == 3


def test_a_person_series_draws_its_first_frame_whole_and_samples_the_rest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    events: list[object] = []
    encoded: list[list[str]] = []
    _recorded(monkeypatch, events)
    result = _series(tmp_path, PERSON, 3, events, _draw(events, encoded))

    assert result.stopped == ""
    assert len(result.kept) == 3
    render = events[events.index("render:start"):events.index("render:end")]
    decode = events[events.index("decode:start"):events.index("decode:end")]
    # The later frames start from the first frame's pixels, so that one is drawn whole first.
    assert [event for event in render if _kind(event) == "draw"] == [("draw", 10, False)]
    assert [event for event in render if _kind(event) == "sample"] == [("sample", 11, True), ("sample", 12, True)]
    assert [event for event in decode if _kind(event) == "draw"] == [("draw", 11, True), ("draw", 12, True)]
    assert series.style_reference(REFERENCE_STRENGTH) == REFERENCE_STRENGTH


def test_a_sample_hook_that_keeps_nothing_draws_the_frame_in_render(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    events: list[object] = []
    _recorded(monkeypatch, events)
    result = _series(tmp_path, IDEA, 2, events, _draw(events, [], keeps=False))

    assert result.stopped == ""
    assert len(result.kept) == 2
    render = events[events.index("render:start"):events.index("render:end")]
    assert [event for event in render if _kind(event) == "draw"] == [("draw", 10, False), ("draw", 11, False)]
    assert "decode:start" not in events


def test_vram_left_after_prep_stops_the_series_before_the_generator(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    events: list[object] = []
    _recorded(monkeypatch, events)
    cuda = _Cuda(allocated=phase_vram.RESIDUAL_LIMIT_BYTES + 1)
    monkeypatch.setattr(phase_vram, "_cuda", lambda: cuda)
    result = _series(tmp_path, IDEA, 2, events, _draw(events, []))

    assert result.stopped == "error"
    assert result.code == "vram.residual"
    assert result.kept == ()
    assert "render:start" not in events
    assert not [event for event in events if _kind(event) in {"sample", "draw"}]
    log = _log(tmp_path / "home")
    assert "stage phase prep " in log and "over=1" in log
    assert "prep.residual" in log


def test_half_a_gigabyte_after_prep_is_still_allowed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    cuda = _Cuda(allocated=phase_vram.RESIDUAL_LIMIT_BYTES)
    monkeypatch.setattr(phase_vram, "_cuda", lambda: cuda)
    assert phase_vram.require_released("prep") == phase_vram.RESIDUAL_LIMIT_BYTES
    cuda.allocated += 1
    with pytest.raises(phase_vram.ResidualVram):
        phase_vram.require_released("prep")


def test_a_phase_keeps_the_peak_across_stage_resets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    cuda = _Cuda()
    monkeypatch.setattr(phase_vram, "_cuda", lambda: cuda)
    opened = phase_vram.start_phase("render")
    assert cuda.resets == 1
    cuda.peak = 7 * _GB
    # A stage inside the phase resets the counter for its own line.
    phase_vram.fold_peak()
    cuda.reset_peak_memory_stats()
    cuda.peak = 2 * _GB
    assert opened.finish() == 7 * _GB
    assert opened.finish() == 7 * _GB
    log = _log(tmp_path / "home")
    assert log.count("stage phase render ") == 1
    assert f"peak={7 * _GB} peak_mib={7 * 1024}" in log


# Runtime: one pipeline load, one denoiser unpark, one VAE session.


class _Device:
    def __init__(self, kind: str) -> None:
        self.type = kind


class _Latent:
    """A latent the doubles hand back with ``output_type="latent"``."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.device = _Device("cuda")

    def detach(self) -> _Latent:
        return self

    def to(self, device: str) -> _Latent:
        self.device = _Device(device)
        return self


class _Module:
    def parameters(self):
        return iter(())

    def to(self, *_args: object, **_kwargs: object) -> _Module:
        return self


def _torch() -> types.SimpleNamespace:
    class _Generator:
        def __init__(self, device: object = None) -> None:
            self.device = device

        def manual_seed(self, seed: int) -> _Generator:
            self.seed = seed
            return self

    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: None,
        synchronize=lambda: None,
        memory_allocated=lambda: 0,
        max_memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
        max_memory_reserved=lambda: 0,
        mem_get_info=lambda: (15 * _GB, 16 * _GB),
        reset_peak_memory_stats=lambda: None,
    )
    return types.SimpleNamespace(bfloat16="bf16", float32="fp32", Generator=_Generator, cuda=cuda)


def _install(monkeypatch, record: dict[str, list[object]]) -> tuple[_Module, _Module]:
    transformer, vae = _Module(), _Module()
    on_cuda: set[int] = set()

    class TextPipe:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> TextPipe:
            record["loads"].append("pipeline")
            return cls()

        def __init__(self) -> None:
            self.text_encoder = None
            self.transformer = transformer
            self.vae = vae
            self.components = {"transformer": transformer, "vae": vae}

        def __call__(self, **kwargs: object) -> types.SimpleNamespace:
            assert kwargs["output_type"] == "latent"
            assert id(transformer) in on_cuda
            generator = kwargs["generator"]
            record["samples"].append(getattr(generator, "seed", None))
            return types.SimpleNamespace(images=[_Latent(int(getattr(generator, "seed", 0)))])

    def activate(name: str, device: str, *, blocking: bool = False) -> bool:
        del device, blocking
        record["activate"].append(name)
        if name == "transformer":
            on_cuda.add(id(transformer))
        return True

    def deactivate(name: str) -> None:
        record["deactivate"].append(name)
        if name == "transformer":
            on_cuda.discard(id(transformer))

    def module_on_cuda(module: object) -> bool:
        return id(module) in on_cuda

    def prepare_vae(pipe: object, _torch: object) -> object:
        record["vae_sessions"].append("open")
        return getattr(pipe, "vae")

    def decode(pipe: object, latents: _Latent, torch: object) -> _Latent:
        with runtime._vae_session(pipe, torch, "vae_decode", restore=False):
            record["decodes"].append(latents.seed)
        return latents

    monkeypatch.setattr(runtime, "_import_torch", _torch)
    monkeypatch.setattr(
        runtime, "_import_diffusers", lambda: types.SimpleNamespace(ZImagePipeline=TextPipe),
    )
    monkeypatch.setattr(runtime, "_place", lambda _pipe, _tier: None)
    monkeypatch.setattr(runtime, "reset_peak_allocated", lambda: None)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: 32 * _GB)
    monkeypatch.setattr(runtime.hot, "activate", activate)
    monkeypatch.setattr(runtime.hot, "deactivate", deactivate)
    monkeypatch.setattr(runtime, "_module_on_cuda", module_on_cuda)
    monkeypatch.setattr(runtime, "_prepare_vae", prepare_vae)
    monkeypatch.setattr(runtime, "_assert_vae_cuda_bf16", lambda _vae, _torch: None)
    monkeypatch.setattr(runtime, "_arm_vae_tiles", lambda _vae: None)
    monkeypatch.setattr(runtime, "_decode_latents", decode)
    monkeypatch.setattr(
        runtime, "_latents_to_pil", lambda _pipe, latents, _torch: Image.new("RGB", (8, 8), (latents.seed, 0, 0)),
    )
    runtime._prompt_embeds = {f"frame {index}": f"embed {index}" for index in range(3)}
    return transformer, vae


def _plan() -> CallPlan:
    return CallPlan("diffusers", PIPELINE_TEXT, ("stub",), "bf16", 8, None, "none", "16gb", "text")


def test_the_generator_loads_once_and_every_frame_decodes_in_one_vae_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    record: dict[str, list[object]] = {
        key: [] for key in ("loads", "samples", "activate", "deactivate", "vae_sessions", "decodes")
    }
    _install(monkeypatch, record)
    folder = tmp_path / "weights"
    images: list[RawImage] = []
    try:
        with runtime.pipeline_scope():
            with runtime.render_phase():
                for index in range(3):
                    screened = issue_input(f"frame {index}", "text", ())
                    try:
                        runtime.sample_latent(_plan(), folder, screened, 8, 8, Tier.GB16, seed=index + 1)
                    finally:
                        screened.wipe()
                # The denoiser stays on the card between frames.
                assert record["activate"].count("transformer") == 1
                assert "transformer" not in record["deactivate"]
            assert record["deactivate"].count("transformer") == 1
            assert runtime.pending_latents() == 3
            assert record["decodes"] == []
            with runtime.decode_phase():
                for index in range(3):
                    screened = issue_input(f"frame {index}", "text", ())
                    try:
                        images.append(
                            runtime.render(_plan(), folder, screened, 8, 8, Tier.GB16, seed=index + 1),
                        )
                    finally:
                        screened.wipe()
    finally:
        runtime.clear_series_embeds()
        runtime.reset_pipeline_cache()
    assert record["loads"] == ["pipeline"]
    assert record["samples"] == [1, 2, 3]
    assert record["decodes"] == [1, 2, 3]
    assert record["vae_sessions"] == ["open"]
    assert [image.to_bytes()[0] for image in images] == [1, 2, 3]
    assert runtime.pending_latents() == 0


def test_a_latent_is_decoded_only_by_the_call_that_sampled_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    record: dict[str, list[object]] = {
        key: [] for key in ("loads", "samples", "activate", "deactivate", "vae_sessions", "decodes")
    }
    _install(monkeypatch, record)
    folder = tmp_path / "weights"
    try:
        with runtime.pipeline_scope():
            with runtime.render_phase():
                screened = issue_input("frame 0", "text", ())
                try:
                    runtime.sample_latent(_plan(), folder, screened, 8, 8, Tier.GB16, seed=1)
                finally:
                    screened.wipe()
            other = issue_input("frame 0", "text", ())
            try:
                runtime.render(_plan(), folder, other, 8, 8, Tier.GB16, seed=2)
            finally:
                other.wipe()
            assert record["samples"] == [1, 2]
            assert runtime.pending_latents() == 1
    finally:
        runtime.clear_series_embeds()
        runtime.reset_pipeline_cache()
    assert runtime.pending_latents() == 0


def test_the_text_encoder_builds_under_the_process_load_lock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    held: list[bool] = []

    class _Param:
        def __init__(self, dtype: str) -> None:
            self.dtype = dtype

    class _Encoder:
        def __init__(self) -> None:
            self.params = [_Param("bf16"), _Param("fp32")]

        def parameters(self):
            return iter(self.params)

    def other_thread_can_build() -> bool:
        from canvas.infra.load_lock import _LOCK

        result: list[bool] = []

        def probe() -> None:
            got = _LOCK.acquire(blocking=False)
            if got:
                _LOCK.release()
            result.append(got)

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()
        return result[0]

    def model(*_args: object, **_kwargs: object) -> _Encoder:
        held.append(not other_thread_can_build())
        return _Encoder()

    transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
        AutoModel=types.SimpleNamespace(from_pretrained=model),
    )
    monkeypatch.setattr(runtime, "_import_transformers", lambda: transformers)
    torch = types.SimpleNamespace(
        bfloat16="bf16", float32="fp32", cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    _tokenizer, encoder = runtime._load_text_pair(tmp_path / "weights", torch)
    assert held == [True]
    assert isinstance(encoder, _Encoder)
    log = _log(tmp_path / "home")
    assert "stage text_encoder dtype " in log
    assert "params=2 fp32=1" in log
    assert "encoder.fp32" in log


def test_run_sampler_takes_only_a_screened_input_and_reports_a_missing_sample() -> None:
    with pytest.raises(TypeError):
        run_sampler(object(), "a prompt", width=8, height=8)  # type: ignore[arg-type]
    screened = issue_input("frame", "text", ())
    try:
        assert run_sampler(object(), screened, width=8, height=8) is False
        seen: list[tuple[int, int]] = []
        generator = types.SimpleNamespace(sample=lambda _screened, width, height: seen.append((width, height)))
        assert run_sampler(generator, screened, width=8, height=4) is True
        assert seen == [(8, 4)]
    finally:
        screened.wipe()


def test_a_single_picture_runner_does_not_sample_ahead() -> None:
    runner = ShellRunner(gate=World().gate(), tier=Tier.GB16, enlarge=object())
    assert runner.cook is False
    assert runner.sample(object(), ()) is False  # type: ignore[arg-type]
