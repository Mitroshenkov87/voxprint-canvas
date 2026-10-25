"""Cook holds the transformer and the VAE. The text encoder stays off the GPU."""
from __future__ import annotations

import types
from pathlib import Path

from PIL import Image

from canvas.infra.errors import note_crash
from canvas.infra.stage_timer import StageTimes, set_current
from canvas.infra.vram_trace import note_vram
from canvas.models import runtime
from canvas.models.modes import Tier
from canvas.models.plan import PIPELINE_IMG2IMG, PIPELINE_TEXT, CallPlan
from canvas.pipeline.series import run_series
from canvas.safety.gate import issue_input
from canvas.safety.image import RawImage
from tests.fakes import World
from tests.test_photoshoot import PERSON, _payload

_GB32 = 32 * 1024**3


class _Device:
    def __init__(self, kind: str) -> None:
        self.type = kind


class _Encoder:
    def __init__(self) -> None:
        self.device = _Device("cuda")

    def parameters(self):
        yield self

    def to(self, *args: object, **kwargs: object) -> _Encoder:
        if kwargs or len(args) != 1 or not isinstance(args[0], str):
            raise AssertionError("the text encoder was recast")
        self.device = _Device(args[0])
        return self


class _Module:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def parameters(self):
        return iter(())

    def to(self, *args: object, **kwargs: object) -> _Module:
        self.calls.append((args, kwargs))
        return self


def _torch() -> types.SimpleNamespace:
    class _Generator:
        def __init__(self, device: object = None) -> None:
            self.device = device

        def manual_seed(self, seed: int) -> _Generator:
            self.seed = seed
            return self

    cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
        mem_get_info=lambda: (0, 0),
    )
    return types.SimpleNamespace(bfloat16="bf16", Generator=_Generator, cuda=cuda)


def _pipelines(encoder: _Encoder, transformer: _Module, vae: _Module, seen: list[str]) -> types.SimpleNamespace:
    class TextPipe:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> TextPipe:
            return cls()

        def __init__(self) -> None:
            self.text_encoder = encoder
            self.transformer = transformer
            self.vae = vae
            self.tokenizer = object()
            self.scheduler = object()
            self.components = {"text_encoder": encoder, "transformer": transformer, "vae": vae}

        def encode_prompt(self, prompt: str, **_kwargs: object) -> tuple[str]:
            return (f"embed:{prompt}",)

        def __call__(self, **kwargs: object) -> types.SimpleNamespace:
            assert runtime.text_encoder_on_cuda(self) is False
            assert kwargs.get("prompt") is None
            assert kwargs.get("prompt_embeds")
            seen.append("text")
            image = Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"])), (20, 40, 60))
            return types.SimpleNamespace(images=[image])

    class ImgPipe:
        def __init__(self, scheduler: object, vae: object, text_encoder: object, tokenizer: object, transformer: object) -> None:
            self.scheduler = scheduler
            self.vae = vae
            self.text_encoder = text_encoder
            self.tokenizer = tokenizer
            self.transformer = transformer
            self.components = {"text_encoder": text_encoder, "transformer": transformer, "vae": vae}

        def __call__(self, **kwargs: object) -> types.SimpleNamespace:
            assert runtime.text_encoder_on_cuda(self) is False
            assert kwargs.get("prompt") is None
            assert kwargs.get("prompt_embeds")
            seen.append("img")
            image = Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"])), (30, 50, 70))
            return types.SimpleNamespace(images=[image])

    return types.SimpleNamespace(ZImagePipeline=TextPipe, ZImageImg2ImgPipeline=ImgPipe)


def _install(monkeypatch, encoder: _Encoder, transformer: _Module, vae: _Module, seen: list[str]) -> None:
    monkeypatch.setattr(runtime, "_import_torch", _torch)
    monkeypatch.setattr(runtime, "_import_diffusers", lambda: _pipelines(encoder, transformer, vae, seen))
    monkeypatch.setattr(runtime, "_place", lambda _pipe, _tier: None)
    monkeypatch.setattr(runtime, "reset_peak_allocated", lambda: None)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: _GB32)


def _plan(pipeline: str) -> CallPlan:
    mode = "text" if pipeline == PIPELINE_TEXT else "restyle"
    return CallPlan("diffusers", pipeline, ("stub",), "bf16", 8, None, "none", "16gb", mode)


def test_the_text_encoder_is_not_on_cuda_during_cook(tmp_path: Path, monkeypatch) -> None:
    encoder, transformer, vae = _Encoder(), _Module(), _Module()
    seen: list[str] = []
    _install(monkeypatch, encoder, transformer, vae, seen)
    runtime.reset_pipeline_cache()
    runtime.clear_series_embeds()
    try:
        with runtime.pipeline_scope():
            runtime.prepare_series_embeds(tmp_path / "weights", Tier.GB16, ("one frame", "two frame"))
            held = runtime._slot
            assert held is not None
            assert runtime.text_encoder_on_cuda(held.pipe) is False
            first = issue_input("one frame", "text", ())
            try:
                runtime.render(_plan(PIPELINE_TEXT), tmp_path / "weights", first, 8, 8, Tier.GB16, seed=1)
            finally:
                first.wipe()
            hero = RawImage(8, 8, bytes([9, 8, 7]) * 64)
            second = issue_input("two frame", "restyle", (("init", 8, 8, hero.to_bytes()),))
            try:
                runtime.render(
                    _plan(PIPELINE_IMG2IMG), tmp_path / "weights", second, 8, 8, Tier.GB16,
                    seed=2, strength=0.8,
                )
            finally:
                second.wipe()
            assert seen == ["text", "img"]
            held = runtime._slot
            assert held is not None
            assert runtime.text_encoder_on_cuda(held.pipe) is False
            assert transformer.calls == []
            assert vae.calls == []
    finally:
        runtime.clear_series_embeds()
        runtime.reset_pipeline_cache()


def test_frame_two_runs_through_the_shared_pipeline(tmp_path: Path, monkeypatch) -> None:
    encoder, transformer, vae = _Encoder(), _Module(), _Module()
    seen: list[str] = []
    order: list[str] = []
    _install(monkeypatch, encoder, transformer, vae, seen)
    runtime.reset_pipeline_cache()
    runtime.clear_series_embeds()
    original = runtime.release_pipeline

    def release() -> None:
        order.append("release")
        original()

    monkeypatch.setattr(runtime, "release_pipeline", release)
    monkeypatch.setattr("canvas.pipeline.series.release_pipeline", release)
    gate = World().gate()
    real = gate.review_image

    def review_image(raw: object, prompt: str = "") -> object:
        order.append("check")
        return real(raw, prompt)

    gate.review_image = review_image

    def prepare(lines: list[str]) -> None:
        runtime.prepare_series_embeds(tmp_path / "weights", Tier.GB16, tuple(lines))

    def draw(line: str, seed: int, hero: object | None, strength: float | None) -> RawImage:
        del seed
        order.append("draw")
        if hero is None:
            screened = issue_input(line, "text", ())
            pipeline = PIPELINE_TEXT
        else:
            screened = issue_input(line, "restyle", (("init", hero.width, hero.height, hero.to_bytes()),))
            pipeline = PIPELINE_IMG2IMG
        try:
            return runtime.render(
                _plan(pipeline), tmp_path / "weights", screened, 8, 8, Tier.GB16, strength=strength,
            )
        finally:
            screened.wipe()

    draw.prepare = prepare  # type: ignore[attr-defined]
    try:
        result = run_series(
            idea=PERSON, count=2, style="photo", place_mode="same", seed=3, strength=None,
            include_camera=lambda: True, rewrite=lambda _prompt: _payload(PERSON, 2),
            age_gate=gate, draw_frame=draw, cancelled=lambda: False, folder=tmp_path / "series",
        )
        assert result.stopped == ""
        assert len(result.kept) == 2
        assert seen == ["text", "img"]
        assert order.index("draw") < order.index("release") < order.index("check")
        assert transformer.calls == []
    finally:
        runtime.clear_series_embeds()
        runtime.reset_pipeline_cache()


def test_from_pipe_does_not_recast_shared_weights() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class _Cast:
        def to(self, *args: object, **kwargs: object) -> _Cast:
            calls.append((args, kwargs))
            return self

    module = _Cast()
    pipe = types.SimpleNamespace(components={"transformer": module, "vae": module})

    def convert(source: object) -> object:
        module.to(dtype=object())
        return source

    runtime._from_pipe_keep_dtype(convert, pipe)
    assert calls == []


def test_an_empty_runtime_error_is_logged(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    note_crash("internal", RuntimeError())
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "crash internal RuntimeError message=''" in log
    assert "Traceback" in log
    note_crash("internal", RuntimeError("boom secret"))
    again = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "secret" not in again


def test_vram_trace_records_the_stage(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("VOXPRINT_VRAM_TRACE", "1")
    note_vram("pipeline_load", "before")
    note_vram("pipeline_load", "after")
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "vram pipeline_load before " in log
    assert "vram pipeline_load after " in log
    assert "allocated=" in log and "reserved=" in log and "free=" in log and "total=" in log
    assert "models=" in log


def test_sampling_time_is_on_the_timing_line() -> None:
    clock = StageTimes()
    previous = set_current(clock)

    class _Pipe:
        def __call__(self, **_kwargs: object) -> str:
            return "ok"

    try:
        assert runtime._timed(_Pipe()) == "ok"
    finally:
        set_current(previous)
    assert "sample=" in clock.log_line()
    assert clock.log_line().startswith("timing wall=")
