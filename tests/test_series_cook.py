"""Cook holds the transformer and the VAE. The text encoder stays off the GPU."""
from __future__ import annotations

import types
from pathlib import Path

from PIL import Image

from canvas.infra.errors import note_crash
from canvas.infra.stage_timer import StageTimes, set_current
from canvas.infra.vram_trace import note_vram
from canvas.models import hot_tier, runtime
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
        self.device = _Device("cpu")
        self.history: list[str] = []

    def parameters(self):
        yield self

    def to(self, *args: object, **kwargs: object) -> _Encoder:
        if kwargs or len(args) != 1 or not isinstance(args[0], str):
            raise AssertionError("the text encoder was recast")
        self.history.append(args[0])
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


_caches = {"n": 0}
_loaded: list[str] = []


def _torch() -> types.SimpleNamespace:
    class _Generator:
        def __init__(self, device: object = None) -> None:
            self.device = device

        def manual_seed(self, seed: int) -> _Generator:
            self.seed = seed
            return self

    def empty_cache() -> None:
        _caches["n"] += 1

    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=empty_cache,
        memory_allocated=lambda: 0,
        max_memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
        mem_get_info=lambda: (15 * 1024**3, 16 * 1024**3),
        reset_peak_memory_stats=lambda: None,
    )
    return types.SimpleNamespace(bfloat16="bf16", Generator=_Generator, cuda=cuda)


def _pipelines(encoder: _Encoder, transformer: _Module, vae: _Module, seen: list[str]) -> types.SimpleNamespace:
    class TextPipe:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> TextPipe:
            _loaded.append("pipeline")
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
    _caches["n"] = 0
    _loaded.clear()
    monkeypatch.setattr(runtime, "_import_torch", _torch)
    monkeypatch.setattr(runtime, "_import_diffusers", lambda: _pipelines(encoder, transformer, vae, seen))
    monkeypatch.setattr(runtime, "_place", lambda _pipe, _tier: None)
    monkeypatch.setattr(runtime, "reset_peak_allocated", lambda: None)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: _GB32)
    monkeypatch.setattr(runtime, "_load_text_pair", lambda _folder, _torch: (object(), encoder))
    monkeypatch.setattr(
        runtime, "_encode_prompt_text", lambda _tokenizer, _encoder, prompt, _torch: f"embed:{prompt}",
    )
    monkeypatch.setattr(runtime, "_encoder_cosine", lambda _folder, _tier, _torch: 1.0)
    monkeypatch.setattr(runtime, "_fp8_ready", lambda _torch: True)
    monkeypatch.setattr(runtime, "cast_module_fp8", lambda *_args, **_kwargs: True)


def _plan(pipeline: str) -> CallPlan:
    mode = "text" if pipeline == PIPELINE_TEXT else "restyle"
    return CallPlan("diffusers", pipeline, ("stub",), "bf16", 8, None, "none", "16gb", mode)


def test_render_logs_the_fit_budget(tmp_path: Path, monkeypatch) -> None:
    """The picture budget is written by render, through the same log as the stage lines."""
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    encoder, transformer, vae = _Encoder(), _Module(), _Module()
    _install(monkeypatch, encoder, transformer, vae, [])
    runtime.reset_pipeline_cache()
    runtime.clear_series_embeds()
    screened = issue_input("one frame", "text", ())
    try:
        with runtime.pipeline_scope():
            runtime.prepare_series_embeds(tmp_path / "weights", Tier.GB16, ("one frame",))
            runtime.render(_plan(PIPELINE_TEXT), tmp_path / "weights", screened, 8, 8, Tier.GB16, seed=1)
    finally:
        screened.wipe()
        runtime.clear_series_embeds()
        runtime.reset_pipeline_cache()
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    assert "stage fit budget" in log
    assert "free=" in log
    assert "total=" in log
    assert "needed=" in log
    assert "residual=" in log
    assert "headroom=" in log
    assert "float8_e4m3fn" in log


def test_the_text_encoder_is_not_on_cuda_during_cook(tmp_path: Path, monkeypatch) -> None:
    encoder, transformer, vae = _Encoder(), _Module(), _Module()
    seen: list[str] = []
    _install(monkeypatch, encoder, transformer, vae, seen)
    runtime.reset_pipeline_cache()
    runtime.clear_series_embeds()
    try:
        with runtime.pipeline_scope():
            runtime.prepare_series_embeds(tmp_path / "weights", Tier.GB16, ("one frame", "two frame"))
            assert _loaded == []
            assert _caches["n"] >= 1
            assert encoder.history[:2] == ["cuda", "cpu"]
            assert runtime.text_encoder_on_cuda(types.SimpleNamespace(text_encoder=encoder)) is False
            assert runtime._slot is None
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
            assert vae.calls == [(("cuda", "bf16"), {})]
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
    original = runtime.park_pipeline

    def release() -> None:
        order.append("release")
        original()

    monkeypatch.setattr(runtime, "park_pipeline", release)
    monkeypatch.setattr("canvas.pipeline.series.park_pipeline", release)
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
    assert "peak=" in log
    assert "models=" in log


def test_sampling_time_is_on_the_timing_line() -> None:
    clock = StageTimes()
    previous = set_current(clock)

    class _Pipe:
        def __call__(self, **_kwargs: object) -> str:
            return "ok"

    try:
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        assert runtime._timed(_Pipe(), torch) == "ok"
    finally:
        set_current(previous)
    assert "sample=" in clock.log_line()
    assert clock.log_line().startswith("timing wall=")


def test_the_second_frame_unparks_the_vae_before_encode(tmp_path: Path, monkeypatch) -> None:
    """Img2img encode unparks the VAE, checks CUDA BF16, and parks it again."""
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("VOXPRINT_RAM_MODE", "ram96")
    seen: dict[str, object] = {}

    class _Image:
        def __init__(self) -> None:
            self.device: object = "cuda"
            self.dtype: object = "fp32"
            self.shape = (1, 3, 64, 64)

        def to(self, *args: object, **kwargs: object) -> _Image:
            if "device" in kwargs:
                self.device = kwargs["device"]
            if "dtype" in kwargs:
                self.dtype = kwargs["dtype"]
            return self

    class _Param:
        def __init__(self) -> None:
            self.device: object = "cpu"
            self.dtype: object = "bf16"

    class _Vae:
        def __init__(self) -> None:
            self.param = _Param()

        def parameters(self):
            yield self.param

        def named_parameters(self):
            return iter(())

        def named_buffers(self):
            return iter(())

        def to(self, *args: object, **kwargs: object) -> _Vae:
            del kwargs
            if args:
                self.param.device = args[0]
                if len(args) > 1 and args[1] is not None:
                    self.param.dtype = args[1]
            return self

        def encode(self, image: _Image, *args: object, **kwargs: object) -> str:
            del args, kwargs
            seen["device"] = self.param.device
            seen["dtype"] = self.param.dtype
            seen["image_device"] = image.device
            seen["image_dtype"] = image.dtype
            return "latent"

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
            image = Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"])), (20, 40, 60))
            return types.SimpleNamespace(images=[image])

    class ImgPipe:
        def __init__(
            self, scheduler: object, vae: object, text_encoder: object, tokenizer: object, transformer: object,
        ) -> None:
            self.scheduler = scheduler
            self.vae = vae
            self.text_encoder = text_encoder
            self.tokenizer = tokenizer
            self.transformer = transformer
            self.components = {"text_encoder": text_encoder, "transformer": transformer, "vae": vae}

        def __call__(self, **kwargs: object) -> types.SimpleNamespace:
            self.vae.encode(_Image())
            image = Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"])), (30, 50, 70))
            return types.SimpleNamespace(images=[image])

    encoder, transformer, vae = _Encoder(), _Module(), _Vae()
    diffusers = types.SimpleNamespace(ZImagePipeline=TextPipe, ZImageImg2ImgPipeline=ImgPipe)
    _install(monkeypatch, encoder, transformer, _Module(), [])
    monkeypatch.setattr(runtime, "_import_diffusers", lambda: diffusers)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: 96 * 1024**3)
    hot_tier.reset()
    runtime.reset_pipeline_cache()
    runtime.clear_series_embeds()
    try:
        with runtime.pipeline_scope():
            runtime.prepare_series_embeds(tmp_path / "weights", Tier.GB16, ("one frame", "two frame"))
            first = issue_input("one frame", "text", ())
            try:
                runtime.render(_plan(PIPELINE_TEXT), tmp_path / "weights", first, 8, 8, Tier.GB16, seed=1)
            finally:
                first.wipe()
            hero = RawImage(8, 8, bytes([9, 8, 7]) * 64)
            second = issue_input("two frame", "restyle", (("init", 8, 8, hero.to_bytes()),))
            try:
                runtime.render(
                    _plan(PIPELINE_IMG2IMG), tmp_path / "weights", second, 8, 8, Tier.GB16, seed=2, strength=0.8,
                )
            finally:
                second.wipe()
    finally:
        runtime.clear_series_embeds()
        runtime.reset_pipeline_cache()
        hot_tier.reset()
    assert seen["device"] == "cuda"
    assert seen["dtype"] == "bf16"
    assert seen["image_device"] == "cuda"
    assert seen["image_dtype"] == "bf16"
    log = (tmp_path / "logs" / "voxprint.log").read_text(encoding="utf-8")
    unpark = log.index("stage unpark vae")
    assert log.index("stage park vae", unpark) > unpark
    assert "stage vae_encode resident" in log
    assert "stage vae_encode peak" in log


def _kind_of(module: object) -> str | None:
    """The device family DiffusionPipeline reads off one component."""
    if module is None:
        return None
    device = getattr(module, "device", None)
    if device is None and callable(getattr(module, "parameters", None)):
        for param in module.parameters():
            device = getattr(param, "device", None)
            break
    if device is None:
        return None
    kind = getattr(device, "type", device)
    text = str(kind)
    if "cuda" in text:
        return "cuda"
    if "cpu" in text:
        return "cpu"
    return text


def _pipeline_latent_device(pipe: object) -> str:
    """The device ``prepare_latents`` captures before encode.

    A component on CUDA wins. Otherwise the first component's device is the
    one ``randn_tensor`` is asked to fill.
    """
    found: list[str] = []
    for name in ("vae", "transformer", "unet", "controlnet", "text_encoder"):
        kind = _kind_of(getattr(pipe, name, None))
        if kind is not None:
            found.append(kind)
    for kind in found:
        if kind not in {"cpu", "meta"}:
            return kind
    return found[0] if found else "cpu"


def test_the_img2img_second_frame_generator_matches_the_latent_device(tmp_path: Path, monkeypatch) -> None:
    """Img2img noise is drawn on the device prepare_latents captured before encode."""
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path))
    monkeypatch.setenv("VOXPRINT_RAM_MODE", "ram96")
    seen: dict[str, str] = {}
    flags: list[tuple[str, bool]] = []
    original_activate = hot_tier.activate

    def activate(name: str, device: str, *, blocking: bool = False) -> bool:
        flags.append((name, blocking))
        return original_activate(name, device, blocking=blocking)

    monkeypatch.setattr(hot_tier, "activate", activate)

    class _Image:
        def __init__(self) -> None:
            self.device: object = "cuda"
            self.dtype: object = "fp32"
            self.shape = (1, 3, 64, 64)

        def to(self, *args: object, **kwargs: object) -> _Image:
            if "device" in kwargs:
                self.device = kwargs["device"]
            if "dtype" in kwargs:
                self.dtype = kwargs["dtype"]
            return self

    class _Weight:
        def __init__(self) -> None:
            self.device = _Device("cpu")
            self.dtype: object = "bf16"
            self.nbytes = 8
            self.data: object = self

        def to(self, *args: object, **kwargs: object) -> _Weight:
            device = kwargs.get("device", args[0] if args else None)
            if device is not None and not isinstance(device, bool):
                self.device = _Device(str(getattr(device, "type", device)))
            if len(args) > 1 and args[1] is not None and not isinstance(args[1], bool):
                self.dtype = args[1]
            if kwargs.get("dtype") is not None:
                self.dtype = kwargs["dtype"]
            return self

        def pin_memory(self) -> _Weight:
            return self

        def detach(self) -> _Weight:
            return self

        def numel(self) -> int:
            return 1

        def element_size(self) -> int:
            return 8

    class _Vae:
        def __init__(self) -> None:
            self.param = _Weight()

        @property
        def device(self) -> _Device:
            return self.param.device

        def parameters(self):
            yield self.param

        def named_parameters(self):
            yield "weight", self.param

        def named_buffers(self):
            return iter(())

        def to(self, *args: object, **kwargs: object) -> _Vae:
            self.param.to(*args, **kwargs)
            return self

        def encode(self, image: _Image, *args: object, **kwargs: object) -> str:
            del image, args, kwargs
            return "latent"

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
            image = Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"])), (20, 40, 60))
            return types.SimpleNamespace(images=[image])

    class ImgPipe:
        def __init__(
            self, scheduler: object, vae: object, text_encoder: object, tokenizer: object, transformer: object,
        ) -> None:
            self.scheduler = scheduler
            self.vae = vae
            self.text_encoder = text_encoder
            self.tokenizer = tokenizer
            self.transformer = transformer
            self.components = {"text_encoder": text_encoder, "transformer": transformer, "vae": vae}

        def __call__(self, **kwargs: object) -> types.SimpleNamespace:
            latent = _pipeline_latent_device(self)
            generator = kwargs["generator"]
            gen_device = getattr(generator, "device", None)
            gen_type = str(getattr(gen_device, "type", gen_device))
            if "cuda" in gen_type:
                gen_type = "cuda"
            elif "cpu" in gen_type:
                gen_type = "cpu"
            if gen_type != latent and gen_type != "cpu":
                raise ValueError(f"Cannot generate a {latent} tensor from a generator of type {gen_type}.")
            seen["generator"] = gen_type
            seen["latents"] = latent
            self.vae.encode(_Image())
            image = Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"])), (30, 50, 70))
            return types.SimpleNamespace(images=[image])

    encoder, transformer, vae = _Encoder(), _Module(), _Vae()
    diffusers = types.SimpleNamespace(ZImagePipeline=TextPipe, ZImageImg2ImgPipeline=ImgPipe)
    _install(monkeypatch, encoder, transformer, _Module(), [])
    monkeypatch.setattr(runtime, "_import_diffusers", lambda: diffusers)
    monkeypatch.setattr(runtime, "system_ram_bytes", lambda: 96 * 1024**3)
    hot_tier.reset()
    runtime.reset_pipeline_cache()
    runtime.clear_series_embeds()
    try:
        with runtime.pipeline_scope():
            runtime.prepare_series_embeds(tmp_path / "weights", Tier.GB16, ("one frame", "two frame"))
            first = issue_input("one frame", "text", ())
            try:
                runtime.render(_plan(PIPELINE_TEXT), tmp_path / "weights", first, 8, 8, Tier.GB16, seed=1)
            finally:
                first.wipe()
            hero = RawImage(8, 8, bytes([9, 8, 7]) * 64)
            second = issue_input("two frame", "restyle", (("init", 8, 8, hero.to_bytes()),))
            try:
                runtime.render(
                    _plan(PIPELINE_IMG2IMG), tmp_path / "weights", second, 8, 8, Tier.GB16, seed=2, strength=0.8,
                )
            finally:
                second.wipe()
    finally:
        runtime.clear_series_embeds()
        runtime.reset_pipeline_cache()
        hot_tier.reset()
    assert seen["generator"] == seen["latents"]
    assert seen["latents"] == "cuda"
    transformer_copies = [blocking for name, blocking in flags if name == "transformer"]
    assert transformer_copies
    assert all(blocking is False for blocking in transformer_copies)
