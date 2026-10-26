"""Screening happens before the generator, and a refusal writes nothing."""
from dataclasses import replace

from canvas.infra.thermal import FULL_SPEED_S, Guard
from canvas.models.loader import ModelNotReady, ZImageGenerator
from canvas.models.modes import Mode, Tier
from canvas.models.plan import BaseNotSelected, ModeUnavailable
from canvas.models.registry import load_catalog
from canvas.pipeline.pipeline import CoolingDown, JobError, Pipeline
from canvas.safety.classifiers import Face
from canvas.safety.gate import GenerationBlocked
from canvas.safety.image import RawImage, SafeImage, ScreenedInput
from canvas.safety.inputs import ImageRole, PendingInput
from tests.fakes import World, avif_file, rgb


class _Gen:
    def __init__(self, image: RawImage | None = None, *, reuse: bool = False) -> None:
        self.image = image if image is not None else rgb()
        self.reuse = reuse
        self.calls = 0
        self.seen = ""

    def generate(self, screened: ScreenedInput) -> RawImage:
        self.calls += 1
        self.seen = screened.prompt
        if self.reuse:
            return self.image
        return RawImage(self.image.width, self.image.height, self.image.to_bytes())


def test_blocked_prompt_never_calls_the_generator_or_writes(tmp_path):
    world = World()
    gen = _Gen()
    pipeline = Pipeline(world.gate(), gen, Tier.GB24)
    path = tmp_path / "out.avif"
    try:
        pipeline.export("nude child", Mode.TEXT, (), path)
    except GenerationBlocked:
        pass
    else:
        raise AssertionError("expected a refusal")
    assert gen.calls == 0
    assert not path.exists()


def test_missing_role_and_hidden_mode_skip_the_generator():
    world = World()
    gen = _Gen()
    pipeline = Pipeline(world.gate(), gen, Tier.GB16)
    try:
        pipeline.generate("a red boat", Mode.INPAINT, [PendingInput(ImageRole.INIT, rgb())])
    except JobError as exc:
        assert str(exc) == "This mode needs another image."
    else:
        raise AssertionError("expected a missing image")
    try:
        pipeline.generate("a red boat", Mode.TEXT_CREATIVE)
    except ModeUnavailable:
        pass
    else:
        raise AssertionError("expected a hidden mode")
    assert gen.calls == 0


def test_allowed_job_saves_only_a_released_image_and_wipes_the_request(tmp_path):
    world = World()
    gen = _Gen(rgb(color=(3, 3, 3)))
    pipeline = Pipeline(world.gate(), gen, Tier.GB16)
    path = tmp_path / "out.avif"
    image = pipeline.generate(
        "a red boat",
        Mode.RESTYLE,
        [PendingInput(ImageRole.INIT, rgb()), PendingInput(ImageRole.REFERENCE, rgb())],
    )
    assert type(image) is SafeImage
    assert gen.calls == 1
    assert world.calls.count("nsfw") == 3  # two inputs and the output (two-stage gate: NSFW pass only)
    pipeline.export("a red boat", Mode.TEXT, (), path)
    assert avif_file(path.read_bytes())


def test_blocked_output_is_wiped_and_not_saved(tmp_path):
    world = World()
    world.nsfw = 0.95
    world.faces = [Face(0, 0, 1, 1)]
    world.ages = [(1.0, 0.0)]
    raw = rgb(color=(9, 9, 9))
    gen = _Gen(raw, reuse=True)
    pipeline = Pipeline(world.gate(), gen, Tier.GB24)
    path = tmp_path / "out.avif"
    try:
        pipeline.export("a red boat", Mode.TEXT, (), path)
    except GenerationBlocked:
        pass
    else:
        raise AssertionError("expected a refusal")
    assert raw.to_bytes() == b""
    assert not path.exists()


def test_stub_generator_becomes_unavailable_after_the_plan():
    world = World()
    catalog = load_catalog()
    screened = world.gate().release_request("a red boat", Mode.TEXT.value, ())
    waiting = ZImageGenerator(replace(catalog, default_text_base=None), Tier.GB16)
    try:
        waiting.generate(screened)
    except BaseNotSelected as exc:
        assert str(exc) == "The text model has not been selected yet."
    else:
        raise AssertionError("expected an unset text model")
    generator = ZImageGenerator(catalog, Tier.GB16)
    try:
        generator.generate(screened)
    except ModelNotReady as exc:
        assert exc.plan.pipeline_class == "ZImagePipeline"
        assert exc.plan.diffusers == "0.41.0"
    else:
        raise AssertionError("expected the stub to refuse")
    pipeline = Pipeline(world.gate(), generator, Tier.GB16)
    try:
        pipeline.generate("a red boat", Mode.TEXT, ())
    except Exception as exc:
        assert str(exc) == "Model missing, fetch it with MDL."
        assert "ZImage" not in str(exc)
    else:
        raise AssertionError("expected an unavailable message")


def test_thermal_pause_does_not_generate():
    world = World()
    gen = _Gen()
    clock = {"t": 0.0}
    guard = Guard(now=lambda: clock["t"])
    for index in range(3):
        clock["t"] = FULL_SPEED_S + index
        guard.observe(90, False)
    pipeline = Pipeline(world.gate(), gen, Tier.GB16, thermal=guard)
    try:
        pipeline.generate("a red boat", Mode.TEXT, ())
    except CoolingDown as exc:
        assert "cooling down" in str(exc)
    else:
        raise AssertionError("expected a pause")
    assert pipeline.paused is True
    assert gen.calls == 0
