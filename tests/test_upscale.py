"""4K is an upscale of the ~2K frame, then the output checks run again."""
from dataclasses import replace

import pytest

from canvas.infra.gpu_lock import lock_path, read_lock
from canvas.models.modes import Mode, Tier
from canvas.models.registry import load_catalog
from canvas.models.sizes import frame_size
from canvas.pipeline.pipeline import EnlargeMissing, Pipeline
from canvas.pipeline.vram_probe import PeakVramProbe
from canvas.safety.classifiers import Face
from canvas.safety.gate import GenerationBlocked
from canvas.safety.image import RawImage, ScreenedInput
from canvas.safety.inputs import ImageRole, PendingInput
from canvas.core.exchange import read_rgb, write_rgb
from tests.fakes import World, rgb


class _Gen:
    def __init__(self, catalog) -> None:
        self.catalog = catalog
        self.order: list[str] = []
        self.sizes: list[tuple[int, int]] = []
        self.lock_held = False

    def generate(self, screened: ScreenedInput, width: int, height: int) -> RawImage:
        del screened
        self.order.append("generate")
        self.sizes.append((width, height))
        self.lock_held = (read_lock(lock_path()) or {}).get("owner") == "canvas"
        return rgb(width, height, color=(1, 2, 3))

    def unload(self) -> None:
        self.order.append("unload")


class _Enlarge:
    def __init__(self, world: World | None = None) -> None:
        self.world = world
        self.scale = 0
        self.unlocked = False
        self.source_long = 0

    def installed(self) -> bool:
        return True

    def enlarge(self, source, *, scale, output, format_name="avif", on_event=None):
        del format_name
        raw = read_rgb(source)
        self.scale = scale
        self.source_long = max(raw.width, raw.height)
        if on_event is not None:
            on_event({"protocol": 1, "type": "progress", "stage": "waiting_gpu", "pct": 1})
        self.unlocked = read_lock(lock_path()) is None
        if self.world is not None:
            self.world.nsfw = 0.95
            self.world.faces = [Face(0, 0, 1, 1)]
            self.world.ages = [8.0]
        width, height = raw.width * scale, raw.height * scale
        write_rgb(output, width, height, bytes((4, 5, 6)) * (width * height))
        return type("Result", (), {"output": output})()


def test_fourk_chain_unloads_on_16gb_and_screens_the_upscale(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "voxprint-gpu.lock"))
    catalog = replace(load_catalog(), default_text_base="qwen-image-2512")
    world = World()
    gen = _Gen(catalog)
    enlarge = _Enlarge()
    probe = PeakVramProbe(sample=lambda: 100)
    pipeline = Pipeline(world.gate(), gen, Tier.GB16, use_gpu_lock=True, catalog=catalog, enlarge=enlarge)
    image = pipeline.generate("a red boat", Mode.TEXT, aspect="16:9", output="4k", probe=probe)
    expected = frame_size(catalog, "qwen-image-2512", "16:9", twok=True)
    assert gen.sizes == [(expected.width, expected.height)]
    assert max(expected.width, expected.height) * enlarge.scale == 3840
    assert enlarge.source_long < 3840
    assert gen.order == ["generate", "unload"]
    assert gen.lock_held is True
    assert enlarge.unlocked is True
    assert image.width == expected.width * enlarge.scale
    assert image.height == expected.height * enlarge.scale
    assert [mark for mark, _used in probe.marks] == [
        "generate-start", "generate-end", "unload", "upscale-start", "upscale-end",
    ]
    assert read_lock(lock_path()) is None
    path = tmp_path / "out.avif"
    pipeline.export("a red boat", Mode.TEXT, (), path, aspect="1:1", output="4k")
    assert path.read_bytes().startswith(b"VOXPRINT avif\n")


def test_24gb_keeps_the_generator_and_image_modes_use_zimage_frames(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "lock"))
    catalog = replace(load_catalog(), default_text_base="qwen-image-2512")
    world = World()
    gen = _Gen(catalog)
    enlarge = _Enlarge()
    pipeline = Pipeline(world.gate(), gen, Tier.GB24, use_gpu_lock=True, enlarge=enlarge)
    pipeline.generate(
        "a red boat",
        Mode.RESTYLE,
        [PendingInput(ImageRole.INIT, rgb())],
        aspect="3:2",
        output="4k",
    )
    expected = frame_size(catalog, "z-image-turbo", "3:2", twok=True)
    assert gen.sizes == [(expected.width, expected.height)]
    assert gen.order == ["generate"]
    assert "unload" not in gen.order


def test_missing_enlarge_does_not_generate_and_a_blocked_upscale_is_not_saved(tmp_path):
    catalog = replace(load_catalog(), default_text_base="z-image-turbo")
    world = World()
    gen = _Gen(catalog)
    pipeline = Pipeline(world.gate(), gen, Tier.GB24, catalog=catalog)
    with pytest.raises(EnlargeMissing):
        pipeline.generate("a red boat", Mode.TEXT, output="4k")
    assert gen.order == []
    blocked = World()
    enlarge = _Enlarge(blocked)
    saving = Pipeline(blocked.gate(), _Gen(catalog), Tier.GB24, catalog=catalog, enlarge=enlarge)
    path = tmp_path / "blocked.avif"
    with pytest.raises(GenerationBlocked):
        saving.export("a red boat", Mode.TEXT, (), path, output="4k")
    assert not path.exists()
