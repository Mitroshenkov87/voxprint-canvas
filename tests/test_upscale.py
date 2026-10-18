"""4K is an upscale of the ~2K frame, then the output checks run again."""
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from canvas.infra.enlarge import EnlargeClient, EnlargeError

from canvas.infra.gpu_lock import lock_path, read_lock
from canvas.models.modes import Mode, Tier
from canvas.models.registry import load_catalog
from canvas.models.sizes import frame_size, upscale_factor
from canvas.pipeline.pipeline import EnlargeMissing, Pipeline
from canvas.pipeline.vram_probe import PeakVramProbe
from canvas.safety.classifiers import Face
from canvas.safety.gate import GenerationBlocked
from canvas.safety.image import RawImage, ScreenedInput
from canvas.safety.inputs import ImageRole, PendingInput
from tests.fakes import World, avif_file, rgb


class _Gen:
    def __init__(self, catalog) -> None:
        self.catalog = catalog
        self.order: list[str] = []
        self.sizes: list[tuple[int, int]] = []
        self.lock_held = False
        self.text_encoder = object()
        self.vae = object()

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
        self.long_side = 0
        self.lock_held = False
        self.source_long = 0

    def installed(self) -> bool:
        return True

    def enlarge(self, source, *, output, format_name="avif", long_side=3840, on_event=None):
        del format_name, on_event
        with Image.open(source) as image:
            raw_w, raw_h = image.size
        self.long_side = long_side
        self.source_long = max(raw_w, raw_h)
        self.lock_held = (read_lock(lock_path()) or {}).get("owner") == "canvas"
        if self.world is not None:
            self.world.nsfw = 0.95
            self.world.faces = [Face(0, 0, 1, 1)]
            self.world.ages = [(1.0, 0.0)]
        factor = long_side // self.source_long
        width, height = raw_w * factor, raw_h * factor
        Image.new("RGB", (width, height), (4, 5, 6)).save(output, format="PNG")
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
    factor = upscale_factor(expected)
    assert gen.sizes == [(expected.width, expected.height)]
    assert enlarge.long_side == 3840
    assert max(expected.width, expected.height) * factor == 3840
    assert enlarge.source_long < 3840
    assert gen.order == ["generate", "unload"]
    assert gen.lock_held is True
    assert enlarge.lock_held is False
    assert gen.text_encoder is None
    assert gen.vae is None
    assert image.width == expected.width * factor
    assert image.height == expected.height * factor
    assert [mark for mark, _used in probe.marks] == [
        "generate-start", "generate-end", "unload", "upscale-start", "upscale-end",
    ]
    assert read_lock(lock_path()) is None
    path = tmp_path / "out.avif"
    pipeline.export("a red boat", Mode.TEXT, (), path, aspect="1:1", output="4k")
    assert avif_file(path.read_bytes())


def test_fourk_cli_releases_the_lock_and_starts_from_a_clean_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "voxprint-gpu.lock"))
    monkeypatch.setenv("CANVAS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("PYTHONPATH", "/from-the-parent")
    monkeypatch.setenv("VIRTUAL_ENV", "/from-the-parent")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:32")
    catalog = replace(load_catalog(), default_text_base="qwen-image-2512")
    world = World()
    gen = _Gen(catalog)
    fake = Path(__file__).resolve().parent / "fixtures" / "enlarge_fake.py"
    client = EnlargeClient([sys.executable, str(fake)])
    pipeline = Pipeline(world.gate(), gen, Tier.GB16, use_gpu_lock=True, catalog=catalog, enlarge=client)
    image = pipeline.generate("a red boat", Mode.TEXT, aspect="16:9", output="4k")
    expected = frame_size(catalog, "qwen-image-2512", "16:9", twok=True)
    factor = upscale_factor(expected)
    assert client.ready()
    assert gen.order == ["generate", "unload"]
    assert image.width == expected.width * factor
    assert image.height == expected.height * factor
    assert read_lock(lock_path()) is None
    assert world.calls.count("nsfw") == 2  # two-stage gate: clothed frames get the NSFW pass only


def test_24gb_unloads_before_enlarge_and_image_modes_use_zimage_frames(monkeypatch, tmp_path):
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
    assert gen.order == ["generate", "unload"]
    assert enlarge.lock_held is False
    assert gen.text_encoder is None
    assert gen.vae is None


def test_fourk_does_not_spawn_enlarge_without_enough_free_memory(tmp_path, monkeypatch):
    monkeypatch.setattr("canvas.pipeline.pipeline.free_vram_gb", lambda: 5.0)
    catalog = replace(load_catalog(), default_text_base="z-image-turbo")
    enlarge = _Enlarge()
    enlarge.capabilities = lambda: {"protocol": 1, "gpu": {"vram_gb": 8}}
    pipeline = Pipeline(World().gate(), _Gen(catalog), Tier.GB24, catalog=catalog, enlarge=enlarge)
    with pytest.raises(EnlargeError) as caught:
        pipeline.generate("a red boat", Mode.TEXT, output="4k")
    assert caught.value.code == "oom"
    assert enlarge.long_side == 0
    monkeypatch.setattr("canvas.pipeline.pipeline.free_vram_gb", lambda: 9.0)
    enlarge.long_side = 0
    pipeline.generate("a red boat", Mode.TEXT, output="4k")
    assert enlarge.long_side == 3840


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


def test_fourk_cli_error_text_is_the_callee_message(tmp_path, monkeypatch):
    monkeypatch.setenv("ENLARGE_FAKE_EXIT", "2")
    catalog = replace(load_catalog(), default_text_base="z-image-turbo")
    fake = Path(__file__).resolve().parent / "fixtures" / "enlarge_fake.py"
    client = EnlargeClient([sys.executable, str(fake)])
    pipeline = Pipeline(World().gate(), _Gen(catalog), Tier.GB24, catalog=catalog, enlarge=client)
    with pytest.raises(EnlargeError) as caught:
        pipeline.generate("a red boat", Mode.TEXT, output="4k")
    assert caught.value.code == "bad_input"
    assert "raw cli detail model-id foo" in str(caught.value)
