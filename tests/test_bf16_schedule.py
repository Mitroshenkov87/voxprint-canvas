"""16 GB Z-Image stays inside the 14 GB budget by loading one stage at a time."""
from dataclasses import replace

import pytest

from canvas.core.exchange import write_rgb
from canvas.models.loader import ModelNotReady, ZImageGenerator
from canvas.models.modes import Mode, Tier, headroom_gb
from canvas.models.registry import load_catalog
from canvas.pipeline.bf16_schedule import (
    EmbeddingCache,
    GpuLedger,
    StageCost,
    VramRefused,
    run_bf16_16gb,
)
from canvas.pipeline.pipeline import Pipeline
from canvas.pipeline.vram_probe import PeakVramProbe
from canvas.pipeline.worker import handle_message
from canvas.safety.image import ScreenedInput
from tests.fakes import World, rgb

_CAP_GB = 14.0
_TEXT_STAGES = (
    "encode-gpu-start",
    "encode-gpu-end",
    "encode-free",
    "denoise-start",
    "denoise-end",
    "transformer-free",
    "decode-start",
    "decode-end",
    "vae-free",
)


def _marks(probe: PeakVramProbe) -> list[str]:
    return [stage for stage, _used in probe.marks]


def _run(ledger: GpuLedger, mode: Mode, width: int, height: int, *, prompt: str = "a red boat",
         seed: int = 0, cache: EmbeddingCache | None = None, cost: StageCost | None = None,
         probe: PeakVramProbe | None = None):
    if cache is None:
        cache = EmbeddingCache()
    if probe is None:
        probe = PeakVramProbe(sample=lambda: int(round(ledger.used_gb * 1024)))
    trace = run_bf16_16gb(
        prompt=prompt,
        mode=mode,
        width=width,
        height=height,
        ledger=ledger,
        cache=cache,
        probe=probe,
        cost=cost,
        seed=seed,
    )
    return trace, probe


def _assert_inside(ledger: GpuLedger, probe: PeakVramProbe) -> None:
    cap = ledger.total_gb - headroom_gb(ledger.total_gb)
    assert ledger.peak_gb <= cap + 1e-9
    assert all(sample <= cap + 1e-9 for sample in ledger.history)
    assert ledger.held == {}
    for _stage, used in probe.marks:
        assert used is not None and used <= cap * 1024 + 1


def test_text_at_1mp_and_2mp_stays_under_the_headroom():
    frames = ((1024, 1024), (1440, 1440), (1920, 1088))
    for width, height in frames:
        ledger = GpuLedger(16)
        trace, probe = _run(ledger, Mode.TEXT, width, height)
        assert trace.device == "gpu" and trace.tiled is False and trace.control is False
        assert trace.peak_gb <= _CAP_GB
        assert ledger.used_gb == 0
        assert _marks(probe) == list(_TEXT_STAGES)
        _assert_inside(ledger, probe)
        assert "control-load" not in _marks(probe)


def test_inpaint_and_guided_modes_load_lite_control_and_stay_under_14gb():
    guided = (Mode.INPAINT, Mode.OUTPAINT, Mode.POSE, Mode.DEPTH, Mode.EDGE)
    for mode in guided:
        for width, height in ((1024, 1024), (1440, 1440), (1920, 1920)):
            ledger = GpuLedger(16)
            trace, probe = _run(ledger, mode, width, height)
            assert trace.control is True and trace.device == "gpu"
            assert trace.peak_gb <= _CAP_GB
            assert ledger.used_gb == 0
            marks = _marks(probe)
            assert marks.index("control-load") < marks.index("denoise-start")
            assert marks.index("transformer-free") < marks.index("control-free")
            assert marks.index("control-free") < marks.index("decode-start")
            _assert_inside(ledger, probe)
    ledger = GpuLedger(16)
    _trace, probe = _run(ledger, Mode.RESTYLE, 1440, 1440)
    assert "control-load" not in _marks(probe)
    _assert_inside(ledger, probe)


def test_a_new_seed_reuses_the_prompt_embedding():
    ledger = GpuLedger(16)
    cache = EmbeddingCache()
    probe = PeakVramProbe()
    _run(ledger, Mode.TEXT, 1024, 1024, seed=1, cache=cache, probe=probe)
    _run(ledger, Mode.TEXT, 1024, 1024, seed=99, cache=cache, probe=probe)
    _run(ledger, Mode.TEXT, 1024, 1024, prompt="a blue boat", seed=99, cache=cache, probe=probe)
    marks = _marks(probe)
    assert marks.count("encode-gpu-start") == 2
    assert marks.count("encode-cache-hit") == 1
    assert len(cache) == 2
    assert ledger.used_gb == 0
    assert ledger.peak_gb <= _CAP_GB


def test_encoder_falls_back_to_cpu_when_the_budget_is_short():
    ledger = GpuLedger(16, used_gb=10)
    cost = StageCost(
        encoder_gb=5.0,
        transformer_gb=3.0,
        control_lite_gb=0.2,
        vae_weight_gb=0.2,
        denoise_per_mp_gb=0.0,
        vae_full_per_mp_gb=0.0,
        vae_tiled_gb=0.1,
    )
    trace, probe = _run(ledger, Mode.TEXT, 1024, 1024, cost=cost)
    assert trace.device == "cpu"
    assert _marks(probe)[:2] == ["encode-cpu-start", "encode-cpu-end"]
    assert "encode-gpu-start" not in _marks(probe)
    assert ledger.used_gb == 10
    assert ledger.held == {}
    assert ledger.peak_gb <= _CAP_GB


def test_a_transformer_that_does_not_fit_frees_the_gpu():
    ledger = GpuLedger(16, used_gb=10)
    cost = StageCost(encoder_gb=5.0, transformer_gb=12.0, denoise_per_mp_gb=0.0)
    with pytest.raises(VramRefused) as caught:
        _run(ledger, Mode.TEXT, 1024, 1024, cost=cost)
    assert str(caught.value) == "This picture does not fit on the GPU."
    assert ledger.used_gb == 10
    assert ledger.held == {}

    guided = GpuLedger(16, used_gb=4)
    guided_cost = StageCost(
        encoder_gb=1.0, transformer_gb=12.0, control_lite_gb=1.0, denoise_per_mp_gb=0.0, vae_weight_gb=0.2,
    )
    probe = PeakVramProbe()
    with pytest.raises(VramRefused):
        _run(guided, Mode.INPAINT, 1024, 1024, cost=guided_cost, probe=probe)
    assert guided.used_gb == 4 and guided.held == {}
    marks = _marks(probe)
    assert "control-load" in marks and "control-free" in marks
    assert "denoise-start" not in marks


def test_vae_tiles_when_a_full_decode_exceeds_the_budget_and_refuses_when_a_tile_does_too():
    ledger = GpuLedger(16)
    cost = StageCost(vae_full_per_mp_gb=30.0)
    trace, probe = _run(ledger, Mode.TEXT, 1024, 1024, cost=cost)
    assert trace.tiled is True
    assert "decode-tiled" in _marks(probe)
    assert ledger.used_gb == 0
    _assert_inside(ledger, probe)

    blocked = GpuLedger(16)
    huge = StageCost(
        encoder_gb=1.0,
        transformer_gb=1.0,
        denoise_per_mp_gb=0.0,
        vae_weight_gb=20.0,
        vae_full_per_mp_gb=0.0,
        vae_tiled_gb=20.0,
    )
    with pytest.raises(VramRefused):
        _run(blocked, Mode.TEXT, 1024, 1024, cost=huge)
    assert blocked.used_gb == 0 and blocked.held == {}


def test_the_40gb_headroom_cap_still_drives_the_encoder():
    ledger = GpuLedger(40)
    assert ledger.room_gb() == pytest.approx(37.0)
    cost = StageCost(
        encoder_gb=37.5,
        transformer_gb=1.0,
        denoise_per_mp_gb=0.0,
        vae_weight_gb=0.1,
        vae_full_per_mp_gb=0.0,
        vae_tiled_gb=0.05,
    )
    trace, _probe = _run(ledger, Mode.TEXT, 1024, 1024, cost=cost)
    assert trace.device == "cpu"
    assert ledger.used_gb == 0
    assert ledger.peak_gb <= 40 - headroom_gb(40) + 1e-9


def test_the_generator_schedules_16gb_bf16_then_stays_a_stub():
    world = World()
    catalog = replace(load_catalog(), default_text_base="z-image-turbo")
    screened = world.gate().release_request("a red boat", Mode.TEXT.value, ())
    generator = ZImageGenerator(catalog, Tier.GB16)
    probe = PeakVramProbe(sample=lambda: int(round(generator.ledger.used_gb * 1024)))
    generator.attach_probe(probe)
    with pytest.raises(ModelNotReady) as caught:
        generator.generate(screened, 1024, 1024)
    assert caught.value.plan.quant == "bf16"
    assert caught.value.plan.text_encoder_offload == "free-after-encode"
    assert _marks(probe) == list(_TEXT_STAGES)
    assert generator.ledger.used_gb == 0
    assert generator.ledger.peak_gb <= _CAP_GB
    with pytest.raises(ModelNotReady):
        generator.generate(screened, 1024, 1024)
    assert _marks(probe).count("encode-cache-hit") == 1
    generator.unload()
    assert generator.unloaded is True
    assert len(generator.embeddings) == 1
    assert generator.ledger.held == {}


def test_24gb_bf16_skips_the_16gb_sequence():
    world = World()
    screened = world.gate().release_request("a red boat", Mode.TEXT.value, ())
    roomy = ZImageGenerator(load_catalog(), Tier.GB24)
    roomy_probe = PeakVramProbe()
    roomy.attach_probe(roomy_probe)
    with pytest.raises(ModelNotReady) as caught:
        roomy.generate(screened, 1024, 1024)
    assert caught.value.plan.quant == "bf16" and caught.value.plan.text_encoder_offload == ""
    assert roomy_probe.marks == []


def test_the_pipeline_refuses_a_picture_that_does_not_fit():
    world = World()
    catalog = replace(load_catalog(), default_text_base="z-image-turbo")
    generator = ZImageGenerator(catalog, Tier.GB16, ledger=GpuLedger(16, used_gb=15))
    pipeline = Pipeline(world.gate(), generator, Tier.GB16, catalog=catalog)
    probe = PeakVramProbe()
    with pytest.raises(VramRefused) as caught:
        pipeline.generate("a red boat", Mode.TEXT, probe=probe)
    assert str(caught.value) == "This picture does not fit on the GPU."
    assert "ZImage" not in str(caught.value)
    assert generator.ledger.used_gb == 15 and generator.ledger.held == {}
    assert _marks(probe)[0] == "generate-start"
    assert "encode-cpu-start" in _marks(probe)
    assert "generate-end" not in _marks(probe)


def test_the_worker_returns_the_fit_refusal():
    world = World()

    class _Refuse:
        def generate(self, screened: ScreenedInput) -> None:
            del screened
            raise VramRefused()

    result = handle_message({"prompt": "a red boat", "mode": "text"}, world.gate(), _Refuse())
    assert result["ok"] is False
    assert result["message"] == "This picture does not fit on the GPU."
    assert "data_b64" not in result


class _Enlarge:
    def installed(self) -> bool:
        return True

    def enlarge(self, source, *, output, format_name="avif", long_side=3840, on_event=None):
        del source, format_name, on_event
        write_rgb(output, 8, 8, bytes((4, 5, 6)) * 64)
        return type("Result", (), {"output": output})()


class _Scheduled:
    """Runs the real 16 GB schedule, then returns pixels so the 4K chain can unload."""

    def __init__(self, catalog) -> None:
        self.inner = ZImageGenerator(catalog, Tier.GB16)
        self.order: list[str] = []
        self.used_at_unload = -1.0

    def attach_probe(self, probe: PeakVramProbe | None) -> None:
        self.inner.attach_probe(probe)

    def generate(self, screened: ScreenedInput, width: int, height: int):
        self.order.append("generate")
        try:
            self.inner.generate(screened, width, height)
        except ModelNotReady:
            pass
        return rgb(8, 8, color=(1, 2, 3))

    def unload(self) -> None:
        self.used_at_unload = self.inner.ledger.used_gb
        self.order.append("unload")
        self.inner.unload()


def test_fourk_still_unloads_after_the_16gb_schedule(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_GPU_LOCK", str(tmp_path / "voxprint-gpu.lock"))
    catalog = replace(load_catalog(), default_text_base="z-image-turbo")
    world = World()
    generator = _Scheduled(catalog)
    probe = PeakVramProbe(sample=lambda: int(round(generator.inner.ledger.used_gb * 1024)))
    pipeline = Pipeline(world.gate(), generator, Tier.GB16, use_gpu_lock=True, catalog=catalog, enlarge=_Enlarge())
    image = pipeline.generate("a red boat", Mode.TEXT, aspect="1:1", output="4k", probe=probe)
    assert generator.order == ["generate", "unload"]
    assert generator.used_at_unload == 0
    assert generator.inner.unloaded is True
    assert generator.inner.ledger.peak_gb <= _CAP_GB
    assert image.width == 8 and image.height == 8
    marks = _marks(probe)
    assert marks[0] == "generate-start"
    assert "encode-free" in marks and "transformer-free" in marks and "vae-free" in marks
    assert marks.index("vae-free") < marks.index("generate-end")
    assert marks.index("generate-end") < marks.index("unload")
    assert marks[-2:] == ["upscale-start", "upscale-end"]
