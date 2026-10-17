"""16 GB schedule selection, the RAM gate, and the disabled INT4 slot."""
from pathlib import Path

import pytest

from canvas.app.main import _parser
from canvas.app.platform import check_gpu
from canvas.infra.gpu import GpuInfo
from canvas.infra.vram import get_vram_budget_gb, set_vram_cap_gb
from canvas.models.loader import ModelNotReady, ZImageGenerator
from canvas.models.modes import Mode, Tier, select_tier
from canvas.models.nunchaku import NUNCHAKU_ID, NunchakuUnavailable, load_nunchaku_generator, nunchaku_ready
from canvas.models.registry import load_catalog
from canvas.models.snapshots import SnapshotError, download_snapshot
from canvas.pipeline.bf16_schedule import EmbeddingCache, GpuLedger, group_offload_cost, run_bf16_16gb
from canvas.pipeline.schedule import (
    PEAK_TARGET_GB,
    Placement,
    SystemMemoryRefused,
    choose_schedule,
    edit_group_peak_gb,
    run_edit_16gb,
    system_ram_gb,
)
from canvas.pipeline.vram_probe import PeakVramProbe
from canvas.safety.inputs import ImageRole, PendingInput
from tests.fakes import World, rgb

_FRAMES = ((2048, 1152), (2048, 2048))


def test_budget_under_24_gb_selects_group_offload_and_24_stays_resident():
    text = choose_schedule(16, Mode.TEXT, ram_gb=64)
    assert text.placement is Placement.GROUP_OFFLOAD
    assert text.peak_target_gb == PEAK_TARGET_GB
    assert choose_schedule(23.49, Mode.INPAINT, ram_gb=64).placement is Placement.GROUP_OFFLOAD
    resident = choose_schedule(23.5, Mode.TEXT, ram_gb=64)
    assert resident.placement is Placement.RESIDENT
    assert choose_schedule(24, Mode.RESTYLE, ram_gb=16).placement is Placement.RESIDENT


def test_qwen_edit_refuses_under_40_gb_and_names_the_requirement():
    with pytest.raises(SystemMemoryRefused, match="40 GB") as caught:
        choose_schedule(16, Mode.INSTRUCTION, ram_gb=31.2)
    assert "31 GB" in str(caught.value)
    assert caught.value.need_gb == 40
    with pytest.raises(SystemMemoryRefused, match="40 GB"):
        choose_schedule(24, Mode.INSTRUCTION, ram_gb=39.9)
    allowed = choose_schedule(16, Mode.INSTRUCTION, ram_gb=40)
    assert allowed.placement is Placement.GROUP_OFFLOAD
    assert choose_schedule(24, Mode.INSTRUCTION, ram_gb=64).placement is Placement.RESIDENT
    assert system_ram_gb() > 0


def test_group_offload_peaks_stay_within_14_5_gb_at_2048():
    for width, height in _FRAMES:
        for mode in (Mode.TEXT, Mode.RESTYLE, Mode.INPAINT):
            ledger = GpuLedger(16)
            trace = run_bf16_16gb(
                prompt="a red boat",
                mode=mode,
                width=width,
                height=height,
                ledger=ledger,
                cache=EmbeddingCache(),
                cost=group_offload_cost(),
            )
            assert trace.peak_gb <= PEAK_TARGET_GB
            assert ledger.held == {}
            assert ledger.used_gb == 0
        edit = run_edit_16gb(width=width, height=height, ledger=GpuLedger(16))
        assert edit.tiled is True
        assert edit.peak_gb <= PEAK_TARGET_GB
        assert edit.peak_gb == pytest.approx(edit_group_peak_gb(width, height))


def test_vram_cap_selects_the_16_gb_tier(monkeypatch):
    assert get_vram_budget_gb(24) == 24
    monkeypatch.setenv("CANVAS_VRAM_CAP_GB", "16")
    assert get_vram_budget_gb(24) == 16
    assert select_tier(24) is Tier.GB16
    assert select_tier(23.99) is Tier.GB16
    checked = check_gpu(GpuInfo("RTX 4090", 24, 20, 8, 9, "fake"))
    assert checked.ok and checked.tier is Tier.GB16
    monkeypatch.setenv("CANVAS_VRAM_CAP_GB", "nope")
    assert select_tier(24) is Tier.GB24
    set_vram_cap_gb(16)
    try:
        assert get_vram_budget_gb(48) == 16
        assert select_tier(48) is Tier.GB16
    finally:
        set_vram_cap_gb(None)
    args = _parser().parse_args(["--vram-cap-gb", "16", "generate", "--prompt", "a red boat", "--out", "out.avif"])
    assert args.vram_cap_gb == 16.0


def test_int4_slot_is_disabled_and_is_not_downloaded(tmp_path: Path):
    spec = load_catalog().by_id(NUNCHAKU_ID)
    assert spec.status == "disabled"
    assert spec.licence == "Apache-2.0"
    assert spec.quant == "int4"
    assert spec.quant_backend == "nunchaku-svdquant"
    assert spec.vram_hint_gb == 12
    assert spec.variant == "r128"
    assert spec.pinned()
    assert spec.files["svdq-int4_r128-z-image-turbo.safetensors"] == (
        "dbf5aba0f7370d1fe88e52823d3d7b9ceb6fa006707f075baf9cfdaf362ea745"
    )
    assert nunchaku_ready(spec) is False
    assert choose_schedule(16, Mode.TEXT, ram_gb=64, nunchaku_ready=False).placement is Placement.GROUP_OFFLOAD
    assert choose_schedule(16, Mode.TEXT, ram_gb=64, nunchaku_ready=True).placement is Placement.NUNCHAKU_INT4
    assert choose_schedule(16, Mode.INSTRUCTION, ram_gb=64, nunchaku_ready=True).placement is Placement.GROUP_OFFLOAD
    with pytest.raises(NunchakuUnavailable, match="not available yet"):
        load_nunchaku_generator(spec)
    with pytest.raises(SnapshotError) as download_error:
        download_snapshot(spec, tmp_path)
    assert download_error.value.detail == "this model is not available for download"
    assert list(tmp_path.iterdir()) == []


def test_instruction_on_16_gb_checks_ram_before_it_schedules():
    catalog = load_catalog()
    world = World()
    screened = world.gate().release_request(
        "paint the door blue", Mode.INSTRUCTION.value, [PendingInput(ImageRole.INIT, rgb())],
    )
    short = ZImageGenerator(catalog, Tier.GB16, ledger=GpuLedger(16), ram_gb=12)
    probe = PeakVramProbe(sample=lambda: 0)
    short.attach_probe(probe)
    with pytest.raises(SystemMemoryRefused, match="40 GB") as caught:
        short.generate(screened, 1024, 1024)
    assert "12 GB" in str(caught.value)
    assert probe.marks == []
    assert short.ledger.used_gb == 0
    ready = ZImageGenerator(catalog, Tier.GB16, ledger=GpuLedger(16), ram_gb=64)
    edit_probe = PeakVramProbe(sample=lambda: int(round(ready.ledger.used_gb * 1024)))
    ready.attach_probe(edit_probe)
    with pytest.raises(ModelNotReady):
        ready.generate(screened, 2048, 1152)
    marks = [stage for stage, _used in edit_probe.marks]
    assert marks[0] == "encode-gpu-start"
    assert "encode-free" in marks
    assert marks.index("encode-free") < marks.index("denoise-start")
    assert "decode-tiled" in marks
    assert ready.ledger.peak_gb <= PEAK_TARGET_GB
    assert ready.ledger.held == {}
