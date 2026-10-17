"""16 GB schedule selection."""
from canvas.app.main import _parser
from canvas.app.platform import check_gpu
from canvas.infra.gpu import GpuInfo
from canvas.infra.vram import get_vram_budget_gb, set_vram_cap_gb
from canvas.models.modes import Mode, Tier, select_tier
from canvas.pipeline.bf16_schedule import EmbeddingCache, GpuLedger, group_offload_cost, run_bf16_16gb
from canvas.pipeline.schedule import PEAK_TARGET_GB, Placement, choose_schedule

_FRAMES = ((2048, 1152), (2048, 2048))


def test_budget_under_24_gb_selects_group_offload_and_24_stays_resident():
    text = choose_schedule(16, Mode.TEXT, ram_gb=64)
    assert text.placement is Placement.GROUP_OFFLOAD
    assert text.peak_target_gb == PEAK_TARGET_GB
    assert choose_schedule(23.49, Mode.INPAINT, ram_gb=64).placement is Placement.GROUP_OFFLOAD
    resident = choose_schedule(23.5, Mode.TEXT, ram_gb=64)
    assert resident.placement is Placement.RESIDENT
    assert choose_schedule(24, Mode.RESTYLE, ram_gb=16).placement is Placement.RESIDENT


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
