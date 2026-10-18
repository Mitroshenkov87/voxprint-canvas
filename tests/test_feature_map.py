"""The feature map records every moved tool, including the removed editor."""
from pathlib import Path


def test_feature_map_keeps_picture_tools_and_records_the_removal() -> None:
    page = Path("docs/ui/FEATURE-MAP.md").read_text(encoding="utf-8")
    folded = page.casefold()
    for phrase in (
        "img2img",
        "inpaint",
        "structure guidance",
        "removed (owner decision 2026-10-17)",
        "manage models",
        "needs owner decision",
        "4k",
        "selftest",
        "pending yitro",
        "source-picture slot",
        "mask slot",
        "shell.steps",
        "shell.guidance",
        "shell.sampler",
        "shell.seed",
        "shell.resolution",
    ):
        assert phrase in folded
    assert "temporarily unavailable" not in folded
    assert "edit.unavailable" not in folded
