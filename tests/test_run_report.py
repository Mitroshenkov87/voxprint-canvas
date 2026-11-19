"""Timing tags and the CLI version stamp."""
from __future__ import annotations

import pytest

from canvas.app.identity import splash_line
from canvas.infra.stage_timer import StageTimes


def test_timing_line_carries_tags() -> None:
    clock = StageTimes()
    clock.add("render", 1.0)
    clock.tag("tier", "gb16")
    clock.tag("weights", "fp8")
    line = clock.log_line()
    assert "tier=gb16" in line and "weights=fp8" in line


def test_tags_are_codes() -> None:
    with pytest.raises(ValueError):
        StageTimes().tag("tier", "16 GB")


def test_cli_version_is_ascii() -> None:
    assert splash_line({}).isascii()


def test_error_stop_rows_carry_the_error_code(tmp_path) -> None:
    pc = pytest.importorskip("canvas.app.photoshoot_cli")

    class _Scenario:
        shots = ("a", "b")

    rows = pc._shot_rows(_Scenario(), {0: "blocked"}, (), "internal")
    assert [row["code"] for row in rows] == ["internal", "internal"]
    assert all("blocked" not in str(row["message"]).lower() for row in rows)
    rows = pc._shot_rows(_Scenario(), {0: "blocked"}, ())
    assert rows[0]["code"] == "not_produced"
