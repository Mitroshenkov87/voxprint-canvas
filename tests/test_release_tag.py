"""A release tag must name the version and rc in BUILD.json (rc.29 shipped saying rc.28)."""
from __future__ import annotations

import json
from pathlib import Path

from canvas.app.identity import check_release_tag, tag_mismatch

ROOT = Path(__file__).resolve().parents[1]
_RC = {"version": "0.1.0", "channel": "RC", "rc": 30, "build": 29}


def test_a_matching_rc_tag_passes() -> None:
    assert tag_mismatch("v0.1.0-rc.30", _RC) is None


def test_an_rc_tag_ahead_of_build_json_fails() -> None:
    reason = tag_mismatch("v0.1.0-rc.29", {**_RC, "rc": 28})
    assert reason is not None
    assert "rc 29" in reason and "rc 28" in reason


def test_a_version_mismatch_fails() -> None:
    reason = tag_mismatch("v0.2.0-rc.30", _RC)
    assert reason is not None
    assert "0.2.0" in reason and "0.1.0" in reason


def test_a_final_tag_needs_build_json_without_rc() -> None:
    assert tag_mismatch("v0.1.0", {"version": "0.1.0"}) is None
    assert tag_mismatch("v0.1.0", {"version": "0.1.0", "rc": 0}) is None
    assert tag_mismatch("v0.1.0", _RC) is not None
    assert tag_mismatch("v0.1.0-rc.30", {"version": "0.1.0"}) is not None


def test_a_tag_that_is_not_a_version_fails() -> None:
    for tag in ("", "v0.1", "0.1.0-rc.30", "v0.1.0-rc.", "v0.1.0-beta.1", "v0.1.0-rc.30x"):
        assert tag_mismatch(tag, _RC) is not None, tag
    assert tag_mismatch("v0.1.0-rc.30", {**_RC, "rc": True}) is not None


def test_every_build_json_is_checked(tmp_path: Path) -> None:
    good = tmp_path / "BUILD.json"
    stale = tmp_path / "packaged.json"
    good.write_text(json.dumps(_RC), encoding="utf-8")
    stale.write_text(json.dumps({**_RC, "rc": 29}), encoding="utf-8")
    assert check_release_tag("v0.1.0-rc.30", [good]) == []
    problems = check_release_tag("v0.1.0-rc.30", [good, stale])
    assert len(problems) == 1
    assert str(stale) in problems[0]


def test_the_repository_build_json_names_its_own_tag() -> None:
    data = json.loads((ROOT / "BUILD.json").read_text(encoding="utf-8"))
    tag = f"v{data['version']}-rc.{data['rc']}"
    assert check_release_tag(tag, [ROOT / "BUILD.json", ROOT / "canvas" / "app" / "BUILD.json"]) == []


def test_the_release_workflow_checks_the_tag_before_building() -> None:
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    jobs = text.split("\njobs:\n", 1)[1]
    assert jobs.startswith("  tag-guard:\n")
    guard = jobs.split("\n  pdf-guide:\n", 1)[0]
    assert "check_release_tag" in guard
    assert 'Path("BUILD.json"), Path("canvas/app/BUILD.json")' in guard
    assert "$GITHUB_REF_NAME" in guard
    assert "\n  pdf-guide:\n    needs: tag-guard\n" in text
    assert "needs: [tag-guard, pdf-guide, linux-tarball, windows-installer]" in text
