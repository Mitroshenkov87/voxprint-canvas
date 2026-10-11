"""The age-check fixtures stay on clothed faces, and adult figures stay clearly adult.

The measurement job scores the face crops. This file only checks the manifest,
so a swimsuit or a young-looking figure fails the default test job.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOUNG = ROOT / "tests" / "fixtures" / "adult_20_25"
ADULT = ROOT / "tests" / "fixtures" / "adult_cc0"
_IMAGES = {".jpg", ".jpeg", ".png", ".webp"}


def _images(folder: Path) -> set[str]:
    return {path.name for path in folder.iterdir() if path.suffix.lower() in _IMAGES}


def test_young_adult_fixtures_are_clothed_face_crops():
    manifest = json.loads((YOUNG / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) >= 20
    assert {int(item["age"]) for item in manifest} >= set(range(20, 26))
    names: set[str] = set()
    for item in manifest:
        assert item["presentation"] == "face-crop"
        assert item["attire"] == "clothed"
        assert item["license"] in {"Public domain", "CC0"}
        age = int(item["age"])
        assert 20 <= age <= 25
        assert (YOUNG / item["file"]).is_file()
        names.add(item["file"])
    assert _images(YOUNG) == names


def test_adult_figure_fixtures_are_clearly_adult():
    manifest = json.loads((ADULT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest
    names: set[str] = set()
    for item in manifest:
        assert item["license"] in {"Public domain", "CC0"}
        age = item["documented_age"]
        if age is None:
            assert item["age_basis"] == "Category:Nude old men"
        else:
            assert int(age) >= 25
        assert (ADULT / item["file"]).is_file()
        names.add(item["file"])
    assert _images(ADULT) == names
