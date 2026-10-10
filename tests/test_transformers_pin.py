"""The generation extra and the bake-off pin stay above the transformers CVE floor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLOOR = "transformers>=5.14.0,<6"
CVE = "CVE-2026-9856"


def test_transformers_pin_names_the_cve_floor():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    bakeoff = (ROOT / "bakeoff" / "requirements.txt").read_text(encoding="utf-8")
    assert FLOOR in pyproject
    assert CVE in pyproject
    assert "diffusers==0.41.0" in pyproject
    assert FLOOR in bakeoff
    assert CVE in bakeoff
