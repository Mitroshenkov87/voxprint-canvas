"""The generation extra and the bake-off pin stay above the transformers CVE floor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = "transformers==5.19.0"
CVE = "CVE-2026-9856"


def test_transformers_pin_names_the_cve_floor():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    bakeoff = (ROOT / "bakeoff" / "requirements.txt").read_text(encoding="utf-8")
    constraints = (ROOT / "constraints.txt").read_text(encoding="utf-8")
    assert PIN in pyproject
    assert CVE in pyproject
    assert "5.14" in pyproject
    assert "diffusers==0.41.0" in pyproject
    assert PIN in bakeoff
    assert CVE in bakeoff
    assert PIN in constraints
    assert "torch==2.11.0" in constraints
    assert "torchvision==0.26.0" in constraints
    for script in (
        ROOT / "installer" / "linux" / "install.sh",
        ROOT / "installer" / "windows" / "install.ps1",
    ):
        text = script.read_text(encoding="utf-8")
        assert "constraints.txt" in text
        assert "-c" in text
