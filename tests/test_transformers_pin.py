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
    assert "torch>=2.11,<2.15" in constraints
    assert "torchvision>=0.26,<0.30" in constraints
    cuda_constraints = (ROOT / "constraints-cuda.txt").read_text(encoding="utf-8")
    assert "torch==2.11.0+cu130" in cuda_constraints
    assert "torchvision==0.26.0+cu130" in cuda_constraints
    for script in (
        ROOT / "installer" / "linux" / "install.sh",
        ROOT / "installer" / "windows" / "install.ps1",
    ):
        text = script.read_text(encoding="utf-8")
        assert "constraints.txt" in text
        assert "-c" in text



def test_gen_torch_ranges_match_constraints_and_cover_the_cuda_default():
    import tomllib

    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    gen = pyproject["project"]["optional-dependencies"]["gen"]

    def extra_spec(name: str) -> str:
        for item in gen:
            if item == name or item.startswith(name) and not item[len(name):len(name)+1].isalpha():
                return item[len(name):]
        raise AssertionError(f"{name} missing from the [gen] extra")

    torch_spec = extra_spec("torch")
    vision_spec = extra_spec("torchvision")
    assert torch_spec == ">=2.11,<2.15"
    assert vision_spec == ">=0.26,<0.30"
    constraints = (ROOT / "constraints.txt").read_text(encoding="utf-8")
    assert f"torch{torch_spec}" in constraints
    assert f"torchvision{vision_spec}" in constraints
    cuda = (ROOT / "constraints-cuda.txt").read_text(encoding="utf-8")
    assert "torch==2.11.0+cu130" in cuda
    assert "torchvision==0.26.0+cu130" in cuda
    torch_set = SpecifierSet(torch_spec)
    vision_set = SpecifierSet(vision_spec)
    assert Version("2.11.0+cu130").public in torch_set
    assert Version("0.26.0+cu130").public in vision_set
    # Documented compatible newer wheels stay inside the range; the upper bound is exclusive.
    assert Version("2.14.1").public in torch_set
    assert Version("0.29.2").public in vision_set
    assert Version("2.15.0").public not in torch_set
    assert Version("0.30.0").public not in vision_set
