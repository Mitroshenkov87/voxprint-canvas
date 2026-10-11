"""The tag workflow publishes a pre-release with checksums. It does not embed weights."""
from pathlib import Path


def test_release_workflow_publishes_a_prerelease_from_the_build_file():
    text = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    assert 'tags:' in text
    assert '- "v*"' in text
    assert "workflow_dispatch:" in text
    assert "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')" in text
    assert 'name: SHA256SUMS' in text
    assert "--prerelease" in text
    assert "release_title" in text
    assert "SHA256SUMS" in text
    assert "gh release delete" in text
    assert "--cleanup-tag" not in text
    assert "voxprint-canvas.iss" in text
    assert "installer/linux/install.sh" in text
    assert "constraints.txt" in text
    assert "uv.exe" in text
    notes = Path("docs/release-notes/0.1.0-rc.1.md").read_text(encoding="utf-8")
    assert notes.startswith("New:")
    assert "Fixed:" in notes


def test_installers_fetch_gpu_libraries_and_the_default_pack():
    linux = Path("installer/linux/install.sh").read_text(encoding="utf-8")
    windows = Path("installer/windows/install.ps1").read_text(encoding="utf-8")
    iss = Path("installer/windows/voxprint-canvas.iss").read_text(encoding="utf-8")
    for script in (linux, windows):
        assert "https://download.pytorch.org/whl/cu130" in script
        assert "--register-models" in script
        assert "huggingface.co" not in script
        folded = script.casefold()
        assert "nunchaku" not in folded
        assert "qwen-image-edit" not in folded
    assert "install.ps1" in iss
    assert "voxprint-canvas.cmd" in iss
    assert "staging\\*" in iss


def test_release_ships_tools_and_vm_scripts():
    text = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "tools" in text
    assert "scripts/vm_test.sh" in text
    assert "scripts/vm_test.ps1" in text
    root = Path(__file__).resolve().parents[1]
    assert (root / "tools" / "vm_test.py").is_file()
    assert (root / "tools" / "__init__.py").is_file()
    assert (root / "scripts" / "vm_test.sh").is_file()
    assert (root / "scripts" / "vm_test.ps1").is_file()
