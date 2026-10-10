"""The tag workflow publishes a pre-release with checksums. It does not embed weights."""
from pathlib import Path


def test_release_workflow_publishes_a_prerelease_from_the_build_file():
    text = Path(".github/workflows/release.yml").read_text(encoding="utf-8")
    assert 'tags:' in text
    assert '- "v*"' in text
    assert "--prerelease" in text
    assert "release_title" in text
    assert "SHA256SUMS" in text
    assert "voxprint-canvas.iss" in text
    assert "installer/linux/install.sh" in text
    notes = Path("docs/release-notes/0.1.0-rc.1.md").read_text(encoding="utf-8")
    assert notes.startswith("New:")
    assert "Fixed:" in notes


def test_installers_fetch_gpu_libraries_and_leave_models_for_later():
    linux = Path("installer/linux/install.sh").read_text(encoding="utf-8")
    windows = Path("installer/windows/install.ps1").read_text(encoding="utf-8")
    iss = Path("installer/windows/voxprint-canvas.iss").read_text(encoding="utf-8")
    for script in (linux, windows):
        assert "https://download.pytorch.org/whl/cu128" in script
        assert "--register-models" in script
        assert "huggingface.co" not in script
    assert "install.ps1" in iss
    assert "voxprint-canvas.cmd" in iss
    assert "staging\\*" in iss
