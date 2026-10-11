"""OS and GPU gates, and the installer tier hook."""
from pathlib import Path

from canvas.app.platform import check_gpu, check_platform, linux_year, read_os_release
from canvas.infra.gpu import GpuInfo
from canvas.models.modes import Tier
from canvas.models.tiers import main as tier_main


def test_windows_and_linux_years():
    assert check_platform(system="Windows", windows_build=26100, linux_release=None).ok
    assert check_platform(system="Windows", windows_build=26099, linux_release=None).ok is False
    assert check_platform(system="Windows", windows_build=None, linux_release=None).ok is False
    assert linux_year({"ID": "ubuntu", "VERSION_ID": "25.04"}) == 2025
    assert linux_year({"ID": "ubuntu", "VERSION_ID": "24.04"}) == 2024
    assert linux_year({"ID": "fedora", "VERSION_ID": "42"}) == 2025
    assert linux_year({"ID": "fedora", "VERSION_ID": "41"}) == 2024
    assert linux_year({"ID": "debian", "VERSION_ID": "13"}) is None
    assert check_platform(system="Linux", windows_build=None, linux_release={"ID": "ubuntu", "VERSION_ID": "25.04"}).ok
    assert check_platform(system="Linux", windows_build=None, linux_release={"ID": "ubuntu", "VERSION_ID": "24.04"}).ok is False
    assert check_platform(system="Linux", windows_build=None, linux_release={}).ok is False
    assert check_platform(system="Darwin", windows_build=None, linux_release=None).ok is False


def test_gpu_tier_requirements():
    refused = check_gpu(None)
    assert refused.ok is False and refused.tier is None
    old = check_gpu(GpuInfo("old", 24, 20, 8, 6, "fake"))
    assert old.ok is False
    small = check_gpu(GpuInfo("small", 12, 8, 8, 9, "fake"))
    assert small.ok is False
    mid = check_gpu(GpuInfo("mid", 16, 12, 8, 9, "fake"))
    assert mid.ok and mid.tier is Tier.GB16
    large = check_gpu(GpuInfo("large", 24, 20, 8, 9, "fake"))
    assert large.tier is Tier.GB24
    reported = check_gpu(GpuInfo("4090", 23.99, 20, 8, 9, "fake"))
    assert reported.ok and reported.tier is Tier.GB24
    sixteen = check_gpu(GpuInfo("16", 15.5, 12, 8, 9, "fake"))
    assert sixteen.ok and sixteen.tier is Tier.GB16
    under = check_gpu(GpuInfo("15", 15.49, 10, 8, 9, "fake"))
    assert under.ok is False


def test_os_release_parser(tmp_path: Path):
    path = tmp_path / "os-release"
    path.write_text('ID="ubuntu"\nVERSION_ID="25.04"\n# comment\n', encoding="utf-8")
    assert read_os_release(path)["VERSION_ID"] == "25.04"


def test_installer_selects_a_tier_and_shares_the_voxprint_group(monkeypatch, capsys):
    monkeypatch.setenv("VOXPRINT_FAKE_VRAM", "RTX 4090,24,18,8.9")
    assert tier_main() == 0
    assert capsys.readouterr().out.strip() == "24gb"
    iss = Path("installer/windows/voxprint-canvas.iss").read_text(encoding="utf-8")
    assert "MinVersion=10.0.26100" in iss
    assert "DefaultGroupName={#MySuiteName}" in iss
    assert 'MySuiteName "VOXPrint AI Suite"' in iss
    assert r"DefaultDirName={autopf}\{#MySuiteName}\Canvas" in iss
    assert "python -m canvas.models.tiers" in iss
    assert "default image pack" in iss
    windows = Path("installer/windows/install.ps1").read_text(encoding="utf-8")
    assert "canvas.models.tiers" in windows
    assert "canvas.app.provision" in windows
    assert "Checking Python..." in windows
    assert "uv.exe" in windows
    assert "could not install Python" in windows
    script = Path("installer/linux/install.sh").read_text(encoding="utf-8")
    assert "voxprint.directory" in script
    assert "Name=VOXPrint AI Suite" in script
    assert 'Name=VOXPrint AI Canvas' in script
    assert "/opt/voxprint-ai-suite/canvas" in script
    assert "canvas.models.tiers" in script
    assert "canvas.app.provision" in script
    assert "Checking Python..." in script
