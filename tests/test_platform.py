"""OS and GPU gates, and the installer tier hook."""
from pathlib import Path

from canvas.app.platform import (
    ALLOW_UNSUPPORTED_DISTRO_VAR,
    check_gpu,
    check_platform,
    linux_year,
    read_os_release,
)
from canvas.infra.gpu import GpuInfo
from canvas.models.modes import Tier
from canvas.models.tiers import main as tier_main


def test_windows_and_linux_years(monkeypatch):
    monkeypatch.delenv(ALLOW_UNSUPPORTED_DISTRO_VAR, raising=False)
    assert check_platform(system="Windows", windows_build=26100, linux_release=None).ok
    assert check_platform(system="Windows", windows_build=26099, linux_release=None).ok is False
    assert check_platform(system="Windows", windows_build=None, linux_release=None).ok is False
    assert linux_year({"ID": "ubuntu", "VERSION_ID": "25.04"}) == 2025
    assert linux_year({"ID": "ubuntu", "VERSION_ID": "24.04"}) == 2024
    assert linux_year({"ID": "fedora", "VERSION_ID": "42"}) == 2025
    assert linux_year({"ID": "fedora", "VERSION_ID": "41"}) == 2024
    assert linux_year({"ID": "debian", "VERSION_ID": "13"}) is None
    assert check_platform(system="Linux", windows_build=None, linux_release={"ID": "ubuntu", "VERSION_ID": "25.04"}).ok
    assert check_platform(system="Linux", windows_build=None, linux_release={"ID": "ubuntu", "VERSION_ID": "24.04"}).ok
    refused = check_platform(system="Linux", windows_build=None, linux_release={"ID": "ubuntu", "VERSION_ID": "22.04"})
    assert refused.ok is False and "2024 or later" in refused.message
    assert check_platform(system="Linux", windows_build=None, linux_release={}).ok is False
    assert check_platform(system="Darwin", windows_build=None, linux_release=None).ok is False


def test_allow_unsupported_distro_override(monkeypatch):
    monkeypatch.setenv(ALLOW_UNSUPPORTED_DISTRO_VAR, "1")
    old = check_platform(system="Linux", windows_build=None, linux_release={"ID": "ubuntu", "VERSION_ID": "22.04"})
    assert old.ok
    assert "VOXPRINT_ALLOW_UNSUPPORTED_DISTRO" in old.message
    assert "unsupported" in old.message
    assert "continuing anyway" in old.message.lower()
    unknown = check_platform(system="Linux", windows_build=None, linux_release={"ID": "debian", "VERSION_ID": "13"})
    assert unknown.ok and unknown.message
    empty = check_platform(system="Linux", windows_build=None, linux_release={})
    assert empty.ok and empty.message
    # The override is Linux-only: Windows and macOS checks are unchanged.
    assert check_platform(system="Windows", windows_build=26099, linux_release=None).ok is False
    assert check_platform(system="Darwin", windows_build=None, linux_release=None).ok is False


def test_allow_unsupported_distro_value_must_be_exactly_one(monkeypatch):
    for value in ("", "0", "true", "yes", "01"):
        monkeypatch.setenv(ALLOW_UNSUPPORTED_DISTRO_VAR, value)
        assert check_platform(system="Linux", windows_build=None, linux_release={}).ok is False
    # Whitespace around the value is stripped before the comparison.
    monkeypatch.setenv(ALLOW_UNSUPPORTED_DISTRO_VAR, " 1 ")
    allowed = check_platform(system="Linux", windows_build=None, linux_release={})
    assert allowed.ok and "VOXPRINT_ALLOW_UNSUPPORTED_DISTRO" in allowed.message


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
    assert "canvas.models.install_fetch" in windows
    assert "e5834ce279c0b3ba6dc5cae527d6270dbdcd5c233120b45a6819bd16c5e88eab" in windows
    assert "Checking Python..." in windows
    assert "uv.exe" in windows
    assert "could not install Python" in windows
    script = Path("installer/linux/install.sh").read_text(encoding="utf-8")
    assert "voxprint.directory" in script
    assert "Name=VOXPrint AI Suite" in script
    assert 'Name=VOXPrint AI Canvas' in script
    assert "/opt/voxprint-ai-suite/canvas" in script
    assert "canvas.models.tiers" in script
    assert "canvas.models.install_fetch" in script
    assert "e5834ce279c0b3ba6dc5cae527d6270dbdcd5c233120b45a6819bd16c5e88eab" in script
    assert "Checking Python..." in script


ANDUINOS_2_0_5 = {
    "PRETTY_NAME": "AnduinOS 2.0.5",
    "NAME": "AnduinOS",
    "VERSION_ID": "2.0.5",
    "VERSION": "2.0.5 (Resolute Ringtail)",
    "VERSION_CODENAME": "resolute",
    "ID": "anduinos",
    "ID_LIKE": "debian",
    "UBUNTU_CODENAME": "resolute",
}


def test_ubuntu_derivatives_use_their_ubuntu_base(monkeypatch, tmp_path):
    monkeypatch.delenv(ALLOW_UNSUPPORTED_DISTRO_VAR, raising=False)
    assert linux_year(ANDUINOS_2_0_5) == 2026
    result = check_platform(system="Linux", windows_build=None, linux_release=ANDUINOS_2_0_5)
    assert result.ok and result.message == ""
    path = tmp_path / "os-release"
    path.write_text("\n".join(f'{k}="{v}"' for k, v in ANDUINOS_2_0_5.items()) + "\n", encoding="utf-8")
    assert check_platform(system="Linux", windows_build=None, linux_release=read_os_release(path)).ok
    # Mint and Pop!_OS name their base too; an old base is still refused.
    assert linux_year({"ID": "linuxmint", "ID_LIKE": "ubuntu debian", "VERSION_ID": "22", "UBUNTU_CODENAME": "noble"}) == 2024
    assert linux_year({"ID": "linuxmint", "ID_LIKE": "ubuntu debian", "VERSION_ID": "21.3", "UBUNTU_CODENAME": "jammy"}) == 2022
    assert check_platform(
        system="Linux", windows_build=None,
        linux_release={"ID": "zorin", "ID_LIKE": "ubuntu debian", "VERSION_ID": "16", "UBUNTU_CODENAME": "focal"},
    ).ok is False
    # VERSION_CODENAME counts only when ID_LIKE says ubuntu.
    assert linux_year({"ID": "pop", "ID_LIKE": "ubuntu debian", "VERSION_ID": "", "VERSION_CODENAME": "noble"}) == 2024
    assert linux_year({"ID": "ubuntu", "VERSION_CODENAME": "plucky"}) == 2025
    # A codename newer than the table counts as the newest known release.
    assert linux_year({"ID": "someos", "VERSION_ID": "3.1", "UBUNTU_CODENAME": "stoic"}) == 2026
    # Ubuntu's own YY.MM still wins over its codename.
    assert linux_year({"ID": "ubuntu", "VERSION_ID": "24.04", "UBUNTU_CODENAME": "noble"}) == 2024
    # Plain Debian and unknown releases are unchanged.
    assert linux_year({"ID": "debian", "VERSION_ID": "13", "VERSION_CODENAME": "trixie"}) is None
    assert linux_year({"ID": "anduinos", "ID_LIKE": "debian", "VERSION_ID": "2.0.5"}) is None
