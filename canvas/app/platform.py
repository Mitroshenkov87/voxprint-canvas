"""OS and GPU requirements. Importing this module does not check the machine."""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from canvas.app.info import MIN_LINUX_YEAR, MIN_WINDOWS_BUILD
from canvas.infra.gpu import GpuInfo
from canvas.models.modes import Tier, select_tier

# Set to exactly "1" to run on an unsupported Linux distribution. The check then
# warns and continues instead of refusing. This override is Linux-only; it does
# not bypass the Windows 11 24H2 requirement and it does not accept macOS.
ALLOW_UNSUPPORTED_DISTRO_VAR = "VOXPRINT_ALLOW_UNSUPPORTED_DISTRO"


@dataclass(frozen=True)
class OsCheck:
    """Whether this operating system can run Canvas."""

    ok: bool
    message: str


@dataclass(frozen=True)
class GpuCheck:
    """Whether the GPU meets the tier rules."""

    ok: bool
    message: str
    tier: Tier | None


# Ubuntu release codenames and the year each one shipped. Derivatives
# (AnduinOS, Linux Mint, Pop!_OS, Zorin, elementary) report their own
# ``VERSION_ID`` but name the Ubuntu base in ``UBUNTU_CODENAME``.
UBUNTU_CODENAME_YEARS: dict[str, int] = {
    "precise": 2012, "quantal": 2012, "raring": 2013, "saucy": 2013,
    "trusty": 2014, "utopic": 2014, "vivid": 2015, "wily": 2015,
    "xenial": 2016, "yakkety": 2016, "zesty": 2017, "artful": 2017,
    "bionic": 2018, "cosmic": 2018, "disco": 2019, "eoan": 2019,
    "focal": 2020, "groovy": 2020, "hirsute": 2021, "impish": 2021,
    "jammy": 2022, "kinetic": 2022, "lunar": 2023, "mantic": 2023,
    "noble": 2024, "oracular": 2024, "plucky": 2025, "questing": 2025,
    "resolute": 2026,
}


def _clean(release: Mapping[str, str], key: str) -> str:
    return str(release.get(key) or "").strip().strip('"').strip("'")


def _ubuntu_based(release: Mapping[str, str]) -> bool:
    ident = _clean(release, "ID").casefold()
    like = _clean(release, "ID_LIKE").casefold().split()
    return ident == "ubuntu" or "ubuntu" in like or bool(_clean(release, "UBUNTU_CODENAME"))


def _ubuntu_codename_year(release: Mapping[str, str]) -> int | None:
    """Year of the Ubuntu base named by the codename, or None when none is named.

    ``UBUNTU_CODENAME`` wins; ``VERSION_CODENAME`` counts only on an Ubuntu-based
    system. The table holds every codename since 2012, so a codename it does not
    know is a release newer than the table and counts as the newest known year.
    """
    codename = _clean(release, "UBUNTU_CODENAME").casefold()
    if not codename and _ubuntu_based(release):
        codename = _clean(release, "VERSION_CODENAME").casefold()
    if not codename:
        return None
    return UBUNTU_CODENAME_YEARS.get(codename, max(UBUNTU_CODENAME_YEARS.values()))


def _version_year(version: str) -> int | None:
    head = version.split(".", 1)[0]
    if not head.isdigit():
        return None
    number = int(head)
    if number >= 2000:
        return number
    if 20 <= number <= 99:
        return 2000 + number
    return None


def linux_year(release: Mapping[str, str]) -> int | None:
    """Year of a Linux distro, or None when the release file does not say.

    Fedora 42 and newer count as 2025. Ubuntu's own ``YY.MM`` becomes 2000+YY.
    An Ubuntu derivative goes by its Ubuntu base (``UBUNTU_CODENAME``, or
    ``VERSION_CODENAME`` when ``ID_LIKE`` names ubuntu), because its own
    ``VERSION_ID`` (AnduinOS 2.0.5, Mint 22) is not a year. Otherwise a
    ``YY.MM`` or four-digit ``VERSION_ID`` is used. Anything else is None.
    """
    ident = _clean(release, "ID").casefold()
    version = _clean(release, "VERSION_ID")
    if ident == "fedora":
        head = version.split(".", 1)[0]
        if not head.isdigit():
            return None
        return 2025 if int(head) >= 42 else 2024
    if ident == "ubuntu" and version:
        year = _version_year(version)
        if year is not None:
            return year
    codename_year = _ubuntu_codename_year(release)
    if codename_year is not None:
        return codename_year
    if not version:
        return None
    return _version_year(version)


def _allow_unsupported_distro() -> bool:
    """True only for the exact value ``"1"`` (surrounding whitespace is ignored)."""
    return os.environ.get(ALLOW_UNSUPPORTED_DISTRO_VAR, "").strip() == "1"


def check_platform(*, system: str, windows_build: int | None, linux_release: Mapping[str, str] | None) -> OsCheck:
    """Windows 11 24H2+ or a 2024-or-later Linux. Other systems are refused.

    An unsupported Linux (unknown or pre-2024 release) still runs when
    ``VOXPRINT_ALLOW_UNSUPPORTED_DISTRO=1`` is set; the check then returns ok
    with a warning message instead of a refusal.
    """
    if system == "Windows":
        if windows_build is None or windows_build < MIN_WINDOWS_BUILD:
            return OsCheck(False, "Windows 11 24H2 or newer is required.")
        return OsCheck(True, "")
    if system == "Linux":
        year = linux_year(linux_release or {})
        if year is None or year < MIN_LINUX_YEAR:
            if _allow_unsupported_distro():
                return OsCheck(
                    True,
                    "Warning: this Linux distribution is unsupported; a distribution from "
                    "2024 or later is required. Continuing anyway because "
                    f"{ALLOW_UNSUPPORTED_DISTRO_VAR}=1 is set.",
                )
            return OsCheck(False, "A Linux distribution from 2024 or later is required.")
        return OsCheck(True, "")
    return OsCheck(False, "Windows 11 24H2 or a 2024 Linux distribution is required.")


def check_gpu(info: GpuInfo | None) -> GpuCheck:
    """RTX 40 series (compute capability 8.9) or newer, at 15.5 GB or 23.5 GB."""
    if info is None or not info.supported:
        return GpuCheck(False, "An NVIDIA RTX 40 series GPU or newer is required.", None)
    tier = select_tier(info.total_gb)
    if tier is None:
        return GpuCheck(False, "This GPU does not have enough memory.", None)
    return GpuCheck(True, "", tier)


def read_os_release(path: Path | None = None) -> dict[str, str]:
    """Parse ``/etc/os-release``. A missing file yields an empty mapping."""
    target = path or Path("/etc/os-release")
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return {}
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip().strip('"')
    return parsed


def current_os_check() -> OsCheck:
    """Check this machine. Used at startup, not at import."""
    if sys.platform == "win32":
        try:
            build = int(sys.getwindowsversion().build)  # type: ignore[attr-defined]
        except (AttributeError, OSError, ValueError):
            build = None
        return check_platform(system="Windows", windows_build=build, linux_release=None)
    if sys.platform.startswith("linux"):
        return check_platform(system="Linux", windows_build=None, linux_release=read_os_release())
    return check_platform(system=sys.platform, windows_build=None, linux_release=None)
