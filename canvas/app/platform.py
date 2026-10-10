"""OS and GPU requirements. Importing this module does not check the machine."""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from canvas.app.info import MIN_LINUX_YEAR, MIN_WINDOWS_BUILD
from canvas.infra.gpu import GpuInfo
from canvas.models.modes import Tier, select_tier


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


def linux_year(release: Mapping[str, str]) -> int | None:
    """Year of a Linux distro, or None when the release file does not say.

    Ubuntu-style ``YY.MM`` becomes 2000+YY. A four-digit ``VERSION_ID`` is that
    year. Fedora 42 and newer count as 2025. Anything else fails closed.
    """
    ident = str(release.get("ID") or "").casefold()
    version = str(release.get("VERSION_ID") or "").strip().strip('"')
    if ident == "fedora":
        head = version.split(".", 1)[0]
        if not head.isdigit():
            return None
        return 2025 if int(head) >= 42 else 2024
    if not version:
        return None
    head = version.split(".", 1)[0]
    if not head.isdigit():
        return None
    number = int(head)
    if number >= 2000:
        return number
    if 20 <= number <= 99:
        return 2000 + number
    return None


def check_platform(*, system: str, windows_build: int | None, linux_release: Mapping[str, str] | None) -> OsCheck:
    """Windows 11 24H2+ or a 2025-or-later Linux. Other systems are refused."""
    if system == "Windows":
        if windows_build is None or windows_build < MIN_WINDOWS_BUILD:
            return OsCheck(False, "Windows 11 24H2 or newer is required.")
        return OsCheck(True, "")
    if system == "Linux":
        year = linux_year(linux_release or {})
        if year is None or year < MIN_LINUX_YEAR:
            return OsCheck(False, "A Linux distribution from 2025 or later is required.")
        return OsCheck(True, "")
    return OsCheck(False, "Windows 11 24H2 or a 2025 Linux distribution is required.")


def check_gpu(info: GpuInfo | None) -> GpuCheck:
    """RTX 40 series (compute capability 8.9) or newer, with a 16 GB or 24 GB tier."""
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
