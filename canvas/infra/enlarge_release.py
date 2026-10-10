"""Pinned Enlarge release that Canvas CI checks. Tag and tarball hash live here only."""
from __future__ import annotations

TAG = "v0.2.0-rc.1"
REPO = "Mitroshenkov87/voxprint-enlarge"
LINUX_ASSET = "voxprint-enlarge-linux-x64.tar.gz"
SUMS_ASSET = "SHA256SUMS"
LINUX_SHA256 = "0ae52900a3a8940e094ce51d481a94e6cef7837c2b7aaaca04e382198151a8f2"


def asset_url(name: str) -> str:
    return f"https://github.com/{REPO}/releases/download/{TAG}/{name}"


def parse_sha256sums(text: str) -> dict[str, str]:
    """``sha256sum`` lines: digest, then the file name."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        found[name.strip().lstrip("*")] = digest.strip().lower()
    return found
