"""Download the pinned Enlarge Linux release, verify SHA256SUMS, and run the client checks."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from canvas.infra.enlarge_release import LINUX_ASSET, LINUX_SHA256, SUMS_ASSET, TAG, asset_url, parse_sha256sums


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "voxprint-canvas"})
    with urllib.request.urlopen(request) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def main() -> int:
    # The checkout is executable. /tmp is not, on some runners, so the bundled interpreter cannot live there.
    root = Path.cwd() / ".enlarge-release"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    sums_path = root / SUMS_ASSET
    archive = root / LINUX_ASSET
    print(f"Enlarge release {TAG}")
    _download(asset_url(SUMS_ASSET), sums_path)
    sums = parse_sha256sums(sums_path.read_text(encoding="utf-8"))
    listed = sums.get(LINUX_ASSET)
    if listed != LINUX_SHA256:
        print(f"SHA256SUMS lists {listed}, pin is {LINUX_SHA256}", file=sys.stderr)
        return 1
    print(f"downloading {LINUX_ASSET}")
    _download(asset_url(LINUX_ASSET), archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != LINUX_SHA256:
        print(f"tarball sha256 {digest} does not match the pin", file=sys.stderr)
        return 1
    print("tarball matches SHA256SUMS and the pin")
    with tarfile.open(archive) as handle:
        handle.extractall(root / "tree", filter="data")
    binary = root / "tree" / "voxprint-enlarge" / "voxprint-enlarge"
    if not binary.is_file():
        print(f"missing launcher {binary}", file=sys.stderr)
        return 1
    binary.chmod(0o755)
    env = os.environ.copy()
    env["CANVAS_ENLARGE_BIN"] = str(binary)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    report = root / "report.txt"
    env["CANVAS_ENLARGE_REPORT"] = str(report)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_enlarge_release.py", "-q", "--tb=short"],
        env=env,
        check=False,
    )
    if report.is_file():
        print(report.read_text(encoding="utf-8"), end="")
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
