"""Python and CUDA wheel locations shared by the suite.

The setup scripts install these when they are missing. Launch repair uses the
same addresses when the manifest says a copy is corrupt.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

from canvas.infra.paths import shared_dir

LINUX_PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20261009/"
    "cpython-3.12.15%2B20261009-x86_64-unknown-linux-gnu-install_only.tar.gz"
)
WINDOWS_PYTHON_URL = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
WHEEL_SOURCE = "pytorch-cu128"


def python_home() -> Path:
    """Directory of a suite-installed interpreter."""
    return shared_dir() / "python"


def wheel_dir() -> Path:
    """Cached CUDA wheels."""
    return shared_dir() / "wheels"


def restore_runtime(progress: Callable[[str], None]) -> None:
    """Download the suite interpreter into the shared folder."""
    progress("Installing Python")
    if sys.platform == "win32":
        _restore_windows()
    else:
        _restore_linux()


def refill_wheels(progress: Callable[[str], None]) -> None:
    """Download the CUDA torch wheels into the shared wheel folder."""
    progress("Installing PyTorch")
    dest = wheel_dir()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call(
            [
                sys.executable, "-m", "pip", "download", "torch", "torchvision",
                "--index-url", TORCH_INDEX, "-d", str(dest),
            ],
        )
    except subprocess.CalledProcessError as exc:
        raise OSError("PyTorch could not be installed.") from exc


def file_sha256(path: Path) -> str:
    """SHA-256 of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(folder: Path) -> dict[str, str]:
    """SHA-256 of each file under ``folder``, keyed by the relative path."""
    if not folder.is_dir():
        return {}
    found: dict[str, str] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name.endswith(".partial"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        found[path.relative_to(folder).as_posix()] = digest.hexdigest()
    return found


def _restore_linux() -> None:
    dest = python_home()
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    archive = dest.with_suffix(".tar.gz")
    _download(LINUX_PYTHON_URL, archive)
    with tarfile.open(archive) as handle:
        handle.extractall(dest, filter="data")
    archive.unlink(missing_ok=True)
    inner = dest / "python"
    if inner.is_dir() and not (dest / "bin").exists():
        for child in inner.iterdir():
            child.rename(dest / child.name)
        inner.rmdir()


def _restore_windows() -> None:
    dest = python_home()
    dest.mkdir(parents=True, exist_ok=True)
    installer = dest / "python-installer.exe"
    _download(WINDOWS_PYTHON_URL, installer)
    completed = subprocess.run(
        [
            str(installer), "/quiet", "InstallAllUsers=0", "PrependPath=0",
            "Include_test=0", "Include_pip=1", f"TargetDir={dest}",
        ],
        check=False,
    )
    installer.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise OSError("Python could not be installed.")


def interpreter_marker() -> Path | None:
    """The suite interpreter, when this machine has one."""
    home = python_home()
    if sys.platform == "win32":
        candidate = home / "python.exe"
    else:
        candidate = home / "bin" / "python3"
    if candidate.is_file():
        return candidate
    return None


def _download(url: str, dest: Path) -> None:
    if not url.startswith("https://"):
        raise OSError("unsupported download address")
    request = urllib.request.Request(url, headers={"User-Agent": "voxprint-canvas"})
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:  # nosec B310
        shutil.copyfileobj(response, handle)
