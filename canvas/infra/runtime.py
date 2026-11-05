"""Python and CUDA wheel locations shared by the suite.

The setup scripts install these when they are missing. Launch repair uses the
same addresses when the manifest says a copy is corrupt. The shared runtime
holds torch for CUDA 13.0 and the GPU ONNX Runtime build. The CPU ONNX Runtime
build is not installed there.
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

from canvas.app.i18n import text
from canvas.infra.manifest import APP_ID, ManifestError, load_manifest
from canvas.infra.paths import shared_dir

LINUX_PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20261026/"
    "cpython-3.15.2%2B20261026-x86_64-unknown-linux-gnu-install_only.tar.gz"
)
LINUX_PYTHON_SHA = "08d4ae9024a7cfbd3b70e559241afa2571cd3208d7bd5f96f88e0080dd4092aa"
WINDOWS_PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20261026/"
    "cpython-3.15.2%2B20261026-x86_64-pc-windows-msvc-install_only.tar.gz"
)
WINDOWS_PYTHON_SHA = "17769f04c4c8c228e9fb838a37d92f6a6401b170c5cd27fb3073f233f73bb2f9"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
RUNTIME_REL = "runtime/py315-torch2.11-cu130"
PYTHON_ID = "python-3.15"
PYTHON_VERSION = "3.15.2"
TORCH_ID = "torch-2.11.0-cu130"
TORCH_VERSION = "2.11.0+cu130"
VISION_VERSION = "0.26.0+cu130"
ORT_ID = "onnxruntime-gpu-1.31"
ORT_VERSION = "1.31.0"
TRT_ID = "tensorrt-10"
TRT_VERSION = "10.16.1.11"
TRT_PATH = "tensorrt-10"
TRT_SPEC = "tensorrt-cu13==10.16.1.11"
TORCHAO_VERSION = "0.18.0"
PYPI_INDEX = "https://pypi.org/simple"


def python_supports_tensorrt() -> bool:
    """False on Python 3.14 and newer, which have no TensorRT wheel."""
    return sys.version_info < (3, 14)


def python_home() -> Path:
    """Directory of the suite Python 3.15 runtime."""
    return shared_dir() / RUNTIME_REL


def wheel_dir() -> Path:
    """Cached CUDA wheels."""
    return shared_dir() / "wheels"


def restore_runtime(progress: Callable[[str], None]) -> None:
    """Download the suite interpreter into the shared folder."""
    progress(text("splash.install_python"))
    if interpreter_marker() is not None:
        return
    if python_home().exists() and _foreign(RUNTIME_REL):
        progress(text("splash.runtime_left"))
        return
    if sys.platform == "win32":
        _restore_windows()
    else:
        _restore_linux()


def _foreign(relative: str) -> bool:
    """True when another app refs this path, or the manifest cannot be read."""
    try:
        components = load_manifest(shared_dir())
    except (ManifestError, OSError):
        return True
    for component in components.values():
        if component.path == relative and any(ref != APP_ID for ref in component.refs):
            return True
    return False


def refill_wheels(progress: Callable[[str], None]) -> None:
    """Download CUDA 13.0 torch wheels. PyPI supplies torchao, numpy, and the GPU ONNX Runtime."""
    progress(text("splash.install_torch"))
    dest = wheel_dir()
    dest.mkdir(parents=True, exist_ok=True)
    _pip(
        [
            sys.executable, "-m", "pip", "download",
            f"torch=={TORCH_VERSION}",
            f"torchvision=={VISION_VERSION}",
            "--index-url", TORCH_INDEX,
            "--extra-index-url", PYPI_INDEX,
            "-d", str(dest),
        ],
    )
    _pip(
        [
            sys.executable, "-m", "pip", "download",
            f"onnxruntime-gpu=={ORT_VERSION}",
            f"torchao=={TORCHAO_VERSION}",
            "--index-url", PYPI_INDEX,
            "-d", str(dest),
        ],
    )


def restore_tensorrt(progress: Callable[[str], None]) -> None:
    """Install TensorRT 10 into its own component directory.

    Python 3.15 has no wheel. That skip is a note, not a failed install.
    """
    if not python_supports_tensorrt():
        progress(text("splash.tensorrt_skip"))
        return
    progress(text("splash.install_tensorrt"))
    if _foreign(TRT_PATH):
        progress(text("splash.tensorrt_left"))
        return
    dest = shared_dir() / TRT_PATH
    dest.mkdir(parents=True, exist_ok=True)
    _pip(
        [
            sys.executable, "-m", "pip", "install", "--target", str(dest),
            "--upgrade-strategy", "only-if-needed", TRT_SPEC,
        ],
    )


def _pip(args: list[str]) -> None:
    """Run pip. Its output stays on stderr so a JSON report can own stdout."""
    completed = subprocess.run(args, stdout=sys.stderr, stderr=sys.stderr, check=False)
    if completed.returncode != 0:
        raise OSError(text("error.package"))


def _restore_linux() -> None:
    _restore_standalone(LINUX_PYTHON_URL, LINUX_PYTHON_SHA)


def _restore_windows() -> None:
    _restore_standalone(WINDOWS_PYTHON_URL, WINDOWS_PYTHON_SHA)


def _restore_standalone(url: str, expected: str) -> None:
    """Install one python-build-standalone install_only archive."""
    dest = python_home()
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    archive = dest.with_suffix(".tar.gz")
    _download(url, archive)
    if _sha256(archive) != expected:
        archive.unlink(missing_ok=True)
        raise OSError(text("error.python"))
    with tarfile.open(archive) as handle:
        handle.extractall(dest, filter="data")
    archive.unlink(missing_ok=True)
    inner = dest / "python"
    if inner.is_dir() and not (dest / "bin").exists() and not (dest / "python.exe").exists():
        for child in inner.iterdir():
            child.rename(dest / child.name)
        inner.rmdir()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise OSError(text("error.address"))
    request = urllib.request.Request(url, headers={"User-Agent": "voxprint-canvas"})
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:  # nosec B310
        shutil.copyfileobj(response, handle)
