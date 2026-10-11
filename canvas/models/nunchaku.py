"""Nunchaku SVDQuant runtime, and the optional INT4 Z-Image generator.

The generator stays disabled until a GPU quality check, so text to image stays
BF16 and nothing downloads those weights. Qwen-Image-Edit-2511 INT4 uses the
tonera vitoom-nunchaku fork because that build accepts the ``zero_cond_t``
flag in the Edit-2511 checkpoint. The fork publishes no cp314 wheel for
torch 2.11 and CUDA 13.0. Linux has no cp312 wheel for that stack either, so
Linux uses the cp311 wheel in a Canvas-private interpreter. Windows uses the
cp312 wheel the same way. The files are fetched only when a caller asks.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

from canvas.app.i18n import text
from canvas.infra.paths import shared_dir
from canvas.models.registry import ModelSpec

NUNCHAKU_ID = "z-image-turbo-svdquant-int4"
NUNCHAKU_BACKEND = "nunchaku-svdquant"
NUNCHAKU_VRAM_GB = 12.0
NUNCHAKU_COMPONENT_ID = "nunchaku-vitoom"
NUNCHAKU_REVISION = "85bad5510d409570527b018387491fd0c81e4727"
NUNCHAKU_LICENCE = "Apache-2.0"
NUNCHAKU_REPO = "https://huggingface.co/tonera/vitoom-nunchaku"
SUITE_ABI = "cp314"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

# cu130 + torch 2.11 wheels published by the fork. cp314 is absent.
# Linux x86_64 also has no cp312 build.
PUBLISHED_CU130_TORCH211 = frozenset({
    ("linux_x86_64", "cp310"),
    ("linux_x86_64", "cp311"),
    ("linux_aarch64", "cp311"),
    ("linux_aarch64", "cp312"),
    ("linux_aarch64", "cp313"),
    ("win_amd64", "cp311"),
    ("win_amd64", "cp312"),
    ("win_amd64", "cp313"),
})
_ABI_ORDER = ("cp314", "cp312", "cp311", "cp313", "cp310")

Fetch = Callable[[str, Path], None]
Extract = Callable[[Path, Path], None]
Pip = Callable[[Path, Path, bool], None]


class NunchakuUnavailable(Exception):
    """The INT4 generator is not confirmed, or its runtime is not installed."""

    def __init__(self, key: str = "nunchaku.unavailable") -> None:
        super().__init__(key)
        self.key = key

    def __str__(self) -> str:
        return text(self.key)


@dataclass(frozen=True)
class PinnedFile:
    """One file the private INT4 runtime downloads."""

    role: str
    filename: str
    url: str
    sha256: str
    nbytes: int


@dataclass(frozen=True)
class NunchakuWheel:
    """The wheel and interpreter for one platform."""

    abi: str
    python: str
    platform_tag: str
    reason: str
    files: tuple[PinnedFile, ...]

    @property
    def side_venv(self) -> bool:
        """True when the wheel cannot be installed into the suite's Python 3.14."""
        return self.abi != SUITE_ABI

    @property
    def nbytes(self) -> int:
        """Bytes fetched for the wheel, the private interpreter, and torch."""
        return sum(item.nbytes for item in self.files)

    def file(self, role: str) -> PinnedFile:
        """The pinned file with this role."""
        for item in self.files:
            if item.role == role:
                return item
        raise NunchakuUnavailable("nunchaku.missing_file")


def _wheel(filename: str, sha256: str, nbytes: int) -> PinnedFile:
    return PinnedFile(
        "nunchaku",
        filename,
        f"{NUNCHAKU_REPO}/resolve/{NUNCHAKU_REVISION}/{filename}",
        sha256,
        nbytes,
    )


def _linux_plan() -> NunchakuWheel:
    return NunchakuWheel(
        abi="cp311",
        python="3.11.17",
        platform_tag="linux_x86_64",
        reason="no cp314 or cp312 cu130 torch 2.11 wheel; cp311 installs in a private side environment",
        files=(
            _wheel(
                "nunchaku-1.3.0.dev20260622+cu13.0torch2.11-cp311-cp311-linux_x86_64.whl",
                "02a762d94ed25ac572dbc59bb8109b0a9df6f9d875aa51228564e70307b3b68c",
                147_399_162,
            ),
            PinnedFile(
                "python",
                "cpython-3.11.17+20261009-x86_64-unknown-linux-gnu-install_only.tar.gz",
                "https://github.com/astral-sh/python-build-standalone/releases/download/20261009/"
                "cpython-3.11.17%2B20261009-x86_64-unknown-linux-gnu-install_only.tar.gz",
                "",
                48_949_967,
            ),
            PinnedFile(
                "torch",
                "torch-2.11.0+cu130-cp311-cp311-manylinux_2_28_x86_64.whl",
                "https://download.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp311-cp311-manylinux_2_28_x86_64.whl",
                "225b22e0a4e36ea573d3a68796e6816a160616f67e8b8c55683a88bf7777f4cd",
                531_045_934,
            ),
        ),
    )


def _windows_plan() -> NunchakuWheel:
    return NunchakuWheel(
        abi="cp312",
        python="3.12.15",
        platform_tag="win_amd64",
        reason="no cp314 cu130 torch 2.11 wheel; cp312 installs in a private side environment",
        files=(
            _wheel(
                "nunchaku-1.3.0.dev20260629+cu13.0torch2.11-cp312-cp312-win_amd64.whl",
                "ddc015bfa6fc6b84957532823c23418f5b986cb61e3c02c97a646cfe2a1fd955",
                92_083_834,
            ),
            PinnedFile(
                "python",
                "cpython-3.12.15+20261009-x86_64-pc-windows-msvc-install_only.tar.gz",
                "https://github.com/astral-sh/python-build-standalone/releases/download/20261009/"
                "cpython-3.12.15%2B20261009-x86_64-pc-windows-msvc-install_only.tar.gz",
                "",
                46_352_482,
            ),
            PinnedFile(
                "torch",
                "torch-2.11.0+cu130-cp312-cp312-win_amd64.whl",
                "https://download.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp312-cp312-win_amd64.whl",
                "ef8beae16d781c3244ef28dc7bee6d8871c26bbde65d5bf66e902cb61972c4ab",
                1_915_224_818,
            ),
        ),
    )


_PLANS = {
    ("linux_x86_64", "cp311"): _linux_plan(),
    ("win_amd64", "cp312"): _windows_plan(),
}


def platform_tag(system: str, machine: str) -> str:
    """``linux_x86_64`` or ``win_amd64`` for the wheels this app pins."""
    arch = machine.lower()
    if system == "linux" and arch in {"x86_64", "amd64"}:
        return "linux_x86_64"
    if system == "win32" and arch in {"x86_64", "amd64"}:
        return "win_amd64"
    raise NunchakuUnavailable("nunchaku.no_wheel")


def select_nunchaku_wheel(system: str, machine: str) -> NunchakuWheel:
    """The pinned wheel for this PC. cp314 first, then cp312, then cp311."""
    tag = platform_tag(system, machine)
    published = {abi for plat, abi in PUBLISHED_CU130_TORCH211 if plat == tag}
    for abi in _ABI_ORDER:
        if abi not in published:
            continue
        plan = _PLANS.get((tag, abi))
        if plan is None:
            raise NunchakuUnavailable("nunchaku.unpinned")
        return plan
    raise NunchakuUnavailable("nunchaku.no_wheel")


def runtime_home() -> Path:
    """Private interpreter and wheels. This is not the suite Python 3.14 runtime."""
    return shared_dir() / "runtime" / "nunchaku-side"


def interpreter_path(root: Path, plan: NunchakuWheel) -> Path:
    """The private interpreter used to import nunchaku."""
    folder = root / "python"
    if plan.platform_tag.startswith("win"):
        return folder / "python.exe"
    return folder / "bin" / "python3"


def runtime_is_installed(plan: NunchakuWheel, root: Path | None = None) -> bool:
    """True when the pinned wheel, torch, and the private interpreter are present."""
    folder = runtime_home() if root is None else root
    if not interpreter_path(folder, plan).is_file():
        return False
    marker = folder / "ready"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != plan.file("nunchaku").sha256:
        return False
    for item in plan.files:
        path = folder / item.filename
        if item.sha256:
            if not _matches(path, item.sha256):
                return False
        elif not path.is_file() or path.stat().st_size != item.nbytes:
            return False
    return True


def install_runtime_files(
    plan: NunchakuWheel,
    root: Path,
    progress: Callable[[str], None] | None = None,
    *,
    fetch: Fetch | None = None,
    extract: Extract | None = None,
    pip: Pip | None = None,
) -> None:
    """Download the pinned files and install them into the private interpreter."""
    fetch_one = _fetch if fetch is None else fetch
    extract_one = _extract_python if extract is None else extract
    pip_one = _pip_install if pip is None else pip
    root.mkdir(parents=True, exist_ok=True)
    for item in plan.files:
        dest = root / item.filename
        if item.sha256 and _matches(dest, item.sha256):
            continue
        if not item.sha256 and dest.is_file() and dest.stat().st_size == item.nbytes:
            continue
        if progress is not None:
            progress(text("splash.starting", name=item.filename))
        fetch_one(item.url, dest)
        if item.sha256 and not _matches(dest, item.sha256):
            dest.unlink(missing_ok=True)
            raise OSError(text("error.package"))
        if not item.sha256 and (not dest.is_file() or dest.stat().st_size != item.nbytes):
            dest.unlink(missing_ok=True)
            raise OSError(text("error.package"))
    interpreter = interpreter_path(root, plan)
    if not interpreter.is_file():
        # The standalone archive already contains a python/ directory.
        extract_one(root / plan.file("python").filename, root)
    if not interpreter.is_file():
        raise OSError(text("error.package"))
    pip_one(interpreter, root / plan.file("torch").filename, True)
    pip_one(interpreter, root / plan.file("nunchaku").filename, False)
    (root / "ready").write_text(plan.file("nunchaku").sha256 + "\n", encoding="utf-8")


def nunchaku_ready(spec: ModelSpec) -> bool:
    """True only when the INT4 generator is enabled, pinned, and permissively licensed."""
    return (
        spec.status == "ready"
        and spec.quant == "int4"
        and spec.quant_backend == NUNCHAKU_BACKEND
        and bool(spec.files)
        and spec.licence.casefold() in {"apache-2.0", "mit"}
    )


def load_nunchaku_generator(spec: ModelSpec) -> Any:
    """Open the INT4 generator. A disabled row raises before any file is fetched.

    The ``nunchaku`` import stays inside this function. The package is optional
    and is not installed with Canvas, so a top-level import would break the
    unit tests. It is not a circular import.
    """
    if not nunchaku_ready(spec):
        raise NunchakuUnavailable()
    try:
        import nunchaku
    except ImportError as exc:
        raise NunchakuUnavailable() from exc
    return nunchaku


def _matches(path: Path, digest: str) -> bool:
    if not path.is_file():
        return False
    return _sha256(path) == digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    if not url.startswith("https://"):
        raise OSError(text("error.package"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    request = Request(url, headers={"User-Agent": "voxprint-canvas"})
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as handle:  # nosec B310
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(dest)


def _extract_python(archive: Path, dest: Path) -> None:
    """Extract a standalone CPython build. The archive's top directory is ``python/``.

    ``dest`` is the runtime folder, which already holds the downloaded wheels.
    Only the extracted interpreter tree is replaced.
    """
    target = dest / "python"
    if target.exists():
        shutil.rmtree(target)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        handle.extractall(dest, filter="data")


def _pip_install(interpreter: Path, wheel: Path, deps: bool) -> None:
    command = [str(interpreter), "-m", "pip", "install", str(wheel)]
    if deps:
        command.extend(["--index-url", TORCH_INDEX, "--extra-index-url", "https://pypi.org/simple"])
    else:
        command.append("--no-deps")
    completed = subprocess.run(command, stdout=sys.stderr, stderr=sys.stderr, check=False)
    if completed.returncode != 0:
        raise OSError(text("error.package"))
