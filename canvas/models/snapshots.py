"""Place pinned weight files on disk and refuse a bad hash.

Downloads happen only when a caller asks. A check that the files are already
present never opens a network connection.
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

from canvas.models.pin import PinError, require_pin, verify_file
from canvas.models.registry import ModelSpec

_PACKAGE = Path(__file__).parent
_CHUNK = 1024 * 1024


class SnapshotError(Exception):
    """A weight file could not be fetched or did not match its pin."""

    def __init__(self, detail: str) -> None:
        super().__init__("The model could not be downloaded.")
        self.detail = detail


def weight_dir(root: Path, repo: str, *, fallback: bool = False) -> Path:
    """Directory for one repo snapshot. The fallback tree is a sibling folder."""
    name = repo.replace("/", "--")
    if fallback:
        name += "--upstream"
    return root / name


def snapshot_ready(spec: ModelSpec, root: Path) -> Path | None:
    """The local folder whose files match the pin, or None when none does."""
    primary = weight_dir(root, spec.repo)
    if _complete(primary, spec.files):
        return primary
    if spec.fallback_files:
        other = weight_dir(root, spec.repo, fallback=True)
        if _complete(other, spec.fallback_files):
            return other
    return None


def download_snapshot(spec: ModelSpec, root: Path) -> Path:
    """Fetch the first snapshot whose files can all be verified."""
    primary = weight_dir(root, spec.repo)
    try:
        _fill(spec, primary, spec.files, spec.revision, fallback=False)
    except SnapshotError as primary_error:
        if not spec.fallback_files:
            raise
        other = weight_dir(root, spec.repo, fallback=True)
        try:
            _fill(spec, other, spec.fallback_files, spec.fallback_revision or spec.revision, fallback=True)
        except SnapshotError as fallback_error:
            raise SnapshotError(f"{primary_error.detail}; fallback: {fallback_error.detail}") from fallback_error
        return other
    return primary


def _complete(folder: Path, files: dict[str, str]) -> bool:
    if not files:
        return False
    for name, digest in files.items():
        path = folder / name
        if not path.is_file():
            return False
        try:
            verify_file(path, digest)
        except PinError:
            return False
    return True


def _fill(spec: ModelSpec, folder: Path, files: dict[str, str], revision: str, *, fallback: bool) -> None:
    errors: list[str] = []
    for name, digest in files.items():
        path = folder / name
        if path.is_file():
            try:
                verify_file(path, digest)
                continue
            except PinError:
                path.unlink()
        try:
            _fetch_one(spec, name, digest, path, revision, fallback=fallback)
        except (OSError, PinError, SnapshotError) as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise SnapshotError("; ".join(errors))


def _fetch_one(spec: ModelSpec, name: str, digest: str, dest: Path, revision: str, *, fallback: bool) -> None:
    pin = require_pin(digest)
    last = "no source"
    for url in _urls(spec, name, revision, fallback=fallback):
        try:
            if url.startswith("package:"):
                _copy_package(url, dest, pin)
            else:
                _stream(url, dest, pin)
            return
        except (OSError, PinError, SnapshotError) as exc:
            last = f"{url}: {exc}"
            if dest.exists():
                dest.unlink()
    raise SnapshotError(last)


def _urls(spec: ModelSpec, name: str, revision: str, *, fallback: bool) -> tuple[str, ...]:
    if fallback:
        remote = name
        found: list[str] = []
        for source in spec.sources:
            if source.kind == "huggingface" and source.uri:
                found.append(_hf(source.uri, revision, remote))
        return tuple(found)
    mirror, upstream = spec.origins.get(name, (name, name))
    found = []
    if mirror.startswith("package:"):
        found.append(mirror)
    if mirror.startswith("http://") or mirror.startswith("https://"):
        found.append(mirror)
    mirror_repo = _uri(spec, "huggingface-mirror")
    mirror_rev = spec.mirror_revision or revision
    if mirror and mirror != "-" and not mirror.startswith(("package:", "http://", "https://")) and mirror_repo:
        found.append(_hf(mirror_repo, mirror_rev, mirror))
    if upstream.startswith("http://") or upstream.startswith("https://"):
        found.append(upstream)
    upstream_repo = _uri(spec, "huggingface")
    direct = _uri(spec, "upstream")
    if upstream and upstream != "-" and not upstream.startswith(("http://", "https://")) and upstream_repo:
        found.append(_hf(upstream_repo, revision, upstream))
    elif upstream and upstream != "-" and direct.startswith(("http://", "https://")) and name == Path(direct).name:
        found.append(direct)
    return tuple(found)


def _uri(spec: ModelSpec, kind: str) -> str:
    for source in spec.sources:
        if source.kind == kind and source.uri:
            return source.uri
    return ""


def _hf(repo_uri: str, revision: str, path: str) -> str:
    base = repo_uri.rstrip("/")
    rev = revision or "main"
    return f"{base}/resolve/{rev}/{path}"


def _copy_package(url: str, dest: Path, pin: str) -> None:
    name = url.split(":", 1)[1]
    source = _PACKAGE / name
    if not source.is_file():
        raise SnapshotError(f"missing package file {name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    verify_file(dest, pin)


def _stream(url: str, dest: Path, pin: str) -> None:
    if not url.startswith("https://"):
        raise SnapshotError("unsupported download address")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "voxprint-canvas"})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:  # nosec B310
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != pin:
        partial.unlink(missing_ok=True)
        raise PinError("hash_mismatch")
    partial.replace(dest)
