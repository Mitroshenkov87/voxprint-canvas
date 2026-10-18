"""Place pinned weight files on disk and refuse a bad hash.

Downloads happen only when a caller asks. A check that the files are already
present never opens a network connection.
"""
from __future__ import annotations

import hashlib
import shutil
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from canvas.app.i18n import text
from canvas.models.features import withheld_from_setup
from canvas.models.pin import PinError, require_pin, verify_file
from canvas.models.registry import ModelSpec
from canvas.models.verified import VerifiedCache
from canvas.models.views import ViewError, blob_path, blobs_present, ensure_view

_PACKAGE = Path(__file__).parent
_CHUNK = 1024 * 1024


class SnapshotError(Exception):
    """A weight file could not be fetched or did not match its pin."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        return text("error.download")


def weight_dir(root: Path, repo: str, *, fallback: bool = False) -> Path:
    """Directory for one repo snapshot. The fallback tree is a sibling folder."""
    name = repo.replace("/", "--")
    if fallback:
        name += "--upstream"
    return root / name


def snapshot_repo(spec: ModelSpec) -> str:
    """Folder key for one registry row."""
    return spec.repo


def snapshot_status(spec: ModelSpec, root: Path, cache: VerifiedCache | None = None) -> str:
    """``ready``, ``not_downloaded``, ``incomplete``, or ``hash_mismatch``.

    Files in the VOXPrint MDL blob store count first; a legacy repo folder still counts.
    """
    primary = _best(_blob_status(root, spec.files, cache),
                    _folder_status(weight_dir(root, snapshot_repo(spec)), spec.files, cache))
    fallback = "not_downloaded"
    if spec.fallback_files:
        fallback = _best(
            _blob_status(root, spec.fallback_files, cache),
            _folder_status(weight_dir(root, snapshot_repo(spec), fallback=True), spec.fallback_files, cache),
        )
    if primary == "ready" or fallback == "ready":
        return "ready"
    for state in (primary, fallback):
        if state == "hash_mismatch":
            return "hash_mismatch"
    for state in (primary, fallback):
        if state == "incomplete":
            return "incomplete"
    return "not_downloaded"


def _best(first: str, second: str) -> str:
    order = ("ready", "hash_mismatch", "incomplete", "not_downloaded")
    return min(first, second, key=order.index)


def _blob_status(root: Path, files: dict[str, str], cache: VerifiedCache | None = None) -> str:
    if not files:
        return "not_downloaded"
    present = [digest for digest in files.values() if blob_path(root, digest).is_file()]
    if not present:
        return "not_downloaded"
    for digest in present:
        if not _matches(blob_path(root, digest), digest, cache):
            return "hash_mismatch"
    return "ready" if len(present) == len(files) else "incomplete"


def _folder_status(folder: Path, files: dict[str, str], cache: VerifiedCache | None = None) -> str:
    if not files:
        return "not_downloaded"
    present = 0
    for name, digest in files.items():
        path = folder / name
        if not path.is_file():
            continue
        present += 1
        if not _matches(path, digest, cache):
            return "hash_mismatch"
    if present == 0:
        return "not_downloaded"
    if present == len(files):
        return "ready"
    return "incomplete"


def snapshot_ready(spec: ModelSpec, root: Path, cache: VerifiedCache | None = None) -> Path | None:
    """The local folder whose files match the pin, or None when none does.

    A view over the MDL blob store comes first, then the legacy ``<org>--<repo>``
    folder, then the same two for the FP32 fallback set.
    """
    found = _view_or_none(spec.id, root, spec.files, cache)
    if found is not None:
        return found
    primary = weight_dir(root, snapshot_repo(spec))
    if _complete(primary, spec.files, cache):
        return primary
    return fallback_ready(spec, root, cache)


def fallback_ready(spec: ModelSpec, root: Path, cache: VerifiedCache | None = None) -> Path | None:
    """The FP32 ``<id>-upstream`` view, or the legacy ``--upstream`` folder while it remains.

    MDL stores that entry as blobs. The view is opened first. The old
    ``<repo>--upstream`` tree is only the migration copy.
    """
    if not spec.fallback_files:
        return None
    found = _view_or_none(f"{spec.id}-upstream", root, spec.fallback_files, cache)
    if found is not None:
        return found
    other = weight_dir(root, snapshot_repo(spec), fallback=True)
    return other if _complete(other, spec.fallback_files, cache) else None


def _view_or_none(key: str, root: Path, files: dict[str, str], cache: VerifiedCache | None) -> Path | None:
    """A verified view for ``files``. Each blob is checked against its pin before the view is used."""
    if not blobs_present(root, files):
        ensure_view(root, key, files)  # removes a stale view; returns None
        return None
    if not all(_matches(blob_path(root, digest), digest, cache) for digest in files.values()):
        return None
    try:
        return ensure_view(root, key, files)
    except ViewError as exc:
        raise SnapshotError(str(exc)) from exc


def download_snapshot(
    spec: ModelSpec, root: Path, progress: Callable[[str], None] | None = None,
) -> Path:
    """Fetch the first snapshot whose files can all be verified."""
    if withheld_from_setup(spec) or spec.status == "disabled":
        raise SnapshotError("this model is not available for download")
    primary = weight_dir(root, snapshot_repo(spec))
    try:
        _fill(spec, primary, spec.files, spec.revision, fallback=False, progress=progress)
    except SnapshotError as primary_error:
        if not spec.fallback_files:
            raise
        other = weight_dir(root, snapshot_repo(spec), fallback=True)
        try:
            _fill(
                spec, other, spec.fallback_files, spec.fallback_revision or spec.revision,
                fallback=True, progress=progress,
            )
        except SnapshotError as fallback_error:
            raise SnapshotError(f"{primary_error.detail}; fallback: {fallback_error.detail}") from fallback_error
        return other
    return primary


def download_files(
    spec: ModelSpec, root: Path, names: tuple[str, ...], progress: Callable[[str], None] | None = None,
) -> Path:
    """Fetch and verify named files from one snapshot. Other files stay as they are."""
    missing = [name for name in names if name not in spec.files]
    if not names or missing:
        raise SnapshotError("a requested file is not pinned")
    files = {name: spec.files[name] for name in names}
    folder = weight_dir(root, snapshot_repo(spec))
    _fill(spec, folder, files, spec.revision, fallback=False, progress=progress)
    return folder


def _complete(folder: Path, files: dict[str, str], cache: VerifiedCache | None = None) -> bool:
    if not files:
        return False
    for name, digest in files.items():
        path = folder / name
        if not path.is_file():
            return False
        if not _matches(path, digest, cache):
            return False
    return True


def _matches(path: Path, digest: str, cache: VerifiedCache | None) -> bool:
    if cache is not None:
        return cache.matches(path, digest)
    try:
        verify_file(path, digest)
    except PinError:
        return False
    return True


def _fill(
    spec: ModelSpec, folder: Path, files: dict[str, str], revision: str, *, fallback: bool,
    progress: Callable[[str], None] | None = None,
) -> None:
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
            _fetch_one(spec, name, digest, path, revision, fallback=fallback, progress=progress)
        except (OSError, PinError, SnapshotError) as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise SnapshotError("; ".join(errors))


def _fetch_one(
    spec: ModelSpec, name: str, digest: str, dest: Path, revision: str, *, fallback: bool,
    progress: Callable[[str], None] | None = None,
) -> None:
    pin = require_pin(digest)
    last = "no source"
    for url in _urls(spec, name, revision, fallback=fallback):
        try:
            if url.startswith("package:"):
                if progress is not None:
                    progress(text("splash.copying", name=name))
                _copy_package(url, dest, pin)
            else:
                _stream(url, dest, pin, progress=progress, label=name)
            return
        except (OSError, PinError, SnapshotError) as exc:
            last = f"{url}: {exc}"
            if dest.exists():
                dest.unlink()
    raise SnapshotError(last)


def _urls(spec: ModelSpec, name: str, revision: str, *, fallback: bool) -> tuple[str, ...]:
    if fallback:
        # The author's original files: the suite mirror copy first, then upstream.
        remote = name
        found: list[str] = []
        mirror_copy = _uri(spec, "huggingface-mirror")
        if mirror_copy and spec.fallback_mirror_revision:
            found.append(_hf(mirror_copy, spec.fallback_mirror_revision, remote))
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


def _stream(
    url: str, dest: Path, pin: str, progress: Callable[[str], None] | None = None, label: str = "",
) -> None:
    if not url.startswith("https://"):
        raise SnapshotError("unsupported download address")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "voxprint-canvas"})
    digest = hashlib.sha256()
    name = label or dest.name
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:  # nosec B310
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            started = time.monotonic()
            if progress is not None:
                progress(text("splash.starting", name=name))
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress is not None and total:
                    elapsed = max(time.monotonic() - started, 0.001)
                    remaining = max(0, total - done) / (done / elapsed)
                    progress(text("splash.progress", name=name, percent=done * 100 // total, seconds=int(remaining)))
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != pin:
        partial.unlink(missing_ok=True)
        raise PinError("hash_mismatch")
    partial.replace(dest)
    if progress is not None:
        progress(text("splash.ready", name=name))
