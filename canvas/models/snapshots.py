"""Resolve a model through its VOXPrint MDL view and refuse a bad hash.

A download is queued with MDL. An incomplete view is not read from a legacy
folder, and that folder is not deleted. A check that the files are already
present never opens a network connection.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

from canvas.app.i18n import text
from canvas.infra.stage_timer import log_detail
from canvas.models.features import withheld_from_setup
from canvas.models.mdl_layout import files_for, has_files
from canvas.models.pin import PinError, verify_file
from canvas.models.registry import ModelSpec
from canvas.models.verified import VerifiedCache
from canvas.models.views import ViewError, blob_path, blobs_present, drop_model_views, ensure_view


class SnapshotError(Exception):
    """A weight file could not be fetched or did not match its pin."""

    def __init__(self, detail: str, *, user_message: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.user_message = user_message

    def __str__(self) -> str:
        if self.user_message:
            return self.user_message
        return text("error.download")


def weight_dir(root: Path, repo: str, *, fallback: bool = False) -> Path:
    """Legacy ``<org>--<repo>`` folder. ``fallback`` selects the ``--upstream`` sibling.

    Loading does not read this folder. :func:`snapshot_ready` and
    :func:`fallback_ready` return a view over the MDL blobs, or nothing.
    The folder is left in place. Nothing here deletes it.
    """
    name = repo.replace("/", "--")
    if fallback:
        name += "--upstream"
    return root / name


def snapshot_repo(spec: ModelSpec) -> str:
    """Folder key for one registry row."""
    return spec.repo


def snapshot_status(spec: ModelSpec, root: Path, cache: VerifiedCache | None = None) -> str:
    """``ready``, ``not_downloaded``, ``incomplete``, or ``hash_mismatch``.

    Only the VOXPrint MDL blob store counts. A missing blob, or a blob of the
    wrong size, is not downloaded. A legacy ``<org>--<name>`` folder is ignored.
    """
    primary = _source_status(spec.id, spec.files, root, cache)
    fallback = "not_downloaded"
    if spec.fallback_files:
        fallback = _source_status(
            f"{spec.id}-upstream",
            spec.fallback_files,
            root,
            cache,
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


def _source_status(
    model_id: str, catalog_files: dict[str, str], root: Path, cache: VerifiedCache | None,
) -> str:
    """Blob status for one id. A legacy repo folder is not a source."""
    if has_files(model_id):
        mapped = files_for(model_id, root)
        if mapped is None:
            return "not_downloaded"
        return _verified({path: blob.name for path, blob in mapped.items()}, root, cache)
    return _blob_status(root, catalog_files, cache)


def _verified(files: dict[str, str], root: Path, cache: VerifiedCache | None) -> str:
    if not files:
        return "not_downloaded"
    for digest in files.values():
        if not _matches(blob_path(root, digest), digest, cache):
            return "hash_mismatch"
    return "ready"


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


def snapshot_ready(spec: ModelSpec, root: Path, cache: VerifiedCache | None = None) -> Path | None:
    """The MDL view whose files match the pin, or None when the view is incomplete.

    A legacy ``<org>--<repo>`` folder is not opened. The FP32 ``<id>-upstream``
    entry is also a view, or nothing.
    """
    found = _open_layout(spec.id, root, spec.files, cache)
    if found is not None:
        return _note_view(spec.id, found)
    return fallback_ready(spec, root, cache)


def fallback_ready(spec: ModelSpec, root: Path, cache: VerifiedCache | None = None) -> Path | None:
    """The FP32 ``<id>-upstream`` view, or None when that view is incomplete.

    The old ``<repo>--upstream`` tree is not opened and is not deleted.
    """
    if not spec.fallback_files:
        return None
    found = _open_layout(f"{spec.id}-upstream", root, spec.fallback_files, cache)
    if found is None:
        return None
    return _note_view(f"{spec.id}-upstream", found)


def request_download(model_id: str) -> bool:
    """Queue ``model_id`` with VOXPrint MDL. False when it cannot be queued.

    Resolved at call time: a missing MDL client returns False, and tests patch
    ``enqueue``. This does not write a file.
    """
    try:
        from voxprint_common.mdl import enqueue
        from voxprint_common.mdl.catalog import CatalogError

        from canvas.models.mdl_registry import manifest_path
    except ImportError:
        return False
    try:
        enqueue("canvas", [model_id], manifests=[manifest_path()])
    except (CatalogError, OSError, ValueError):
        return False
    return True


def _note_view(model_id: str, folder: Path) -> Path:
    """Record that ``model_id`` resolved to an MDL view."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in model_id)[:80]
    log_detail("model", "path", "cpu", f"id={safe} kind=view")
    return folder


def _open_layout(
    model_id: str, root: Path, catalog_files: dict[str, str], cache: VerifiedCache | None,
) -> Path | None:
    """The view for ``model_id``.

    An installed file list wins. A missing blob or a wrong size drops a stale
    view and is not downloaded. With no file list, the catalog pins are used.
    """
    if has_files(model_id):
        mapped = files_for(model_id, root)
        if mapped is None:
            drop_model_views(root, model_id)
            return None
        files = {path: blob.name for path, blob in mapped.items()}
        if not all(_matches(blob_path(root, digest), digest, cache) for digest in files.values()):
            drop_model_views(root, model_id)
            return None
        return _view_or_none(model_id, root, files, cache)
    return _view_or_none(model_id, root, catalog_files, cache)


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
        raise SnapshotError(exc.detail, user_message=str(exc)) from exc


def download_snapshot(
    spec: ModelSpec, root: Path, progress: Callable[[str], None] | None = None,
) -> Path:
    """Queue the model with MDL and return its view.

    An incomplete view raises :class:`SnapshotError` whose message tells the
    caller to fetch the model with MDL. No ``<org>--<name>`` copy is written.
    """
    if withheld_from_setup(spec) or spec.status == "disabled":
        raise SnapshotError("this model is not available for download")
    request_download(spec.id)
    folder = snapshot_ready(spec, root)
    if folder is None:
        _missing(progress)
    return folder


def download_files(
    spec: ModelSpec, root: Path, names: tuple[str, ...], progress: Callable[[str], None] | None = None,
) -> Path:
    """Queue the model with MDL and return its view when every named file is there."""
    missing = [name for name in names if name not in spec.files]
    if not names or missing:
        raise SnapshotError("a requested file is not pinned")
    request_download(spec.id)
    folder = snapshot_ready(spec, root)
    if folder is None or any(not (folder / name).is_file() for name in names):
        _missing(progress)
    return folder


def _missing(progress: Callable[[str], None] | None) -> NoReturn:
    message = text("model.missing")
    if progress is not None:
        progress(message)
    raise SnapshotError("model missing", user_message=message)


def _matches(path: Path, digest: str, cache: VerifiedCache | None) -> bool:
    if cache is not None:
        return cache.matches(path, digest)
    try:
        verify_file(path, digest)
    except PinError:
        return False
    return True
