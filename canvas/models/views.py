"""Folder views over the VOXPrint MDL blob store.

MDL downloads each file once into ``models_dir()/blobs/<sha256>``. diffusers and
transformers open a model folder with its repo layout, so Canvas builds a view:
``models_dir()/views/canvas/<key>-<digest12>/`` whose entries point at the blobs.

- An entry is a hard link to its blob (same volume as the blobs, no admin rights on
  Windows). When the link fails, Linux and macOS use a symbolic link. A small config
  file (JSON, text, tokenizer data up to ``SMALL_COPY_BYTES``) may be copied; a
  weight file is never copied. On Windows a failed link refuses the weights.
- A view is built in a staging folder and renamed into place, with a marker file
  that lists every path and digest. A view whose marker or blobs no longer match is
  rebuilt, or removed when a blob is gone.
- Only view folders are ever deleted. The blobs belong to MDL and are never
  removed, renamed or written here.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


VIEWS_DIR = "views"
APP_DIR = "canvas"
BLOBS_DIR = "blobs"
MARKER = ".voxprint-view.json"
MARKER_SCHEMA = 1
SMALL_COPY_BYTES = 4 * 1024 * 1024
_COPY_SUFFIXES = frozenset({".json", ".txt", ".model", ".tiktoken", ".jinja", ".md"})


class ViewError(Exception):
    """A view could not be built without copying a weight file."""


def blob_path(root: Path, digest: str) -> Path:
    """Where MDL keeps the file with ``digest``."""
    return root / BLOBS_DIR / digest.lower()


def views_root(root: Path) -> Path:
    """Folder that holds every Canvas view under ``root``."""
    return root / VIEWS_DIR / APP_DIR


def view_dir(root: Path, key: str, files: dict[str, str]) -> Path:
    """The view folder for one file set. The digest changes when any pin changes."""
    listing = "\n".join(f"{name}\t{digest.lower()}" for name, digest in sorted(files.items()))
    digest = hashlib.sha256(listing.encode("utf-8")).hexdigest()[:12]
    safe = key.replace("/", "--").replace("\\", "--")
    return views_root(root) / f"{safe}-{digest}"


def is_view(root: Path, folder: Path) -> bool:
    """True when ``folder`` is a Canvas view under ``root`` (MDL already refs its blobs)."""
    base = views_root(root).resolve()
    return base in folder.resolve().parents


def blobs_present(root: Path, files: dict[str, str]) -> bool:
    """True when every pinned file is in the blob store with a plausible digest name."""
    return bool(files) and all(blob_path(root, digest).is_file() for digest in files.values())


def ensure_view(root: Path, key: str, files: dict[str, str]) -> Path | None:
    """Return a view folder for ``files``, building it when needed.

    Returns None when a blob is missing; a stale view for the same set is removed.
    The caller still verifies each file against its pin before loading it.

    Raises:
        ViewError: A weight file could not be linked and copying it is refused.
    """
    folder = view_dir(root, key, files)
    if not blobs_present(root, files):
        _remove_view(root, folder)
        return None
    if _view_valid(root, folder, files):
        return folder
    staging = folder.with_name(f"{folder.name}.staging-{os.getpid()}")
    _remove_view(root, staging)
    staging.mkdir(parents=True)
    try:
        for name, digest in sorted(files.items()):
            target = staging / _safe_name(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            _place(blob_path(root, digest), target)
        marker = {"schema": MARKER_SCHEMA, "key": key, "files": {n: d.lower() for n, d in sorted(files.items())}}
        (staging / MARKER).write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _remove_view(root, folder)
        os.replace(staging, folder)
    except BaseException:
        _remove_view(root, staging)
        raise
    return folder


def remove_views(root: Path) -> None:
    """Remove every Canvas view under ``root``. The blobs stay."""
    base = views_root(root)
    if not base.is_dir():
        return
    for child in base.iterdir():
        _remove_view(root, child)


def _safe_name(name: str) -> Path:
    parts = Path(name.replace("\\", "/")).parts
    if not parts or Path(name).is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise ViewError("a pinned file name must stay inside its model folder")
    return Path(*parts)


def _place(blob: Path, target: Path) -> None:
    try:
        os.link(blob, target)
        return
    except OSError:
        pass
    if sys.platform != "win32":
        try:
            os.symlink(blob, target)
            return
        except OSError:
            pass
    if _copy_allowed(target, blob):
        shutil.copyfile(blob, target)
        return
    raise ViewError(f"{target.name} could not be linked to the model store; weights are not copied")


def _view_valid(root: Path, folder: Path, files: dict[str, str]) -> bool:
    try:
        marker = json.loads((folder / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(marker, dict) or marker.get("schema") != MARKER_SCHEMA:
        return False
    if marker.get("files") != {n: d.lower() for n, d in files.items()}:
        return False
    for name, digest in files.items():
        entry = folder / _safe_name(name)
        blob = blob_path(root, digest)
        try:
            if entry.is_symlink():
                if entry.resolve() != blob.resolve():
                    return False
            elif not os.path.samefile(entry, blob) and not _small_copy(entry, blob):
                return False
        except OSError:
            return False
    return True


def _copy_allowed(target: Path, blob: Path) -> bool:
    return target.suffix.lower() in _COPY_SUFFIXES and blob.stat().st_size <= SMALL_COPY_BYTES


def _small_copy(entry: Path, blob: Path) -> bool:
    return _copy_allowed(entry, blob) and entry.stat().st_size == blob.stat().st_size


def _remove_view(root: Path, folder: Path) -> None:
    """Delete one view or staging folder. Anything outside the views root is left alone."""
    base = views_root(root).resolve()
    if not (folder.exists() or folder.is_symlink()):
        return
    if folder.is_symlink():
        return
    target = folder.resolve()
    if target == base or base not in target.parents:
        return
    blobs = (root / BLOBS_DIR).resolve()
    if target == blobs or blobs in target.parents:
        return
    # rmtree removes hard links and symbolic links themselves, never what they point at.
    shutil.rmtree(target)
