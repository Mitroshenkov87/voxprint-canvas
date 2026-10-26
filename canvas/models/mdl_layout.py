"""Pair an installed MDL model with the files in the blob store.

VOXPrint MDL writes each file once to ``models_dir()/blobs/<sha256>``. The
installed ``mdl.json`` says which repo path that blob belongs to, including an
``<id>-upstream`` FP32 entry. A missing blob, or a blob of the wrong size, means
the model is not downloaded.
"""
from __future__ import annotations

from pathlib import Path

from voxprint_common.mdl.catalog import CatalogError, Model, load_manifest

from canvas.infra.paths import models_dir
from canvas.models.mdl_registry import manifest_path
from canvas.models.views import blob_path


def files_for(model_id: str, root: Path | None = None) -> dict[str, Path] | None:
    """Map each repo path to its blob.

    The installed ``mdl.json`` is read with
    :func:`voxprint_common.mdl.catalog.load_manifest`. The result is ``None``
    when that id has no file list, a blob is missing, or a blob has the wrong
    size. :func:`has_files` is false only in the first of those cases. A missing
    blob is not downloaded from a legacy folder.

    Args:
        model_id: A model id from the installed manifest, including ``<id>-upstream``.
        root: Models directory. Defaults to :func:`canvas.infra.paths.models_dir`.
    """
    rows = _rows(model_id)
    if not rows:
        return None
    folder = models_dir() if root is None else root
    for model in rows:
        mapped = _ready(folder, model)
        if mapped is not None:
            return mapped
    return None


def has_files(model_id: str) -> bool:
    """True when the installed manifest lists at least one file for ``model_id``."""
    return bool(_rows(model_id))


def _rows(model_id: str) -> tuple[Model, ...]:
    try:
        manifest = load_manifest(manifest_path())
    except CatalogError:
        return ()
    return tuple(model for model in manifest.models if model.id == model_id and model.files)


def _ready(root: Path, model: Model) -> dict[str, Path] | None:
    mapped: dict[str, Path] = {}
    for item in model.files:
        blob = blob_path(root, item.sha256)
        try:
            if not blob.is_file() or blob.stat().st_size != item.size:
                return None
        except OSError:
            return None
        mapped[item.path] = blob
    return mapped or None
