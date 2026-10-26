"""Picture and gate models a run must already have as MDL views."""
from __future__ import annotations

from pathlib import Path

from voxprint_common.mdl.catalog import CatalogError, load_manifest

from canvas.models.mdl_fetch import MANIFEST
from canvas.models.registry import Catalog, ModelSpec, RegistryError, load_catalog
from canvas.models.views import ViewError, blobs_present, ensure_view

# The check models, both Z-Image-Turbo packs, and the scenario writer.
_GATE_ROLES = ("prompt-guard", "zero-shot", "face-detector", "age", "nsfw", "age-fine")
_MODEL_IDS = ("z-image-turbo-bf16-16gb", "z-image-turbo-bf16", "qwen3-4b-writer")


def required_specs(catalog: Catalog | None = None) -> tuple[ModelSpec, ...]:
    """Gate models, both Turbo packs, and the writer. A missing id is skipped."""
    chosen = catalog or load_catalog()
    specs: list[ModelSpec] = []
    for role in _GATE_ROLES:
        rows = chosen.by_role(role)
        if len(rows) == 1:
            specs.append(rows[0])
    for model_id in _MODEL_IDS:
        try:
            specs.append(chosen.by_id(model_id))
        except RegistryError:
            continue
    return tuple(specs)


def gate_specs(catalog: Catalog | None = None) -> tuple[ModelSpec, ...]:
    """Content-check models, one per role. An empty result means one role is missing."""
    chosen = catalog or load_catalog()
    found: list[ModelSpec] = []
    for role in _GATE_ROLES:
        rows = chosen.by_role(role)
        if len(rows) != 1:
            return ()
        found.append(rows[0])
    return tuple(found)


def pins_for(spec: ModelSpec) -> dict[str, str]:
    """SHA-256 for each catalog path. The MDL manifest wins when it lists the path."""
    catalog = {path: digest.lower() for path, digest in spec.files.items()}
    if not catalog:
        return {}
    try:
        manifest = load_manifest(MANIFEST)
    except (CatalogError, OSError, ValueError):
        return catalog
    listed = next((model for model in manifest.models if model.id == spec.id and model.files), None)
    if listed is None:
        return catalog
    published = {item.path: item.sha256.lower() for item in listed.files}
    return {path: published.get(path, digest) for path, digest in catalog.items()}


def view_complete(spec: ModelSpec, root: Path) -> bool:
    """True when every pin is a blob and a view can be built. The bytes are not hashed."""
    pins = pins_for(spec)
    if not pins or not blobs_present(root, pins):
        return False
    try:
        return ensure_view(root, spec.id, pins) is not None
    except ViewError:
        return False
