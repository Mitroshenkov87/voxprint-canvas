"""The Generate-model picker list, read from the app manifest.

voxprint-common 0.7.1 surfaces the optional ``label``, ``purpose`` and ``group``
fields on ``Model``. Canvas still reads them from the raw manifest JSON so the
picker stays consistent with synthetic test manifests. In a real install the
list comes from the pinned installer catalog (repo-root ``mdl.json``, 14 rows)
via :func:`canvas.models.mdl_registry.manifest_path`. The packaged
``canvas/mdl.json`` stub is only the last resort when no pinned catalog is
found; it carries the same 14 ids plus two uncatalogued SVDQuant placeholders.
A row joins the picker when it is installed — not an unavailable placeholder
and its files are on disk — and its group is ``generate``. A row without an
explicit ``group`` is ``generate`` only when the Canvas catalog lists it with
the ``generator`` role; every other row stays out of the picker.

``group`` must be a recognised task group from the 0.7.1 field spec
(``generate``, ``edit``, ``upscale``, ``restore``, ``voice``, ``transcribe``,
``translate``, ``writer``, ``safety``, ``internal``). An unknown value falls
under ``other`` in Manage models. This field is unrelated to the
``file_groups`` file-grouping field on common's Model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from canvas.infra.paths import models_dir
from canvas.models.mdl_registry import manifest_path
from canvas.models.modes import Tier
from canvas.models.registry import Catalog, ModelSpec, RegistryError, load_catalog
from canvas.models.views import blob_path

#: The Canvas setting the pick is stored under (a TEXT setting on the layout).
SETTING_KEY = "canvas.generate_model"

#: Picker sections. Everything that is not the full-size tier lands in the
#: compact section, so an 8 GB build would sit beside the 16 GB ones.
FULL_TIER = "24gb"

_GROUP_GENERATE = "generate"


@dataclass(frozen=True)
class GenerateChoice:
    """One installed generate-group model the picker can show.

    Attributes:
        id: The manifest model id. Words-to-image plans use it when the
            catalog lists the row.
        text: The visible name: ``label``, else ``display_name``, else ``id``.
        tip: The ``purpose`` line for the tooltip; empty when the row has none.
        tier: The row's ``vram_tier``. ``"24gb"`` picks the full-size section,
            anything else the compact one.
        catalogued: The id has a row in ``canvas/models/catalog.json``, so the
            generation plan can run it. A model that is installed but not
            catalogued (for example a compact 4-bit build) stays listed; the
            plan then falls back to the family mapping.
    """

    id: str
    text: str
    tip: str
    tier: str
    catalogued: bool


def _raw_rows(manifest: Path | None) -> tuple[dict[str, Any], ...]:
    """The raw model rows of the manifest JSON, fields common does not keep."""
    target = manifest_path() if manifest is None else manifest
    try:
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return ()
    rows = payload.get("models")
    if not isinstance(rows, list):
        return ()
    return tuple(row for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str))


def _catalog_roles() -> dict[str, str]:
    """The Canvas catalog's role by id. An unreadable catalog maps nothing."""
    try:
        catalog = load_catalog()
    except (RegistryError, OSError, ValueError):
        return {}
    return {spec.id: spec.role for spec in catalog.models}


def _row_label(row: dict[str, Any]) -> str:
    for key in ("label", "display_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(row["id"])


def _row_tip(row: dict[str, Any]) -> str:
    value = row.get("purpose")
    return value.strip() if isinstance(value, str) else ""


def _row_group(row: dict[str, Any], roles: dict[str, str]) -> str:
    value = row.get("group")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _GROUP_GENERATE if roles.get(str(row["id"]), "") == "generator" else ""


def _installed(row: dict[str, Any], root: Path | None) -> bool:
    """True when the row is no placeholder and every listed blob is on disk.

    The check is the one :func:`canvas.models.mdl_layout.files_for` uses on the
    installed manifest: a listed file with a missing or wrong-size blob means
    the model is not downloaded. It runs on the raw row so a caller-supplied
    manifest and models folder stay consistent.
    """
    if row.get("status") == "unavailable":
        return False
    files = row.get("files")
    if not isinstance(files, list) or not files:
        return False
    folder = models_dir() if root is None else root
    for item in files:
        if not isinstance(item, dict):
            return False
        digest = str(item.get("sha256") or "")
        size = item.get("size")
        if not digest.strip() or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            return False
        blob = blob_path(folder, digest)
        try:
            if not blob.is_file() or blob.stat().st_size != size:
                return False
        except OSError:
            return False
    return True


def choices(manifest: Path | None = None, root: Path | None = None) -> tuple[GenerateChoice, ...]:
    """Every installed generate-group model in the app manifest, in manifest order.

    Encoders, VAEs, the writer, edit-group (ControlNet) rows and safety-group
    rows stay out, installed or not. Optional plug-in generators (MOSS) join
    when they are installed, whether or not the Canvas manifest table lists
    them; ``catalogued`` says if the Canvas catalog can plan them.
    """
    rows = _raw_rows(manifest)
    if not rows:
        return ()
    roles = _catalog_roles()
    found: list[GenerateChoice] = []
    for row in rows:
        if not _installed(row, root):
            continue
        if _row_group(row, roles) != _GROUP_GENERATE:
            continue
        tier = row.get("vram_tier")
        found.append(
            GenerateChoice(
                id=str(row["id"]),
                text=_row_label(row),
                tip=_row_tip(row),
                tier=tier.strip().lower() if isinstance(tier, str) else "",
                catalogued=str(row["id"]) in roles,
            )
        )
    return tuple(found)


def split(choices_: tuple[GenerateChoice, ...]) -> tuple[tuple[GenerateChoice, ...], tuple[GenerateChoice, ...]]:
    """The compact (16 GB-class) and full-size (24 GB+) section rows, in order."""
    compact = tuple(choice for choice in choices_ if choice.tier != FULL_TIER)
    full = tuple(choice for choice in choices_ if choice.tier == FULL_TIER)
    return compact, full


def preselect(choices_: tuple[GenerateChoice, ...], tier: Tier | None, stored: str = "") -> str:
    """The id the picker selects.

    The stored pick wins while its model is still installed. Otherwise the
    first row of the section that fits the detected card is selected — on a
    16 GB card never a full-size row. Empty string when nothing is installed.
    """
    if not choices_:
        return ""
    by_id = {choice.id: choice for choice in choices_}
    if stored and stored in by_id:
        return stored
    want_full = tier is Tier.GB24
    for choice in choices_:
        if (choice.tier == FULL_TIER) is want_full:
            return choice.id
    return choices_[0].id


def picked_spec(catalog: Catalog, stored: str) -> ModelSpec | None:
    """The catalog row for a stored pick, or None when it cannot plan it.

    Only generator rows count: a stale or foreign stored value never routes a
    picture onto an edit or safety model.
    """
    if not stored:
        return None
    try:
        spec = catalog.by_id(stored)
    except RegistryError:
        return None
    return spec if spec.role == "generator" else None


def implied_family(model_id: str, catalog: Catalog | None = None) -> str:
    """The shared ``model.default`` family a pick clearly implies, else "".

    Only catalogued rows imply one: their text base maps onto common's family
    ids (``z-image-turbo`` or ``moss``). Creative and not-yet-catalogued builds
    imply nothing, so the shared family is left alone.
    """
    if catalog is None:
        try:
            catalog = load_catalog()
        except (RegistryError, OSError, ValueError):
            return ""
    spec = picked_spec(catalog, model_id)
    if spec is None:
        return ""
    return {"z-image-turbo": "z-image-turbo", "moss-image-sft": "moss"}.get(spec.text_base, "")
