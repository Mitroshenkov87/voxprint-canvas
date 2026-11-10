"""Model Forge LoRAs under the shared models folder.

A folder is listed when its record matches a text base this app
runs. One adapter is loaded on the transformer at a time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from canvas.infra.paths import models_dir
from canvas.models.default_base import selected_text_base
from canvas.models.registry import load_catalog

SCHEMA = "voxprint-model/1"
ZIMAGE_TURBO_MDL_ID = "z-image-turbo"
FORGE_BASE_ID = "z-image"
# LoRA base tag for a MOSS adapter. The catalog text base stays ``moss-image-sft``.
MOSS_LORA_BASE = "moss5b"
MOSS_IMAGE_SFT = "moss-image-sft"
# Expected ``voxprint.lora.base_variant`` on a moss5b file. ``moss-image-base`` still loads.
EXPECTED_BASE_VARIANT = "moss-image-sft"
_VARIANT_BASE = "moss-image-base"
# User-facing values and older header ids. All of them normalise to ``moss5b``.
_MOSS_TAGS = frozenset({
    "moss",
    "moss-image",
    "moss-image-base",
    "moss-image-sft",
    "moss5b",
})
# Forge family strings. Full Z-Image is ``z-image``; Turbo is ``z-image-turbo``.
# Forge never trains on Turbo. The legacy short tags are ``zimg6b`` and ``moss5b``.
FAMILY_ZIMAGE = "z-image"
FAMILY_TURBO = "z-image-turbo"
ZIMAGE_SHORT_TAG = "zimg6b"
# Default strength for a Z-Image family LoRA on Turbo. No auto-scaling.
ZIMAGE_LORA_DEFAULT_STRENGTH = 1.0
_ZIMAGE_FULL_TAGS = frozenset({FAMILY_ZIMAGE, ZIMAGE_SHORT_TAG})
_TURBO_TAGS = frozenset({FAMILY_TURBO})
_HEADER_TRAINED_BASE = "voxprint.lora.trained_base"
_HEADER_COMPATIBLE_BASES = "voxprint.lora.compatible_bases"
# Strength for a LoRA whose record names none.
CLI_DEFAULT_STRENGTH = 0.7
CLI_STRENGTH_RANGE = (0.0, 2.0)
CODE_BASE_MISMATCH = "lora.base_mismatch"
CODE_INVALID = "lora.invalid"
CODE_SCALE = "lora.scale"
CODE_RETRAIN = "lora.retrain"
CODE_NEEDS_OPTIONAL_MOSS = "lora.needs_optional_moss"
_KINDS = frozenset({"person", "object", "style"})
_ADAPTER = "canvas-lora"
_A_WEIGHT = ".lora_A.weight"
_ALPHA = ".alpha"
_KOHYA = "lora_unet_"
THIRD_PARTY_KIND = "unknown"
_B_WEIGHT = ".lora_B.weight"
_HEADER_TRIGGER = "voxprint.subject.trigger"
_HEADER_ALIAS = "voxprint.subject.alias"
_HEADER_STRENGTH = "voxprint.lora.recommended_strength"
_HEADER_KIND = "voxprint.lora.kind"
_HEADER_BASE = "voxprint.lora.base"
_HEADER_BASE_VARIANT = "voxprint.lora.base_variant"
_HEADER_BASE_SHA256 = "voxprint.lora.base_sha256"
_HEADER_BASE_SHA256_ALT = "base_sha256"
_HEADER_CONSENT = "voxprint.subject.consent_obtained"
_HEADER_CONFIRMED = "voxprint.subject.adult_confirmed"
_ADULT_LABEL = "voxprint.lora.adult_content"
_LOG = logging.getLogger(__name__)
# Older Model Forge exports wrote "thing" for an object LoRA; a .voxlora package maps it the same way.
_HEADER_KIND_ALIAS = {"thing": "object"}
_HEADER_KEYS = (
    _HEADER_TRIGGER,
    _HEADER_ALIAS,
    _HEADER_STRENGTH,
    _HEADER_KIND,
    _HEADER_BASE,
    _HEADER_TRAINED_BASE,
    _HEADER_COMPATIBLE_BASES,
    _HEADER_BASE_VARIANT,
    _HEADER_BASE_SHA256,
    _HEADER_BASE_SHA256_ALT,
    _HEADER_CONSENT,
    _HEADER_CONFIRMED,
    _ADULT_LABEL,
)
_SIDECAR = "voxprint-model.json"

REASON_RECORD = "canvas.addons.reason.record"
REASON_SCHEMA = "canvas.addons.reason.schema"
REASON_TYPE = "canvas.addons.reason.type"
REASON_BASE = "canvas.addons.reason.base"
REASON_FILE = "canvas.addons.reason.file"
REASON_PERSON = "canvas.addons.reason.person"
REASON_INCOMPLETE = "canvas.addons.reason.incomplete"
REASON_PREVIEW = "canvas.addons.reason.preview"


@dataclass(frozen=True)
class UserLora:
    """One LoRA the add-ons list can select."""

    id: str
    kind: str
    folder: Path
    weights: Path
    trigger: str
    default_strength: float
    strength_range: tuple[float, float]
    adult_content: bool
    name: str = ""
    previews: tuple[Path, ...] = ()
    base: str = ""
    sha256: str = ""
    base_variant: str = ""
    base_sha256: str = ""
    warnings: tuple[str, ...] = ()
    alias: str = ""


@dataclass(frozen=True)
class SkippedLora:
    """A folder that stays out of the selectable list, with a catalog reason."""

    name: str
    reason_key: str


@dataclass(frozen=True)
class LoraIndex:
    """Selectable LoRAs, then folders that failed the record check."""

    available: tuple[UserLora, ...]
    skipped: tuple[SkippedLora, ...]


class LoraRejected(Exception):
    """A LoRA file cannot be loaded. ``code`` is a stable catalog key.

    ``details`` fills the catalog sentence. ``lora.retrain`` carries ``base``.
    """

    def __init__(self, code: str, **details: str) -> None:
        super().__init__(code)
        self.code = code
        self.details = details


def discover(root: Path | None = None) -> LoraIndex:
    """Read ``user-models/lora`` under the shared models folder.

    Args:
        root: Models folder. The shared models folder is used when omitted.
    """
    base = _lora_root(root)
    if not base.is_dir():
        return LoraIndex((), ())
    available: list[UserLora] = []
    skipped: list[SkippedLora] = []
    for folder in sorted((path for path in base.iterdir() if path.is_dir()), key=lambda path: path.name):
        if folder.name.startswith("."):
            continue
        found, reason = inspect_folder(folder)
        if found is None:
            skipped.append(SkippedLora(folder.name, reason or REASON_RECORD))
        else:
            available.append(found)
    available.sort(key=lambda item: (item.adult_content, item.id))
    return LoraIndex(tuple(available), tuple(skipped))


def inspect_folder(folder: Path) -> tuple[UserLora | None, str | None]:
    """Return a listed LoRA, or a catalog reason key when the folder is skipped.

    A LoRA trained on another base model is listed too; its trained base id is
    recorded on the entry (see :func:`base_choice`).

    Args:
        folder: One ``<id>`` directory.
    """
    document = _document(folder / "voxprint-model.json")
    if document is None:
        return None, REASON_RECORD
    schema_reason = _schema_reason(document)
    if schema_reason is not None:
        return None, schema_reason
    if document.get("type") != "lora":
        return None, REASON_TYPE
    kind = document.get("kind")
    lora_id = document.get("id")
    if kind not in _KINDS or not isinstance(lora_id, str) or lora_id != folder.name or _bad_id(lora_id):
        return None, REASON_RECORD
    weights_name = document.get("file")
    if weights_name != f"{lora_id}.safetensors":
        return None, REASON_FILE
    weights = folder / str(weights_name)
    if not _inside(folder, weights) or not _inside(folder, folder / "LICENSE") or not _inside(folder, folder / "NOTICE"):
        return None, REASON_INCOMPLETE
    base = document.get("base")
    mdl_id = str(base.get("mdl_id") or "") if isinstance(base, dict) else ""
    if not mdl_id:
        return None, REASON_BASE
    file_reason = _file_reason(document, weights)
    if file_reason is not None:
        return None, file_reason
    strength = _strength(document)
    if strength is None:
        return None, REASON_RECORD
    default_strength, low, high = strength
    adult = _adult_label(document)
    if adult is None:
        return None, REASON_RECORD
    preview_reason = _preview_reason(document, folder)
    if preview_reason is not None:
        return None, preview_reason
    header = _header(weights)
    if header is None:
        return None, REASON_RECORD
    header_kind = header.get(_HEADER_KIND)
    if header_kind is not None and header_kind != kind:
        return None, REASON_RECORD
    if kind == "person" and not _person_flags(header):
        return None, REASON_PERSON
    licence = document.get("licence")
    if not isinstance(licence, str) or not licence.strip():
        return None, REASON_RECORD
    variant, base_sha, warnings = _provenance(header, weights, mdl_id)
    return UserLora(
        lora_id,
        str(kind),
        folder,
        weights,
        header.get(_HEADER_TRIGGER, ""),
        recommended_or_default(header.get(_HEADER_STRENGTH)),
        (low, high),
        adult,
        base=mdl_id,
        sha256=str(document.get("sha256") or ""),
        base_variant=variant,
        base_sha256=base_sha,
        warnings=warnings,
        alias=parse_alias(header.get(_HEADER_ALIAS)) if header.get(_HEADER_TRIGGER, "").strip() else "",
    ), None


def resolve_selected(path: str | Path, root: Path | None = None) -> UserLora | None:
    """Return the listed LoRA for a folder, a weight file, or an id.

    Args:
        path: Expert field value. Empty means nothing is selected.
        root: Models folder used when ``path`` is a bare id.
    """
    text = str(path).strip()
    if not text:
        return None
    from canvas.models.lora_library import resolve_library

    imported = resolve_library(text)
    if imported is not None:
        return _unique_alias(imported, root)
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = _lora_root(root) / text
    if candidate.is_file() and candidate.suffix == ".safetensors":
        found, _reason = inspect_folder(candidate.parent)
        if found is not None and _same_file(found.weights, candidate):
            return _unique_alias(found, root)
        try:
            return _unique_alias(open_weights(candidate), root)
        except LoraRejected:
            return None
    if candidate.is_file():
        candidate = candidate.parent
    found, _reason = inspect_folder(candidate)
    return None if found is None else _unique_alias(found, root)


def installed_loras(
    root: Path | None = None, library: Path | None = None,
) -> tuple[tuple[UserLora, ...], tuple[tuple[str, str, str], ...]]:
    """Library LoRAs, then the listed add-ons, with aliases made unique, and a note per renamed alias.

    Args:
        root: Models folder holding the add-on folders.
        library: Library folder. The app's library when omitted.
    """
    from canvas.models.lora_library import list_library

    items: list[UserLora] = []
    for entry in list_library(library):
        try:
            items.append(entry.as_lora())
        except LoraRejected:
            continue
    items.extend(discover(root).available)
    return resolve_aliases(tuple(items))


def _unique_alias(found: UserLora, root: Path | None) -> UserLora:
    """``found`` with the alias it was given across the installed set, so two LoRAs never share one."""
    if not found.alias:
        return found
    try:
        listed, _notes = installed_loras(root)
    except OSError:
        return found
    for item in listed:
        if item.id == found.id and _same_file(item.weights, found.weights):
            return replace(found, alias=item.alias)
    # A file chosen by path that is not installed: it yields to every installed alias.
    taken = {item.alias for item in listed if item.alias}
    if found.alias not in taken:
        return found
    return resolve_aliases((*listed, found))[0][-1]


def _strip_revision(token: str) -> str:
    """Drop a ``@revision`` suffix from a Forge ``trained_base`` value."""
    text = token.strip()
    if "@" in text:
        text = text.split("@", 1)[0].strip()
    return text


def family_of(token: str) -> str:
    """Return the Forge family id for a catalog base, LoRA tag, or trained_base.

    ``z-image@04cc4abb`` and the short tag ``zimg6b`` are ``z-image``.
    MOSS aliases are the catalog family ``moss-image-sft``. Unknown tokens
    are returned case-folded, so ``sdxl`` stays ``sdxl``.
    """
    raw = _strip_revision(token)
    if not raw:
        return ""
    folded = raw.casefold()
    if folded in _MOSS_TAGS:
        return MOSS_IMAGE_SFT
    if folded in _ZIMAGE_FULL_TAGS:
        return FAMILY_ZIMAGE
    if folded in _TURBO_TAGS:
        return FAMILY_TURBO
    return folded


def parse_compatible_bases(raw: object) -> tuple[str, ...] | None:
    """Parse ``voxprint.lora.compatible_bases``. Malformed values are absent."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw, list):
        value = raw
    else:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return tuple(value)


def _legacy_compatible_families(trained: str) -> frozenset[str]:
    """Families a LoRA may load on when ``compatible_bases`` is absent.

    Legacy ``voxprint.lora.base="z-image"`` or the short tag ``zimg6b``
    (and an untagged or Turbo-tagged Z-Image file) is compatible with both
    ``z-image`` and ``z-image-turbo``. Legacy ``moss-image-base`` / ``moss5b``
    is compatible only with MOSS.
    """
    family = family_of(trained)
    if family in {FAMILY_ZIMAGE, FAMILY_TURBO}:
        return frozenset({FAMILY_ZIMAGE, FAMILY_TURBO})
    if family == MOSS_IMAGE_SFT:
        return frozenset({MOSS_IMAGE_SFT})
    return frozenset()


def _lora_fields(path: Path | None, recorded: str = "") -> dict[str, object]:
    """Safetensors header fields, filled from a sidecar when a key is missing."""
    fields: dict[str, object] = dict(_string_metadata(path) if path is not None else {})
    if path is not None:
        sidecar = path.parent / _SIDECAR
        document = _document(sidecar) if sidecar.is_file() else None
        if isinstance(document, dict):
            for key in (
                _HEADER_TRAINED_BASE,
                _HEADER_COMPATIBLE_BASES,
                _HEADER_BASE,
                "trained_base",
                "compatible_bases",
            ):
                if not fields.get(key) and key in document:
                    fields[key] = document[key]
            base = document.get("base")
            if not fields.get(_HEADER_BASE) and isinstance(base, dict):
                mdl_id = base.get("mdl_id")
                if isinstance(mdl_id, str) and mdl_id.strip():
                    fields[_HEADER_BASE] = mdl_id
    if recorded and not fields.get(_HEADER_BASE) and not fields.get(_HEADER_TRAINED_BASE):
        fields[_HEADER_BASE] = recorded
    return fields


def lora_accepts_family(lora: UserLora | Path | str, loaded: str) -> bool:
    """True when ``lora`` may load on the running Forge family ``loaded``.

    When ``voxprint.lora.compatible_bases`` is present and well-formed, the
    loaded family must be in that list. Otherwise the legacy single-tag
    check applies.
    """
    if isinstance(lora, UserLora):
        path, recorded = lora.weights, str(lora.base or "")
    else:
        path, recorded = Path(lora), ""
    fields = _lora_fields(path, recorded)
    compatible = parse_compatible_bases(
        fields.get(_HEADER_COMPATIBLE_BASES) or fields.get("compatible_bases")
    )
    if compatible is not None:
        return loaded in compatible
    return loaded in _legacy_compatible_families(detect_base(lora))


def _catalog_base_for_family(loaded: str, text_bases: tuple[str, ...]) -> str | None:
    """The catalog text base that runs family ``loaded``, if the catalog lists one."""
    for candidate in text_bases:
        if family_of(candidate) == loaded:
            return candidate
    return None


def _is_zimage_family(token: str) -> bool:
    """True when ``token`` names full Z-Image or Z-Image-Turbo."""
    return family_of(token) in {FAMILY_ZIMAGE, FAMILY_TURBO}


def _is_moss_tag(value: str) -> bool:
    """True when ``value`` is a MOSS LoRA tag or a user-facing alias of one."""
    token = value.strip()
    folded = token.casefold()
    return token in _MOSS_TAGS or folded in _MOSS_TAGS


def _is_zimage_tag(value: str) -> bool:
    """True when ``value`` is a Z-Image family tag (full, Turbo, or ``zimg6b``)."""
    return _is_zimage_family(value)


def base_matches(mdl_id: str, expected: str | None = None) -> bool:
    """True when ``mdl_id`` is the base this app runs.

    Model Forge records a Z-Image LoRA as ``z-image``. That id matches the
    Z-Image-Turbo base. A MOSS LoRA's tag is ``moss5b``. The user-facing
    values ``moss`` and ``moss-image``, and the older ids ``moss-image-sft``
    and ``moss-image-base``, match ``moss5b`` and the ``moss-image-sft``
    catalog base.

    Args:
        mdl_id: ``base.mdl_id`` or the ``voxprint.lora.base`` header value.
        expected: Base id to accept. The running base is used when omitted.
    """
    if expected is None:
        expected = running_base_id()
    if not expected or not mdl_id:
        return False
    if mdl_id == expected:
        return True
    if _is_zimage_tag(expected) and _is_zimage_tag(mdl_id):
        return True
    return _is_moss_tag(expected) and _is_moss_tag(mdl_id)


def cli_scale(value: object) -> float:
    """Command-line LoRA strength. Values stay inside 0..2.

    Args:
        value: The ``--lora-scale`` number.

    Raises:
        LoraRejected: The value is outside 0..2. ``code`` is ``lora.scale``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LoraRejected(CODE_SCALE)
    scale = float(value)
    if scale < 0 or scale > 2:
        raise LoraRejected(CODE_SCALE)
    return scale


def prepare_cli_lora(path: str | Path) -> UserLora:
    """Open one LoRA for a command. Strength stays inside 0..2.

    Args:
        path: A ``.safetensors`` file, a ``.voxlora`` package, an add-on folder,
            or the name of an imported LoRA.

    Raises:
        LoraRejected: The file is unreadable, its base does not match, or a
            matrix does not fit its Z-Image-Turbo layer.
    """
    opened = _cli_lora(path)
    fits_layers(opened.weights)
    return opened


def _cli_lora(path: str | Path) -> UserLora:
    from canvas.models.lora_library import PACKAGE_SUFFIX, open_package_weights, resolve_library

    imported = resolve_library(str(path))
    if imported is not None:
        return replace(imported, strength_range=CLI_STRENGTH_RANGE)
    candidate = Path(path)
    if candidate.is_file() and candidate.suffix == PACKAGE_SUFFIX:
        opened = open_package_weights(candidate)
        return replace(opened, strength_range=CLI_STRENGTH_RANGE)
    if candidate.is_dir():
        found, _reason = inspect_folder(candidate)
        if found is None:
            raise LoraRejected(CODE_INVALID)
        weights = _loader_weights(found.weights)
        return replace(
            found,
            weights=weights,
            strength_range=CLI_STRENGTH_RANGE,
        )
    if not candidate.is_file():
        raise LoraRejected(CODE_INVALID)
    if candidate.suffix == ".safetensors" and not _forge_file(candidate):
        opened = open_third_party(candidate)
    else:
        opened = open_weights(candidate)
    return replace(
        opened,
        strength_range=CLI_STRENGTH_RANGE,
    )


def _forge_file(path: Path) -> bool:
    """True when the header or a matching sidecar marks a Model Forge export."""
    header = _header(path)
    if header is None:
        raise LoraRejected(CODE_INVALID)
    if header.get(_HEADER_BASE) or header.get(_HEADER_TRAINED_BASE) or parse_compatible_bases(header.get(_HEADER_COMPATIBLE_BASES)):
        return True
    sidecar = path.parent / _SIDECAR
    document = _document(sidecar) if sidecar.is_file() else None
    return document is not None and document.get("file") == path.name


def open_weights(path: Path) -> UserLora:
    """Read one Forge ``.safetensors`` file and the sidecar beside it, when present.

    A LoRA trained on another base model opens all the same; its trained base
    is recorded on the entry.

    Args:
        path: The weight file. The default Forge path is ``project/export/<id>.safetensors``.

    Raises:
        LoraRejected: The file is unreadable.
    """
    meta, weights = _opened_weights(path)
    base = meta.get(_HEADER_BASE, "")
    trained = str(meta.get(_HEADER_TRAINED_BASE) or "").strip()
    compatible = parse_compatible_bases(meta.get(_HEADER_COMPATIBLE_BASES))
    if not base and not trained and compatible is None:
        raise LoraRejected(CODE_INVALID)
    stored_base = base or (_canonical_base(trained) if trained else "")
    sidecar = _matching_sidecar(path)
    kind = _HEADER_KIND_ALIAS.get(meta.get(_HEADER_KIND, ""), meta.get(_HEADER_KIND, ""))
    lora_id = path.stem
    default_strength = (
        ZIMAGE_LORA_DEFAULT_STRENGTH if _is_zimage_family(stored_base or trained or FAMILY_ZIMAGE) else CLI_DEFAULT_STRENGTH
    )
    low, high = CLI_STRENGTH_RANGE
    adult = meta.get(_ADULT_LABEL) == "true"
    if sidecar is not None:
        kind = str(sidecar.get("kind") or kind)
        recorded = sidecar.get("id")
        if isinstance(recorded, str) and recorded.strip() and not _bad_id(recorded):
            lora_id = recorded
        strength = _strength(sidecar)
        if strength is None:
            raise LoraRejected(CODE_INVALID)
        default_strength, low, high = strength
        labelled = _adult_label(sidecar)
        if labelled is None:
            raise LoraRejected(CODE_INVALID)
        adult = labelled
    if not kind:
        # Raw Forge checkpoints record voxprint.lora.base but not kind.
        # A Kohya file that only claims a Forge base header is still invalid.
        if _has_kohya_keys(path):
            raise LoraRejected(CODE_INVALID)
        kind = THIRD_PARTY_KIND
    if kind not in _KINDS and kind != THIRD_PARTY_KIND:
        raise LoraRejected(CODE_INVALID)
    if _bad_id(lora_id):
        raise LoraRejected(CODE_INVALID)
    variant, base_sha, warnings = _provenance(meta, path, stored_base)
    return UserLora(
        lora_id,
        kind,
        path.parent,
        weights,
        meta.get(_HEADER_TRIGGER, ""),
        recommended_or_default(meta.get(_HEADER_STRENGTH)),
        (low, high),
        adult,
        base=stored_base,
        sha256=_sha256(path),
        base_variant=variant,
        base_sha256=base_sha,
        warnings=warnings,
        alias=parse_alias(meta.get(_HEADER_ALIAS)) if meta.get(_HEADER_TRIGGER, "").strip() else "",
    )


_ALIAS_RE = re.compile(r"vox_[a-z0-9_]+")
ALIAS_MAX = 32
_STRENGTH_TEXT = re.compile(r"\s*[0-9]+(?:\.[0-9]+)?\s*")


def parse_alias(value: object) -> str:
    """The alias in a weight header, or empty when it is absent or not ``vox_<slug>``.

    The slug is lowercase ASCII letters, digits, and underscores. At most 32 characters in all.
    """
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if len(text) > ALIAS_MAX or len(text) == len("vox_") or _ALIAS_RE.fullmatch(text) is None:
        return ""
    return text


DEFAULT_RECOMMENDED_STRENGTH = 1.0


def recommended_or_default(value: object) -> float:
    """The recommended strength key when it parses, else 1.0 (not the record's or the family's default)."""
    found = parse_recommended_strength(value)
    return DEFAULT_RECOMMENDED_STRENGTH if found is None else found


def parse_recommended_strength(value: object) -> float | None:
    """A decimal string such as ``"1.0"`` as a strength, or None when absent or unparsable."""
    if not isinstance(value, str) or _STRENGTH_TEXT.fullmatch(value) is None:
        return None
    number = float(value)
    return number if 0.0 < number <= 4.0 else None


def chip_word(lora: UserLora | None) -> str:
    """The word the chip shows and inserts: the alias when there is one, else the trigger."""
    if lora is None:
        return ""
    return lora.alias or lora.trigger


def _word_pattern(word: str) -> re.Pattern[str]:
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])", re.IGNORECASE)


def rewrite_alias(text: str, lora: UserLora | None) -> str:
    """``text`` with the LoRA's alias replaced by its trigger. Unchanged without both."""
    if lora is None or not lora.alias or not lora.trigger.strip():
        return text
    trigger = lora.trigger.strip()
    return _word_pattern(lora.alias).sub(lambda _match: trigger, text)


def apply_trigger(prompt: str, lora: UserLora | None) -> str:
    """The final prompt for a selected LoRA: alias replaced by the trigger, trigger led in when missing.

    The alias rewrite runs first, then the insert check. A trigger already in
    the prompt is not added again. A LoRA without a trigger leaves the prompt as it is.
    Run it on the composed prompt that is encoded and reviewed, not on the prompt box.
    """
    trigger = (lora.trigger if lora is not None else "").strip()
    if lora is None or not trigger:
        return prompt
    text = rewrite_alias(prompt, lora)
    if _word_pattern(trigger).search(text) is not None:
        return text
    body = text.strip()
    return f"{trigger}, {body}" if body else trigger


def resolve_aliases(loras: tuple[UserLora, ...]) -> tuple[tuple[UserLora, ...], tuple[tuple[str, str, str], ...]]:
    """Make aliases unique across a set. A later clash gets ``_2``, ``_3`` and so on.

    Returns:
        The LoRAs with their resolved alias, and a ``(lora id, wanted, given)`` note per clash.
    """
    taken: set[str] = set()
    out: list[UserLora] = []
    notes: list[tuple[str, str, str]] = []
    for item in loras:
        alias = item.alias
        if alias:
            given = alias
            count = 1
            while given in taken:
                count += 1
                suffix = f"_{count}"
                given = alias[: ALIAS_MAX - len(suffix)] + suffix
            taken.add(given)
            if given != alias:
                notes.append((item.id, alias, given))
                item = replace(item, alias=given)
        out.append(item)
    return tuple(out), tuple(notes)


def running_base_id() -> str:
    """The shared ``model.default`` mapped to a Canvas text base.

    An untagged LoRA is a Z-Image LoRA. :func:`detect_base` applies that.
    This function does not.
    """
    return selected_text_base()


def strength_or_default(lora: UserLora, requested: object) -> float:
    """The strength to use for ``lora``: the requested number, else the LoRA's recorded default.

    Args:
        lora: The selected LoRA.
        requested: The strength a screen or command carries; None when nothing was chosen.
    """
    if isinstance(requested, bool) or not isinstance(requested, (int, float)):
        return lora.default_strength
    return float(requested)


_META_ARCHITECTURE = "modelspec.architecture"
_META_BASE_VERSION = "ss_base_model_version"
_FILENAME_BASES = (
    (".moss5b.safetensors", MOSS_LORA_BASE),
    (".moss.safetensors", MOSS_LORA_BASE),
    (".zimg6b.safetensors", FAMILY_ZIMAGE),
    (".z-image.safetensors", FAMILY_ZIMAGE),
)


def _canonical_base(raw: str) -> str:
    """``moss5b``, ``z-image``, or ``z-image-turbo`` when ``raw`` is one of those families.

    ``moss`` and ``moss-image`` are the user-facing MOSS values. Older headers
    ``moss-image-sft`` and ``moss-image-base`` normalise to ``moss5b`` too.
    ``z-image@04cc4abb`` and the short tag ``zimg6b`` become ``z-image``.
    Any other id is kept, so ``sdxl`` stays ``sdxl``.
    """
    token = _strip_revision(raw)
    family = family_of(token)
    if family == MOSS_IMAGE_SFT:
        return MOSS_LORA_BASE
    if family in {FAMILY_ZIMAGE, FAMILY_TURBO}:
        return family
    return token


def _detect_target(path_or_lora: Path | UserLora | str) -> tuple[Path | None, str]:
    """The weight file and a recorded base id, when the caller passed a LoRA."""
    if isinstance(path_or_lora, UserLora):
        return path_or_lora.weights, str(path_or_lora.base or "").strip()
    return Path(path_or_lora), ""


def _string_metadata(path: Path | None) -> dict[str, str]:
    """String entries of a safetensors header. Empty when the file cannot be read."""
    if path is None or not path.is_file():
        return {}
    opened = _safetensors(path)
    if opened is None:
        return {}
    raw = opened[0].get("__metadata__") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if isinstance(key, str) and isinstance(value, str)}


def _filename_base(path: Path | None) -> str:
    """A ``<name>.moss`` or ``<name>.z-image`` tag on the weight file name."""
    if path is None:
        return ""
    name = path.name.casefold()
    for suffix, base in _FILENAME_BASES:
        if name.endswith(suffix):
            return base
    return ""


def _family_token(value: str) -> str:
    """A MOSS or Z-Image family id found inside one metadata string.

    A string that names MOSS, including the user-facing values ``moss`` and
    ``moss-image``, is the ``moss5b`` tag. ``zimg6b`` and a bare ``z-image``
    are the full Z-Image family. A Turbo name stays ``z-image-turbo``.
    """
    folded = value.casefold()
    if not folded:
        return ""
    if "moss" in folded:
        return MOSS_LORA_BASE
    compact = folded.replace("-", "").replace("_", "").replace(" ", "")
    if "zimg6b" in compact:
        return FAMILY_ZIMAGE
    if "zimageturbo" in compact or "z-image-turbo" in folded:
        return FAMILY_TURBO
    if "z-image" in folded or "zimage" in compact:
        return FAMILY_ZIMAGE
    return ""


def _metadata_family(meta: dict[str, str]) -> str:
    """The family named by ``modelspec.architecture`` or ``ss_base_model_version``."""
    for key in (_META_ARCHITECTURE, _META_BASE_VERSION):
        found = _family_token(meta.get(key, ""))
        if found:
            return found
    return ""


def _trained_tag(meta: dict[str, str], path: Path | None, recorded: str = "") -> str:
    """The normalised base tag for one header, filename, or recorded id."""
    trained = str(meta.get(_HEADER_TRAINED_BASE) or meta.get("trained_base") or "").strip()
    if trained:
        return _canonical_base(trained)
    header = str(meta.get(_HEADER_BASE) or "").strip() or recorded.strip()
    if header:
        return _canonical_base(header)
    found = _metadata_family(meta)
    if found:
        return found
    named = _filename_base(path)
    if named:
        return named
    return FAMILY_ZIMAGE


def detect_base(path_or_lora: Path | UserLora | str) -> str:
    """The base a LoRA was trained on.

    ``voxprint.lora.trained_base`` wins when present (``z-image@04cc4abb``
    becomes ``z-image``). Otherwise ``voxprint.lora.base`` wins, including
    over a filename tag, and is normalised to ``moss5b``, ``z-image``, or
    ``z-image-turbo``. ``moss`` and ``moss-image`` both become ``moss5b``.
    ``zimg6b`` becomes ``z-image``. Any other header value is kept, so
    ``sdxl`` stays ``sdxl``. A recorded base on a :class:`UserLora` is that
    header when the file does not carry one.

    Otherwise ``modelspec.architecture`` and ``ss_base_model_version`` are
    read: ``moss`` is ``moss5b``, a Turbo name is ``z-image-turbo``, and
    ``z-image`` or ``zimg6b`` is ``z-image``. A filename
    ``<name>.moss.safetensors``, ``<name>.moss5b.safetensors``,
    ``<name>.z-image.safetensors``, or ``<name>.zimg6b.safetensors`` is used
    when those strings are absent. An untagged file, including an older LoRA,
    is the Z-Image family (``z-image``) and still loads on Turbo.

    Args:
        path_or_lora: A weight file, or a LoRA whose ``weights`` file is read.
    """
    path, recorded = _detect_target(path_or_lora)
    fields = _lora_fields(path, recorded)
    meta = {key: value for key, value in fields.items() if isinstance(key, str) and isinstance(value, str)}
    return _trained_tag(meta, path, recorded)


def _variant_warnings(tag: str, variant: str) -> tuple[str, ...]:
    """A warning when a moss5b LoRA names the pre-SFT base. The file still loads."""
    written = variant.strip()
    if tag != MOSS_LORA_BASE or written.casefold() != _VARIANT_BASE:
        return ()
    message = (
        f"LoRA base tag is {MOSS_LORA_BASE} but {_HEADER_BASE_VARIANT} is {written}; "
        f"expected {EXPECTED_BASE_VARIANT}. The file still loads."
    )
    _LOG.warning("%s", message)
    return (message,)


def _provenance(meta: dict[str, str], path: Path, recorded: str = "") -> tuple[str, str, tuple[str, ...]]:
    """``base_variant``, ``base_sha256``, and warnings from one safetensors header.

    ``voxprint.lora.base_sha256`` wins. A file that only carries ``base_sha256``
    is read the same way.
    """
    variant = meta.get(_HEADER_BASE_VARIANT, "").strip()
    digest = meta.get(_HEADER_BASE_SHA256, "").strip() or meta.get(_HEADER_BASE_SHA256_ALT, "").strip()
    return variant, digest, _variant_warnings(_trained_tag(meta, path, recorded), variant)


@dataclass(frozen=True)
class LoraRoute:
    """Where a LoRA is allowed to run.

    Attributes:
        base: A text base this app can run. Empty when the LoRA must not load.
        retrain_from: The trained base when ``base`` is empty.
    """

    base: str
    retrain_from: str = ""


def route_lora(lora: UserLora, text_bases: tuple[str, ...] | None = None, default: str | None = None) -> LoraRoute:
    """Route ``lora`` onto the loaded family when that family is compatible.

    When ``voxprint.lora.compatible_bases`` is present, the LoRA loads only if
    the running family is in that list. Otherwise the legacy tag check
    applies: ``z-image`` / ``zimg6b`` load on full Z-Image and on Turbo;
    ``moss5b`` / ``moss-image-base`` load only on MOSS. Any other combination
    is a retrain.

    Args:
        lora: The selected LoRA.
        text_bases: Text bases the catalog lists. The catalog is read when omitted.
        default: The loaded (running) text base. The shared ``model.default``
            is used when omitted.

    Returns:
        ``LoraRoute(base)`` when the LoRA may load on the running family.
        ``LoraRoute("", retrain_from=trained)`` otherwise. ``retrain_from`` is
        the normalised tag, so a MOSS file reports ``moss5b``.
    """
    if text_bases is None:
        catalog = load_catalog()
        text_bases = catalog.text_bases
    if default is None:
        default = selected_text_base()
    trained = detect_base(lora)
    loaded = family_of(default)
    if not loaded or not lora_accepts_family(lora, loaded):
        return LoraRoute("", trained)
    target = _catalog_base_for_family(loaded, text_bases)
    if target is None:
        return LoraRoute("", trained)
    return LoraRoute(target)


def moss_weights_ready(root: Path | None = None) -> bool:
    """True when a MOSS-Image generator snapshot is already on disk.

    This does not download or queue a fetch. Missing or unreadable weights
    count as not ready, so an optional MOSS LoRA can show a download prompt.
    """
    try:
        from canvas.models.snapshots import snapshot_ready
        from canvas.models.verified import flush_process_cache, process_cache

        catalog = load_catalog()
        models = root if root is not None else models_dir()
        cache = process_cache()
        try:
            for spec in catalog.models:
                if spec.role != "generator" or spec.text_base != MOSS_IMAGE_SFT:
                    continue
                if snapshot_ready(spec, models, cache) is not None:
                    return True
        finally:
            flush_process_cache()
    except Exception:  # noqa: BLE001 - an unreadable install counts as missing
        return False
    return False


def retrain_rejection(
    lora: UserLora, text_bases: tuple[str, ...] | None = None, default: str | None = None,
) -> LoraRejected | None:
    """A refusal when ``lora`` cannot run on an installed base. None when it can."""
    route = route_lora(lora, text_bases, default)
    if not route.base:
        return LoraRejected(CODE_RETRAIN, base=route.retrain_from)
    if route.base == MOSS_IMAGE_SFT and not moss_weights_ready():
        return LoraRejected(CODE_NEEDS_OPTIONAL_MOSS, base=route.base)
    return None


@dataclass(frozen=True)
class BaseChoice:
    """Which base a LoRA runs on, and the trained base to mention when it cannot.

    Attributes:
        run_on: The text base the job uses. Empty when the LoRA must not load.
        note_base: :attr:`LoraRoute.retrain_from`. Empty when ``run_on`` is set.
    """

    run_on: str
    note_base: str = ""


def base_choice(lora: UserLora, text_bases: tuple[str, ...] | None = None, default: str | None = None) -> BaseChoice:
    """The route for ``lora``. ``note_base`` is the trained base when a retrain is required.

    Args:
        lora: The selected LoRA.
        text_bases: Text bases the catalog lists. The catalog is read when omitted.
        default: The default text base. Passed through to :func:`route_lora`.
    """
    route = route_lora(lora, text_bases, default)
    return BaseChoice(route.base, route.retrain_from)


def clamp_strength(lora: UserLora, requested: float) -> float:
    """Keep a requested strength inside the LoRA's recorded range.

    Args:
        lora: The selected LoRA.
        requested: The expert LoRA strength field.
    """
    requested = strength_or_default(lora, requested)
    low, high = lora.strength_range
    return min(max(float(requested), low), high)


@dataclass(frozen=True)
class LoraApplyResult:
    """What ``apply_bound_lora`` loaded onto the transformer."""

    lora_id: str
    alias: str
    strength: float
    base: str
    modules_patched: int


def apply_bound_lora(
    lora: UserLora,
    strength: float,
    *,
    holder: object | None = None,
    transformer: object | None = None,
) -> LoraApplyResult:
    """Set the bound LoRA and strength, then load and fuse or apply the adapter.

    GUI and CLI both go through this function. ``holder`` is the generator,
    runner, or pipeline that should remember the selection. When a transformer
    is available (passed in, or on ``holder``), the adapter is loaded, the
    strength is set, and it is fused when the module supports that. The
    function raises ``LoraRejected`` if the adapter reaches no layer.
    """
    applied = clamp_strength(lora, strength)
    _bind_holder(holder, lora, applied)
    target = transformer if transformer is not None else _holder_transformer(holder)
    if target is None:
        return LoraApplyResult(lora.id, lora.alias, applied, lora.base, 0)
    applied = apply_lora(target, lora, applied)
    patched = int(getattr(apply_lora, "last_modules_patched", 0) or 0)
    return LoraApplyResult(lora.id, lora.alias, applied, lora.base, patched)


def _bind_holder(holder: object | None, lora: UserLora, strength: float) -> None:
    """Record the selected LoRA on a generator, runner, or job-like object."""
    if holder is None:
        return
    for name, value in (
        ("bound_lora", lora),
        ("lora_strength", strength),
        ("lora_weight", strength),
        ("lora_path", str(lora.weights)),
    ):
        try:
            setattr(holder, name, value)
        except (AttributeError, TypeError):
            continue


def _holder_transformer(holder: object | None) -> object | None:
    if holder is None:
        return None
    transformer = getattr(holder, "transformer", None)
    if transformer is not None:
        return transformer
    pipe = getattr(holder, "pipe", None)
    return None if pipe is None else getattr(pipe, "transformer", None)


def apply_lora(transformer: object, lora: UserLora, strength: float) -> float:
    """Load one adapter on the transformer. Any loaded adapter is removed first.

    Args:
        transformer: The Z-Image transformer.
        lora: The selected LoRA.
        strength: The expert LoRA strength field.

    Returns:
        The strength that was applied.
    """
    applied = clamp_strength(lora, strength)
    unload_lora(transformer)
    loader = getattr(transformer, "load_lora_adapter", None)
    if not callable(loader):
        loader = getattr(transformer, "load_lora_weights", None)
    if not callable(loader):
        raise RuntimeError("the transformer has no adapter loader")
    fits_layers(lora.weights, _live_shapes(transformer))
    try:
        _load_adapter(loader, str(lora.weights))
    except RuntimeError as exc:
        unload_lora(transformer)
        if "size mismatch" in str(exc):
            raise LoraRejected(CODE_INVALID) from exc
        raise
    if not _adapter_attached(transformer):
        # diffusers only warns when no key matches a layer of the right shape.
        unload_lora(transformer)
        raise LoraRejected(CODE_INVALID)
    setter = getattr(transformer, "set_adapters", None)
    if not callable(setter):
        raise RuntimeError("the transformer has no adapter loader")
    try:
        setter([_ADAPTER], weights=[applied])
    except TypeError:
        setter([_ADAPTER], adapter_weights=[applied])
    patched = _modules_patched(transformer)
    if patched == 0 and callable(getattr(transformer, "named_modules", None)):
        unload_lora(transformer)
        raise LoraRejected(CODE_INVALID)
    _fuse_adapter(transformer)
    _LOG.info(
        "LoRA applied id=%s alias=%s strength=%s base=%s modules=%s",
        lora.id,
        lora.alias or "-",
        applied,
        lora.base or "-",
        patched,
    )
    apply_lora.last_modules_patched = patched
    return applied


def _fuse_adapter(transformer: object) -> None:
    """Bake the adapter into the weights when the module exposes a fuse hook."""
    fuse = getattr(transformer, "fuse_lora", None)
    if not callable(fuse):
        fuse = getattr(transformer, "fuse_lora_weights", None)
    if not callable(fuse):
        return
    try:
        fuse()
    except TypeError:
        try:
            fuse(adapter_names=[_ADAPTER])
        except TypeError:
            return


def _modules_patched(transformer: object) -> int:
    """How many modules received the bound adapter. ``peft_config`` is the fallback."""
    modules = getattr(transformer, "named_modules", None)
    if callable(modules):
        count = 0
        for _name, module in modules():
            layers = getattr(module, "lora_A", None)
            if isinstance(layers, dict) and _ADAPTER in layers:
                count += 1
        return count
    config = getattr(transformer, "peft_config", None)
    if isinstance(config, dict) and _ADAPTER in config:
        return 1
    return 0


def _load_adapter(loader: Callable[..., object], weights: str) -> None:
    """Load with names relative to the transformer, as Model Forge and ``_peft_key`` write them.

    diffusers keeps only keys that start with ``prefix``; its default ``transformer``
    drops every key of such a file.
    """
    try:
        loader(weights, adapter_name=_ADAPTER, prefix=None)
    except TypeError as exc:
        if "prefix" not in str(exc):
            raise
        loader(weights, adapter_name=_ADAPTER)


def _live_shapes(transformer: object) -> dict[str, tuple[int, int]]:
    """``(out, in)`` of each linear layer of the loaded transformer. Empty for a model without modules."""
    modules = getattr(transformer, "named_modules", None)
    if not callable(modules):
        return {}
    shapes: dict[str, tuple[int, int]] = {}
    for name, module in modules():
        out_size = getattr(module, "out_features", None)
        in_size = getattr(module, "in_features", None)
        if isinstance(out_size, int) and isinstance(in_size, int):
            shapes[str(name)] = (out_size, in_size)
    return shapes


def fits_layers(path: Path, shapes: dict[str, tuple[int, int]] | None = None) -> None:
    """Every ``lora_A`` / ``lora_B`` matrix must fit the layer it targets.

    Args:
        path: The adapter file.
        shapes: ``(out, in)`` per layer. Defaults to the shipped Z-Image-Turbo table.
            A layer that is not listed, and a tensor that is not a matrix, are left to the loader.

    Raises:
        LoraRejected: ``lora.invalid`` when a matrix does not fit.
    """
    table = _base_shapes() if shapes is None else shapes
    if not table:
        return
    opened = _safetensors(path)
    if opened is None:
        raise LoraRejected(CODE_INVALID)
    pairs: dict[str, dict[str, list[object]]] = {}
    for key, entry in opened[0].items():
        if key == "__metadata__" or not isinstance(entry, dict):
            continue
        shape = entry.get("shape")
        name = _peft_key(str(key))
        for suffix in (_A_WEIGHT, _B_WEIGHT):
            if name.endswith(suffix) and isinstance(shape, list) and len(shape) == 2:
                pairs.setdefault(name[: -len(suffix)], {})[suffix] = shape
    for stem, pair in pairs.items():
        layer = table.get(stem)
        if layer is None:
            continue
        down = pair.get(_A_WEIGHT)
        up = pair.get(_B_WEIGHT)
        if down is not None and down[1] != layer[1]:
            raise LoraRejected(CODE_INVALID)
        if up is not None and up[0] != layer[0]:
            raise LoraRejected(CODE_INVALID)
        if down is not None and up is not None and down[0] != up[1]:
            raise LoraRejected(CODE_INVALID)


def _adapter_attached(transformer: object) -> bool:
    """True when the adapter reached at least one layer. A transformer without modules is trusted."""
    modules = getattr(transformer, "named_modules", None)
    if not callable(modules):
        return True
    for _name, module in modules():
        layers = getattr(module, "lora_A", None)
        if layers is not None and _ADAPTER in layers:
            return True
    return False


def unload_lora(transformer: object) -> None:
    """Remove adapters from the transformer.

    Args:
        transformer: The Z-Image transformer.
    """
    config = getattr(transformer, "peft_config", None)
    delete = getattr(transformer, "delete_adapters", None)
    if callable(delete) and isinstance(config, dict) and config:
        delete(list(config))
    unload = getattr(transformer, "unload_lora_weights", None)
    if callable(unload):
        try:
            unload()
        except (ValueError, RuntimeError):
            return


def _lora_root(root: Path | None) -> Path:
    base = root if root is not None else models_dir()
    return base / "user-models" / "lora"


def _document(path: Path) -> dict[str, object] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def _schema_reason(document: dict[str, object]) -> str | None:
    if document.get("schema") != SCHEMA:
        return REASON_SCHEMA
    minimum = document.get("min_schema")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or not 1 <= minimum <= 1:
        return REASON_SCHEMA
    return None


def _bad_id(lora_id: str) -> bool:
    return not lora_id or lora_id in {".", ".."} or "/" in lora_id or "\\" in lora_id


def _inside(folder: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
    except (OSError, ValueError):
        return False
    return path.is_file()


def _file_reason(document: dict[str, object], weights: Path) -> str | None:
    size = document.get("size")
    digest = document.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size != weights.stat().st_size:
        return REASON_FILE
    if not isinstance(digest, str) or digest.casefold() != _sha256(weights):
        return REASON_FILE
    return None


def _strength(document: dict[str, object]) -> tuple[float, float, float] | None:
    default = document.get("default_strength")
    span = document.get("strength_range")
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        return None
    if not isinstance(span, list) or len(span) != 2:
        return None
    low, high = span
    if isinstance(low, bool) or isinstance(high, bool):
        return None
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return None
    if float(low) > float(high):
        return None
    return float(default), float(low), float(high)


def _adult_label(document: dict[str, object]) -> bool | None:
    if _ADULT_LABEL not in document:
        return False
    value = document[_ADULT_LABEL]
    if value is True:
        return True
    if value is False:
        return False
    return None


def _preview_reason(document: dict[str, object], folder: Path) -> str | None:
    if "preview" not in document and "preview_sha256" not in document:
        return None
    if document.get("preview") != "preview.webp":
        return REASON_PREVIEW
    preview = folder / "preview.webp"
    expected = document.get("preview_sha256")
    if not isinstance(expected, str) or not _inside(folder, preview):
        return REASON_PREVIEW
    if expected.casefold() != _sha256(preview):
        return REASON_PREVIEW
    return None


def _person_flags(header: dict[str, str]) -> bool:
    return header.get(_HEADER_CONSENT) == "true" and header.get(_HEADER_CONFIRMED) == "true"


def _has_kohya_keys(path: Path) -> bool:
    """True when the file still uses Kohya ``lora_unet_*`` / text-encoder names."""
    opened = _safetensors(path)
    if opened is None:
        return False
    for key in opened[0]:
        if key == "__metadata__":
            continue
        name = str(key)
        if name.startswith(_KOHYA) or name.startswith(("lora_te", "lora_te1_", "lora_te2_")):
            return True
    return False


def _header(path: Path) -> dict[str, str] | None:
    opened = _safetensors(path)
    if opened is None:
        return None
    raw = opened[0].get("__metadata__") or {}
    if not isinstance(raw, dict):
        return None
    picked: dict[str, str] = {}
    for key in _HEADER_KEYS:
        value = raw.get(key)
        if isinstance(value, str):
            picked[key] = value
    return picked


def _opened_weights(path: Path) -> tuple[dict[str, str], Path]:
    """Header strings and the file the loader should open.

    Kohya ``lora_unet_*`` names are mapped to the z-image modules, and each
    ``alpha`` is folded into its ``lora_B`` weight.
    """
    opened = _safetensors(path)
    if opened is None:
        raise LoraRejected(CODE_INVALID)
    header, blob = opened
    raw = header.get("__metadata__") or {}
    if not isinstance(raw, dict):
        raise LoraRejected(CODE_INVALID)
    meta = {str(key): value for key, value in raw.items() if isinstance(key, str) and isinstance(value, str)}
    if not _offsets_fit(header, blob):
        raise LoraRejected(CODE_INVALID)
    names = [str(key) for key in header if key != "__metadata__" and isinstance(key, str)]
    renamed = {name: _peft_key(name) for name in names}
    alphas = {new[: -len(_ALPHA)]: old for old, new in renamed.items() if new.endswith(_ALPHA)}
    weights = {old: new for old, new in renamed.items() if not new.endswith(_ALPHA)}
    if len(set(weights.values())) != len(weights) or not _paired(list(weights.values())):
        raise LoraRejected(CODE_INVALID)
    stems = {new[: -len(_A_WEIGHT)] for new in weights.values() if new.endswith(_A_WEIGHT)}
    if any(stem not in stems for stem in alphas):
        raise LoraRejected(CODE_INVALID)
    if not alphas and list(renamed.values()) == names:
        return meta, path
    return meta, _write_normalised(path, header, blob, weights, alphas)


def fits_base(path: Path) -> None:
    """Every Z-Image adapter must target a weight of the z-image transformer.

    Module names are checked only when :func:`detect_base` is Z-Image. A
    MOSS-tagged file is not rejected for naming other modules.

    Raises:
        LoraRejected: ``lora.invalid`` for a damaged file, ``lora.base_mismatch``
            when a Z-Image adapter names a module the base does not have.
    """
    opened = _safetensors(path)
    if opened is None:
        raise LoraRejected(CODE_INVALID)
    if not _is_zimage_family(detect_base(path)):
        return
    modules = _modules()
    for key in opened[0]:
        if key == "__metadata__":
            continue
        name = _peft_key(str(key))
        for suffix in (_A_WEIGHT, _B_WEIGHT, _ALPHA):
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                break
        else:
            raise LoraRejected(CODE_INVALID)
        if f"{stem}.weight" not in modules:
            raise LoraRejected(CODE_BASE_MISMATCH)


def open_third_party(path: Path) -> UserLora:
    """A ``.safetensors`` LoRA made outside Model Forge, for one command.

    A LoRA trained on another base model opens all the same; its trained base
    is recorded on the entry when the file names one.

    Raises:
        LoraRejected: The file is damaged, or its adapters do not fit the base.
    """
    meta, weights = _opened_weights(path)
    base = meta.get(_HEADER_BASE, "") or _canonical_base(str(meta.get(_HEADER_TRAINED_BASE) or ""))
    fits_base(path)
    variant, base_sha, warnings = _provenance(meta, path, base)
    return UserLora(
        path.stem,
        THIRD_PARTY_KIND,
        path.parent,
        weights,
        "",
        recommended_or_default(meta.get(_HEADER_STRENGTH)),
        CLI_STRENGTH_RANGE,
        False,
        name=path.stem,
        base=base,
        sha256=_sha256(path),
        base_variant=variant,
        base_sha256=base_sha,
        warnings=warnings,
    )


def _matching_sidecar(path: Path) -> dict[str, object] | None:
    """The sidecar that names this file, or None when that file is absent or names another."""
    sidecar = path.parent / _SIDECAR
    if not sidecar.is_file():
        return None
    document = _document(sidecar)
    if document is None:
        raise LoraRejected(CODE_INVALID)
    if document.get("file") != path.name:
        return None
    if _schema_reason(document) is not None or document.get("type") != "lora":
        raise LoraRejected(CODE_INVALID)
    if _file_reason(document, path) is not None:
        raise LoraRejected(CODE_INVALID)
    base = document.get("base")
    mdl_id = str(base.get("mdl_id") or "") if isinstance(base, dict) else ""
    if not isinstance(base, dict) or not mdl_id:
        raise LoraRejected(CODE_INVALID)
    return document


def _safetensors(path: Path) -> tuple[dict[str, object], bytes] | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 8:
        return None
    size = int.from_bytes(data[:8], "little")
    if size <= 0 or 8 + size > len(data):
        return None
    try:
        loaded = json.loads(data[8:8 + size].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded, data[8 + size:]


def _peft_key(key: str) -> str:
    """A transformer-relative ``lora_A`` / ``lora_B`` / ``alpha`` name.

    PEFT prefixes (``base_model.model.``) and ``.lora_A.default.weight`` names
    are folded into the same form as a Forge export.

    Raises:
        LoraRejected: A Kohya name that maps to no z-image module (``lora.base_mismatch``).
    """
    name = key
    if name.startswith(_KOHYA):
        prefix, _dot, rest = name.partition(".")
        stem = _kohya_table().get(prefix)
        if stem is None:
            raise LoraRejected(CODE_BASE_MISMATCH)
        name = f"{stem}.{rest}"
    elif name.startswith(("lora_te", "lora_te1_", "lora_te2_")):
        raise LoraRejected(CODE_BASE_MISMATCH)
    for prefix in ("base_model.model.", "base_model.", "transformer.", "diffusion_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
    return name.replace(".lora_down.weight", _A_WEIGHT).replace(".lora_up.weight", _B_WEIGHT)


_MODULES: frozenset[str] | None = None
_SHAPES: dict[str, tuple[int, int]] | None = None
_KOHYA_TABLE: dict[str, str] | None = None


def _modules() -> frozenset[str]:
    """Weight names of the Z-Image-Turbo transformer, from the shipped index."""
    global _MODULES
    if _MODULES is None:
        index = Path(__file__).with_name("turbo_transformer_index.json")
        raw = json.loads(index.read_text(encoding="utf-8"))
        _MODULES = frozenset(str(key) for key in raw.get("weight_map", {}))
    return _MODULES


def _base_shapes() -> dict[str, tuple[int, int]]:
    """``(out, in)`` of each linear layer of the Z-Image-Turbo transformer, from the shipped table."""
    global _SHAPES
    if _SHAPES is None:
        table = Path(__file__).with_name("turbo_transformer_shapes.json")
        raw = json.loads(table.read_text(encoding="utf-8"))
        _SHAPES = {str(name): (int(shape[0]), int(shape[1])) for name, shape in raw.items()}
    return _SHAPES


def _kohya_table() -> dict[str, str]:
    """``lora_unet_<module with underscores>`` to the module name."""
    global _KOHYA_TABLE
    if _KOHYA_TABLE is None:
        table: dict[str, str] = {}
        for weight in _modules():
            if weight.endswith(".weight"):
                stem = weight[: -len(".weight")]
                table[_KOHYA + stem.replace(".", "_")] = stem
        _KOHYA_TABLE = table
    return _KOHYA_TABLE


def _paired(keys: list[str]) -> bool:
    if not keys or any(not key.endswith((_A_WEIGHT, _B_WEIGHT)) for key in keys):
        return False
    stems_a = {key[: -len(_A_WEIGHT)] for key in keys if key.endswith(_A_WEIGHT)}
    stems_b = {key[: -len(_B_WEIGHT)] for key in keys if key.endswith(_B_WEIGHT)}
    return bool(stems_a) and stems_a == stems_b


def _offsets_fit(header: dict[str, object], blob: bytes) -> bool:
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, dict):
            return False
        offsets = value.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            return False
        start, stop = offsets
        if isinstance(start, bool) or isinstance(stop, bool):
            return False
        if not isinstance(start, int) or not isinstance(stop, int):
            return False
        if start < 0 or stop < start or stop > len(blob):
            return False
    return True


def _write_normalised(
    path: Path, header: dict[str, object], blob: bytes, weights: dict[str, str], alphas: dict[str, str],
) -> Path:
    """Write the loader copy: PEFT names, with each alpha folded into ``lora_B``."""
    scale: dict[str, float] = {}
    for stem, alpha_key in alphas.items():
        alpha = _scalar(header[alpha_key], blob)
        down = next(old for old, new in weights.items() if new == f"{stem}{_A_WEIGHT}")
        down_entry = header[down]
        shape = down_entry.get("shape") if isinstance(down_entry, dict) else None
        if not isinstance(shape, list) or not shape or not isinstance(shape[0], int) or shape[0] <= 0:
            raise LoraRejected(CODE_INVALID)
        scale[f"{stem}{_B_WEIGHT}"] = alpha / shape[0]
    new_header: dict[str, object] = {}
    if "__metadata__" in header:
        new_header["__metadata__"] = header["__metadata__"]
    parts: list[bytes] = []
    offset = 0
    for old, new in weights.items():
        source = header[old]
        if not isinstance(source, dict):
            raise LoraRejected(CODE_INVALID)
        entry = dict(source)
        start, stop = entry["data_offsets"]
        data = blob[start:stop]
        if new in scale and scale[new] != 1.0:
            data = _scaled(str(entry.get("dtype")), data, scale[new])
        entry["data_offsets"] = [offset, offset + len(data)]
        offset += len(data)
        new_header[new] = entry
        parts.append(data)
    payload = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    target = Path(tempfile.gettempdir()) / "canvas-lora" / _sha256(path)[:16] / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(len(payload).to_bytes(8, "little") + payload + b"".join(parts))
    return target


def _scalar(entry: object, blob: bytes) -> float:
    if not isinstance(entry, dict):
        raise LoraRejected(CODE_INVALID)
    start, stop = entry["data_offsets"]
    values = _floats(str(entry.get("dtype")), blob[start:stop])
    if values.size != 1:
        raise LoraRejected(CODE_INVALID)
    value = float(values.reshape(-1)[0])
    if not value > 0 or value != value or value == float("inf"):
        raise LoraRejected(CODE_INVALID)
    return value


def _floats(dtype: str, data: bytes):  # noqa: ANN202 - numpy array
    import numpy as np

    if dtype == "F32":
        return np.frombuffer(data, dtype="<f4").astype(np.float32)
    if dtype == "F16":
        return np.frombuffer(data, dtype="<f2").astype(np.float32)
    if dtype == "BF16":
        raw = np.frombuffer(data, dtype="<u2").astype(np.uint32) << 16
        return raw.view(np.float32)
    raise LoraRejected(CODE_INVALID)


def _scaled(dtype: str, data: bytes, factor: float) -> bytes:
    import numpy as np

    values = _floats(dtype, data) * np.float32(factor)
    if dtype == "F32":
        return values.astype("<f4").tobytes()
    if dtype == "F16":
        return values.astype("<f2").tobytes()
    bits = values.astype(np.float32).view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
    return rounded.tobytes()


def _loader_weights(path: Path) -> Path:
    _meta, weights = _opened_weights(path)
    return weights


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
