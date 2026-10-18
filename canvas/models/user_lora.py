"""Model Forge LoRAs under the shared models folder.

A folder is listed when its record matches the Z-Image-Turbo base this app
runs. One adapter is loaded on the transformer at a time.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from canvas.infra.paths import models_dir
from canvas.models.registry import load_catalog

SCHEMA = "voxprint-model/1"
ZIMAGE_TURBO_MDL_ID = "z-image-turbo"
_KINDS = frozenset({"person", "object", "style"})
_ADAPTER = "canvas-lora"
_HEADER_TRIGGER = "voxprint.subject.trigger"
_HEADER_KIND = "voxprint.lora.kind"
_HEADER_CONSENT = "voxprint.subject.consent_obtained"
_HEADER_CONFIRMED = "voxprint.subject.adult_confirmed"
_ADULT_LABEL = "voxprint.lora.adult_content"

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
    expected = running_base_id()
    for folder in sorted((path for path in base.iterdir() if path.is_dir()), key=lambda path: path.name):
        if folder.name.startswith("."):
            continue
        found, reason = inspect_folder(folder, expected_base=expected)
        if found is None:
            skipped.append(SkippedLora(folder.name, reason or REASON_RECORD))
        else:
            available.append(found)
    available.sort(key=lambda item: (item.adult_content, item.id))
    return LoraIndex(tuple(available), tuple(skipped))


def inspect_folder(folder: Path, *, expected_base: str | None = None) -> tuple[UserLora | None, str | None]:
    """Return a listed LoRA, or a catalog reason key when the folder is skipped.

    Args:
        folder: One ``<id>`` directory.
        expected_base: ``base.mdl_id`` this app accepts. The running base is used when omitted.
    """
    if expected_base is None:
        expected_base = running_base_id()
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
    if not isinstance(base, dict) or str(base.get("mdl_id") or "") != expected_base or not expected_base:
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
    return UserLora(
        lora_id,
        str(kind),
        folder,
        weights,
        header.get(_HEADER_TRIGGER, ""),
        default_strength,
        (low, high),
        adult,
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
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = _lora_root(root) / text
    if candidate.is_file():
        candidate = candidate.parent
    found, _reason = inspect_folder(candidate)
    return found


def running_base_id() -> str:
    """The Z-Image-Turbo base id this app runs, or empty when the registry differs."""
    base = load_catalog().default_text_base or ""
    if base != ZIMAGE_TURBO_MDL_ID:
        return ""
    return ZIMAGE_TURBO_MDL_ID


def clamp_strength(lora: UserLora, requested: float) -> float:
    """Keep a requested strength inside the LoRA's recorded range.

    Args:
        lora: The selected LoRA.
        requested: The expert LoRA strength field.
    """
    if isinstance(requested, bool) or not isinstance(requested, (int, float)):
        requested = lora.default_strength
    low, high = lora.strength_range
    return min(max(float(requested), low), high)


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
    loader(str(lora.weights), adapter_name=_ADAPTER)
    setter = getattr(transformer, "set_adapters", None)
    if not callable(setter):
        raise RuntimeError("the transformer has no adapter loader")
    try:
        setter([_ADAPTER], weights=[applied])
    except TypeError:
        setter([_ADAPTER], adapter_weights=[applied])
    return applied


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


def _header(path: Path) -> dict[str, str] | None:
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
    raw = loaded.get("__metadata__") or {}
    if not isinstance(raw, dict):
        return None
    picked: dict[str, str] = {}
    for key in (_HEADER_TRIGGER, _HEADER_KIND, _HEADER_CONSENT, _HEADER_CONFIRMED):
        value = raw.get(key)
        if isinstance(value, str):
            picked[key] = value
    return picked


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
