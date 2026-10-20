"""Model Forge LoRAs under the shared models folder.

A folder is listed when its record matches the Z-Image-Turbo base this app
runs. One adapter is loaded on the transformer at a time.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from canvas.infra.paths import models_dir
from canvas.models.registry import load_catalog

SCHEMA = "voxprint-model/1"
ZIMAGE_TURBO_MDL_ID = "z-image-turbo"
FORGE_BASE_ID = "z-image"
CLI_DEFAULT_STRENGTH = 0.7
CLI_STRENGTH_RANGE = (0.0, 2.0)
CODE_BASE_MISMATCH = "lora.base_mismatch"
CODE_INVALID = "lora.invalid"
CODE_SCALE = "lora.scale"
_KINDS = frozenset({"person", "object", "style"})
_ADAPTER = "canvas-lora"
_A_WEIGHT = ".lora_A.weight"
_ALPHA = ".alpha"
_KOHYA = "lora_unet_"
THIRD_PARTY_KIND = "unknown"
_B_WEIGHT = ".lora_B.weight"
_HEADER_TRIGGER = "voxprint.subject.trigger"
_HEADER_KIND = "voxprint.lora.kind"
_HEADER_BASE = "voxprint.lora.base"
_HEADER_CONSENT = "voxprint.subject.consent_obtained"
_HEADER_CONFIRMED = "voxprint.subject.adult_confirmed"
_ADULT_LABEL = "voxprint.lora.adult_content"
_HEADER_KEYS = (
    _HEADER_TRIGGER,
    _HEADER_KIND,
    _HEADER_BASE,
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
    """A LoRA file cannot be loaded. ``code`` is a stable catalog key."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


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
    mdl_id = str(base.get("mdl_id") or "") if isinstance(base, dict) else ""
    if not isinstance(base, dict) or not base_matches(mdl_id, expected_base):
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
    header_base = header.get(_HEADER_BASE)
    if header_base is not None and not base_matches(header_base, expected_base):
        return None, REASON_BASE
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
    from canvas.models.lora_library import resolve_library

    imported = resolve_library(text)
    if imported is not None:
        return imported
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = _lora_root(root) / text
    if candidate.is_file() and candidate.suffix == ".safetensors":
        found, _reason = inspect_folder(candidate.parent)
        if found is not None and _same_file(found.weights, candidate):
            return found
        try:
            return open_weights(candidate)
        except LoraRejected:
            return None
    if candidate.is_file():
        candidate = candidate.parent
    found, _reason = inspect_folder(candidate)
    return found


def base_matches(mdl_id: str, expected: str | None = None) -> bool:
    """True when ``mdl_id`` is the base this app runs.

    Model Forge records a Z-Image LoRA as ``z-image``. That id matches the
    Z-Image-Turbo base this app runs.

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
    return expected == ZIMAGE_TURBO_MDL_ID and mdl_id == FORGE_BASE_ID


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
        LoraRejected: The file is unreadable or its base does not match.
    """
    from canvas.models.lora_library import PACKAGE_SUFFIX, open_package_weights, resolve_library

    imported = resolve_library(str(path))
    if imported is not None:
        return replace(imported, default_strength=CLI_DEFAULT_STRENGTH, strength_range=CLI_STRENGTH_RANGE)
    candidate = Path(path)
    if candidate.is_file() and candidate.suffix == PACKAGE_SUFFIX:
        opened = open_package_weights(candidate)
        return replace(opened, default_strength=CLI_DEFAULT_STRENGTH, strength_range=CLI_STRENGTH_RANGE)
    if candidate.is_dir():
        found, reason = inspect_folder(candidate)
        if found is None:
            if reason == REASON_BASE:
                raise LoraRejected(CODE_BASE_MISMATCH)
            raise LoraRejected(CODE_INVALID)
        weights = _loader_weights(found.weights)
        return replace(
            found,
            weights=weights,
            default_strength=CLI_DEFAULT_STRENGTH,
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
        default_strength=CLI_DEFAULT_STRENGTH,
        strength_range=CLI_STRENGTH_RANGE,
    )


def _forge_file(path: Path) -> bool:
    """True when the header or a matching sidecar marks a Model Forge export."""
    header = _header(path)
    if header is None:
        raise LoraRejected(CODE_INVALID)
    if header.get(_HEADER_BASE):
        return True
    sidecar = path.parent / _SIDECAR
    document = _document(sidecar) if sidecar.is_file() else None
    return document is not None and document.get("file") == path.name


def open_weights(path: Path) -> UserLora:
    """Read one Forge ``.safetensors`` file and the sidecar beside it, when present.

    Args:
        path: The weight file. The default Forge path is ``project/export/<id>.safetensors``.

    Raises:
        LoraRejected: The file is unreadable or its base does not match.
    """
    meta, weights = _opened_weights(path)
    base = meta.get(_HEADER_BASE, "")
    if not base:
        raise LoraRejected(CODE_INVALID)
    if not base_matches(base):
        raise LoraRejected(CODE_BASE_MISMATCH)
    sidecar = _matching_sidecar(path)
    kind = meta.get(_HEADER_KIND, "")
    lora_id = path.stem
    default_strength = CLI_DEFAULT_STRENGTH
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
    if kind not in _KINDS or _bad_id(lora_id):
        raise LoraRejected(CODE_INVALID)
    return UserLora(
        lora_id,
        kind,
        path.parent,
        weights,
        meta.get(_HEADER_TRIGGER, ""),
        default_strength,
        (low, high),
        adult,
    )


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
    """Every adapter must target a weight of the z-image transformer.

    Raises:
        LoraRejected: ``lora.invalid`` for a damaged file, ``lora.base_mismatch``
            when an adapter names a module the base does not have.
    """
    opened = _safetensors(path)
    if opened is None:
        raise LoraRejected(CODE_INVALID)
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

    Raises:
        LoraRejected: The file is damaged, or its adapters do not fit the base.
    """
    meta, weights = _opened_weights(path)
    base = meta.get(_HEADER_BASE, "")
    if base and not base_matches(base):
        raise LoraRejected(CODE_BASE_MISMATCH)
    fits_base(path)
    return UserLora(
        path.stem,
        THIRD_PARTY_KIND,
        path.parent,
        weights,
        "",
        CLI_DEFAULT_STRENGTH,
        CLI_STRENGTH_RANGE,
        False,
        name=path.stem,
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
    if not base_matches(mdl_id):
        raise LoraRejected(CODE_BASE_MISMATCH)
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
    for prefix in ("transformer.", "diffusion_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace(".lora_down.weight", _A_WEIGHT).replace(".lora_up.weight", _B_WEIGHT)


_MODULES: frozenset[str] | None = None
_KOHYA_TABLE: dict[str, str] | None = None


def _modules() -> frozenset[str]:
    """Weight names of the Z-Image-Turbo transformer, from the shipped index."""
    global _MODULES
    if _MODULES is None:
        index = Path(__file__).with_name("turbo_transformer_index.json")
        raw = json.loads(index.read_text(encoding="utf-8"))
        _MODULES = frozenset(str(key) for key in raw.get("weight_map", {}))
    return _MODULES


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
