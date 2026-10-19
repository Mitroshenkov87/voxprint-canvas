"""The Canvas LoRA library: Model Forge exports imported from the GUI.

A Forge export is a ``.voxlora`` stored zip with ``model.safetensors``,
``voxprint-model.json`` (schema ``voxprint-model/1``), ``preview.webp`` and up
to six ``previews/NN.webp``. A bare ``.safetensors`` with that json beside it
is accepted too. Accepted packages are copied to
``<data home>/canvas/lora-library/<id>``.

Refusals carry stable codes: ``lora.invalid``, ``lora.base_mismatch``,
``lora.kind``, ``lora.preview`` and ``lora.base_missing``.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from canvas.infra.paths import lora_library_dir
from canvas.models.user_lora import (
    CODE_BASE_MISMATCH,
    CODE_INVALID,
    FORGE_BASE_ID,
    SCHEMA,
    ZIMAGE_TURBO_MDL_ID,
    LoraRejected,
    UserLora,
    _opened_weights,
    base_matches,
)

PACKAGE_SUFFIX = ".voxlora"
WEIGHTS_NAME = "model.safetensors"
MANIFEST_NAME = "voxprint-model.json"
PREVIEW_NAME = "preview.webp"
MAX_PREVIEWS = 6
MAX_PREVIEW_PX = 1024
MAX_PACKAGE_BYTES = 4 * 1024 * 1024 * 1024
STRENGTH_LIMITS = (0.0, 2.0)
DEFAULT_RANGE = (0.0, 1.2)

# The kinds the library takes. One constant, set by the owner's decision.
ACCEPTED_KINDS = frozenset({"object", "style"})
KNOWN_KINDS = frozenset({"object", "style", "subject"})

CODE_KIND = "lora.kind"
CODE_PREVIEW = "lora.preview"
CODE_BASE_MISSING = "lora.base_missing"

_HEADER_KIND = "voxprint.lora.kind"
_HEADER_BASE = "voxprint.lora.base"
_HEADER_TRIGGER = "voxprint.subject.trigger"
_HEADER_KIND_MAP = {"person": "subject", "thing": "object"}
_PREVIEW_ENTRY = re.compile(r"^previews/(\d{2})\.webp$")
_ALLOWED_ENTRY = re.compile(r"^(model\.safetensors|voxprint-model\.json|preview\.webp|previews/\d{2}\.webp|previews/)$")
_SLUG = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True)
class Manifest:
    """The fields Canvas reads from ``voxprint-model.json``."""

    name: str
    base: str
    trigger: str
    kind: str
    scale_default: float
    scale_min: float
    scale_max: float
    forge_version: str
    created: str
    weights_sha256: str | None
    attest: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LibraryEntry:
    """One imported LoRA in the library."""

    id: str
    folder: Path
    manifest: Manifest
    weights: Path
    previews: tuple[Path, ...]

    def as_lora(self) -> UserLora:
        """The entry as a selectable LoRA. The weight keys are normalised for the loader."""
        loader_weights = _loader_path(self.weights)
        low, high = _clamped_range(self.manifest)
        return UserLora(
            self.id,
            self.manifest.kind,
            self.folder,
            loader_weights,
            self.manifest.trigger,
            min(max(self.manifest.scale_default, low), high),
            (low, high),
            False,
            name=self.manifest.name,
            previews=self.previews,
        )


@dataclass
class _Staged:
    manifest: Manifest
    raw_manifest: dict[str, object]
    weights: bytes
    previews: list[tuple[str, bytes]]


def library_root(root: Path | None = None) -> Path:
    """The library folder."""
    return root if root is not None else lora_library_dir()


def import_lora(
    source: Path,
    gate: object,
    *,
    root: Path | None = None,
    base_ready: Callable[[], bool] | None = None,
    fetch_base: Callable[[Callable[[str], None]], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> LibraryEntry:
    """Validate one Forge export and copy it into the library.

    Args:
        source: A ``.voxlora`` package, or a ``.safetensors`` with ``voxprint-model.json`` beside it.
        gate: The app's check stack. Every preview goes through its input check.
        root: Library folder. The app data folder is used when omitted.
        base_ready: True when the base model is installed. Checked on disk when omitted.
        fetch_base: Asks VOXPrint MDL for the base model. MDL is used when omitted.
        progress: Status lines while the base model is requested.

    Raises:
        LoraRejected: The export is refused. Nothing is stored.
    """
    staged = _stage(source)
    _check_previews(staged.previews, gate)
    ready = base_ready if base_ready is not None else _base_on_disk
    if not ready():
        fetch = fetch_base if fetch_base is not None else _fetch_with_mdl
        if not fetch(progress or (lambda _line: None)):
            raise LoraRejected(CODE_BASE_MISSING)
    return _store(staged, library_root(root))


def read_export(source: Path) -> Manifest:
    """Validate one export without checking previews or storing it."""
    return _stage(source).manifest


def open_package_weights(source: Path) -> UserLora:
    """Open the weights of a ``.voxlora`` for one command. Previews are not read or kept.

    Raises:
        LoraRejected: The package is refused.
    """
    staged = _stage(source, with_previews=False)
    folder = Path(tempfile.gettempdir()) / "canvas-lora" / hashlib.sha256(staged.weights).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / WEIGHTS_NAME
    weights.write_bytes(staged.weights)
    _meta, loader_weights = _opened_weights(weights)
    low, high = _clamped_range(staged.manifest)
    manifest = staged.manifest
    return UserLora(
        _slug(manifest.name) or "lora",
        manifest.kind,
        folder,
        loader_weights,
        manifest.trigger,
        min(max(manifest.scale_default, low), high),
        (low, high),
        False,
        name=manifest.name,
    )


def list_library(root: Path | None = None) -> tuple[LibraryEntry, ...]:
    """Library entries that still pass the format and kind checks, sorted by name."""
    base = library_root(root)
    if not base.is_dir():
        return ()
    found: list[LibraryEntry] = []
    for folder in sorted(path for path in base.iterdir() if path.is_dir() and not path.name.startswith(".")):
        entry = read_entry(folder)
        if entry is not None:
            found.append(entry)
    found.sort(key=lambda item: (item.manifest.name.casefold(), item.id))
    return tuple(found)


def read_entry(folder: Path) -> LibraryEntry | None:
    """One stored entry, or None when its files no longer agree with the record."""
    try:
        raw = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
        manifest = _manifest(raw)
        weights = folder / WEIGHTS_NAME
        digest = _file_digest(weights)
    except (OSError, UnicodeError, json.JSONDecodeError, LoraRejected):
        return None
    if manifest.kind not in ACCEPTED_KINDS:
        return None
    if manifest.weights_sha256 is None or digest != manifest.weights_sha256:
        return None
    previews = tuple(
        path for path in [folder / PREVIEW_NAME, *sorted((folder / "previews").glob("*.webp"))] if path.is_file()
    )
    return LibraryEntry(folder.name, folder, manifest, weights, previews)


_DIGESTS: dict[tuple[str, int, int], str] = {}
_LOADER: dict[tuple[str, int, int], Path] = {}


def _stat_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


def _loader_path(path: Path) -> Path:
    """The file the loader opens, read once while the stored file stays the same."""
    key = _stat_key(path)
    cached = _LOADER.get(key)
    if cached is not None and cached.is_file():
        return cached
    _meta, loader = _opened_weights(path)
    _LOADER[key] = loader
    return loader


def _file_digest(path: Path) -> str:
    """SHA-256 of a stored file, kept while its size and time stay the same."""
    key = _stat_key(path)
    cached = _DIGESTS.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    _DIGESTS[key] = digest.hexdigest()
    return _DIGESTS[key]


def resolve_library(value: str, root: Path | None = None) -> UserLora | None:
    """The imported LoRA named by an id or a library folder path, or None."""
    text = str(value).strip()
    if not text:
        return None
    base = library_root(root)
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            relative = candidate.resolve().relative_to(base.resolve())
        except (OSError, ValueError):
            return None
        if not relative.parts:
            return None
        folder = base / relative.parts[0]
    elif "/" in text or "\\" in text or text in {".", ".."}:
        return None
    else:
        folder = base / text
    if not folder.is_dir():
        return None
    entry = read_entry(folder)
    if entry is None:
        return None
    try:
        return entry.as_lora()
    except LoraRejected:
        return None


def remove_entry(entry_id: str, root: Path | None = None) -> bool:
    """Delete one entry from the library. False when it is not there."""
    if not entry_id or "/" in entry_id or "\\" in entry_id or entry_id in {".", ".."}:
        return False
    folder = library_root(root) / entry_id
    if not folder.is_dir():
        return False
    shutil.rmtree(folder)
    return True


def _stage(source: Path, *, with_previews: bool = True) -> _Staged:
    source = Path(source)
    if source.suffix == PACKAGE_SUFFIX:
        files = _read_package(source)
    elif source.suffix == ".safetensors":
        files = _read_bare(source)
        if files.get(MANIFEST_NAME) is None or "name" not in _json(files[MANIFEST_NAME]):
            files[MANIFEST_NAME] = _rc3_manifest(source)
    else:
        raise LoraRejected(CODE_INVALID)
    raw_manifest = _json(files.get(MANIFEST_NAME))
    manifest = _manifest(raw_manifest)
    weights = files.get(WEIGHTS_NAME)
    if weights is None:
        raise LoraRejected(CODE_INVALID)
    digest = hashlib.sha256(weights).hexdigest()
    if manifest.weights_sha256 is not None and manifest.weights_sha256 != digest:
        raise LoraRejected(CODE_INVALID)
    meta = _weights_meta(weights)
    _agree(manifest, meta)
    if not base_matches(manifest.base, ZIMAGE_TURBO_MDL_ID):
        raise LoraRejected(CODE_BASE_MISMATCH)
    if manifest.kind not in ACCEPTED_KINDS:
        raise LoraRejected(CODE_KIND)
    previews: list[tuple[str, bytes]] = []
    if with_previews:
        previews = _previews(files, raw_manifest, package=source.suffix == PACKAGE_SUFFIX)
    if manifest.weights_sha256 is None:
        manifest = _with_digest(manifest, digest)
    return _Staged(manifest, raw_manifest, weights, previews)


def _read_package(source: Path) -> dict[str, bytes]:
    try:
        if source.stat().st_size > MAX_PACKAGE_BYTES:
            raise LoraRejected(CODE_INVALID)
        with zipfile.ZipFile(source) as archive:
            files: dict[str, bytes] = {}
            for info in archive.infolist():
                name = info.filename
                if not _ALLOWED_ENTRY.match(name) or info.compress_type != zipfile.ZIP_STORED:
                    raise LoraRejected(CODE_INVALID)
                if info.is_dir():
                    continue
                if name in files:
                    raise LoraRejected(CODE_INVALID)
                files[name] = archive.read(info)
    except (OSError, RuntimeError, NotImplementedError, EOFError, zipfile.BadZipFile, zipfile.LargeZipFile,
            KeyError, ValueError) as exc:
        raise LoraRejected(CODE_INVALID) from exc
    return files


def _read_bare(source: Path) -> dict[str, bytes]:
    sidecar = source.parent / MANIFEST_NAME
    try:
        files: dict[str, bytes] = {WEIGHTS_NAME: source.read_bytes()}
        if sidecar.is_file():
            files[MANIFEST_NAME] = sidecar.read_bytes()
    except OSError as exc:
        raise LoraRejected(CODE_INVALID) from exc
    return files


def _rc3_manifest(source: Path) -> bytes:
    """An earlier Forge export: the weight header, with its own sidecar when present."""
    from canvas.models.user_lora import open_weights

    opened = open_weights(source)
    meta, _loader = _opened_weights(source)
    kind = _HEADER_KIND_MAP.get(opened.kind, opened.kind)
    low, high = opened.strength_range
    record = {
        "schema": SCHEMA,
        "name": opened.id,
        "base": meta.get(_HEADER_BASE, ""),
        "trigger": opened.trigger,
        "kind": kind,
        "scale_default": opened.default_strength,
        "scale_min": low,
        "scale_max": high,
        "forge_version": "",
        "created": "",
    }
    return json.dumps(record).encode("utf-8")


def _json(data: bytes | None) -> dict[str, object]:
    if data is None:
        raise LoraRejected(CODE_INVALID)
    try:
        loaded = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LoraRejected(CODE_INVALID) from exc
    if not isinstance(loaded, dict):
        raise LoraRejected(CODE_INVALID)
    return loaded


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _manifest(raw: dict[str, object]) -> Manifest:
    if raw.get("schema") != SCHEMA:
        raise LoraRejected(CODE_INVALID)
    texts = {}
    for key in ("name", "base", "trigger", "kind", "forge_version", "created"):
        value = raw.get(key)
        if not isinstance(value, str):
            raise LoraRejected(CODE_INVALID)
        texts[key] = value.strip()
    if not texts["name"] or not texts["base"] or texts["kind"] not in KNOWN_KINDS:
        raise LoraRejected(CODE_INVALID)
    default = _number(raw.get("scale_default"))
    low = _number(raw.get("scale_min", DEFAULT_RANGE[0]))
    high = _number(raw.get("scale_max", DEFAULT_RANGE[1]))
    if default is None or low is None or high is None or low > high:
        raise LoraRejected(CODE_INVALID)
    digest = raw.get("weights_sha256")
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)):
        raise LoraRejected(CODE_INVALID)
    attest = raw.get("attest")
    return Manifest(
        texts["name"],
        texts["base"],
        texts["trigger"],
        texts["kind"],
        default,
        low,
        high,
        texts["forge_version"],
        texts["created"],
        None if digest is None else str(digest).casefold(),
        dict(attest) if isinstance(attest, dict) else {},
    )


def _with_digest(manifest: Manifest, digest: str) -> Manifest:
    from dataclasses import replace

    return replace(manifest, weights_sha256=digest)


def _weights_meta(weights: bytes) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="canvas-lora-check-") as scratch:
        path = Path(scratch) / WEIGHTS_NAME
        path.write_bytes(weights)
        meta, _loader = _opened_weights(path)
    return meta


def _agree(manifest: Manifest, meta: dict[str, str]) -> None:
    """The weight header and the json must name the same kind, base and trigger."""
    kind = meta.get(_HEADER_KIND)
    if kind is not None and _HEADER_KIND_MAP.get(kind, kind) != manifest.kind:
        raise LoraRejected(CODE_INVALID)
    base = meta.get(_HEADER_BASE)
    if base is not None and _base_family(base) != _base_family(manifest.base):
        raise LoraRejected(CODE_INVALID)
    trigger = meta.get(_HEADER_TRIGGER)
    if trigger is not None and trigger.strip() != manifest.trigger:
        raise LoraRejected(CODE_INVALID)


def _base_family(base: str) -> str:
    return FORGE_BASE_ID if base in {FORGE_BASE_ID, ZIMAGE_TURBO_MDL_ID} else base


def _previews(files: dict[str, bytes], raw: dict[str, object], *, package: bool) -> list[tuple[str, bytes]]:
    names = sorted(name for name in files if _PREVIEW_ENTRY.match(name))
    if len(names) > MAX_PREVIEWS:
        raise LoraRejected(CODE_INVALID)
    listed = raw.get("previews")
    if listed is not None:
        if not isinstance(listed, list) or any(not isinstance(item, str) or item not in files for item in listed):
            raise LoraRejected(CODE_INVALID)
    if package and PREVIEW_NAME not in files:
        raise LoraRejected(CODE_INVALID)
    ordered = ([PREVIEW_NAME] if PREVIEW_NAME in files else []) + names
    return [(name, files[name]) for name in ordered]


def _check_previews(previews: list[tuple[str, bytes]], gate: object) -> None:
    """Every preview goes through the app's input check. One failure refuses the package."""
    from PIL import Image, UnidentifiedImageError

    from canvas.safety.gate import SafetyError
    from canvas.safety.image import RawImage
    from canvas.safety.inputs import ImageRole, PendingInput

    for _name, data in previews:
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "WEBP" or max(image.size) > MAX_PREVIEW_PX:
                    raise LoraRejected(CODE_PREVIEW)
                rgb = image.convert("RGB")
                raw = RawImage(rgb.width, rgb.height, rgb.tobytes())
        except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError) as exc:
            raise LoraRejected(CODE_PREVIEW) from exc
        try:
            gate.release_request(  # type: ignore[attr-defined]
                "", "text", [PendingInput(ImageRole.REFERENCE, raw)], explicit=True,
            )
        except SafetyError as exc:
            raise LoraRejected(CODE_PREVIEW) from exc
        except Exception as exc:  # noqa: BLE001 - any failure refuses the package
            raise LoraRejected(CODE_PREVIEW) from exc


def _store(staged: _Staged, base: Path) -> LibraryEntry:
    base.mkdir(parents=True, exist_ok=True)
    stem = _slug(staged.manifest.name) or f"lora-{(staged.manifest.weights_sha256 or '')[:8]}"
    entry_id = stem
    counter = 2
    while (base / entry_id).exists():
        entry_id = f"{stem}-{counter}"
        counter += 1
    temp = Path(tempfile.mkdtemp(prefix=".import-", dir=base))
    try:
        (temp / WEIGHTS_NAME).write_bytes(staged.weights)
        record = dict(staged.raw_manifest)
        record["weights_sha256"] = staged.manifest.weights_sha256
        record["imported_as"] = entry_id
        (temp / MANIFEST_NAME).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for name, data in staged.previews:
            target = temp / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        temp.rename(base / entry_id)
    except OSError as exc:
        shutil.rmtree(temp, ignore_errors=True)
        raise LoraRejected(CODE_INVALID) from exc
    entry = read_entry(base / entry_id)
    if entry is None:
        shutil.rmtree(base / entry_id, ignore_errors=True)
        raise LoraRejected(CODE_INVALID)
    return entry


def _slug(name: str) -> str:
    return _SLUG.sub("-", name.casefold()).strip("-.")[:64]


def _clamped_range(manifest: Manifest) -> tuple[float, float]:
    low = min(max(manifest.scale_min, STRENGTH_LIMITS[0]), STRENGTH_LIMITS[1])
    high = min(max(manifest.scale_max, STRENGTH_LIMITS[0]), STRENGTH_LIMITS[1])
    return low, max(low, high)


def _base_on_disk() -> bool:
    try:
        from canvas.infra.paths import models_dir
        from canvas.models.registry import load_catalog
        from canvas.models.snapshots import snapshot_ready

        root = models_dir()
        for spec in load_catalog().models:
            if spec.role == "generator" and spec.text_base == ZIMAGE_TURBO_MDL_ID and snapshot_ready(spec, root) is not None:
                return True
    except Exception:  # noqa: BLE001 - an unreadable install counts as missing
        return False
    return False


def _fetch_with_mdl(progress: Callable[[str], None]) -> bool:
    """Queue the default base model with VOXPrint MDL. MDL shows the download progress."""
    try:
        from voxprint_common.mdl import enqueue
        from voxprint_common.mdl.catalog import CatalogError

        from canvas.models.mdl_registry import manifest_path
        from canvas.models.provision import default_specs
        from canvas.models.registry import load_catalog
    except ImportError:
        return False
    try:
        ids = [spec.id for spec in default_specs(load_catalog()) if spec.role == "generator"]
        if not ids:
            return False
        enqueue("canvas", ids, manifests=[manifest_path()])
    except (CatalogError, OSError, ValueError):
        return False
    from canvas.app.i18n import text

    progress(text("canvas.lora.base_queued"))
    return True
