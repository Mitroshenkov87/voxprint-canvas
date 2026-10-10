"""Opt-in third-party weights.

A download runs only after the feature is turned on and the user asks for one
entry. The file comes from that entry's upstream URL and is stored under
``third-party`` in the shared models folder. Hard-coded repository ids and
hashes are refused even when an entry or a manual path names them. The content
checks are not configurable from an entry.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from canvas.models.downloader import DownloadError
from canvas.models.excluded_weights import EXCLUDED_MODEL_IDS, EXCLUDED_SHA256
from canvas.models.pin import PinError, require_pin, verify_bytes
from canvas.models.registry import RegistryError
from canvas.safety.gate import AgeGate
from canvas.safety.image import ScreenedInput
from canvas.safety.inputs import PendingInput

_CATALOG = Path(__file__).with_name("third_party.json")
_FOLDER = "third-party"
_KINDS = frozenset({"lora", "model"})
_BASES = frozenset({"Z-Image-Turbo", "Qwen-Image-2512", "Edit-2511"})
_MIRROR_KINDS = frozenset({"huggingface-mirror", "kaggle", "torrent"})
_MIRROR_HOSTS = frozenset({"hf-mirror.com", "kaggle.com", "www.kaggle.com"})

Transport = Callable[[str], bytes]


@dataclass(frozen=True)
class Addon:
    """One opt-in third-party model. It stays off until you ask for it."""

    id: str
    kind: str
    base: str
    upstream: str
    revision: str
    sha256: str
    licence: str
    author: str
    files: dict[str, str]


@dataclass(frozen=True)
class AddonCatalog:
    """The curated add-on list."""

    addons: tuple[Addon, ...]

    def by_id(self, addon_id: str) -> Addon:
        """The add-on with this id."""
        for addon in self.addons:
            if addon.id == addon_id:
                return addon
        raise RegistryError(f"unknown add-on {addon_id}")


def third_party_dir(models_root: Path) -> Path:
    """Folder for opt-in weights. It is not part of the published model set."""
    return models_root / _FOLDER


def _names_excluded_model(url: str) -> bool:
    folded = url.strip().casefold().rstrip("/")
    for model_id in EXCLUDED_MODEL_IDS:
        needle = f"/models/{model_id}"
        if needle in folded or folded.endswith(needle):
            return True
    return False


def refuse_excluded(url: str, sha256: str = "") -> None:
    """Refuse the hard-coded model id or a known file hash, including a manual path."""
    digest = sha256.strip().casefold()
    if digest and digest in EXCLUDED_SHA256:
        raise RegistryError("a listed add-on is excluded")
    if url and _names_excluded_model(url):
        raise RegistryError("a listed add-on is excluded")


def _https_upstream(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    if parsed.scheme != "https" or not host or host in _MIRROR_HOSTS:
        raise RegistryError("a third-party add-on is downloaded only from upstream")
    return url.strip()


def _relative_name(name: str) -> str:
    parts = Path(name).parts
    if not name or name.startswith(("/", "\\")) or ".." in parts or any(part in {"", ".", ".."} for part in parts):
        raise RegistryError("a third-party file is unpinned")
    return name


def _sha_map(row: dict[str, object]) -> tuple[dict[str, str], str]:
    raw = row.get("sha256")
    if isinstance(raw, dict):
        files: dict[str, str] = {}
        try:
            for name, meta in raw.items():
                if isinstance(meta, dict):
                    digest = require_pin(str(meta.get("sha256") or ""))
                else:
                    digest = require_pin(str(meta or ""))
                files[_relative_name(str(name))] = digest
        except PinError as exc:
            raise RegistryError("a third-party file is unpinned") from exc
        if not files:
            raise RegistryError("a third-party file is unpinned")
        top = [name for name in files if "/" not in name]
        primary = top[0] if len(top) == 1 else next(iter(files))
        return files, files[primary]
    try:
        digest = require_pin(str(raw or ""))
    except PinError as exc:
        raise RegistryError("a third-party file is unpinned") from exc
    return {}, digest


def _identifier(value: str) -> str:
    if not value or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-" for ch in value):
        raise RegistryError("a third-party add-on id is not usable")
    return value


def _parse_addon(row: dict[str, object]) -> Addon:
    sources = row.get("sources")
    if isinstance(sources, list):
        for item in sources:
            if isinstance(item, dict) and str(item.get("kind") or "") in _MIRROR_KINDS:
                raise RegistryError("a third-party add-on cannot use a mirror")
    if "gate" in row and str(row.get("gate") or "") != "always":
        raise RegistryError("the content checks stay on")
    kind = str(row.get("kind") or "")
    base = str(row.get("base") or "")
    if kind not in _KINDS:
        raise RegistryError("a third-party add-on has an unknown kind")
    if kind == "lora" and base not in _BASES:
        raise RegistryError("a third-party add-on has an unknown base")
    if kind == "model" and base and base not in _BASES:
        raise RegistryError("a third-party add-on has an unknown base")
    upstream = _https_upstream(str(row.get("upstream") or ""))
    files, digest = _sha_map(row)
    revision = str(row.get("revision") or "").strip()
    author = str(row.get("author") or "").strip()
    licence = str(row.get("licence") or "").strip()
    if not revision or not author or not licence:
        raise RegistryError("a third-party add-on is missing a field")
    refuse_excluded(upstream, digest)
    for name, file_digest in files.items():
        _relative_name(name)
        refuse_excluded(upstream, file_digest)
    return Addon(
        id=_identifier(str(row.get("id") or "")),
        kind=kind,
        base=base,
        upstream=upstream,
        revision=revision,
        sha256=digest,
        licence=licence,
        author=author,
        files=files,
    )


def load_addons(path: Path | None = None) -> AddonCatalog:
    """Read the curated list. A mirror source or an excluded id does not load."""
    raw = json.loads((path or _CATALOG).read_text(encoding="utf-8"))
    rows = raw.get("addons") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise RegistryError("the third-party registry is unreadable")
    return AddonCatalog(tuple(_parse_addon(row) for row in rows if isinstance(row, dict)))


def release_addon_request(gate: AgeGate, prompt: str, mode: str = "text",
                          inputs: tuple[PendingInput, ...] | list[PendingInput] = ()) -> ScreenedInput:
    """Screen one add-on request. There is no argument that turns the checks off."""
    return gate.release_request(prompt, mode, inputs)


def _save_pinned(folder: Path, name: str, data: bytes, digest: str) -> Path:
    actual = hashlib.sha256(data).hexdigest()
    if actual in EXCLUDED_SHA256:
        raise DownloadError("excluded")
    try:
        verify_bytes(data, digest)
    except PinError as exc:
        raise DownloadError("hash_mismatch") from exc
    try:
        relative = Path(_relative_name(name))
    except RegistryError as exc:
        raise DownloadError("unpinned") from exc
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def download_addon(addon: Addon, models_root: Path, transport: Transport, *,
                   enabled: bool = False, user_initiated: bool = False) -> Path:
    """Fetch pinned files from the upstream URL into the third-party folder."""
    if not enabled or not user_initiated:
        raise DownloadError("not-requested")
    try:
        _https_upstream(addon.upstream)
        refuse_excluded(addon.upstream, addon.sha256)
        for digest in addon.files.values():
            refuse_excluded(addon.upstream, digest)
    except RegistryError as exc:
        message = str(exc).casefold()
        raise DownloadError("excluded" if "excluded" in message else "mirror") from exc
    folder = third_party_dir(models_root) / addon.id
    folder.mkdir(parents=True, exist_ok=True)
    if not addon.files:
        try:
            data = transport(addon.upstream)
        except OSError as exc:
            raise DownloadError(str(exc)) from exc
        name = urlparse(addon.upstream).path.rstrip("/").split("/")[-1] or f"{addon.id}.bin"
        return _save_pinned(folder, name, data, addon.sha256)
    saved: Path | None = None
    for name, digest in addon.files.items():
        url = f"{addon.upstream.rstrip('/')}/{name}"
        try:
            data = transport(url)
        except OSError as exc:
            raise DownloadError(str(exc)) from exc
        path = _save_pinned(folder, name, data, digest)
        if saved is None or "/" not in name:
            saved = path
    if saved is None:
        raise DownloadError("unpinned")
    return saved
