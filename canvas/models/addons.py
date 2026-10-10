"""Opt-in third-party weights. The curated list ships empty.

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
from canvas.models.excluded_weights import EXCLUDED_REPOS, EXCLUDED_SHA256
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
    id: str
    kind: str
    base: str
    upstream: str
    revision: str
    sha256: str
    licence: str
    author: str


@dataclass(frozen=True)
class AddonCatalog:
    addons: tuple[Addon, ...]

    def by_id(self, addon_id: str) -> Addon:
        for addon in self.addons:
            if addon.id == addon_id:
                return addon
        raise RegistryError(f"unknown add-on {addon_id}")


def third_party_dir(models_root: Path) -> Path:
    """Folder for opt-in weights. It is not part of the published model set."""
    return models_root / _FOLDER


def _repo_key(repo: str) -> str:
    return repo.strip().casefold()


def _repo_in_url(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    if host != "huggingface.co":
        return ""
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return ""
    return f"{parts[0]}/{parts[1]}"


def _names_excluded_repo(url: str) -> bool:
    named = _repo_key(_repo_in_url(url))
    if named and named in {_repo_key(repo) for repo in EXCLUDED_REPOS}:
        return True
    folded = url.casefold()
    for repo in EXCLUDED_REPOS:
        needle = repo.casefold()
        if f"/{needle}/" in folded or folded.rstrip("/").endswith("/" + needle):
            return True
    return False


def refuse_excluded(url: str, sha256: str = "") -> None:
    """Refuse a hard-coded repository id or weight hash, including a manual path."""
    digest = sha256.strip().casefold()
    if digest and digest in EXCLUDED_SHA256:
        raise RegistryError("a listed add-on is excluded")
    if url and _names_excluded_repo(url):
        raise RegistryError("a listed add-on is excluded")


def _https_upstream(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    if parsed.scheme != "https" or not host or host in _MIRROR_HOSTS:
        raise RegistryError("a third-party add-on is downloaded only from upstream")
    return url.strip()


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
    if base not in _BASES:
        raise RegistryError("a third-party add-on has an unknown base")
    upstream = _https_upstream(str(row.get("upstream") or ""))
    try:
        digest = require_pin(str(row.get("sha256") or ""))
    except PinError as exc:
        raise RegistryError("a third-party file is unpinned") from exc
    revision = str(row.get("revision") or "").strip()
    author = str(row.get("author") or "").strip()
    licence = str(row.get("licence") or "").strip()
    if not revision or not author or not licence:
        raise RegistryError("a third-party add-on is missing a field")
    refuse_excluded(upstream, digest)
    return Addon(
        id=_identifier(str(row.get("id") or "")),
        kind=kind,
        base=base,
        upstream=upstream,
        revision=revision,
        sha256=digest,
        licence=licence,
        author=author,
    )


def load_addons(path: Path | None = None) -> AddonCatalog:
    """Read the curated list. An empty list is the shipped registry."""
    raw = json.loads((path or _CATALOG).read_text(encoding="utf-8"))
    rows = raw.get("addons") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise RegistryError("the third-party registry is unreadable")
    return AddonCatalog(tuple(_parse_addon(row) for row in rows if isinstance(row, dict)))


def release_addon_request(gate: AgeGate, prompt: str, mode: str = "text",
                          inputs: tuple[PendingInput, ...] | list[PendingInput] = ()) -> ScreenedInput:
    """Screen one add-on request. There is no argument that turns the checks off."""
    return gate.release_request(prompt, mode, inputs)


def download_addon(addon: Addon, models_root: Path, transport: Transport, *,
                   enabled: bool = False, user_initiated: bool = False) -> Path:
    """Fetch one pinned file from its upstream URL into the third-party folder."""
    if not enabled or not user_initiated:
        raise DownloadError("not-requested")
    try:
        _https_upstream(addon.upstream)
        refuse_excluded(addon.upstream, addon.sha256)
    except RegistryError as exc:
        message = str(exc).casefold()
        raise DownloadError("excluded" if "excluded" in message else "mirror") from exc
    try:
        data = transport(addon.upstream)
    except OSError as exc:
        raise DownloadError(str(exc)) from exc
    actual = hashlib.sha256(data).hexdigest()
    if actual in EXCLUDED_SHA256:
        raise DownloadError("excluded")
    try:
        verify_bytes(data, addon.sha256)
    except PinError as exc:
        raise DownloadError("hash_mismatch") from exc
    name = urlparse(addon.upstream).path.rstrip("/").split("/")[-1] or f"{addon.id}.bin"
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise DownloadError("unpinned")
    folder = third_party_dir(models_root) / addon.id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(data)
    return path
