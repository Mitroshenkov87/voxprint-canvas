"""Forge-verified LoRA records: metadata only, no effect on the content gate.

A LoRA counts as *verified* when a signed record sits next to its weights
file and both of these hold:

1. the record's signature checks out against this install's key, and
2. the record's ``lora_sha256`` matches the exact weights file being loaded.

Canvas screens the request, not the result (owner decision 2026-11-16, which
supersedes the 2026-11-15 verified-LoRA rule): every generation runs the
prompt check and, when the request carries an uploaded image, the input-image
checks in :meth:`canvas.safety.gate.AgeGate.release_request`. No classifier
runs on generated pictures, with any LoRA or none. Verification therefore no
longer changes what the gate does; :func:`output_check_decision` and
:func:`verify_lora` stay as record metadata (Forge runs the input-image checks
on its training set, and the record says so).

This module never relaxes the minors gate: it cannot widen what the prompt
filter, refusal_stage, thresholds, or the input-image checks do.

The signing scheme is Ed25519 with a per-install key kept in the shared suite
folder. Forge signs records with the same key; see ``tools/forge_sign_lora.py``
and the ``SCHEMA`` section of the design note.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from canvas.infra.paths import voxprint_home

if TYPE_CHECKING:
    from canvas.safety.gate import AgeGate
    from canvas.safety.image import RawImage, SafeImage

RECORD_SUFFIX = ".voxprint-verify.json"
SCHEMA_ID = "voxprint-lora-verify/1"
ALGORITHM = "ed25519"
_KEY_NAME = "lora-verify-ed25519.key"
_READ_CHUNK = 1024 * 1024

# Fields covered by the signature, in a fixed set. The canonical message is
# the JSON of exactly these fields with sorted keys; ``signature`` and any
# extra fields are never trusted because they are not signed.
SIGNED_FIELDS = (
    "schema",
    "record_id",
    "lora_file",
    "lora_sha256",
    "project_id",
    "dataset_image_count",
    "age_gate_version",
    "verified",
    "result",
    "verified_at",
    "algorithm",
)


@dataclass(frozen=True)
class VerifyResult:
    """The outcome of checking one LoRA's record."""

    path: Path
    verified: bool
    reason: str
    record_id: str = ""
    lora_sha256: str = ""


@dataclass(frozen=True)
class SkipDecision:
    """Whether every LoRA of a generation is Forge-verified. Metadata only.

    ``skip`` keeps its historical name: it is True when every loaded LoRA is
    verified. The gate no longer reads it (no output check runs either way).
    """

    skip: bool
    reason: str
    records: tuple[VerifyResult, ...] = ()


def keys_dir() -> Path:
    """The per-install key folder inside the shared suite folder."""
    path = voxprint_home() / "keys"
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def key_path() -> Path:
    """Absolute path of this install's Ed25519 signing key."""
    return keys_dir() / _KEY_NAME


def _load_private_key():
    """Load this install's signing key, creating it (0600) on first use."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    path = key_path()
    if not path.exists():
        return _create_private_key(path)
    raw = path.read_bytes()
    if len(raw) != 32:
        raise ValueError("lora-verify key is not a 32-byte Ed25519 seed")
    return Ed25519PrivateKey.from_private_bytes(raw)


def _create_private_key(path: Path):
    """Generate and persist a fresh Ed25519 key with 0600 permissions.

    The write is exclusive. A second process that loses the race reads the key
    the winner wrote instead of overwriting it.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    keys_dir()
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        existing = path.read_bytes()
        if len(existing) != 32:
            raise ValueError("lora-verify key is not a 32-byte Ed25519 seed")
        return Ed25519PrivateKey.from_private_bytes(existing)
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


def canonical_message(record: dict) -> bytes:
    """The exact bytes signed and verified. Raises ``KeyError`` if incomplete."""
    payload = {field: record[field] for field in SIGNED_FIELDS}
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def sign_record(record: dict) -> str:
    """Return the hex Ed25519 signature over ``record``'s signed fields."""
    key = _load_private_key()
    return key.sign(canonical_message(record)).hex()


def _verify_signature(message: bytes, signature_hex: str) -> bool:
    from cryptography.exceptions import InvalidSignature

    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    key = _load_private_key()
    try:
        key.public_key().verify(signature, message)
        return True
    except InvalidSignature:
        return False


def file_sha256(path: Path) -> str:
    """Streaming sha256 of a file, lower-case hex."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_path_for(weights: Path) -> Path:
    """The ``<weights>.voxprint-verify.json`` path next to a LoRA file."""
    weights = Path(weights)
    return weights.with_name(weights.name + RECORD_SUFFIX)


def verify_lora(weights: Path) -> VerifyResult:
    """Check one LoRA's record. Fail-closed: any problem means not verified."""
    weights = Path(weights)
    try:
        if not weights.is_file():
            return VerifyResult(weights, False, "weights-missing")
        record_path = record_path_for(weights)
        if not record_path.is_file():
            return VerifyResult(weights, False, "record-missing")
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return VerifyResult(weights, False, "record-unreadable")
        if not isinstance(record, dict):
            return VerifyResult(weights, False, "record-malformed")
        if record.get("schema") != SCHEMA_ID:
            return VerifyResult(weights, False, "schema-mismatch")
        if record.get("algorithm") != ALGORITHM:
            return VerifyResult(weights, False, "algorithm-mismatch")
        if record.get("verified") is not True or record.get("result") != "pass":
            return VerifyResult(weights, False, "not-a-pass")
        signature = record.get("signature")
        if not isinstance(signature, str) or not signature:
            return VerifyResult(weights, False, "signature-missing")
        record_id = str(record.get("record_id") or "")
        try:
            message = canonical_message(record)
        except KeyError:
            return VerifyResult(weights, False, "record-incomplete", record_id=record_id)
        if not _verify_signature(message, signature):
            return VerifyResult(weights, False, "bad-signature", record_id=record_id)
        actual = file_sha256(weights)
        claimed = str(record.get("lora_sha256") or "").lower()
        if actual.lower() != claimed:
            return VerifyResult(weights, False, "hash-mismatch", record_id=record_id, lora_sha256=actual)
        return VerifyResult(weights, True, "verified", record_id=record_id, lora_sha256=actual)
    except Exception:  # noqa: BLE001 - fail-closed on anything unexpected
        return VerifyResult(weights, False, "error")


def _as_path(item: object) -> Path | None:
    if isinstance(item, Path):
        return item
    if isinstance(item, str) and item:
        return Path(item)
    return None


def loaded_lora_files(holder: object) -> list[Path]:
    """Weights files for every LoRA applied to a generator or draw-frame.

    Looks for a ``lora_files`` sequence first (a future multi-LoRA stack),
    then a single ``bound_lora`` or ``lora`` object with a ``weights`` path,
    then a ``lora_path`` string. Returns an empty list when nothing is loaded.
    """
    explicit = getattr(holder, "lora_files", None)
    if explicit:
        files: list[Path] = []
        for item in explicit:
            path = _as_path(item)
            if path is not None:
                files.append(path)
        return files
    for attr in ("bound_lora", "lora"):
        weights = getattr(getattr(holder, attr, None), "weights", None)
        path = _as_path(weights)
        if path is not None:
            return [path]
    path = _as_path(getattr(holder, "lora_path", ""))
    return [path] if path is not None else []


def output_check_decision(lora_files: Sequence[Path]) -> SkipDecision:
    """Whether a generation's LoRAs are all Forge-verified. Metadata only.

    ``skip`` is True only when there is at least one LoRA and every one of
    them is verified. The content gate does not depend on this.
    """
    files = [path for path in (_as_path(item) for item in lora_files) if path is not None]
    if not files:
        return SkipDecision(False, "no-lora")
    results = tuple(verify_lora(path) for path in files)
    if all(result.verified for result in results):
        return SkipDecision(True, "all-verified", results)
    return SkipDecision(False, "unverified-lora", results)


def review_or_skip(
    gate: AgeGate, raw: RawImage, prompt: str, lora_files: Sequence[Path],
) -> SafeImage:
    """Deprecated: release a generated picture. Same as :func:`release_generated`.

    Kept as a thin alias for older callers. ``gate``, ``prompt`` and
    ``lora_files`` are ignored: LoRA verification no longer changes the gate,
    and no classifier runs on generated pictures. The raw buffer is wiped.
    """
    from canvas.safety.gate import release_generated

    return release_generated(raw)
