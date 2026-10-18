"""Tiny LoRA folders for tests. The application does not import this module."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def write_safetensors(path: Path, tensors: dict[str, bytes], metadata: dict[str, str]) -> None:
    """Write one safetensors file from raw little-endian tensor bytes."""
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = metadata
    offset = 0
    parts: list[bytes] = []
    for name in sorted(tensors):
        raw = tensors[name]
        header[name] = {"dtype": "U8", "shape": [len(raw)], "data_offsets": [offset, offset + len(raw)]}
        parts.append(raw)
        offset += len(raw)
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(payload).to_bytes(8, "little") + payload + b"".join(parts))


def write_lora(
    root: Path,
    lora_id: str,
    *,
    kind: str = "object",
    tensors: dict[str, bytes] | None = None,
    metadata: dict[str, str] | None = None,
    adult_content: bool | None = None,
    base: str = "z-image-turbo",
    trigger: str = "triggerword",
    default_strength: float = 0.8,
    strength_range: tuple[float, float] = (0.0, 1.5),
    person_flags: str | None = "true",
    extra: dict[str, object] | None = None,
    notice: bool = True,
    license_file: bool = True,
    preview: bytes | None = None,
    preview_hash: str | None = None,
    file_hash: str | None = None,
    file_size: int | None = None,
) -> Path:
    """Write ``root/user-models/lora/<id>`` and return the folder."""
    folder = root / "user-models" / "lora" / lora_id
    folder.mkdir(parents=True)
    weights = tensors if tensors is not None else {"lora.down": b"\x01\x02\x03\x04"}
    header = {
        "voxprint.subject.trigger": trigger,
        "voxprint.lora.kind": kind,
        "voxprint.lora.base": base,
    }
    if kind == "person" and person_flags is not None:
        header["voxprint.subject.consent_obtained"] = person_flags
        header["voxprint.subject.adult_confirmed"] = person_flags
    if metadata:
        header.update(metadata)
    weight_path = folder / f"{lora_id}.safetensors"
    write_safetensors(weight_path, weights, header)
    if license_file:
        (folder / "LICENSE").write_text("apache-2.0\n", encoding="utf-8")
    if notice:
        (folder / "NOTICE").write_text("notice\n", encoding="utf-8")
    if preview is not None:
        (folder / "preview.webp").write_bytes(preview)
    document: dict[str, object] = {
        "schema": "voxprint-model/1",
        "min_schema": 1,
        "type": "lora",
        "kind": kind,
        "id": lora_id,
        "file": f"{lora_id}.safetensors",
        "sha256": file_hash if file_hash is not None else hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        "size": file_size if file_size is not None else weight_path.stat().st_size,
        "base": {"mdl_id": base, "revision": "f332072aa78be7aecdf3ee76d5c247082da564a6"},
        "default_strength": default_strength,
        "strength_range": [strength_range[0], strength_range[1]],
        "licence": "apache-2.0",
        "tags": [],
        "producer": {"name": "Test", "url": "https://example.com"},
    }
    if adult_content is not None:
        document["voxprint.lora.adult_content"] = adult_content
    if preview is not None:
        document["preview"] = "preview.webp"
        document["preview_sha256"] = preview_hash if preview_hash is not None else hashlib.sha256(preview).hexdigest()
    if extra:
        document.update(extra)
    (folder / "voxprint-model.json").write_text(json.dumps(document), encoding="utf-8")
    return folder
