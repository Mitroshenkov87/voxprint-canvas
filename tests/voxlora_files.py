"""Synthetic Model Forge ``.voxlora`` packages for tests. The application does not import this module."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image

from tests.lora_files import write_safetensors

TENSORS = {"layers.0.attn.to_q.lora_A.weight": b"\x01\x02\x03\x04", "layers.0.attn.to_q.lora_B.weight": b"\x05\x06\x07\x08"}


def webp(color: tuple[int, int, int] = (90, 120, 160), size: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, "WEBP")
    return buffer.getvalue()


def weights_bytes(tmp: Path, header: dict[str, str]) -> bytes:
    path = tmp / "w.safetensors"
    write_safetensors(path, TENSORS, header)
    data = path.read_bytes()
    path.unlink()
    return data


def manifest(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": "voxprint-model/1",
        "name": "Brass Lamp",
        "base": "z-image",
        "trigger": "brsslmp",
        "scale_default": 0.8,
        "kind": "object",
        "forge_version": "0.1.0-rc.4",
        "created": "2026-10-19T20:00:00Z",
    }
    record.update(overrides)
    return {key: value for key, value in record.items() if value is not None}


def write_voxlora(
    folder: Path,
    *,
    name: str = "lamp.voxlora",
    record: dict[str, object] | None = None,
    header: dict[str, str] | None = None,
    sha: str | None = "auto",
    previews: int = 2,
    preview: bytes | None = None,
    compress: bool = False,
) -> Path:
    """Write one package and return its path."""
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(record if record is not None else manifest())
    meta = header if header is not None else {
        "voxprint.lora.kind": str(record.get("kind", "object")),
        "voxprint.lora.base": str(record.get("base", "z-image")),
        "voxprint.subject.trigger": str(record.get("trigger", "")),
    }
    weights = weights_bytes(folder, meta)
    if sha == "auto":
        record["weights_sha256"] = hashlib.sha256(weights).hexdigest()
    elif sha is not None:
        record["weights_sha256"] = sha
    extra = {f"previews/{index:02d}.webp": webp((40 * index, 90, 160)) for index in range(1, previews + 1)}
    if extra:
        record["previews"] = sorted(extra)
    path = folder / name
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", mode) as archive:
        archive.writestr("model.safetensors", weights)
        archive.writestr("voxprint-model.json", json.dumps(record))
        archive.writestr("preview.webp", preview if preview is not None else webp())
        for entry, data in extra.items():
            archive.writestr(entry, data)
    return path
