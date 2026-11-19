"""Offline count report for a folder of images and one LoRA.

The application does not import this module.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from pathlib import Path

from PIL import Image

from canvas.infra.paths import models_dir
from canvas.models.loader import load_catalog_stack
from canvas.safety.gate import AgeGate, GenerationBlocked, SafetyError
from canvas.safety.image import RawImage

KEY = "voxprint.subject.attr_v"
LOW = 18
HIGH = 99
_SPAN = 2
_TAG = hashlib.sha256().digest_size
_KEY = bytes.fromhex(
    "6ac13e9457b208d6419fe32c7015a84b"
    "c0d7e91a3f58264b8e0c5d7a19b4f263"
)
_RULE_VALUES = frozenset({18, 19, 20})
_ALTERNATE_PROMPT = "19 years old"
_REPORT_KEYS = ("frames", "gate_drops", "alternate_rule_drops", "disagreements")
_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".avif"})


def read_declared(
    path_or_metadata: str | Path | Mapping[str, object],
    tensors: Mapping[str, bytes] | None = None,
) -> int | None:
    """Return the bound value when it verifies and is from 18 to 99.

    A missing field, a failed binding, or a value outside that range yields
    ``None``. This function does not raise.

    Args:
        path_or_metadata: A ``.safetensors`` path, or a metadata mapping used with ``tensors``.
        tensors: Raw little-endian tensor bytes keyed by name. Loaded from the file when omitted.
    """
    try:
        return _read(path_or_metadata, tensors)
    except Exception:
        return None


def build_report(
    images: Path,
    lora: Path,
    *,
    declared_age: int | None = None,
    gate: AgeGate | None = None,
) -> dict[str, int]:
    """Count frames and the two drop totals for one folder.

    Args:
        images: Folder of pictures.
        lora: A ``.safetensors`` file or a LoRA folder.
        declared_age: Optional override. Only 18, 19, and 20 change the second count.
        gate: The content gate. The installed gate is opened when omitted.
    """
    active = gate if gate is not None else AgeGate(load_catalog_stack(models_dir()))
    value = _rule_value(lora, declared_age)
    frames = 0
    gate_drops = 0
    alternate_drops = 0
    disagreements = 0
    for path in _frames(images):
        image = _load_image(path)
        if image is None:
            continue
        frames += 1
        real_drop = _dropped(active, image, "")
        if value is None:
            alternate_drop = real_drop
        else:
            alternate_drop = _dropped(active, image, _ALTERNATE_PROMPT)
        if real_drop:
            gate_drops += 1
        if alternate_drop:
            alternate_drops += 1
        if real_drop != alternate_drop:
            disagreements += 1
    return {
        "frames": frames,
        "gate_drops": gate_drops,
        "alternate_rule_drops": alternate_drops,
        "disagreements": disagreements,
    }


def write_report(path: Path, report: Mapping[str, int]) -> None:
    """Write the count report. A path inside the app state directory is refused.

    Args:
        path: Destination. Parent directories are not created inside the state directory.
        report: The four counts from :func:`build_report`.

    Raises:
        SystemExit: ``path`` is inside the app state directory.
    """
    if _in_state(path):
        raise SystemExit(2)
    payload = {key: int(report[key]) for key in _REPORT_KEYS}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Read a folder and a LoRA and write the count report.

    Args:
        argv: Command arguments, excluding the program name.
    """
    parser = argparse.ArgumentParser(prog="declared_age_shadow")
    parser.add_argument("images", type=Path)
    parser.add_argument("lora", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--declared-age", type=int, default=None)
    args = parser.parse_args(argv)
    if _in_state(args.report) or not args.images.is_dir() or not args.lora.exists():
        return 2
    try:
        report = build_report(args.images, args.lora, declared_age=args.declared_age)
    except SafetyError:
        return 2
    write_report(args.report, report)
    return 0


def _read(
    path_or_metadata: str | Path | Mapping[str, object],
    tensors: Mapping[str, bytes] | None,
) -> int | None:
    meta: Mapping[str, object]
    if isinstance(path_or_metadata, (str, Path)):
        loaded = _load(Path(path_or_metadata))
        if loaded is None:
            return None
        meta, file_tensors = loaded
        if tensors is None:
            tensors = file_tensors
    elif isinstance(path_or_metadata, Mapping):
        meta = path_or_metadata
    else:
        return None
    if tensors is None:
        return None
    raw = meta.get(KEY)
    if not isinstance(raw, str) or not raw:
        return None
    blob = base64.b64decode(raw, validate=True)
    if len(blob) != _SPAN + _TAG:
        return None
    hidden, tag = blob[:_SPAN], blob[_SPAN:]
    if not hmac.compare_digest(tag, _tag(_digest(tensors), hidden)):
        return None
    return _accepted(_show(hidden))


def _accepted(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if LOW <= value <= HIGH:
        return value
    return None


def _rule_value(lora: Path, override: int | None) -> int | None:
    if override is not None:
        value: int | None = override if _accepted(override) is not None else None
    else:
        weights = _weights_path(lora)
        value = None if weights is None else read_declared(weights)
    if value in _RULE_VALUES:
        return value
    return None


def _weights_path(lora: Path) -> Path | None:
    if lora.is_file():
        return lora
    if lora.is_dir():
        named = lora / f"{lora.name}.safetensors"
        if named.is_file():
            return named
    return None


def _load(path: Path) -> tuple[dict[str, object], dict[str, bytes]] | None:
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
        header = json.loads(data[8:8 + size].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(header, dict):
        return None
    raw_meta = header.get("__metadata__") or {}
    if not isinstance(raw_meta, dict):
        return None
    meta: dict[str, object] = {str(key): value for key, value in raw_meta.items()}
    base = 8 + size
    tensors: dict[str, bytes] = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(key, str) or not isinstance(info, dict):
            return None
        offsets = info.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            return None
        start, end = offsets
        if isinstance(start, bool) or isinstance(end, bool):
            return None
        if not isinstance(start, int) or not isinstance(end, int):
            return None
        if start < 0 or end < start or base + end > len(data):
            return None
        tensors[key] = data[base + start:base + end]
    return meta, tensors


def _show(hidden: bytes) -> int:
    mask = hashlib.sha256(_KEY).digest()
    raw = bytes(byte ^ mask[index] for index, byte in enumerate(hidden))
    return int.from_bytes(raw, "big")


def _tag(digest: str, hidden: bytes) -> bytes:
    return hmac.new(_KEY, digest.encode("ascii") + hidden, hashlib.sha256).digest()


def _digest(tensors: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for key in sorted(tensors):
        raw = tensors[key]
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("tensor bytes are required")
        digest.update(key.encode())
        digest.update(bytes(raw))
    return digest.hexdigest()


def _frames(folder: Path) -> tuple[Path, ...]:
    if not folder.is_dir():
        return ()
    return tuple(sorted(path for path in folder.iterdir() if path.suffix.casefold() in _SUFFIXES))


def _load_image(path: Path) -> RawImage | None:
    try:
        with Image.open(path) as handle:
            rgb = handle.convert("RGB")
            return RawImage(rgb.width, rgb.height, rgb.tobytes())
    except OSError:
        return None


def _dropped(gate: AgeGate, image: RawImage, prompt: str) -> bool:
    copy = RawImage(image.width, image.height, image.to_bytes())
    try:
        gate.review_image(copy, prompt)
    except (GenerationBlocked, SafetyError):
        return True
    return False


def _in_state(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return True
    for root in _state_roots():
        try:
            base = root.resolve()
        except OSError:
            continue
        if resolved == base or base in resolved.parents:
            return True
    return False


def _state_roots() -> tuple[Path, ...]:
    override = os.environ.get("VOXPRINT_HOME", "").strip()
    if override:
        home = Path(override)
    elif os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "")
        home = Path(local) / "VOXPrint AI Suite" if local else Path.home() / "AppData" / "Local" / "VOXPrint AI Suite"
    else:
        home = Path.home() / ".local" / "share" / "voxprint-ai-suite"
    return (home / "state",)


if __name__ == "__main__":
    raise SystemExit(main())
