"""Optional ``canvas.toml``: a few knobs that already exist as flags or environment variables.

The file is ``<data home>/canvas/canvas.toml``; ``CANVAS_CONFIG`` names another path.
It is read once per process. A missing file gives the defaults. An unknown key or
an invalid value gives the default for that key and one log line. Command-line
flags and environment variables still win over the file.

Keys:
    gpu: CUDA device index, used when ``CUDA_VISIBLE_DEVICES`` is not set.
    placement: ``ram32``, ``ram64`` or ``ram96``, used when ``VOXPRINT_RAM_MODE`` is not set.
    keep_warm_s: seconds the window keeps its loaded pipeline while idle. 0 keeps it until close.
    writer: true runs the prompt writer on ``generate`` without ``--enhance``.
    steps: sampler steps for every picture. 0 uses the model default.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from canvas.infra.errors import note_error

PLACEMENTS = ("ram32", "ram64", "ram96")
MAX_STEPS = 150
MAX_KEEP_WARM_S = 24 * 3600
MAX_GPU = 63
_KEYS = frozenset({"gpu", "placement", "keep_warm_s", "writer", "steps"})


@dataclass(frozen=True)
class Settings:
    """The knobs. ``None`` or 0 means the built-in behaviour."""

    gpu: int | None = None
    placement: str | None = None
    keep_warm_s: int = 0
    writer: bool = False
    steps: int = 0


_cached: Settings | None = None


def config_path() -> Path:
    """The ``canvas.toml`` path."""
    named = os.environ.get("CANVAS_CONFIG", "").strip()
    if named:
        return Path(named)
    from canvas.infra.paths import voxprint_home

    return voxprint_home() / "canvas" / "canvas.toml"


def _whole(value: object, low: int, high: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        return None
    return value


def parse(data: dict[str, Any]) -> Settings:
    """Settings from a parsed table. Each bad entry is logged once and keeps its default."""
    fields: dict[str, Any] = {}
    for key, value in data.items():
        if key not in _KEYS:
            note_error(f"settings: unknown key {key}")
            continue
        if key == "gpu":
            parsed: Any = _whole(value, 0, MAX_GPU)
        elif key == "placement":
            parsed = value if isinstance(value, str) and value in PLACEMENTS else None
        elif key == "keep_warm_s":
            parsed = _whole(value, 0, MAX_KEEP_WARM_S)
        elif key == "writer":
            parsed = value if isinstance(value, bool) else None
        else:
            parsed = _whole(value, 0, MAX_STEPS)
        if parsed is None:
            note_error(f"settings: invalid value for {key}")
            continue
        fields[key] = parsed
    return Settings(**fields)


def load(path: Path) -> Settings:
    """Read one file. A missing file is the defaults; an unreadable one is the defaults plus a log line."""
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        return Settings()
    except (OSError, tomllib.TOMLDecodeError):
        note_error("settings: unreadable file")
        return Settings()
    return parse(data)


def get() -> Settings:
    """The settings for this process, read on first use."""
    global _cached
    if _cached is None:
        _cached = load(config_path())
    return _cached


def reset() -> None:
    """Forget the cached settings so the next ``get`` reads the file again."""
    global _cached
    _cached = None
