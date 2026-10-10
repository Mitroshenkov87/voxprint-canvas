"""Try each listed source in order and keep a file only when the SHA-256 matches."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from canvas.models.pin import PinError, require_pin, verify_bytes
from canvas.models.registry import ModelSpec, Source


class DownloadError(Exception):
    """A model file was not saved. ``detail`` is for logs, not the window."""

    def __init__(self, detail: str) -> None:
        super().__init__("The model could not be downloaded.")
        self.detail = detail


Transport = Callable[[Source], bytes]


def attempt_sources(spec: ModelSpec) -> tuple[Source, ...]:
    """Sources that have a URI, in registry order."""
    return tuple(source for source in spec.sources if source.uri)


def download_file(spec: ModelSpec, filename: str, dest_dir: Path, transport: Transport) -> Path:
    """Fetch ``filename`` from the first source that returns the pinned bytes."""
    expected = spec.files.get(filename, "")
    try:
        require_pin(expected)
    except PinError:
        raise DownloadError("unpinned") from None
    errors: list[str] = []
    for source in attempt_sources(spec):
        try:
            data = transport(source)
            verify_bytes(data, expected)
        except (OSError, PinError, DownloadError) as exc:
            errors.append(f"{source.kind}: {exc}")
            continue
        folder = dest_dir / spec.repo.replace("/", "--")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename
        path.write_bytes(data)
        return path
    raise DownloadError("; ".join(errors) or "no source")
