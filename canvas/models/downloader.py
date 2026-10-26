"""Ask VOXPrint MDL for a model. A plain copy is never written."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from canvas.app.i18n import text
from canvas.models.pin import PinError, require_pin
from canvas.models.registry import ModelSpec, Source
from canvas.models.snapshots import request_download, snapshot_ready


class DownloadError(Exception):
    """A model file was not saved. ``detail`` is for logs, not the window."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail

    def __str__(self) -> str:
        if self.detail == "model missing":
            return text("model.missing")
        return text("error.download")


Transport = Callable[[Source], bytes]


def attempt_sources(spec: ModelSpec) -> tuple[Source, ...]:
    """Sources that have a URI, in registry order."""
    return tuple(source for source in spec.sources if source.uri)


def download_file(spec: ModelSpec, filename: str, dest_dir: Path, transport: Transport) -> Path:
    """Queue ``filename`` with MDL and return it from the view.

    ``transport`` is not used. MDL writes the blob. This function does not
    create ``<org>--<name>`` or copy bytes into it.
    """
    del transport
    expected = spec.files.get(filename, "")
    try:
        require_pin(expected)
    except PinError:
        raise DownloadError("unpinned") from None
    request_download(spec.id)
    folder = snapshot_ready(spec, dest_dir)
    path = (folder / filename) if folder is not None else None
    if path is None or not path.is_file():
        raise DownloadError("model missing")
    return path
