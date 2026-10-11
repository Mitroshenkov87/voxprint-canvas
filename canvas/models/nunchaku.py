"""Plan A: a Nunchaku SVDQuant INT4 Z-Image-Turbo generator.

The registry row is disabled until Model Forge confirms the weights and the
licence. This module does not download anything. A ready row would load the
optional ``nunchaku`` package; that package is not a Canvas dependency.
"""
from __future__ import annotations

from typing import Any

from canvas.models.registry import ModelSpec

NUNCHAKU_ID = "z-image-turbo-svdquant-int4"
NUNCHAKU_BACKEND = "nunchaku-svdquant"
NUNCHAKU_VRAM_GB = 12.0


class NunchakuUnavailable(Exception):
    """The INT4 generator is not confirmed, or its runtime is not installed."""

    def __init__(self) -> None:
        super().__init__("The INT4 generator is not available yet.")


def nunchaku_ready(spec: ModelSpec) -> bool:
    """True only when Forge has confirmed weights, a licence, and a pin."""
    return (
        spec.status == "ready"
        and spec.quant == "int4"
        and spec.quant_backend == NUNCHAKU_BACKEND
        and bool(spec.files)
        and spec.licence.casefold() in {"apache-2.0", "mit"}
    )


def load_nunchaku_generator(spec: ModelSpec) -> Any:
    """Open the INT4 generator. A disabled row raises before any file is fetched.

    The ``nunchaku`` import stays inside this function. The package is optional
    and is not installed with Canvas, so a top-level import would break the
    unit tests. It is not a circular import.
    """
    if not nunchaku_ready(spec):
        raise NunchakuUnavailable()
    try:
        import nunchaku
    except ImportError as exc:
        raise NunchakuUnavailable() from exc
    return nunchaku
