"""Product switches from the registry."""
from __future__ import annotations

from canvas.models.registry import Catalog, ModelSpec


def withheld_from_setup(spec: ModelSpec, catalog: Catalog | None = None) -> bool:
    """Setup, repair, and first launch must not fetch a disabled pack."""
    del catalog
    return spec.status == "disabled"
