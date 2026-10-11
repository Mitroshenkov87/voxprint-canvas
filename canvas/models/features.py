"""Product switches from the registry.

``features.instruction_edit`` is the only switch for instruction editing.
Changing its status to ``available`` turns the window, the command, and the
download back on. The loader, the pins, and the tests stay either way.
"""
from __future__ import annotations

from canvas.models.edit import EDIT_ID
from canvas.models.registry import Catalog, FeatureFlag, ModelSpec, load_catalog

INSTRUCTION_EDIT = "instruction_edit"


def instruction_edit_flag(catalog: Catalog | None = None) -> FeatureFlag:
    """The instruction-edit switch. A missing switch is available."""
    source = load_catalog() if catalog is None else catalog
    return source.features.get(INSTRUCTION_EDIT, FeatureFlag("available", ""))


def instruction_edit_available(catalog: Catalog | None = None) -> bool:
    """True when instruction editing may be offered and downloaded."""
    return instruction_edit_flag(catalog).status != "unavailable"


def withheld_from_setup(spec: ModelSpec, catalog: Catalog | None = None) -> bool:
    """Setup, repair, and first launch must not fetch this pack.

    Disabled rows, the Nunchaku runtime, and INT4 packs stay out. Instruction
    edit stays out while its feature switch is unavailable.
    """
    if spec.status == "disabled":
        return True
    if spec.runtime == "nunchaku" or spec.quant == "int4":
        return True
    if spec.quant_backend.casefold().startswith("nunchaku"):
        return True
    if spec.id == EDIT_ID and not instruction_edit_available(catalog):
        return True
    return False
