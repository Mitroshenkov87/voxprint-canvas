"""Test defaults. The offscreen Qt platform is set before any window test imports PySide6."""
import os

import pytest

from canvas.infra.vram import set_vram_cap_gb
from canvas.models.edit_route import ENV_EDIT_ROUTE, set_edit_route_override

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _clear_vram_cap():
    """Drop a VRAM cap and an edit-route override left by an earlier test."""
    set_vram_cap_gb(None)
    set_edit_route_override(None)
    os.environ.pop(ENV_EDIT_ROUTE, None)
    yield
    set_vram_cap_gb(None)
    set_edit_route_override(None)
    os.environ.pop(ENV_EDIT_ROUTE, None)
