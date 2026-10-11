"""Test defaults. The offscreen Qt platform is set before any window test imports PySide6."""
import os

import pytest

from canvas.infra.vram import set_vram_cap_gb

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _clear_vram_cap():
    """Drop a VRAM cap left by an earlier test. The cap is process-wide."""
    set_vram_cap_gb(None)
    yield
    set_vram_cap_gb(None)
