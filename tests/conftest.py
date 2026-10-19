"""Test defaults. The offscreen Qt platform is set before any window test imports PySide6."""
import os

import pytest

from canvas.infra.vram import set_vram_cap_gb
from canvas.models.runtime import reset_pipeline_cache

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _clear_vram_cap():
    """Drop a VRAM cap left by an earlier test."""
    set_vram_cap_gb(None)
    yield
    set_vram_cap_gb(None)


@pytest.fixture(autouse=True)
def _reset_pipeline_cache():
    """Drop a pipeline left open by a window or a parked run."""
    reset_pipeline_cache()
    yield
    reset_pipeline_cache()
