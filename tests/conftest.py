"""Test defaults. The offscreen Qt platform is set before any window test imports PySide6."""
import os

import pytest

from canvas.infra.vram import set_vram_cap_gb

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _clear_vram_cap():
    """Drop a VRAM cap left by an earlier test."""
    set_vram_cap_gb(None)
    yield
    set_vram_cap_gb(None)


def _pipeline_reset():
    """Load the pipeline reset.

    Imported inside the function because the PDF guide job loads this file
    without the runtime dependencies.
    """
    try:
        from canvas.models.runtime import reset_pipeline_cache
    except ImportError:
        return None
    return reset_pipeline_cache


@pytest.fixture(autouse=True)
def _reset_pipeline_cache():
    """Drop a pipeline left open by a window or a parked run."""
    reset = _pipeline_reset()
    if reset is not None:
        reset()
    yield
    reset = _pipeline_reset()
    if reset is not None:
        reset()


@pytest.fixture(autouse=True)
def _unknown_free_ram(monkeypatch):
    """Tests do not depend on the free RAM of the machine that runs them."""
    try:
        from canvas.models import hot_tier
    except ImportError:
        return
    monkeypatch.setattr(hot_tier, "free_bytes", lambda: 0)


@pytest.fixture(autouse=True)
def _private_ram_ledger(monkeypatch, tmp_path_factory):
    """Each test sees only the RAM ledger entries it writes."""
    monkeypatch.setenv("VOXPRINT_RAM_LEDGER", str(tmp_path_factory.mktemp("ram-ledger")))


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "real_shapes: check LoRA matrices against the shipped Z-Image-Turbo layer table")


@pytest.fixture(autouse=True)
def _toy_lora_shapes(request, monkeypatch):
    """Toy LoRAs use real layer names with small matrices, so the shipped shape table is off unless asked for."""
    if request.node.get_closest_marker("real_shapes") is None:
        from canvas.models import user_lora

        monkeypatch.setattr(user_lora, "_base_shapes", lambda: {})
