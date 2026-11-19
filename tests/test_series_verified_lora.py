"""Photoshoot series frames are released without an output check.

Owner rule 2026-11-16: Canvas screens the request (prompt and input images),
not the generated pixels. ``_released_frame`` takes no LoRA and no gate, so a
Forge-verified record cannot change it; the LoRA cases at series level are in
``tests/test_series_phases.py``.
"""
from __future__ import annotations

from canvas.pipeline.series import _released_frame
from canvas.safety.image import SafeImage


class _Frame:
    """A minimal SeriesFrame: 2x2 RGB pixels."""

    width = 2
    height = 2

    def to_bytes(self) -> bytes:
        return bytes([10, 20, 30]) * 4


def test_series_frame_is_released_without_output_check(monkeypatch):
    def no_check(*_args, **_kwargs):
        raise AssertionError("no classifier may run on a generated frame")

    monkeypatch.setattr("canvas.safety.gate.AgeGate.review_image", no_check)
    monkeypatch.setattr("canvas.safety.gate.AgeGate._check_image", no_check)
    released, refusal = _released_frame(_Frame())
    assert refusal is None
    assert type(released.safe) is SafeImage
    assert released.safe.data == bytes([10, 20, 30]) * 4
    assert (released.width, released.height) == (2, 2)
