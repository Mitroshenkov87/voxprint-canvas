"""Photoshoot series constants shared by the planner, the writer, the CLI and the window.

This module stays free of model and pipeline imports so the window can read the
series limits at startup without pulling in the writer pipeline.
"""
from __future__ import annotations

MAX_FRAMES = 99
DEFAULT_FRAMES = 6
TEST_SERIES_CHARACTER = "an Asian woman, about 30, blonde, athletic with defined muscles"
