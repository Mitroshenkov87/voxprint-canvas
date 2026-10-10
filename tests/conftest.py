"""Test defaults. The offscreen Qt platform is set before any window test imports PySide6."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
