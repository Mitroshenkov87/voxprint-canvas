"""Acrylic backdrop on Windows 11. Other platforms stay on the plain dark theme.

Adapted from Voxprint AI Audiobook Builder ``infra/platform_win.py``
(Apache-2.0, Copyright Aleksandr Mitroshenkov).
"""
from __future__ import annotations

import sys

import ctypes
from ctypes import wintypes

DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_SYSTEMBACKDROP_TYPE = 38
DWMSBT_TRANSIENTWINDOW = 3
DWMWCP_ROUND = 2


def apply_backdrop(hwnd: int) -> str:
    """Ask DWM for Acrylic. Returns ``acrylic`` or ``plain``."""
    if sys.platform != "win32" or not hwnd:
        return "plain"
    try:
        dwm = ctypes.windll.dwmapi

        def _set(attr: int, value: int) -> int:
            packed = ctypes.c_int(value)
            return dwm.DwmSetWindowAttribute(wintypes.HWND(hwnd), ctypes.c_uint(attr), ctypes.byref(packed), ctypes.sizeof(packed))

        _set(DWMWA_USE_IMMERSIVE_DARK_MODE, 1)
        _set(DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND)
        hr = _set(DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_TRANSIENTWINDOW)
    except (AttributeError, OSError):
        return "plain"
    return "acrylic" if hr == 0 else "plain"
