"""Windows taskbar identity and the window icon (suite installer standard)."""
from __future__ import annotations

import sys
from pathlib import Path

APP_USER_MODEL_ID = "VOXPrint.AISuite.Canvas"
ICON_FILE = "voxprint-canvas.ico"


def icon_path(app_dir: Path | None = None) -> Path:
    """``<app dir>/assets/voxprint-canvas.ico``. The app dir is the install folder or the repository root."""
    root = app_dir if app_dir is not None else Path(__file__).resolve().parents[2]
    return root / "assets" / ICON_FILE


def set_app_user_model_id(ident: str = APP_USER_MODEL_ID) -> bool:
    """Group the window under the Start menu shortcut, not under Python. Windows only; False elsewhere."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        shell32 = getattr(ctypes, "windll").shell32
        return int(shell32.SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(ident))) == 0
    except (AttributeError, OSError):
        return False
