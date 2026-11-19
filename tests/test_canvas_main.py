"""``python -m canvas`` launches the same entry as main.py."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from canvas.app.identity import splash_line

ROOT = Path(__file__).resolve().parents[1]


def test_python_module_canvas_prints_the_version() -> None:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.pop("PYTHONPATH", None)
    done = subprocess.run(
        [sys.executable, "-m", "canvas", "--version"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == splash_line()
