"""Measure the window startup import cost.

Runs ``python -X importtime -c "import canvas.app.main"`` and prints the wall
time plus the 15 modules with the largest cumulative import time. Run it in an
environment with the app dependencies installed:

    python tools/measure_startup.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP_N = 15
IMPORT_TIME = re.compile(r"^import time:\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.+?)\s*$")


def main() -> int:
    started = time.perf_counter()
    done = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import canvas.app.main"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - started
    if done.returncode != 0:
        print(done.stderr[-4000:], file=sys.stderr)
        return done.returncode
    rows: list[tuple[int, int, str]] = []
    for line in done.stderr.splitlines():
        match = IMPORT_TIME.match(line)
        if match:
            rows.append((int(match.group(2)), int(match.group(1)), match.group(3)))
    rows.sort(reverse=True)
    print(f"wall {wall:.3f}s  (python -X importtime -c 'import canvas.app.main')")
    print(f"{'cumulative':>12} {'self':>10}  module")
    for cumulative, self_us, name in rows[:TOP_N]:
        print(f"{cumulative / 1000:>10.1f}ms {self_us / 1000:>9.1f}ms  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
