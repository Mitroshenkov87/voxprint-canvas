"""Write ``docs/cli.md`` from ``voxprint-canvas --help``. Do not hand-edit that page."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "docs" / "cli.md"


def help_text() -> str:
    """Stdout of ``python -m canvas.app.main --help``."""
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [sys.executable, "-m", "canvas.app.main", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    text = proc.stdout.strip()
    if proc.returncode != 0 or "--version" not in text or "usage:" not in text.casefold():
        raise RuntimeError(proc.stderr.strip() or text or "help text did not come from the app")
    return text


def generate_help() -> str:
    """Stdout of ``python -m canvas.app.main generate --help``."""
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [sys.executable, "-m", "canvas.app.main", "generate", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    text = proc.stdout.strip()
    if proc.returncode != 0 or "--enhance" not in text:
        raise RuntimeError(proc.stderr.strip() or text or "generate help did not come from the app")
    return text


def render(text: str, generate: str = "") -> str:
    """The command-line page. The fenced block is the captured help."""
    extra = ""
    if generate.strip():
        extra = (
            "\n## generate\n\n"
            "Generated from `voxprint-canvas generate --help` when the docs are built.\n\n"
            "```text\n"
            f"{generate.strip()}\n"
            "```\n"
        )
    return (
        "# Command line\n\n"
        "Generated from `voxprint-canvas --help` when the docs are built.\n\n"
        "```text\n"
        f"{text}\n"
        "```\n"
        f"{extra}"
    )


def main() -> int:
    """Capture help and write ``docs/cli.md``."""
    DEST.write_text(render(help_text(), generate_help()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
