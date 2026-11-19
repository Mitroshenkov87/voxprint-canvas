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


def _help(command: str, marker: str) -> str:
    """Stdout of ``python -m canvas.app.main <command> --help``."""
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [sys.executable, "-m", "canvas.app.main", command, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    text = proc.stdout.strip()
    if proc.returncode != 0 or marker not in text:
        raise RuntimeError(proc.stderr.strip() or text or f"{command} help did not come from the app")
    return text


def generate_help() -> str:
    """Stdout of ``python -m canvas.app.main generate --help``."""
    return _help("generate", "--enhance")


def photoshoot_help() -> str:
    """Stdout of ``python -m canvas.app.main photoshoot --help``."""
    return _help("photoshoot", "--frames")


def _section(name: str, body: str) -> str:
    if not body.strip():
        return ""
    return (
        f"\n## {name}\n\n"
        f"Generated from `voxprint-canvas {name} --help` when the docs are built.\n\n"
        "```text\n"
        f"{body.strip()}\n"
        "```\n"
    )


def render(text: str, generate: str = "", photoshoot: str = "") -> str:
    """The command-line page. The fenced block is the captured help."""
    extra = _section("generate", generate) + _section("photoshoot", photoshoot)
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
    DEST.write_text(render(help_text(), generate_help(), photoshoot_help()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
