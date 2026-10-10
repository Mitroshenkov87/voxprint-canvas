"""The published docs are generated, and the check page stays on the public entry points."""
from pathlib import Path

from canvas.safety.gate import AgeGate, GenerationBlocked, SafetyError, SafetyStack, SafetyUnavailable
from tools.audit import INTERROGATE_MIN
from tools.render_cli_docs import help_text, render

BANNED = ("nsfw", "uncensored", "nude", "nudenet", "classifier")


def test_cli_page_is_captured_from_help(tmp_path: Path):
    captured = help_text()
    page = render(captured)
    assert "Generated from `voxprint-canvas --help`" in page
    assert "--version" in page
    assert "--selftest" in page
    assert "--register-models" in page
    folded = page.casefold()
    for word in BANNED:
        assert word not in folded
    dest = tmp_path / "cli.md"
    dest.write_text(page, encoding="utf-8")
    assert dest.read_text(encoding="utf-8") == page


def test_safety_reference_lists_entry_points_only():
    page = Path("docs/api/safety.md").read_text(encoding="utf-8")
    for name in ("AgeGate", "SafetyError", "GenerationBlocked", "SafetyUnavailable", "SafetyStack"):
        assert name in page
    folded = page.casefold()
    for word in ("denylist", "classifiers", "adapters", "nsfw", "uncensored", "nudenet", "qwen3guard", "siglip"):
        assert word not in folded
    assert AgeGate.__doc__ == "Decides whether a request may continue. There is no off switch."
    assert SafetyError.__doc__ == "The request stopped. The message is safe to show."
    assert GenerationBlocked.__doc__ == "The request cannot be completed."
    assert SafetyUnavailable.__doc__ == "A check could not finish, so the request stops."
    assert SafetyStack.__doc__ == "What the gate needs before it can run."


def test_interrogate_gate_is_65_percent():
    assert INTERROGATE_MIN == 65.0
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "fail-under = 65" in text
    workflow = Path(".github/workflows/docs.yml").read_text(encoding="utf-8")
    assert "actions/deploy-pages@v4" in workflow
    assert "actions/upload-pages-artifact@v3" in workflow
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "https://mitroshenkov87.github.io/voxprint-canvas/" in readme
