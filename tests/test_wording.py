"""User-facing text stays on the one responsibility line."""
from pathlib import Path

from canvas.app.info import RESPONSIBILITY

SUPPORT = """## ☕ Support the Project

If you find this project useful and would like to support its development, you can buy me a coffee!

- **Network:** TRON (TRC-20)
- **Accepted:** USDT or TRX
- **Address:** `TYveZBXaSpM4FEHa6iGrc7A6zuLfS4z3x6`

> ⚠️ **Important:** Please ensure you are sending funds **only** via the TRON (TRC-20) network. 
> Sending assets from other networks (such as Ethereum ERC-20, BSC BEP-20, etc.) to this address will result in **permanent loss of funds**.
"""

BANNED = (
    "nsfw",
    "uncensored",
    "nude",
    "minor",
    "classifier",
    "screenedinput",
    "safeimage",
    "qwen3guard",
    "siglip",
    "fairface",
    "yunet",
    "nudenet",
    "falconsai",
)


def _user_text() -> list[tuple[str, str]]:
    files = [Path("README.md"), *Path("docs").rglob("*.md"), *Path("canvas/ui").rglob("*.py")]
    return [(path.as_posix(), path.read_text(encoding="utf-8")) for path in files]


def test_support_section_and_responsibility_line():
    readme = Path("docs/README-full.md").read_text(encoding="utf-8")
    assert SUPPORT in readme
    assert RESPONSIBILITY in readme
    assert RESPONSIBILITY in Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    assert not Path("docs/SAFETY.md").exists()


def test_user_facing_files_do_not_describe_the_checks():
    for name, text in _user_text():
        folded = text.casefold()
        for word in BANNED:
            assert word not in folded, (name, word)
    architecture = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8").casefold()
    assert "content safety checks" in architecture
