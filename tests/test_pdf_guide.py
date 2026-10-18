"""The user-guide sources agree, and a built PDF contains its chapters."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

GUIDE = Path("docs/guide")
LANGUAGES = ("en", "ru")
_HEADING = re.compile(r"^(#{1,3}) (.+)$")
_CHAPTER = re.compile(r"^# (.+)$", re.MULTILINE)
_IMAGE = re.compile(r"screens/([a-z0-9-]+)\.png")


def _text(language: str) -> str:
    return (GUIDE / f"{language}.md").read_text(encoding="utf-8")


def _headings(language: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for line in _text(language).splitlines():
        match = _HEADING.match(line)
        if match is not None:
            found.append((match.group(1), match.group(2).strip()))
    return found


def test_guides_have_no_todo_markers() -> None:
    for language in LANGUAGES:
        assert "TODO(check)" not in _text(language)


def test_guides_share_structure() -> None:
    english = _headings("en")
    russian = _headings("ru")
    assert len(english) == len(russian)
    assert [level for level, _title in english] == [level for level, _title in russian]
    assert _IMAGE.findall(_text("en")) == _IMAGE.findall(_text("ru"))
    assert len(_CHAPTER.findall(_text("en"))) == len(_CHAPTER.findall(_text("ru")))
    assert len(english) > 0


def test_built_pdfs_contain_every_chapter() -> None:
    raw = os.environ.get("VOXPRINT_PDF_GUIDE_DIR", "")
    if not raw:
        pytest.skip("PDF guide is built in the pdf-guide job")
    from pypdf import PdfReader

    folder = Path(raw)
    counts: dict[str, int] = {}
    for language in LANGUAGES:
        path = folder / f"VOXPrint-AI-Canvas-Guide-{language}.pdf"
        assert path.is_file(), path
        assert path.stat().st_size > 100 * 1024
        reader = PdfReader(str(path))
        counts[language] = len(reader.pages)
        assert counts[language] > 1
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        collapsed = re.sub(r"\s+", " ", text)
        assert "TODO(check)" not in collapsed
        for title in _CHAPTER.findall(_text(language)):
            assert title in collapsed, title
    assert counts["en"] > 0
    assert counts["ru"] > 0
