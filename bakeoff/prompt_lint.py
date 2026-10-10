"""Reject youth terms and stated ages under 20 in bake-off prompts."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

MIN_STATED_AGE = 20

_DECADE_WORD = {
    "twenties": 20,
    "thirties": 30,
    "forties": 40,
    "fifties": 50,
    "sixties": 60,
    "seventies": 70,
    "eighties": 80,
    "nineties": 90,
}

# Numeric ages. Ordinals stop at 19 so "40th" on a sign is not treated as an age.
_AGE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(\d{1,3})\s*-\s*years?\s*-\s*old\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s+years?\s+old\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*yo\b", re.IGNORECASE),
    re.compile(r"\b(?:aged|age)\s+(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(?:his|her|their)\s+(\d{2})s\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*-\s*летн", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s+лет\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s+года\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s+год\b", re.IGNORECASE),
    re.compile(r"\b(1[0-9]|[1-9])(?:st|nd|rd|th)\b", re.IGNORECASE),
)
_DECADE_RE = re.compile(
    r"\b(twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b",
    re.IGNORECASE,
)

_EN_EXACT = {
    "girl", "girls", "boy", "boys", "petite", "student", "students",
    "kid", "kids", "minor", "minors", "youth", "youths",
    "juvenile", "juveniles", "adolescent", "adolescents",
    "toddler", "toddlers", "infant", "infants", "newborn", "newborns",
    "baby", "babies", "preteen", "preteens", "jailbait",
    "loli", "lolicon", "shota", "shotacon",
    "schoolgirl", "schoolgirls", "schoolboy", "schoolboys",
    "devochka", "devushka", "malchik", "podrostok", "shkolnitsa", "shkolnik",
    "rebenok", "yunyi", "yunaya",
}
_EN_PREFIX = ("teen", "young", "child", "underage", "schoolgirl", "schoolboy")
_RU_EXACT = {
    "юный", "юная", "юное", "юные", "юного", "юной", "юную", "юным", "юных", "юными",
    "молодой", "молодая", "молодое", "молодые", "молодого", "молодую", "молодым",
    "молодых", "молодыми", "молодость", "молодости",
    "дети", "детей", "детям", "детьми", "детях",
}
_RU_PREFIX = (
    "девоч", "мальчик", "мальчиш", "подрост", "школьн", "студент", "ученик", "учениц",
    "девуш", "юнош", "малолет", "несовершеннолет", "малыш", "младен", "тинейдж",
    "парен", "ребен",
)


def _age_fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower().replace("ё", "е")


def _term_fold(text: str) -> str:
    folded = _age_fold(text).replace("-", " ").replace("_", " ")
    return re.sub(r"[^\w\s]+", " ", folded, flags=re.UNICODE)


def stated_ages(text: str) -> list[int]:
    """Stated ages in reading order. Values above 120 are ignored."""
    folded = _age_fold(text)
    found: list[tuple[int, int]] = []
    for pattern in _AGE_RES:
        for match in pattern.finditer(folded):
            age = int(match.group(1))
            if 1 <= age <= 120:
                found.append((match.start(), age))
    for match in _DECADE_RE.finditer(folded):
        found.append((match.start(), _DECADE_WORD[match.group(1).lower()]))
    found.sort()
    ages: list[int] = []
    seen_at: set[int] = set()
    for start, age in found:
        if start in seen_at:
            continue
        seen_at.add(start)
        ages.append(age)
    return ages


def youth_terms(text: str) -> list[str]:
    """Age-ambiguous and youth tokens in ``text``, in token order."""
    tokens = re.findall(r"[0-9a-zа-я]+", _term_fold(text))
    hits: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        term = _youth_token(token)
        if term and term not in seen:
            seen.add(term)
            hits.append(term)
    return hits


def _youth_token(token: str) -> str:
    if token in _EN_EXACT or token in _RU_EXACT:
        return token
    for prefix in _EN_PREFIX:
        if token.startswith(prefix):
            return token
    for prefix in _RU_PREFIX:
        if token.startswith(prefix):
            return token
    return ""


def lint_text(text: str) -> list[str]:
    """Reasons ``text`` cannot be a renderable prompt. Empty means it passes."""
    reasons: list[str] = []
    for age in stated_ages(text):
        if age < MIN_STATED_AGE:
            reasons.append(f"stated age {age} is under {MIN_STATED_AGE}")
    for term in youth_terms(text):
        reasons.append(f"youth term {term}")
    return reasons


def lint_jsonl(path: Path) -> list[str]:
    """Lint one prompt file. Each hit is ``id: reason``."""
    hits: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row["prompt"]
        if row.get("target"):
            text = f"{text}\n{row['target']}"
        for reason in lint_text(text):
            hits.append(f"{row['id']}: {reason}")
    return hits
