"""Minor-term and sexual-term lists for English, Russian and German.

Matching is deterministic and runs even when a classifier says the prompt is safe.
Obfuscation covered here: leetspeak, inserted separators, Latin/Cyrillic confusables,
German umlaut folding, and Russian-to-Latin transliteration. The lists are source
constants. They are not loaded from settings.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
_LETTERS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "ё": "е"})
_LEET = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
    "@": "a", "$": "s", "!": "i", "|": "i",
})
# Cyrillic letters that are used to disguise Latin words. Applied only to Latin-majority tokens.
_CONFUSABLE = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ї": "i",
})
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n",
    "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y",
    "ь": "", "э": "e", "ю": "yu", "я": "ya",
}

# mode "exact": the token is one of the needles. mode "prefix": token equals or starts with the needle.
# mode "stem": the needle occurs inside a token (needles are long enough to stay distinctive).
_MINOR: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("exact", "en", ("kid", "kids", "minor", "minors", "csam", "jailbait", "baby", "babies")),
    ("prefix", "en", ("child", "underage", "preteen", "toddler", "infant", "newborn", "loli", "shota",
                      "lolicon", "shotacon", "schoolgirl", "schoolboy", "pedo", "paedo", "paedophile")),
    ("exact", "en", ("teen", "teens", "teenager", "teenagers", "teenage")),
    ("stem", "ru", ("ребенок", "ребенк", "малолет", "несовершеннолет", "подрост", "школьниц", "школьник",
                    "младен", "педофил", "малыш")),
    ("stem", "ru-lat", ("rebenok", "rebyonok", "malolet", "nesovershennoletn", "podrostok", "shkolnits",
                        "shkolnik", "mladen", "pedofil", "malysh")),
    ("exact", "de", ("kind", "kinder", "kindes", "kindern", "kinds")),
    ("stem", "de", ("minderjaehrig", "kleinkind", "saeugling", "schulmaedchen", "schuljunge",
                    "vorpubertaer", "unmuendig", "kindesmissbrauch", "kinderporn", "paedophil")),
)
_SEXUAL: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("exact", "en", ("sex", "sexy", "nude", "naked", "porn", "porno", "nsfw", "erotic", "erotica",
                     "explicit", "lewd", "hentai", "topless", "bottomless", "lingerie", "xxx")),
    ("prefix", "en", ("sexual", "intercourse")),
    ("stem", "ru", ("секс", "порно", "эротич", "обнаж", "голая", "голый", "голые", "интим")),
    ("stem", "ru-lat", ("seks", "porno", "erotich")),
    ("stem", "de", ("nackt", "porn", "erotisch", "explizit", "geschlechtsverkehr", "dessous")),
    ("exact", "de", ("sex",)),
)

_AGE_DIGIT = re.compile(r"\b(?:1[0-7]|[1-9])\s*(?:years?\s*old|year\s*old|yo|jahre?\s*alt|jaehrig|лет|года)\b")
_AGE_COMPACT = re.compile(r"\b(?:1[0-7]|[1-9])yo\b")
_UNDER = re.compile(r"\b(?:under|unter|младше)\s*(?:18|1[0-7]|[1-9])\b")
_WORD_AGE = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eins", "zwei", "drei", "vier", "fuenf", "sechs", "sieben", "acht", "neun", "zehn", "elf", "zwoelf",
    "dreizehn", "vierzehn", "fuenfzehn", "sechzehn", "siebzehn",
)
_WORD_AGE_RE = re.compile(
    r"\b(?:" + "|".join(_WORD_AGE) + r")\s*(?:years?\s*old|year\s*old|jahre?\s*alt|jaehrig)\b"
)


@dataclass(frozen=True)
class TextSignals:
    """Whether the prompt names a minor and whether it is sexual or suggestive."""

    minor: bool
    sexual: bool


def _cyrillic_count(token: str) -> int:
    return sum(1 for ch in token if "а" <= ch <= "я")


def _translit(token: str) -> str:
    return "".join(_TRANSLIT.get(ch, ch) for ch in token)


def _glue(tokens: list[str]) -> list[str]:
    """Join runs of single letters so 'c h i l d' and spaced Cyrillic match a word."""
    found: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= 4:
            word = "".join(run)
            found.append(word)
            if _cyrillic_count(word):
                found.append(_translit(word))
        run.clear()

    for token in tokens:
        if len(token) == 1:
            run.append(token)
        else:
            flush()
    flush()
    return found


def _base(text: str) -> str:
    """NFKC, lower case, umlaut folding. Digits stay digits so ages still match."""
    return unicodedata.normalize("NFKC", text).translate(_ZW).lower().translate(_LETTERS)


def tokenise(text: str) -> list[str]:
    """Normalised tokens plus glued spellings and Russian transliterations."""
    folded = _base(text).translate(_LEET)
    raw = re.findall(r"[0-9a-zа-я]+", folded)
    primary: list[str] = []
    extra: list[str] = []
    for token in raw:
        if _cyrillic_count(token) * 2 >= len(token) and _cyrillic_count(token):
            primary.append(token)
            extra.append(_translit(token))
        else:
            primary.append(token.translate(_CONFUSABLE))
    return primary + _glue(primary) + extra


def _matches(tokens: list[str], mode: str, needles: tuple[str, ...]) -> bool:
    if mode == "exact":
        wanted = set(needles)
        return any(token in wanted for token in tokens)
    if mode == "prefix":
        return any(token.startswith(needle) for token in tokens for needle in needles)
    if mode == "stem":
        return any(needle in token for token in tokens for needle in needles)
    raise RuntimeError("unknown term mode")


def _group_hit(tokens: list[str], groups: tuple[tuple[str, str, tuple[str, ...]], ...]) -> bool:
    return any(_matches(tokens, mode, needles) for mode, _lang, needles in groups)


def _age_hit(text: str) -> bool:
    spaced = " ".join(re.findall(r"[0-9a-zа-я]+", _base(text)))
    return bool(_AGE_DIGIT.search(spaced) or _AGE_COMPACT.search(spaced) or _UNDER.search(spaced) or _WORD_AGE_RE.search(spaced))


def scan(text: str) -> TextSignals:
    """Return minor and sexual signals. A minor signal alone does not block."""
    tokens = tokenise(text)
    minor = _group_hit(tokens, _MINOR) or _age_hit(text)
    sexual = _group_hit(tokens, _SEXUAL)
    return TextSignals(minor=minor, sexual=sexual)
