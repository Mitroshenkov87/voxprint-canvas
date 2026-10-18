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
    ("exact", "en", ("kid", "kids", "minor", "minors", "baby", "babies")),
    ("prefix", "en", ("child", "underage", "preteen", "toddler", "infant", "newborn", "schoolgirl", "schoolboy")),
    ("exact", "en", ("teen", "teens", "teenager", "teenagers", "teenage")),
    ("stem", "ru", ("ребенок", "ребенк", "дети", "детей", "детям", "детьми", "дитя", "малолет",
                    "несовершеннолет", "подрост", "школьниц", "школьник", "младен", "малыш", "детск",
                    "тинейджер")),
    ("stem", "ru-lat", ("rebenok", "rebyonok", "malolet", "nesovershennoletn", "podrostok", "shkolnits",
                        "shkolnik", "mladen", "malysh", "deti", "tineidzher")),
    # "kind" is checked in _german_kind: in an English prompt it is "a kind woman".
    # "kinds" is left out ("all kinds of").
    ("exact", "de", ("kinder", "kindes", "kindern")),
    ("stem", "de", ("minderjaehrig", "kleinkind", "saeugling", "schulmaedchen", "schuljunge",
                    "vorpubertaer", "unmuendig")),
)
# These name sexual content involving a minor. They block on their own.
_MINOR_SEXUAL: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("exact", "en", ("loli", "lolis", "lolita", "shota", "shotas", "csam", "jailbait", "pedo", "paedo")),
    ("prefix", "en", ("lolicon", "shotacon", "pedophile", "paedophile", "pedophil")),
    ("stem", "ru", ("лоли", "шота", "лоликон", "шотакон", "педофил")),
    ("stem", "ru-lat", ("loli", "shota", "lolicon", "shotacon", "pedofil")),
    ("stem", "de", ("kindesmissbrauch", "kinderporn", "paedophil")),
)
# School and student context. Alone, or in a clothed scene, these pass. Together
# with explicit or nude content they block (product rule 2026-10-18).
_YOUTH_CONTEXT: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("exact", "en", ("school", "schools", "schoolkid", "schoolkids", "highschool", "student", "students",
                     "pupil", "pupils", "classroom", "classrooms", "schoolyard",
                     "juvenile", "juveniles", "adolescent", "adolescents")),
    ("prefix", "en", ("schooluniform", "barelylegal")),
    ("stem", "ru", ("школ", "школьн", "студент", "ученик", "учениц", "гимназист", "старшеклас",
                    "первоклас", "одноклассниц", "одноклассник")),
    ("stem", "ru-lat", ("shkol", "student", "uchenik", "uchenits", "starsheklas")),
    ("exact", "de", ("schule", "schueler", "schuelerin", "schuluniform", "klassenzimmer", "student", "studentin")),
)
# Anime and other stylised characters. With explicit or nude content these block,
# even when an adult age is stated (product rule 2026-10-18). Alone they pass.
_STYLISED: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("exact", "en", ("waifu", "waifus", "chibi", "moe", "kawaii", "anime", "manga", "hentai")),
    ("prefix", "en", ("animegirl", "animeboy", "animestyle")),
    ("stem", "ru", ("вайфу", "чиби", "аниме", "манга", "хентай", "кавай")),
    ("exact", "ru", ("сета", "моэ")),
    ("stem", "ru-lat", ("vaifu", "chibi", "anime", "hentai")),
)
_SEXUAL: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("exact", "en", ("sex", "sexy", "nude", "naked", "porn", "porno", "nsfw", "erotic", "erotica",
                     "explicit", "lewd", "hentai", "topless", "bottomless", "lingerie", "xxx")),
    ("prefix", "en", ("sexual", "intercourse")),
    ("stem", "ru", ("секс", "порно", "эротич", "эротик", "обнаж", "голая", "голый", "голые", "интим", "хентай")),
    ("stem", "ru-lat", ("seks", "porno", "erotich", "erotik")),
    ("stem", "de", ("nackt", "porn", "erotisch", "explizit", "geschlechtsverkehr", "dessous")),
    ("exact", "de", ("sex",)),
)

# The number is a whole token under 18. "30" and "85" do not match, and a digit
# inside a longer number does not match.
_AGE_NUMBER = r"(?<!\d)\b(?:1[0-7]|[1-9])(?!\d)"
_AGE_DIGIT = re.compile(_AGE_NUMBER + r"\s*(?:years?\s*old|year\s*old|yo|jahre?\s*alt|jaehrig|лет|года)\b")
_AGE_COMPACT = re.compile(_AGE_NUMBER + r"yo\b")
_UNDER = re.compile(r"\b(?:under|unter|младше)\s*(?:18|1[0-7]|[1-9])(?!\d)\b")
_WORD_AGE = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eins", "zwei", "drei", "vier", "fuenf", "sechs", "sieben", "acht", "neun", "zehn", "elf", "zwoelf",
    "dreizehn", "vierzehn", "fuenfzehn", "sechzehn", "siebzehn",
)
_WORD_AGE_RE = re.compile(
    r"\b(?:" + "|".join(_WORD_AGE) + r")\s*(?:years?\s*old|year\s*old|jahre?\s*alt|jaehrig)\b"
)
_RU_WORD_AGE = (
    "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
    "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать",
)
_RU_WORD_AGE_RE = re.compile(r"\b(?:" + "|".join(_RU_WORD_AGE) + r")\s*(?:лет|года|год)\b")
_DECLARED_AGE = re.compile(
    r"\b(?:age|aged|возраст)\s*(?:1[0-7]|[1-9])(?!\d)\b|" + _AGE_NUMBER + r"\s*летн"
)


@dataclass(frozen=True)
class TextSignals:
    """Whether the prompt names a minor, sexual content, or sexual content about a minor."""

    minor: bool
    sexual: bool
    minor_sexual: bool = False
    youth_context: bool = False
    stylised: bool = False


@dataclass(frozen=True)
class ScanTrace:
    """Lexicon hits for one prompt. ``age`` is an explicit age under 18."""

    minor: bool
    sexual: bool
    minor_sexual: bool
    age: bool
    minor_terms: tuple[str, ...]
    sexual_terms: tuple[str, ...]
    minor_sexual_terms: tuple[str, ...]
    youth_context_terms: tuple[str, ...] = ()
    stylised_terms: tuple[str, ...] = ()


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


# Adult phrases that contain a listed word. Only these exact neighbours exempt it;
# the word anywhere else still counts.
_NEXT_EXEMPT = {
    "baby": frozenset({"oil", "doll", "blue", "pink"}),
    "minor": frozenset({"key", "detail", "details", "flaw", "flaws"}),
    "school": frozenset({"of"}),
}
_PREV_EXEMPT = {"school": frozenset({"old", "art", "film"})}
_NEVER_EXEMPT_BEFORE = {"school": frozenset({"uniform", "uniforms", "girl", "girls", "boy", "boys"})}


def _drop_exempt(raw: list[str]) -> list[str]:
    """Blank a listed word inside one of the adult phrases above ('baby oil', 'old school')."""
    kept = list(raw)
    for index, token in enumerate(raw):
        before = raw[index - 1] if index > 0 else ""
        after = raw[index + 1] if index + 1 < len(raw) else ""
        if after in _NEVER_EXEMPT_BEFORE.get(token, frozenset()):
            continue
        if after in _NEXT_EXEMPT.get(token, frozenset()) or before in _PREV_EXEMPT.get(token, frozenset()):
            kept[index] = ""
    return kept


# Two-word phrases matched as one token ("barely legal" -> "barelylegal").
_PHRASES = frozenset({("barely", "legal")})


def _phrases(tokens: list[str]) -> list[str]:
    return [a + b for a, b in zip(tokens, tokens[1:]) if (a, b) in _PHRASES]


def tokenise(text: str) -> list[str]:
    """Normalised tokens plus glued spellings and Russian transliterations."""
    folded = _base(text).translate(_LEET)
    raw = _drop_exempt(re.findall(r"[0-9a-zа-я]+", folded))
    primary: list[str] = []
    extra: list[str] = []
    for token in raw:
        if _cyrillic_count(token) * 2 >= len(token) and _cyrillic_count(token):
            primary.append(token)
            extra.append(_translit(token))
        else:
            primary.append(token.translate(_CONFUSABLE))
    return primary + _glue(primary) + _phrases(primary) + extra


# A stem is a prefix of an inflected form, not a substring. The remainder is empty,
# starts with a vowel, or is one of these endings. "секс" does not match "секстант".
_INFLECTION = frozenset("аеёиоуыэюяьъaeiouy")
_ENDINGS = (
    "граф", "ный", "ная", "ное", "ные", "ного", "ному", "ным", "ными", "ных",
    "ний", "няя", "нее", "ние", "него", "нему", "ним", "ними", "них",
    "ский", "ская", "ское", "ские", "ского", "скому", "ским", "скими", "ских",
    "ка", "ки", "ку", "кой", "ке", "ок",
)


def _stem_match(token: str, needle: str) -> bool:
    """True when ``needle`` is the token or its morphological stem."""
    if not token.startswith(needle):
        return False
    rest = token[len(needle):]
    if rest == "" or rest[0] in _INFLECTION:
        return True
    return any(rest.startswith(ending) for ending in _ENDINGS)


def _needle_hit(tokens: list[str], mode: str, needle: str) -> bool:
    if mode == "exact":
        return any(token == needle for token in tokens)
    if mode == "prefix":
        return any(token.startswith(needle) for token in tokens)
    if mode == "stem":
        return any(_stem_match(token, needle) for token in tokens)
    raise RuntimeError("unknown term mode")


def _hits(tokens: list[str], groups: tuple[tuple[str, str, tuple[str, ...]], ...]) -> tuple[str, ...]:
    found: list[str] = []
    for mode, _lang, needles in groups:
        for needle in needles:
            if _needle_hit(tokens, mode, needle):
                found.append(needle)
    return tuple(found)


def _age_hit(text: str) -> bool:
    spaced = " ".join(re.findall(r"[0-9a-zа-я]+", _base(text)))
    return bool(
        _AGE_DIGIT.search(spaced)
        or _AGE_COMPACT.search(spaced)
        or _UNDER.search(spaced)
        or _WORD_AGE_RE.search(spaced)
        or _RU_WORD_AGE_RE.search(spaced)
        or _DECLARED_AGE.search(spaced)
    )


# German words that mark a prompt as German, so a bare "kind" there means a child.
_GERMAN_MARKERS = frozenset({
    "ein", "eine", "einen", "einem", "einer", "eines", "das", "der", "dem", "den", "des", "und", "mit",
    "ist", "im", "beim", "zum", "zur", "fuer", "kleines", "kleine", "kleiner", "kleinen", "junges", "junge",
})
_GERMAN_SEXUAL_STEMS = ("nackt", "erotisch", "explizit", "dessous", "geschlechtsverkehr")


def _german_kind(tokens: list[str]) -> bool:
    """True for "kind" in a German prompt ("ein nacktes Kind"), not in English ("a kind woman")."""
    if "kind" not in tokens:
        return False
    if any(token in _GERMAN_MARKERS for token in tokens):
        return True
    return any(_stem_match(token, stem) for token in tokens for stem in _GERMAN_SEXUAL_STEMS)


def explain(text: str) -> ScanTrace:
    """Terms and the age flag that ``scan`` used. This is the lexicon half of a refusal trace."""
    tokens = tokenise(text)
    minor_terms = _hits(tokens, _MINOR) + (("kind",) if _german_kind(tokens) else ())
    sexual_terms = _hits(tokens, _SEXUAL)
    minor_sexual_terms = _hits(tokens, _MINOR_SEXUAL)
    youth_context_terms = _hits(tokens, _YOUTH_CONTEXT)
    stylised_terms = _hits(tokens, _STYLISED)
    age = _age_hit(text)
    minor_sexual = bool(minor_sexual_terms)
    return ScanTrace(
        minor=minor_sexual or bool(minor_terms) or age,
        sexual=bool(sexual_terms),
        minor_sexual=minor_sexual,
        age=age,
        minor_terms=minor_terms,
        sexual_terms=sexual_terms,
        minor_sexual_terms=minor_sexual_terms,
        youth_context_terms=youth_context_terms,
        stylised_terms=stylised_terms,
    )


def scan(text: str) -> TextSignals:
    """Return minor and sexual signals. A minor signal alone does not block."""
    detail = explain(text)
    return TextSignals(
        minor=detail.minor,
        sexual=detail.sexual,
        minor_sexual=detail.minor_sexual,
        youth_context=bool(detail.youth_context_terms),
        stylised=bool(detail.stylised_terms),
    )
