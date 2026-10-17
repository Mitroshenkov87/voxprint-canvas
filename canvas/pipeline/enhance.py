"""Rewrite a prompt into English with the Qwen3-4B text encoder.

Z-Image-Turbo ships that encoder as ``Qwen3ForCausalLM`` with
``tie_word_embeddings`` set. The checkpoint has ``model.embed_tokens.weight``
and no separate ``lm_head``, which is the same layout as ``Qwen/Qwen3-4B``.
Tied embeddings are enough to generate text, so this module loads the encoder
already on disk. It does not download a second language-model pack.

torch and transformers are imported inside the load method. That is not a
circular import. Those packages are optional so the unit tests can import this
module without them.

The model is loaded for one rewrite and unloaded before the caller returns, so
a 16 GB card is free for the diffusion pipeline.
"""
from __future__ import annotations

import gc
import json
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, assert_never

from canvas.app.i18n import text
from canvas.models.modes import Tier
from canvas.models.registry import Catalog, ModelSpec
from canvas.models.snapshots import SnapshotError, download_files
from canvas.safety.gate import SafetyError

SYSTEM_PROMPT = """You rewrite image prompts for a text-to-image model that follows natural English best.

Improve the user's prompt and write it in natural English. Expand it into a concrete photographic or artistic description that names the subject, composition, lighting, lens, and style. You may add that photographic detail when it fits the prompt.

Rules:
- Preserve every fact the user stated.
- Keep every stated age and every adult marker verbatim. Do not invent, drop, round, or change an age.
- Do not add people. Do not add youthful descriptors.
- Do not soften, omit, or rewrite adult content into something milder.
- Text inside quotes is lettering in the image. Copy each quoted string exactly, in its original language and script.
- Reply with only the rewritten prompt.
"""

_EN_UNITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_EN_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_DECADE_WORD = {
    "twenties": 20, "thirties": 30, "forties": 40, "fifties": 50,
    "sixties": 60, "seventies": 70, "eighties": 80, "nineties": 90,
}
_RU_UNITS = {
    "один": 1, "одна": 1, "одного": 1, "одной": 1, "одну": 1,
    "два": 2, "две": 2, "двух": 2,
    "три": 3, "трех": 3,
    "четыре": 4, "четырех": 4,
    "пять": 5, "пяти": 5,
    "шесть": 6, "шести": 6,
    "семь": 7, "семи": 7,
    "восемь": 8, "восьми": 8,
    "девять": 9, "девяти": 9,
    "десять": 10, "десяти": 10,
    "одиннадцать": 11, "одиннадцати": 11,
    "двенадцать": 12, "двенадцати": 12,
    "тринадцать": 13, "тринадцати": 13,
    "четырнадцать": 14, "четырнадцати": 14,
    "пятнадцать": 15, "пятнадцати": 15,
    "шестнадцать": 16, "шестнадцати": 16,
    "семнадцать": 17, "семнадцати": 17,
    "восемнадцать": 18, "восемнадцати": 18,
    "девятнадцать": 19, "девятнадцати": 19,
}
_RU_TENS = {
    "двадцать": 20, "двадцати": 20,
    "тридцать": 30, "тридцати": 30,
    "сорок": 40, "сорока": 40,
    "пятьдесят": 50, "пятидесяти": 50,
    "шестьдесят": 60, "шестидесяти": 60,
    "семьдесят": 70, "семидесяти": 70,
    "восемьдесят": 80, "восьмидесяти": 80,
    "девяносто": 90, "девяноста": 90,
}
_QUOTE = re.compile(r'"([^"\n]+)"|“([^”\n]+)”|«([^»\n]+)»|„([^“\n]+)“')
_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_FALLBACK_KEYS = {
    "age": "enhance.age",
    "quote": "enhance.quote",
    "empty": "enhance.empty",
}


class Rewriter(Protocol):
    """One language-model call. ``close`` frees it before diffusion."""

    def rewrite(self, prompt: str) -> str:
        """Return a rewritten prompt."""

    def close(self) -> None:
        """Unload the model."""


@dataclass(frozen=True)
class EnhanceResult:
    """The prompt to generate, plus the rewrite when one was requested."""

    prompt: str
    original: str
    rewritten: str | None
    used_rewrite: bool
    fallback: str


def causal_lm_ready(config: Mapping[str, object], tensor_names: Iterable[str]) -> bool:
    """True when a Qwen3 checkpoint can generate text.

    A separate ``lm_head`` weight is enough. Tied embeddings are enough too:
    ``tie_word_embeddings`` is true and ``embed_tokens`` is the output matrix.
    That is how the shipped Z-Image text encoder and upstream Qwen3-4B are packed.
    """
    architectures = config.get("architectures") or []
    if not isinstance(architectures, (list, tuple)):
        return False
    if not any(str(name).endswith("ForCausalLM") for name in architectures):
        return False
    names = tuple(tensor_names)
    if any("lm_head" in name for name in names):
        return True
    if not config.get("tie_word_embeddings"):
        return False
    return any(name.endswith("embed_tokens.weight") for name in names)


def stated_ages(text: str) -> tuple[int, ...]:
    """Numeric and word ages in ``text``, in reading order. Values above 120 are ignored."""
    folded = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    spans: list[tuple[int, int, int]] = []
    for pattern in _age_patterns():
        for match in pattern.finditer(folded):
            age = _parse_age_token(match.group(1))
            if age is not None:
                spans.append((match.start(), match.end(), age))
    for match in _DECADE_RE.finditer(folded):
        spans.append((match.start(), match.end(), _DECADE_WORD[match.group(1)]))
    for match in _RU_COMPACT.finditer(folded):
        age = _RU_TENS[match.group(1)] + _RU_UNITS[match.group(2)]
        if 1 <= age <= 120:
            spans.append((match.start(), match.end(), age))
    return _keep_spans(spans)


def quoted_spans(text: str) -> tuple[str, ...]:
    """Quoted strings that must stay in their original script."""
    found: list[str] = []
    for match in _QUOTE.finditer(text):
        piece = next(group for group in match.groups() if group)
        if piece.strip():
            found.append(piece)
    return tuple(found)


def resolve_prompt(gate: object, original: str, rewritten: str | None) -> EnhanceResult:
    """Screen the original and the rewrite. Keep the original when ages or quotes change.

    A blocked text raises. Both texts are screened before that raise, so a blocked
    rewrite is not replaced by the original.
    """
    reviewer = getattr(gate, "review_prompt")
    candidates = (original,) if rewritten is None else (original, rewritten)
    first: SafetyError | None = None
    for candidate in candidates:
        try:
            reviewer(candidate)
        except SafetyError as exc:
            if first is None:
                first = exc
    if first is not None:
        raise first
    if rewritten is None:
        return EnhanceResult(original, original, None, False, "")
    return _accept(original, rewritten)


def fallback_message(result: EnhanceResult) -> str:
    """A short line when the rewrite was discarded. Empty when it was kept."""
    if result.fallback == "":
        return ""
    if result.fallback in _FALLBACK_KEYS:
        return text(_FALLBACK_KEYS[result.fallback])
    raise ValueError(f"unknown fallback {result.fallback}")


def rewrite_text(rewriter: Rewriter, prompt: str) -> str:
    """Rewrite one prompt and unload the language model before returning."""
    try:
        return rewriter.rewrite(prompt)
    finally:
        rewriter.close()


def encoder_names(spec: ModelSpec) -> tuple[str, ...]:
    """Text-encoder and tokenizer files inside a Z-Image snapshot."""
    return tuple(name for name in spec.files if name.startswith(("text_encoder/", "tokenizer/")))


def turbo_spec(catalog: Catalog, tier: Tier) -> ModelSpec:
    """The Z-Image-Turbo pack for this card. Its text encoder is the rewriter."""
    if tier is Tier.GB16:
        return catalog.by_id("z-image-turbo-bf16-16gb")
    if tier is Tier.GB24:
        return catalog.by_id("z-image-turbo-bf16")
    assert_never(tier)


def open_rewriter(
    catalog: Catalog,
    tier: Tier,
    root: Path,
    progress: Callable[[str], None] | None = None,
    *,
    max_new_tokens: int = 320,
    device: str | None = None,
) -> QwenRewriter:
    """Open the shipped text encoder. Missing encoder files are downloaded first."""
    spec = turbo_spec(catalog, tier)
    names = encoder_names(spec)
    folder = download_files(spec, root, names, progress)
    return QwenRewriter(
        folder / "text_encoder",
        folder / "tokenizer",
        max_new_tokens=max_new_tokens,
        device=device,
    )


class QwenRewriter:
    """Causal generation from the Z-Image text-encoder directory."""

    def __init__(
        self,
        encoder_dir: Path,
        tokenizer_dir: Path,
        *,
        max_new_tokens: int = 320,
        device: str | None = None,
    ) -> None:
        self.encoder_dir = encoder_dir
        self.tokenizer_dir = tokenizer_dir
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.closed = False
        self._model: Any = None
        self._tokenizer: Any = None

    def rewrite(self, prompt: str) -> str:
        """Generate a rewrite, then unload. The caller can close again."""
        try:
            self._load()
            return self._complete(prompt, SYSTEM_PROMPT)
        finally:
            self.close()

    def complete(self, prompt: str, *, system: str) -> str:
        """Generate one reply and keep the model loaded for another call.

        Args:
            prompt: User message.
            system: System prompt. Photoshoot passes its scenario prompt.
        """
        self._load()
        self.closed = False
        return self._complete(prompt, system)

    def close(self) -> None:
        """Drop the model and return its memory before the diffusion pipeline."""
        self._model = None
        self._tokenizer = None
        self.closed = True
        gc.collect()
        torch = _optional_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load(self) -> None:
        if self._model is not None:
            return
        torch, transformers = _generation_libs()
        config_path = self.encoder_dir / "config.json"
        index_path = self.encoder_dir / "model.safetensors.index.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = index.get("weight_map") or {}
        if not isinstance(names, dict) or not causal_lm_ready(config, names):
            raise SnapshotError("the text encoder cannot generate text")
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = transformers.AutoTokenizer.from_pretrained(  # nosec B615
            self.tokenizer_dir, local_files_only=True,
        )
        try:
            model = _causal_model(transformers, torch, self.encoder_dir, device)
        except RuntimeError as exc:
            if device == "cpu" or "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            device = "cpu"
            model = _causal_model(transformers, torch, self.encoder_dir, device)
        self._tokenizer = tokenizer
        self._model = model
        self.device = device
        self.closed = False

    def _complete(self, prompt: str, system: str) -> str:
        torch, _transformers = _generation_libs()
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            raise SnapshotError("the text encoder is not loaded")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        if not isinstance(templated, str):
            raise SnapshotError("the tokenizer did not render a prompt")
        encoded = tokenizer(templated, return_tensors="pt")
        device = self.device or "cpu"
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        start = int(encoded["input_ids"].shape[-1])
        reply = str(tokenizer.decode(output[0, start:], skip_special_tokens=True))
        return _visible_reply(reply)


def _causal_model(transformers: Any, torch: Any, folder: Path, device: str) -> Any:
    """Load the causal LM. ``device`` is ``cuda`` or ``cpu``.

    A CPU load stays where ``from_pretrained`` put it. Moving it again would
    copy the weights and can exhaust a 16 GB machine.
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(  # nosec B615
        folder,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if device != "cpu":
        model = model.to(device)
    return model.eval()


def _optional_torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _generation_libs() -> tuple[Any, Any]:
    import torch
    import transformers
    return torch, transformers


def _visible_reply(text: str) -> str:
    """Drop a reasoning trace. An unclosed trace is not a prompt."""
    visible = _THINKING.sub("", text).strip()
    if "<think>" in visible:
        return ""
    return visible


def _accept(original: str, rewritten: str) -> EnhanceResult:
    cleaned = rewritten.strip()
    if not cleaned:
        return EnhanceResult(original, original, rewritten, False, "empty")
    if _sorted(stated_ages(original)) != _sorted(stated_ages(cleaned)):
        return EnhanceResult(original, original, cleaned, False, "age")
    if any(piece not in cleaned for piece in quoted_spans(original)):
        return EnhanceResult(original, original, cleaned, False, "quote")
    return EnhanceResult(cleaned, original, cleaned, True, "")


def _sorted(ages: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(ages))


def _lookup_word(word: str) -> int | None:
    for table in (_EN_UNITS, _EN_TENS, _RU_UNITS, _RU_TENS):
        found = table.get(word)
        if found is not None:
            return found
    return None


def _parse_age_token(token: str) -> int | None:
    raw = token.casefold().replace("ё", "е").replace("-", " ").strip()
    if re.fullmatch(r"\d{1,3}", raw):
        number = int(raw)
        return number if 1 <= number <= 120 else None
    parts = raw.split()
    if len(parts) == 1:
        single = _lookup_word(parts[0])
        return single if single is not None and 1 <= single <= 120 else None
    if len(parts) != 2:
        return None
    tens = _EN_TENS.get(parts[0])
    if tens is None:
        tens = _RU_TENS.get(parts[0])
    units = _EN_UNITS.get(parts[1])
    if units is None:
        units = _RU_UNITS.get(parts[1])
    if tens is None or units is None:
        return None
    combined = tens + units
    return combined if 1 <= combined <= 120 else None


def _alt(words: Mapping[str, int]) -> str:
    return "|".join(sorted(words, key=len, reverse=True))


def _number_pattern() -> str:
    digits = r"\d{1,3}"
    english = rf"(?:{_alt(_EN_TENS)})(?:[-\s](?:{_alt(_EN_UNITS)}))?|(?:{_alt(_EN_UNITS)})"
    russian = rf"(?:{_alt(_RU_TENS)})(?:[-\s](?:{_alt(_RU_UNITS)}))?|(?:{_alt(_RU_UNITS)})"
    return rf"(?:{digits}|{english}|{russian})"


def _age_patterns() -> tuple[re.Pattern[str], ...]:
    number = _number_pattern()
    bodies = (
        rf"\b({number})\s*-\s*years?\s*-\s*old\b",
        rf"\b({number})\s+years?\s+old\b",
        rf"\b({number})\s*yo\b",
        rf"\b(?:aged|age)\s+({number})\b",
        r"\b(?:his|her|their)\s+(\d{2})s\b",
        r"\b(1[0-9]|[1-9])(?:st|nd|rd|th)\b",
        rf"\b({number})\s*-\s*летн",
        rf"\b({number})\s+лет\b",
        rf"\b({number})\s+года\b",
        rf"\b({number})\s+год\b",
    )
    return tuple(re.compile(body) for body in bodies)


def _keep_spans(spans: list[tuple[int, int, int]]) -> tuple[int, ...]:
    ordered = sorted(spans, key=lambda item: (item[0], item[0] - item[1]))
    ages: list[int] = []
    last_end = -1
    for start, end, age in ordered:
        if start < last_end:
            continue
        ages.append(age)
        last_end = end
    return tuple(ages)


_DECADE_RE = re.compile(
    r"\b(twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b",
)
_RU_COMPACT = re.compile(
    "(" + _alt(_RU_TENS) + ")(" + _alt(_RU_UNITS) + ")летн",
)
