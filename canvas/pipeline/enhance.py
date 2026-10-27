"""Rewrite a prompt into English with the scenario writer."""
from __future__ import annotations

import gc
import json
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, assert_never

from canvas.app.i18n import text
from canvas.infra.errors import note_error
from canvas.infra.load_lock import pretrained_lock
from canvas.infra.stage_guard import StageBlocked, admit, begin_stage, finish_stage
from canvas.infra.stage_timer import log_detail, log_stage, note_tag, timed
from canvas.infra.vram_trace import drop_model, hold_model, note_vram
from canvas.models.modes import Tier
from canvas.pipeline.bf16_schedule import VramRefused
from canvas.pipeline.device_guard import (
    DeviceRefused,
    place_on_cuda,
    reclaim_cuda,
    require_cuda,
    require_writer_materialized,
    weights_on_cuda,
)
from canvas.pipeline.generation_weights import (
    dtype_label,
    kv_cache_label,
    prepare_generation,
)
from canvas.pipeline.writer_decode import GraphDecode, cache_len, try_decode
from canvas.models.registry import Catalog, ModelSpec
from canvas.models import hot_tier as hot
from canvas.models.runtime import park_pipeline
from canvas.models.snapshots import SnapshotError, download_snapshot, snapshot_ready
from canvas.models.verified import flush_process_cache, process_cache
from canvas.safety.gate import SafetyError

AGING_DESCRIPTOR_MIN_AGE = 45
"""Aging descriptors may be added only when every stated age is at least this."""

SYSTEM_PROMPT = f"""You rewrite image prompts for a text-to-image model that follows natural English best.

Improve the user's prompt and write it in natural English. Expand it into a concrete photographic or artistic description that names the subject, composition, lighting, lens, and style. You may add that photographic detail when it fits the prompt.

Rules:
- Preserve every fact the user stated.
- Keep every stated age and every adult marker verbatim. Do not invent, drop, round, or change an age.
- Do not add people. Do not add youthful descriptors.
- Add words about age or skin (mature, middle-aged, older, aged, wrinkles, smile lines, laugh lines, crow's feet, grey hair, weathered) only when every stated age is {AGING_DESCRIPTOR_MIN_AGE} or older. Below that age, or with no stated age, do not add them. Never add youthful-appearance words: no young-looking, youthful, petite, smooth skin, round face, baby face. Describe the person with the stated age number.
- In an explicit or nude scene, every person is an ordinary adult: an adult face, an adult body and adult styling. Never add school, student, uniform, anime, chibi or young-looking details to such a scene.
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
    "age_descriptor": "enhance.age_descriptor",
    "quote": "enhance.quote",
    "empty": "enhance.empty",
}

AGING_TERMS = (
    "mature", "middle-aged", "middle aged", "older", "elderly", "aged", "aging", "ageing",
    "wrinkles", "wrinkled", "wrinkle", "fine lines", "smile lines", "laugh lines", "crow's feet",
    "grey hair", "gray hair", "greying", "graying", "weathered", "sagging",
)
_AGING_RE = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(term) for term in sorted(AGING_TERMS, key=len, reverse=True))
    + r")(?![\w-])",
    re.IGNORECASE,
)


def aging_terms(text: str) -> tuple[str, ...]:
    """Aging descriptors in ``text``, case-folded, in order."""
    return tuple(match.group(1).casefold() for match in _AGING_RE.finditer(text))


def aging_terms_allowed(original: str) -> bool:
    """True when ``original`` states ages and all are ``AGING_DESCRIPTOR_MIN_AGE`` or older."""
    ages = stated_ages(original)
    return bool(ages) and min(ages) >= AGING_DESCRIPTOR_MIN_AGE


def added_aging_terms(original: str, rewritten: str) -> tuple[str, ...]:
    """Aging descriptors added to ``rewritten``. Empty when the stated ages allow them."""
    if aging_terms_allowed(original):
        return ()
    known = set(aging_terms(original))
    return tuple(term for term in aging_terms(rewritten) if term not in known)


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
    """True when a Qwen3 checkpoint can generate text."""
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
    """Stated ages."""
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


def mask_numbers(text: str) -> str:
    """The text with every digit and every English or Russian number word replaced by ``#``."""
    folded = unicodedata.normalize("NFKC", text)
    masked = re.sub(r"\d", "#", folded)
    return _NUMBER_WORDS.sub("#", masked)


def quoted_spans(text: str) -> tuple[str, ...]:
    """Quoted strings that must stay in their original script."""
    found: list[str] = []
    for match in _QUOTE.finditer(text):
        piece = next(group for group in match.groups() if group)
        if piece.strip():
            found.append(piece)
    return tuple(found)


def resolve_prompt(gate: object, original: str, rewritten: str | None) -> EnhanceResult:
    """Resolve prompt."""
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
    """Fallback message."""
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
    """The Z-Image-Turbo pack for this card. Its text encoder stays the generator's encoder."""
    if tier is Tier.GB16:
        return catalog.by_id("z-image-turbo-bf16-16gb")
    if tier is Tier.GB24:
        return catalog.by_id("z-image-turbo-bf16")
    assert_never(tier)


WRITER_ID = "qwen3-4b-writer"


def writer_spec(catalog: Catalog) -> ModelSpec:
    """The text model that writes a Photoshoot scenario and an improved prompt."""
    return catalog.by_id(WRITER_ID)


def open_rewriter(
    catalog: Catalog,
    tier: Tier,
    root: Path,
    progress: Callable[[str], None] | None = None,
    *,
    max_new_tokens: int = 320,
    device: str | None = None,
    stage: str = "writer",
) -> QwenRewriter:
    """Open the scenario writer. Missing files are downloaded on first use.

    The caller closes the writer before diffusion. On a 16 GB card the weights
    are stored as FP8. ``stage`` is ``writer`` or ``enhancer``.
    """
    if tier is not Tier.GB16 and tier is not Tier.GB24:
        assert_never(tier)
    if stage not in {"writer", "enhancer"}:
        raise ValueError(f"unknown stage {stage}")
    spec = writer_spec(catalog)
    folder = snapshot_ready(spec, root, process_cache())
    if folder is None:
        if progress is not None:
            progress(text("writer.download"))
        folder = download_snapshot(spec, root, progress)
    else:
        flush_process_cache()
    return QwenRewriter(
        folder,
        folder,
        max_new_tokens=max_new_tokens,
        device=device,
        tier=tier,
        stage=stage,
    )


class QwenRewriter:
    """Causal generation from a local text-model directory."""

    def __init__(
        self,
        encoder_dir: Path,
        tokenizer_dir: Path,
        *,
        max_new_tokens: int = 320,
        device: str | None = None,
        tier: Tier | None = None,
        stage: str = "writer",
    ) -> None:
        self.encoder_dir = encoder_dir
        self.tokenizer_dir = tokenizer_dir
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.tier = tier
        self.stage = stage
        self.closed = False
        self._model: Any = None
        self._tokenizer: Any = None
        self._dtype = "bf16"
        self._graph: GraphDecode | None = None

    def rewrite(self, prompt: str) -> str:
        """Generate a rewrite, then unload. The caller can close again."""
        try:
            self._load()
            return self._complete(prompt, SYSTEM_PROMPT)
        finally:
            self.close()

    def complete(self, prompt: str, *, system: str) -> str:
        """Generate one reply and keep the model loaded for another call."""
        self._load()
        self.closed = False
        return self._complete(prompt, system)

    def close(self) -> None:
        """Park the writer and release every VRAM byte it held. RAM stays when the tier keeps it.

        The graph, its private pool, and the StaticCache go first. The park then
        points the weights at RAM, and a tensor still on CUDA is moved off one at
        a time. The residual is logged after the allocator is emptied.
        """
        note_vram(self.stage, "before", self._model)
        try:
            graph = self._graph
            self._graph = None
            if graph is not None:
                graph.release()
            del graph
            if self._model is not None:
                _drop_generation_cache(self._model)
            if self._model is not None and not hot.retained("writer"):
                hot.remember("writer", self._model)
            hot.deactivate("writer")
            if self._model is not None:
                _evict_cuda(self._model)
            if not hot.retained("writer") and self._model is not None:
                mover = getattr(self._model, "to", None)
                if callable(mover):
                    try:
                        self._model = mover("cpu")
                    except (TypeError, RuntimeError, AttributeError):
                        pass
                self._model = None
                self._tokenizer = None
        finally:
            drop_model(self.stage)
            drop_model("writer")
            self.closed = True
            _release_cuda(_optional_torch(), self.stage)
            note_vram(self.stage, "after")
            if self.tier is Tier.GB16:
                try:
                    finish_stage(self.stage, self.tier)
                except StageBlocked as exc:
                    raise VramRefused() from exc

    def _load(self) -> None:
        if self._model is not None:
            self.closed = False
            self._seat_cuda(self._model)
            self._capture_decode(rebuild=True)
            return
        found = hot.cached("writer")
        if found is not None:
            self._model = found
            tokenizer = hot.bundle("writer")
            if tokenizer is not None:
                self._tokenizer = tokenizer
            self.closed = False
            self._seat_cuda(found)
            self._capture_decode(rebuild=True)
            return
        park_pipeline()
        torch, transformers = _generation_libs()
        reclaim_cuda(torch)
        if self.tier is Tier.GB16:
            try:
                begin_stage(self.stage, self.tier)
            except StageBlocked as exc:
                raise VramRefused() from exc
        note_vram("writer_load", "before")
        with timed("writer_load"):
            config_path = self.encoder_dir / "config.json"
            index_path = self.encoder_dir / "model.safetensors.index.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            index = json.loads(index_path.read_text(encoding="utf-8"))
            names = index.get("weight_map") or {}
            if not isinstance(names, dict) or not causal_lm_ready(config, names):
                raise SnapshotError("the text encoder cannot generate text")
            cuda = bool(torch.cuda.is_available())
            if self.device is None and not cuda:
                note_error("writer.nocuda")
                note_tag("writer_fallback", "nocuda")
            device = self.device or ("cuda" if cuda else "cpu")
            if device == "cpu" and cuda:
                note_error("writer.device")
                raise DeviceRefused()
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.tokenizer_dir, local_files_only=True,
            )
            try:
                # Straight onto the GPU in bf16, the load bench G measured at
                # 7.75 GB. A CPU load with low_cpu_mem_usage builds meta
                # tensors (and an fp32 copy) and model_park cannot copy them.
                started_ssd = time.perf_counter()
                model = _causal_model(transformers, torch, self.encoder_dir, device)
                hot.note_ssd("writer", model, started_ssd)
                # One BF16 copy. FP8 is not installed: it kept the original
                # weights alive next to the cast copy.
                if device != "cpu" and self.tier is Tier.GB16:
                    with timed("cast"):
                        chosen, _fell_back = prepare_generation(model, torch)
                    self._dtype = chosen
                elif device != "cpu":
                    self._dtype = "bf16"
                require_writer_materialized(model)
                if device != "cpu":
                    require_cuda(model, "writer.device", torch)
                if hot.remember("writer", model):
                    hot.remember_bundle("writer", tokenizer)
                if device != "cpu" and hot.retained("writer"):
                    with timed("place"):
                        hot.activate("writer", "cuda", blocking=True)
                if device != "cpu":
                    require_cuda(model, "writer.device", torch)
                reclaim_cuda(torch)
            except DeviceRefused:
                raise
            except RuntimeError as exc:
                if device == "cpu" or "out of memory" not in str(exc).lower():
                    raise
                if cuda:
                    torch.cuda.empty_cache()
                note_error("writer.oom")
                raise DeviceRefused() from exc
            self._tokenizer = tokenizer
            self._model = model
            self.device = device
            self.closed = False
            if self.tier is Tier.GB16:
                try:
                    admit(self.stage, model, self.tier)
                except StageBlocked as exc:
                    raise VramRefused() from exc
            else:
                hold_model(self.stage, model)
            note_vram("writer_load", "after", model)
            if device in {"cpu", "cuda"}:
                note_tag("writer", device)
        self._capture_decode()

    def _seat_cuda(self, model: Any) -> None:
        """Unpark onto CUDA. A parameter or buffer left on CPU or meta stops the writer.

        A machine with no CUDA keeps the existing CPU fallback. A CUDA machine
        never generates with the writer still on the host.
        """
        torch, _transformers = _generation_libs()
        try:
            cuda = bool(torch.cuda.is_available())
        except (AttributeError, TypeError, RuntimeError):
            cuda = False
        if not cuda:
            return
        if self.device == "cpu":
            note_error("writer.device")
            raise DeviceRefused()
        self.device = "cuda"
        activated = hot.activate("writer", "cuda", blocking=True)
        if _has_resident_tensors(model):
            if not weights_on_cuda(model):
                place_on_cuda(model)
            require_cuda(model, "writer.device", torch)
        elif not activated:
            place_on_cuda(model)

    def _capture_decode(self, *, rebuild: bool = False) -> None:
        """Capture the 1-token graph once the weights are on the card.

        ``rebuild`` drops the previous graph and its StaticCache. Unpark moves
        the weights, so the cache has to be allocated on CUDA again.
        """
        if self.device != "cuda" or self._model is None:
            return
        torch, _transformers = _generation_libs()
        if rebuild and self._graph is not None:
            self._graph.invalidate()
        if self._graph is None or self._graph.model is not self._model:
            self._graph = GraphDecode(
                self._model, torch, cache_len(self.max_new_tokens), stage=self.stage,
            )
        if self._graph.stale():
            self._graph.capture()

    def _complete(self, prompt: str, system: str) -> str:
        torch, _transformers = _generation_libs()
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            raise SnapshotError("the text encoder is not loaded")
        try:
            cuda_up = bool(torch.cuda.is_available())
        except (AttributeError, TypeError, RuntimeError):
            cuda_up = False
        if cuda_up and self.device != "cuda":
            note_error("writer.device")
            raise DeviceRefused()
        if self.device == "cuda":
            require_cuda(model, "writer.device", torch)
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
        count = 0
        kv_cache = kv_cache_label(model)
        graph_label = "eager"
        prefill_ms = 0
        with timed("writer_generate"):
            started = time.perf_counter()
            decoded = try_decode(model, torch, encoded["input_ids"], self.max_new_tokens, self._graph)
            if decoded is None:
                with torch.inference_mode():
                    # use_cache keeps the KV cache on the GPU. No streamer and no
                    # stopping criterion: those call .item() and stall the card
                    # between tokens. inference_mode keeps autograd off.
                    output = model.generate(
                        **encoded,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        use_cache=kv_cache == "on",
                    )
                elapsed = time.perf_counter() - started
                start = int(encoded["input_ids"].shape[-1])
                count = _generated_count(output, start)
                reply = str(tokenizer.decode(output[0, start:], skip_special_tokens=True))
                rate = 0 if elapsed <= 0 else min(1_000_000, int(count / elapsed))
                decode_rate = rate
            else:
                count = decoded.count
                prefill_ms = decoded.prefill_ms
                decode_rate = decoded.decode_tok_s
                graph_label = decoded.graph
                reply = str(tokenizer.decode(decoded.ids, skip_special_tokens=True))
                rate = decode_rate
        note_tag("writer_tps", f"tps{rate}")
        logged = device if device in {"cpu", "cuda"} else "none"
        resident = dtype_label(model, self._dtype)
        log_stage(
            "writer", "tokens", logged,
            tokens=count, tok_s=rate, prefill_ms=prefill_ms, decode_tok_s=decode_rate,
            graph=graph_label, dtype=resident, kv_cache=kv_cache,
        )
        return _visible_reply(reply)


class WriterSession:
    """One writer for every scenario chunk of a series.

    Calling the session is one reply. The first call opens the writer, later
    calls reuse it while it stays on the card, and ``release`` unloads it once.
    """

    def __init__(
        self, open_writer: Callable[[int], QwenRewriter], system: str, budget: Callable[[str], int],
    ) -> None:
        self._open = open_writer
        self._system = system
        self._budget = budget
        self._writer: QwenRewriter | None = None

    def __call__(self, prompt: str) -> str:
        """Reply to one scenario request. ``budget`` sets its new-token cap."""
        budget = self._budget(prompt)
        if self._writer is None:
            self._writer = self._open(budget)
        self._writer.max_new_tokens = budget
        return self._writer.complete(prompt, system=self._system)

    def release(self) -> None:
        """Unload the writer. A second call does nothing."""
        writer = self._writer
        self._writer = None
        if writer is not None:
            writer.close()


def _has_resident_tensors(module: Any) -> bool:
    """True when the module exposes a parameter or a buffer."""
    for name in ("parameters", "buffers"):
        method = getattr(module, name, None)
        if not callable(method):
            continue
        try:
            if any(True for _tensor in method()):
                return True
        except (TypeError, RuntimeError):
            continue
    return False


def _writer_load_attempts(dtype: object, device: str) -> tuple[dict[str, Any], ...]:
    """``from_pretrained`` kwargs for the writer, as in the Proving Ground bench G load.

    CUDA is ``dtype=bfloat16`` with ``device_map="cuda"``, so the weights are
    one real bf16 copy on the card. transformers 5 reads ``dtype``, and
    ``torch_dtype`` is the retry for a build that rejects it.
    ``low_cpu_mem_usage`` is not passed: transformers 5 drops it, and 4.x
    refuses ``False`` next to a device map. A CPU machine gets the same bf16
    load with no device map.
    """
    if device != "cuda" and device != "cpu":
        raise ValueError(f"unknown device {device}")
    common: dict[str, Any] = {"local_files_only": True}
    if device == "cuda":
        common["device_map"] = "cuda"
    return (
        {**common, "dtype": dtype},
        {**common, "torch_dtype": dtype},
    )


def _causal_model(transformers: Any, torch: Any, folder: Path, device: str) -> Any:
    """Load the writer in bf16. CUDA goes straight to ``device_map="cuda"``.

    The build holds the process load lock, so a check loading on the side
    thread cannot change torch's default dtype under it.
    """
    loader = transformers.AutoModelForCausalLM.from_pretrained
    with pretrained_lock():
        model = _open_causal(loader, folder, torch, device)
    settle_bf16(model, torch, device)
    return model.eval()


def settle_bf16(model: Any, torch: Any, device: str) -> int:
    """Log the loaded parameter dtype and cast any fp32 parameter to bf16.

    The load lock is the fix. This is the backstop: a stray fp32 parameter is
    cast in place, one tensor at a time, so the card never holds a second
    copy of the model. Returns how many parameters were cast.
    """
    fp32 = getattr(torch, "float32", None)
    target = getattr(torch, "bfloat16", None)
    params = _module_parameters(model)
    found = [param for param in params if fp32 is not None and getattr(param, "dtype", None) == fp32]
    loaded = dtype_label(model, "bf16")
    cast = 0
    for param in found:
        caster = getattr(getattr(param, "data", None), "to", None)
        if target is None or not callable(caster):
            continue
        try:
            param.data = caster(target)
        except (TypeError, RuntimeError, AttributeError):
            continue
        cast += 1
    kind = device if device in {"cpu", "cuda"} else "cpu"
    log_detail(
        "writer", "dtype", kind,
        f"loaded={loaded} params={len(params)} fp32={len(found)} cast={cast} now={dtype_label(model, 'bf16')}",
    )
    if found:
        note_error("writer.fp32_cast")
    return cast


def _open_causal(loader: Any, folder: Path, torch: Any, device: str) -> Any:
    """Open the writer. A build that rejects ``dtype`` is tried with ``torch_dtype``."""
    attempts = _writer_load_attempts(torch.bfloat16, device if device == "cuda" else "cpu")
    error: Exception | None = None
    last = len(attempts) - 1
    for index, kwargs in enumerate(attempts):
        try:
            return loader(folder, **kwargs)
        except TypeError as exc:
            error = exc
        except (ValueError, ImportError) as exc:
            if index == last:
                raise
            error = exc
    if error is None:
        raise SnapshotError("the text encoder cannot be loaded")
    raise error


# A writer close that leaves more than this allocated on the card is logged as a leak.
WRITER_RESIDUAL_LIMIT_BYTES = 512 * 1024**2


def _module_parameters(module: Any) -> list[Any]:
    method = getattr(module, "parameters", None)
    if not callable(method):
        return []
    try:
        return list(method())
    except (TypeError, RuntimeError):
        return []


def _drop_generation_cache(model: Any) -> None:
    """Forget a cache ``generate`` stored on the model for reuse."""
    for name in ("_cache", "_static_cache"):
        if getattr(model, name, None) is None:
            continue
        try:
            setattr(model, name, None)
        except (AttributeError, TypeError):
            continue


def _evict_cuda(module: Any) -> int:
    """Move each parameter or buffer still on CUDA to the CPU, one at a time.

    The park reassigns the tensors it holds. Anything it does not hold, such
    as a buffer missing from the park dict, would otherwise keep its VRAM.
    Returns how many tensors moved.
    """
    moved = 0
    for name in ("parameters", "buffers"):
        method = getattr(module, name, None)
        if not callable(method):
            continue
        try:
            tensors = list(method())
        except (TypeError, RuntimeError):
            continue
        for tensor in tensors:
            if not _tensor_on_cuda(tensor):
                continue
            mover = getattr(getattr(tensor, "data", None), "to", None)
            if not callable(mover):
                continue
            try:
                tensor.data = mover("cpu")
            except (TypeError, RuntimeError, AttributeError):
                continue
            moved += 1
    return moved


def _tensor_on_cuda(tensor: Any) -> bool:
    device = getattr(tensor, "device", None)
    kind = getattr(device, "type", None)
    text = kind if isinstance(kind, str) else str(device or "")
    return "cuda" in text


def _release_cuda(torch: Any, stage: str) -> int:
    """Collect, sync, empty the allocator, and log what is still allocated.

    Returns the allocated bytes, or 0 when there is no CUDA.
    """
    gc.collect()
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    if cuda is None:
        return 0
    try:
        if not bool(cuda.is_available()):
            return 0
    except (TypeError, RuntimeError, AttributeError):
        return 0
    for name in ("synchronize", "empty_cache", "ipc_collect"):
        fn = getattr(cuda, name, None)
        if not callable(fn):
            continue
        try:
            fn()
        except (TypeError, RuntimeError):
            continue
    allocated = _cuda_counter(cuda, "memory_allocated")
    reserved = _cuda_counter(cuda, "memory_reserved")
    over = allocated > WRITER_RESIDUAL_LIMIT_BYTES
    log_detail(
        stage if stage in {"writer", "enhancer"} else "writer", "release", "cuda",
        f"allocated={allocated} reserved={reserved} over={int(over)}",
    )
    if over:
        note_error(f"{stage}.residual")
    return allocated


def _cuda_counter(cuda: Any, name: str) -> int:
    fn = getattr(cuda, name, None)
    if not callable(fn):
        return 0
    try:
        return max(0, int(fn()))
    except (TypeError, ValueError, RuntimeError):
        return 0


def _generated_count(output: Any, start: int) -> int:
    """New tokens in one generate result. A short fake reply still counts."""
    try:
        row = output[0, start:]
    except (TypeError, IndexError, KeyError):
        try:
            row = output[0]
        except (TypeError, IndexError, KeyError):
            return 0
    if isinstance(row, (list, tuple)):
        return len(row)
    shape = getattr(row, "shape", None)
    if shape is None:
        return 0
    try:
        return int(shape[-1])
    except (TypeError, ValueError, IndexError):
        return 0


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
    """Visible reply."""
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
    if added_aging_terms(original, cleaned):
        return EnhanceResult(original, original, cleaned, False, "age_descriptor")
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

_NUMBER_WORDS = re.compile(
    r"(?<![\w])(?:" + "|".join(
        re.escape(word) for word in sorted(
            {*_EN_UNITS, *_EN_TENS, *_DECADE_WORD, *_RU_UNITS, *_RU_TENS}, key=len, reverse=True,
        )
    ) + r")(?![\w])",
    re.IGNORECASE,
)
