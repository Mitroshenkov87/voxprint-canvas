"""Greedy writer decode: one captured CUDA graph around a single token.

Prefill stays eager because the prompt length changes. The decode step is
captured once after the weights are on the card and again if an unpark moves
those buffers. The decode clock waits for ``torch.cuda.synchronize`` so the
rate is the tokens that finished, not the replay launches. A failed capture
uses the same argmax loop without a graph. ``generate`` remains the last
resort for a stand-in that has no ``forward``. CUDA never falls through to a
CPU writer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from canvas.infra.stage_timer import log_detail
from canvas.pipeline.device_guard import require_cuda

# Room for the system prompt, the scenario JSON, and the chat wrapper.
WRITER_MAX_PROMPT_TOKENS = 2048
_WARMUP = 3


@dataclass(frozen=True)
class DecodeRun:
    """New token ids and the split clock. ``graph`` is ``captured`` or ``eager``."""

    ids: Any
    count: int
    prefill_ms: int
    decode_tok_s: int
    graph: str


class GraphDecode:
    """A static cache and, when capture works, a 1-token CUDA graph."""

    def __init__(self, model: Any, torch: Any, max_cache_len: int, *, stage: str = "writer") -> None:
        self.model = model
        self.torch = torch
        self.max_cache_len = max(1, int(max_cache_len))
        self.stage = stage if stage in {"writer", "enhancer"} else "writer"
        self.graph: Any = None
        self.cache: Any = None
        self.static_input: Any = None
        self.cache_position: Any = None
        self.static_logits: Any = None
        self.label = "eager"
        self.reason = "unready"
        self._ptr = -1

    def stale(self) -> bool:
        """True when the graph is missing or the weight storage has moved."""
        if self.graph is None:
            return True
        return _weight_ptr(self.model) != self._ptr

    def invalidate(self) -> None:
        """Drop a graph whose weight addresses are about to change."""
        self.graph = None
        self.label = "eager"
        self.reason = "moved"
        self._ptr = -1

    def release(self) -> None:
        """Drop the graph, its private pool, the StaticCache, and the model.

        ``invalidate`` keeps the cache and the static buffers for a recapture.
        The writer stage ends with this instead, so nothing here holds VRAM
        when the picture stage starts.
        """
        self.invalidate()
        self.reason = "released"
        cache = self.cache
        self.cache = None
        self.static_input = None
        self.cache_position = None
        self.static_logits = None
        self.model = None
        _drop_cache(cache)

    def capture(self) -> None:
        """Capture the 1-token step. Failure stays on the eager loop."""
        try:
            if not _cuda_ready(self.torch):
                self._mark("eager", "cpu")
                return
            if getattr(self.model, "config", None) is None:
                self._mark("eager", "no_config")
                return
            self._capture()
            self._ptr = _weight_ptr(self.model)
            self._mark("captured", "")
        except Exception as exc:
            self.graph = None
            self._mark("eager", _reason(exc))

    def run(self, input_ids: Any, new_tokens: int) -> DecodeRun | None:
        """Prefill, then replay the graph. None when this call cannot use it."""
        count = _count(new_tokens)
        try:
            prompt_len = int(input_ids.shape[-1])
        except (TypeError, AttributeError, ValueError):
            return None
        if (
            self.graph is None
            or self.cache is None
            or self.static_input is None
            or self.static_logits is None
            or prompt_len + count > self.max_cache_len
        ):
            return None
        self.cache.reset()
        started = time.perf_counter()
        with self.torch.inference_mode():
            position = self.torch.arange(prompt_len, device=input_ids.device)
            output = self.model(
                input_ids=input_ids,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=position,
            )
            token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        _sync(self.torch)
        prefill_ms = _ms(started)
        pieces = [token]
        started = time.perf_counter()
        position_value = prompt_len
        for _step in range(count - 1):
            self.static_input.copy_(token)
            self.cache_position.fill_(position_value)
            self.graph.replay()
            token = self.static_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            pieces.append(token)
            position_value += 1
        _sync(self.torch)
        return _run(self.torch, pieces, prefill_ms, time.perf_counter() - started, "captured")

    def _capture(self) -> None:
        torch = self.torch
        self.cache = _new_cache(self.model, self.max_cache_len)
        self.static_input = torch.zeros((1, 1), dtype=torch.long, device="cuda")
        self.cache_position = torch.zeros((1,), dtype=torch.long, device="cuda")
        # no_grad, not inference_mode: inference tensors cannot be replayed by a CUDA graph.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.no_grad():
                output = None
                for _step in range(_WARMUP):
                    output = self._step()
        torch.cuda.current_stream().wait_stream(stream)
        if output is None:
            raise RuntimeError("decode warmup produced no logits")
        self.static_logits = torch.empty_like(output.logits)
        graph = torch.cuda.CUDAGraph()

        def replay_body() -> None:
            produced = self._step()
            self.static_logits.copy_(produced.logits)

        with torch.cuda.graph(graph):
            with torch.no_grad():
                replay_body()
        # A real CUDA graph replays the captured kernels. A test double stores
        # the same body and runs it again, so the tokens stay the eager ones.
        bind = getattr(graph, "bind_replay", None)
        if callable(bind):
            bind(replay_body)
        self.graph = graph
        self.cache.reset()

    def _step(self) -> Any:
        return self.model(
            input_ids=self.static_input,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=self.cache_position,
        )

    def _mark(self, label: str, reason: str) -> None:
        self.label = label
        self.reason = reason
        suffix = f"graph={label}"
        if reason:
            suffix += f" reason={reason}"
        device = "cuda" if _cuda_ready(self.torch) else "cpu"
        log_detail(self.stage, "decode", device, suffix)


def try_decode(
    model: Any, torch: Any, input_ids: Any, new_tokens: int, graph: GraphDecode | None,
) -> DecodeRun | None:
    """Graph replay when it is current, otherwise the eager argmax loop.

    None means the caller should use ``generate``. That is the stand-in path
    and the path taken when the static cache cannot be built.
    """
    require_cuda(model, "writer.device", torch)
    if graph is not None:
        if graph.stale():
            graph.capture()
        try:
            found = graph.run(input_ids, new_tokens)
        except Exception as exc:
            graph._mark("eager", _reason(exc))
            found = None
        if found is not None:
            return found
    return eager_decode(model, torch, input_ids, new_tokens)


def eager_decode(model: Any, torch: Any, input_ids: Any, new_tokens: int) -> DecodeRun | None:
    """Argmax one token at a time with a static cache and no CUDA graph."""
    if not callable(getattr(model, "forward", None)):
        return None
    count = _count(new_tokens)
    try:
        prompt_len = int(input_ids.shape[-1])
    except (TypeError, AttributeError, ValueError):
        return None
    try:
        cache = _new_cache(model, prompt_len + count)
    except (OSError, RuntimeError, TypeError, ValueError, AttributeError, ImportError):
        return None
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            position = torch.arange(prompt_len, device=getattr(input_ids, "device", "cpu"))
            output = model(
                input_ids=input_ids, past_key_values=cache, use_cache=True, cache_position=position,
            )
            token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        _sync(torch)
        prefill_ms = _ms(started)
        pieces = [token]
        started = time.perf_counter()
        for index in range(count - 1):
            with torch.inference_mode():
                position = torch.arange(1, device=getattr(token, "device", "cpu"))
                position = position + (prompt_len + index)
                output = model(
                    input_ids=token, past_key_values=cache, use_cache=True, cache_position=position,
                )
                token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            pieces.append(token)
        _sync(torch)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return _run(torch, pieces, prefill_ms, time.perf_counter() - started, "eager")


def cache_len(new_tokens: int) -> int:
    """Static-cache length: the prompt budget plus the new-token cap."""
    return WRITER_MAX_PROMPT_TOKENS + _count(new_tokens)


def _run(
    torch: Any, pieces: list[Any], prefill_ms: int, decode_s: float, graph: str,
) -> DecodeRun:
    ids = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)
    produced = _produced(ids, len(pieces))
    # The first id comes from prefill. The rate covers the tokens decoded after it.
    steps = max(0, produced - 1)
    rate = 0 if steps == 0 or decode_s <= 0 else min(1_000_000, int(steps / decode_s))
    return DecodeRun(ids, produced, prefill_ms, rate, graph)


def _produced(ids: Any, fallback: int) -> int:
    """How many new token ids were actually built."""
    if isinstance(ids, (list, tuple)):
        return len(ids)
    shape = getattr(ids, "shape", None)
    if shape is not None:
        try:
            width = int(shape[-1])
        except (TypeError, ValueError, IndexError):
            width = 0
        if width > 0:
            return width
    return fallback if fallback > 0 else 0


def _sync(torch: Any) -> None:
    """Wait until queued CUDA work has finished. A CPU double has nothing to wait for."""
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    try:
        if not bool(cuda.is_available()):
            return
    except (TypeError, RuntimeError):
        return
    sync = getattr(cuda, "synchronize", None)
    if not callable(sync):
        return
    try:
        sync()
    except (TypeError, RuntimeError):
        return


def _new_cache(model: Any, length: int) -> Any:
    # transformers is an optional extra. Importing it here keeps Canvas importable
    # when the package is absent. This is not a circular import.
    from transformers import StaticCache

    return StaticCache(config=model.config, max_cache_len=length)


def _drop_cache(cache: Any) -> None:
    """Let go of each StaticCache layer's key and value tensors.

    The cache object can outlive this call in a traceback or a log, so its
    layers are emptied rather than relying on the last reference going away.
    """
    if cache is None:
        return
    layers = getattr(cache, "layers", None)
    if isinstance(layers, list):
        for layer in layers:
            for name in ("keys", "values"):
                if getattr(layer, name, None) is not None:
                    try:
                        setattr(layer, name, None)
                    except (AttributeError, TypeError):
                        continue
        layers.clear()


def _cuda_ready(torch: Any) -> bool:
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except (TypeError, RuntimeError):
        return False


def _weight_ptr(model: Any) -> int:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return 0
    try:
        rows = list(parameters())
    except (TypeError, RuntimeError):
        return 0
    for param in rows:
        data = getattr(param, "data", param)
        pointer = getattr(data, "data_ptr", None)
        if not callable(pointer):
            continue
        try:
            return int(pointer())
        except (TypeError, ValueError, RuntimeError):
            return 0
    return 0


def _count(new_tokens: int) -> int:
    if isinstance(new_tokens, bool) or not isinstance(new_tokens, int) or new_tokens < 1:
        return 1
    return new_tokens


def _ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def _reason(exc: BaseException) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in type(exc).__name__.lower())
    return cleaned[:40] or "error"
