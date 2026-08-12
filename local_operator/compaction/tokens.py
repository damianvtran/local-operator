"""Token estimation with per-message memoization.

Why this module exists
-----------------------
Every compaction decision (threshold checks, cut-point search, pruning guards)
is driven by token counts. Providers only report real usage after a request,
so between requests we estimate locally with tiktoken's ``cl100k_base``.
Estimation runs on hot paths (per-index suffix sums, backwards cut-point
walks), so results are memoized.

tiktoken lives behind the ``tokenizer`` extra: it is a compiled wheel that
also fetches its BPE ranks over the network on first use, neither of which
the default install should require. Without it every estimate falls back to
the ``len(text) // 4`` heuristic below, which is coarser but keeps compaction
working — real usage from the provider still corrects the accounting after
each request.

Settle rule / cache contract
----------------------------
The cache is a module-level dict keyed on the message's stable ``id``. Two
invariants keep it honest:

1. **Settle gate.** A streaming assistant is mutated under one identity while
   its ``usage``/``stop_reason`` are provisional, so assistants are cached
   only when settled — real ``usage`` (not ``None``) and a terminal
   ``stop_reason`` that is not ``None`` / ``"aborted"`` / ``"error"``.
   Unsettled assistants never read or insert. Non-assistant messages are
   immutable once appended and always cache. Loop finalization
   (``harness/loop.py``) sets ``usage``/``stop_reason`` BEFORE any estimate
   of the finished message runs and drops a provisional entry at that point.
2. **Owner invalidation.** Any in-place mutation of a message MUST call
   :func:`invalidate_message_cache`. Owners that mutate messages in place are
   the pruning passes (``local_operator.compaction.pruning``), streaming
   finalize, and any future image/reasoning stripper.

Cross-package consumers that keep derived caches can subscribe via
:func:`register_invalidator` and are notified on every invalidation.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from typing import Callable, Sequence

from local_operator.harness.types import Message, TextContent
from local_operator.optional import missing_extra_error

logger = logging.getLogger(__name__)

#: Flat token cost charged per image block. Matches
#: ``IMAGE_TOKEN_ESTIMATE``: vision providers bill images in fixed visual-token
#: chunks, and a flat estimate keeps the estimator deterministic and
#: independent of base64 payload length.
IMAGE_TOKEN_ESTIMATE = 1200

#: Fallback ratio when tiktoken is unavailable: ~4 chars per token.
_CHARS_PER_TOKEN_FALLBACK = 4

# ---------------------------------------------------------------------------
# Lazy tiktoken singleton
# ---------------------------------------------------------------------------

_ENCODING: object | None = None
_ENCODING_FAILED = False


def _get_encoding() -> object | None:
    """Lazily load the cl100k_base encoding; never raise.

    A missing or broken tiktoken install must degrade to the ``len(text) //
    4`` fallback, not crash compaction (or startup — loading is deferred to
    first use). The warning fires once per process, so it names the extra
    without turning into per-turn noise.
    """
    global _ENCODING, _ENCODING_FAILED
    if _ENCODING is not None or _ENCODING_FAILED:
        return _ENCODING
    try:
        import tiktoken

        _ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001 - degrade, never raise
        _ENCODING_FAILED = True
        logger.warning(
            "%s; falling back to the chars/4 token estimate (%s)",
            missing_extra_error("tokenizer", "Exact token counting"),
            exc,
        )
    return _ENCODING


#: Passed to every ``encode`` call. tiktoken REFUSES by default to encode text
#: containing a special-token literal such as ``<|endoftext|>`` and raises
#: ``ValueError``. Everything counted here is untrusted content — tool output,
#: file contents, model text — so that default turns any transcript that
#: merely MENTIONS the literal into a crash in the estimator, and the
#: estimator runs on every turn from pruning and the compaction threshold
#: check. Reproduced: a bash result containing ``<|endoftext|>`` raised out of
#: ``estimate_tokens``. An empty set means "treat them as ordinary text",
#: which is both non-raising and the correct count for content that is being
#: measured rather than sent as a control token.
_DISALLOWED_SPECIAL: tuple[str, ...] = ()


def _encode_len(text: str) -> int:
    """Token count of ``text`` via cl100k_base, or the chars/4 fallback."""
    if not text:
        return 0
    encoding = _get_encoding()
    if encoding is not None:
        encode = encoding.encode  # type: ignore[attr-defined]
        return len(encode(text, disallowed_special=_DISALLOWED_SPECIAL))
    return len(text) // _CHARS_PER_TOKEN_FALLBACK


#: Per-model encodings, resolved lazily and cached. Separate from the compaction
#: singleton on purpose: compaction wants ONE stable ruler for its budget
#: arithmetic, while an API reporting token stats should answer for the model the
#: caller actually named.
_MODEL_ENCODINGS: dict[str, object | None] = {}


def _get_model_encoding(model: str) -> object | None:
    """Encoding for ``model``, or None when tiktoken is unavailable.

    Falls back to the shared cl100k_base encoding for an unknown model name,
    which is what the previous inline implementation did.
    """
    if model in _MODEL_ENCODINGS:
        return _MODEL_ENCODINGS[model]
    encoding: object | None = None
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:  # noqa: BLE001 - unknown model name, not a failure
            encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 - degrade, never raise
        encoding = _get_encoding()  # logs the extra hint once
    _MODEL_ENCODINGS[model] = encoding
    return encoding


def count_text_tokens(text: str, model: str | None = None) -> int:
    """Public token count for a plain string.

    Callers outside compaction (the HTTP API's token stats, for example) need a
    count without reaching for tiktoken themselves. Going through here means
    they inherit the degradation ladder — exact counts when the ``tokenizer``
    extra is installed, the chars/4 estimate otherwise — instead of raising
    ImportError on an install that never asked for it.

    ``model`` selects the model's own encoding. It matters more than it looks:
    cl100k_base and o200k_base agree on ASCII prose and code, but differ by
    ~23% on CJK, so hardcoding one ruler made a reported ``total_tokens``
    over-report for non-Latin content on every o200k model (gpt-4o and later).
    Omit it only when any consistent ruler will do.
    """
    if not text:
        return 0
    encoding = _get_model_encoding(model) if model else _get_encoding()
    if encoding is not None:
        encode = encoding.encode  # type: ignore[attr-defined]
        return len(encode(text, disallowed_special=_DISALLOWED_SPECIAL))
    return len(text) // _CHARS_PER_TOKEN_FALLBACK


def approx_text_tokens(text: str) -> int:
    """Token count for ``text`` that NEVER loads the tokenizer.

    :func:`count_text_tokens` is exact when the ``tokenizer`` extra is present,
    and reaching for that exactness has a price paid once per process: loading
    cl100k_base costs ~84 ms and ~43.6 MB RSS (``scripts/bench_base_overhead``),
    and on a cold cache tiktoken fetches the ranks over the NETWORK — which
    offline is a connection timeout rather than a slow answer. Both
    :func:`local_operator.compaction.pruning.prune_transcript` and the session's
    compaction gate go out of their way to defer that load, precisely so a short
    run does not buy a 43.6 MB table to be told there is nothing to do.

    This is for callers who want a number NOW and would rather be somewhat out
    than spend that: a status readout, a progress hint, a size warning.

    **The error is a property of the content, not a bound.** ``chars // 4`` is
    exactly right only where the text averages four characters per token, and
    real text does not. Measured against cl100k_base:

    ==========================  ============  ========
    payload                     chars/token   error
    ==========================  ============  ========
    English prose                      5.59    +39.8%
    Python source                      3.61     -9.8%
    system prompt + tool JSON          4.30     +7.0%
    minified JSON                      1.82    -54.4%
    CJK                                1.36    -65.9%
    ==========================  ============  ========

    So the honest claim is narrow: on the mix this exists to measure — a Latin
    system prompt plus JSON tool schemas — it runs roughly +7% to +17% high,
    which a percentage rendered to one decimal can carry. It is NOT a
    general-purpose estimator, and a caller measuring CJK or minified payloads
    would be told they had spent a third of what they actually had. Anything
    load-bearing wants :func:`count_text_tokens` and should pay for it.

    Deliberately NOT "use the encoder when it is already resident". That would
    make the same session report different numbers depending on whether
    compaction happened to have run, and an estimate that moves for reasons the
    user cannot see is worse than one that is consistently approximate.
    """
    return len(text) // _CHARS_PER_TOKEN_FALLBACK if text else 0


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Prefix of ``text`` that fits within ``max_tokens`` tokens.

    Token-aligned truncation (decode of the first ``max_tokens`` ids) when
    tiktoken is available; otherwise a conservative ``max_tokens * 4`` chars
    prefix to match the estimator's fallback ratio.
    """
    if max_tokens <= 0 or not text:
        return ""
    encoding = _get_encoding()
    if encoding is not None:
        encode = encoding.encode  # type: ignore[attr-defined]
        ids = encode(text, disallowed_special=_DISALLOWED_SPECIAL)
        if len(ids) <= max_tokens:
            return text
        return encoding.decode(ids[:max_tokens])  # type: ignore[attr-defined]
    limit = max_tokens * _CHARS_PER_TOKEN_FALLBACK
    return text if len(text) <= limit else text[:limit]


# ---------------------------------------------------------------------------
# Estimate cache
# ---------------------------------------------------------------------------

#: Memoized estimates keyed by ``message.id``, LRU-bounded. The cache used to
#: be an unbounded dict that no session ever cleared — the server facade
#: builds and disposes a session per request, and converters mint fresh
#: message ids per call, so every message ever estimated leaked an entry for
#: the life of the process. The bound keeps the memoization useful for a
#: turn's working set without unbounded growth.
_ESTIMATE_CACHE: OrderedDict[str, int] = OrderedDict()
_ESTIMATE_CACHE_MAX = 4096

#: Cross-package subscribers notified on every invalidation. Insertion-ordered
#: dict keyed by ``id(callable)`` so registration is idempotent per callable
#: and unsubscribe is O(1).
_INVALIDATORS: dict[int, Callable[[Message], None]] = {}


def estimate_tokens(message: Message) -> int:
    """Estimated token cost of ``message`` as sent on the wire.

    Memoized per ``message.id``. Any in-place mutation of the message MUST be
    followed by :func:`invalidate_message_cache` (see module docstring) or a
    stale value is served. Images count as :data:`IMAGE_TOKEN_ESTIMATE` each;
    text is tokenized with cl100k_base or ``len(text) // 4`` when tiktoken is
    unavailable. Tool calls contribute their name + serialized arguments.
    """
    if not _is_settled(message):
        # Streaming assistant, still provisional: compute but never cache.
        return _compute_tokens(message)
    cached = _ESTIMATE_CACHE.get(message.id)
    if cached is not None:
        _ESTIMATE_CACHE.move_to_end(message.id)
        return cached

    tokens = _compute_tokens(message)
    _ESTIMATE_CACHE[message.id] = tokens
    while len(_ESTIMATE_CACHE) > _ESTIMATE_CACHE_MAX:
        _ESTIMATE_CACHE.popitem(last=False)
    return tokens


def _is_settled(message: Message) -> bool:
    """Cacheability gate (``isEstimateCacheable``): non-assistant messages
    are immutable once appended; assistants only count as settled once the
    loop finalization has stored real ``usage`` and a terminal ``stop_reason``
    (not ``None`` / ``"aborted"`` / ``"error"``)."""
    if message.role != "assistant":
        return True
    return message.usage is not None and message.stop_reason not in (None, "aborted", "error")


def _compute_tokens(message: Message) -> int:
    """Uncached token computation for a single message."""
    total = 0
    text_parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            text_parts.append(block.text)
        else:
            total += IMAGE_TOKEN_ESTIMATE

    if text_parts:
        total += _encode_len("\n".join(text_parts))

    if message.tool_calls:
        # Tool-call scaffolding + arguments ride along in the wire payload.
        for call in message.tool_calls:
            args = call.raw_arguments or json.dumps(call.arguments, sort_keys=True)
            total += _encode_len(call.name + args)
    return total


def estimate_messages_tokens(messages: Sequence[Message]) -> int:
    """Sum of per-message estimates — the local context-size estimator."""
    return sum(estimate_tokens(m) for m in messages)


def messages_tokens_upper_bound(messages: Sequence[Message]) -> int:
    """A cheap value that is NEVER below :func:`_compute_tokens`.

    Exists so a caller can prove "nowhere near the compaction threshold"
    without touching tiktoken. The first real estimate in a process loads the
    cl100k_base BPE table, which costs ~84 ms and ~43.6 MB RSS (measured with
    scripts/bench_base_overhead.py) — the single largest item in the peak RSS
    of a short session, spent so a threshold check on a few thousand tokens can
    return False.

    The bound is rigorous, not heuristic, which is what lets a caller
    substitute it into a monotonic ``context_tokens > threshold`` test:

    * cl100k_base is a byte-level BPE, so every token consumes at least one
      UTF-8 byte and ``tokens <= utf8_bytes`` always holds. The chars/4
      fallback used when tiktoken is missing is smaller still.
    * ``str.isascii()`` is an O(1) flag read on CPython, so the common
      all-ASCII block gets the tight bound ``len(text)`` for free; anything
      else uses ``4 * len(text)``, the maximum UTF-8 width per code point.
      Neither path encodes the string, so nothing is allocated.
    * Images and tool calls are counted exactly as :func:`_compute_tokens`
      counts them, so the bound cannot slip under the real estimate when a
      history is mostly images.

    PRECONDITION for the claim about :func:`estimate_messages_tokens`. The
    bound dominates :func:`_compute_tokens` for ANY input, unconditionally.
    It therefore dominates :func:`estimate_messages_tokens` only while this
    module's invalidation contract holds, because that path serves
    ``_ESTIMATE_CACHE`` entries keyed on ``message.id``: a message mutated in
    place without :func:`invalidate_message_cache` keeps the estimate of its
    OLD content, which the bound on the NEW content has no reason to exceed.
    Blanking a settled 1501-token message to ``"[pruned]"`` without
    invalidating yields ``est=1501`` against ``bound=8``; invalidating first
    yields ``est=4`` against the same bound. The failure is
    silent and one-directional: a stale LARGER cached estimate makes the
    bound's early return claim "nowhere near the threshold" for a session that
    is over it, so compaction never fires. All three in-place mutation owners
    (``pruning._blank``, loop finalize, and streaming deltas — never cached
    while unsettled) invalidate today; that is what this bound rests on.

    Deliberately NOT memoized, but not because it is cheap in relative terms —
    on a 400-message tool-heavy history it is SLOWER than the cached estimate,
    measured on this repo (Python 3.13, M3 Max): 0.107 ms vs 0.050 ms when
    every call carries ``raw_arguments`` (~2x), and 3.2 ms vs 0.050 ms when it
    does not (~60x), because the ``json.dumps`` fallback re-serializes every
    tool call in the history on every turn. The reason it stays unmemoized is
    that single-digit milliseconds per turn is nothing against the ~84 ms and
    ~43.6 MB the tokenizer costs once, while a second cache keyed on message
    id would be one more structure every mutation owner must invalidate — a
    new way to produce exactly the stale-value failure described above.

    Scale note: the bound is loose, so the early return only covers the low
    end of the range. Measured ratios of bound to exact estimate: ~3.5-4.5x on
    ASCII prose, chat text and source code (cl100k averages ~4 bytes/token and
    the bound charges one token per byte), ~3.8x on pure CJK, but ~18x on
    ASCII text sprinkled with non-ASCII, where a single non-ASCII character
    makes the whole block take the 4x-per-code-point branch. So the tokenizer
    is skipped only while real context is roughly under ``threshold / 4`` and
    possibly much less; past that the caller falls through to the real
    estimator and loads tiktoken anyway. The substitution buys the cheap
    common case, not the whole range.
    """
    total = 0
    for message in messages:
        text_blocks = 0
        for block in message.content:
            if isinstance(block, TextContent):
                text = block.text
                total += len(text) if text.isascii() else 4 * len(text)
                text_blocks += 1
            else:
                total += IMAGE_TOKEN_ESTIMATE
        # _compute_tokens encodes the text blocks as ONE "\n"-joined string, so
        # the separators are inside what it tokenizes. Charging them here keeps
        # the bound above the estimate for a many-block message whose parts are
        # each a single token.
        total += max(0, text_blocks - 1)
        for call in message.tool_calls or ():
            args = call.raw_arguments or json.dumps(call.arguments, sort_keys=True)
            text = call.name + args
            total += len(text) if text.isascii() else 4 * len(text)
    return total


def invalidate_message_cache(message: Message) -> None:
    """Drop the cached estimate for ``message`` and notify subscribers.

    MUST be called after any in-place mutation of a message (pruning blanks,
    streaming finalize). Idempotent; safe for never-cached messages.
    """
    _ESTIMATE_CACHE.pop(message.id, None)
    for invalidator in list(_INVALIDATORS.values()):
        try:
            invalidator(message)
        except Exception:  # noqa: BLE001 - a subscriber must not break pruning
            logger.exception("message-cache invalidator failed")


def register_invalidator(invalidator: Callable[[Message], None]) -> Callable[[], None]:
    """Subscribe ``invalidator(message)`` to every cache invalidation.

    Returns an unsubscribe callable. For cross-package consumers that keep
    their own derived caches (e.g. provider payload caches) coherent with the
    token estimates.
    """
    _INVALIDATORS[id(invalidator)] = invalidator

    def _unsubscribe() -> None:
        _INVALIDATORS.pop(id(invalidator), None)

    return _unsubscribe


def clear_estimate_cache() -> None:
    """Wipe the whole estimate cache (tests, memory pressure)."""
    _ESTIMATE_CACHE.clear()
