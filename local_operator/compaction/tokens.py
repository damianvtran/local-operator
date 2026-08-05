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


def _encode_len(text: str) -> int:
    """Token count of ``text`` via cl100k_base, or the chars/4 fallback."""
    if not text:
        return 0
    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text))  # type: ignore[attr-defined]
    return len(text) // _CHARS_PER_TOKEN_FALLBACK


def count_text_tokens(text: str) -> int:
    """Public token count for a plain string.

    Callers outside compaction (the HTTP API's token stats, for example) need
    a count without reaching for tiktoken themselves. Going through here means
    they inherit the degradation ladder — exact counts when the ``tokenizer``
    extra is installed, the chars/4 estimate otherwise — instead of raising
    ImportError on an install that never asked for it.
    """
    return _encode_len(text)


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
        ids = encoding.encode(text)  # type: ignore[attr-defined]
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
