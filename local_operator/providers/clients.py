"""httpx wire clients streaming provider SSE into harness ``StreamEvent``s.

Four clients over the harness contract:

- :class:`OpenAICompatClient` — ``{base}/chat/completions`` (covers openai,
  openrouter, deepseek, kimi, alibaba, mistral, xai, ollama, radient).
- :class:`AnthropicClient` — ``/v1/messages`` with cache-control breakpoints
  on system blocks.
- :class:`GoogleClient` — ``generateContent`` / ``streamGenerateContent``
  (minimal).
- :class:`MockClient`` — deterministic canned events for ``--hosting test``.

All accept an injected ``httpx.AsyncClient`` so tests can use
``httpx.MockTransport`` without touching the network. Error mapping raises
:class:`~local_operator.providers.failover.ProviderError` with status/
retryable/auth flags for the failover layer.
"""

from __future__ import annotations

import asyncio
import email.utils
import json
import logging
import math
import re
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import timezone
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx

from local_operator.compaction.thresholds import (
    CompactionSettings,
    resolve_threshold_percent,
)
from local_operator.compaction.tokens import estimate_messages_tokens
from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    ImageContent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsageEvent,
    TextContent,
    ToolCall,
    Usage,
)
from local_operator.providers.failover import ProviderError

if TYPE_CHECKING:
    from local_operator.providers.auth_store import OAuthAccess


#: "no payload supplied" — distinct from ``None``, which is the legitimate
#: result of a non-JSON body. Lets the two extractors share one parse when the
#: caller has it and still stand alone when it does not.
_UNSET: Any = object()

#: Config/transport problems go to the LOG, never the terminal: this module runs
#: under a full-screen TUI that owns stderr.
logger = logging.getLogger("local_operator.providers.clients")


@runtime_checkable
class WireClient(Protocol):
    """The one method the harness needs from a provider client."""

    def stream(
        self,
        request: ChatRequest,
        api_key: str | None,
        oauth_access: "OAuthAccess | None" = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one completion. Raises :class:`ProviderError` on failure.

        ``oauth_access`` carries the resolved credential record (kind,
        account/org identity) so OAuth bearers can take provider-specific
        headers/routes that a bare API key must not. Declared without ``async``
        because implementations are async generators: callers get the iterator
        from the bare call and drive it with ``async for``, never ``await``.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def openrouter_attribution_headers() -> dict[str, str]:
    """Attribution headers identifying Local Operator to OpenRouter.

    OpenRouter uses these headers to credit requests to Local Operator in app
    rankings and public showcase discovery. Both `X-Title` (backward-compatible
    alias) and `X-OpenRouter-Title` (preferred modern header) are sent along with
    the canonical referer and marketplace category tags.
    """
    return {
        "HTTP-Referer": "https://local-operator.com",
        "X-OpenRouter-Title": "Local Operator",
        "X-Title": "Local Operator",
        "X-OpenRouter-Categories": "cli-agent,personal-agent",
    }


def _error_payload(response: httpx.Response) -> Any:
    """The parsed error body, or ``None`` when it is not JSON.

    A single-element LIST is unwrapped: google's ``streamGenerateContent``
    answers a pre-stream failure with ``[{"error": {...}}]``, and the mapping-only
    extractor read straight past it to ``response.text``.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if isinstance(payload, list) and len(payload) == 1:
        return payload[0]
    return payload


def _first_text(*candidates: Any) -> str:
    """The first candidate that is a non-blank string."""
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _usage_token(raw_usage: Mapping[str, Any], *names: str) -> int:
    """Return the first provider token counter present under ``names``.

    OpenAI-compatible describes a transport, not a shared usage schema. Kimi and
    DeepSeek put cache hits directly on ``usage`` while OpenAI, xAI, Mistral,
    Z.AI, Alibaba, and OpenRouter use ``prompt_tokens_details.cached_tokens``.
    Normalizing the aliases at the wire boundary keeps analytics and pricing from
    silently charging a cache hit at the full input rate.
    """
    for name in names:
        value = raw_usage.get(name)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                # A malformed optional detail must not abort an otherwise valid
                # stream; treating it as absent preserves the authoritative totals.
                continue
    return 0


def _compat_cache_usage(raw_usage: Mapping[str, Any]) -> tuple[int, int]:
    """Normalize cache read/write counters across OpenAI-compatible providers."""
    details = raw_usage.get("prompt_tokens_details") or {}
    if not isinstance(details, Mapping):
        details = {}
    if "cached_tokens" in details:
        # Presence wins even for zero: a standard field explicitly reporting a
        # miss is more authoritative than a stray provider compatibility alias.
        cache_read = _usage_token(details, "cached_tokens")
    else:
        # Kimi's documented field is ``cached_tokens``; DeepSeek's is
        # ``prompt_cache_hit_tokens``. Both are subsets of ``prompt_tokens``.
        cache_read = _usage_token(raw_usage, "cached_tokens", "prompt_cache_hit_tokens")
    cache_write = _usage_token(details, "cache_write_tokens")
    return cache_read, cache_write


def _usd_cost(raw_usage: Mapping[str, Any] | None) -> float | None:
    """A provider's own precomputed dollar cost for one request, or ``None``.

    Some providers (OpenRouter's ``usage.cost``) bill the request themselves and
    return the amount they charged. That number is authoritative in a way no
    token×rate reconstruction can be: it already reflects the per-route price the
    request actually landed on, reasoning-token splits, cache discounts, and any
    time- or value-banded overrides — none of which a single flat table row can
    express. Return it as ``None`` when absent, so a caller can tell "the
    provider didn't say" apart from a real ``0.0`` (billed as free).

    Defensive about shape: the field is a JSON number, but a provider that spells
    it as a numeric string must not turn into an exception that aborts a stream.
    """
    if not isinstance(raw_usage, Mapping):
        return None
    value = raw_usage.get("cost")
    if value is None:
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cost) or cost < 0:
        # A negative bill is malformed provider data, not a credit to hand back,
        # and a non-finite one (``json.loads`` parses the non-standard literals
        # ``Infinity``/``NaN`` by default, so ``inf`` is wire-reachable) would
        # poison every summed total forever — ``inf + x == inf`` never recovers.
        # Both fall through to ``None`` so the token×rate estimate answers.
        return None
    return cost


#: Longest provider message carried into an error frame. Every branch of the
#: cascade below is bounded by it: the frame is one wrapped notice line in a
#: terminal, and a provider that answers with a 3 KB error object must not spend
#: the transcript on it.
MAX_ERROR_MESSAGE_CHARS = 500


def _extract_error_message(response: httpx.Response, payload: Any = _UNSET) -> str:
    """The provider's OWN words about the failure, from whichever slot it used.

    A cascade rather than one lookup, because the shapes genuinely differ and
    the message is the useful half of the frame — it says WHICH limit and WHEN
    it resets. Covered here:

    - ``{"error": {"message": ...}}`` — openai, anthropic, google.
    - ``{"error": "..."}`` — ollama and several openai-compatible servers.
    - ``{"message": ...}`` / ``{"detail": ...}`` — gateways and FastAPI-shaped
      proxies (litellm, vLLM).
    - ``{"error": {"metadata": {"raw": "<json>"}}}`` — openrouter puts a bare
      ``"Provider returned error"`` in ``message`` and the UPSTREAM provider's
      real text, JSON-encoded, in ``metadata.raw``; both are kept, in that
      order, because the raw part is the one that names the limit.
    - anything else — the raw body, capped.

    ``{"error": {"message": ""}}`` falls THROUGH to the next slot: the previous
    ``error.get("message", error)`` treated a present-but-empty key as an
    answer, so a provider that sent the field blank produced an error that
    printed nothing at all.

    Nothing here ever renders a Python ``repr``. An ``error`` object with no
    readable text falls through to the raw BODY — the real wire bytes, capped —
    and a body that is empty too returns ``""`` so that ``ProviderError``'s own
    floor speaks instead. ``str(error)`` used to stand in that slot and put
    ``{'message': ''}`` in the frame, which is both uglier than the status phrase
    and, uncapped, unbounded.
    """
    if payload is _UNSET:
        payload = _error_payload(response)
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            message = _attributed_relay_message(
                _first_text(error.get("message"), error.get("detail"), error.get("msg")), error
            )
            upstream = _openrouter_upstream_text(error)
            if message and upstream and upstream not in message:
                return _capped(f"{message}: {upstream}")
            resolved = message or upstream or _first_text(error.get("status"), error.get("code"))
            if resolved:
                return _capped(resolved)
        else:
            direct = _first_text(error, payload.get("message"), payload.get("detail"))
            if direct:
                return _capped(direct)
    return _capped(response.text)


def _capped(text: str) -> str:
    return text.strip()[:MAX_ERROR_MESSAGE_CHARS].strip()


def _attributed_relay_message(message: str, error: Mapping[str, Any]) -> str:
    """Name the ORIGIN provider in an aggregator's generic relay message.

    "Provider returned error" is the least useful sentence in any frame: it
    says a provider failed without saying WHICH, and the reader is already
    looking at an aggregator, so "provider" is ambiguous between the gateway
    and the model host. ``metadata.provider_name`` carries the answer, so the
    frame reads ``Meta returned error: ...`` instead — which is the difference
    between a user suspecting their own model id and knowing the upstream host
    is unwell.

    Only the exact generic phrase is rewritten. A relay message that already
    says something specific is left alone, and a body with no
    ``provider_name`` keeps the original wording rather than inventing an
    attribution the wire did not supply.
    """
    if message.strip().lower() != _OPAQUE_AGGREGATOR_MESSAGE:
        return message
    metadata = error.get("metadata")
    if not isinstance(metadata, Mapping):
        return message
    provider = _first_text(metadata.get("provider_name"))
    return f"{provider} returned error" if provider else message


def _openrouter_upstream_text(error: Mapping[str, Any]) -> str:
    """The upstream provider's message out of openrouter's ``metadata.raw``.

    ``raw`` is a JSON *string* holding the origin provider's own error body, so
    it is re-parsed one level. Anything unexpected degrades to the raw string
    itself, which still says more than "Provider returned error".
    """
    metadata = error.get("metadata")
    if not isinstance(metadata, Mapping):
        return ""
    raw = metadata.get("raw")
    if not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        inner = json.loads(raw)
    except ValueError:
        return raw.strip()[:500]
    if isinstance(inner, Mapping):
        nested = inner.get("error")
        if isinstance(nested, Mapping):
            return _first_text(nested.get("message")) or raw.strip()[:500]
        return _first_text(inner.get("message"), nested) or raw.strip()[:500]
    return raw.strip()[:500]


#: Aggregators answer an UPSTREAM provider failure with an HTTP 400 whose body
#: names nothing: the outer message is exactly "Provider returned error" and
#: ``metadata.raw`` holds a bare sentinel instead of the origin provider's
#: diagnostics. Session e13d092c093c recorded this shape intermittently on a
#: request that live probes at ~750k tokens answered 200 seconds later — an
#: upstream blip wearing a 400. Compared case-insensitively because the casing
#: on the wire ("Provider returned error", raw "ERROR") is the aggregator's,
#: not part of the signal.
_OPAQUE_AGGREGATOR_MESSAGE = "provider returned error"

#: Raw bodies short enough to prove nobody tried to say anything. A real
#: upstream body — "context length exceeded", a JSON error object — is
#: actionable and must keep its ``request`` classification.
_OPAQUE_RAW_SENTINELS = frozenset({"error"})

#: Statuses an aggregator will RELAY from an origin provider that are otherwise
#: read as "the request itself was refused". 401/403 stay out: those describe
#: the caller's credential at the aggregator and must keep reaching credential
#: rotation rather than being retried as weather. 429 is already quota and 5xx
#: is already transient, so neither needs this path.
#:
#: 404 is the member that matters and the reason this set exists. OpenRouter
#: answers a model id it does not know with a FLAT 400 that names the slug
#: ("meta/muse-spark-9.9 is not a valid model ID"), and a routing refusal it
#: decided itself with a FLAT 404 whose message names the routing preference
#: ("No allowed providers are available for the selected model...") — both are
#: the aggregator's own words, and neither carries ``metadata.raw``. Verified
#: against the live API on 2026-09-04. A 404 that arrives WRAPPED in the relay
#: envelope is therefore never "the client asked for a model that does not
#: exist": it is the ORIGIN provider's own 404 forwarded verbatim, which for a
#: single-endpoint model is a transient failure to resolve its own snapshot.
_RELAYED_UPSTREAM_STATUSES = frozenset({400, 404})

#: Upstream wordings that stay ``request`` even when relayed, because they
#: describe the BYTES we sent: they will fail identically on a retry and on
#: every fallback target, so retrying them buys minutes of backoff and the
#: same answer. Deliberately NARROW and matched as phrases rather than single
#: words — a loose marker ("invalid", "error") would drag genuine upstream
#: weather back into ``request`` and re-create the dead turn this predicate
#: exists to prevent. When a wording is ambiguous the tie is broken toward
#: transient, because the cost of that mistake is bounded retries while the
#: cost of the opposite is a killed turn.
_DETERMINISTIC_UPSTREAM_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "is not a valid model",
    "max_tokens",
    "max_output_tokens",
)

#: How many matched quote pairs :func:`_strip_quote_pair` will peel. Real
#: bodies use at most one or two layers; the cap exists so a body made of
#: nothing but quotes cannot spend quadratic CPU (each peel copies the string)
#: on a request that is already failing.
_MAX_QUOTE_PEELS = 4


def _strip_quote_pair(text: str) -> str:
    """Peel MATCHED surrounding quote pairs off ``text``.

    ``str.strip('"\'')`` would take any leading/trailing run of either
    character, so unbalanced junk like ``\'\'\'ERROR""`` would reduce to the
    sentinel and be treated as no-information. That only ever widens the match,
    but the widening should be a decision rather than an accident of the API —
    so pairs are peeled one at a time and an unbalanced body keeps its quotes
    and stays ``request``.

    The peel LOOPS rather than running once to cover a value that arrives
    already wrapped more than once — e.g. ``'"ERROR"'`` — which costs nothing
    to absorb and keeps the predicate insensitive to how many layers of
    quoting an aggregator applied.

    A JSON-encoded ``raw`` whose inner value is itself a string (the
    double-encoding case) DOES reach this function. :func:`_openrouter_upstream_text`
    ``json.loads`` it, then — because the inner is a ``str``, not a
    ``Mapping`` — returns the original unparsed ``raw`` rather than the
    decoded inner. The peel then takes matched quote pairs off that original.
    (A ``Mapping`` inner is unwrapped to its ``message`` instead, and never
    gets here.) The earlier claim that double-encoding is "parsed away before
    this sees it" was wrong; the behaviour was always this, only the stated
    reason was not.

    ``_MAX_QUOTE_PEELS`` bounds the loop because slicing copies the string on
    every pass: an adversarial body of a few MB of quotes would otherwise cost
    quadratic time on a request that is already failing. Nothing legitimate
    nests quotes more than a couple of layers deep, so a low cap costs real
    traffic nothing and denies a hostile aggregator the CPU.
    """
    for _ in range(_MAX_QUOTE_PEELS):
        if not (len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'"):
            break
        text = text[1:-1].strip()
    return text


def _relayed_upstream_failure(status: int | None, error: Mapping[str, Any]) -> bool:
    """Is this 4xx an UPSTREAM failure the aggregator merely relayed?

    OpenRouter forwards an origin provider's failure as ``{"message":
    "Provider returned error", "metadata": {"raw": ...}}``. The presence of
    that envelope is the load-bearing signal: the aggregator's OWN refusals
    (an unknown model slug, a routing preference it cannot satisfy) are flat
    bodies that name the problem and carry no ``metadata.raw`` at all. So a
    wrapped 4xx describes something that happened at the ORIGIN, on the far
    side of a network hop the caller cannot see or influence.

    Two shapes take this path, for the same reason — the status line is the
    aggregator's relay choice rather than evidence our request was wrong:

    - ``raw`` is a bare sentinel (observed: ``"ERROR"``, provider "Stealth"),
      so the body says nothing a caller could act on.
    - ``raw`` holds real upstream prose that nevertheless describes the
      PROVIDER's state rather than our bytes. Session 2be018a98088 recorded
      ``"The requested model was not found."`` relayed under a 404 from Meta
      for ``meta/muse-spark-1.3`` — twice in 75 seconds, with six successful
      calls on the identical model id in between, on a model whose single
      endpoint reported 99.98% uptime. Read literally the text looks like an
      answer, which is why the old wording-only predicate missed it; but a
      model id that works either side of the failure cannot be the cause, and
      classifying it ``request`` killed the turn with no retry and no
      failover.

    :data:`_DETERMINISTIC_UPSTREAM_MARKERS` carves back out the relayed
    wordings that genuinely describe our bytes (context length, a malformed
    parameter): those fail identically on every retry and every fallback, so
    they keep ``request`` and surface immediately.

    Two deliberate widenings, both toward "treat no-information as transient":

    - The outer message is matched after trimming and case-folding, so
      ``"Provider returned error "`` matches too. The casing and padding on the
      wire are the aggregator's formatting, not part of the signal.
    - :func:`_openrouter_upstream_text` unwraps a nested ``error`` object, so
      ``{"error": {"message": "ERROR"}}`` resolves to the same bare sentinel and
      also matches. A body that structurally tried to say something but whose
      content is still the word "error" carries no more actionable information
      than the flat sentinel. This is the case most likely to shadow a real
      upstream error whose message happens to be that single word; the cost of
      that collision is bounded retries, and the cost of the opposite mistake
      is a dead turn.

    The price is latency, not correctness, and it is not small: a genuinely
    broken request now saturates the driver's server-fault budget (12 requests
    per target) before surfacing. Measured against the default 500ms base delay
    and the 8s backoff cap, that is 64-76s of sleep on a single target and
    roughly 4-5 minutes across a 3-target fallback chain. A user hitting a
    PERSISTENT relayed 4xx therefore waits minutes for a failure that
    previously surfaced immediately. That is the accepted trade: this shape is
    by construction unattributable, so the alternative is killing a turn that
    rotation or a fallback could have served — but it is a real cost, not the
    rounding error an earlier draft of this docstring implied.

    The marker carve-out is what keeps that cost bounded to the cases that
    deserve it: the failures a user can actually fix by changing the request
    still fail fast, and only the ones no retry could have prevented pay for
    the cascade.
    """
    if status is not None and status not in _RELAYED_UPSTREAM_STATUSES:
        return False
    message = _first_text(error.get("message"))
    if message.lower() != _OPAQUE_AGGREGATOR_MESSAGE:
        return False
    metadata = error.get("metadata")
    if not isinstance(metadata, Mapping) or not _first_text(metadata.get("raw")):
        # The relay envelope is the whole signal. "Provider returned error"
        # with nothing wrapped inside it is not evidence of an upstream hop,
        # so it keeps whatever its status already meant.
        return False
    upstream = _openrouter_upstream_text(error)
    stripped = _strip_quote_pair(upstream.strip()).lower()
    if stripped in _OPAQUE_RAW_SENTINELS:
        return True
    return not any(marker in stripped for marker in _DETERMINISTIC_UPSTREAM_MARKERS)


def _compat_stream_error(chunk: Mapping[str, Any]) -> ProviderError:
    """An in-band mid-stream failure on an OpenAI-compatible stream.

    Once a gateway has committed HTTP 200 it can no longer signal an upstream
    failure via the status line, so OpenRouter delivers it INSIDE the stream:
    a ``chat.completion.chunk`` carrying a top-level ``error`` object (``code``,
    ``message``, ``metadata.error_type``) alongside ``finish_reason: "error"``.
    The chunk parser used to read only ``choices`` and ``usage``, so the error
    object was dropped and the terminal finish reason surfaced later as a
    wordless ``stop_reason="error"`` — the turn died as a silent interruption:
    no incident for the model, no retry, no failover, no credential rotation.
    Raising here hands the failure to the failover driver, which names it,
    journals it, and retries or rotates while the budget lasts.

    Simpler compatible servers send the error as a bare string instead of an
    object. The message cascade follows :func:`_extract_error_message`'s slots
    (message, detail, msg, then the ``metadata.raw`` upstream text) but not its
    floors: that function's last resort is ``error.status``/``error.code`` and
    its response-body fallback, neither of which belongs here — ``code`` is
    already consumed as the status, and the body is a live stream that must
    not be read. ``error_type`` plus a substantive default is the better
    floor. The composed message is capped like every other error frame.
    """
    error = chunk.get("error")
    status: int | None = None
    error_type = ""
    relayed_upstream = False
    if isinstance(error, Mapping):
        message = _attributed_relay_message(
            _first_text(error.get("message"), error.get("detail"), error.get("msg")), error
        )
        upstream = _openrouter_upstream_text(error)
        if message and upstream and upstream not in message:
            message = f"{message}: {upstream}"
        else:
            message = message or upstream
        code = error.get("code")
        if isinstance(code, int):
            status = code
        elif isinstance(code, str) and code.isdigit():
            status = int(code)
        metadata = error.get("metadata")
        if isinstance(metadata, Mapping):
            error_type = str(metadata.get("error_type") or "")
        # The SAME relay body arrives on this channel whenever the gateway had
        # already committed HTTP 200 before the upstream failed, so it has to
        # classify the same way here as in `raise_for_status` -- a turn must
        # not die on the relay channel the aggregator happened to pick. Judged
        # on the chunk's `code` rather than a status line, because in-band that
        # is where the status lives.
        relayed_upstream = _relayed_upstream_failure(status, error)
    else:
        message = _first_text(error)
    if not message:
        message = error_type or "provider ended the stream with an error"
    elif error_type and error_type not in message:
        message = f"{error_type}: {message}"
    return ProviderError(
        status,
        _capped(message),
        retryable=(relayed_upstream or status is None or status == 429 or status >= 500),
        auth_error=status in (401, 403),
    )


#: Ceiling on any advertised wait. A ``Retry-After`` is provider-supplied and
#: reaches SQLite: a usage-limit failure feeds ``retry_after_ms_from_error``
#: into ``AuthStore.rotate_sibling`` → ``block_credential(block_ms=...)``, which
#: floors the value but has no ceiling of its own. A single ``retryDelay:
#: "99999999s"`` would therefore write a 27,777-hour block against a working
#: credential and print ``retry in 27777h46m`` at the user. Past a day the number
#: is not a wait any interactive session can act on, so it is clamped and the
#: original is logged.
MAX_RETRY_AFTER_MS = 24 * 60 * 60 * 1000


def _clamp_retry_after(delay_ms: int, source: str) -> int:
    if delay_ms <= MAX_RETRY_AFTER_MS:
        return delay_ms
    logger.warning(
        "provider advertised a %d ms wait via %s; clamped to %d ms",
        delay_ms,
        source,
        MAX_RETRY_AFTER_MS,
    )
    return MAX_RETRY_AFTER_MS


def _parse_retry_after(response: httpx.Response, payload: Any = _UNSET) -> int | None:
    """How long to wait, as milliseconds, from the header OR the body.

    ``Retry-After`` (seconds or HTTP-date) first. Google is the reason the body
    is consulted too: gemini sends NO ``Retry-After`` on a quota 429 and puts
    the delay in ``error.details[].retryDelay`` as ``"41s"``. That figure is the
    single most actionable fact in a rate-limit error, and dropping it left the
    frame saying only that the limit was hit.

    A NON-POSITIVE header falls through to the body rather than winning. Zero is
    not an answer to "how long": it renders as no wait at all (``__str__`` tests
    the value for truth) and, worse, ``_same_credential_retry_allowed`` reads it
    as a short throttle and grants an immediate same-key retry of a quota error.
    A gateway that sends ``Retry-After: 0`` alongside google's real ``retryDelay``
    should not erase it.
    """
    header = response.headers.get("retry-after")
    if header is not None:
        parsed = _retry_after_from_header(header)
        if parsed:
            return _clamp_retry_after(parsed, "the Retry-After header")
    if payload is _UNSET:
        payload = _error_payload(response)
    delay = _retry_delay_from_payload(payload)
    return None if delay is None else _clamp_retry_after(delay, "the response body")


def _retry_after_from_header(header: str) -> int | None:
    try:
        # OverflowError as well as ValueError: `float("1e400")` is `inf`, and
        # `int(inf * 1000)` raises rather than returning a number. Unhandled it
        # escaped `raise_for_status` entirely, and in `ApiEmbedder._fetch` it
        # escaped the graceful-degradation handlers too.
        return max(0, int(float(header) * 1000))
    except (ValueError, OverflowError):
        pass
    try:
        when = email.utils.parsedate_to_datetime(header)
    except (TypeError, ValueError, OverflowError):
        return None
    if when.tzinfo is None:
        # HTTP dates are GMT; parsedate yields a naive datetime when the
        # zone is absent.
        when = when.replace(tzinfo=timezone.utc)
    delta_ms = int(when.timestamp() * 1000) - int(time.time() * 1000)
    return max(0, delta_ms)


#: Google's ``RetryInfo.retryDelay``: a protobuf Duration rendered as seconds
#: with an ``s`` suffix (``"41s"``, ``"1.5s"``). Bounded to 12 digits so a
#: pathological body cannot build an enormous int before the clamp sees it.
_RETRY_DELAY_RE = re.compile(r"^\s*(\d{1,12}(?:\.\d{1,6})?)s\s*$")


def _retry_delay_from_payload(payload: Any) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    details = error.get("details") if isinstance(error, Mapping) else None
    if not isinstance(details, Sequence) or isinstance(details, (str, bytes)):
        return None
    for detail in details:
        if not isinstance(detail, Mapping):
            continue
        match = _RETRY_DELAY_RE.match(str(detail.get("retryDelay", "")))
        if match:
            return max(0, int(float(match.group(1)) * 1000))
    return None


def raise_for_status(response: httpx.Response) -> None:
    """Map HTTP errors onto ProviderError with failover-relevant flags.

    The body is parsed ONCE and handed to both extractors: the message and the
    retry delay can live in the same payload (google puts them there), and
    re-reading ``response.json()`` per lookup parsed it three times.
    """
    status = response.status_code
    if status < 400:
        return
    payload = _error_payload(response)
    auth_error = status in (401, 403)
    # 408/504 are the two "ran out of time" statuses; 429 and 5xx are the
    # classic retryables. Everything else in 4xx is an answer, not a blip.
    retryable = status == 429 or status >= 500 or status in (408, 504)
    # ...except a 4xx the aggregator RELAYED from an origin provider (the
    # `metadata.raw` envelope). That is an upstream failure wearing a client
    # status, not a request the provider read and refused, so it must reach
    # the failover cascade. See :func:`_relayed_upstream_failure` for the
    # shapes and the recorded evidence.
    error = payload.get("error") if isinstance(payload, Mapping) else None
    if isinstance(error, Mapping) and _relayed_upstream_failure(status, error):
        retryable = True
    raise ProviderError(
        status,
        _extract_error_message(response, payload),
        retryable=retryable,
        retry_after_ms=_parse_retry_after(response, payload),
        auth_error=auth_error,
    )


def _iter_sse_lines(response: httpx.Response) -> AsyncIterator[str]:
    """Yield decoded ``data:`` payloads from an SSE byte stream.

    Once bytes have started arriving, a silence longer than
    ``STREAM_READ_TIMEOUT_S`` is treated as a stalled stream. The distinction
    this makes — and that httpx's own read timeout cannot — is between a model
    that has not started answering (legitimately minutes of silence for a
    reasoning model) and a connection that accepted the request and died
    (indistinguishable from thinking, for the whole request budget, with the UI
    spinning).
    """

    async def _gen() -> AsyncIterator[str]:
        buffer = ""
        async for chunk in _guarded_chunks(response):
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data:
                        yield data

    return _gen()


async def _guarded_chunks(response: httpx.Response) -> AsyncIterator[str]:
    """``response.aiter_text()`` with a stall watchdog after the first chunk."""
    iterator = response.aiter_text().__aiter__()
    started = False
    while True:
        try:
            if started:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=STREAM_READ_TIMEOUT_S)
            else:
                chunk = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise httpx.ReadTimeout(
                f"stream stalled: no data for {STREAM_READ_TIMEOUT_S:.0f}s",
                request=response.request,
            ) from exc
        started = True
        yield chunk


def _sampling_params(request: ChatRequest, *, top_p_key: str = "top_p") -> dict[str, float]:
    """The sampling pair for this request, or nothing when the model rejects it.

    Empty dict rather than ``{"temperature": None}``: httpx serialises ``None``
    as JSON ``null``, and a provider that rejects the key rejects it just as
    hard with a null value — the whole point is that the key must not appear.

    An explicit ``request.temperature`` loses to the capability on purpose. A
    caller-set value the model cannot accept is a turn that 400s, not a
    preference worth honouring, and every caller inherits the spec's defaults
    anyway so "explicit" here rarely means "deliberate".

    ``top_p_key`` exists only because Gemini spells it ``topP``; the capability
    itself is provider-independent and lives on the spec.

    Each knob is resolved INDEPENDENTLY, and a ``None`` on the spec means OMIT
    that key so the vendor's own default applies (see ``_SAMPLING_POLICY``).
    Vendors diverge on the two constantly — Qwen documents temperature 0.7 with
    top_p 0.8 — so a family may legitimately seed one and omit the other, and
    an all-or-nothing pair would be unable to express that.
    """
    if not request.model.supports_sampling_params:
        return {}
    params: dict[str, float] = {}
    temperature = (
        request.temperature if request.temperature is not None else request.model.temperature
    )
    if temperature is not None:
        params["temperature"] = temperature
    top_p = request.top_p if request.top_p is not None else request.model.top_p
    if top_p is not None:
        params[top_p_key] = top_p
    return params


#: Multiplier applied to a LOCAL token estimate before it is compared against a
#: provider-scale window.
#:
#: ``compaction/tokens.py`` is explicit that its numbers are "a RULER OF ITS OWN,
#: never a prediction of the bill": it counts with ``cl100k_base``, which is
#: OpenAI's tokenizer, so the gap against what a provider actually bills is a
#: per-MODEL property rather than a constant. The window is a PROVIDER figure, so
#: subtracting a raw local estimate from it mixes two rulers — the exact class of
#: bug that module records as having already shipped twice.
#:
#: **This scaling is a safety margin, not a prediction, and it is deliberately
#: confined to sizing the ASK.** Two independent measurements of the same
#: Anthropic ratio disagree by a lot: ``slope_fit.txt`` fits a real session
#: transcript (tool calls, images, cache scaffolding) at p50 1.82-1.96, while QA
#: measured plain conversational text live at 1.21-1.34. Both are probably right
#: about their own content, which is the point — a single number cannot predict
#: this, so it is not asked to. It is rounded UP to 2.0 for the Claude family so
#: the ask stays admissible on the expensive end, and the cost of being wrong on
#: the cheap end is bounded output headroom near the window and nothing below it.
#: Note also that ``ratio p50`` is a median: half the observed requests exceed it
#: by construction, which is another reason to round up rather than to treat the
#: figure as a forecast.
#:
#: What the scaling must NEVER do is decide a refusal — see
#: :func:`_estimated_prompt_tokens`, which returns the unscaled measurement
#: separately for exactly that reason.
#:
#: **Why 1.25, and why not the 2.00 an earlier revision used.** That 2.00 came
#: from reading ``slope_fit.txt``'s Anthropic ``ratio p50`` of 1.82-1.96 as a
#: multiplier. It is not one: those fits carry an INTERCEPT of 24k-50k tokens
#: (``inter=`` in the table), so the median is inflated by a fixed additive term
#: that a pure multiplier then re-applies proportionally to every prompt.
#:
#: A reviewer correctly noted that the same file's ``slope`` column is the fitted
#: coefficient with the intercept already removed, and reads 1.62-1.69 — above
#: this constant. That column is cited here so a future reader does not "correct"
#: this number upward on the strength of it. **It was tested directly and the
#: measurement does not support it**: if a 1.6x coefficient were a per-prompt
#: cost, the observed ratio would climb toward it as prompts grow. Measured live
#: across a 23x range it is flat — 1.174 at 4.5k local tokens, 1.174 at 18k,
#: 1.192 at 53k, 1.201 at 104k. The intercept is per-SESSION scaffolding those
#: fits absorbed, not a term each request pays again.
#:
#: Measured directly instead, live ``prompt_tokens`` over the local estimate on
#: the content agent sessions actually carry:
#:
#: =========================  =======
#: content                      ratio
#: =========================  =======
#: prose / markdown             1.098
#: source code                  1.166
#: log output                   1.167
#: mixed transcript             1.210
#: base64-ish text              1.219
#: =========================  =======
#:
#: QA measured 1.18 independently on its own corpus, and 1.183 as its worst
#: cross-family figure. 1.25 clears every one of those with headroom.
#:
#: **A known gap, stated rather than papered over.** Emoji-interleaved prose
#: measures **1.471**, above this constant. It is not covered, and raising the
#: constant to cover it would be a poor trade: at that ratio the prompt alone
#: crosses the window before any ask is added, so the case belongs to the refusal
#: (which keys on the unscaled measurement) rather than to a multiplier — while
#: every point of extra pessimism shrinks the reply budget of ordinary sessions,
#: which is the silent-truncation failure rounds 1, 3 and 4 each rejected.
#:
#: State it honestly: this is a CHOSEN CONSTANT bounded by measurement, not a
#: fitted parameter. An earlier revision defended 1.75 as the p50 of
#: ``slope_fit.txt``'s unattributed ``None/None`` rows, which claimed more
#: support than one median over an unlabelled bucket gives — and a median is the
#: wrong statistic for a safety margin anyway, since half the population exceeds
#: it by construction (review R18).
#:
#: A SINGLE value, not a per-family table. An earlier revision kept one and keyed
#: it on ``ModelSpec.provider`` — the registry hosting id — so every
#: aggregator-served Claude (``openrouter``, ``radient``, both real registry ids)
#: took a different number than the same model served directly, re-opening the
#: original HTTP 400 on the very provider this bug was reported against (review
#: R9). Measured properly, Anthropic lands where every other family does, so the
#: table decided nothing while still offering a way to get the routing wrong.
#:
#: Cross-family measurements agree: QA measured 1.005-1.183 across families, and
#: ``slope_fit.txt``'s non-Anthropic fits sit at 1.02-1.04. A family that is
#: genuinely more expensive earns a documented exception here, backed by live
#: measurement rather than by a fitted ratio carrying an intercept.
DEFAULT_ESTIMATE_SLOPE = 1.25


def _estimate_slope(model: ModelSpec) -> float:
    """Local-estimate-to-provider ratio for ``model``.

    Kept as a function rather than inlining :data:`DEFAULT_ESTIMATE_SLOPE` at the
    one call site because the ratio IS a per-model property — the value is
    uniform today only because measurement said so. A family that proves more
    expensive gets its exception here, where the routing question (match the
    model, never the provider that serves it — review R9) has already been
    settled, instead of reintroducing a lookup at the call site.
    """
    return DEFAULT_ESTIMATE_SLOPE


#: Characters per token for the system-block and tool-schema term, matching the
#: ratio ``compaction/tokens.py`` uses when tiktoken is absent. Deliberately a
#: local constant rather than an import of that module's private
#: ``_CHARS_PER_TOKEN_FALLBACK``: this is a coarse sizing input for one piece of
#: arithmetic, not a claim to share that module's estimator contract.
_CHARS_PER_TOKEN = 4

#: Tokens held back on top of the prompt estimate, covering what no estimate of
#: OUR message list can see: the provider's own per-request scaffolding (chat
#: templates, injected tool preambles, role framing). Re-derived for the precise
#: estimator — the previous 2048 was chosen against a 3.5-4.5x byte bound that
#: already dwarfed it, so it was inert wherever it was supposed to help (review
#: R4). 4096 is roughly a page of injected scaffolding and is the term that keeps
#: a SMALL prompt against a SMALL window from landing exactly on the boundary.
OUTPUT_CLAMP_SAFETY_MARGIN = 4_096

#: Smallest output ask worth sending on a window large enough to afford it.
#: Below this the reply cannot be a reply: QA measured ``x-ai/grok-4.6`` spending
#: **689 tokens on reasoning alone** at a 512-token cap and emitting zero visible
#: text, because reasoning tokens are billed against this same budget.
#:
#: This is a REFUSAL threshold, not a value that goes on the wire. See
#: :func:`_effective_max_tokens` for why silently sending a doomed cap is the one
#: outcome this must not produce.
#:
#: It is an ABSOLUTE token count, so it cannot be the whole story: on a small
#: window a constant reserve is a large fraction of the model. 4096 + the 4096
#: margin is the ENTIRE 8k window of ``gpt-4`` and ``moonshot-v1-8k``, which is
#: how every request to those models — including a one-token ``"hi"`` — came to
#: be refused. :func:`_output_reserve_tokens` scales it down for exactly that
#: case.
MIN_OUTPUT_TOKENS = 4_096

#: The most aggressive compaction trigger a user can configure
#: (``compaction.threshold_percent`` is a FLOAT setting bounded at 100%, see
#: ``settings_io.py``). :func:`_output_reserve_tokens` sizes itself against THIS
#: rather than the configured value, because its production call site cannot see
#: the configured value and a reserve that is safe at the extreme is safe at
#: every setting below it.
#:
#: Not the 1.0 the setting technically accepts. ``resolve_threshold_tokens``
#: clamps the trigger to ``window - 1``, so at extreme settings compaction fires
#: only when the session is already at the wall and leaves single-digit tokens
#: behind. No reserve can sit above such a trigger and still be a reserve — the
#: ordering would demand a reserve of zero, i.e. no refusal at all — so a strict
#: ordering is arithmetically impossible there, not merely untuned.
#:
#: What the ordering exists to protect is narrower than the ordering itself: the
#: refusal must never pre-empt a compaction pass that could ACTUALLY rescue the
#: turn. Past ~0.9 a pass reclaims less than a usable reply, so there is nothing
#: to pre-empt and the refusal is the correct outcome rather than a wedge. 0.90 is
#: the most aggressive setting at which a pass still frees enough to answer with
#: (20,000 tokens on a 200k window), which makes it the right bound to size
#: against.
MAX_SUPPORTED_THRESHOLD_PERCENT = 0.90


def _output_reserve_tokens(window: int, settings: CompactionSettings | None = None) -> int:
    """Tokens this clamp insists remain for the reply, or it refuses the request.

    The number that decides a refusal has to keep ONE ordering true at EVERY
    window size: **the refusal must never fire where compaction could still have
    rescued the session.** If it fires first, the turn dies non-retryably while
    the session sits below its compaction threshold — and the compaction
    summarizer takes this same code path, so the one remedy the error names
    cannot run either.

    The previous revision reserved a CONSTANT ``MIN_OUTPUT_TOKENS +
    OUTPUT_CLAMP_SAFETY_MARGIN`` (8192) while the compaction trigger is a
    FRACTION of the window. Two different shapes cannot hold a fixed ordering
    across a range, and this one inverted below ~41k: at 32,768 the refusal fired
    at 24,576 against a 26,214 trigger, and at 8,192 it fired unconditionally.
    That is the same wedge twice — which is why the reserve is now expressed in
    the trigger's own shape rather than re-tuned.

    The reserve is therefore the SMALLER of the absolute floor and a fraction of
    the window strictly under the compaction headroom, making the ordering true
    by construction:

        reserve <= window * (1 - trigger_fraction) * SAFETY
        =>  refusal point = window - reserve  >  window * trigger_fraction

    **The fraction is the one the trigger CANNOT exceed, not the configured
    one.** An earlier revision derived it from ``resolve_threshold_percent`` and
    documented that as the reason the ordering holds — but the sole production
    call site has no settings to pass (``ChatRequest`` carries none, and threading
    the session's config through every provider client is a far wider change than
    this fix). The parameter was therefore decorative in production: the reserve
    was pinned to the 0.80 default while the trigger used the user's real
    ``compaction.threshold_percent``, a first-class setting that reaches 1.0, and
    the two re-diverged for anyone who raised it — reproducibly refusing turns
    below the trigger at 0.90 on a 32k model, with ``/compact`` blocked in the
    same band (review R19, QA Q12).

    Deriving from ``MAX_SUPPORTED_THRESHOLD_PERCENT`` closes that by construction:
    a reserve small enough for the most aggressive trigger a user can configure is
    small enough for every less aggressive one, since the refusal point
    ``window - reserve`` only moves further above a trigger that moves down. The
    ordering then holds for EVERY setting without the function needing to observe
    any of them — which is the only way it can be true on a call site that cannot
    supply them.

    ``settings`` is retained for tests that pin the relationship against a
    specific configuration; production correctness must not depend on it.
    """
    if window <= 0:
        return MIN_OUTPUT_TOKENS
    # The configured value only ever makes the trigger EARLIER than this bound,
    # so honouring the maximum covers every configuration including the default.
    percent = MAX_SUPPORTED_THRESHOLD_PERCENT
    if settings is not None:
        percent = max(percent, resolve_threshold_percent(settings))
    # A QUARTER of the post-trigger headroom, not a half. The refusal subtracts
    # BOTH this reserve and a margin clamped to the same value, so a half left
    # `window - 2*reserve == percent * window` — the trigger exactly, with the
    # only separation coming from `int()` truncation (1-2 tokens, review R20).
    # A quarter makes the two subtractions sum to half the headroom, so the
    # designed cushion survives the margin instead of being cancelled by it.
    proportional = int(window * (1.0 - percent) * 0.25)
    return max(1, min(MIN_OUTPUT_TOKENS, proportional))


def _effective_max_tokens(request: ChatRequest) -> int:
    """The output cap to put on the wire, clamped to fit inside the window.

    Providers count ``prompt + max_tokens`` against the context window AT
    ADMISSION, before a single token is generated, so the output reservation is
    not free headroom — it is input capacity spent in advance. A listing that
    advertises a large completion cap therefore silently shrinks the usable
    prompt by exactly that amount.

    That is not hypothetical. OpenRouter advertises ``meta/muse-spark-1.3`` as
    ``context_length: 1048576`` with ``top_provider.max_completion_tokens:
    943718`` (0.9 of the window), which reaches the spec as
    ``max_output_tokens`` and goes out verbatim as ``max_tokens``. The provider
    then admits only ~104,858 tokens of prompt — 10% of a 1M model — and a real
    session died at ~113k input with ``requested about 1057079 tokens (102961 of
    text input, 10400 of tool input, 943718 in the output)``. 82 models in the
    live OpenRouter catalogue advertise exactly this 0.9 ratio (``x-ai/grok-4.6``
    at 500000/450000, ``x-ai/grok-4.20`` at 2000000/1800000), so this is latent
    for a large slice of the catalogue rather than one bad row.

    Compaction cannot rescue it: the trigger is a FRACTION of the window (~838k
    at the default 0.8), far above where the 400 lands, and a compacted prompt
    still carries the same reservation. The clamp has to live here, at body-build
    time, because this is the first point that knows both the window and the
    actual prompt — the spec knows the window and never sees the messages.

    **Sizing the prompt: two rulers, in preference order.** The window is a
    PROVIDER-scale number, so the subtrahend has to be one too. This mirrors
    ``AnthropicClient._cache_ttl_for``, which faces the same problem and resolves
    it the same way:

    1. ``request.context_tokens_hint`` — the provider's OWN count from the
       session's previous call. It is on the right ruler by construction and is
       used unscaled and WHOLE: it already covers the system blocks and tool
       schemas, because ``Usage.context_tokens`` is normalized in this same file
       as ``input + cache_read + cache_write``, i.e. the entire prompt the
       provider read. Adding a locally-estimated prefix on top of it double-counts
       (~21.8k phantom tokens with this repo's default tool set), which is review
       finding R7.
    2. :func:`estimate_messages_tokens` scaled by the model family's measured
       ratio, when no hint exists (a session's first call, a fork, a one-shot
       errand). Only this branch adds the system/tool term, because only here is
       the term genuinely missing.

    An earlier revision used :func:`messages_tokens_upper_bound` on the argument
    that over-estimating is "the safe direction". That reasoning was wrong and is
    recorded here so it is not reintroduced: the bound counts one token per BYTE,
    which is ~4.5x the real count on ASCII and jumps another ~4x the moment a
    single non-ASCII character (a curly apostrophe, an em dash, an accented name)
    flips a block to its ``4 * len`` branch. Subtracting that does not shave the
    ask, it consumes the window several times over — measured, a Claude Sonnet
    200k/64k session at **24% of its window** collapsed from 64000 to the floor
    and returned a real answer truncated mid-sentence with
    ``finish_reason='length'``. Trading a loud HTTP 400 on a handful of models
    for silent truncation on every long session is a worse failure in kind,
    because the user cannot see it happened. The direction-of-error argument
    holds only for a THRESHOLD test that must never read low; here the magnitude
    of the over-estimate is itself the cost.

    System blocks and tool schemas are charged too, not just messages. The 400
    above itemised **10,400 tokens of tool input** separately, and a system
    prompt plus JSON tool schemas is routinely tens of thousands of tokens — a
    term this size is the difference between a clamp that fits and one that
    overflows by exactly the part it forgot to count.

    The clamp only ever LOWERS the ask. That is what preserves
    ``Session.ERRAND_MAX_TOKENS``: a deliberate small ``request.max_tokens``
    (1024 for auto-naming) stays 1024 rather than being raised to fill the
    window. Returns ``0`` when neither the request nor the spec asks for a cap,
    which every call site already spells as "omit the key".

    Raises:
        ProviderError: when the window cannot fund even
            :data:`MIN_OUTPUT_TOKENS` of output. See below for why this refuses
            rather than sending a doomed cap.
    """
    requested = request.max_tokens or request.model.max_output_tokens
    if not requested or requested <= 0:
        # No cap asked for anywhere: the caller omits the key entirely and lets
        # the provider apply its own default. Clamping a value nobody set would
        # turn an absent key into a present one and CAP a model that currently
        # has no ceiling — strictly worse than the status quo.
        return 0

    window = request.model.context_window
    if not window or window <= 0:
        # An unknown window (the -1/0 sentinels the registry still produces for
        # an unlisted model) gives nothing to clamp against. Arithmetic on it
        # would invent a limit from a number that means "no data".
        return requested

    reserve = _output_reserve_tokens(window)
    # The margin is scaled by the same rule and for the same reason: it is an
    # absolute allowance for provider-side scaffolding, and on an 8k window a
    # flat 4096 of it is half the model.
    margin = min(OUTPUT_CLAMP_SAFETY_MARGIN, reserve)

    prompt, measured_prompt = _estimated_prompt_tokens(request)
    available = window - prompt - margin
    measured_available = window - measured_prompt - margin
    if available < reserve <= measured_available:
        # The scaled ask has collapsed below a usable reply while the MEASURED
        # prompt proves the session is not in the state the refusal exists for.
        # Only this narrow case is rescued: when the scaled ask is already
        # healthy it is the slope-protected number and must stand, or the
        # rescue would hand back an ask larger than the safety scaling permits
        # and re-open the overflow (caught by the muse-spark admission test).
        #
        # Restore a PROPORTIONATE ask, not a constant floor. Flooring at
        # ``MIN_OUTPUT_TOKENS`` was round-1's truncation defect returning by
        # another route: on the hint-less path (first call, forks, errands, the
        # compaction summarizer) a Sonnet session at 50% occupancy asked for
        # 4,096 with ~95k tokens of real headroom, and returned an answer cut
        # mid-word with ``finish_reason='length'`` where main completed it. The
        # margin may size the ask DOWN from what the spec wanted; it may not
        # decide the size on its own once the measurement has shown the request
        # is healthy.
        #
        # The restore is bounded by the reserve rather than allowed to consume
        # ``measured_available`` outright. Handing back the full measured
        # headroom would discard the family scaling completely and re-open the
        # very overflow this clamp exists to prevent — the measured figure is a
        # LOCAL count, and a Claude prompt really bills up to ~1.9x it, so an ask
        # sized against it admits a request the provider then rejects. Verified
        # by the suite: the unbounded form failed both the muse-spark admission
        # test and the expensive-tokenizer one.
        #
        # It continues the TAPER on the measured figure rather than flattening to
        # a constant. Granting exactly ``reserve`` made the ask a cliff: 15,908 at
        # 75% occupancy and then 4,096 flat at 80%, 88% and 95% alike, discarding
        # up to 264k tokens on a 1M model and truncating a live answer mid-sentence
        # that main completed (QA Q11). That is the margin deciding the size, which
        # is the failure rounds 1 and 3 both rejected.
        #
        # The bound is ``measured_available`` discounted by the worst ratio
        # actually OBSERVED on real content, not by the slope. Measured live
        # against this provider, agent-shaped content bills 1.10-1.21x the local
        # estimate (prose 1.098, code 1.166, logs 1.167, mixed transcript 1.210,
        # flat across a 23x size range), so the slope's 1.35 is a deliberate
        # over-estimate for the TAPER — where being wrong costs only headroom —
        # while the rescue needs the tighter figure to stay proportionate.
        #
        # Erring here is bounded in the direction this project has chosen three
        # times: an ask that is slightly too large produces a loud, retryable HTTP
        # 400 — exactly what main already does at these sizes — whereas an ask that
        # is too small produces a silent truncation the user cannot see.
        #
        # A RESIDUAL GAP remains above ~78% occupancy and is stated rather than
        # hidden: the slope is a bound, not a prediction, so where content bills
        # nearer 1.10 the ask is smaller than the provider would have allowed.
        # Measured live at 78.2% occupancy the branch asked 18,128 and the model
        # used 18,128 of the 21,174 tokens main's answer took — a graceful
        # truncation of the tail rather than the mid-sentence cut at 4,096 that
        # this replaced. Closing it entirely means lowering the slope until it no
        # longer bounds the overflow it exists to bound; the trade is deliberate,
        # and compaction fires at 80% of the window, so this band is narrow.
        available = max(available, min(requested, reserve))
    if available >= requested:
        # The overwhelmingly common case, and the one an earlier revision broke:
        # an ordinary prompt against a sanely advertised cap sends the spec's
        # number untouched. Anthropic (200k/64k, 1M/128k) and OpenAI (272k/128k)
        # come out byte-identical at every realistic session size.
        return requested

    # There is deliberately NO separate escape hatch for an explicit small ask
    # here. One existed for two rounds and was dead code both times: first
    # comparing against ``available`` (a contradiction no input could satisfy,
    # review R16), then against ``measured_available``, which the rescue above now
    # subsumes — ``min(requested, reserve)`` already returns an explicit ask that
    # is smaller than the reserve, so the caller's own number is honoured before
    # this point. Verified across 585 (window, explicit, occupancy) states: an
    # explicit ask is never raised and never refused while the measurement can
    # fund it, which is the ``Session.ERRAND_MAX_TOKENS`` guarantee. Re-adding a
    # branch here would restore the dead code, not the protection.
    if available < reserve:
        # REFUSE rather than send a cap too small to answer with. Sending it
        # anyway is the one outcome worse than the bug this fixes: reasoning
        # tokens are billed against this same budget (grok-4.6 spent 689 of them
        # thinking at a 512 cap), and ``harness/loop.py`` only retries a
        # COMPLETELY silent truncation — ``silent = not assistant.text and not
        # assistant.tool_calls`` — so a partial answer is accepted with no notice
        # and the user reads a confidently truncated reply.
        #
        # The refusal is judged on ``measured_prompt``, NOT on the scaled
        # estimate, and that distinction is the whole of review finding R8. A
        # scaled figure can exceed the window on a session that is genuinely
        # fine, and refusing there wedges it: the turn dies non-retryably while
        # the session sits BELOW its compaction threshold, and the compaction
        # summarizer — which takes this same path — is refused too, so the one
        # remedy the error names cannot run. Measured, a 200k Anthropic model
        # past ~135k local tokens could not compact at all.
        #
        # Keying on the unscaled figure is half of what makes the invariant hold;
        # :func:`_output_reserve_tokens` is the other half. An earlier revision
        # reserved a constant 8192 and claimed the ordering held "for every window
        # above ~41k, below which a model is too small for the trigger to help
        # anyway". The first clause was true and the second was false —
        # ``resolve_threshold_tokens`` returns a working trigger at 16k and 32k,
        # and the refusal fired beneath it there, wedging exactly as before one
        # window-band lower. The reserve is now a fraction of the same window the
        # trigger is a fraction of, so the ordering holds BY CONSTRUCTION at every
        # size rather than over a tested range.
        #
        # Compaction therefore always gets its chance first, and the summarizer's
        # own call — a fresh prefix, not the bloated transcript — measures small
        # and is never refused.
        #
        # A residual false refusal remains and is ACCEPTED rather than
        # overlooked: the local estimator can itself over-state a prompt (QA
        # measured ~8.8k high at ~93% occupancy), so a request with a little real
        # headroom left can be refused. That band sits far above the compaction
        # trigger, it is not a regression — the pre-fix behaviour there is an
        # HTTP 400 from the provider — and the two failures differ only in which
        # side reports them. Closing it would mean trusting a local count at
        # exactly the occupancy where being wrong is most expensive.
        raise ProviderError(
            None,
            (
                f"prompt is too large for {request.model.model_id}: about "
                f"{measured_prompt:,} tokens of input against a {window:,}-token "
                f"context window leaves under {reserve:,} tokens for the "
                f"reply. Compact the conversation or start a new session."
            ),
            kind="request",
        )
    return available


def _estimated_prompt_tokens(request: ChatRequest) -> tuple[int, int]:
    """Prompt size for :func:`_effective_max_tokens`, as ``(scaled, measured)``.

    TWO numbers, because they answer two different questions and conflating them
    is review finding R8:

    * ``scaled`` carries the safety margin and sizes the ASK. Erring high here
      costs a little output headroom near the window and nothing at all below it.
    * ``measured`` is the best unembellished figure available — the provider's
      own count when there is one, the raw local estimate otherwise. It decides
      whether the request is REFUSED. A refusal must never rest on our own
      inflation: that is what wedged sessions that were fine and blocked the
      compaction pass that would have rescued them.

    On the hinted branch the two are equal, because there is nothing to be
    uncertain about.

    The hint is used WHOLE. ``Usage.context_tokens`` is normalized in this file
    as ``input + cache_read + cache_write`` — the entire prompt the provider
    read, system blocks and tool schemas included — so the prefix term below
    belongs only to the estimated branch. Adding it to a hint double-counts it
    (R7).
    """
    hint = request.context_tokens_hint
    window = request.model.context_window
    if hint is not None and hint > 0 and not (window > 0 and hint > window):
        # Already a provider figure, and about a model this size: no slope, no
        # prefix, nothing to add. A hint LARGER than the window is deliberately
        # excluded — it cannot describe this request, so believing it would
        # refuse a session whose real context fits. That state is reachable two
        # ways, both reproduced by reviewers: `Session.set_model` swaps to a
        # smaller-window model without clearing the hint (a `/model` down-switch),
        # and the failover clone keeps the primary's hint while moving to a
        # smaller fallback spec. Falling through to the local estimate re-measures
        # the messages actually in hand, which is the only honest answer.
        return hint, hint

    local = estimate_messages_tokens(request.messages)

    extra_chars = sum(len(block) for block in request.system_blocks)
    for tool in request.tools:
        # ``len(str(...))`` rather than a json.dumps round trip: this runs per
        # tool on every request across all four clients, and a character count of
        # the schema is as good an input to a /4 estimate as its exact
        # serialization would be (review R6).
        extra_chars += len(tool.name) + len(tool.description) + len(str(tool.parameters))
    prefix = int(extra_chars / _CHARS_PER_TOKEN)

    measured = local + prefix
    return int(measured * _estimate_slope(request.model)), measured


def _reasoning_effort(request: ChatRequest) -> str | None:
    """The effort level to send, or ``None`` when the key must not appear.

    Same omission rule as :func:`_sampling_params`, and for the same reason: a
    model with no effort ladder rejects the key however politely it is spelled,
    so the body must not carry it at all. Each client then places the value
    under its own family's key — the level names are shared vocabulary, the
    key is not.

    The level is re-checked against the spec's ladder rather than trusted,
    because the spec is mutable at runtime: ``/effort`` and ``shift+tab`` write
    it, and a fallback can swap the model underneath it. A value the model does
    not accept is dropped here rather than sent, which costs one turn's worth of
    depth instead of the whole turn.
    """
    level = request.model.reasoning_effort
    if not level or level not in request.model.reasoning_efforts:
        return None
    return level


def _replayable_tool_arguments(call: ToolCall) -> dict[str, Any]:
    """The argument OBJECT to replay for ``call``, never a parse failure.

    ``raw_arguments`` is the provider's verbatim argument string and is
    normally replayed byte-for-byte, because a model that emitted
    non-canonical JSON must see its own bytes back. But the string is NOT
    guaranteed to be valid JSON: :meth:`AgentLoop._assemble_tool_call` builds
    it by concatenating the streamed argument deltas, so a turn that ends
    mid-call — an abort, a dropped stream, a provider 5xx between deltas —
    stores a truncated fragment like ``{"path": "/tmp/x.py"`` and leaves
    ``arguments`` empty because it could not parse it either.

    That fragment is then permanent: it is written to the transcript and
    replayed on EVERY later request in the session. Parsing it here without a
    guard raised ``JSONDecodeError`` out of body construction, which
    :func:`~local_operator.providers.failover.wrap_transport_error` could only
    read as a transient provider fault — so the harness retried, rebuilt the
    same body, failed identically, and reported the session's own corrupt row
    as the provider being unwell. Every subsequent turn died the same way and
    the session could not be continued at all.

    A fragment carries no recoverable intent, so the fallback is
    ``call.arguments`` (``{}`` for a call the loop could not parse). The pairing
    tool result already tells the model the call did not complete, which is the
    truth the next turn needs; replaying an unparseable argument list buys
    nothing and costs the session.
    """
    # Empty joins `None` rather than being read as `{}`: an empty string is not
    # a call the model made, it is the absence of one, and `call.arguments`
    # is the better answer for it whenever the loop managed to fill it.
    if not call.raw_arguments:
        return call.arguments
    try:
        parsed = json.loads(call.raw_arguments)
    except json.JSONDecodeError:
        logger.warning(
            "tool call %s (%s) carries unparseable raw arguments; "
            "replaying parsed arguments instead",
            call.id,
            call.name,
        )
        return call.arguments
    # A non-object parse (a bare string or list from a confused model) is just
    # as unusable as a fragment: the wire shape here is an object.
    return parsed if isinstance(parsed, dict) else call.arguments


def _replayable_tool_arguments_json(call: ToolCall) -> str:
    """The argument STRING to replay, for the providers that take one.

    Verbatim when ``raw_arguments`` is valid JSON — byte fidelity matters for a
    model reading back its own call — and a re-encode of the salvaged object
    otherwise. See :func:`_replayable_tool_arguments` for why the raw string
    cannot be trusted. OpenAI-shaped providers did not crash on the fragment
    the way Anthropic did; they were handed invalid JSON on the wire and
    rejected the request instead, which is the same dead session by a longer
    route.
    """
    raw = call.raw_arguments
    # Parse the raw value ITSELF, never `raw or "{}"`: that spelling validates
    # the placeholder and then returns `raw`, so an empty string passed the
    # check and went out as an empty body — invalid JSON on the wire, the exact
    # failure this function exists to prevent. Today's assembler normalizes
    # empty to None (`loop.py`, `raw or None`), but the field is typed `str |
    # None` and transcripts are external input, so the guard cannot lean on it.
    #
    # The `startswith` pre-check skips the parse for strings that CANNOT be an
    # object. It does not speed up the common path — a valid object starts with
    # `{` and is still parsed in full — so the win is confined to the
    # non-object case, where a large string is otherwise decoded in its
    # entirety only to be thrown away. Worth keeping because this runs for
    # every tool call in the whole history on every request, and `lstrip` on a
    # string with nothing to strip returns the same object.
    if raw is not None and raw.lstrip().startswith("{"):
        try:
            if isinstance(json.loads(raw), dict):
                return raw
        except json.JSONDecodeError:
            pass
    return json.dumps(_replayable_tool_arguments(call))


def _is_empty_assistant(message: Message) -> bool:
    """Is ``message`` an assistant turn with nothing on it a provider accepts?

    The harness persists an assistant message for EVERY model turn, including
    turns that died before producing a single token — an errored stream, an
    abort during thinking, a rate-limited request (``harness/loop.py`` appends
    the message and then sets ``stop_reason="error"/"aborted"``). Replayed
    verbatim, that turn serializes as an assistant message with empty content,
    and strict OpenAI-compatible providers reject the whole request: Moonshot/
    Kimi answers HTTP 400 "the message at position N with role 'assistant'
    must not be empty" (observed live 2026-08-19 switching a session from Qwen,
    which tolerates the empty turn, to Kimi, which does not). Anthropic
    likewise 400s on an empty content array.

    So every wire client drops these turns at body-build time — the same
    boundary where empty TOOL results are already backfilled (see
    ``EMPTY_TOOL_RESULT_TEXT``). Dropped rather than backfilled because a
    backfill would put words in the assistant's mouth that it never said, and
    the turn carries zero information: no text, no images, and — decisive for
    wire legality — no tool calls, so nothing downstream (a tool message, an
    Anthropic ``tool_result`` pairing) references it. Fixing the render rather
    than the transcript also repairs every EXISTING session that already
    carries such a turn, which is the actual failure mode: the 400 appears
    hundreds of messages deep, long after the errored turn was written.

    Whitespace-only text counts as empty: it renders to content a strict
    provider may still reject, and dropping it loses nothing a model could
    read. Assistant turns with tool calls are NEVER empty in this sense —
    the calls are the content.
    """
    if message.role != "assistant" or message.tool_calls:
        return False
    if any(isinstance(block, ImageContent) for block in message.content):
        return False
    return not message.text.strip()


def _message_to_openai(message: Message) -> dict[str, Any]:
    """Render one harness message into OpenAI chat-completions shape."""
    if message.role == "assistant" and message.tool_calls:
        # Same whitespace rule as `_is_empty_assistant`: a whitespace-only
        # text next to tool calls is noise, not content, and shipping it
        # would make the two paths disagree about what counts as empty.
        text = message.text if message.text.strip() else ""
        entry: dict[str, Any] = {"role": "assistant"}
        if text:
            entry["content"] = text
        entry["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": _replayable_tool_arguments_json(call),
                },
            }
            for call in message.tool_calls
        ]
        return entry
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id or "",
            "content": _tool_content_openai(message),
        }
    parts: list[dict[str, Any]] = []
    plain_only = True
    for block in message.content:
        if isinstance(block, TextContent):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            plain_only = False
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                }
            )
    role = message.role
    if plain_only:
        return {"role": role, "content": "".join(p["text"] for p in parts)}
    return {"role": role, "content": parts}


def _tool_output_to_openai_responses(
    output: str | list[dict[str, Any]],
) -> str | list[dict[str, Any]]:
    """Chat-completions tool content -> Responses function output content.

    Responses accepts a string OR native input content blocks. JSON-encoding
    chat's image_url parts turns base64 into megabytes of plain text and makes
    the screenshot invisible to the model; translate to input_text/input_image
    instead.
    """
    if isinstance(output, str):
        return output
    blocks: list[dict[str, Any]] = []
    for part in output:
        if part.get("type") == "text":
            blocks.append({"type": "input_text", "text": part.get("text", "")})
            continue
        if part.get("type") == "image_url":
            image = part.get("image_url") or {}
            url = image.get("url") if isinstance(image, Mapping) else ""
            if url:
                blocks.append({"type": "input_image", "image_url": url})
    return blocks


def _messages_to_openai_responses(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Render harness history as Responses input items, including tool turns."""
    items: list[dict[str, Any]] = []
    for message in messages:
        # Same normalization as chat/completions: an errored/aborted turn's
        # empty assistant message is dead weight a strict provider rejects.
        if _is_empty_assistant(message):
            continue
        if message.role == "assistant" and message.tool_calls:
            # Whitespace-only text is dropped for the same reason
            # `_is_empty_assistant` treats it as empty — see that predicate.
            if message.text.strip():
                items.append({"role": "assistant", "content": message.text})
            for call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": _replayable_tool_arguments_json(call),
                    }
                )
            continue
        if message.role == "tool":
            output = _tool_content_openai(message)
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id or "",
                    "output": _tool_output_to_openai_responses(output),
                }
            )
            continue

        content: list[dict[str, Any]] = []
        text_type = "output_text" if message.role == "assistant" else "input_text"
        for block in message.content:
            if isinstance(block, TextContent):
                content.append({"type": text_type, "text": block.text})
            elif isinstance(block, ImageContent):
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{block.mime_type};base64,{block.data}",
                    }
                )
        items.append({"role": message.role, "content": content})
    return items


EMPTY_TOOL_RESULT_TEXT = "[tool returned no output]"


def _tool_content_openai(message: Message) -> str | list[dict[str, Any]]:
    """Render a tool result from its content blocks — never ``message.text``.

    Flattening via ``.text`` drops image-only results to ``""``; render text
    blocks as text and image blocks as data-URL ``image_url`` parts. An empty
    result is backfilled so providers never receive empty content.
    """
    parts: list[dict[str, Any]] = []
    has_image = False
    for block in message.content:
        if isinstance(block, TextContent):
            if block.text:
                parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            has_image = True
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                }
            )
    if not parts:
        return EMPTY_TOOL_RESULT_TEXT
    if not has_image:
        return "".join(part["text"] for part in parts)
    return parts


def _tools_to_openai(tools: Sequence[AgentTool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


def _tools_to_openai_responses(tools: Sequence[AgentTool]) -> list[dict[str, Any]]:
    """Responses tools are flat; chat-completions nests fields under ``function``."""
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters or {"type": "object", "properties": {}},
        }
        for tool in tools
    ]


_FINISH_TO_STOP_REASON = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "toolUse",
    "function_call": "toolUse",
    # A filtered completion is a REFUSAL, not a clean stop. Mapping it to
    # "stop" ended the turn with an empty frame and no explanation — the user
    # saw nothing and could not tell a refusal from a no-op, which matters
    # because the remedy (rephrase, or switch models) is theirs to choose.
    "content_filter": "refusal",
    # OpenRouter terminates a mid-stream upstream failure with this finish
    # reason. The accompanying chunk normally carries the top-level ``error``
    # object the parser raises; when a gateway sends the reason WITHOUT the
    # object, the end event below still names the failure instead of passing
    # the raw word through as an exotic-but-successful stop reason.
    "error": "error",
}


def _refusal_error(marker: str, refusal_text: str, *, streamed_text: bool = False) -> str:
    """The one visible line a refusal produces, always naming the wire marker.

    ``marker`` is the provider's own terminal signal (``finish_reason=
    content_filter``, ``stop_reason=refusal``, ``finishReason=SAFETY``…) and is
    kept in the message even when refusal prose exists: the prose says what the
    model would not do, the marker says which provider mechanism fired, and
    deciding whether to rephrase or switch models needs both. Providers
    frequently send NO prose at all — that silent case is the whole reason this
    line exists, so it must read as a diagnosis rather than an empty string.

    ``streamed_text`` is whether the client forwarded any answer prose before
    the refusal terminal (design review D1): Anthropic and Gemini safety stops
    often cut a partial answer, and "sent no message" directly under a
    partially-rendered reply asserts the opposite of what is on screen. The
    line must describe the frame the user is looking at.

    Parentheses, not square brackets: this string reaches ``console.print`` in
    the headless renderer, where ``[marker]`` parses as rich markup and the
    diagnosis silently vanishes from its own error line.
    """
    if refusal_text:
        return f"model refused: {_capped(refusal_text)} ({marker})"
    if streamed_text:
        return f"model refused and cut the reply short ({marker})"
    return f"model refused and sent no message ({marker})"


# ---------------------------------------------------------------------------
# OpenAI-compatible
# ---------------------------------------------------------------------------


#: Longest silence tolerated BETWEEN chunks of a streaming response. Providers
#: emit tokens or keep-alives well inside this; a stream that says nothing for
#: three minutes has stalled, and treating that as patience turns a dead
#: connection into a UI that looks frozen for the whole request timeout.
STREAM_READ_TIMEOUT_S = 180.0
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_BETA_HEADER = "responses=experimental"


def _stream_timeout(total: float) -> httpx.Timeout:
    """Timeouts shaped for a STREAMING response, not a single request/response.

    ``read`` stays at the caller's total on purpose. httpx applies its read
    timeout to EVERY read on the stream, including the wait for response headers
    and the first body chunk — and a reasoning model over ``/chat/completions``
    emits nothing at all until it has finished thinking, which is legitimately
    minutes. Bounding time-to-first-byte at the inter-chunk budget would kill
    exactly the requests the budget was meant to protect, and `failover` would
    retry them, re-billing the prefill each time.

    The gap BETWEEN chunks is bounded separately, by :func:`_iter_sse_lines`,
    which is the only place that knows a stream has started.
    """
    return httpx.Timeout(total, connect=30.0, read=total, write=120.0)


_OPENAI_RESPONSE_ERROR_STATUS = {
    "invalid_api_key": 401,
    "authentication_error": 401,
    "permission_denied": 403,
    "rate_limit_exceeded": 429,
    "server_error": 500,
    "context_length_exceeded": 400,
}


def _openai_response_error(payload: Mapping[str, Any]) -> ProviderError:
    """Failed/top-level Responses terminal event -> shared ProviderError."""
    response_obj = payload.get("response") or {}
    error = payload.get("error") or (
        response_obj.get("error") if isinstance(response_obj, Mapping) else None
    )
    if not isinstance(error, Mapping):
        error = {"message": str(error or payload)}
    code = str(error.get("code") or error.get("type") or "")
    message = _first_text(error.get("message")) or code or "OpenAI Responses request failed"
    status = _OPENAI_RESPONSE_ERROR_STATUS.get(code)
    return ProviderError(
        status,
        f"{code}: {message}" if code and code not in message else message,
        retryable=status is None or status == 429 or status >= 500,
        auth_error=status in (401, 403),
    )


class OpenAICompatClient:
    """Stream OpenAI-compatible chat/completions or OpenAI Responses.

    Tool-call deltas are normalized onto the same harness events on both
    routes. Usage includes cached input tokens when the provider reports them.

    ChatGPT OAuth credentials (``oauth_access`` with ``kind == "oauth"`` and
    an ``org_id``) always use ChatGPT's private Codex Responses endpoint and
    headers. Ordinary API keys use the public ``/responses`` route only when
    this client is configured for it and the ``ModelSpec`` advertises support;
    compatibility providers and explicit opt-outs keep chat/completions.
    """

    def __init__(
        self,
        base_url: str,
        *,
        http_client: httpx.AsyncClient | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: float = 600.0,
        openai_api: str | None = None,
        oauth_base_url: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # Some providers serve subscription OAuth and pay-as-you-go API keys
        # from different hosts (see ``ProviderDefinition.oauth_base_url``). The
        # client is built before the credential is resolved -- failover may
        # rotate between kinds mid-turn -- so the host is chosen per REQUEST
        # from the credential actually presented, not fixed at construction.
        self._oauth_base_url = oauth_base_url.rstrip("/") if oauth_base_url else None
        if openai_api is None:
            # Direct construction remains useful in wire tests and extensions.
            # Only the canonical public base is safe to recognize implicitly;
            # every provider-registry construction passes an explicit mode.
            openai_api = (
                "responses" if self._base_url == "https://api.openai.com/v1" else "chat_completions"
            )
        self._openai_api = openai_api
        self._extra_headers = dict(extra_headers or {})
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=_stream_timeout(timeout))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    def _request_base_url(self, oauth_access: "OAuthAccess | None") -> str:
        """The host this REQUEST goes to, given the credential it carries.

        An OAuth bearer goes to ``oauth_base_url`` when the provider declares
        one; everything else uses the ordinary base. Without this, listing a
        subscription's models (which discovery now does against the OAuth host)
        would advertise models that inference then sends to the API-key host,
        where they 404 -- worse than not listing them at all.
        """
        if (
            self._oauth_base_url
            and oauth_access is not None
            and oauth_access.kind == "oauth"
            and oauth_access.access_token
        ):
            return self._oauth_base_url
        return self._base_url

    def _headers(
        self, api_key: str | None, oauth_access: "OAuthAccess | None" = None
    ) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._extra_headers}
        bearer = api_key
        if oauth_access is not None and oauth_access.kind == "oauth" and oauth_access.access_token:
            bearer = oauth_access.access_token
            if oauth_access.org_id:
                # ChatGPT subscription scope: which account pays for this call.
                headers["chatgpt-account-id"] = oauth_access.org_id
            headers.update(
                {
                    "Accept": "text/event-stream",
                    "OpenAI-Beta": CODEX_BETA_HEADER,
                    "originator": "local-operator",
                    "User-Agent": "local-operator",
                }
            )
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        return headers

    def _build_body(self, request: ChatRequest) -> dict[str, Any]:
        messages = [
            *self._system_messages(request),
            # Empty assistant turns (errored/aborted model turns the harness
            # persists) are dropped, not sent: Moonshot/Kimi 400s on them.
            # See `_is_empty_assistant`.
            *[_message_to_openai(m) for m in request.messages if not _is_empty_assistant(m)],
        ]
        if request.model.supports_prompt_cache:
            self._message_cache_markers(messages)
        body: dict[str, Any] = {
            "model": request.model.model_id,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": messages,
        }
        if request.tools:
            body["tools"] = _tools_to_openai(request.tools)
            # Safe default: unmapped values fall back to "auto" instead of KeyError.
            body["tool_choice"] = {"auto": "auto", "none": "none", "required": "required"}.get(
                request.tool_choice, "auto"
            )
        max_tokens = _effective_max_tokens(request)
        if max_tokens and max_tokens > 0:
            body["max_tokens"] = max_tokens
        body.update(_sampling_params(request))
        effort = _reasoning_effort(request)
        if effort is not None:
            # Chat-completions spells it flat and top-level; the Responses body
            # below nests the same value under `reasoning`. Aggregators fronting
            # OpenAI-shaped endpoints take the same key: measured live through
            # OpenRouter on 2026-08-11, `reasoning_effort: "low"` to
            # `openai/gpt-5.4` and `openai/o4-mini` both answered 200. That is
            # the extent of what was measured — an Anthropic model reached
            # through an aggregator, and the top rungs (`xhigh`/`max`), were not
            # exercised, so treat those as expected-to-work rather than proven.
            body["reasoning_effort"] = effort
        if request.stop_sequences:
            body["stop"] = list(request.stop_sequences)
        return body

    def _system_messages(self, request: ChatRequest) -> list[dict[str, Any]]:
        """System blocks → messages; stable blocks carry ``cache_control``.

        Mirrors the Anthropic client's all-but-last-two breakpoint policy so
        OpenRouter BYOK / OpenAI-compatible pools that honor ephemeral markers
        get the same warm prefix. Providers that ignore the field are
        unaffected; gated on ``supports_prompt_cache``.
        """
        blocks = request.system_blocks
        out: list[dict[str, Any]] = []
        stable_cutoff = max(len(blocks) - 2, 0)
        for i, block in enumerate(blocks):
            if request.model.supports_prompt_cache and i < stable_cutoff:
                out.append(
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": block, "cache_control": {"type": "ephemeral"}}
                        ],
                    }
                )
            else:
                out.append({"role": "system", "content": block})
        return out

    def _message_cache_markers(self, messages: list[dict[str, Any]]) -> None:
        """Mark the final message (and the previous user turn) for caching.

        Same economics as the Anthropic client: system-only markers stop the
        warm prefix before the conversation. OpenAI-compatible pools that
        honor ``cache_control`` on content parts (OpenRouter BYOK) then keep
        the previous request's prefix warm; providers that ignore the field
        are unaffected. Only applied when the model reports prompt caching.
        """
        targets: list[dict[str, Any]] = []
        if messages:
            last = messages[-1]
            content = last.get("content")
            if isinstance(content, str):
                last["content"] = [{"type": "text", "text": content}]
                content = last["content"]
            if isinstance(content, list) and content:
                targets.append(content[-1])
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if len(user_indices) >= 2:
            prev = messages[user_indices[-2]]
            content = prev.get("content")
            if isinstance(content, str):
                prev["content"] = [{"type": "text", "text": content}]
                content = prev["content"]
            if isinstance(content, list) and content and content[-1] not in targets:
                targets.append(content[-1])
        for block in targets:
            if isinstance(block, dict):
                block["cache_control"] = {"type": "ephemeral"}

    def _codex_responses_mode(self, oauth_access: "OAuthAccess | None") -> bool:
        """Whether this credential is a ChatGPT subscription OAuth grant."""
        return bool(
            oauth_access is not None and oauth_access.kind == "oauth" and oauth_access.org_id
        )

    def _public_responses_mode(
        self, request: ChatRequest, oauth_access: "OAuthAccess | None"
    ) -> bool:
        """Whether an ordinary OpenAI API key should use public Responses."""
        return (
            not self._codex_responses_mode(oauth_access)
            and self._openai_api == "responses"
            and request.model.supports_responses_api
        )

    def _build_responses_body(self, request: ChatRequest) -> dict[str, Any]:
        """Public Responses body using native input items and flat tools."""
        body: dict[str, Any] = {
            "model": request.model.model_id,
            "stream": True,
            "input": _messages_to_openai_responses(request.messages),
        }
        if request.system_blocks:
            # The stable system prefix rides top-level ``instructions``, exactly
            # as real Codex does (client.rs: ``instructions = base_instructions``).
            # We deliberately do NOT move it into ``developer`` messages or attach
            # ``prompt_cache_breakpoint`` markers: the ChatGPT-subscription Codex
            # backend rejects both ``prompt_cache_breakpoint`` and
            # ``prompt_cache_options`` with HTTP 400 (matches OpenAI Codex bug
            # #35300), and that OAuth backend is the only path in use here.
            body["instructions"] = "\n\n".join(request.system_blocks)
        if request.tools:
            body["tools"] = _tools_to_openai_responses(request.tools)
            body["tool_choice"] = {"auto": "auto", "none": "none", "required": "required"}.get(
                request.tool_choice, "auto"
            )
        max_tokens = _effective_max_tokens(request)
        if max_tokens and max_tokens > 0:
            body["max_output_tokens"] = max_tokens
        body.update(_sampling_params(request))
        effort = _reasoning_effort(request)
        if effort is not None:
            body["reasoning"] = {"effort": effort}
            # Defect #2: mirror OpenAI Codex (client.rs L720-724), which requests
            # encrypted reasoning items whenever reasoning is enabled. On the
            # ``store:false`` codex backend these ``reasoning.encrypted_content``
            # items are what let the SAME response reuse its reasoning KV state,
            # and they cost nothing when reasoning is off — so this is gated on
            # reasoning being present, not on the model version. See
            # ``_stream_responses`` for how the resulting reasoning items are
            # handled on the wire (safely skipped; we do not yet replay them).
            #
            # Extend rather than assign: nothing else sets ``include`` on this
            # path today, but a future include-bearing field must not be
            # silently clobbered by this line (review round 1, N1). Dedupe so a
            # re-entry cannot list the same value twice.
            include = body.setdefault("include", [])
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
        if request.model.supports_prompt_cache and request.prompt_cache_key:
            # The 24h policy is meaningful only with a stable key. SessionStreamFn
            # supplies one per transcript, and retries preserve it on ChatRequest.
            body["prompt_cache_key"] = request.prompt_cache_key
            body["prompt_cache_retention"] = "24h"
        return body

    def _build_codex_responses_body(self, request: ChatRequest) -> dict[str, Any]:
        """ChatGPT Codex body: Responses-shaped, on the ``store:false`` backend.

        Reuses the public Responses body, then strips the fields the codex
        backend does not take. Note that ``prompt_cache_key`` is deliberately
        NOT stripped any more: an earlier version popped it under a comment
        calling it "public-API-only", which was a wrong assumption. Real Codex
        (client.rs, ``build_responses_request``) sets ``prompt_cache_key``
        UNCONDITIONALLY on this same ``store:false`` backend for routing
        stickiness, and the model's ~89-90% cache-read rate versus ~97-98% for
        OpenAI-shaped peers was traced to us stripping it. ``prompt_cache_key``
        and ``include`` (defect #2's encrypted reasoning) flow through from
        ``_build_responses_body``. Only ``prompt_cache_retention`` is popped:
        the codex backend is ``store:false``, so public retention does not
        apply.
        """
        body = self._build_responses_body(request)
        body["store"] = False
        body.pop("prompt_cache_retention", None)
        body.pop("max_output_tokens", None)
        body.pop("temperature", None)
        body.pop("top_p", None)
        return body

    async def stream(
        self,
        request: ChatRequest,
        api_key: str | None,
        oauth_access: "OAuthAccess | None" = None,
    ) -> AsyncIterator[StreamEvent]:
        codex_responses = self._codex_responses_mode(oauth_access)
        if codex_responses or self._public_responses_mode(request, oauth_access):
            async for event in self._stream_responses(
                request, api_key, oauth_access, codex=codex_responses
            ):
                yield event
            return
        url = f"{self._request_base_url(oauth_access)}/chat/completions"
        finish_reason: str | None = None
        usage: Usage | None = None
        provider_payload: dict[str, Any] | None = None
        # Refusal prose arrives in its own delta slot (``delta.refusal``), not
        # ``delta.content``. It is collected rather than yielded as text: it is
        # not an answer, and forwarding it as prose would leave the transcript
        # reading as if the model replied normally.
        refusal_parts: list[str] = []
        streamed_text = False

        async with self._http.stream(
            "POST",
            url,
            json=self._build_body(request),
            headers=self._headers(api_key, oauth_access),
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise_for_status(response)
            async for data in _iter_sse_lines(response):
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk.get("error"), (Mapping, str)):
                    # In-band mid-stream failure: the status line already said
                    # 200, so this chunk is the only channel the failure has.
                    # Raise it NAMED; swallowing it left turns dying as
                    # wordless interruptions (see _compat_stream_error).
                    raise _compat_stream_error(chunk)
                if isinstance(chunk.get("usage"), Mapping):
                    raw = chunk["usage"]
                    cache_read, cache_write = _compat_cache_usage(raw)
                    completion_details = raw.get("completion_tokens_details") or {}
                    usage = Usage(
                        input_tokens=int(raw.get("prompt_tokens", 0)),
                        output_tokens=int(raw.get("completion_tokens", 0)),
                        cache_read_tokens=cache_read,
                        cache_write_tokens=cache_write,
                        # The thinking slice of the completion, when the wire
                        # separates it. A SUBSET of ``completion_tokens`` (see
                        # ``Usage.reasoning_tokens``), so analytics can split
                        # output into thinking vs generation.
                        reasoning_tokens=int(
                            completion_details.get("reasoning_tokens", 0)
                            if isinstance(completion_details, Mapping)
                            else 0
                        ),
                        # Prompt tokens ARE the context the provider just read:
                        # this is the authoritative context size the compaction
                        # trigger prefers over its own estimate, and what the
                        # TUI status line reports.
                        context_tokens=int(raw.get("prompt_tokens", 0)) or None,
                        # OpenRouter (and any OpenAI-compatible aggregator that
                        # precomputes billing) reports the dollar amount it
                        # actually charged here. Prefer it over the token×rate
                        # estimate: the routed provider's own price, reasoning-
                        # token split, cache discount and any override are all
                        # already baked in, and none of them are recoverable from
                        # the flat per-model table price.
                        usd_cost=_usd_cost(raw),
                    )
                    yield StreamUsageEvent(usage=usage)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    streamed_text = True
                    yield StreamTextDelta(delta=text)
                refusal = delta.get("refusal")
                if refusal:
                    refusal_parts.append(str(refusal))
                for tool_delta in delta.get("tool_calls") or []:
                    index = int(tool_delta.get("index", 0))
                    function = tool_delta.get("function") or {}
                    call_id = tool_delta.get("id")
                    name = function.get("name")
                    if call_id:
                        yield StreamToolCallDelta(index=index, id=call_id)
                    if name:
                        yield StreamToolCallDelta(index=index, name=name)
                    argument_delta = function.get("arguments")
                    if argument_delta:
                        yield StreamToolCallDelta(index=index, argument_delta=argument_delta)
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                if chunk.get("id") or chunk.get("system_fingerprint"):
                    provider_payload = {
                        "id": chunk.get("id"),
                        "system_fingerprint": chunk.get("system_fingerprint"),
                    }

        stop_reason = _FINISH_TO_STOP_REASON.get(finish_reason or "", finish_reason or "stop")
        # A refusal delta with a non-filter finish (OpenAI sends
        # ``finish_reason=stop`` for its own refusals; only third-party filters
        # send ``content_filter``) is still a refusal: the prose slot is the
        # authoritative signal that no answer was produced. ``length`` counts
        # too — refusal prose truncated by the token cap is a refusal whose
        # message got cut, and ending the turn as a bare "length" dropped the
        # collected prose entirely (review R1-3). ``toolUse`` is left alone: a
        # turn that produced executable calls is actionable output.
        if refusal_parts and stop_reason in ("stop", "length"):
            stop_reason = "refusal"
        error: str | None = None
        if stop_reason == "refusal":
            # Name what actually detected the refusal (design review D2): when
            # the finish reason was an ordinary stop/length and the prose slot
            # was the signal, a bare "(finish_reason=stop)" argues with the
            # word "refused" and names a mechanism that did not fire.
            marker = f"finish_reason={finish_reason or 'stop'}"
            if finish_reason != "content_filter":
                marker = f"delta.refusal, {marker}"
            # ``streamed_text`` matters on the content_filter path, where a
            # third-party filter commonly terminates AFTER answer chunks have
            # rendered and there is no refusal prose: "sent no message" under a
            # partial reply is the D1 contradiction again (review R3-1).
            error = _refusal_error(marker, "".join(refusal_parts), streamed_text=streamed_text)
        elif stop_reason == "error":
            # A wordless error end is exactly the silent-interruption defect:
            # the loop journals `error` as the incident, so name the failure,
            # not just the wire field that signalled it.
            error = f"provider reported a mid-stream failure (finish_reason '{finish_reason}')"
        yield StreamEndEvent(
            stop_reason=stop_reason,
            usage=usage,
            provider_payload=provider_payload,
            error=error,
        )

    async def _stream_responses(
        self,
        request: ChatRequest,
        api_key: str | None,
        oauth_access: "OAuthAccess | None",
        *,
        codex: bool,
    ) -> AsyncIterator[StreamEvent]:
        """Normalize public OpenAI and private ChatGPT Responses SSE."""
        url = CODEX_RESPONSES_URL if codex else f"{self._base_url}/responses"
        body = (
            self._build_codex_responses_body(request)
            if codex
            else self._build_responses_body(request)
        )
        usage: Usage | None = None
        provider_payload: dict[str, Any] | None = None
        tool_call_count = 0
        terminal_stop: str | None = None
        terminal_error: str | None = None
        # Refusal prose streams in its own event type (``response.refusal.delta``)
        # and is collected, not yielded as text: it is not an answer, and the
        # transcript must not read as if the model replied normally.
        refusal_parts: list[str] = []
        streamed_text = False
        # Output item/call ids -> normalized tool-call index.
        call_indexes: dict[str, int] = {}

        async with self._http.stream(
            "POST",
            url,
            json=body,
            headers=self._headers(api_key, oauth_access),
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise_for_status(response)
            async for data in _iter_sse_lines(response):
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                event_type = payload.get("type", "")
                # Requesting ``include: ["reasoning.encrypted_content"]`` (defect
                # #2) makes the stream carry ``reasoning`` output items and their
                # ``response.reasoning*`` deltas that a non-reasoning include
                # never produced. We deliberately DROP them: the harness has no
                # channel to replay an OpenAI encrypted reasoning item back on
                # the next turn's ``input`` (unlike Anthropic's thinking blocks,
                # which ride ``provider_payload``), and wiring that state through
                # the loop is out of this fix's scope. The include still earns
                # its keep — encrypted reasoning improves SAME-response cache
                # reuse — but cross-turn reasoning replay is intentionally not
                # attempted here. Skipping is explicit rather than incidental so
                # a future ``else`` branch cannot accidentally render a reasoning
                # item's encrypted blob as assistant text. (Called out for review.)
                if event_type.startswith("response.reasoning"):
                    continue
                if event_type == "response.output_item.added":
                    item = payload.get("item") or {}
                    if item.get("type") == "reasoning":
                        # Same rationale as above: acknowledge the item, drop it.
                        continue
                    if item.get("type") == "function_call":
                        index = tool_call_count
                        tool_call_count += 1
                        call_id = item.get("call_id") or item.get("id") or ""
                        item_id = item.get("id") or ""
                        if call_id:
                            call_indexes[call_id] = index
                        if item_id:
                            call_indexes[item_id] = index
                        yield StreamToolCallDelta(index=index, id=call_id, name=item.get("name"))
                elif event_type == "response.function_call_arguments.delta":
                    call_id = payload.get("call_id") or payload.get("item_id") or ""
                    delta = payload.get("delta")
                    if delta:
                        fallback_index = int(payload.get("output_index", 0) or 0)
                        yield StreamToolCallDelta(
                            index=call_indexes.get(call_id, fallback_index),
                            argument_delta=delta,
                        )
                elif event_type == "response.output_text.delta":
                    delta = payload.get("delta")
                    if delta:
                        streamed_text = True
                        yield StreamTextDelta(delta=delta)
                elif event_type == "response.refusal.delta":
                    delta = payload.get("delta")
                    if delta:
                        refusal_parts.append(str(delta))
                elif event_type in ("response.completed", "response.incomplete"):
                    response_obj = payload.get("response") or {}
                    if response_obj.get("id"):
                        provider_payload = {"id": response_obj["id"]}
                    raw = response_obj.get("usage") or {}
                    if raw:
                        details = raw.get("input_tokens_details") or {}
                        if not isinstance(details, Mapping):
                            details = {}
                        output_details = raw.get("output_tokens_details") or {}
                        usage = Usage(
                            input_tokens=int(raw.get("input_tokens", 0)),
                            output_tokens=int(raw.get("output_tokens", 0)),
                            cache_read_tokens=_usage_token(details, "cached_tokens"),
                            cache_write_tokens=_usage_token(details, "cache_write_tokens"),
                            # Responses breaks reasoning out under
                            # ``output_tokens_details``. A SUBSET of
                            # ``output_tokens`` (see ``Usage.reasoning_tokens``).
                            reasoning_tokens=int(
                                output_details.get("reasoning_tokens", 0)
                                if isinstance(output_details, Mapping)
                                else 0
                            ),
                            context_tokens=int(raw.get("input_tokens", 0)) or None,
                        )
                        yield StreamUsageEvent(usage=usage)
                    if event_type == "response.completed":
                        if refusal_parts:
                            # A completed response whose only output was a
                            # refusal item: the wire says "completed", the
                            # content says "no". The content wins — reporting
                            # "stop" here is the silent-empty-turn bug.
                            terminal_stop = "refusal"
                            terminal_error = _refusal_error(
                                "response.completed with a refusal item",
                                "".join(refusal_parts),
                            )
                        else:
                            terminal_stop = "toolUse" if tool_call_count else "stop"
                    else:
                        incomplete = response_obj.get("incomplete_details") or {}
                        reason = str(
                            incomplete.get("reason")
                            if isinstance(incomplete, Mapping)
                            else incomplete
                        )
                        if reason in ("max_output_tokens", "max_output_chars"):
                            if refusal_parts:
                                # Refusal prose truncated by the output cap is
                                # still a refusal; a bare "length" terminal
                                # dropped the collected prose (review R1-3).
                                # The refusal slot is named as the detection
                                # signal (D2): the incomplete reason alone
                                # describes the truncation, not the refusal.
                                terminal_stop = "refusal"
                                terminal_error = _refusal_error(
                                    f"response.refusal, incomplete_details.reason={reason}",
                                    "".join(refusal_parts),
                                )
                            else:
                                # Length means the loop pairs placeholders and
                                # NEVER executes a partial function call.
                                terminal_stop = "length"
                        elif reason == "content_filter":
                            # A filtered response is a refusal, not a transport
                            # fault: raising ProviderError here sent it into
                            # failover's retry machinery for a request the
                            # provider had already declined on content grounds.
                            terminal_stop = "refusal"
                            # ``streamed_text`` for the same reason as the
                            # chat-completions filter terminal: a filter that
                            # cut a partially-rendered answer must not claim
                            # "sent no message" under it (review R3-1).
                            terminal_error = _refusal_error(
                                "incomplete_details.reason=content_filter",
                                "".join(refusal_parts),
                                streamed_text=streamed_text,
                            )
                        else:
                            raise ProviderError(
                                400,
                                f"OpenAI Responses incomplete: {reason or 'unknown reason'}",
                                retryable=False,
                            )
                elif event_type in ("response.failed", "error"):
                    raise _openai_response_error(payload)

        if terminal_stop is None:
            raise ProviderError(
                None,
                "OpenAI Responses stream ended without a terminal event",
                retryable=True,
            )
        yield StreamEndEvent(
            stop_reason=terminal_stop,
            usage=usage,
            provider_payload=provider_payload,
            error=terminal_error,
        )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

ANTHROPIC_API_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"

#: Anthropic's mid-stream ``error`` events carry the diagnosis in ``type``, not
#: in a status: the HTTP response was 200 long before one arrives. Mapping the
#: type back to the status it would have had is what lets the shared classifier
#: call an ``overloaded_error`` transient and an ``authentication_error`` auth,
#: instead of the blanket ``retryable=True`` that used to re-send a request the
#: API had already refused.
_ANTHROPIC_ERROR_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "conflict_error": 409,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}


def _anthropic_stream_error(error: Mapping[str, Any]) -> ProviderError:
    """One mid-stream anthropic ``error`` event as a ``ProviderError``.

    The ``type`` is prepended to the message because anthropic's text alone is
    frequently one bare word ("Overloaded"), and an unknown type keeps
    ``retryable=True`` — the pre-existing assumption, correct for the api_error
    and overloaded cases this event mostly carries.
    """
    error_type = str(error.get("type") or "")
    status = _ANTHROPIC_ERROR_STATUS.get(error_type)
    message = _first_text(error.get("message"))
    if not message:
        message = error_type or (str(error) if error else "")
    elif error_type and error_type not in message:
        message = f"{error_type}: {message}"
    return ProviderError(
        status,
        message,
        retryable=status is None or status == 429 or status >= 500,
        auth_error=status in (401, 403),
    )


class AnthropicClient:
    """``POST {base}/v1/messages`` streaming.

    System blocks are sent as an array with ``cache_control: {type:
    "ephemeral"}`` on every block EXCEPT the last (the breakpoint policy:
    the volatile tail stays un-cached). Content arrives as
    ``content_block_start/delta/stop`` events for ``text`` and ``tool_use``
    blocks; tool arguments stream via ``input_json_delta``.

    Large contexts switch every marker to the 1-hour cache TTL; see
    ``_cache_ttl_for`` for the economics and the wire constraint.
    """

    def __init__(
        self,
        base_url: str = ANTHROPIC_API_URL,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 600.0,
        cache_ttl_1h_min_context_tokens: int = 0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=_stream_timeout(timeout))
        # Context size (tokens) from which requests carry the 1h TTL; 0 keeps
        # every request on the default 5m. Defaults OFF at the constructor so a
        # bare ``AnthropicClient()`` (tests, scripts) sends the exact body it
        # always did; ``client_for_spec`` passes the configured threshold.
        self._cache_ttl_1h_min_context_tokens = max(0, int(cache_ttl_1h_min_context_tokens))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    @staticmethod
    def _is_oauth(oauth_access: "OAuthAccess | None") -> bool:
        """True when this request carries a subscription (OAuth) credential.

        One definition, because two things now depend on it and they MUST agree:
        the auth header scheme and the Claude Code identity block. A request that
        sends the Bearer token without the block is refused, and one that sends
        the block with ``x-api-key`` alters an API-key user's prompt for nothing.
        """
        return (
            oauth_access is not None
            and oauth_access.kind == "oauth"
            and bool(oauth_access.access_token)
        )

    def _headers(
        self,
        api_key: str | None,
        oauth_access: "OAuthAccess | None" = None,
        *,
        effort: str | None = None,
    ) -> dict[str, str]:
        headers = {"anthropic-version": ANTHROPIC_VERSION, "Content-Type": "application/json"}
        if self._is_oauth(oauth_access):
            # Claude Pro/Max OAuth: Bearer + the oauth beta header (the
            # ``x-api-key`` scheme 401s OAuth-issued access tokens).
            assert oauth_access is not None  # narrowed by _is_oauth
            headers["Authorization"] = f"Bearer {oauth_access.access_token}"
            headers["anthropic-beta"] = "oauth-2025-04-20"
        elif api_key:
            headers["x-api-key"] = api_key
        if effort:
            beta = "effort-2025-11-24"
            existing = headers.get("anthropic-beta")
            headers["anthropic-beta"] = f"{existing},{beta}" if existing else beta
        return headers

    # Anthropic caps cache_control markers per request; the harness keeps the
    # first 3 stable system blocks breakpointed and never exceeds the cap.
    MAX_CACHE_BREAKPOINTS = 4

    #: Anthropic gates OAuth (Claude Pro/Max subscription) credentials to Claude
    #: Code: a request whose FIRST system block is not this exact identity is
    #: refused. The refusal is an opaque ``HTTP 429 Error``, not a 401 or a 403,
    #: so it reads as rate limiting and sends the operator looking at their quota
    #: instead of at their request. Measured against the live endpoint with a
    #: valid subscription token and ``model: claude-opus-5``:
    #:
    #:     no system block          -> HTTP 429
    #:     ordinary system block    -> HTTP 429
    #:     this block first         -> HTTP 200
    #:
    #: API-key credentials are NOT gated and must not receive it — an identity
    #: line changes how the model answers, and paying customers did not ask to be
    #: told they are a CLI.
    CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

    #: Rough bytes-per-token for the byte-based context estimate in
    #: ``_cache_ttl_for``. English prose and JSON both sit near 4; the estimate
    #: only has to land on the right side of a 150k threshold, not be exact.
    _BYTES_PER_TOKEN_ESTIMATE = 4

    #: Token FLOOR counted per image in the byte-based context estimate in
    #: ``_cache_ttl_for``. The serialized body carries images as base64 in
    #: ``source.data``, so a naive ``len(json) / 4`` turns one ~1 MB screenshot
    #: into ~330k "tokens". Anthropic bills an image by its PIXEL AREA
    #: (``width × height / 750``, capped at ~1.15 megapixels), so a worst-case
    #: image is ~1.6k tokens no matter how many bytes its base64 spelling
    #: takes. A floor rather than a parse: media_type/size metadata is not
    #: reliably present on every path that builds an ``ImageContent``, and
    #: the estimate only has to land on the right side of the threshold.
    _IMAGE_TOKENS_ESTIMATE = 1_600

    @staticmethod
    def _estimate_context_tokens(body: dict[str, Any]) -> int:
        """Size the request for ``_cache_ttl_for``'s threshold comparison.

        Counts the serialized body's characters EXCEPT base64 image payloads,
        each of which contributes ``_IMAGE_TOKENS_ESTIMATE`` instead: bytes of
        base64 say nothing about the tokens a provider bills (see that
        constant), and leaving them in makes any first call, fork or resume
        carrying a screenshot flip to the 1h TTL on a ~1.6k-token image.
        Images are located by walking the body's shape — any dict with a
        ``source`` whose ``type`` is ``base64`` and whose ``data`` is a string
        — rather than by key position, because tool results nest them one
        level deeper than plain message content.
        """
        images = 0

        def _swap_out_images(value: Any) -> Any:
            nonlocal images
            if isinstance(value, dict):
                source = value.get("source")
                if (
                    isinstance(source, dict)
                    and source.get("type") == "base64"
                    and isinstance(source.get("data"), str)
                ):
                    images += 1
                    # The key stays so the shape (and key order) matches what
                    # gets serialized; only the counted payload shrinks.
                    return {**value, "source": {**source, "data": ""}}
                return {k: _swap_out_images(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_swap_out_images(item) for item in value]
            return value

        # ``default=str`` guards against any non-JSON-native value a caller
        # slipped into the body; ``separators`` matches what httpx sends.
        serialized = json.dumps(_swap_out_images(body), default=str, separators=(",", ":"))
        return len(serialized) // AnthropicClient._BYTES_PER_TOKEN_ESTIMATE + (
            images * AnthropicClient._IMAGE_TOKENS_ESTIMATE
        )

    @staticmethod
    def _cache_control(ttl: str | None) -> dict[str, Any]:
        """One ``cache_control`` marker. ``ttl`` is ``"1h"`` or None (5m).

        The 5m marker deliberately omits the ``ttl`` key rather than spelling
        ``"5m"``: it is the exact shape every prior request sent, so a client
        below the threshold (or with the feature off) produces a byte-identical
        body to the one before this feature existed, and nothing about the
        default path has to be re-validated.
        """
        marker: dict[str, Any] = {"type": "ephemeral"}
        if ttl:
            marker["ttl"] = ttl
        return marker

    def _cache_ttl_for(self, request: ChatRequest, body: dict[str, Any]) -> str | None:
        """Pick the prompt-cache TTL for this request: ``"1h"`` or None (5m).

        WHY: a 5m entry expires while a large session waits on a subagent, a
        scheduled wake, or the user, and its next call then rewrites the whole
        prefix at 1.25× base. Measured over 24h of this harness's own traffic:
        276 such rewrites of >150k contexts cost 89.5M write tokens (~112M
        base-equivalent), while ALL incremental writes on those contexts were
        14.7M tokens — which at the 1h rate (2× base instead of 1.25×) cost
        ~11M base-equivalent extra. Above the threshold the 1h TTL is a
        ~10:1 win; below it, small prefixes are cheap to rewrite and the 2×
        write rate on every turn would cost more than the expiries it avoids.

        CONSTRAINT (Anthropic, "Mixing different TTLs"): a 1h marker must
        appear BEFORE every 5m marker in the request, so a body cannot carry
        5m markers on the system blocks and 1h markers on the messages. The
        result of this method is therefore applied to EVERY marker in the
        request — system blocks and message breakpoints alike — never to a
        subset.

        The decision is one-directional in practice and needs no hysteresis:
        context only grows within a turn, and once it crosses the threshold
        the 1h markers stay until compaction shrinks it. Dropping back to 5m
        markers after compaction is fine — the compacted prefix is new
        content that has to be written either way, and a smaller context is
        exactly the case where the cheaper write rate wins again.

        Two size sources, in preference order:

        1. ``request.context_tokens_hint`` — the provider's OWN count from
           the session's previous call, stamped by ``SessionStreamFn``. It is
           exact for the prefix this request replays, and off only by the
           one new turn appended since.
        2. A byte estimate of the serialized body when no hint exists: a
           session's first call, a fork's first request, a one-shot errand
           with no usage history. The estimate (``_estimate_context_tokens``)
           excludes base64 image payloads — Anthropic bills an image by pixel
           area, not bytes — but is otherwise coarse; it only has to land on
           the right side of a 150k threshold, and the hint replaces it from
           the second call on. Without the fallback a resumed 400k session
           would send its first (and most expensive) request at 5m.
        """
        threshold = self._cache_ttl_1h_min_context_tokens
        if threshold <= 0:
            return None
        size = request.context_tokens_hint
        if size is None:
            size = self._estimate_context_tokens(body)
        return "1h" if size >= threshold else None

    @classmethod
    def _system_blocks(
        cls, blocks: Sequence[str], *, ttl: str | None = None
    ) -> list[dict[str, Any]]:
        """System blocks → Anthropic ``system`` array with cache breakpoints.

        The harness sends [instructions, tool inventory, skills, env/date];
        the trailing blocks are VOLATILE (skills change per turn, env/date
        changes per day) and must stay breakpoint-free so the prompt-cache
        prefix covers only the stable head. Generic for any block count:
        every block except the last two gets an ephemeral breakpoint, CAPPED
        at ``MAX_CACHE_BREAKPOINTS`` — Anthropic rejects requests carrying
        more than 4 ``cache_control`` markers, so surplus stable blocks keep
        the cache prefix intact without adding markers.

        ``ttl`` is the request-wide TTL from ``_cache_ttl_for``; every marker
        placed here carries it, because a 1h marker may not follow a 5m one.
        """
        rendered: list[dict[str, Any]] = []
        stable_count = min(cls.MAX_CACHE_BREAKPOINTS, max(0, len(blocks) - 2))
        for index, block in enumerate(blocks):
            entry: dict[str, Any] = {"type": "text", "text": block}
            if index < stable_count:
                entry["cache_control"] = cls._cache_control(ttl)
            rendered.append(entry)
        return rendered

    @staticmethod
    def _message_blocks(message: Message) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextContent):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageContent):
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": block.mime_type,
                            "data": block.data,
                        },
                    }
                )
        return blocks

    @staticmethod
    def _tool_result_blocks(message: Message) -> list[dict[str, Any]]:
        """Render a tool result's content as Anthropic blocks.

        Uses the message's content blocks (never ``message.text``) so
        image-only results survive; an empty result is backfilled because
        Anthropic 400s on empty ``tool_result`` content.
        """
        blocks = AnthropicClient._message_blocks(message)
        if not blocks:
            return [{"type": "text", "text": EMPTY_TOOL_RESULT_TEXT}]
        return blocks

    def _message_cache_breakpoints(
        self, messages: list[dict[str, Any]], body: dict[str, Any], *, ttl: str | None = None
    ) -> None:
        """Place cache_control on the conversation, not just the system head.

        Anthropic caches the prefix ending at each marker; system-only
        markers stop the cached prefix before the first message, so the
        growing conversation (every tool result, every assistant turn) is
        re-processed at full price on every request. Mark the last content
        block of the final message and of the second-to-last user turn so
        the previous prefix stays warm across turns.

        Budget: MAX_CACHE_BREAKPOINTS total. System markers are counted
        first; when the sum would exceed the cap the LOWEST-value system
        breakpoint (the last stable block) is dropped in favour of the
        message markers, which cover far more tokens.

        ``ttl`` must be the SAME value the system markers were rendered with:
        Anthropic rejects a 1h marker that follows a 5m one, and the message
        markers always follow the system ones.
        """
        system = body.get("system") or []
        system_markers = [i for i, entry in enumerate(system) if "cache_control" in entry]
        message_targets: list[dict[str, Any]] = []
        if messages:
            last = messages[-1]
            if isinstance(last.get("content"), list) and last["content"]:
                message_targets.append(last["content"][-1])
        # Second-to-last USER turn keeps the previous request's prefix warm.
        user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if len(user_indices) >= 2:
            prev_user = messages[user_indices[-2]]
            if isinstance(prev_user.get("content"), list) and prev_user["content"]:
                block = prev_user["content"][-1]
                if block not in message_targets:
                    message_targets.append(block)
        if not message_targets:
            return
        budget = self.MAX_CACHE_BREAKPOINTS - len(message_targets)
        # Drop the lowest-value system breakpoints (the last stable blocks)
        # until system + message markers fit the cap.
        for index in system_markers[budget:]:
            system[index].pop("cache_control", None)
        for block in message_targets:
            if isinstance(block, dict):
                block["cache_control"] = self._cache_control(ttl)

    def _build_body(self, request: ChatRequest, *, oauth: bool = False) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            # Anthropic 400s on an assistant message whose content array is
            # empty, which is exactly what an errored/aborted model turn
            # replays as. Same drop as the OpenAI paths; see
            # `_is_empty_assistant`.
            if _is_empty_assistant(message):
                continue
            if message.role == "assistant" and message.tool_calls:
                content = self._message_blocks(message)
                content.extend(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": _replayable_tool_arguments(call),
                    }
                    for call in message.tool_calls
                )
                messages.append({"role": "assistant", "content": content})
            elif message.role == "tool":
                # Anthropic groups tool results under one user message.
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": self._tool_result_blocks(message),
                        **({"is_error": True} if message.is_error else {}),
                    }
                ]
                if (
                    messages
                    and messages[-1]["role"] == "user"
                    and isinstance(messages[-1]["content"], list)
                ):
                    messages[-1]["content"].extend(content)
                else:
                    messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": message.role, "content": self._message_blocks(message)})

        body: dict[str, Any] = {
            "model": request.model.model_id,
            "stream": True,
            "messages": messages,
            "max_tokens": _effective_max_tokens(request),
        }
        # The identity block is PREPENDED, not appended, because Anthropic checks
        # the first block specifically. It is a constant, so it makes ideal
        # cache-prefix material and keeps the prefix byte-stable across turns —
        # and because it is added on every OAuth request, the breakpoint policy
        # below sees the same block count each time rather than shifting.
        blocks = list(request.system_blocks)
        if oauth:
            blocks.insert(0, self.CLAUDE_CODE_IDENTITY)
        # Rendered marker-free first: the TTL every marker will carry is
        # decided once messages, tools and system text are ALL in the body, so
        # the byte-estimate fallback sees everything the provider will count.
        # See ``_cache_ttl_for`` for why one value applies to every marker.
        if blocks:
            body["system"] = [{"type": "text", "text": block} for block in blocks]
        if request.tools:
            body["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters or {"type": "object", "properties": {}},
                }
                for tool in request.tools
            ]
        ttl = self._cache_ttl_for(request, body)
        if blocks:
            body["system"] = self._system_blocks(blocks, ttl=ttl)
        # System-only breakpoints stop the cached prefix before the first
        # message: the entire growing conversation would be re-processed at
        # full price on every request. Mark the last content block of the
        # final message and the second-to-last user turn so the previous
        # prefix stays warm across turns, within MAX_CACHE_BREAKPOINTS by
        # dropping the lowest-value system breakpoint.
        self._message_cache_breakpoints(messages, body, ttl=ttl)
        if request.tools:
            # Safe default: unmapped values fall back to auto (PR-22).
            #
            # ``none`` is DELIBERATELY sent as ``auto`` on this wire, and only
            # here. Anthropic's cache hierarchy is tools -> system -> messages,
            # and per the prompt-caching docs ("What invalidates the cache")
            # ``tool_choice`` is rendered into the MESSAGES level: a request
            # whose ``tool_choice`` differs from the one that wrote the prefix
            # keeps the tools+system head warm and re-writes every message
            # block. The callers that send a non-empty tool list with
            # ``tool_choice="none"`` — ``Session.complete_aside`` (``/btw`` and
            # the ``/loop`` judge) and ``Session.advise_compaction`` — exist to
            # ride the working turn's cached prefix, and the turn sends
            # ``{"type": "auto"}``.
            #
            # This mapping is hygiene, not a measured saving. Measured live
            # (``scripts/measure_aside_tool_choice_cache.py``, result in
            # ``docs/evidence/compaction-advisor/aside-tool-choice-measurement.txt``):
            # a ~37k-token aside sent with ``none`` read the turn's FULL prefix
            # and wrote only its appended question, identical to ``auto`` — no
            # cache break reproduced on either model tried. The fleet's
            # head-only-hit signature that first pointed here was root-caused
            # elsewhere: per-account cache isolation when the quota preflight
            # moved a session between OAuth accounts under a reserve verdict
            # (PR #537). ``auto`` is kept because it makes the aside body
            # byte-identical to the turn's request, which is the only shape the
            # invalidation rule is guaranteed not to bite, and nothing about an
            # aside changes the tool surface the turn declared.
            #
            # The "reads the turn, calls nothing" contract those callers promise
            # is therefore NOT enforced on the wire for Anthropic. It is
            # enforced by the callers: both consume only ``StreamTextDelta`` /
            # ``StreamUsageEvent`` and never execute a ``StreamToolCallDelta``,
            # so a ``tool_use`` block in the answer is inert (the model is also
            # told in the appended prompt to answer in text). ``complete_aside``
            # retries once WITHOUT tools when the answer was a tool call and
            # nothing else, which is the only way the mapping is observable.
            #
            # The no-tools callers (naming, compaction summary, server operator)
            # send ``tools=[]`` and never reach this branch, so they carry no
            # ``tool_choice`` key at all, exactly as before. The OpenAI and
            # Gemini builders keep a literal ``none``: neither documents a
            # cache penalty for it, and Gemini's mode MUST stay ``NONE`` because
            # its default with tools present is to allow calls.
            body["tool_choice"] = {
                "auto": {"type": "auto"},
                "none": {"type": "auto"},
                "required": {"type": "any"},
            }.get(request.tool_choice, {"type": "auto"})
        effort = _reasoning_effort(request)
        if effort is not None:
            # `output_config`, NOT a `thinking` budget: Anthropic's effort
            # parameter covers ALL tokens in the response — text, tool calls and
            # thinking — and needs no beta header, where a thinking budget only
            # bounds the thinking block and requires it to be enabled. See
            # https://platform.claude.com/docs/en/build-with-claude/effort.
            #
            # `thinking: adaptive` is a different thing from a budget and rides
            # along from main: it lets the model choose its own depth instead of
            # being handed a token ceiling. Kept INSIDE the validated gate, so a
            # model with no effort ladder is still sent neither key.
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": effort}
        else:
            # The sampling pair goes out ONLY when thinking does not. Anthropic
            # rejects the combination — HTTP 400 ``` `temperature` may only be
            # set to 1 when thinking is enabled or in adaptive mode ``` (and the
            # same for `top_p`), observed live on `claude-opus-4-8` on
            # 2026-08-21 — and every turn on such a model died on it. Generation
            # 5+ never hit this because `_NO_SAMPLING_PARAMS` already drops the
            # pair, but 4.5–4.9 models accept temperature on its own while
            # carrying an effort ladder, and the ladder is what switches
            # adaptive thinking on above. Keying the omission on the SAME gate
            # that writes `thinking` (rather than on a second model-name list)
            # means any future model with an effort ladder is automatically
            # safe: whichever way the gate resolves, the body is one Anthropic
            # accepts. Omitting the pair costs nothing real — with thinking on,
            # the only accepted value is the provider default.
            body.update(_sampling_params(request))
        if request.stop_sequences:
            body["stop_sequences"] = list(request.stop_sequences)
        return body

    async def stream(
        self,
        request: ChatRequest,
        api_key: str | None,
        oauth_access: "OAuthAccess | None" = None,
    ) -> AsyncIterator[StreamEvent]:
        url = f"{self._base_url}/v1/messages"
        stop_reason = "stop"
        usage = Usage()
        streamed_text = False
        block_index_to_call: dict[int, tuple[str, str]] = {}

        async with self._http.stream(
            "POST",
            url,
            json=self._build_body(request, oauth=self._is_oauth(oauth_access)),
            headers=self._headers(
                api_key,
                oauth_access,
                effort=request.model.reasoning_effort,
            ),
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise_for_status(response)
            async for data in _iter_sse_lines(response):
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type")
                if event_type == "message_start":
                    raw_usage = (event.get("message") or {}).get("usage") or {}
                    usage.input_tokens = int(raw_usage.get("input_tokens", usage.input_tokens))
                    usage.cache_read_tokens = int(raw_usage.get("cache_read_input_tokens", 0))
                    usage.cache_write_tokens = int(raw_usage.get("cache_creation_input_tokens", 0))
                    # The per-TTL split of the write count. Anthropic documents
                    # ``cache_creation_input_tokens`` as the SUM of these two,
                    # and the two are priced differently (1.25× vs 2× base), so
                    # analytics needs them apart to judge the 1h trade. The
                    # object is absent on older API versions; both then stay 0.
                    creation = raw_usage.get("cache_creation") or {}
                    if isinstance(creation, dict):
                        usage.cache_write_5m_tokens = int(
                            creation.get("ephemeral_5m_input_tokens", 0) or 0
                        )
                        usage.cache_write_1h_tokens = int(
                            creation.get("ephemeral_1h_input_tokens", 0) or 0
                        )
                    # Anthropic reports input_tokens EXCLUDING cached blocks
                    # (OpenAI includes them), so the context actually read is
                    # the sum of the three. The compaction trigger and the TUI
                    # status line both consume context_tokens, so the provider
                    # difference has to be normalized here, not downstream.
                    usage.context_tokens = (
                        usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
                    ) or None
                elif event_type == "content_block_start":
                    block = event.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        index = int(event.get("index", 0))
                        block_index_to_call[index] = (block.get("id", ""), block.get("name", ""))
                        yield StreamToolCallDelta(
                            index=index, id=block.get("id"), name=block.get("name")
                        )
                elif event_type == "content_block_delta":
                    delta = event.get("delta") or {}
                    delta_type = delta.get("type")
                    if delta_type == "text_delta":
                        text = delta.get("text")
                        if text:
                            streamed_text = True
                            yield StreamTextDelta(delta=text)
                    elif delta_type == "input_json_delta":
                        index = int(event.get("index", 0))
                        partial = delta.get("partial_json")
                        if partial:
                            yield StreamToolCallDelta(index=index, argument_delta=partial)
                elif event_type == "message_delta":
                    delta = event.get("delta") or {}
                    if delta.get("stop_reason"):
                        stop_reason = str(delta["stop_reason"])
                    raw_usage = event.get("usage") or {}
                    if "output_tokens" in raw_usage:
                        usage.output_tokens = int(raw_usage["output_tokens"])
                elif event_type == "error":
                    raise _anthropic_stream_error(event.get("error") or {})

        mapped = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "toolUse",
            "stop_sequence": "stop",
            # Anthropic's documented refusal terminal (classifier-stopped
            # output). It passed through this map UNMAPPED, and downstream —
            # which only branches on error/aborted/length — treated the unknown
            # value as a clean stop: an empty turn with no explanation.
            "refusal": "refusal",
        }.get(stop_reason, stop_reason)
        error: str | None = None
        if mapped == "refusal":
            # Anthropic sends no refusal prose alongside this stop_reason; any
            # text it did stream has already been forwarded, so the terminal
            # line only needs to name the mechanism — and whether it cut a
            # partially-streamed answer or produced nothing at all (D1).
            error = _refusal_error("stop_reason=refusal", "", streamed_text=streamed_text)
        yield StreamUsageEvent(usage=usage)
        yield StreamEndEvent(stop_reason=mapped, usage=usage, error=error)


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

GOOGLE_API_URL = "https://generativelanguage.googleapis.com"


class GoogleClient:
    """Minimal Gemini client: ``streamGenerateContent?alt=sse``."""

    def __init__(
        self,
        base_url: str = GOOGLE_API_URL,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 600.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=_stream_timeout(timeout))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    def _build_body(self, request: ChatRequest) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        for message in request.messages:
            # The `if parts or ...` guard below already skips an assistant
            # turn with NO parts, but a whitespace-only text still renders
            # one. Route through the shared predicate so all three clients
            # agree on what an empty assistant turn is; see
            # `_is_empty_assistant`.
            if _is_empty_assistant(message):
                continue
            role = "user" if message.role == "user" else "model"
            parts: list[dict[str, Any]] = [{"text": message.text}] if message.text else []
            for block in message.content:
                if isinstance(block, ImageContent):
                    parts.append(
                        {"inline_data": {"mime_type": block.mime_type, "data": block.data}}
                    )
            if message.role == "assistant" and message.tool_calls:
                parts.extend(
                    {"functionCall": {"name": call.name, "args": call.arguments}}
                    for call in message.tool_calls
                )
            if message.role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": message.tool_name or "",
                                    "response": {"content": self._tool_response_content(message)},
                                }
                            }
                        ],
                    }
                )
                continue
            if parts or message.role == "user":
                contents.append({"role": role, "parts": parts or [{"text": ""}]})

        body: dict[str, Any] = {"contents": contents}
        if request.system_blocks:
            # Gemini's dedicated system slot (not folded into a user turn).
            body["systemInstruction"] = {
                "parts": [{"text": block} for block in request.system_blocks]
            }
        generation_config: dict[str, Any] = {}
        max_tokens = _effective_max_tokens(request)
        if max_tokens and max_tokens > 0:
            generation_config["maxOutputTokens"] = max_tokens
        generation_config.update(_sampling_params(request, top_p_key="topP"))
        # No effort key here, deliberately: Gemini's named thinking tiers belong
        # to the Interactions API, while this client speaks `generateContent`,
        # where the shipped 2.5-series models take a token budget instead. No
        # Gemini model is given an effort ladder for that reason
        # (``model.effort``), so `_reasoning_effort` would return None anyway —
        # the note is here because its absence is a decision, not an oversight.
        if request.stop_sequences:
            generation_config["stopSequences"] = list(request.stop_sequences)
        body["generationConfig"] = generation_config
        if request.tools:
            body["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters or {"type": "object", "properties": {}},
                        }
                        for tool in request.tools
                    ]
                }
            ]
            # Gemini has no per-request tool_choice field; it takes the mode
            # under toolConfig.functionCallingConfig instead. It defaults to
            # AUTO, so before an aside started sending the live tool schema this
            # branch was never reached with tools present and the mode never
            # mattered. Now that complete_aside sends tools with
            # tool_choice="none" (to stay on the working turn's cache prefix),
            # the mode MUST be pinned or the aside could newly call a tool — the
            # opposite of the "reads the turn, calls nothing" contract. Map the
            # cross-provider tool_choice onto Gemini's mode; unknown values fall
            # back to AUTO, matching the other builders' safe default.
            body["toolConfig"] = {
                "functionCallingConfig": {
                    "mode": {"auto": "AUTO", "none": "NONE", "required": "ANY"}.get(
                        request.tool_choice, "AUTO"
                    )
                }
            }
        return body

    @staticmethod
    def _tool_response_content(message: Message) -> str:
        """Render a tool result from its content blocks — never ``message.text``.

        Same policy as the other two clients: text blocks concatenated,
        image-only results summarized, empty results backfilled so the
        provider never receives an empty ``functionResponse``.
        """
        texts: list[str] = []
        has_image = False
        for block in message.content:
            if isinstance(block, TextContent):
                if block.text:
                    texts.append(block.text)
            elif isinstance(block, ImageContent):
                has_image = True
        if texts and not has_image:
            return "".join(texts)
        if texts:
            return "".join(texts) + "\n[attached image content omitted]"
        if has_image:
            return "[tool returned image content]"
        return EMPTY_TOOL_RESULT_TEXT

    async def stream(
        self,
        request: ChatRequest,
        api_key: str | None,
        oauth_access: "OAuthAccess | None" = None,
    ) -> AsyncIterator[StreamEvent]:
        url = (
            f"{self._base_url}/v1beta/models/{request.model.model_id}:streamGenerateContent?alt=sse"
        )
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["x-goog-api-key"] = api_key
        usage: Usage | None = None
        stop_reason = "stop"
        # Gemini's abnormal finish reasons split into two families, and the
        # split decides the diagnosis the user reads (review R1-2). Content
        # classifiers (the allowlist below) are REFUSALS — "rephrase or switch
        # models" is the right advice. Model/tooling defects
        # (MALFORMED_FUNCTION_CALL, UNEXPECTED_TOOL_CALL…) are ERRORS — a plain
        # retry usually works, and calling them refusals steers the user away
        # from it. Both used to collapse into the "stop" fallback of the
        # finishReason map below: the silent empty turn this field exists to
        # prevent. The marker is kept verbatim either way so the visible line
        # names WHICH terminal fired. Unknown reasons land on the error side:
        # "the model refused" is a strong claim to make about a marker we have
        # never seen, while an error line with the verbatim marker stays true.
        refusal_marker: str | None = None
        defect_marker: str | None = None
        streamed_text = False
        # Gemini returns one complete functionCall per part with no ids and
        # no part indexes, so the harness must mint both. They must be UNIQUE
        # per response: the loop dedups tool calls by id (first-wins), and a
        # parallel batch of same-tool calls with a shared id silently drops
        # every call after the first — the model believes two reads ran when
        # only one did. The index doubles as the stream slot used to assemble
        # argument deltas, matching the OpenAI per-index contract.
        call_index = 0

        async with self._http.stream(
            "POST", url, json=self._build_body(request), headers=headers
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                raise_for_status(response)
            async for data in _iter_sse_lines(response):
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for candidate in chunk.get("candidates") or []:
                    for part in (candidate.get("content") or {}).get("parts") or []:
                        text = part.get("text")
                        if text:
                            streamed_text = True
                            yield StreamTextDelta(delta=text)
                        function_call = part.get("functionCall")
                        if function_call:
                            name = function_call.get("name")
                            yield StreamToolCallDelta(
                                index=call_index,
                                id=f"fc_{call_index}_{name or 'call'}",
                                name=name,
                                argument_delta=json.dumps(function_call.get("args") or {}),
                            )
                            call_index += 1
                    if candidate.get("finishReason"):
                        reason = str(candidate["finishReason"])
                        normal = {"STOP": "stop", "MAX_TOKENS": "length", "TOOL_USE": "toolUse"}
                        refusals = (
                            "SAFETY",
                            "RECITATION",
                            "PROHIBITED_CONTENT",
                            "SPII",
                            "BLOCKLIST",
                            "IMAGE_SAFETY",
                            "OTHER",
                        )
                        if reason in normal:
                            stop_reason = normal[reason]
                        elif reason in refusals:
                            stop_reason = "refusal"
                            refusal_marker = f"finishReason={reason}"
                        else:
                            stop_reason = "error"
                            defect_marker = f"finishReason={reason}"
                # A prompt blocked outright never produces a candidate, only
                # ``promptFeedback.blockReason`` — without this the stream ends
                # on the "stop" default with nothing on screen.
                feedback = chunk.get("promptFeedback")
                if isinstance(feedback, Mapping) and feedback.get("blockReason"):
                    stop_reason = "refusal"
                    refusal_marker = f"promptFeedback.blockReason={feedback['blockReason']}"
                raw_usage = chunk.get("usageMetadata")
                if raw_usage:
                    # Gemini reports thinking and tool-use prompt tokens outside
                    # the ordinary candidate/prompt counters, but includes both in
                    # ``totalTokenCount`` and bills them at the corresponding
                    # output/input rates. Dropping them makes a thinking or grounded
                    # call look materially cheaper than the provider's own usage.
                    prompt_tokens = _usage_token(raw_usage, "promptTokenCount")
                    tool_tokens = _usage_token(raw_usage, "toolUsePromptTokenCount")
                    candidate_tokens = _usage_token(raw_usage, "candidatesTokenCount")
                    thought_tokens = _usage_token(raw_usage, "thoughtsTokenCount")
                    input_tokens = prompt_tokens + tool_tokens
                    usage = Usage(
                        input_tokens=input_tokens,
                        output_tokens=candidate_tokens + thought_tokens,
                        cache_read_tokens=_usage_token(raw_usage, "cachedContentTokenCount"),
                        context_tokens=input_tokens or None,
                        reasoning_tokens=thought_tokens,
                    )

        if usage is not None:
            yield StreamUsageEvent(usage=usage)
        error: str | None = None
        if stop_reason == "refusal":
            # Gemini sends no refusal prose; any text it did stream has been
            # forwarded already, so the line names the classifier that fired —
            # and whether it cut a partial answer or produced nothing (D1).
            error = _refusal_error(
                refusal_marker or "finishReason=OTHER", "", streamed_text=streamed_text
            )
        elif stop_reason == "error":
            error = f"model call failed ({defect_marker or 'unknown finishReason'})"
        yield StreamEndEvent(stop_reason=stop_reason, usage=usage, error=error)


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------


class MockClient:
    """Deterministic canned stream for ``--hosting test``.

    Emits two text deltas + usage + end; when the last user message contains
    ``[tool]`` it emits one tool call (``echo`` with ``{"text": "hi"}``) and
    stops with ``toolUse`` instead. ``[refuse]`` ends the stream as a refusal
    with a canned provider message — the only way to exercise the whole
    refusal path (loop → event → TUI notice → transcript replay) against a
    real running app without needing a provider to actually decline.
    """

    async def stream(
        self,
        request: ChatRequest,
        api_key: str | None,
        oauth_access: "OAuthAccess | None" = None,
    ) -> AsyncIterator[StreamEvent]:
        if any("[refuse]" in message.text for message in request.messages):
            yield StreamUsageEvent(usage=Usage(input_tokens=10, output_tokens=0))
            yield StreamEndEvent(
                stop_reason="refusal",
                usage=Usage(input_tokens=10, output_tokens=0),
                error=_refusal_error("stop_reason=mock_refusal", "I can't help with that request."),
            )
            return
        wants_tool = any("[tool]" in message.text for message in request.messages)
        if wants_tool:
            yield StreamToolCallDelta(index=0, id="call_mock_1", name="echo")
            yield StreamToolCallDelta(index=0, argument_delta=json.dumps({"text": "hi"}))
            yield StreamUsageEvent(usage=Usage(input_tokens=10, output_tokens=5))
            yield StreamEndEvent(
                stop_reason="toolUse", usage=Usage(input_tokens=10, output_tokens=5)
            )
            return
        yield StreamTextDelta(delta="Hello")
        yield StreamTextDelta(delta=" from the mock provider!")
        yield StreamUsageEvent(usage=Usage(input_tokens=10, output_tokens=8))
        yield StreamEndEvent(stop_reason="stop", usage=Usage(input_tokens=10, output_tokens=8))


def client_for_spec(
    spec: ModelSpec,
    *,
    http_client: httpx.AsyncClient | None = None,
    openai_api: str = "responses",
    anthropic_cache_ttl_1h_min_context_tokens: int = 0,
) -> WireClient:
    """Build the wire client for a ``ModelSpec`` via the provider registry.

    Unknown providers raise :class:`ValueError` — the legacy fallback to the
    local ollama endpoint silently served the wrong wire shape.

    ``anthropic_cache_ttl_1h_min_context_tokens`` is the configured
    ``providers.anthropic.cache_ttl_1h_min_context_tokens`` (see
    ``AnthropicClient._cache_ttl_for``); it only reaches the Anthropic wire.
    Like ``openai_api`` it is a per-provider setting resolved by the caller
    (``SessionStreamFn._client_for``) so this function stays settings-free.
    """
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition(spec.provider)
    if definition is None:
        raise ValueError(f"Unknown provider: {spec.provider!r}")
    wire = definition.wire
    if wire == "mock":
        return MockClient()
    if wire == "anthropic":
        base = spec.base_url or (definition.base_url if definition else None) or ANTHROPIC_API_URL
        return AnthropicClient(
            base_url=base,
            http_client=http_client,
            cache_ttl_1h_min_context_tokens=anthropic_cache_ttl_1h_min_context_tokens,
        )
    if wire == "google":
        base = spec.base_url or GOOGLE_API_URL
        return GoogleClient(base_url=base, http_client=http_client)
    base = (
        spec.base_url
        or (definition.base_url if definition else None)
        or "http://localhost:11434/v1"
    )
    extra_headers = None
    if spec.provider == "openrouter":
        extra_headers = openrouter_attribution_headers()
    elif spec.provider == "kimi":
        # The OAuth grant is minted against a device fingerprint; every
        # inference call must present the same X-Msh-* headers or the provider
        # can reject the session as an unknown device (kimi.py invariant).
        from local_operator.providers.oauth.kimi import kimi_common_headers

        extra_headers = kimi_common_headers()
    return OpenAICompatClient(
        base_url=base,
        http_client=http_client,
        extra_headers=extra_headers,
        openai_api=openai_api if spec.provider == "openai" else "chat_completions",
        # Suppressed only when the spec names a base the registry did NOT
        # supply, which is a deliberate endpoint override (a gateway, a proxy)
        # and must not be second-guessed per credential. `build_model_spec`
        # copies `definition.base_url` onto every spec, so testing
        # `spec.base_url` alone would disable the OAuth host for every request
        # and silently send the coding-plan bearer to the API-key platform.
        oauth_base_url=(
            definition.oauth_base_url
            if not spec.base_url or spec.base_url == definition.base_url
            else None
        ),
    )
