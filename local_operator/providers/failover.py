"""Credential rotation, model fallback chains, and failover streaming.

Three independent tiers from the provider-rotation design (§5):

- **Tier 1** — a/b/c credential rotation inside one provider
  (:func:`resolve_next_key`): initial resolve / force-refresh same account /
  rotate to a sibling. 403 and usage-limit errors skip the refresh step
  (a valid-but-denied token cannot be fixed by refreshing). Attempted keys
  are tracked in a set and capped at 64 so sibling pools cannot loop.
- **Tier 2** — model fallback chains (:func:`resolve_chain` /
  :func:`expand_fallback_candidates`), configured in config.yml
  ``values.retry.fallbackChains``.
- **High level** — :func:`stream_with_failover` composes both: rotate the
  credential, then walk the chain, backing off between attempts.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    RenderedStreamError,
    StreamEvent,
)
from local_operator.model.effort import EFFORT_ORDER, resolve_effort

if TYPE_CHECKING:  # import cycle: both modules import this one at runtime
    from local_operator.providers.auth_store import OAuthAccess, StoredCredential
    from local_operator.providers.clients import WireClient

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

#: The KINDS of provider failure. This is the one thing an error frame has to
#: get across — what SORT of problem this is, and therefore what to do about
#: it — and it is exactly what the reported frame `✕ HTTP 404:` did not say.
#: Names, not codes: nobody reads a status and knows whether to wait, re-login,
#: or fix their request.
ProviderErrorKind = Literal[
    "quota",  # rate limited or out of quota: wait, or use another credential
    "auth",  # the bearer was rejected: re-login or rotate
    "timeout",  # the request or the stream ran out of time: retry
    "transient",  # 5xx, dropped connection, stalled stream: retry
    "request",  # the provider read the request and refused it: fix the request
    "aborted",  # the user stopped it
    "unknown",
]

#: What each kind is CALLED in the frame the user reads. Lowercase to match the
#: harness's notice voice ("turn failed", "interrupted", "usage panel
#: unavailable") — a capitalised label in that column reads as a different app.
_KIND_LABELS: dict[str, str] = {
    "quota": "rate limit or quota exceeded",
    "auth": "authentication failed",
    "timeout": "provider timeout",
    "transient": "transient provider error",
    "request": "invalid request",
    "aborted": "aborted",
    "unknown": "provider error",
}

#: Substrings that mean "you have run out" UNAMBIGUOUSLY, wherever the provider
#: chose to put them. Matched against the provider's own message because the
#: status alone is not enough: anthropic answers an OAuth credential used
#: off-Claude-Code with an opaque 429, and google/openrouter both report quota
#: through a 403 or a bare body.
_USAGE_LIMIT_MARKERS = (
    "quota",
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "limit reached",
)

#: Words that USUALLY mean exhaustion and sometimes mean permission. They are
#: separated from the set above because they are the ones that misfire: google's
#: real 403 PERMISSION_DENIED text is "Request had insufficient authentication
#: scopes.", which the combined set read as a quota exhaustion and rendered as
#: `rate limit or quota exceeded (HTTP 403)` — sending the user to wait out a
#: problem only a re-login or a scope change clears. So on a status that is
#: ITSELF about the bearer (401/403), only the unambiguous set counts.
_WEAK_USAGE_LIMIT_MARKERS = ("usage", "insufficient")

_TIMEOUT_MARKERS = ("timeout", "timed out", "deadline exceeded", "stream stalled")


def _is_usage_limit(status: int | None, message: str) -> bool:
    """429, or a body that SAYS it ran out.

    Three bounds, each of which was a real misclassification:

    - **Not above 500.** A 5xx is the server failing, so a 5xx body mentioning a
      limit is still the server failing and still worth retrying. Without this an
      empty-bodied 507 read as quota, because :func:`_describe_bare_status` fills
      the message from the status phrase and "Insufficient Storage" contains
      ``insufficient``. (Found by enumerating every ``HTTPStatus`` phrase against
      these markers. 408/504 collide with the timeout markers too, harmlessly:
      both are already timeouts by status.)
    - **Unambiguous markers only on 401/403**, per
      :data:`_WEAK_USAGE_LIMIT_MARKERS`.
    - **402 is NOT here**, though it is a quota problem: this predicate also
      drives ``AuthStore.rotate_sibling``, which PRESERVES the sticky credential
      for a usage limit on the reasoning that the same account is first choice
      again once the window passes. A spent balance does not come back in sixty
      seconds, so 402 is named as quota in :func:`_classify_fields` and routed
      past the pointless token refresh by
      :func:`is_direct_credential_rotation_error`, without claiming that its
      credential is worth staying on.
    """
    if status == 429:
        return True
    if status is not None and status >= 500:
        return False
    lowered = message.lower()
    if any(marker in lowered for marker in _USAGE_LIMIT_MARKERS):
        return True
    if status in (401, 403):
        return False
    return any(marker in lowered for marker in _WEAK_USAGE_LIMIT_MARKERS)


def _classify_fields(
    status: int | None, message: str, *, retryable: bool, auth_error: bool
) -> ProviderErrorKind:
    """The kind, decided ONCE from the raw fields.

    ``message`` MUST be the provider's own words, and empty when it sent none.
    Never a message the harness synthesized: text this function wrote itself
    coming back through the markers is how an empty-bodied 507 became a quota
    exhaustion. :class:`ProviderError` classifies before it substitutes its
    floor text, and :func:`wrap_transport_error` states its kind outright.

    Order is the whole content of this function, so it is written down:

    1. A 401 is always the bearer. Nothing else produces one, and a 401 body
       mentioning "insufficient permissions" must not read as a quota problem.
       :func:`_is_usage_limit` extends the same care to 403 for the same reason.
    2. Quota next, ahead of 403: google and openrouter both report exhausted
       quota with a 403, and telling that user "authentication failed" sends
       them to re-login for a problem that a login cannot fix. This is why the
       display order differs from :func:`is_auth_error`, which stays a pure
       401/403 rotation predicate. 402 joins it here rather than inside
       ``_is_usage_limit`` — see that function for the sticky-credential reason.
    3. Timeout ahead of transient: both retry, but only one of them tells the
       user the request was too slow rather than the provider being unwell.
    4. Anything else in the 4xx range is a request the provider READ and
       refused — retrying it unchanged burns quota for the same answer.
    """
    if status == 401:
        return "auth"
    if status == 402 or _is_usage_limit(status, message):
        return "quota"
    if auth_error or status == 403:
        return "auth"
    if status in (408, 504) or any(m in message.lower() for m in _TIMEOUT_MARKERS):
        return "timeout"
    if retryable or (status is not None and status >= 500):
        return "transient"
    if status is not None and 400 <= status < 500:
        return "request"
    return "unknown"


def _describe_bare_status(status: int | None) -> str:
    """A message for a provider that sent NONE.

    An empty message is its own defect: it was half of the reported
    ``✕ HTTP 404:`` frame, and no amount of downstream care can render text
    that was never captured. Providers really do answer with an empty body —
    a gateway rejecting an unknown model is the common one — so the floor is
    set here, at construction, rather than trusted to every call site.
    """
    if status is None:
        return "the provider failed without reporting a reason"
    try:
        phrase = HTTPStatus(status).phrase
    except ValueError:  # a status no stdlib release has heard of
        return f"the provider sent no error message with its HTTP {status}"
    return f"{phrase} — the provider sent no error message"


def _format_retry_delay(delay_ms: int) -> str:
    """A wait as the phrase a waiting human wants: ``42s``, ``1m30s``.

    Neither existing formatter fits: ``wake.format_duration`` renders only
    EXACT units (a 41.6s ``Retry-After`` becomes ``41600ms``) and
    ``tui.widgets.tool_card.format_duration`` is a widget-layer import that has
    no business inside the provider driver. Rounded to the second because the
    fact being communicated is "come back in about this long".
    """
    if delay_ms < 1000:
        return f"{delay_ms}ms"
    seconds = round(delay_ms / 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes}m" if minutes else f"{hours}h"
    if minutes:
        return f"{minutes}m{seconds}s" if seconds else f"{minutes}m"
    return f"{seconds}s"


class ProviderError(RenderedStreamError):
    """A provider call failed, in a way the frame has to NAME.

    ``status`` is the HTTP status when known. ``retryable`` reflects whether
    the SAME credential may succeed again (429/5xx/timeout/network); auth
    errors are retryable only via rotation.

    A provider's answer, not a defect: ``RenderedStreamError`` tells the loop
    that ``__str__`` below is the complete diagnosis, so it logs the line
    without a stack. That makes ``__str__`` the whole of what the user is told,
    which is why it leads with :attr:`kind` and why an empty ``message`` is
    refused at construction rather than rendered as ``HTTP 404:``.

    ``message`` stays the provider's OWN words (the classifiers read it, and it
    usually says the useful thing — which limit, when it resets); the kind and
    the wait are composed in only on the way out.
    """

    def __init__(
        self,
        status: int | None,
        message: str,
        *,
        retryable: bool = False,
        retry_after_ms: int | None = None,
        auth_error: bool = False,
        kind: ProviderErrorKind | None = None,
    ) -> None:
        provider_text = message.strip() if isinstance(message, str) else str(message)
        #: Classified BEFORE the floor text is substituted, so the classifier only
        #: ever reads the provider's own words. Feeding it a message the harness
        #: wrote is how an empty-bodied 507 became a quota exhaustion —
        #: "Insufficient Storage" is the status phrase, not a provider saying it
        #: ran out. Empty text matches no marker, so a wordless failure is
        #: classified from its status alone, which is all the evidence there is.
        self.kind: ProviderErrorKind = kind or _classify_fields(
            status, provider_text, retryable=retryable, auth_error=auth_error
        )
        text = provider_text or _describe_bare_status(status)
        super().__init__(text)
        self.status = status
        self.message = text
        self.retryable = retryable
        self.retry_after_ms = retry_after_ms
        self.auth_error = auth_error

    def __str__(self) -> str:
        """``<kind> (HTTP <status>, retry in <wait>): <the provider's words>``.

        The kind comes first because it is the actionable part, and the wait
        rides in the parenthetical rather than the tail so that "try again in
        40s" survives a long provider message being wrapped or clipped.

        Two cases render as the bare message, because a label would restate what
        the text already says. An abort: the user pressed the key. And an error
        with NO status and NO kind — which is only ever one the harness wrote
        about itself ("No API key configured for provider 'openai'", "Failover
        exhausted for …"); prefixing those with ``provider error:`` produced a
        stutter and named a provider that was never called.
        """
        if self.kind == "aborted" or (self.kind == "unknown" and self.status is None):
            return self.message
        facts = []
        if self.status is not None:
            facts.append(f"HTTP {self.status}")
        if self.retry_after_ms:
            facts.append(f"retry in {_format_retry_delay(self.retry_after_ms)}")
        label = _KIND_LABELS.get(self.kind, _KIND_LABELS["unknown"])
        detail = f" ({', '.join(facts)})" if facts else ""
        return f"{label}{detail}: {self.message}"


#: Provider wordings that mean "an image block in this request is unusable".
#: Matched on the provider's OWN message, because no status distinguishes it:
#: it arrives as a plain 400 alongside every other malformed-request refusal.
#:
#: This is worth naming as its own condition because of how it FAILS. The image
#: lives in the conversation history, so once one is rejected every subsequent
#: request carries it again and gets the same 400 — including compaction, which
#: has to send the history to summarise it. The session is then unrecoverable
#: from inside: reload replays the same blocks, and /compact cannot run either.
#: Anthropic have had this open for over a year across many reports
#: (anthropics/claude-code#19031, #24387, #47391, #50708), where the same bytes
#: are accepted for hours and then refused, so the client cannot prevent it by
#: validating on the way in — it can only stop being poisoned by it.
_IMAGE_REJECTION_MARKERS = (
    "could not process image",
    "image could not be processed",
    "unable to process image",
    "invalid image",
    "does not match the provided media type",
    "image exceeds",
    "unsupported image",
    # The DIMENSION refusals, which the list above missed entirely and which
    # are the ones a long screenshot-taking session actually hits. Anthropic's
    # wording is "At least one of the image dimensions exceed max allowed
    # size: 8000 pixels" for a single oversized image, and "... max allowed
    # size for many-image requests: 2000 pixels" once a request carries more
    # than twenty images.
    #
    # The second is the nasty one: no image changed, the CONVERSATION grew, so
    # a block that was accepted for a hundred turns starts being refused and
    # keeps being refused forever. Without a marker here the degrade never
    # fired, and the session answered every prompt — and every /compact — with
    # the same 400 until it was abandoned. Observed live on 2026-08-18.
    #
    # Matched on the shared prefix so both variants and any future pixel
    # ceiling are covered by one marker, since the number is the part that
    # moves.
    "image dimensions exceed max allowed size",
)


def is_image_rejection(error: BaseException | str) -> bool:
    """Did the provider refuse the request BECAUSE of an image block?

    Accepts either the exception or its RENDERED form, because the two callers
    hold different things: the client layer catches a :class:`ProviderError`,
    while ``AgentEndEvent.error`` carries the already-rendered
    ``"<kind> (HTTP <status>): <provider's words>"`` string that the UI shows.

    Gated on the kind as well as the wording, either way round. A 5xx that
    happens to mention an image is the provider failing, not a bad block, and
    the degrade this drives is STICKY — so misreading weather as a poisoned
    image would quietly strip every image from the rest of the session for no
    reason. On the rendered form the gate is the kind's own label, so the two
    paths agree by construction instead of by a duplicated status range.
    """
    if isinstance(error, ProviderError):
        if error.status is not None and not 400 <= error.status < 500:
            return False
        haystack = error.message.lower()
    else:
        text = error if isinstance(error, str) else str(error)
        haystack = text.lower()
        if not haystack.startswith(_KIND_LABELS["request"]):
            return False
    return any(marker in haystack for marker in _IMAGE_REJECTION_MARKERS)


def classify_provider_error(error: BaseException) -> ProviderErrorKind:
    """The failure kind for anything the harness can catch.

    A raw exception is read best-effort from its CLASS and, for the timeout
    family, its text — which is enough for the stdlib timeout family and honest
    about the rest: a bare ``ValueError`` is a defect, not weather, and must NOT
    come back retryable.

    Deliberately NOT through the usage markers. The text of a raw exception is
    the harness's own words — a client's ``KeyError('usage')``, a parser's
    ``ValueError('insufficient data in chunk')`` — and reading them as evidence
    about the user's account is how the harness came to diagnose its own bugs as
    ``rate limit or quota exceeded``. :func:`wrap_transport_error` states its
    kind outright for exactly this reason; the same care belongs here, because
    this is the entry point every caller holding an unwrapped exception uses.
    A class name IS legitimate evidence of a timeout. Nothing in an exception's
    text is evidence about anyone's quota.

    A caller holding a TRANSPORT exception — ``httpx.TransportError`` and
    friends — should pass it through :func:`wrap_transport_error` first. Only
    the raiser knows that a dropped connection is weather; this function cannot
    tell that from a parse failure, and guessing in either direction is worse
    than being asked.
    """
    if isinstance(error, ProviderError):
        return error.kind
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return "timeout"
    haystack = f"{type(error).__name__}: {error}".lower()
    if any(marker in haystack for marker in _TIMEOUT_MARKERS):
        return "timeout"
    return "unknown"


def is_auth_error(error: BaseException) -> bool:
    """401/403-class failures (bearer rejected or denied).

    Deliberately NOT ``kind == "auth"``: this is the rotation predicate, and a
    403 whose body says "quota exceeded" must still rotate to a sibling
    credential even though the user is told it is a quota problem.
    """
    if isinstance(error, ProviderError):
        return error.auth_error or error.status in (401, 403)
    return False


def is_usage_limit_error(error: BaseException) -> bool:
    """402/429 or provider-reported quota/rate exhaustion."""
    if not isinstance(error, ProviderError):
        return False
    return _is_usage_limit(error.status, error.message)


def is_timeout_error(error: BaseException) -> bool:
    """The request or the stream ran out of time (408/504, or a timeout
    exception the driver wrapped). Retryable: the next attempt may be served
    by a healthy node."""
    return classify_provider_error(error) == "timeout"


def is_transient_error(error: BaseException) -> bool:
    """Worth trying again UNCHANGED — 5xx, a dropped connection, a stalled
    stream, a timeout.

    Excludes quota (a wait, not a blip — retrying it immediately burns the
    quota that is already gone), auth (no retry can mint a valid bearer) and
    a refused request (the same bytes get the same answer). Those three are
    the errors that must NOT be retried, and this is the one place that says so.
    """
    return classify_provider_error(error) in ("transient", "timeout")


def is_invalidated_credential_error(error: BaseException) -> bool:
    """The credential was EXPLICITLY revoked by the IdP — soft-delete worthy.

    Narrow on purpose (PR-03): only true invalidation signals qualify —
    anthropic ``invalid_request_error`` + ``revoked``, openai
    ``token_revoked``/``invalid_grant``, or a generic ``revoked`` /
    ``invalid_grant`` payload. An ordinary expired-token 401 is NOT
    invalidated: it goes through the refresh step (b) and the row stays
    enabled. Generic "invalid"/"unauthorized"/"expired" 401s never
    soft-delete.
    """
    if not isinstance(error, ProviderError):
        return False
    if error.status != 401:
        return False
    lowered = error.message.lower()
    if "invalid_grant" in lowered or "token_revoked" in lowered:
        return True
    # anthropic surfaces revocation as invalid_request_error + "revoked";
    # "revoked" alone is the generic marker both shapes share.
    return "revoked" in lowered


def is_direct_credential_rotation_error(error: BaseException) -> bool:
    """Skip the refresh-same-account step for these: refreshing a
    valid-but-denied token cannot help, so rotate through the pool.

    402 is listed explicitly rather than through :func:`is_usage_limit_error`,
    which deliberately excludes it: a refreshed token still has no credits, so
    the refresh step is as pointless here as for a 403 — but a spent balance is
    not a window that reopens, so the credential must not keep its sticky place.

    Server-side failures (5xx, including Anthropic's 529 ``overloaded_error``,
    and transport timeouts) are here too. A refresh cannot fix the PROVIDER
    being overloaded, so the refresh step is equally pointless, and the sticky
    credential is not at fault so it keeps its place — that is exactly the
    contract this predicate expresses.

    The same answer decides which errors may cycle EVERY sibling rather than
    switching once and stopping. The ``legacy_auth_switch_used`` cap in
    :class:`AuthRetryKeyState` exists for the ordinary-401 case, where a second,
    third and fourth bearer are all equally likely to be rejected and cycling
    them only delays a login prompt the user has to answer anyway. That is the
    wrong rule for a pool of accounts that fail INDEPENDENTLY: with four
    Anthropic accounts and a 529 storm the cap stopped after the second account
    and reported the turn dead while two healthy accounts were never asked --
    the reported failure. Quota is the same story, since one account's spent
    weekly window says nothing about the other three. So exhaustion, payment,
    permission and server-side faults may walk the whole pool; only an ordinary
    401 keeps the single-switch cap.
    """
    if not isinstance(error, ProviderError):
        return False
    return (
        is_usage_limit_error(error) or error.status in (402, 403) or is_server_side_failure(error)
    )


def is_server_side_failure(error: BaseException) -> bool:
    """The PROVIDER failed, not the credential: 5xx, or a timeout/transport blip.

    Kept separate from :func:`is_usage_limit_error` because the two want
    opposite blocking behaviour — a quota error blocks the credential for the
    reset window, while an overloaded provider must not get four credentials
    blocked for a fault none of them caused. ``AuthStore.rotate_sibling``
    deprioritises instead of blocking on the strength of this predicate.
    """
    if not isinstance(error, ProviderError):
        return False
    if error.kind in ("timeout", "transient"):
        return True
    return error.status is not None and error.status >= 500


def retry_after_ms_from_error(error: BaseException) -> int | None:
    if isinstance(error, ProviderError):
        return error.retry_after_ms
    return None


#: How much a failure tells the USER, which is what decides who owns the
#: reported-error slot when several attempts fail. The reported frame
#: ``✕ HTTP 404:`` was a real 429 quota exhaustion on the requested model,
#: overwritten by a bare 404 from a fallback selector the account cannot
#: serve — last-wins reported the least informative failure of the set.
_KIND_DIAGNOSTIC_RANK: dict[str, int] = {
    "quota": 5,  # names the user's actual problem AND when it clears
    "auth": 4,  # names a credential they have to fix
    "request": 3,  # names something wrong with the request itself
    "timeout": 2,
    "transient": 2,
    "unknown": 1,
    "aborted": 0,  # they already know; never worth reporting over a real failure
}


def error_report_score(error: ProviderError, *, primary: bool) -> int:
    """Rank a failure for the reported-error slot; higher wins.

    The PRIMARY selector dominates every fallback: the user asked for that
    model, so its failure is the news of the turn, and a fallback's failure is
    a problem with the configured chain (logged, and reported only when the
    primary never failed at all).
    """
    return (10 if primary else 0) + _KIND_DIAGNOSTIC_RANK.get(error.kind, 1)


# ---------------------------------------------------------------------------
# Tier 1 — a/b/c credential rotation
# ---------------------------------------------------------------------------

AUTH_RETRY_MAX_ATTEMPTS = 64


@dataclasses.dataclass
class ApiKeyResolveContext:
    """What the resolver needs to pick a key: ``error is None`` ⇒ initial
    resolve; ``last_chance`` ⇒ rotate to a sibling credential."""

    last_chance: bool
    error: BaseException | None = None
    previous_key: str | None = None


ApiKeyResolver = Callable[[ApiKeyResolveContext], Awaitable[str | None] | str | None]


@dataclasses.dataclass
class AuthRetryKeyState:
    """Mutable rotation state shared across attempts for one request.

    ``legacy_auth_switch_used`` carries the legacy auth-switch semantics: an
    ORDINARY 401 gets exactly one refresh-same-account plus one sibling
    switch, then rotation is exhausted. Usage-limit/403 errors skip the
    refresh step and may cycle every distinct sibling instead.
    """

    #: Every DISTINCT bearer rotation has returned for this request. Also the
    #: retry budget's evidence that rotation happened at all: see
    #: :func:`_request_has_rotated`.
    attempted_keys: set[str] = dataclasses.field(default_factory=set)
    last_key: str | None = None
    refreshed_current: bool = False
    legacy_auth_switch_used: bool = False
    attempts: int = 0


async def _call_resolver(resolver: ApiKeyResolver, ctx: ApiKeyResolveContext) -> str | None:
    """Call a resolver that may be sync or async and normalise the result."""
    result = resolver(ctx)
    if inspect.isawaitable(result):
        return await result
    return result


async def resolve_next_key(
    state: AuthRetryKeyState,
    resolver: ApiKeyResolver,
    error: BaseException | None = None,
    *,
    signal: AbortSignal | None = None,
) -> str | None:
    """Next key to try, or ``None`` when rotation is exhausted.

    - ``error is None`` → (a) initial resolve (cheap, cached token OK).
    - ``error`` → (b) force-refresh the same account, then (c) rotate to a
      sibling — except 403/usage-limit skip (b) entirely.

    Termination: key already attempted, ≥64 attempts, resolver gives up, the
    signal aborted, or — for an ordinary 401 — the single refresh + single
    sibling switch (``legacy_auth_switch_used``) has been spent.
    """
    if signal is not None and signal.aborted:
        return None
    if state.attempts >= AUTH_RETRY_MAX_ATTEMPTS:
        return None
    state.attempts += 1

    async def _accept(key: str | None) -> str | None:
        if key is None or key in state.attempted_keys:
            return None
        state.attempted_keys.add(key)
        state.last_key = key
        return key

    if error is None:
        return await _accept(
            await _call_resolver(resolver, ApiKeyResolveContext(last_chance=False))
        )

    direct_rotation = is_direct_credential_rotation_error(error)
    # Ordinary 401: one refresh + one sibling switch, then stop
    # (legacyAuthSwitchUsed). Usage-limit/403 may keep cycling siblings.
    if not direct_rotation and state.legacy_auth_switch_used:
        return None

    # (b) Force-refresh the same account — skipped for valid-but-denied errors.
    if not state.refreshed_current and not direct_rotation:
        state.refreshed_current = True
        key = await _accept(
            await _call_resolver(
                resolver,
                ApiKeyResolveContext(last_chance=False, error=error, previous_key=state.last_key),
            )
        )
        if key is not None:
            return key

    # (c) Rotate to a sibling credential.
    if not direct_rotation:
        state.legacy_auth_switch_used = True
    return await _accept(
        await _call_resolver(
            resolver,
            ApiKeyResolveContext(last_chance=True, error=error, previous_key=state.last_key),
        )
    )


# ---------------------------------------------------------------------------
# Tier 2 — model fallback chains
# ---------------------------------------------------------------------------

#: Config problems are reported to the LOG, never the terminal: this module runs
#: under a full-screen TUI that owns stderr.
logger = logging.getLogger("local_operator.providers.failover")

DEFAULT_CHAIN_KEY = "default"
SUPPORTED_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})

#: The same vocabulary as an ordinal ladder, for anything a person READS.
#: ``sorted()`` puts ``max`` between ``low`` and ``medium`` and sits ``minimal``
#: next to it - the two opposite ends of the scale, adjacent - which reads as
#: noise to someone who already knows the ladder from ``/effort`` (design round
#: 28).
#:
#: ``none`` is absent because :data:`SUPPORTED_EFFORTS` predates this and does
#: not admit it. An earlier version of this comment justified that as "no
#: reasoning is a property of the model, not a rung to fall back to", which the
#: set itself refutes: ``minimal`` is admitted, and no shipped model has a
#: ``minimal`` rung either, so that is the same kind of value being treated the
#: opposite way. (:func:`resolve_effort` clamps it to ``none`` on the gpt-5
#: family - the only tables carrying a ``none`` rung - and UP to ``low``
#: everywhere else.) So the boundary is the set's, not a semantic line:
#: recorded rather than rationalised, because the next person to move it should
#: know there is no principle holding it in place.
#:
#: What it costs, which is the part worth knowing before moving it: on the
#: Anthropic and o-series tables there is no way to say "retry with reasoning
#: off" in a chain hop at all. ``none`` is refused here, and ``minimal`` is not
#: a substitute because it clamps upward to ``low`` (design round 30).
CHAIN_EFFORT_LADDER = tuple(e for e in EFFORT_ORDER if e in SUPPORTED_EFFORTS)


@dataclasses.dataclass(frozen=True)
class FallbackTarget:
    """One resolved cascade entry.

    ``selector`` keeps routing compatible with the existing
    ``provider/model`` chain format. ``effort`` is optional because many
    providers either do not expose reasoning levels or should use their model
    default.
    """

    selector: str
    effort: str | None = None


def _chain_specificity(key: str, selector: str) -> int | None:
    """Higher = more specific; ``None`` = no match."""
    if key == selector:
        return 1 << 30  # exact beats every wildcard
    if key.endswith("/*"):
        prefix = key[:-2]
        if selector.startswith(prefix + "/"):
            return len(prefix)
    return None


def resolve_chain(selector: str, chains: Mapping[str, Sequence[Any]]) -> list[Any] | None:
    """Pick the fallback chain for ``selector`` by specificity:
    exact ``provider/model`` → longest matching wildcard prefix → ``default``.
    """
    best_key: str | None = None
    best_score = -1
    for key in chains:
        if key == DEFAULT_CHAIN_KEY:
            continue
        score = _chain_specificity(key, selector)
        if score is not None and score > best_score:
            best_score = score
            best_key = key
    if best_key is None:
        if DEFAULT_CHAIN_KEY in chains:
            return list(chains[DEFAULT_CHAIN_KEY])
        return None
    return list(chains[best_key])


def _fallback_target(entry: Any) -> FallbackTarget | None:
    """Normalize a legacy selector string or a provider/model/effort mapping."""
    effort: str | None = None
    if isinstance(entry, str):
        selector = entry.strip()
    elif isinstance(entry, Mapping):
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or entry.get("model_id") or "").strip()
        selector = str(entry.get("selector") or "").strip()
        if not selector and provider and model:
            selector = f"{provider}/{model}"
        raw_effort = entry.get("effort")
        if raw_effort is not None:
            effort = str(raw_effort).strip().lower()
            if effort not in SUPPORTED_EFFORTS:
                return None
    else:
        return None
    provider, model_id = parse_selector(selector)
    if not provider or not model_id:
        return None
    return FallbackTarget(selector=selector, effort=effort)


def expand_fallback_targets(selector: str, chain: Sequence[Any]) -> list[FallbackTarget]:
    """Materialize configured entries into unique provider/model/effort targets.

    ``provider/*`` keeps the failing model id. A mapping may explicitly repeat
    the current selector with a different effort; that is a real fallback
    route, while an unchanged legacy string is still suppressed.
    """
    _, _, bare_id = selector.partition("/")
    targets: list[FallbackTarget] = []
    for entry in chain:
        target = _fallback_target(entry)
        if target is None:
            continue
        if target.selector.endswith("/*"):
            target = dataclasses.replace(target, selector=f"{target.selector[:-1]}{bare_id}")
        if target.selector == selector and target.effort is None:
            continue
        if target not in targets:
            targets.append(target)
    return targets


def expand_fallback_candidates(selector: str, chain: Sequence[Any]) -> list[str]:
    """Backward-compatible selector-only view of :func:`expand_fallback_targets`."""
    return [target.selector for target in expand_fallback_targets(selector, chain)]


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------

BACKOFF_CAP_MS = 8000
BACKOFF_JITTER_FRACTION = 0.25
# Interactive sessions must never disappear into a provider's quota-reset
# window. A 429 can carry Retry-After values of many minutes or hours; sleeping
# that duration before credential rotation makes the TUI look hung and prevents
# configured fallback models from running. Short throttles get two same-key
# retries — burst limits and OAuth concurrency limits ("another request is
# already being processed") clear in seconds, and one attempt only ever saw
# the collision that created it — while long waits rotate or surface
# immediately. Two, not the full ``max_retries`` budget: each attempt sleeps
# the ADVERTISED delay, so this cap is also the longest a screen sits waiting
# on one credential before a sibling or fallback is tried.
MAX_USAGE_RETRY_AFTER_MS = 30_000
MAX_SAME_CREDENTIAL_USAGE_RETRIES = 2

# Same-credential retries for a SERVER-side fault (5xx/529/timeout) ONCE this
# request has rotated onto a second bearer. The first credential keeps the full
# configured budget -- a lone credential, an override bearer or a pool whose
# siblings are all spent must not lose retries it has nowhere else to spend --
# and the cap engages only when the multiplication it exists to prevent has
# actually begun. See `_request_has_rotated` for why this is
# observed rather than predicted.
MAX_SAME_CREDENTIAL_SERVER_RETRIES = 2

#: Hard ceiling on the requests one turn will aim at ONE PROVIDER for
#: server-side faults, across every credential it rotates through. The
#: per-credential cap above bounds each account; this bounds their PRODUCT,
#: which is the quantity that actually reaches that provider. Counted per
#: fallback target, because each target is a different service: rationing the
#: fallback for the primary's outage is how a chain stops being a chain.
#: Without it, widening rotation to walk the
#: whole pool multiplied the load by the pool size at exactly the moment the
#: provider was asking for less: 44 requests over ~190s for four accounts,
#: measured, against 22 before. Twelve keeps a four-account pool walkable (each
#: account still gets more than one look) while never exceeding the pre-change
#: behaviour, whatever the pool size.
MAX_SERVER_FAULT_REQUESTS_PER_TURN = 12

#: Attempts the FIRST bearer may spend on a server-side fault before the turn
#: tries another credential. It is deliberately below ``max_retries``: until a
#: rotation has happened nothing knows whether a sibling exists, and spending
#: the full configured budget in place is how four healthy accounts got two
#: attempts between them (the reported bug). Small enough to leave the turn
#: ceiling room for a real pool walk, large enough to ride out the brief blips
#: that make up most 5xx traffic. A bearer that turns out to be alone gets the
#: rest of its configured budget back on the next pass, because rotation
#: exhausting is what ends the walk.
MAX_FIRST_CREDENTIAL_SERVER_RETRIES = 3


def backoff_delay_ms(base_delay_ms: int, attempt: int, *, rng: random.Random | None = None) -> int:
    """``min(base * 2^(attempt-1), 8000)`` with 25% downward jitter."""
    raw = min(base_delay_ms * (2 ** max(0, attempt - 1)), BACKOFF_CAP_MS)
    jitter_source = rng or random
    return max(0, int(raw - raw * BACKOFF_JITTER_FRACTION * jitter_source.random()))


def _same_credential_retry_allowed(
    error: ProviderError,
    transport_retries: int,
    retry: "RetrySettings",
    *,
    has_rotated: bool = False,
    rotation_exhausted: bool = False,
    server_fault_requests: int = 0,
) -> bool:
    if not error.retryable:
        return False
    if is_usage_limit_error(error):
        if (error.retry_after_ms or 0) > MAX_USAGE_RETRY_AFTER_MS:
            return False
        return transport_retries < min(retry.max_retries, MAX_SAME_CREDENTIAL_USAGE_RETRIES)
    if is_server_side_failure(error):
        # The turn-wide ceiling applies unconditionally: it is about what the
        # PROVIDER receives, so it cannot depend on how the requests were
        # distributed across credentials.
        if server_fault_requests >= MAX_SERVER_FAULT_REQUESTS_PER_TURN:
            return False
        # Rotation comes FIRST while it still has somewhere to go.
        #
        # A provider-side fault is not the credential's fault, but another
        # ACCOUNT is nonetheless the thing most likely to succeed -- the pool
        # walk is the fix this whole change exists to deliver. Spending a full
        # `max_retries` on the first bearer before rotating starves that walk
        # against the turn ceiling: four accounts, and only two were ever tried,
        # which is the original reported bug arriving by a new route.
        #
        # So each bearer gets a small allowance and the turn moves on, until
        # rotation reports it has nowhere left to go. `has_rotated` and
        # `rotation_exhausted` are both OBSERVATIONS the driver made -- what
        # rotation already did -- never predictions about the credential table.
        if rotation_exhausted:
            # Rotation has been tried and there is no other credential, so the
            # small allowances below -- which exist ONLY to get a turn moving to
            # another account -- have nothing left to buy. The bearer in hand
            # gets the budget the user configured rather than an allowance
            # sized for a pool that does not exist.
            return transport_retries < retry.max_retries
        if has_rotated:
            return transport_retries < min(retry.max_retries, MAX_SAME_CREDENTIAL_SERVER_RETRIES)
        # Not yet rotated. The first bearer gets a SMALL allowance rather than
        # the configured budget, because nothing yet knows whether a sibling
        # exists and finding out is cheap: rotation either produces another
        # credential (in which case the turn should be walking the pool, not
        # burning ten attempts in place -- the reported bug) or is exhausted, and
        # exhaustion ends the walk without costing anything. This deliberately
        # does NOT consult the credential table: every version that did drifted
        # from what the cascade actually selects (blocked rows, credential
        # types, override bearers, rows split across tiers), and each drift took
        # retries away from someone who had nowhere else to spend them.
        return transport_retries < min(retry.max_retries, MAX_FIRST_CREDENTIAL_SERVER_RETRIES)
    return transport_retries < retry.max_retries


def _normalize_chain_entry(entry: Any, chain_key: str) -> Any:
    """One fallback-chain entry, checked and returned in its own shape.

    The declared shape is a list of ``provider/model`` selector strings, and
    until this existed that shape was assumed rather than checked: a config
    written with structured entries parsed cleanly, resolved cleanly, and then
    died in the expansion with ``'dict' object has no attribute 'endswith'``.
    Not at startup, where a config error belongs: on EVERY turn, because that
    is the first moment a fallback chain is walked.

    So a mapping is accepted rather than rejected. Two forms are honoured:

    - ``{provider, model}`` becomes the selector string, which is all it ever
      meant;
    - a mapping that ALSO carries ``effort`` - the "retry cheaper on failure"
      form - is a real routing decision the chain's (selector, effort) shape
      now supports, so the effort is carried on a cleaned mapping. Flattening
      it to a selector would silently discard the one key that makes the entry
      mean something different.

    The effort's VALUE is checked here too, even though :func:`_fallback_target`
    is what ultimately rejects it. That function returns ``None`` for an
    unreadable target and cannot say why, so a single typo used to delete a
    whole fallback hop in silence: the operator configured failover, got none
    during an outage, and nothing connected it to the YAML (review round 29).
    It belongs here rather than at the point of rejection because this is the
    only layer that still holds the user's own text - the key it was written
    under and the value they typed - so the message can name the config rather
    than an internal target. It is NOT emitted once: ``from_settings`` is not
    memoized and re-normalizes per model call (review round 30), so a standing
    typo repeats in the log exactly as the sibling unsupported-key warning
    beside it always has.

    ``None`` means "not something this can turn into an entry"; the caller
    warns and drops it rather than letting it reach the wire.
    """
    if isinstance(entry, str):
        return entry.strip() or None
    if isinstance(entry, Mapping):
        provider = str(entry.get("provider", "") or "").strip()
        model = str(entry.get("model", entry.get("model_id", "")) or "").strip()
        if provider and model:
            # ``effort`` is honoured, so it is NOT an unsupported key - listing
            # it here is what let the old warning claim, in one sentence, both
            # that it was being ignored and that it was supported (R29-2).
            extra = sorted(set(entry) - {"provider", "model", "model_id", "effort"})
            if extra:
                # Named rather than swallowed: a chain entry that silently
                # drops half of what the user wrote is the next bug report.
                logger.warning(
                    "retry.fallbackChains[%s]: ignoring unsupported key(s) %s on entry %s/%s"
                    " - only provider, model and effort are honoured",
                    chain_key,
                    ", ".join(extra),
                    provider,
                    model,
                )
            raw_effort = entry.get("effort")
            if raw_effort is None:
                return f"{provider}/{model}"
            if str(raw_effort).strip().lower() not in SUPPORTED_EFFORTS:
                logger.warning(
                    "retry.fallbackChains[%s]: dropping entry %s/%s - %r is not accepted in"
                    " a fallback chain hop; expected one of %s",
                    chain_key,
                    provider,
                    model,
                    raw_effort,
                    ", ".join(CHAIN_EFFORT_LADDER),
                )
            return {"provider": provider, "model": model, "effort": raw_effort}
    return None


def _normalize_chains(chains: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Coerce the configured chains to ``{key: [entry, ...]}``, selector strings
    where that is what was written and structured entries where an ``effort``
    made the mapping load-bearing.

    Follows the convention the session factory uses for its compaction block:
    a malformed value degrades with a warning and never blocks. A bad chain is
    a preference about what to do when a model fails — losing it must not be
    more disruptive than the failure it was written to handle.
    """
    normalized: dict[str, list[str]] = {}
    for key, raw in chains.items():
        if not isinstance(key, str):
            logger.warning("retry.fallbackChains: ignoring non-string key %r", key)
            continue
        if isinstance(raw, str) or not isinstance(raw, Sequence):
            logger.warning("retry.fallbackChains[%s]: expected a list, got %s", key, type(raw))
            continue
        entries = []
        for entry in raw:
            kept = _normalize_chain_entry(entry, key)
            if kept is None:
                logger.warning("retry.fallbackChains[%s]: ignoring unreadable entry %r", key, entry)
                continue
            entries.append(kept)
        if entries:
            normalized[key] = entries
    return normalized


@dataclasses.dataclass(frozen=True)
class RetrySettings:
    """The ``values.retry.*`` config surface."""

    enabled: bool = True
    max_retries: int = 10
    base_delay_ms: int = 500
    model_fallback: bool = True
    usage_aware_fallback: bool = False
    usage_reserve_percent: float = 10.0
    fallback_chains: Mapping[str, Sequence[Any]] = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_settings(settings: Mapping[str, Any] | None) -> "RetrySettings":
        retry = (settings or {}).get("retry", {}) if isinstance(settings, Mapping) else {}
        if not isinstance(retry, Mapping):
            retry = {}
        chains = retry.get("fallbackChains", retry.get("fallback_chains", {}))
        if not isinstance(chains, Mapping):
            chains = {}
        reserve = retry.get("usageReservePercent", retry.get("usage_reserve_percent", 10.0))
        try:
            reserve_percent = min(100.0, max(0.0, float(reserve)))
        except (TypeError, ValueError):
            reserve_percent = 10.0
        return RetrySettings(
            enabled=bool(retry.get("enabled", True)),
            max_retries=int(retry.get("maxRetries", retry.get("max_retries", 10))),
            base_delay_ms=int(retry.get("baseDelayMs", retry.get("base_delay_ms", 500))),
            model_fallback=bool(retry.get("modelFallback", retry.get("model_fallback", True))),
            usage_aware_fallback=bool(
                retry.get("usageAwareFallback", retry.get("usage_aware_fallback", False))
            ),
            usage_reserve_percent=reserve_percent,
            # Normalized HERE, at the one place config crosses into the module,
            # so every consumer downstream can rely on the declared type.
            fallback_chains=_normalize_chains(chains),
        )


RouteChangeHandler = Callable[[FallbackTarget, str], Awaitable[None] | None]


@dataclasses.dataclass
class FailoverRouteState:
    """Session-sticky fallback route with a primary-probe cooldown.

    A successful fallback stays active for later model calls in the same user
    message. At later message boundaries, quota-aware preflight may return to
    the primary only after ``primary_retry_at_ms``. Without that suppression a
    healthy quota endpoint would make a transport-broken primary consume the
    full prompt once per user message before failing over again.
    """

    active: FallbackTarget | None = None
    on_change: RouteChangeHandler | None = None
    primary_retry_at_ms: int = 0

    async def activate(
        self,
        target: FallbackTarget,
        reason: str,
        *,
        cooldown_ms: int = 0,
    ) -> None:
        if self.active == target:
            # Already on this route: nothing changed, so nothing is re-armed.
            # The cooldown MUST NOT be bumped here. `stream_with_failover`
            # calls `activate` on every request that enters the sticky
            # fallback route, so bumping before this return turned a fixed
            # post-failure cooldown into a sliding window: a user sending
            # messages more often than the cooldown never reached
            # `primary_retry_due()`, and stayed pinned to the fallback for the
            # whole session even after the primary recovered. The docstring
            # above promises a retry "only after primary_retry_at_ms", which
            # is a deadline set when the route CHANGES, not on every use.
            return
        if cooldown_ms > 0:
            self.primary_retry_at_ms = max(
                self.primary_retry_at_ms,
                int(time.time() * 1000) + cooldown_ms,
            )
        self.active = target
        if self.on_change is None:
            return
        result = self.on_change(target, reason)
        if inspect.isawaitable(result):
            await result

    def primary_retry_due(self, now_ms: int | None = None) -> bool:
        now = int(time.time() * 1000) if now_ms is None else now_ms
        return now >= self.primary_retry_at_ms

    def clear(self) -> None:
        self.active = None
        self.primary_retry_at_ms = 0


def parse_selector(selector: str) -> tuple[str, str]:
    provider, _, model_id = selector.partition("/")
    return provider, model_id


def spec_for_target(base: ModelSpec, target: FallbackTarget) -> ModelSpec:
    """Build the fallback model's OWN spec, then carry only sampling choices.

    Cloning the primary spec kept its base URL, context window and capabilities;
    a cross-provider fallback could therefore send an OpenAI model to the
    Anthropic endpoint. Model metadata and transport identity belong to the
    target, while temperature/top-p remain session preferences.

    Reasoning effort is the exception in BOTH directions, and has to be. Its
    valid values are a property of the MODEL, so carrying the primary's level
    across a swap sends ``xhigh`` to a model whose ladder stops at ``high`` -
    a 400 on the request that was supposed to rescue the turn. But dropping it
    is not right either: a cascade entry that names no effort should not cost
    the user the level they asked for wherever the fallback accepts it. So an
    explicit ``target.effort`` wins, and otherwise :func:`resolve_effort` keeps
    the chosen level when the fallback accepts it and falls back to that
    model's own default when it does not.

    ``supports_sampling_params`` needs no such care here, unlike in the
    clone-based version this replaced: it comes from the target's own
    ``build_model_spec`` rather than from the primary.
    """
    from local_operator.model.configure import build_model_spec

    provider, model_id = parse_selector(target.selector)
    target_spec = build_model_spec(provider, model_id)
    return target_spec.model_copy(
        update={
            "temperature": base.temperature,
            "top_p": base.top_p,
            "reasoning_effort": (
                target.effort
                if target.effort is not None
                # Clamped to the nearest rung rather than dropped or defaulted: see
                # `resolve_effort` for why "the model's default" silently inverts
                # a `none`, and why clamping preserves the direction of the ask.
                else resolve_effort(model_id, base.reasoning_effort or target_spec.reasoning_effort)
            ),
        }
    )


def spec_for_selector(base: ModelSpec, selector: str) -> ModelSpec:
    """Backward-compatible selector-only wrapper."""
    return spec_for_target(base, FallbackTarget(selector))


# ---------------------------------------------------------------------------
# High-level streaming with failover
# ---------------------------------------------------------------------------

# Factories may be sync (the usual case) or async, so the driver awaits when
# it has to. ``WireClient`` is quoted: clients.py imports this module.
ClientFactory = Callable[[ModelSpec], "WireClient | Awaitable[WireClient]"]


@runtime_checkable
class FailoverAuthStore(Protocol):
    """The credential-store slice the failover driver needs.

    Structural rather than the concrete ``AuthStore``: that module imports
    this one, and hosts (plus test doubles) supply only these members.
    """

    async def get_api_key(
        self,
        provider: str,
        session_id: str | None = None,
        *,
        force_refresh: bool = False,
        read_only: bool = False,
    ) -> str | None: ...  # pragma: no cover

    def rotate_sibling(
        self,
        provider: str,
        session_id: str | None,
        error: BaseException,
        api_key: str | None = None,
    ) -> bool: ...  # pragma: no cover


@runtime_checkable
class CredentialLister(Protocol):
    """A store that can enumerate what it holds for a provider.

    Optional, like :class:`OAuthAccessSource`: it exists only so a failure can
    say WHY no bearer was resolved (nothing configured, versus everything
    temporarily blocked). A store without it gets the unqualified wording,
    which is the honest answer when the distinction cannot be checked.
    """

    def list_credentials(self, provider: str) -> "list[StoredCredential]": ...  # pragma: no cover


@runtime_checkable
class OAuthAccessSource(Protocol):
    """A store that can also hand back the identity-carrying OAuth record.

    Deliberately NOT a subclass of :class:`FailoverAuthStore`: the record is
    an optional capability, so the driver tests for this ONE member with an
    ``isinstance`` against this runtime-checkable protocol. A store exposing
    only ``get_api_key`` yields bare bearers instead.
    """

    async def get_oauth_access(
        self,
        provider: str,
        session_id: str | None = None,
        *,
        force_refresh: bool = False,
        read_only: bool = False,
    ) -> "OAuthAccess | None": ...  # pragma: no cover


def _selector_for_request(request: ChatRequest) -> str:
    return f"{request.model.provider}/{request.model.model_id}"


async def _abortable_sleep(delay_ms: int, signal: AbortSignal | None) -> None:
    """Backoff sleep that returns early when the abort signal fires (PR-19)."""
    if delay_ms <= 0:
        return
    if signal is None:
        await asyncio.sleep(delay_ms / 1000)
        return
    loop = asyncio.get_running_loop()
    sleeper = loop.create_task(asyncio.sleep(delay_ms / 1000))
    abort_waiter = loop.create_task(signal.wait())
    try:
        done, _pending = await asyncio.wait(
            {sleeper, abort_waiter}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        sleeper.cancel()
        abort_waiter.cancel()
    if abort_waiter in done:
        raise ProviderError(None, signal.reason or "aborted", retryable=False, kind="aborted")


def wrap_transport_error(exc: BaseException) -> ProviderError:
    """A non-``ProviderError`` failure as one, without losing what it was.

    The exception CLASS is kept in the message because for this family it is
    most of the diagnosis and often all of it: ``httpx.ConnectTimeout()`` and
    ``httpx.RemoteProtocolError()`` are routinely raised with no arguments at
    all, and ``ProviderError(None, str(exc))`` turned those into an error that
    printed nothing.

    The kind is stated OUTRIGHT rather than left to the text classifier, because
    the text here is the harness's own and running it through the quota markers
    made the harness misdiagnose its own bugs: a ``KeyError('usage')`` from a
    client parsing a usage block rendered as ``rate limit or quota exceeded``,
    and ``ValueError('insufficient data in chunk')`` did the same. The class name
    IS legitimate evidence of a timeout; nothing in an exception's text is
    evidence about the user's quota.
    """
    detail = str(exc).strip()
    name = type(exc).__name__
    haystack = f"{name} {detail}".lower()
    timed_out = isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or any(
        marker in haystack for marker in _TIMEOUT_MARKERS
    )
    return ProviderError(
        None,
        f"{name}: {detail}" if detail else name,
        retryable=True,
        kind="timeout" if timed_out else "transient",
    )


async def stream_with_failover(
    request: ChatRequest,
    auth: FailoverAuthStore,
    settings: Mapping[str, Any] | None,
    client_for: ClientFactory,
    *,
    session_id: str | None = None,
    signal: AbortSignal | None = None,
    route_state: FailoverRouteState | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream one provider call with tier-1 + tier-2 failover.

    ONE rotation path (PR-04/05): failures are classified here, but every
    key decision — refresh-same-account (b), sibling rotation (c), the
    ordinary-401 single-switch cap — is delegated to
    :func:`resolve_next_key`. The driver never rotates directly.

    Transport-retryable errors (429/5xx/timeout/network) consume an
    INDEPENDENT retry budget (``retry.max_retries``) with backoff on the
    SAME credential before any rotation (PR-06); the budget resets when the
    credential changes.

    Events are forwarded as they arrive, so once anything has been yielded a
    mid-stream failure is re-raised: partial output cannot be un-shown. A
    ``request.replayable`` call opts out of that by BUFFERING instead of
    forwarding, which is what extends retry coverage to the one-shot errands
    (compaction summary, auto-naming) whose streams used to die permanently on
    a single stalled read — see the field's own docstring.

    Raises :class:`ProviderError` with the most diagnostic failure seen — see
    :func:`error_report_score` — when every option is spent.
    """
    retry = RetrySettings.from_settings(settings)
    primary_selector = _selector_for_request(request)
    primary_target = FallbackTarget(primary_selector, request.model.reasoning_effort)

    if request.isolated:
        # DECORATION: one attempt on the model it named, and no reach into
        # anything the concurrent turn depends on. Expressed by disabling retry
        # rather than by a second code path, because "retry disabled" already
        # means exactly the three things needed here — no fallback chain below,
        # no transport-retry budget, and no credential rotation (every rotation
        # `continue` sits behind a `retry.enabled` raise). Dropping the route
        # state removes the fourth: a decorative call neither pins the session
        # to a fallback nor clears a pin the turn is relying on. The fifth is
        # not expressible here — the credential cascade takes routing decisions
        # of its own on a read — so the resolve below is asked for read-only
        # (`read_only=request.isolated`).
        retry = dataclasses.replace(retry, enabled=False)
        route_state = None

    targets = [primary_target]
    if retry.enabled and retry.model_fallback:
        chain = resolve_chain(primary_selector, retry.fallback_chains)
        if chain:
            for candidate in expand_fallback_targets(primary_selector, chain):
                if candidate not in targets:
                    targets.append(candidate)
    if route_state is not None and route_state.active in targets:
        targets = targets[targets.index(route_state.active) :]

    reported: ProviderError | None = None
    reported_score = -1
    clients: dict[tuple[str, str | None], "WireClient"] = {}
    rng = random.Random()

    def record(error: ProviderError, *, primary: bool) -> None:
        """Offer ``error`` the reported-error slot, keeping the best so far.

        ``>=`` so that at equal rank the LATEST attempt wins, which is the
        long-standing behaviour within one selector.

        EVERY error that does not win the slot is logged, not just a fallback's.
        Only one failure can reach the frame, and picking the most diagnostic one
        means the others now go somewhere the user cannot see — so they have to
        go somewhere the reader of a log file can. A 429 on the first key
        followed by "api key is invalid or revoked" on the sibling reports the
        429 correctly and would otherwise mention the revoked credential
        nowhere at all.
        """
        nonlocal reported, reported_score
        score = error_report_score(error, primary=primary)
        if score >= reported_score:
            if reported is not None:
                logger.warning("superseded by a more diagnostic failure: %s", reported)
            reported, reported_score = error, score
        else:
            logger.warning(
                "%s failed: %s", "requested model" if primary else "fallback selector", error
            )

    for target in targets:
        selector = target.selector
        if signal is not None and signal.aborted:
            raise ProviderError(None, signal.reason or "aborted", retryable=False, kind="aborted")

        is_primary = selector == primary_selector
        provider, _model_id = parse_selector(selector)
        spec = request.model if target == primary_target else spec_for_target(request.model, target)
        route_key = (selector, target.effort)
        client = clients.get(route_key)
        if client is None:
            built = client_for(spec)
            client = await built if inspect.isawaitable(built) else built
            clients[route_key] = client
        current_request = (
            request if target == primary_target else request.model_copy(update={"model": spec})
        )
        if route_state is not None and target != primary_target:
            cooldown_ms = max(60_000, reported.retry_after_ms or 0) if reported else 60_000
            await route_state.activate(
                target,
                "provider failure",
                cooldown_ms=cooldown_ms,
            )
        state = AuthRetryKeyState()
        error: BaseException | None = None
        access: "OAuthAccess | None" = None  # credential record for this attempt
        current_token: str | None = None
        transport_retries = 0
        retry_same_key = False
        # Requests aimed at THIS target's provider that came back a server-side
        # fault, counted across every credential it rotates through.
        #
        # Per target, not per turn. A turn-wide counter let the primary's storm
        # spend the whole allowance and leave each fallback a single attempt
        # with no retries -- so a fallback that would have succeeded on its
        # second try never got one, which defeats the entire point of having a
        # chain. Each provider is a different service having a different day,
        # and the ceiling is about what ONE provider receives.
        server_fault_requests = 0
        # Attempts this target's FIRST bearer spent before rotation started, so
        # the restore below can hand back the remainder of the user's budget
        # instead of a fresh one.
        spent_before_rotation = 0
        # The last credential that actually produced a bearer, and whether its
        # configured budget has already been handed back once rotation ran out.
        last_access: "OAuthAccess | None" = None
        exhausted_budget_restored = False

        while state.attempts <= AUTH_RETRY_MAX_ATTEMPTS:
            if signal is not None and signal.aborted:
                raise ProviderError(
                    None, signal.reason or "aborted", retryable=False, kind="aborted"
                )
            if not retry_same_key:
                access = await _resolve_access_for_provider(
                    auth,
                    provider,
                    session_id,
                    state,
                    error,
                    read_only=request.isolated,
                )
                token = access.access_token if access is not None else None
                if token != current_token:
                    if not exhausted_budget_restored:
                        # Attempts already charged to this bearer, INCLUDING the
                        # one the rotation cycle itself spends re-presenting it
                        # after a refresh. Missing that one is a whole extra
                        # request against a provider that is already failing.
                        spent_before_rotation = max(spent_before_rotation, transport_retries + 1)
                    current_token = token
                    transport_retries = 0  # fresh credential ⇒ fresh budget
                if access is None and error is not None:
                    # Rotation is exhausted: no other credential exists. The
                    # small pre-rotation allowance exists ONLY to get a turn
                    # moving to another account, so with nowhere to go the
                    # bearer already in hand finishes the budget the user
                    # configured -- once. Without this a lone credential, and
                    # the last account standing during an outage, lost every
                    # retry past the allowance (the regression rounds 2 and 3
                    # both filed).
                    if (
                        not exhausted_budget_restored
                        and last_access is not None
                        and spent_before_rotation <= retry.max_retries
                        and is_server_side_failure(error)
                        and server_fault_requests < MAX_SERVER_FAULT_REQUESTS_PER_TURN
                    ):
                        exhausted_budget_restored = True
                        access = last_access
                        # Carry the attempts already spent, and re-pin
                        # `current_token`, so the restored pass finishes the
                        # user's budget rather than starting a second one.
                        #
                        # Both halves are load-bearing. Rotation returning
                        # nothing makes `token` None, which trips the
                        # `token != current_token` reset a few lines above and
                        # zeroes the counter; re-pinning here stops the NEXT
                        # pass doing the same thing again. Without this a lone
                        # credential spent 2 x (max_retries + 1) requests
                        # against one provider -- twice the configured budget,
                        # in a change whose whole purpose is to stop hammering a
                        # provider that is already failing.
                        #
                        # `spent_before_rotation <= max_retries` above is part of
                        # the same contract, and the comparison is inclusive on
                        # purpose. `transport_retries` counts retries, so a
                        # bearer that has spent exactly `max_retries` has issued
                        # `max_retries + 1` requests and has none left; but the
                        # allowance can also land ON the budget (at
                        # `maxRetries: 4` the pre-rotation allowance of 3 makes
                        # `spent_before_rotation` 4), and an exclusive test
                        # skipped the restore there and cost the user their last
                        # request. Only that one value diverged, which is why a
                        # sweep of every setting -- not a sample -- is what the
                        # test does. `maxRetries: 0` is still exactly one
                        # request: `max()` below leaves the counter at the
                        # budget, so no retry is allowed through.
                        # Never LESS than what has already been spent: the
                        # restored pass finishes the configured budget, it does
                        # not reopen it. With a small `max_retries` the
                        # pre-rotation allowance can already have met or passed
                        # the budget, in which case there is nothing to restore
                        # and the loop simply ends.
                        transport_retries = max(spent_before_rotation, transport_retries)
                        current_token = last_access.access_token
                        error = None
                    else:
                        break  # rotation exhausted for this provider
                if access is None and not _provider_allows_missing(provider):
                    # "No API key configured" is only true when the provider has
                    # NO credential at all. Said unconditionally it was actively
                    # misleading: a signed-in OAuth account that was merely
                    # rate-limited (blocked) resolves to None here too, and the
                    # reported frame sent the user off to configure a key they
                    # already had -- the reported `No API key configured for
                    # provider 'openai'` on an account signed in via OAuth.
                    record(
                        _no_credential_error(auth, provider),
                        primary=is_primary,
                    )
                    break
                error = None
            retry_same_key = False
            if access is not None:
                last_access = access
            key = access.access_token if access is not None else None

            forwarded_any = False
            # A replayable call holds its events back so a failed attempt can be
            # discarded whole. Bounded by the errand's own output (a summary, a
            # title), which the caller was going to hold in full regardless.
            buffered: list[StreamEvent] | None = [] if current_request.replayable else None
            try:
                async for event in client.stream(current_request, key, oauth_access=access):
                    if buffered is not None:
                        buffered.append(event)
                        continue
                    forwarded_any = True
                    yield event
                if route_state is not None and target == primary_target:
                    route_state.clear()
                # This credential just served a request, so whatever provider
                # fault demoted it has passed. Without this the mark outlived
                # the outage for the life of the process, contradicting the
                # "lasts seconds" reasoning it is justified by and leaving a
                # perfectly good account permanently last in the pool.
                #
                # Skipped for an isolated request, which is the same rule the
                # cascade applies (`read_only`): a decorative call running beside
                # a user's turn must not move that turn's routing, and restoring
                # priority is as much a routing decision as removing it.
                if not request.isolated:
                    _clear_demotion(auth, provider, access)
                if buffered is None:
                    return  # clean completion; a buffered stream flushes below
            except asyncio.CancelledError:
                raise
            except ProviderError as exc:
                if forwarded_any:
                    raise  # partial output already reached the caller
                record(exc, primary=is_primary)
                if is_server_side_failure(exc):
                    server_fault_requests += 1
                if not retry.enabled:
                    raise
                if _same_credential_retry_allowed(
                    exc,
                    transport_retries,
                    retry,
                    # OBSERVED, not predicted: has rotation actually produced a
                    # different bearer for this request? See
                    # `_request_has_rotated`.
                    has_rotated=_request_has_rotated(state),
                    rotation_exhausted=exhausted_budget_restored,
                    server_fault_requests=server_fault_requests,
                ):
                    # 5xx/network-style failures use the configured budget.
                    # Rate limits retry once only when the advertised delay is
                    # short; long quota resets rotate or surface immediately.
                    transport_retries += 1
                    delay = max(
                        exc.retry_after_ms or 0,
                        backoff_delay_ms(retry.base_delay_ms, transport_retries, rng=rng),
                    )
                    await _abortable_sleep(delay, signal)
                    retry_same_key = True
                    continue
                if _server_fault_budget_spent(exc, server_fault_requests):
                    # The turn has aimed its whole server-fault budget at this
                    # provider. Rotating again would keep spending it on an
                    # outage the credentials are not responsible for, so hand
                    # over to the fallback chain (a DIFFERENT provider) instead,
                    # which is the thing left that might actually succeed.
                    break
                if exc.retryable or exc.auth_error or exc.status in (401, 403):
                    # Delegate: (b) refresh same account, then (c) rotate —
                    # resolve_next_key owns the decision (PR-04/05).
                    error = exc
                    continue
                break  # non-retryable for this provider
            except Exception as exc:  # network errors et al.
                wrapped = wrap_transport_error(exc)
                if forwarded_any:
                    # Partial output already reached the caller, so this attempt
                    # cannot be retried — but the failure still has to arrive
                    # NAMED. Re-raised raw it was not a ``RenderedStreamError``,
                    # so the loop printed ``str(httpx.ConnectTimeout())`` (which
                    # is the empty string) into the frame AND a full traceback
                    # into the log.
                    raise wrapped from exc
                record(wrapped, primary=is_primary)
                if is_server_side_failure(wrapped):
                    server_fault_requests += 1
                if not retry.enabled:
                    raise wrapped from exc
                # Same budget rule as the ProviderError arm above, asked the same
                # way. This branch used to test `retry.max_retries` directly,
                # which quietly exempted it from the per-account server cap --
                # and this is the arm RAW transport failures take (no client in
                # clients.py catches httpx; `_guarded_chunks` deliberately raises
                # ReadTimeout on a stall), so a timeout storm still sent
                # max_retries x pool-size requests. `wrap_transport_error` stamps
                # these `kind="timeout"`, so the shared predicate already
                # classifies them correctly; the branch just was not asking.
                if _same_credential_retry_allowed(
                    wrapped,
                    transport_retries,
                    retry,
                    has_rotated=_request_has_rotated(state),
                    rotation_exhausted=exhausted_budget_restored,
                    server_fault_requests=server_fault_requests,
                ):
                    transport_retries += 1
                    await _abortable_sleep(
                        backoff_delay_ms(retry.base_delay_ms, transport_retries, rng=rng), signal
                    )
                    retry_same_key = True
                    continue
                if _server_fault_budget_spent(wrapped, server_fault_requests):
                    break  # same ceiling, same reasoning as the arm above
                error = wrapped
                continue
            else:
                # Clean completion. The buffer is flushed OUTSIDE the try so a
                # consumer raising back into this generator is not mistaken for
                # a provider failure and retried.
                for event in buffered or ():
                    yield event
                return

        # Provider exhausted — walk on to the next fallback selector.

    if reported is not None:
        raise reported
    raise ProviderError(None, f"Failover exhausted for '{primary_selector}'", retryable=False)


async def _resolve_access_for_provider(
    auth: FailoverAuthStore,
    provider: str,
    session_id: str | None,
    state: AuthRetryKeyState,
    error: BaseException | None,
    *,
    read_only: bool = False,
) -> "OAuthAccess | None":
    """Bridge AuthStore into the a/b/c resolver shape, returning the
    :class:`~local_operator.providers.auth_store.OAuthAccess` record (or
    ``None``) so wire clients get identity headers alongside the bearer.

    ``read_only`` is the isolated request's sixth denial: the cascade itself
    mutates session-shared routing state on a READ — it blocks an OAuth row
    whose refresh raises and it writes (or clears) the session's sticky
    credential — and neither ``retry.enabled=False`` nor a dropped
    ``route_state`` is upstream of that. A decorative call resolves the account
    the turn is already on and decides nothing.
    """
    # Presence test, not a nominal one: stores exposing only get_api_key take
    # the bare-bearer path and get wrapped at the bottom of this function.
    oauth_store = auth if isinstance(auth, OAuthAccessSource) else None
    records: dict[str, "OAuthAccess"] = {}

    def _flags(force_refresh: bool) -> dict[str, bool]:
        """Each flag is passed only when set, so stores declaring the bare
        ``(provider, session_id)`` signature keep working."""
        flags: dict[str, bool] = {}
        if force_refresh:
            flags["force_refresh"] = True
        if read_only:
            flags["read_only"] = True
        return flags

    async def _access(*, force_refresh: bool = False) -> "OAuthAccess | None":
        if oauth_store is None:
            return None
        return await oauth_store.get_oauth_access(provider, session_id, **_flags(force_refresh))

    async def _key(*, force_refresh: bool = False) -> str | None:
        return await auth.get_api_key(provider, session_id, **_flags(force_refresh))

    async def resolver(ctx: ApiKeyResolveContext) -> str | None:
        try:
            if ctx.error is None:
                record = await _access()
                if record is None:
                    return await _key()
            elif ctx.last_chance:
                auth.rotate_sibling(provider, session_id, ctx.error, api_key=ctx.previous_key)
                record = await _access()
                if record is None:
                    return await _key()
            else:
                record = await _access(force_refresh=True)
                if record is None:
                    return await _key(force_refresh=True)
        except Exception:
            return None
        if record is None:
            return None
        records[record.access_token] = record
        return record.access_token

    token = await resolve_next_key(state, resolver, error)
    if token is None:
        return None
    record = records.get(token)
    if record is None:
        # Auth stores without get_oauth_access (test fakes) yield bare keys;
        # wrap them so clients see one uniform shape.
        from local_operator.providers.auth_store import OAuthAccess

        record = OAuthAccess(access_token=token, credential_id=0, kind="api_key")
    return record


def _clear_demotion(auth: FailoverAuthStore, provider: str, access: "OAuthAccess | None") -> None:
    """Restore a credential's priority after it successfully served a request.

    Best-effort and silent: a store need not implement demotion at all (it is an
    ``AuthStore`` detail, not part of the failover protocol), and a bookkeeping
    failure must never turn a SUCCESSFUL turn into an error.
    """
    if access is None or not access.credential_id:
        return
    clear = getattr(auth, "clear_deprioritized", None)
    if not callable(clear):
        return
    try:
        clear(provider, access.credential_id)
    except Exception:  # noqa: BLE001 - never fail a served request on bookkeeping
        logger.debug("could not clear demotion for %s/%s", provider, access.credential_id)


def _server_fault_budget_spent(error: ProviderError, server_fault_requests: int) -> bool:
    """Has this turn spent its whole server-fault allowance on one provider?

    Checked before ROTATING as well as before retrying, because rotation is the
    other way the same turn sends the provider another request: bounding only
    the same-credential path let a large pool walk straight past the ceiling.
    """
    return (
        is_server_side_failure(error)
        and server_fault_requests >= MAX_SERVER_FAULT_REQUESTS_PER_TURN
    )


def _request_has_rotated(state: AuthRetryKeyState) -> bool:
    """Has this request already been handed more than one distinct bearer?

    This decides ONE thing: whether the small per-credential allowance applies,
    because the budget is about to be spent AGAIN on another account -- the
    multiplication the cap exists to prevent.

    It reports what ALREADY HAPPENED -- more than one distinct bearer has been
    handed to this request -- rather than modelling the credential table. Five
    review rounds each found a predictive version drifting from what the cascade
    actually does: raw row counts, then blocked rows, then credential types,
    then override bearers with no row at all, then ``api_key`` rows split across
    cascade tiers where one always wins. Every drift cost a real user retries
    they needed. A fact about the past cannot drift, which is why this version
    has no sixth.
    """
    return len(state.attempted_keys) > 1


def _no_credential_error(auth: FailoverAuthStore, provider: str) -> ProviderError:
    """Say WHY no bearer could be resolved, distinguishing two opposite causes.

    Resolution returns ``None`` both when a provider has never been configured
    and when every credential it has is temporarily blocked (rate limit, a
    failed refresh). Those need opposite actions from the user -- go and sign
    in, versus wait or top up -- and reporting the first for the second is what
    told a user with a working OAuth login that they had no API key.

    Stores that cannot enumerate credentials (the structural protocol only
    promises ``get_api_key``) fall back to the original wording, which is the
    correct message whenever the answer is genuinely unknown.
    """
    rows: list["StoredCredential"] = []
    if isinstance(auth, CredentialLister):
        try:
            rows = [row for row in auth.list_credentials(provider) if row.disabled_cause is None]
        except Exception:  # noqa: BLE001 - diagnosis must never mask the real failure
            rows = []
    if not rows:
        return ProviderError(
            None, f"No API key configured for provider '{provider}'", retryable=False
        )
    kinds = {row.credential_type for row in rows}
    what = "OAuth sign-in" if "oauth" in kinds else "API key"
    count = len(rows)
    subject = (
        f"The {what} credential for provider '{provider}' is"
        if count == 1
        else f"All {count} {what} credentials for provider '{provider}' are"
    )
    # Retryable: the credential comes back on its own once the block expires,
    # which is materially different from a missing configuration.
    #
    # The wording hedges on WHY deliberately. A row can also be present and
    # permanently unreadable -- a login flow storing a shape no cascade tier
    # resolves, which is what R21 was -- and for that user "temporarily
    # unavailable, retry once the limit resets" is advice that can never come
    # true, and it hides the real fix (sign in again). Naming the third cause
    # costs a clause and stops the message actively misdirecting; the rate-limit
    # case, which is the common one, still reads first.
    return ProviderError(
        None,
        (
            f"{subject} not usable right now (rate limited, a token refresh "
            "failed, or the stored credential could not be read). The "
            "credentials are still configured; retry once the limit resets, or "
            "sign in again to replace them."
        ),
        retryable=True,
        kind="quota",
    )


def _provider_allows_missing(provider: str) -> bool:
    """Providers that self-authenticate (ollama/test) need no key at all."""
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition(provider)
    return bool(definition and definition.allows_missing_api_key)
