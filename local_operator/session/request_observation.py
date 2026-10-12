"""Typed request-final observations for a session tree, opt-in and allocation-free when unused.

WHY THIS EXISTS. Callers that must ACCOUNT for every model request a session
tree makes -- the benchmark pilot record is the first, but nothing here is
benchmark-specific -- need one record per finished logical request: parent
turns, tool-loop continuations, compaction and naming helpers, and every
delegated child/grandchild call, each exactly once, with enough identity and
provenance to attribute spend without parsing prompts or trusting model names.

WHERE THE RECORDS COME FROM. The existing ``SessionStreamFn._record_stream``
wrapper is the one place every provider call already funnels through; it
already assigns the request UUID and holds the final ``Usage``. This module
owns the TYPES and the optional subscriber carrier; the stream wrapper builds
one observation in its own ``finally`` so the record exists on success,
failure and cancellation alike. No second decoder, no parallel ledger, no
per-delta work: the carrier is ``None`` unless a caller has subscribed, and
that ``None`` is resolved once per request (see
``SessionStreamFn.subscribe_requests``).

WHAT A RECORD MUST NOT DO. It carries no prompt, tool arguments, credentials
or provider payloads -- identity, provenance and accounting ONLY. It never
prices anything: money lives in ``model.costs`` over the ``Usage`` the
provider already reported, and a missing receipt stays UNKNOWN, never zero.
``usage=None``/``usage_reported=False`` means "unknown", NOT ``Usage()``; a
provider-reported ``usd_cost=0.0`` is a known zero and is kept as one. A final
duplicate usage/end snapshot REPLACES the provisional one upstream, so a
record built from the final ``Usage`` can never double-count it.

COST COVERAGE, STATED HONESTLY. One logical request can encompass several wire
attempts (retry/fallback live below this wrapper, in ``stream_with_failover``).
``usage_coverage="reported_final"`` therefore claims only what its name says --
the FINAL attempt's receipt is present and raw -- and is NOT proof that every
retry bill was reported. ``"missing"`` means no usage frame arrived at all.
Consumers that need invoice totals must mark fallback-involved requests as
incomplete coverage; a transport-attempt observation would be a separate,
narrower seam.

CALLBACK CONTRACT. Delivery is synchronous and bounded: handlers append or
enqueue and return. No HTTP, SQLite, pricing or heavy serialization here --
this runs on the model loop that is unwinding a request. A handler raising is
logged and counted on the carrier (``callback_errors``) and must never replace
or erase the provider's outcome; that count is surfaced on the owning
session's shutdown receipt so a report cannot silently claim complete
observation after a failed subscriber.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from local_operator.harness.types import ChatRequest, Usage

logger = logging.getLogger("local_operator.session.request_observation")

#: Version marker for the record SHAPE. Bump when a field's MEANING changes --
#: consumers may retain rows across versions and must be able to tell.
REQUEST_OBSERVATION_SCHEMA_VERSION = "request-final-observation-v1"

#: One observer: called once per finished request, synchronously, with an
#: immutable record. It must return promptly; it must not await.
RequestObserver = Callable[["RequestFinalObservation"], None]


@dataclass(frozen=True)
class RequestFinalObservation:
    """One finished logical request, as the session's stream wrapper saw it.

    Frozen as a VALUE, with one honest caveat stated where it matters: the
    nested ``usage`` is a provider-typed object that is not itself frozen, so
    the carrier hands each handler its own detached copy when more than one
    handler is subscribed (``RequestSubscriberHolder.notify``). Freezing only
    this outer object would not freeze it.

    ``served_identity_source`` records HOW the served fields were derived:

    * ``"usage_stamp"`` -- the ``Usage`` carried a failover-stamped
      provider/model id (at least one side); the other side falls back to the
      request exactly as ``_record_usage`` already does, so the two readers of
      one call cannot disagree.
    * ``"request_assumed"`` -- no stamp existed: either the usage frame
      arrived without serving identity (a primary success or a direct call;
      ``request.model`` is the honest reading, same rule as the module
      comment ``_record_usage`` follows), or NO usage frame arrived at all --
      and then the served fields stay ``None``: the request ran but nothing
      reported who served it, so nothing is claimed.
    """

    #: Shape version; see REQUEST_OBSERVATION_SCHEMA_VERSION.
    schema_version: str = REQUEST_OBSERVATION_SCHEMA_VERSION
    #: The existing per-request UUID from ``_record_stream`` -- the one and
    #: only deduplication key a consumer may use. Provider/model/cache keys
    #: are NOT unique per request and must never dedupe.
    request_id: str = ""
    #: The ROOT owning session id of this session tree; survives forks.
    tree_id: str | None = None
    #: This stream owner's session id (root or a delegated child).
    session_id: str | None = None
    #: The IMMEDIATE parent session id; ``None`` on the root.
    parent_session_id: str | None = None
    #: The child Session's job id / role label (existing subagent metadata),
    #: ``None`` for root-owned requests.
    job_id: str | None = None
    actor_role: str | None = None
    #: ``ChatRequest.purpose`` -- "turn", "compaction", "naming", ...
    purpose: str = "turn"
    #: What the request ASKED for (the original ChatRequest still names it
    #: after a failover rewrite, which is why both sides are recorded).
    requested_provider: str | None = None
    requested_model_id: str | None = None
    #: What actually served, per ``served_identity_source`` above.
    served_provider: str | None = None
    served_model_id: str | None = None
    served_identity_source: Literal["usage_stamp", "request_assumed"] = "request_assumed"
    #: The aggregator upstream host from ``StreamEndEvent.served_provider``,
    #: recorded SEPARATELY from the hosting provider -- they are different
    #: facts (one is a route, one is a host) and replacing one with the other
    #: is the bug class this separation prevents.
    upstream_host: str | None = None
    #: A detached copy of the FINAL reported usage, or ``None`` when no usage
    #: frame arrived (unknown spend -- never ``Usage()`` priced as zero).
    #: Raw buckets are exactly as the provider convention delivered them:
    #: cache read/write stay separate, reasoning stays a subset of output.
    usage: Usage | None = None
    usage_reported: bool = False
    #: ``"reported_final"`` covers the final attempt's receipt only (earlier
    #: retry/fallback attempts may be unreported -- see the module docstring);
    #: ``"missing"`` means nothing arrived.
    usage_coverage: Literal["reported_final", "missing"] = "missing"
    #: The wrapper's own outcome vocabulary: a stop reason ("stop",
    #: "tool_use", ...), an exception class name, or "incomplete".
    outcome: str = "unknown"
    ok: bool = False
    duration_ms: float = -1.0
    #: The provider-reported stamp of the call when the usage carried one
    #: (epoch ms UTC); ``None`` otherwise. Reused, never re-minted.
    at_ms: int | None = None
    #: The request-prefix cache key the call was sent with, for cache
    #: identity checks; a key, not a credential.
    requested_prompt_cache_key: str | None = None


@dataclass(frozen=True)
class OwnedShutdownResult:
    """The truthful outcome of ``Session.shutdown_owned_work``.

    ``complete`` is True ONLY when the cleanup was joined within the deadline,
    every owned job settled, and no descendant request is still in flight. A
    timeout is ``complete=False`` UNVERIFIED -- pending cleanup is not success,
    and no caller may score or delete on an incomplete result while owned
    work could still be alive. The cleanup task is RETAINED (never cancelled
    on the way out), so the real shutdown still lands; Python cannot forcibly
    stop a cancellation-resistant coroutine, and pretending otherwise is the
    failure this field exists to prevent.
    """

    complete: bool
    reason: str
    elapsed_ms: float
    #: Owned jobs still running at the deadline. Each running child session is
    #: its own job, so jobs are the honest pending-id list.
    pending_job_ids: tuple[str, ...] = ()
    #: Short, sanitized error strings from the cleanup path (class name plus a
    #: bounded message), empty on a clean shutdown.
    errors: tuple[str, ...] = ()
    #: Observer handlers that raised while delivering records for this tree
    #: (0 when none did). Surfaced HERE, not only in the log, so a report
    #: cannot claim complete observation after a failed subscriber.
    observer_callback_errors: int = 0
    #: Whether a descendant provider stream was still in flight at the
    #: deadline (the stream wrapper's own O(1) counter).
    descendant_requests_in_flight: bool = False


@dataclass
class _SubscriberState:
    handlers: list[RequestObserver] = field(default_factory=list)
    callback_errors: int = 0


class RequestSubscriberHolder:
    """The tree-scoped subscriber set, allocated only when someone subscribes.

    The stream wrapper holds this as ``None`` until the first subscription;
    that ``None`` (resolved once per request through the root of the fork
    tree) is the whole fast path -- no allocation, no callback, no per-delta
    check, no extra clock on the ordinary path.

    ``subscribe`` returns an idempotent unsubscribe: calling it twice, or from
    any thread of the event loop, is safe and removes the registration once.
    Unsubscribing does NOT retroactively drop requests already in flight --
    their handler set was snapshotted at request start, so a request begun
    while subscribed is delivered even if the caller unsubscribes before it
    finishes. That is what lets a caller hold the subscription until shutdown
    settles to capture cancellation-final usage.

    Delivery is synchronous and per-request bounded (one record; one call per
    handler). Handler exceptions are logged and counted, never raised out: an
    observer bug must not mask the provider's outcome or erase a completed
    request.
    """

    __slots__ = ("_state",)

    def __init__(self) -> None:
        self._state = _SubscriberState()

    def subscribe(self, handler: RequestObserver) -> Callable[[], None]:
        """Register ``handler``; returns the idempotent unsubscribe closure."""
        state = self._state
        state.handlers.append(handler)

        def unsubscribe() -> None:
            try:
                state.handlers.remove(handler)
            except ValueError:  # already removed -- idempotent by contract
                pass

        return unsubscribe

    @property
    def callback_errors(self) -> int:
        """How many handler invocations have raised so far on this tree."""
        return self._state.callback_errors

    @property
    def subscriber_count(self) -> int:
        return len(self._state.handlers)

    def snapshot(self) -> tuple[RequestObserver, ...]:
        """The handler set for ONE request, captured at its start.

        The tuple is immutable, so a handler subscribing or unsubscribing
        while this request is in flight cannot change what this request
        delivers to -- the request-start snapshot is the recorded contract.
        """
        return tuple(self._state.handlers)

    def notify(
        self, handlers: tuple[RequestObserver, ...], observation: RequestFinalObservation
    ) -> None:
        """Deliver one finished request to a start-time snapshot.

        Bounded, synchronous, non-raising. When more than one handler is
        subscribed each receives its OWN record with its own detached Usage
        copy -- the nested ``Usage`` is mutable in the provider type, and a
        handler rewriting a counter must not rewrite what the next handler
        sees.
        """
        if not handlers:
            return
        state = self._state
        for handler in handlers:
            view = observation if len(handlers) == 1 else _with_detached_usage(observation)
            try:
                handler(view)
            except Exception:  # noqa: BLE001 -- a subscriber must never break a turn
                state.callback_errors += 1
                logger.warning("request observation handler failed", exc_info=True)


def _with_detached_usage(observation: RequestFinalObservation) -> RequestFinalObservation:
    """A copy of ``observation`` whose Usage is nobody else's object."""
    if observation.usage is None:
        return observation
    return RequestFinalObservation(
        schema_version=observation.schema_version,
        request_id=observation.request_id,
        tree_id=observation.tree_id,
        session_id=observation.session_id,
        parent_session_id=observation.parent_session_id,
        job_id=observation.job_id,
        actor_role=observation.actor_role,
        purpose=observation.purpose,
        requested_provider=observation.requested_provider,
        requested_model_id=observation.requested_model_id,
        served_provider=observation.served_provider,
        served_model_id=observation.served_model_id,
        served_identity_source=observation.served_identity_source,
        upstream_host=observation.upstream_host,
        usage=observation.usage.model_copy(deep=True),
        usage_reported=observation.usage_reported,
        usage_coverage=observation.usage_coverage,
        outcome=observation.outcome,
        ok=observation.ok,
        duration_ms=observation.duration_ms,
        at_ms=observation.at_ms,
        requested_prompt_cache_key=observation.requested_prompt_cache_key,
    )


def build_request_final_observation(
    *,
    request_id: str,
    tree_id: str | None,
    session_id: str | None,
    parent_session_id: str | None,
    job_id: str | None,
    actor_role: str | None,
    request: ChatRequest,
    usage: Usage | None,
    upstream_host: str | None,
    ok: bool,
    outcome: str,
    duration_ms: float,
) -> RequestFinalObservation:
    """Build the one immutable record for a finished request.

    Runs in the stream wrapper's ``finally`` on the model loop, so it is
    deliberately arithmetic-only: an attribute read, a detach copy of the
    usage WHEN one exists, and a dataclass construction -- no clock beyond the
    duration the caller already measured, no pricing, no I/O.

    Served identity, in the three cases the contract names (see
    ``RequestFinalObservation.served_identity_source``):

    1. usage with a stamp -> ``usage_stamp``; the unstamped side falls back to
       the request exactly as ``_record_usage`` already does.
    2. usage without a stamp -> ``request_assumed``; the request is the honest
       reading (primary success / direct call).
    3. NO usage -> ``request_assumed`` with ``served_provider``/
       ``served_model_id`` left ``None``: missing usage must not claim a
       served identity.
    """
    usage_provider = getattr(usage, "provider", None) if usage is not None else None
    usage_model_id = getattr(usage, "model_id", None) if usage is not None else None
    if usage is not None and (usage_provider or usage_model_id):
        served_provider: str | None = usage_provider or request.model.provider
        served_model_id: str | None = usage_model_id or request.model.model_id
        served_source: Literal["usage_stamp", "request_assumed"] = "usage_stamp"
    elif usage is not None:
        served_provider = request.model.provider
        served_model_id = request.model.model_id
        served_source = "request_assumed"
    else:
        served_provider = None
        served_model_id = None
        served_source = "request_assumed"

    detached_usage = usage.model_copy(deep=True) if usage is not None else None
    return RequestFinalObservation(
        request_id=request_id,
        tree_id=tree_id,
        session_id=session_id,
        parent_session_id=parent_session_id,
        job_id=job_id,
        actor_role=actor_role,
        purpose=getattr(request, "purpose", "turn") or "turn",
        requested_provider=request.model.provider,
        requested_model_id=request.model.model_id,
        served_provider=served_provider,
        served_model_id=served_model_id,
        served_identity_source=served_source,
        upstream_host=upstream_host,
        usage=detached_usage,
        usage_reported=usage is not None,
        usage_coverage="reported_final" if usage is not None else "missing",
        outcome=outcome,
        ok=ok,
        duration_ms=duration_ms,
        at_ms=getattr(usage, "at_ms", None) if usage is not None else None,
        requested_prompt_cache_key=getattr(request, "prompt_cache_key", None),
    )
