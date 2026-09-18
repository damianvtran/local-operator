"""``ClassificationService``: one classification per user message, never a failure.

THE CONTRACT THIS CLASS IMPLEMENTS (§4)
=======================================

* :meth:`ClassificationService.recommend_resources` returns an empty
  :class:`~local_operator.classification.recommend.Recommendation` with
  ``skipped`` set for EVERY failure mode. It never raises. A layer whose whole
  purpose is to add an advisory line to a prompt may not be able to break a
  turn, and "it raised and the turn died" is not a failure mode a caller can
  defend against from outside.
* No vendor call happens when the setting is off, the roster is empty, no leg is
  usable, or the circuit breaker is open.
* Results are cached by ``sha256(user_message + candidates_digest)`` in a bounded
  per-session LRU (64 entries). A hit costs nothing, including no tokens, and
  reports ``latency_s`` ≈ 0.
* Three consecutive failures open the breaker for the rest of the session.
* A 422 — or, as measured, a 400 whose body names one of our question fields —
  is NOT a transport failure: it disables that question shape for the session
  and is logged at ERROR, because it is a bug in this layer rather than weather.
* Concurrent callers share one in-flight call per cache key.

THE LATENCY BUDGET (the constraint that shapes this class)
=========================================================

This layer runs ONCE PER USER MESSAGE, on the critical path, before the turn's
first token — and **the caller owns how long the turn may wait for it**: §7's
wiring gives it ``values.classification.waitMs`` (default 50 ms) and delivers a
late answer on a later turn rather than blocking. So this class makes no claim
about the turn's added wall clock. What it does promise is its own overhead and
its own deadline:

* **OUR OWN OVERHEAD, measured: 0.011-0.2 ms per message** — the work that
  happens whether or not a vendor answers: settings reads, the cache key, the
  roster memo, the state build, the question build, the credential memo hit and
  the serialization. ``tests/unit/classification/test_latency_budget.py`` pins it
  with a deliberately loose 10 ms assertion so a loaded shared host cannot flake
  it; the measured figure is in the report on the MR.
* ``values.classification.timeoutMs`` (default 1500) is the **ceiling on the call
  itself** — the deadline this service enforces around the vendor request — not a
  promise that a turn waits that long. A caller that stops waiting earlier simply
  gets the empty outcome on its side, and a call that does land is cached for the
  next message either way.

That is why §7's wiring runs this inside an ``asyncio.gather`` beside the
existing skill/guide selection: the added wall time is then the DIFFERENCE
between the two, not their sum. Three things in this class exist only to keep
the per-message cost down, and they are the three to preserve if this is ever
refactored:

* the credential memo (per leg, on this instance, TTL 5 minutes) — because
  ``AuthStore.get_api_key`` walks a 7-step SQLite cascade and can refresh an
  OAuth grant (``vendors.CREDENTIAL_TTL_S``);
* ONE keep-alive ``httpx.AsyncClient`` for the session
  (:meth:`ClassificationService._client`), because a per-message TLS handshake
  to ``api.radienthq.com`` is 50-150 ms and would blow the budget on its own;
* the roster memo (:class:`~local_operator.classification.context.RosterCache`),
  because the candidate lines change only when the roster does, not per message.

THE GENERAL PRINCIPLE, since the same shape will be wanted elsewhere: on a
per-message path, anything slow to look up — credentials, pricing rows,
catalogue entries — is resolved once and cached with a SHORT TTL, never looked
up per message; and never cached for so long that a revoked credential outlives
its usefulness. The TTL is what makes the cache safe; the cache is what makes
the path affordable.

There is exactly ONE network call in here. No credential preflight, no model
listing, no usage probe, no warm-up request: the decision call is the only thing
that leaves the process, and a leg that needs a token exchange does it once at
resolve time.

WHAT "SESSION" MEANS
====================

The instance. One service is built per session (§8: the settings are
``Section``-scoped because the service is per session), so every piece of
mutable state here — the cache, the breaker, the disabled shapes, the resolved
leg — is session-scoped by construction rather than by a key that could collide
between two sessions in one process. That is also why the cache key does not
include a session id.

THE BREAKER HAS NO HALF-OPEN STATE
==================================

§4 says it opens after three consecutive failures and "leaves it open for the
rest of the session". There is deliberately no clock-driven half-open probe: the
failure this guards against is a vendor being down or a key being exhausted, a
session's worth of retries is a rounding error of quota, and a half-open state
would mean a doomed network call per user message for an unknown length of time.
The next session starts closed, which is the natural retry.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import httpx

from local_operator.classification.cascade import (
    DEFAULT_VENDOR,
    classification_section,
    leg_order,
    model_override,
    pinned_vendor,
    resolve_vendor,
    vendor_status,
)
from local_operator.classification.context import (
    RosterCache,
    build_state,
    candidates_digest,
    max_candidates,
    select_candidates,
    setting_int,
)
from local_operator.classification.recommend import (
    DEFAULT_MAX_RECOMMENDATIONS,
    QuestionPlan,
    Recommendation,
    RecommendationRequest,
    build_decision_request,
    build_questions,
    collect_resources,
    max_recommendations,
    render_block,
)
from local_operator.classification.types import (
    DecisionRequest,
    DecisionResponse,
    DecisionSchemaError,
    DecisionVendor,
    DecisionVendorError,
)
from local_operator.classification.vendors import build_vendor
from local_operator.settings_io import strict_bool

if TYPE_CHECKING:
    from local_operator.credentials import CredentialManager

logger = logging.getLogger(__name__)

#: ``values.classification.auto`` — off means the prompt is byte-identical to a
#: harness without this feature. Default false, matching ``values.effort.auto``:
#: an upgrade must never silently change behaviour or spend (§8).
DEFAULT_AUTO = False

#: ``values.classification.timeoutMs`` — per-call deadline, in milliseconds.
DEFAULT_TIMEOUT_MS = 1500

#: ``values.classification.notice`` — whether the host emits its one-line notice.
DEFAULT_NOTICE = True

#: Consecutive failures after which the breaker opens for the session (§4).
CIRCUIT_FAILURE_THRESHOLD = 3

#: Cache entries before the oldest is evicted (§4: 64).
CACHE_SIZE = 64


def _cache_key(request: RecommendationRequest, limit: int | None) -> str:
    """``sha256(user_message + candidates_digest)`` over the roster AS SENT.

    The digest covers the roster's names, URLs and descriptions, so a session
    whose discovered descriptions changed does not serve a recommendation built
    from the old rubric. Deliberately NOT part of the key: ``max_recommendations``
    and the timeout. Both are session settings rather than per-request ones, and
    :meth:`ClassificationService.recommend_resources` re-applies the cap to a
    cached result anyway, which keeps the key exactly what the contract says it
    is.

    WHY THE CAPPED ROSTER AND NOT THE DISCOVERED ONE
    ------------------------------------------------
    ``limit`` is ``values.classification.maxCandidates``, and the roster is
    capped by it BEFORE it is sent — :func:`select_candidates`, called by both
    :func:`~local_operator.classification.context.build_state` and
    :func:`~local_operator.classification.recommend.build_questions`. So an
    entry past the cap never reaches the vendor and cannot change the question
    that was asked. Digesting the DISCOVERED list instead made the key sensitive
    to roster changes the request cannot carry: one more skill on disk moved the
    key while the body stayed byte-identical, and the session paid for a second
    call to re-ask a question it had already answered. Keying on the capped roster
    is what makes the ROSTER half of the key a digest of the ROSTER half of the
    body.

    WHAT IS STILL OUTSIDE THE KEY, since this docstring is a future caller's only
    warning. The key is ``user_message`` plus this digest, so two other body
    inputs are not covered: ``context`` (a state field when a caller supplies
    one — none does today) and the ladder rung that ``maxStateChars`` selects, a
    session setting whose change moves the body without moving the key. Both are
    benign while the first caller-side half is unbuilt, and closing them belongs
    with whoever builds it rather than with the roster fix.

    It stays a plain function taking the limit rather than reading the settings
    itself, because the caller is the one that knows which limit its request
    builder will apply — and the two must agree, or the key covers a different
    roster than the body. ``limit`` has no default for that reason: a caller that
    forgot it would key on the DEFAULT cap while its body carried the configured
    one, and a configured cap ABOVE the default would then answer a changed
    roster out of cache — a false hit, which is the expensive direction of this
    mistake.
    """
    roster = select_candidates(request.candidates, limit)
    payload = f"{request.user_message}\n{candidates_digest(roster)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _NoUsableLeg(DecisionVendorError):
    """No leg resolved a credential — a configuration state, not an outage.

    Kept apart from an ordinary leg failure because the service reports it as
    ``skipped="no-vendor"`` rather than as an error, and because it must NOT
    count toward the circuit breaker: a session with no credentials is not a
    vendor being down, and three messages into it the operator should still be
    told "no vendor", not "circuit open".
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, kind="auth", status=401)


class ClassificationService:
    """Recommends skills, guides and MCP servers for one session's prompts."""

    def __init__(
        self,
        *,
        manager: "CredentialManager",
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        self._manager = manager
        self._settings = settings
        self._cache: OrderedDict[str, Recommendation] = OrderedDict()
        self._inflight: dict[str, asyncio.Task[Recommendation]] = {}
        self._consecutive_failures = 0
        self._circuit_open = False
        #: Question ids the vendor rejected on this session, disabled until the
        #: session ends (a 422/field-level 400 is our bug, and re-asking it every
        #: message would spend a request per message to re-learn one fact).
        self._disabled_shapes: set[str] = set()
        self._vendor_name: str | None = None
        self._legs: tuple[DecisionVendor, ...] | None = None
        self._vendors: dict[str, DecisionVendor] = {}
        #: One keep-alive client for the session; created on first use so a
        #: session that never classifies never opens a socket, and closed by
        #: :meth:`aclose`.
        self._http: httpx.AsyncClient | None = None
        #: The candidate lines, built once per roster (see the class docstring).
        self._roster_cache = RosterCache()

    # -- configuration -----------------------------------------------------

    @property
    def enabled(self) -> bool:
        """``values.classification.auto``, read as a REAL boolean.

        Through ``settings_io.strict_bool`` rather than ``bool(...)``: a
        hand-edited ``auto: "false"`` is a non-empty string, and reading it as
        truthy is exactly the toggle-that-does-nothing defect ``strict_bool``
        exists to close.
        """
        return strict_bool(classification_section(self._settings).get("auto"), DEFAULT_AUTO)

    @property
    def timeout_s(self) -> float:
        """The per-call deadline in seconds, from ``timeoutMs``."""
        return setting_int(self._settings, "timeoutMs", DEFAULT_TIMEOUT_MS) / 1000.0

    @property
    def notice_enabled(self) -> bool:
        """``values.classification.notice``."""
        return strict_bool(classification_section(self._settings).get("notice"), DEFAULT_NOTICE)

    @property
    def vendor_name(self) -> str | None:
        """The leg this session will use — lazily resolved, cached for the session.

        Before the first call this is an ESTIMATE: it reports the pin, or the
        first leg whose credential is visible without I/O
        (:func:`~local_operator.classification.cascade.vendor_status`), which
        cannot see a Radient OAuth session. After the first call the property
        reports the leg that actually answered or was resolved, so a notice or a
        diagnostics line is precise from the second message onwards. Making this
        a plain property rather than a coroutine is §4's choice, and the price of
        it is exactly this documented estimation window.
        """
        if self._vendor_name is not None:
            return self._vendor_name
        pin = pinned_vendor(self._settings)
        if pin != DEFAULT_VENDOR:
            return pin
        for name, available in vendor_status(self._manager, self._settings):
            if available:
                return name
        return None

    # -- public API --------------------------------------------------------

    async def recommend_resources(self, request: RecommendationRequest) -> Recommendation:
        """Recommend resources for one user message. Never raises; see the module docstring.

        The only exception that may leave this coroutine is cancellation, which
        is not a failure to swallow: a cancelled turn must stay cancelled rather
        than be turned into an empty recommendation the caller then injects.
        """
        try:
            return await self._recommend(request)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the classifier may never fail a turn
            logger.warning("classification: recommendation failed", exc_info=True)
            self._record_failure("error")
            return Recommendation(skipped="error")

    def notice(self, recommendation: Recommendation) -> str | None:
        """The one-line host notice, or ``None`` when there is nothing to announce.

        Silent on every empty outcome — disabled, no vendor, timeout, error,
        circuit open, or a vendor that chose nothing. Those are the states where
        the feature is working as designed (the prompt is unchanged), and a line
        of chrome per user message telling the operator that an advisory feature
        had no opinion is noise; the failures are logged instead.

        The line names the RESOURCES and which message they were asked for. The
        vendor, the duration and the cost were on it and came off in the design
        round (D2/D3), each for its own reason:

        - ``$0.000157`` bypassed the repo's one money formatter (``format_usd``,
          ``tui/costs.py``, whose docstring claims every money surface reads it).
          This package cannot import that module — it would drag the terminal UI
          into the server and phone planes — and hand-rolling a second ladder is
          the failure that docstring warns about. The 18-character tail is also
          what pushed a 103-character line onto a second row at 100 columns;
        - a cache hit rendered ``(0.00s)``, a duration for a call that never
          happened;
        - ``via <vendor>`` named an implementation leg at a user, and printed
          ``via unknown vendor`` when no leg answered at all.

        The spend is not lost: the caller records it at INFO
        (``session_factory._log_classification_cost``), which is the operator's
        surface for it, and a user's money belongs in ``/usage``.

        ATTRIBUTION is the other half of the design round. ``Recommendation.late_urls``
        names the resources that were asked for by an EARLIER message — against a real
        vendor (~250 ms) and a 50 ms wait that is the ORDINARY case, delivered by the
        next message — so the sentence says which message each resource belongs to.
        Without that it reads as advice about the question it happens to sit under,
        which is actively wrong rather than merely unhelpful (D2). A prompt that gained
        BOTH sets is the one case that tags each resource separately: one label over a
        union would have to be false of half of it (QA round 4, Q1).
        """
        if not self.notice_enabled or not recommendation.resources:
            return None
        resources = tuple(recommendation.resources)
        # ``getattr``: the seam contract is these two methods, and a host's own
        # recommendation type is whatever it says it is — the harness only guarantees
        # the field set it was compiled against. Without the attribute the line falls
        # back to the uniform "for this message", which is the pre-attribution sentence:
        # worse than the truth, and much better than the AttributeError that made the
        # whole notice disappear.
        late = set(getattr(recommendation, "late_urls", ()) or ())
        if not late:
            urls = ", ".join(candidate.resource_url for candidate in resources)
            return f"Suggestion added for this message: {urls}"
        if len(late) == len(resources):
            urls = ", ".join(candidate.resource_url for candidate in resources)
            return f"Suggestion added for your previous message: {urls}"
        # MIXED, and this is the only shape that tags per resource: the union cannot
        # carry one attribution without lying about half of it (QA round 4, Q1 — the
        # late-first branch used to print "for your previous message" over a set whose
        # other half was chosen for the message the line sits under). Longer by design,
        # and only when the two sets really did arrive together.
        tagged = ", ".join(
            f"{candidate.resource_url} "
            f"({'your previous message' if candidate.resource_url in late else 'this message'})"
            for candidate in resources
        )
        return f"Suggestion added: {tagged}"

    # -- internals ---------------------------------------------------------

    async def _recommend(self, request: RecommendationRequest) -> Recommendation:
        """The real body, with the gates applied in their documented order."""
        if not self.enabled:
            return Recommendation(skipped="disabled")
        if not request.candidates:
            # Before the breaker and before the vendor: an empty roster is not a
            # failure, it is a session with nothing installed, and it must not
            # count toward opening the breaker.
            return Recommendation(skipped="empty-roster")
        if self._circuit_open:
            return Recommendation(skipped="circuit-open")

        key = _cache_key(request, max_candidates(self._settings))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            # A hit costs nothing — no tokens, no time. ``cost_usd`` becomes
            # None rather than the original spend, because THIS call spent
            # nothing; the resources themselves are free to reuse. The token
            # counts are cleared with it, so the cost line cannot report a spend
            # for a call that was not made.
            return self._capped(
                dataclasses.replace(
                    cached,
                    cost_usd=None,
                    input_tokens=None,
                    output_tokens=None,
                    latency_s=0.0,
                ),
                request.max_recommendations,
            )

        task = self._inflight.get(key)
        if task is None:
            # One in-flight call per (session, cache key): the second caller for
            # the same message joins the first rather than spending a second
            # request against a 0.5 req/s key (§3).
            task = asyncio.create_task(self._attempt(request, key))
            self._inflight[key] = task
            task.add_done_callback(lambda _finished, _key=key: self._inflight.pop(_key, None))
        # ``shield`` keeps the shared call alive when a joiner is cancelled: the
        # other callers are still waiting for it, and its result is about to be
        # cached for the messages after this one.
        return await asyncio.shield(task)

    async def _attempt(self, request: RecommendationRequest, key: str) -> Recommendation:
        """Build, call the cascade, map the answers — and cache the outcome."""
        started = time.monotonic()
        plan = self._plan_for(request)
        if not plan.questions:
            # Every question shape was disabled by a schema error, so there is
            # nothing to ask. Reported as an error rather than as "nothing
            # relevant": the vendor never had a chance to answer.
            logger.error(
                "classification: every question shape is disabled for this session (%s)",
                sorted(self._disabled_shapes),
            )
            return Recommendation(skipped="error", latency_s=time.monotonic() - started)

        state = build_state(
            user_message=request.user_message,
            context=request.context,
            candidates=request.candidates,
            settings=self._settings,
            roster_cache=self._roster_cache,
        )
        decision_request = build_decision_request(plan, state)
        timeout_s = self.timeout_s
        try:
            async with asyncio.timeout(timeout_s):
                response = await self._call_cascade(decision_request, timeout_s)
        except TimeoutError:
            logger.warning("classification: timed out after %.0f ms", timeout_s * 1000)
            self._record_failure("timeout")
            return Recommendation(skipped="timeout", latency_s=time.monotonic() - started)
        except DecisionSchemaError as exc:
            self._disable_shape(exc.shape)
            return Recommendation(skipped="error", latency_s=time.monotonic() - started)
        except _NoUsableLeg:
            # Neither an outage nor our bug: there is simply nothing to call, so
            # it neither opens the breaker nor logs a warning.
            return Recommendation(skipped="no-vendor", latency_s=time.monotonic() - started)
        except DecisionVendorError as exc:
            logger.warning("classification: no leg answered (%s: %s)", exc.kind, exc)
            self._record_failure(exc.kind)
            return Recommendation(skipped="error", latency_s=time.monotonic() - started)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — see the module docstring
            logger.warning("classification: vendor call failed", exc_info=True)
            self._record_failure("error")
            return Recommendation(skipped="error", latency_s=time.monotonic() - started)

        resources = collect_resources(
            response,
            plan,
            max_recommendations=min(
                request.max_recommendations, max_recommendations(self._settings)
            ),
        )
        recommendation = Recommendation(
            resources=resources,
            block=render_block(resources),
            vendor=response.vendor,
            cost_usd=response.cost_usd,
            # The vendor's own counts, carried so the harness's cost line can print
            # them: input is what this layer is billed for, so a line that could not
            # show it could not answer "what does one of these calls cost".
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_s=time.monotonic() - started,
        )
        self._consecutive_failures = 0
        self._remember(key, recommendation)
        return recommendation

    def _plan_for(self, request: RecommendationRequest) -> QuestionPlan:
        """Questions for this request, minus the shapes this session disabled."""
        return build_questions(
            request.candidates,
            limit=max_candidates(self._settings),
            skip_question_ids=frozenset(self._disabled_shapes),
        )

    async def _call_cascade(self, request: DecisionRequest, timeout_s: float) -> DecisionResponse:
        """Walk the legs in order; the first that answers wins.

        A ``DecisionSchemaError`` is re-raised rather than caught: it is our bug
        and §4 forbids pretending it is weather. Every other
        :class:`DecisionVendorError` moves to the next leg, and having exhausted
        the legs, the last error is raised for the caller to classify.

        A failed leg invalidates the resolved leg set so the next message
        re-resolves credentials: a key revoked or topped up mid-session is
        picked up without restarting the session.
        """
        legs = await self._available_legs()
        if not legs:
            raise _NoUsableLeg("no leg has a usable credential")
        last: DecisionVendorError | None = None
        for vendor in legs:
            try:
                response = await vendor.decide(request, timeout_s=timeout_s)
            except DecisionSchemaError:
                raise
            except DecisionVendorError as exc:
                logger.debug(
                    "classification: leg %s failed (%s), trying next", vendor.name, exc.kind
                )
                last = exc
                self._legs = None
                self._vendor_name = None
                continue
            self._vendor_name = vendor.name
            return response
        raise last if last is not None else DecisionVendorError("no leg answered", kind="transport")

    async def _available_legs(self) -> tuple[DecisionVendor, ...]:
        """Resolve the cascade once per session, keeping the instances.

        Uses :func:`~local_operator.classification.cascade.resolve_vendor` — the
        one definition of "first usable leg" — and then builds the legs AFTER the
        resolved one, because a leg ahead of it has just been proven unusable.
        Instances are cached, so a leg's credential is resolved once no matter
        how many messages the session classifies.
        """
        if self._legs is not None:
            return self._legs
        first = await resolve_vendor(self._manager, self._settings, client=self._client())
        if first is None:
            self._legs = ()
            return self._legs
        order = leg_order(self._settings)
        start = order.index(first.name) if first.name in order else 0
        self._vendors[first.name] = first
        self._vendor_name = self._vendor_name or first.name
        self._legs = tuple(self._vendor_for(name) for name in order[start:])
        return self._legs

    def _client(self) -> httpx.AsyncClient:
        """The session's keep-alive HTTP client, created on first use.

        One client for the session, not one per call: a connection plus TLS
        handshake is 50-150 ms, and this path runs before the turn's first token.
        The pool is deliberately small (a decision call is at most one request per
        user message) and the per-request timeout is still passed by each leg, so
        a settings change cannot leave a stale deadline on the client.
        """
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s),
                limits=httpx.Limits(
                    max_connections=4, max_keepalive_connections=2, keepalive_expiry=30.0
                ),
            )
        return self._http

    async def aclose(self) -> None:
        """Release the session's client and abandon whatever it still has in flight.

        The credential and roster memos are dropped with it, so the service is
        usable again afterwards (a fresh client is created on the next call) —
        which is what makes this safe to call from a teardown path that may run
        before a later message, rather than only at process exit.

        IN-FLIGHT FIRST, and that is the whole reason this is not just a client
        close: the session's wait is 50 ms against a ~250 ms answer, so a session
        disposed right after a message ALWAYS has one attempt running. Closing the
        keep-alive client under it failed that attempt with a transport error — a
        warning with a traceback, from a call whose answer nobody could use any
        more (review round 2, MINOR 2). Cancelling it first is the honest outcome:
        the work belonged to a session that is ending. ``gather(...,
        return_exceptions=True)`` because a cancelled attempt may finish with
        ``CancelledError`` rather than returning, and because teardown must never
        fail on the failure of the thing it is tearing down.
        """
        pending = [task for task in self._inflight.values() if not task.done()]
        for task in pending:
            # ``_recommend`` shields the attempt so a JOINER's cancellation cannot
            # kill a call other callers are waiting for. This is not a joiner: the
            # session is going away, so the attempt itself is what has to stop.
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._inflight.clear()
        client, self._http = self._http, None
        if client is not None:
            await client.aclose()
        self._vendors.clear()
        self._legs = None
        self._vendor_name = None

    def warm_up(self) -> None:
        """Build the keep-alive client NOW, so a session's first message does not.

        WHAT IT COSTS AND WHAT IT BUYS, measured (``scripts/classification_latency_probe.py
        --clients-only``, a fresh process per run, five constructions each): **tens of
        milliseconds for the first ``httpx.AsyncClient``, then 3-6 ms each** — 19.4 /
        23.0 / 36.0 ms in three quiet runs here and 27.0 / 81.3 / 42.0 ms in three on a
        loaded machine, which is the spread to expect, because the SSL context it builds
        dominates and the host is shared. It is built synchronously, before the call
        reaches its first await, so no wait budget can bound it; ``--prewarm`` pays it at
        session build.

        AN HONEST LIMIT ON THE CLAIM: paired runs of that probe (three per arm, same
        machine, same roster) put a session's FIRST message at +22 to +32 ms of our own
        time either way — 30.7 ms median without this call, 32.0 ms with it — so the
        prewarm did NOT measurably move the first message here, and an earlier draft of
        this docstring (which attributed the whole first-message cost to the client
        construction, at a 139 ms figure that does not reproduce) was wrong. What this
        does demonstrably is take a known one-off out of the first call and pay it where
        the operator is already waiting on skill discovery. The residual first-message
        cost is the package's own first-call setup — ``build_state`` measures 0.05 ms,
        so it is not the state build either — and the contract records it as open
        rather than explaining it away here.

        NOT a network call and NOT a credential read: this builds the client object
        only, and nothing connects until a request is made, which is what keeps it
        side-effect-free at boot. Enabled-only, so a default install — which never
        builds a service at all — still opens no client and imports nothing.
        """
        if not self.enabled:
            return
        self._client()

    def _vendor_for(self, name: str) -> DecisionVendor:
        """One cached leg instance per name for this session."""
        vendor = self._vendors.get(name)
        if vendor is None:
            vendor = build_vendor(
                name, self._manager, model=model_override(self._settings), client=self._client()
            )
            self._vendors[name] = vendor
        return vendor

    def _record_failure(self, kind: str) -> None:
        """Count a consecutive failure and open the breaker at the threshold.

        Only transport-class failures ever reach here (the schema path is
        handled separately), which is what keeps a malformed question from
        tripping a breaker that is meant to stop network calls.
        """
        self._consecutive_failures += 1
        if self._consecutive_failures >= CIRCUIT_FAILURE_THRESHOLD and not self._circuit_open:
            self._circuit_open = True
            logger.warning(
                "classification: circuit breaker opened after %d consecutive failures (last: %s)",
                self._consecutive_failures,
                kind,
            )

    def _disable_shape(self, shape: str | None) -> None:
        """Log a schema rejection loudly and stop asking that question this session.

        ``logger.error`` and not ``logger.warning``: this is our bug — a request
        shape the vendor refuses — and §4 is explicit that it must never be
        smoothed over into "try the next vendor". When the body named a question,
        that question id is disabled; when it named none, nothing is disabled and
        the next message will try again (and fail the same way, loudly, which is
        the correct signal for a bug we cannot yet attribute).
        """
        if shape is None:
            logger.error(
                "classification: vendor rejected our request shape but named no question; "
                "the next attempt will repeat it"
            )
            return
        self._disabled_shapes.add(shape)
        logger.error(
            "classification: vendor rejected question %r; disabled for this session",
            shape,
        )

    def _remember(self, key: str, recommendation: Recommendation) -> None:
        """Insert into the bounded LRU, evicting the oldest entry past the cap."""
        self._cache[key] = recommendation
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)

    def _capped(self, recommendation: Recommendation, cap: int) -> Recommendation:
        """Apply the caller's cap to a cached recommendation.

        Needed because the cache key deliberately excludes the cap (§4's key is
        message + roster): a cached three-resource result served to a caller who
        asked for one must be trimmed, block included, rather than silently
        exceeding the caller's budget.
        """
        limit = max(0, min(cap, max_recommendations(self._settings)))
        if len(recommendation.resources) <= limit:
            return recommendation
        resources = recommendation.resources[:limit]
        return dataclasses.replace(
            recommendation, resources=resources, block=render_block(resources)
        )


__all__ = [
    "CACHE_SIZE",
    "CIRCUIT_FAILURE_THRESHOLD",
    "DEFAULT_AUTO",
    "DEFAULT_MAX_RECOMMENDATIONS",
    "DEFAULT_NOTICE",
    "DEFAULT_TIMEOUT_MS",
    "ClassificationService",
]
