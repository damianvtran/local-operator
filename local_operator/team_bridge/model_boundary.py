"""H3 — the production model boundary: default resolution, wire guard, runner.

THREE PIECES, all narrow:

1. :func:`resolve_default_selection` — the CANONICAL configured default
   (``session/cold_model``'s reading of hosting/model_name, model defaulted
   through the provider's own default), built into a real ``ModelSpec`` by
   the same builder a session uses. A missing or unusable default returns
   ``None`` and the pipeline holds.
2. :class:`ScreeningWireGuard` — the physical pre-I/O guard. httpx routes
   REDIRECT hops and transport retries through ``handle_async_request``
   again, so the charge lives UNDER the client (a transport proxy), not
   around ``WireClient.stream``: every admitted physical request, nested
   resend/redirect included, reserves the account's HTTP bucket before the
   transport handler can touch the network. Provider errors are SANITIZED
   here, at the boundary — fixed text, typed flags only, ``__cause__`` AND
   ``__context__`` cleared — before shared failover logging can see vendor
   prose. Cancellation is control flow: it unwinds untouched.
3. :class:`ScreenModelRunner` — one fresh user request per screening: a
   single ``ChatRequest`` with ONE user message, empty tools, ``tool_choice
   "none"``, ``isolated=True``, no system blocks, no history, no cache
   lineage, no callbacks. Typed events only (``text_delta``/``delta``);
   reasoning is discarded; tool-call deltas and abnormal end reasons refuse.
   The serving identity comes from the stream's own request-local model
   event — never a fallback to the configured selection: unevidenced
   identity returns ``None`` and the pipeline holds.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

from local_operator.team_bridge.store import (
    SCREENING_FAILURE_CLASSES as _STORE_FAILURE_CLASSES,
)
from local_operator.team_bridge.store import Store, StoreRefused
from local_operator.team_bridge.types import OwnerSnapshot

__all__ = [
    "DefaultSelection",
    "SCREENING_FAILURE_CLASSES",
    "ScreenChainRunner",
    "ScreenModelRunner",
    "ScreeningWireGuard",
    "make_screening_runner",
    "resolve_default_selection",
    "resolve_screening_chain",
    "screening_failure_class",
]

#: Output ceiling for one screening reply. The frozen grammar is two short
#: lines, and a bounded stream is part of the cost contract -- but on a
#: reasoning model this ceiling ALSO bounds the thinking that precedes those
#: lines (Anthropic counts adaptive-thinking tokens against ``max_tokens``, and
#: the default spec seeds an effort level, so the screener thinks). 256 was
#: sized for the two lines alone: measured 2026-10-30 on ``claude-opus-5-5`` at
#: its seeded ``medium`` effort, 22 of 71 live screens ended ``stop=length`` at
#: exactly 256 output tokens with no verdict line, mostly ordinary messages
#: whose screen needed 220-350 tokens (xhigh peaked at 715). Those
#: truncations were what opened the account breaker in issue #3195. 1024 clears
#: the measured peak with headroom; the visible text stays bounded by
#: ``_SCREEN_TEXT_CAP`` and the strict two-line grammar either way, and output
#: is billed as used, not as reserved.
_SCREEN_MAX_TOKENS = 1024
#: Bound on consumed reply text (defence against a runaway stream).
_SCREEN_TEXT_CAP = 4096
#: The screening wall deadline, per the frozen contract — NOT the ordinary
#: session HTTP read timeout (600 s) and not max_tokens alone.
_SCREEN_WALL_DEADLINE_S = 45.0
_SCREEN_CONNECT_TIMEOUT_S = 10.0
#: Fixed text for sanitized wire failures — never vendor prose.
_SANITIZED_WIRE_ERROR = "screening provider call failed"


#: Stream refusals that mean the provider DID answer -- the route is healthy,
#: the reply just cannot admit -- mapped to the closed hold code the pipeline
#: records for each. Anything not listed (``error``/``aborted`` terminals, a
#: stream that ended without its end event, a pre-dispatch refusal) stays a
#: provider failure. See ``ScreenModelRunner._drive`` for where each is raised.
_ANSWERED_REFUSAL_HOLDS: dict[str, str] = {
    # The model's own safety stop on the screening prompt. A hostile message
    # is exactly what provokes it, so counting it as a route failure let a
    # sender trip the account breaker (#3195).
    "refusal": "model_refused",
    # The reply hit ``max_tokens`` before the verdict was complete.
    "length": "reply_truncated",
    # Replies that break the strict grammar in a way the parser never sees.
    # ``tool_use`` is the runner's own reason (a tool-call delta, or tool
    # content in a clean end's payload); ``toolUse`` is a ``StreamEndEvent``'s
    # stop reason passed through verbatim (``harness.types.StreamEndEvent``).
    "tool_use": "parse_failed",
    "toolUse": "parse_failed",
    "over_cap": "parse_failed",
}


class _ScreeningStreamRefused(Exception):
    """A typed refusal inside a screening stream (never carries prose)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    @property
    def answered_hold_code(self) -> str | None:
        """The hold code when the provider answered, else ``None``.

        Non-``None`` means a completed provider RESPONSE that is not an
        admissible verdict: the pipeline records a provider success (it must
        not advance the account breaker) and holds under this code.
        ``None`` means the route itself failed and the breaker counts it.
        """

        return _ANSWERED_REFUSAL_HOLDS.get(self.reason)


@dataclass(frozen=True)
class AdmissionJob:
    """The ONE immutable admission proof for a screening dispatch.

    Minted by the pipeline AFTER the logical attempt reservation and the
    native account turn claim, bound into the runner before dispatch, and
    re-validated by the native reservation transaction before EVERY actual
    send: its ``digest`` is the canonical full-job fingerprint carried by
    the logical row, every physical row and the finish/receipt binding.
    Nothing here is a caller-supplied name or a fallback.
    """

    owner_generation: int
    credential_id: int
    account_id: str
    tenant_id: str
    endpoint_id: str
    hub_url: str
    message_id: str
    attempt_id: str
    body_sha256: str
    template_sha256: str
    template_version: str
    provider: str
    model: str
    approved: tuple[str, ...]
    envelope_digest: str
    request_digest: str
    display_digest: str
    revision: str
    epoch: int
    turn: str

    @property
    def digest(self) -> str:
        """The canonical full-job digest (stable field order, no defaults)."""

        payload = json.dumps(
            [
                self.owner_generation,
                self.credential_id,
                self.account_id,
                self.tenant_id,
                self.endpoint_id,
                self.hub_url,
                self.message_id,
                self.attempt_id,
                self.body_sha256,
                self.template_sha256,
                self.template_version,
                self.provider,
                self.model,
                list(self.approved),
                self.envelope_digest,
                self.request_digest,
                self.display_digest,
                self.revision,
                self.epoch,
                self.turn,
            ],
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _owner_key_text(owner: OwnerSnapshot) -> str:
    """Stable owner text for the no-fingerprint fallback (never NULL ledgers)."""

    return f"{owner.credential_id}|{owner.account_id}|{owner.tenant_id}|{owner.endpoint_id}"


def _payload_has_tool_content(payload: Any, depth: int = 0) -> bool:
    """Whether a terminal/provider payload carries function/tool content.

    The normal Responses decoder usually maps a terminal function_call to
    ``stop_reason='toolUse'``, but QA observed a real final-only call whose
    end event kept ``stop_reason='stop'`` with the marker visible only in
    the payload. A screening verdict must never ride tool content, so the
    payload is scanned (bounded) regardless of the stop reason.
    """

    if depth > 4 or payload is None:
        return False
    if isinstance(payload, dict):
        if payload.get("type") == "function_call" or payload.get("function_call"):
            return True
        if isinstance(payload.get("tool_calls"), list) and payload["tool_calls"]:
            return True
        return any(_payload_has_tool_content(value, depth + 1) for value in payload.values())
    if isinstance(payload, list):
        return any(_payload_has_tool_content(item, depth + 1) for item in payload)
    return False


@dataclass
class DefaultSelection:
    """The captured canonical default, with its approved identity set."""

    provider: str
    model_id: str
    spec: Any  # a real ModelSpec; Any keeps this module import-light
    approved_identities: frozenset[str] = field(default_factory=frozenset)
    #: The canonical (provider, model) TUPLES this selection approves: the
    #: serving identity check compares tuples, so a foreign provider
    #: offering the same model id is refused (R6/Q10).
    approved_tuples: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # A selection ALWAYS approves its own resolved identity: callers that
        # construct one directly (tests, adapters) must not accidentally
        # disable evidence matching by leaving the set empty.
        if not self.approved_identities:
            object.__setattr__(self, "approved_identities", frozenset({self.model_id}))
        if not self.approved_tuples:
            object.__setattr__(
                self,
                "approved_tuples",
                frozenset(
                    {(self.provider, mid) for mid in self.approved_identities}
                    | {(self.provider, self.model_id)}
                ),
            )

    @property
    def default_model(self) -> str:
        return self.model_id


def resolve_default_selection(config_dir: Path) -> DefaultSelection | None:
    """The canonical configured default for screening, or None when unusable.

    Reads the same pair ``Session`` starts from and resolves it through the
    standard spec builder, so an explicitly configured alias keeps its
    normal server-side resolution. The approved identity set is derived
    HERE from that resolution — never from caller strings.
    """

    try:
        from local_operator.model.configure import build_model_spec
        from local_operator.session.cold_model import configured_model_pair

        provider, model_id = configured_model_pair(config_dir)
        if not provider or not model_id:
            return None
        spec = build_model_spec(provider, model_id)
        resolved = str(getattr(spec, "model_id", "") or "")
        if not resolved:
            return None
        return DefaultSelection(
            provider=provider,
            model_id=resolved,
            spec=spec,
            approved_identities=frozenset({resolved}),
        )
    except Exception:  # noqa: BLE001 — an unusable default is a HOLD, not a crash
        return None


def _retry_values(config_dir: Path, settings: Any) -> Any:
    """The harness settings whose ``retry`` block governs the screening chain.

    WHY THE DISK READ: the production connector calls the screening cycle with
    ``settings=None`` (``connector/__main__.py`` hands ``run_forever`` none), so
    a chain read only from the argument would be empty live and screening
    would stay single-shot — the exact outage this exists to end. A caller
    that DOES pass a mapping carrying ``retry`` (tests, an embedding host) is
    authoritative; otherwise the operator's config on disk is read fresh, the
    same never-writing reader the connector's own fence uses.
    """

    if isinstance(settings, Mapping) and "retry" in settings:
        return settings
    try:
        from local_operator.config import read_config_values

        return read_config_values(config_dir) or {}
    except Exception:  # noqa: BLE001 — an unreadable config means "no chain"
        return {}


def resolve_screening_chain(config_dir: Path, settings: Any = None) -> list[DefaultSelection]:
    """The ordered screening hops: the configured default, then its fallbacks.

    WHY THIS EXISTS: a screening request is ``isolated=True`` and the failover
    layer turns retry OFF for isolated requests (``providers/failover.py``,
    the ``if request.isolated`` arm of the stream), so the session-level
    fallback chain never runs for screening. One flaky default then parked
    every inbound message behind the breaker while the operator's configured
    ``retry.fallbackChains`` sat unused. The chain is walked HERE instead,
    one isolated attempt per hop, leaving failover's decorative-errand
    semantics untouched for every other isolated caller.

    Element 0 is always :func:`resolve_default_selection` (empty list when
    there is no usable default). Further hops come from the SAME chain
    resolution failover uses (``resolve_chain`` + ``expand_fallback_targets``)
    and only when ``retry.enabled`` and ``retry.modelFallback`` both hold.
    CONSTRAINTS: each hop approves ONLY its own resolved (provider, model)
    tuple — approval never leaks across hops, so a foreign provider serving
    another hop's model id is still refused; a hop whose spec cannot be built
    is skipped; duplicates (by tuple) are dropped keeping first order. A
    target's per-entry ``effort`` is not carried: screening is a fixed,
    bounded two-line grammar and builds its spec exactly as the default does.
    """

    primary = resolve_default_selection(config_dir)
    if primary is None:
        return []
    hops = [primary]
    try:
        from local_operator.model.configure import build_model_spec
        from local_operator.providers.failover import (
            RetrySettings,
            expand_fallback_targets,
            parse_selector,
            resolve_chain,
        )

        retry = RetrySettings.from_settings(_retry_values(config_dir, settings))
        if not (retry.enabled and retry.model_fallback):
            return hops
        primary_selector = f"{primary.provider}/{primary.model_id}"
        chain = resolve_chain(primary_selector, retry.fallback_chains)
        targets = expand_fallback_targets(primary_selector, chain) if chain else []
    except Exception:  # noqa: BLE001 — no readable chain degrades to the default alone
        return hops
    seen = {(primary.provider, primary.model_id)}
    for target in targets:
        provider, model_id = parse_selector(target.selector)
        try:
            spec = build_model_spec(provider, model_id)
        except Exception:  # noqa: BLE001 — an unbuildable hop is skipped, not fatal
            continue
        resolved = str(getattr(spec, "model_id", "") or "")
        if not provider or not resolved or (provider, resolved) in seen:
            continue
        seen.add((provider, resolved))
        hops.append(
            DefaultSelection(
                provider=provider,
                model_id=resolved,
                spec=spec,
                approved_identities=frozenset({resolved}),
            )
        )
    return hops


#: The CLOSED vocabulary of ``screening_holds.failure_class`` — owned by the
#: store, which clamps every write to it (anything else becomes ``other``).
SCREENING_FAILURE_CLASSES = _STORE_FAILURE_CLASSES


#: HTTP statuses that mean "this route is unavailable right now", which no
#: submission can produce: gateway/overload responses. 500 is deliberately
#: absent — providers report an output-side content filter as a 5xx, and a
#: body crafted to trip it must not buy a different screener (S-1).
_ADVANCE_STATUSES = frozenset({502, 503, 504, 529})
#: Rate limit / out of quota. Floodable in principle, but bounded by the
#: per-account HTTP budget and accepted by the operator (PR #3194 decisions).
_QUOTA_STATUSES = frozenset({429})
#: The provider rejected OUR credential, never the content. 401 ONLY: a 403 is
#: how OpenRouter reports moderation flags, guardrail blocks,
#: ``content_policy_violation`` and ``refusal`` (its documented error codes),
#: and failover's ``_classify_fields`` labels every marker-less 403
#: ``kind="auth"`` — so neither the status nor the kind can separate a bad
#: credential from a content verdict. A 403 HOLDS (PR #3194 review M2 / S-5).
_AUTH_STATUSES = frozenset({401})
#: Content-verdict wording a provider can relay on an otherwise-advancing
#: gateway status — Alibaba's "Output data may contain inappropriate content"
#: arrives as a relayed 502 (``failover.py`` ``is_input_refusal`` docstring).
#: Matched case-insensitively on the RAW message. ONE-WAY by construction: it
#: can only turn an advance into a hold, never the reverse, so a provider
#: phrasing an outage this way costs a held message, not a weaker screener.
_CONTENT_VERDICT_MARKERS = (
    "inappropriate content",
    "moderation",
    "flagged",
    "content_policy",
    "content policy",
    "guardrail",
)

#: Typed flag the guard sets on a scrubbed wire failure. ``True`` only when
#: the RAW failure was on the allow-list below; the walk reads nothing else.
_ADVANCE_ATTR = "tb_screen_advance"


def _sender_cannot_cause(failure: BaseException) -> bool:
    """Whether the next hop may screen after THIS raw failure (S-1/S-2).

    WHY AN ALLOW-LIST: the threat is a sender who controls the body and
    attachments. Any failure their bytes can provoke — a content-filter 400,
    a 413 / bad-image 400, an output-filter 5xx, a length or refusal stop —
    must HOLD, because falling over would let crafted content buy a weaker
    screener, and the first provider refusing the input is itself evidence.
    Only failures of the ROUTE advance: our credential rejected (401, or an
    OAuth refresh that failed — never 403, see ``_AUTH_STATUSES``), quota
    (429), a timeout, a gateway/overload status (502/503/504/529) whose raw
    text names no content verdict, or a transport error our own client raised
    before the body was sent.

    MUST be evaluated on the RAW exception, before ``_scrub_wire_failure``
    erases everything but typed fields; it reads types, ``status``, typed
    flags and, one-way and hold-only, the content-verdict markers on an
    advancing gateway status; no other prose. The guard's own fail-closed refusals
    (owner moved, policy stale, budget exhausted, route changed) are built
    AFTER this point and never carry the flag, so they stop the walk.
    """

    if isinstance(failure, (asyncio.TimeoutError, TimeoutError)):
        return True
    try:
        import httpx

        if isinstance(failure, httpx.TimeoutException):
            return True
        if isinstance(failure, (httpx.ConnectError, httpx.ConnectTimeout)):
            return True  # raised before any request byte reached the provider
    except ImportError:  # pragma: no cover - httpx is a hard dependency
        pass
    try:
        from local_operator.providers.auth_store import AuthStoreError

        if isinstance(failure, AuthStoreError):
            return True  # credential resolution / OAuth refresh failed
    except ImportError:  # pragma: no cover
        pass
    from local_operator.providers.failover import ProviderError

    if not isinstance(failure, ProviderError):
        return False
    if getattr(failure, "fail_closed", False) and not getattr(failure, "request_sent", True):
        return False  # a local refusal before any send: never provider weather
    status = getattr(failure, "status", None)
    kind = getattr(failure, "kind", None)
    if isinstance(status, int) and status:
        if status in _AUTH_STATUSES or status in _QUOTA_STATUSES:
            return True
        if status not in _ADVANCE_STATUSES:
            return False
        raw = str(getattr(failure, "message", "") or "").lower()
        return not any(marker in raw for marker in _CONTENT_VERDICT_MARKERS)
    # No status: a status-less ``kind`` is classified from PROVIDER TEXT (an
    # in-band error chunk on a 200 stream), which a crafted body can steer — so
    # it never advances on its own. Only typed provenance does: a failure OUR
    # client observed (``transport``, set solely by ``wrap_transport_error``)
    # that is a timeout or a connect error. (``auth_error`` is NOT read: the
    # clients set it for 401 AND 403, and a 403 can be a content verdict.)
    if not getattr(failure, "transport", False):
        return False
    if kind == "timeout":
        return True
    try:
        import httpx

        return isinstance(failure.__cause__, httpx.ConnectError)
    except ImportError:  # pragma: no cover
        return False


def screening_failure_class(failure: BaseException) -> str:
    """Map a dispatch failure to :data:`SCREENING_FAILURE_CLASSES` (never prose).

    ``chain_exhausted`` wins: the chain runner tags the last failure of a
    multi-hop walk, and "every configured route failed" is the fact the
    operator acts on, not the last hop's own kind. ``identity_refused`` is
    part of the shared vocabulary (PR #3189 reads it) but an unevidenced
    identity is a ``("", None)`` RETURN, not an exception, so it is held as
    ``model_identity_unknown`` and never reaches this mapping today.
    """

    if getattr(failure, "tb_chain_exhausted", False):
        return "chain_exhausted"
    if isinstance(failure, _ScreeningStreamRefused):
        return "stream_refused"
    if isinstance(failure, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    from local_operator.providers.failover import ProviderError

    if isinstance(failure, ProviderError):
        kind = getattr(failure, "kind", None)
        status = getattr(failure, "status", None)
        if kind == "auth" or getattr(failure, "auth_error", False):
            return "auth_unusable"
        if kind == "quota":
            return "quota"
        if kind == "timeout":
            return "timeout"
        if kind == "transient" or (isinstance(status, int) and 500 <= status <= 599):
            return "provider_5xx"
    return "other"


def _running_loop() -> bool:
    """Whether THIS thread already has an event loop running (never nest one)."""

    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class _ChargingAsyncTransport:
    """An httpx transport proxy that charges EVERY physical send.

    Wrapping the CLIENT is not enough: with redirect following enabled httpx
    performs the hop inside one ``.stream`` call, invisible to a WireClient
    callback (QA R3 proved two POSTs under one charge). httpx re-enters
    ``handle_async_request`` per hop, so this layer is the pre-I/O boundary.
    Transport-level retries INSIDE the wrapped transport remain invisible;
    the factory's dedicated client disables them (``retries=0``), and a
    caller-supplied client's own retries are a documented residual limit.
    """

    def __init__(
        self, inner: Any, charge: Callable[[], None], owner_token: object | None = None
    ) -> None:
        self._inner = inner
        self._charge = charge
        # Identity of the guard that installed this proxy: a second guard
        # meeting an existing proxy can tell "mine" from "foreign" and must
        # never wrap again (double-charge) or adopt a foreign job's guard.
        self.owner_token = owner_token

    async def handle_async_request(self, request: Any) -> Any:
        self._charge()
        return await self._inner.handle_async_request(request)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _scrub_wire_failure(failure: BaseException) -> BaseException:
    """A content-free replacement for ANY wire-layer failure.

    Preserves only typed flags; substitutes the message; clears BOTH
    ``__cause__`` and ``__context__`` (``raise ... from None`` still leaves
    ``__context__`` set, which is exactly where QA found the raw vendor
    body), and drops transport metadata. An UNKNOWN send stays conservative:
    ``request_sent`` is never False unless the refusal happened provably
    before any dispatch.
    """

    from local_operator.providers.failover import ProviderError

    if isinstance(failure, ProviderError):
        scrub = ProviderError(
            failure.status or 0,
            _SANITIZED_WIRE_ERROR,
            retryable=False,
            retry_after_ms=failure.retry_after_ms,
            auth_error=failure.auth_error,
            kind=failure.kind,
            request_sent=failure.request_sent,
            fail_closed=True,
        )
    else:
        scrub = ProviderError(
            0,
            _SANITIZED_WIRE_ERROR,
            retryable=False,
            request_sent=True,  # unknown, conservative — never "unsent" without proof
            fail_closed=True,
        )
    scrub.__cause__ = None
    scrub.__context__ = None
    # The ONE fact the chain walk needs, decided on the raw failure before its
    # evidence is erased (S-1): a typed boolean, never prose.
    setattr(scrub, _ADVANCE_ATTR, _sender_cannot_cause(failure))
    return scrub


class ScreeningWireGuard:
    """Charge every physical send; sanitize every wire failure.

    Install by wrapping the WireClient (``wrap_client`` seam). The guard
    locates the client's httpx transport — from ``http_client`` when the
    caller passes it, else from the wire client's own ``_http`` — and parks
    a charging proxy under it. When no transport is reachable the guard
    still charges once per stream call (best effort) and says so in a
    comment: callbacks alone are NOT the physical boundary.
    """

    def __init__(
        self,
        client: Any,
        *,
        store: Store,
        owner: OwnerSnapshot,
        message_id: str,
        fingerprint: str | None = None,
        http_client: Any | None = None,
        expected_epoch: int | None = None,
        job_ref: dict[str, Any] | None = None,
    ) -> None:
        # The shared job cell the runner binds the minted AdmissionJob into:
        # the dispatch reservation REQUIRES it — a factory acall with no
        # logical job refuses BEFORE any I/O (R5).
        self._job_ref = job_ref if job_ref is not None else {}
        self._client = client
        self._store = store
        self._owner = owner
        self._message_id = message_id
        self._guard_token = object()
        # A physical reservation must NEVER ride anonymous: when no full job
        # fingerprint was supplied, a deterministic owner/message-scoped one
        # still binds the charge to this job (never NULL).
        self._fingerprint = fingerprint or f"job|{_owner_key_text(owner)}|{message_id}"
        self._expected_epoch = expected_epoch
        self._transport_patched = False
        # Pin the job's policy identity where the guard is BUILT: a trusted
        # publisher moving the row afterwards (any direction) refuses every
        # later charge with zero handler calls, and a missing/unreadable row
        # at build time can never be silently adopted mid-stream.
        try:
            self._pinned_policy = store.current_screening_policy(owner)
        except Exception:  # noqa: BLE001 - an unreadable store pins NOTHING
            self._pinned_policy = None
        self._proxies: list[_ChargingAsyncTransport] = []
        # (container, key, proxy, kind): kind 'attr' reads via getattr,
        # 'item' via container.get(key) — a MOUNT dict is keyed by an
        # httpx URLPattern, which getattr cannot address (R6/Q14).
        self._installed: list[tuple[Any, Any, _ChargingAsyncTransport, str]] = []
        http = http_client if http_client is not None else getattr(client, "_http", None)
        if http is not None:
            patched_any = self._patch_transport(http, "_transport")
            # Mounted transports are SEPARATE send paths (an https:// mount
            # bypasses _transport entirely): every non-None mount must be
            # guarded too, or the client is opaque and every send refuses.
            mounts = getattr(http, "_mounts", None)
            if isinstance(mounts, dict):
                for pattern, transport in list(mounts.items()):
                    if transport is not None:
                        patched_any = self._patch_mount_transport(http, pattern) or patched_any
            self._transport_patched = patched_any and self._all_proxies_current(http)

    def _patch_transport(self, http: Any, attr: str) -> bool:
        transport = getattr(http, attr, None)
        if transport is None:
            return False
        if isinstance(transport, _ChargingAsyncTransport):
            # Already proxied: never wrap twice (that would double-charge);
            # the proxy is adopted only if it is THIS job's own guard.
            if transport.owner_token is self._guard_token:
                self._proxies.append(transport)
                return True
            return False
        proxy = _ChargingAsyncTransport(transport, self._charge, owner_token=self._guard_token)
        setattr(http, attr, proxy)
        self._proxies.append(proxy)
        self._installed.append((http, attr, proxy, "attr"))
        return True

    def _patch_mount_transport(self, http: Any, pattern: Any) -> bool:
        mounts = getattr(http, "_mounts", None)
        if not isinstance(mounts, dict):
            return False
        transport = mounts.get(pattern)
        if transport is None:
            return False
        if isinstance(transport, _ChargingAsyncTransport):
            if transport.owner_token is self._guard_token:
                self._proxies.append(transport)
                return True
            return False
        proxy = _ChargingAsyncTransport(transport, self._charge, owner_token=self._guard_token)
        mounts[pattern] = proxy
        self._proxies.append(proxy)
        self._installed.append((mounts, pattern, proxy, "item"))
        return True

    def _all_proxies_current(self, http: Any) -> bool:
        if not self._proxies:
            return False
        transport = getattr(http, "_transport", None)
        if (
            isinstance(transport, _ChargingAsyncTransport)
            and transport.owner_token is self._guard_token
        ):
            pass
        mounts = getattr(http, "_mounts", None)
        if isinstance(mounts, dict):
            for transport in mounts.values():
                if transport is not None and not (
                    isinstance(transport, _ChargingAsyncTransport)
                    and transport.owner_token is self._guard_token
                ):
                    return False
        return True

    def route_ok(self, route: Any) -> bool:
        """ACTUAL selected-send boundary check (R7/Q14): the transport httpx
        SELECTED for this request must be one of THIS guard's own proxies.

        Called from the dedicated client's synchronous `_transport_for_url`
        immediately before the send, so a primary replacement, a late
        foreign mount, a reused proxy or an opaque route is refused with
        ZERO I/O — the outer stream check alone cannot see the selection.
        """

        return any(route is proxy for proxy in self._proxies)

    def route_error(self) -> BaseException:
        """The typed refusal for a selected route that is not ours."""

        return self._refuse("screening selected route is not this job's guard")

    def _guard_current(self) -> bool:
        """Whether every recorded transport is still THIS guard's proxy."""

        for container, key, proxy, kind in self._installed:
            current = getattr(container, key, None) if kind == "attr" else container.get(key)
            if current is not proxy:
                return False
        return True

    def _verify_job(self) -> None:
        """The immutable CURRENT job re-checked before EVERY physical send.

        Owner generation must still be live, the pinned policy row must be
        byte-equal to the current one (a trusted publisher advancing the
        revision/epoch between construction and dispatch refuses with ZERO
        handler calls), and — when the caller supplied one — the charged
        epoch must still be the published one.
        """

        from local_operator.providers.failover import ProviderError

        refusal = None
        try:
            live = self._store.active_owner(credential_id=self._owner.credential_id)
            if live != self._owner:
                refusal = "owner_changed"
            else:
                current = self._store.current_screening_policy(self._owner)
                if current != self._pinned_policy:
                    refusal = "policy_stale"
                elif self._expected_epoch is not None and (
                    current is None or int(current["epoch"]) != int(self._expected_epoch)
                ):
                    refusal = "policy_stale"
        except Exception:  # noqa: BLE001 - unreadable authority = refuse
            refusal = "owner_changed"
        if refusal is not None:
            hold = ProviderError(
                409,
                f"screening job refused before send ({refusal})",
                retryable=False,
                kind="request",
                request_sent=False,
                fail_closed=True,
            )
            hold.__cause__ = None
            hold.__context__ = None
            raise hold

    def _charge(self) -> None:
        job = self._job_ref.get("job")
        if job is None:
            raise self._refuse("screening job not bound; refusing to send")
        if not self._guard_current():
            # The selected transport is no longer the proxy THIS guard
            # installed (replaced/mounted/reused foreign): a send could
            # escape uncharged — refuse before I/O, every request.
            raise self._refuse("screening route changed after guard installation")
        try:
            self._store.reserve_screening_dispatch(
                self._owner, message_id=self._message_id, job=job, turn=job.turn
            )
        except StoreRefused as failure:
            raise self._refuse(f"screening dispatch refused ({failure.code})") from None

    def _refuse(self, reason: str) -> BaseException:
        from local_operator.providers.failover import ProviderError

        hold = ProviderError(
            409, reason, retryable=False, kind="request", request_sent=False, fail_closed=True
        )
        hold.__cause__ = None
        hold.__context__ = None
        return hold

    async def stream(  # pragma: no cover - thin delegation, driven by tests
        self, request: Any, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        if not self._transport_patched:
            # OPAQUE ROUTE: nothing we can charge at the send boundary.
            # Best-effort per-stream charging is forbidden fallback authority
            # (it cannot see nested sends), so the stream refuses BEFORE any
            # I/O — a bare wire object with no patchable transport must not
            # screen at all.
            from local_operator.providers.failover import ProviderError

            opaque = ProviderError(
                409,
                "screening route is not chargeable (no interceptable transport)",
                retryable=False,
                kind="request",
                request_sent=False,
                fail_closed=True,
            )
            opaque.__cause__ = None
            opaque.__context__ = None
            raise opaque
        if not self._guard_current():
            # The CURRENT selected route (primary or mount) is no longer the
            # proxy THIS guard installed for THIS job — replaced, foreign or
            # re-mounted after installation. A historical install flag is
            # not evidence: refuse BEFORE any I/O (R6/Q14).
            raise self._refuse("screening route changed after guard installation")
        pending: BaseException | None = None
        try:
            # The CALL itself sits inside the scrub: a client that raises at
            # call time must not escape with vendor prose intact.
            inner = self._client.stream(request, api_key, oauth_access)
            async for event in inner:
                yield event
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit, GeneratorExit):
            # Control flow, not a provider fault: unwind exactly as received
            # (a cancelled post-POST send must never be laundered into a
            # request_sent=False hold).
            raise
        except BaseException as failure:  # noqa: BLE001 - sanitize EVERYTHING here
            pending = _scrub_wire_failure(failure)  # noqa: F841 - used below
        if pending is not None:
            # Raised OUTSIDE the except block on purpose: ``raise`` inside an
            # except suite re-attaches the raw failure as ``__context__``
            # (``from None`` only suppresses DISPLAY), which is exactly where
            # QA found the vendor canary riding along.
            raise pending


class ScreenModelRunner:
    """Drive ONE isolated screening request through the normal stream fn.

    Async is the primary public interface (:meth:`acall`); the synchronous
    facade exists for callers already outside a loop and deliberately does
    NOT nest one — inside an active loop it returns the unevidenced
    ("", None) hold signal instead of raising, and the caller should use
    ``acall``. Wall time is bounded; owned resources close in ``finally``.
    """

    def __init__(
        self,
        stream_fn: Any | None,
        selection: DefaultSelection,
        *,
        deadline_s: float = _SCREEN_WALL_DEADLINE_S,
        owned_http: Any | None = None,
        owned_stream_fn: Any | None = None,
        job_ref: dict[str, Any] | None = None,
        ensure_fn: Any | None = None,
        lazy_state: dict[str, Any] | None = None,
    ) -> None:
        self._stream_fn = stream_fn
        self._selection = selection
        self._deadline_s = deadline_s
        self._owned_http = owned_http
        self._owned_stream_fn = owned_stream_fn
        self._job_ref = job_ref if job_ref is not None else {}
        # Lazy production path (R6): the client/stream fn are built on first
        # use; the state cell is what _close_owned tears down.
        self._ensure_fn = ensure_fn
        self._lazy_state = lazy_state

    def bind_job(self, job: AdmissionJob) -> None:
        """Bind the minted AdmissionJob for the enclosing dispatch (R5).

        The guard's native reservation REQUIRES it; binding happens after
        the logical attempt and the native account turn are held, right
        before the public dispatch, so no send can ride an unbound job.
        """

        self._job_ref["job"] = job

    def __call__(
        self,
        prompt: str,
        *,
        default_model: str,
        images: "Sequence[tuple[bytes, str]] | None" = None,
    ) -> tuple[str, str | None]:
        if self._selection.default_model != default_model:
            # Built for a DIFFERENT default than the pipeline screens under:
            # identity cannot be evidenced, so hold.
            return "", None
        if _running_loop():
            # Never nest a loop or borrow a thread; the public async entry
            # (acall) is the correct call inside a running loop.
            return "", None
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.acall(prompt, default_model=default_model, images=images)
            )
        finally:
            loop.close()

    async def acall(
        self,
        prompt: str,
        *,
        default_model: str,
        images: "Sequence[tuple[bytes, str]] | None" = None,
    ) -> tuple[str, str | None]:
        """The PUBLIC async entry: bounded wall time, owned cleanup in finally.

        ``images`` are ``(bytes, mime)`` pairs from the screen-purpose fetch,
        already verified and bounded. They share this call's single dispatch,
        deadline, token cap and budget unit — an image screen is ONE screening,
        not a second channel.
        """

        try:
            if self._selection.default_model != default_model:
                return "", None
            return await asyncio.wait_for(
                self._drive(prompt, images=images), timeout=self._deadline_s
            )
        finally:
            # One-shot by contract: owned resources close on EVERY completion
            # — identity hold, deadline, cancel, error, success alike (QA: an
            # open client after an early hold is a leak).
            await self._close_owned()

    async def aclose(self) -> None:
        """Close whatever this runner owns (idempotent)."""

        await self._close_owned()

    async def _close_owned(self) -> None:
        if self._lazy_state is not None:
            stream_fn = self._lazy_state.pop("stream_fn", None)
            client = self._lazy_state.pop("client", None)
            if stream_fn is not None:
                try:
                    await stream_fn.close()
                except Exception:  # noqa: BLE001 — teardown, never fatal
                    pass
            if client is not None:
                try:
                    await client.aclose()
                except Exception:  # noqa: BLE001 — teardown, never fatal
                    pass
            return
        if self._owned_stream_fn is not None:
            try:
                await self._owned_stream_fn.close()
            except Exception:  # noqa: BLE001 — teardown, never fatal
                pass
            self._owned_stream_fn = None
        if self._owned_http is not None:
            try:
                await self._owned_http.aclose()
            except Exception:  # noqa: BLE001 — teardown, never fatal
                pass
            self._owned_http = None

    async def _drive(
        self, prompt: str, *, images: "Sequence[tuple[bytes, str]] | None" = None
    ) -> tuple[str, str | None]:
        from local_operator.harness.types import (
            ChatRequest,
            ImageContent,
            Message,
            TextContent,
        )

        spec = self._selection.spec
        sees_images = bool(getattr(spec, "supports_images", False)) and bool(
            getattr(spec, "supports_images_stated", True)
        )
        if images and not sees_images:
            # D5, fail closed: a screener that cannot see images must never be
            # sent them (a provider would strip or 400 them and the verdict
            # would then cover pixels nobody looked at). The connector screens
            # text-only for such a model; reaching here means the selection
            # changed under it, which is a hold, never a text-only allow.
            # "Can see" means STATED: an undescribed model's spec defaults to
            # True, which is not knowledge (PR #3071 review m2).
            raise _ScreeningStreamRefused("images_unsupported")
        image_blocks = [
            ImageContent(data=base64.b64encode(data).decode("ascii"), mime_type=mime)
            for data, mime in (images or ())
        ]
        request = ChatRequest(
            model=self._selection.spec,
            # ONE fresh user turn: no system blocks, no history, no tools,
            # no cache lineage, no callbacks — the isolation contract. Images
            # follow the prompt text inside that same turn (Message.user's
            # order), so the instructions frame them as data.
            messages=[
                (
                    Message.user(prompt, images=image_blocks)
                    if image_blocks
                    else Message(role="user", content=[TextContent(text=prompt)])
                )
            ],
            tools=[],
            tool_choice="none",
            isolated=True,
            purpose="team_bridge_screen",
            max_tokens=_SCREEN_MAX_TOKENS,
            # A verdict should be a function of the bytes, not of sampling
            # noise: identical submissions were measured flipping 2 of 4
            # verdicts at the model's default temperature. Safe on every hop:
            # ``providers.clients._sampling_params`` omits the key for a model
            # whose spec rejects sampling params. top_p is deliberately left
            # to the vendor default.
            temperature=0.0,
        )
        stream_fn = self._stream_fn
        if stream_fn is None and self._ensure_fn is not None:
            # First use: build the dedicated client + stream fn NOW, where a
            # construction failure can await the cleanup instead of
            # abandoning a task (R6/Q10).
            stream_fn = await self._ensure_fn()
            self._stream_fn = stream_fn
        if stream_fn is None:
            return "", None
        parts: list[str] = []
        consumed = 0
        completed = False
        candidates: list[tuple[str, str]] = []

        def _consider_usage(usage: Any) -> None:
            # The failover stamps usage.provider/model_id from the spec that
            # ACTUALLY went on this request's wire (_stamp_serving_spec).
            # That stamp is request-local provenance; the provider BODY's
            # model string is never read for identity.
            if usage is None:
                return
            provider = getattr(usage, "provider", None)
            mid = getattr(usage, "model_id", None)
            if isinstance(mid, str) and mid:
                candidates.append((str(provider or ""), mid))

        async for event in stream_fn(request, None):
            etype = str(getattr(event, "type", "") or "")
            if etype == "model":
                model = getattr(event, "model", None)
                mid = getattr(model, "model_id", None)
                provider = getattr(model, "provider", None)
                if isinstance(mid, str) and mid:
                    candidates.append((str(provider or ""), mid))
            elif etype == "usage":
                _consider_usage(getattr(event, "usage", None))
            elif etype == "text_delta":
                text = getattr(event, "delta", None)
                if isinstance(text, str) and text:
                    consumed += len(text)
                    if consumed > _SCREEN_TEXT_CAP:
                        raise _ScreeningStreamRefused("over_cap")
                    parts.append(text)
            elif etype == "reasoning_delta":
                continue  # consumed and discarded
            elif etype == "end":
                stop_reason = str(getattr(event, "stop_reason", "") or "")
                if stop_reason != "stop":
                    # length / toolUse / refusal / error / aborted (and a
                    # blank/missing stop) can NEVER admit.
                    raise _ScreeningStreamRefused(stop_reason or "missing_stop")
                if _payload_has_tool_content(getattr(event, "provider_payload", None)):
                    # Final-only function content with stop='stop' (QA's
                    # observed decoder shape) still refuses.
                    raise _ScreeningStreamRefused("tool_use")
                _consider_usage(getattr(event, "usage", None))
                completed = True
            elif etype == "tool_call_delta":
                # A tool call in a screening reply is never invoked and can
                # never ride a verdict.
                raise _ScreeningStreamRefused("tool_use")
        if not completed:
            # Generator exhaustion is NOT completion (Q13): an authoritative
            # end event with a clean stop is required.
            raise _ScreeningStreamRefused("missing_end")
        if not candidates:
            return "".join(parts), None
        distinct = {(provider, mid) for provider, mid in candidates}
        if len(distinct) != 1:
            # Conflicting request-local provenance can never be averaged.
            return "", None
        provider, mid = distinct.pop()
        # The TUPLE is the identity (R6/Q10): a foreign provider offering
        # the same approved model id is NOT this selection's identity.
        canonical = (self._selection.provider, self._selection.model_id)
        if (provider, mid) != canonical and (provider, mid) not in getattr(
            self._selection, "approved_tuples", set()
        ):
            return "", None
        return "".join(parts), mid


class ScreenChainRunner:
    """Walk the screening hops in order: one isolated attempt per hop.

    WHY: failover disables retry for ``isolated=True`` requests, so the
    operator's fallback chain never served screening (see
    :func:`resolve_screening_chain`). This runner is the chain walk, kept in
    the team-bridge boundary so failover's isolated semantics stay unchanged.

    CONTRACT (same public shape as :class:`ScreenModelRunner`: ``acall``,
    ``bind_job``, ``aclose`` and a sync ``__call__``):

    * Each hop's runner is built LAZILY through ``factory(hop)`` — production
      passes ``make_screening_runner(..., selection=hop)`` — so every hop gets
      a FRESH wire guard and dedicated client: per-send charging, the route
      check and the hop's own (provider, model) tuple check all still apply.
    * Advance ONLY on a failure a sender cannot cause (:func:`_sender_cannot_cause`:
      credential unusable, quota, timeout, 502/503/504/529, a pre-send
      transport error), or on the unevidenced ``("", None)`` signal (serving
      identity is request-local provenance, not something a body controls).
      EVERYTHING ELSE STOPS AND HOLDS (S-1/S-2): any other 4xx (content-filter
      400, 413), any other 5xx, every :class:`_ScreeningStreamRefused`
      (refusal, length, tool use, over-cap, missing end/stop, error, aborted,
      images_unsupported), the guard's own fail-closed refusals, and local
      defects. A sender-triggerable failure must never buy a weaker screener.
      A return carrying a model identity STOPS the walk whatever its text: a
      model HOLD is a verdict, not a failure.
    * Every hop failing re-raises the LAST exception, tagged
      ``tb_chain_exhausted`` when more than one hop ran, so the pipeline
      records exactly ONE breaker failure and ONE ``provider_error`` hold.
      When no hop raised (all returned ``("", None)``) the signal is returned
      as-is, which is the single-hop behaviour this replaced.
    * Cancellation and other ``BaseException`` control flow are never caught.
    """

    def __init__(
        self,
        hops: Sequence[DefaultSelection],
        factory: Callable[[DefaultSelection], Any],
    ) -> None:
        self._hops = list(hops)
        self._factory = factory
        self._built: list[Any] = []
        self._job: AdmissionJob | None = None

    @property
    def hops(self) -> list[DefaultSelection]:
        return list(self._hops)

    def bind_job(self, job: AdmissionJob) -> None:
        """Bind the minted job to every hop built so far and every later one."""

        self._job = job
        for runner in self._built:
            bind = getattr(runner, "bind_job", None)
            if callable(bind):
                bind(job)

    def __call__(
        self,
        prompt: str,
        *,
        default_model: str,
        images: "Sequence[tuple[bytes, str]] | None" = None,
    ) -> tuple[str, str | None]:
        # Same sync-facade rule as ScreenModelRunner: never nest a loop.
        if not self._hops or self._hops[0].default_model != default_model:
            return "", None
        if _running_loop():
            return "", None
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.acall(prompt, default_model=default_model, images=images)
            )
        finally:
            loop.close()

    def _build(self, hop: DefaultSelection) -> Any:
        runner = self._factory(hop)
        if runner is None:
            return None
        self._built.append(runner)
        if self._job is not None:
            bind = getattr(runner, "bind_job", None)
            if callable(bind):
                bind(self._job)
        return runner

    async def acall(
        self,
        prompt: str,
        *,
        default_model: str,
        images: "Sequence[tuple[bytes, str]] | None" = None,
    ) -> tuple[str, str | None]:
        # The pipeline screens under hop 0 (the published default); a chain
        # built for a different default cannot evidence identity.
        if not self._hops or self._hops[0].default_model != default_model:
            return "", None
        last_failure: BaseException | None = None
        attempts = 0
        unevidenced: tuple[str, str | None] = ("", None)
        for hop in self._hops:
            runner = self._build(hop)
            if runner is None:
                continue
            attempts += 1
            try:
                # Each hop is asked under ITS OWN model id: the hop runner's
                # identity gate compares against its own selection. Invoked
                # through the pipeline's own runner contract, so a hop that
                # is a plain synchronous callable (an injected factory) is
                # driven exactly as the pipeline drives a bare runner.
                from local_operator.team_bridge.screening import (
                    _invoke_runner,
                    _maybe_await,
                )

                reply, actual = await _maybe_await(
                    _invoke_runner(
                        runner,
                        prompt,
                        hop.model_id,
                        images=list(images) if images is not None else None,
                    )
                )
            except _ScreeningStreamRefused:
                raise
            except Exception as failure:  # noqa: BLE001 — classified below; control flow propagates
                # A wire failure was classified on its RAW form by the guard
                # before scrubbing; a failure raised outside the guard (the
                # credential cascade, a local defect) is still raw here.
                advance = getattr(failure, _ADVANCE_ATTR, None)
                if advance is None:
                    advance = _sender_cannot_cause(failure)
                if not advance:
                    raise
                last_failure = failure
                continue
            if actual is not None:
                return reply, actual
            unevidenced = (reply, actual)
        if last_failure is None:
            return unevidenced
        if attempts > 1:
            # Typed attribute, not prose: screening_failure_class reads it.
            setattr(last_failure, "tb_chain_exhausted", True)
        raise last_failure

    async def aclose(self) -> None:
        """Close every hop runner that was built (each close is idempotent)."""

        for runner in self._built:
            close = getattr(runner, "aclose", None)
            if callable(close):
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except Exception:  # noqa: BLE001 — teardown, never fatal
                    pass


_GUARDED_CLIENT_CLS: Any = None


def _guarded_screening_client(**kwargs: Any) -> Any:
    """Build the dedicated screening client whose TRANSPORT SELECTION is
    itself validated (R7/Q14).

    ``_transport_for_url`` is the exact synchronous call httpx makes to
    choose the transport for every send — primary or mounted, including
    redirects and retries. Validating there closes the gap where a route
    replaced between the outer stream check and the selection still got a
    POST with no physical charge. The guard is attached as ``_tb_guard``
    once it exists; a client without a guard (test seams) is unchanged.
    """

    import httpx

    global _GUARDED_CLIENT_CLS
    if _GUARDED_CLIENT_CLS is None:

        class _GuardedScreeningClient(httpx.AsyncClient):  # type: ignore[misc]
            def _transport_for_url(self, url: Any) -> Any:
                route = super()._transport_for_url(url)
                guard = getattr(self, "_tb_guard", None)
                if guard is not None and not guard.route_ok(route):
                    raise guard.route_error()
                return route

        _GUARDED_CLIENT_CLS = _GuardedScreeningClient
    return _GUARDED_CLIENT_CLS(**kwargs)


def make_screening_runner(
    *,
    config_dir: Path,
    store: Store,
    owner: OwnerSnapshot,
    message_id: str,
    auth_store: Any,
    settings: Any = None,
    fingerprint: str | None = None,
    expected_epoch: int | None = None,
    selection: "DefaultSelection | None" = None,
) -> "ScreenModelRunner | None":
    """The production builder: canonical default + a fresh GUARDED transport.

    Builds its own dedicated httpx client — narrow read deadline, no
    redirect following, no transport retries — so every physical send is
    visible to the charging proxy parked underneath it, and owns the
    client's lifetime (the runner closes both the client and the stream fn
    in ``finally``). Ordinary ``create_stream_fn`` callers pass neither seam
    and stay byte-identical. The auth store handed in is the caller's normal
    INFERENCE auth, separate from the bridge's pinned person OAuth.
    """

    # ONE canonical capture (R8/Q3): when the caller already resolved the
    # selection (the composition captures provider/model/approved ONCE), the
    # factory MUST use that same object — resolving again here could certify
    # a config the earlier await never saw.
    if selection is None:
        selection = resolve_default_selection(config_dir)
    if selection is None:
        return None
    # LAZY construction (R6/Q10): the dedicated client and stream fn are
    # built on FIRST USE, inside the async path, so a create_stream_fn
    # failure can await the client's close and JOIN the cleanup — a sync
    # constructor has no legal way to await (the old fire-and-forget task
    # was abandoned) and a nested loop raises inside an active loop.
    job_state: dict[str, Any] = {}
    lazy: dict[str, Any] = {"client": None, "stream_fn": None}

    def _guard(client: Any) -> Any:
        guard = ScreeningWireGuard(
            client,
            store=store,
            owner=owner,
            message_id=message_id,
            fingerprint=fingerprint,
            http_client=lazy["client"],
            expected_epoch=expected_epoch,
            job_ref=job_state,
        )
        # Attach to the DEDICATED client so its transport SELECTION is
        # validated at the actual send boundary (R7/Q14).
        if lazy["client"] is not None:
            lazy["client"]._tb_guard = guard
        return guard

    async def _ensure() -> Any:
        if lazy["stream_fn"] is not None:
            return lazy["stream_fn"]
        import httpx

        from local_operator.model.configure import create_stream_fn

        dedicated = _guarded_screening_client(
            transport=httpx.AsyncHTTPTransport(retries=0),
            timeout=httpx.Timeout(_SCREEN_WALL_DEADLINE_S, connect=_SCREEN_CONNECT_TIMEOUT_S),
            follow_redirects=False,
        )
        lazy["client"] = dedicated
        try:
            stream_fn = create_stream_fn(
                auth_store,
                settings,
                session_id="team-bridge-screen",
                wrap_client=_guard,
                http_client=dedicated,
            )
        except BaseException:
            lazy["client"] = None
            await dedicated.aclose()
            raise
        lazy["stream_fn"] = stream_fn
        return stream_fn

    return ScreenModelRunner(
        None,
        selection,
        ensure_fn=_ensure,
        lazy_state=lazy,
        job_ref=job_state,
    )
