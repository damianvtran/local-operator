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
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Sequence

from local_operator.team_bridge.store import Store, StoreRefused
from local_operator.team_bridge.types import OwnerSnapshot

__all__ = [
    "DefaultSelection",
    "ScreenModelRunner",
    "ScreeningWireGuard",
    "make_screening_runner",
    "resolve_default_selection",
]

#: Output ceiling for one screening reply: the frozen grammar is two short
#: lines, and a bounded stream is part of the cost contract.
_SCREEN_MAX_TOKENS = 256
#: Bound on consumed reply text (defence against a runaway stream).
_SCREEN_TEXT_CAP = 4096
#: The screening wall deadline, per the frozen contract — NOT the ordinary
#: session HTTP read timeout (600 s) and not max_tokens alone.
_SCREEN_WALL_DEADLINE_S = 45.0
_SCREEN_CONNECT_TIMEOUT_S = 10.0
#: Fixed text for sanitized wire failures — never vendor prose.
_SANITIZED_WIRE_ERROR = "screening provider call failed"


class _ScreeningStreamRefused(Exception):
    """A typed refusal inside a screening stream (never carries prose)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


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

        if images and not bool(getattr(self._selection.spec, "supports_images", False)):
            # D5, fail closed: a screener that cannot see images must never be
            # sent them (a provider would strip or 400 them and the verdict
            # would then cover pixels nobody looked at). The connector screens
            # text-only for such a model; reaching here means the selection
            # changed under it, which is a hold, never a text-only allow.
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
