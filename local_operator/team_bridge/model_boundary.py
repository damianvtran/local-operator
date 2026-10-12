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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

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


@dataclass
class DefaultSelection:
    """The captured canonical default, with its approved identity set."""

    provider: str
    model_id: str
    spec: Any  # a real ModelSpec; Any keeps this module import-light
    approved_identities: frozenset[str] = field(default_factory=frozenset)

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

    def __init__(self, inner: Any, charge: Callable[[], None]) -> None:
        self._inner = inner
        self._charge = charge

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
    ) -> None:
        self._client = client
        self._store = store
        self._owner = owner
        self._message_id = message_id
        self._fingerprint = fingerprint
        self._transport_patched = False
        http = http_client if http_client is not None else getattr(client, "_http", None)
        if http is not None:
            transport = getattr(http, "_transport", None)
            if transport is not None and not isinstance(transport, _ChargingAsyncTransport):
                http._transport = _ChargingAsyncTransport(transport, self._charge)
                self._transport_patched = True

    def _charge(self) -> None:
        try:
            self._store.reserve_http_attempt(
                self._owner, message_id=self._message_id, fingerprint=self._fingerprint
            )
        except StoreRefused:
            from local_operator.providers.failover import ProviderError

            refusal = ProviderError(
                429,
                "screening HTTP budget exhausted",
                retryable=False,
                kind="quota",
                request_sent=False,  # provably before any dispatch
            )
            refusal.__cause__ = None
            refusal.__context__ = None
            raise refusal from None

    async def stream(  # pragma: no cover - thin delegation, driven by tests
        self, request: Any, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        if not self._transport_patched:
            # No reachable transport: this charge may UNDER-COUNT nested
            # sends. The factory always patches; this branch is the
            # documented best-effort for bare wire objects.
            self._charge()
        pending: BaseException | None = None
        try:
            # The CALL itself sits inside the scrub: a client that raises at
            # call time must not escape with vendor prose intact.
            inner = self._client.stream(request, api_key, oauth_access)
            async for event in inner:
                yield event
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
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
        stream_fn: Callable[..., AsyncIterator[Any]],
        selection: DefaultSelection,
        *,
        deadline_s: float = _SCREEN_WALL_DEADLINE_S,
        owned_http: Any | None = None,
        owned_stream_fn: Any | None = None,
    ) -> None:
        self._stream_fn = stream_fn
        self._selection = selection
        self._deadline_s = deadline_s
        self._owned_http = owned_http
        self._owned_stream_fn = owned_stream_fn

    def __call__(self, prompt: str, *, default_model: str) -> tuple[str, str | None]:
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
            return loop.run_until_complete(self.acall(prompt, default_model=default_model))
        finally:
            loop.close()

    async def acall(self, prompt: str, *, default_model: str) -> tuple[str, str | None]:
        """The PUBLIC async entry: bounded wall time, owned cleanup in finally."""

        if self._selection.default_model != default_model:
            return "", None
        try:
            return await asyncio.wait_for(self._drive(prompt), timeout=self._deadline_s)
        finally:
            await self._close_owned()

    async def aclose(self) -> None:
        """Close whatever this runner owns (idempotent)."""

        await self._close_owned()

    async def _close_owned(self) -> None:
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

    async def _drive(self, prompt: str) -> tuple[str, str | None]:
        from local_operator.harness.types import ChatRequest, Message, TextContent

        request = ChatRequest(
            model=self._selection.spec,
            # ONE fresh user turn: no system blocks, no history, no tools,
            # no cache lineage, no callbacks — the isolation contract.
            messages=[Message(role="user", content=[TextContent(text=prompt)])],
            tools=[],
            tool_choice="none",
            isolated=True,
            purpose="team_bridge_screen",
            max_tokens=_SCREEN_MAX_TOKENS,
        )
        parts: list[str] = []
        consumed = 0
        serving: str | None = None
        async for event in self._stream_fn(request, None):
            etype = str(getattr(event, "type", "") or "")
            if etype == "model":
                model = getattr(event, "model", None)
                mid = getattr(model, "model_id", None)
                if isinstance(mid, str) and mid:
                    # Request-local serving metadata, from the normal failover
                    # path — evidence, never a configured-selection guess.
                    serving = mid
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
                if stop_reason not in ("stop", ""):
                    # length / toolUse / refusal / error / aborted can NEVER
                    # admit (the typed terminal contract).
                    raise _ScreeningStreamRefused(stop_reason)
            elif "tool" in etype:
                # A tool call in a screening reply is never invoked and can
                # never ride a verdict.
                raise _ScreeningStreamRefused("tool_use")
        return "".join(parts), serving


def make_screening_runner(
    *,
    config_dir: Path,
    store: Store,
    owner: OwnerSnapshot,
    message_id: str,
    auth_store: Any,
    settings: Any = None,
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

    selection = resolve_default_selection(config_dir)
    if selection is None:
        return None
    import httpx

    from local_operator.model.configure import create_stream_fn

    dedicated = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(retries=0),
        timeout=httpx.Timeout(_SCREEN_WALL_DEADLINE_S, connect=_SCREEN_CONNECT_TIMEOUT_S),
        follow_redirects=False,
    )

    def _guard(client: Any) -> Any:
        return ScreeningWireGuard(
            client,
            store=store,
            owner=owner,
            message_id=message_id,
            http_client=dedicated,
        )

    try:
        stream_fn = create_stream_fn(
            auth_store,
            settings,
            session_id="team-bridge-screen",
            wrap_client=_guard,
            http_client=dedicated,
        )
    except BaseException:
        import asyncio as _asyncio

        _asyncio.new_event_loop().run_until_complete(dedicated.aclose())
        raise
    return ScreenModelRunner(
        stream_fn,
        selection,
        owned_http=dedicated,
        owned_stream_fn=stream_fn,
    )
