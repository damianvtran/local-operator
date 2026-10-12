"""H3 — the production model boundary: default resolution, wire guard, runner.

THREE PIECES, all narrow:

1. :func:`resolve_default_selection` — the CANONICAL configured default
   (``session/cold_model``'s reading of hosting/model_name, model defaulted
   through the provider's own default), built into a real ``ModelSpec`` by
   the same builder a session uses. NOT an active conversation override, not
   a classifier tier, not a cheap substitute: a missing or unusable default
   returns ``None`` and the pipeline holds.
2. :class:`ScreeningWireGuard` — a ``WireClient`` decorator that charges the
   PHYSICAL HTTP bucket (``reserve_http_attempt``) immediately before EVERY
   nested/resend attempt the failover layer makes, and SANITIZES provider
   errors here at the boundary — before shared failover logging can ever see
   vendor prose from a screening request. Only typed status/kind/retry flags
   survive; message text is substituted and exception causes are dropped.
3. :class:`ScreenModelRunner` — one fresh user request per screening: a
   single ``ChatRequest`` with ONE user message, empty tools, ``tool_choice
   "none"``, ``isolated=True``, no system blocks, no history, no cache
   lineage, no callbacks. Text is consumed up to a bound; reasoning is
   discarded; ANY tool-call delta refuses (it is never invoked). The actual
   serving identity comes from the stream's own model events when present;
   an identity the reader cannot evidence holds rather than guesses.
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
    "resolve_default_selection",
]

#: Output ceiling for one screening reply: the frozen grammar is two short
#: lines, and a bounded stream is part of the cost contract.
_SCREEN_MAX_TOKENS = 256
#: Bound on consumed reply text (defence against a runaway stream).
_SCREEN_TEXT_CAP = 4096
#: Fixed text for sanitized wire failures — never vendor prose.
_SANITIZED_WIRE_ERROR = "screening provider call failed"


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
    standard spec builder, so aliases like an explicitly configured ``auto``
    selector keep their normal server-side resolution. The approved identity
    set is derived HERE from that resolution — never from caller strings.
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
        approved = {resolved}
        return DefaultSelection(
            provider=provider,
            model_id=resolved,
            spec=spec,
            approved_identities=frozenset(approved),
        )
    except Exception:  # noqa: BLE001 — an unusable default is a HOLD, not a crash
        return None


class ScreeningWireGuard:
    """Charge every nested wire attempt; scrub vendor prose at the boundary.

    ``store``/``owner``/``message_id``/``fingerprint`` identify the account
    budget the attempts draw from. The guard wraps the REAL client so the
    failover layer's auth re-resolve and fast-mode resend are charged exactly
    like the primary attempt, and a refusal raises BEFORE any I/O.
    """

    def __init__(
        self,
        client: Any,
        *,
        store: Store,
        owner: OwnerSnapshot,
        message_id: str,
        fingerprint: str | None = None,
    ) -> None:
        self._client = client
        self._store = store
        self._owner = owner
        self._message_id = message_id
        self._fingerprint = fingerprint

    def _charge(self) -> None:
        try:
            self._store.reserve_http_attempt(
                self._owner, message_id=self._message_id, fingerprint=self._fingerprint
            )
        except StoreRefused as failure:
            from local_operator.providers.failover import ProviderError

            # Non-retryable by construction: a budget refusal must not be
            # re-attempted by any layer, and carries no vendor content.
            raise ProviderError(
                429,
                "screening HTTP budget exhausted",
                retryable=False,
                kind="quota",
                request_sent=False,
            ) from failure

    async def stream(  # pragma: no cover - thin delegation, driven by tests
        self, request: Any, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        self._charge()
        try:
            # The call itself must be inside the scrub: a client that raises
            # at CALL time (rather than mid-iteration) would otherwise escape
            # with vendor prose intact, exactly what failover's logger reads.
            inner = self._client.stream(request, api_key, oauth_access)
            async for event in inner:
                yield event
        except BaseException as failure:  # noqa: BLE001 - sanitize EVERYTHING here
            from local_operator.providers.failover import ProviderError

            if isinstance(failure, ProviderError):
                # Preserve ONLY typed flags; substitute text; drop the cause
                # chain so no vendor echo can ride ctx/logging (H3 pre-log
                # sanitization happens exactly here, before failover's logger).
                raise ProviderError(
                    failure.status or 0,
                    _SANITIZED_WIRE_ERROR,
                    retryable=False,
                    retry_after_ms=failure.retry_after_ms,
                    auth_error=failure.auth_error,
                    kind=failure.kind,
                    request_sent=failure.request_sent,
                    fail_closed=True,
                ) from None
            raise ProviderError(
                0,
                _SANITIZED_WIRE_ERROR,
                retryable=False,
                request_sent=False,
                fail_closed=True,
            ) from None


def _running_loop() -> bool:
    """Whether THIS thread already has an event loop running (never nest one)."""

    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class ScreenModelRunner:
    """Drive ONE isolated screening request through the normal stream fn.

    The stream function is a real ``create_stream_fn`` product (or any
    equivalent); this class only shapes the request and reads the reply. It
    is synchronous on the outside because the screening pipeline is
    synchronous, and it refuses (typed) rather than nesting event loops when
    called from a running one.
    """

    def __init__(
        self, stream_fn: Callable[..., AsyncIterator[Any]], selection: DefaultSelection
    ) -> None:
        self._stream_fn = stream_fn
        self._selection = selection

    def __call__(self, prompt: str, *, default_model: str) -> tuple[str, str | None]:
        if self._selection.default_model != default_model:
            # The runner was built for a DIFFERENT default than the pipeline
            # is screening under: identity cannot be evidenced, so hold.
            return "", None
        if _running_loop():
            raise RuntimeError(
                "ScreenModelRunner cannot run inside an active event loop; "
                "use the async entry point"
            )
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._drive(prompt))
        finally:
            loop.close()

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
                    # Request-local serving metadata: the spec this request
                    # resolved to. Evidence, not a guess.
                    serving = mid
            elif "tool" in etype and "delta" in etype:
                # A tool call in a screening reply is never invoked and can
                # never ride a verdict.
                raise ValueError("tool-call delta in a screening reply")
            elif etype == "stream_reasoning_delta" or etype == "reasoning_delta":
                continue  # consumed and discarded
            else:
                text = getattr(event, "text", None)
                if isinstance(text, str) and text:
                    consumed += len(text)
                    if consumed > _SCREEN_TEXT_CAP:
                        raise ValueError("screening reply exceeded its bound")
                    parts.append(text)
        actual = serving or self._selection.default_model
        return "".join(parts), actual


def make_screening_runner(
    *,
    config_dir: Path,
    store: Store,
    owner: OwnerSnapshot,
    message_id: str,
    auth_store: Any,
    settings: Any = None,
) -> "ScreenModelRunner | None":
    """The production builder: canonical default + guarded stream fn.

    Ordinary ``create_stream_fn`` callers pass neither seam and are
    unchanged; this path threads the wire guard (physical HTTP budget +
    pre-log sanitization) for exactly this screening, and NEVER touches the
    model-routing key cascade — the auth store handed in is the caller's
    normal inference auth, separate from the bridge's pinned person OAuth.
    """

    selection = resolve_default_selection(config_dir)
    if selection is None:
        return None
    from local_operator.model.configure import create_stream_fn

    def _guard(client: Any) -> Any:
        return ScreeningWireGuard(client, store=store, owner=owner, message_id=message_id)

    stream_fn = create_stream_fn(
        auth_store,
        settings,
        session_id="team-bridge-screen",
        wrap_client=_guard,
    )
    return ScreenModelRunner(stream_fn, selection)
