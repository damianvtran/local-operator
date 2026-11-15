"""PR-D (Root 0): a turn whose owner is away and whose lease has ended waits, quietly.

THE GAP AFTER PR-A (#3439): a held grant bridges an owner's absence only while
the owner's own lease lasts (0-900 s, depending on the grant's age). Past it the
turn failed about half a second later with "owner offline", even if the owner
was a lid-open away. Under PR-D the SAME turn keeps its place, before any byte is
sent, for up to ``grant_ttl_s / 2``. It watches the one durable fact the node's
relay writes when the owner's link is admitted (``member.last_seen_at``), dials
nothing while it waits, and then runs the driver's existing single loop-back
sweep: served once if the owner came back, or failed with a sentence that says
how long it waited.

What must NOT happen is pinned as hard as what must: no extra asks (the same two
as before, only spread out), no grant life extended, no IdP POST, no new target,
no wait for a process the owner never leased, for an owner ANSWER, for MCP,
GitHub, device-bound keys or isolated calls, and none in front of a configured
fallback.

Real path: two relays on loopback (the repo's ``mesh`` fixture: real pairing, the
real ``credential share``/``revoke`` CLI, a loopback IdP counting POSTs), the real
``MeshCredentialClient``/``MeshAwareAuthStore``, and ``stream_with_failover`` with
a real ``OpenAICompatClient`` talking to an httpx stub provider. The owner is
stopped and brought back as a new relay with the same identity, root and port.
Synthetic ids only.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import httpx
import pytest
import yaml

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
)
from local_operator.network import credentials as credentials_pkg
from local_operator.network.credentials import effective
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.store import MeshAwareAuthStore
from local_operator.network.credentials.types import BrokerError, Grant
from local_operator.providers.failover import (
    PREWIRE_REFUSAL_ATTR,
    FailoverRouteState,
    FallbackTarget,
    ProviderError,
    stream_with_failover,
)
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credential_limit_skip import _NoLocalLogins
from tests.unit.network.test_credentials_held_offline import _restart_owner, _stop_owner
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    _lop_network,
    _pull,
    _share,
    mesh,
)
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixture by import

SESSION = "sess-1"
MODEL = "claude-opus-5-5"
#: No configured fallback: the walk is the primary alone, then the one loop-back.
SETTINGS: dict[str, Any] = {"retry": {"baseDelayMs": 1, "maxRetries": 0, "fallbackChains": {}}}


class _Provider:
    """An OpenAI-compatible endpoint that serves every bearer and records each one."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request.headers.get("authorization", "").removeprefix("Bearer "))
        chunk = {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
        return httpx.Response(
            200,
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(),
            headers={"content-type": "text/event-stream"},
        )


class _KeylessLimitedOnce(_Provider):
    """As :class:`_Provider`, but the first KEYLESS request gets 429 ``retry-after``."""

    def __init__(self, retry_after_s: int) -> None:
        super().__init__()
        self.retry_after_s = retry_after_s
        self.served_at: list[tuple[float, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
        if not bearer and not any(not seen for seen in self.seen):
            self.seen.append(bearer)
            return httpx.Response(
                429,
                headers={"retry-after": str(self.retry_after_s)},
                json={"error": {"message": "rate limited (stub)", "type": "rate_limit_error"}},
            )
        self.served_at.append((time.monotonic(), bearer))
        return super().__call__(request)


class _OpenAIKey(_NoLocalLogins):
    """A local login for ``openai`` only; every other provider is borrowed or absent."""

    async def get_api_key(self, *args: Any, **_k: Any) -> Any:
        return "local-fallback-key" if args and args[0] == "openai" else None


async def _turn(
    auth: Any,
    provider: _Provider,
    *,
    signal: AbortSignal | None = None,
    settings: dict[str, Any] | None = None,
    model: ModelSpec | None = None,
    route_state: FailoverRouteState | None = None,
    isolated: bool = False,
) -> list[Any]:
    from local_operator.providers.clients import OpenAICompatClient

    request = ChatRequest(
        model=model or ModelSpec(provider=STUB_PROVIDER, model_id=MODEL), isolated=isolated
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:

        async def client_for(spec: ModelSpec) -> Any:
            return OpenAICompatClient("https://stub.invalid/v1", http_client=http)

        return [
            event
            async for event in stream_with_failover(
                request,
                auth,
                settings or SETTINGS,
                client_for,
                session_id=SESSION,
                signal=signal,
                route_state=route_state,
            )
        ]


class _Asks:
    """Every credential ask THIS session makes of its client, with its time."""

    def __init__(self, client: MeshCredentialClient, monkeypatch: pytest.MonkeyPatch) -> None:
        self.at: list[float] = []
        original = client.request_grant_sync

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.at.append(time.monotonic())
            return original(*args, **kwargs)

        monkeypatch.setattr(client, "request_grant_sync", counted)


def _bound(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    # ``raising=False``: on a tree without PR-D the name does not exist, and the
    # cell must fail on BEHAVIOUR there, not on its own set-up.
    monkeypatch.setattr(credentials_pkg, "owner_wait_s", lambda root=None: seconds, raising=False)


def _lapsed(mesh: Any) -> tuple[MeshCredentialClient, MeshAwareAuthStore, Grant]:  # noqa: F811
    """Borrow once through the store, stop the owner, and let the lease AND hold run out.

    PR #3922 added the owner-unreachable hold (default 1 h): a held grant is served
    while the owner is away, so this wait only engages once that hold has ended.
    The borrower's own ``offline_hold_s`` caps the hold and is clamped up to
    ``grant_ttl_s``, so pinning it to the TTL makes "lease ran out" and "hold ran
    out" the same moment again (measured from receipt, hence the second wait term).
    """
    (mesh.b.root / "config.yml").write_text(
        yaml.safe_dump(
            {"values": {"network": {"credentials": {"grant_ttl_s": 2.0, "offline_hold_s": 2.0}}}}
        ),
        encoding="utf-8",
    )
    effective._receipts.clear()
    _share(mesh)
    _pull(mesh)
    client = MeshCredentialClient.for_this_device(mesh.b.root)
    assert client is not None, "the share never reached the borrower"
    store = MeshAwareAuthStore(_NoLocalLogins(), mesh=client, config_dir=mesh.b.root)
    grant = client.request_grant_sync(STUB_PROVIDER, session_id=SESSION)
    assert isinstance(grant, Grant), grant
    effective.note_resolution(mesh.b.root, STUB_PROVIDER, grant)
    _stop_owner(mesh)
    hold_end_ms = max(grant.grant_expires_at_ms, int(grant.received_wall * 1000 + 2000))
    assert net_fixtures.wait_for(lambda: time.time() * 1000 > hold_end_ms + 50)
    return client, store, grant


def _ended(events: list[Any]) -> bool:
    return any(isinstance(event, StreamEndEvent) for event in events)


# ---------------------------------------------------------------------------
# W: behaviour PR-D adds (each fails on a tree without it)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_owner_returns_mid_wait_and_the_same_turn_is_served_once(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """W1: the owner comes back inside the bound; ONE request reaches the provider."""
    client, store, grant = _lapsed(mesh)
    _bound(monkeypatch, 30.0)
    provider = _Provider()
    task = asyncio.create_task(_turn(store, provider))
    await asyncio.sleep(1.0)
    assert not task.done(), "the turn did not wait for its owner"
    assert provider.seen == [], "a request left before the owner was back"
    owner = _restart_owner(mesh, monkeypatch)
    try:
        returned_at = time.monotonic()
        events = await asyncio.wait_for(task, timeout=30.0)
        served_in = time.monotonic() - returned_at
        assert _ended(events), events
        assert len(provider.seen) == 1, provider.seen
        fresh = client.grants.peek(STUB_PROVIDER, SESSION)
        assert fresh is not None and fresh.granted_at > grant.granted_at
        assert provider.seen == [fresh.access_token], "not the fresh grant's bearer"
        assert effective.effective_access(STUB_PROVIDER, root=mesh.b.root, mesh=client).state == (
            "effective_borrow"
        )
        print(f"W1: served {served_in:.2f}s after the owner came back")
    finally:
        owner.stop()


@pytest.mark.asyncio
async def test_the_bound_expires_and_the_turn_names_the_wait_and_the_remedy(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """W2: the owner stays away; the turn fails at the bound, saying it waited."""
    _client, store, _grant = _lapsed(mesh)
    _bound(monkeypatch, 1.5)
    provider = _Provider()
    started = time.monotonic()
    with pytest.raises(ProviderError) as failed:
        await asyncio.wait_for(_turn(store, provider), timeout=20.0)
    elapsed = time.monotonic() - started
    error = failed.value
    message = str(error)
    assert error.retryable is True and error.kind == "unknown", (error.retryable, error.kind)
    assert getattr(error, PREWIRE_REFUSAL_ATTR, False) is True, "nothing was sent; say so"
    assert "waited 1 s" in message, message
    assert "Bring" in message and f"lop login {STUB_PROVIDER}" in message, message
    assert elapsed >= 1.5, elapsed
    assert provider.seen == []


@pytest.mark.asyncio
async def test_the_wait_dials_nothing_between_the_walk_and_the_sweep(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """W3: two asks, the bound apart; no IdP POST, no grant, no lease moved."""
    client, store, _grant = _lapsed(mesh)
    _bound(monkeypatch, 3.0)
    posts = len(mesh.idp.posts)
    lease = effective.effective_access(
        STUB_PROVIDER, root=mesh.b.root, mesh=client
    ).last_grant_expires_at
    asks = _Asks(client, monkeypatch)
    held: list[Any] = []
    task = asyncio.create_task(_turn(store, _Provider()))
    while not task.done():
        # From the walk's first ask on: before it, the cache still holds the LAPSED
        # grant, which that ask drops (``_held_or``); the wait must not put one back.
        if asks.at:
            held.append(client.grants.peek(STUB_PROVIDER, SESSION))
        await asyncio.sleep(0.5)
    with pytest.raises(ProviderError):
        await task
    # The walk and the sweep each make the SAME asks they make today (a resolve
    # asks once per credential shape it tries); the wait makes none between them.
    gaps = [later - earlier for earlier, later in zip(asks.at, asks.at[1:])]
    assert gaps and max(gaps) >= 3.0, asks.at
    split = gaps.index(max(gaps)) + 1
    assert all(gap < 1.0 for gap in gaps[: split - 1] + gaps[split:]), asks.at
    assert split * 2 == len(asks.at), f"the sweep asked {len(asks.at) - split}, the walk {split}"
    assert len(mesh.idp.posts) == posts
    assert held and all(value is None for value in held), "the wait put a grant in the cache"
    after = effective.effective_access(STUB_PROVIDER, root=mesh.b.root, mesh=client)
    assert after.last_grant_expires_at == lease, "the lease moved"


@pytest.mark.asyncio
async def test_user_abort_ends_the_wait_promptly_without_another_dial(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """W4: a stop is a stop: ``aborted`` at once, and no sweep ask after it."""
    client, store, _grant = _lapsed(mesh)
    _bound(monkeypatch, 30.0)
    asks = _Asks(client, monkeypatch)
    signal = AbortSignal()
    task = asyncio.create_task(_turn(store, _Provider(), signal=signal))
    await asyncio.sleep(0.5)
    walked = len(asks.at)
    fired = time.monotonic()
    signal.abort("user stop")
    with pytest.raises(ProviderError) as stopped:
        await asyncio.wait_for(task, timeout=5.0)
    assert stopped.value.kind == "aborted", stopped.value
    assert time.monotonic() - fired < 1.5
    assert walked and len(asks.at) == walked, f"asked after the stop: {asks.at}"


@pytest.mark.asyncio
async def test_cancelling_the_turn_task_reaps_the_wait(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """W5: a cancelled turn (session close, move teardown) leaves nothing polling."""
    from local_operator.network.credentials import store as store_mod

    client, store, _grant = _lapsed(mesh)
    _bound(monkeypatch, 30.0)
    asks = _Asks(client, monkeypatch)
    polls: list[float] = []
    poll = getattr(store_mod, "_OwnerWait", None)
    if poll is not None:
        original = poll.poll

        def counted(self: Any) -> str:
            polls.append(time.monotonic())
            return original(self)

        monkeypatch.setattr(poll, "poll", counted)
    task = asyncio.create_task(_turn(store, _Provider()))
    await asyncio.sleep(0.5)
    walked = len(asks.at)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    settled = len(polls)
    await asyncio.sleep(2.0)
    assert len(polls) == settled, "a poll kept running after the turn was cancelled"
    assert walked and len(asks.at) == walked, f"asked after the cancel: {asks.at}"


@pytest.mark.asyncio
async def test_band_holds_owner_unavailable_while_waiting_and_clears_on_return(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """W6: the band says what is true while the turn waits, and clears when served."""
    from concurrent.futures import ThreadPoolExecutor

    from local_operator.session.runtime import serving
    from local_operator.session.runtime.serving import ServingSessionHandle
    from tests.unit.network.test_owner_return_clears_access_band import _band, _settle
    from tests.unit.session.runtime.test_serving import _AccessSession

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fixture-quiet-wait-band")
    monkeypatch.setattr(serving, "_switch_executor_pool", pool)
    _client, store, _grant = _lapsed(mesh)
    _bound(monkeypatch, 30.0)
    session = _AccessSession(f"{STUB_PROVIDER}/stub-model")
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(mesh.b.root), config_dir=mesh.b.root
    )
    owner = None
    try:
        task = asyncio.create_task(_turn(store, _Provider()))
        await asyncio.sleep(1.0)
        await _settle(handle)
        first = _band(session)
        await asyncio.sleep(0.6)
        await _settle(handle)
        second = _band(session)
        assert not task.done(), "the turn did not wait"
        for band in (first, second):
            assert (band.state, band.effective_state) == (
                "temporarily_unavailable",
                "owner_unavailable",
            ), band
        assert first.observed_at == second.observed_at, "the wait wrote a receipt"
        owner = _restart_owner(mesh, monkeypatch)
        assert _ended(await asyncio.wait_for(task, timeout=30.0))
        await _settle(handle)
        band = _band(session)
        assert (band.state, band.effective_state) == ("ok", "effective_borrow"), band
    finally:
        if owner is not None:
            owner.stop()
        await handle.dispose()
        pool.shutdown(wait=True, cancel_futures=True)


def test_hint_is_the_owners_link_admission(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """W7: the hint moves only when the owner's link is admitted again."""
    client, _store, grant = _lapsed(mesh)
    owner_id = client.owner_of(STUB_PROVIDER)
    before = client.owner_seen_at(owner_id)
    time.sleep(3.0)
    assert client.owner_seen_at(owner_id) == before, "the hint moved with the owner away"
    owner = _restart_owner(mesh, monkeypatch)
    try:
        assert net_fixtures.wait_for(
            lambda: (client.owner_seen_at(owner_id) or 0.0) > (before or 0.0), timeout_s=10.0
        ), "the owner's admission never moved the hint"
        # ``owner_last_seen_s`` still counts a served grant as a sighting (Q6).
        last = client.owner_last_seen_s(owner_id)
        assert last is not None and last <= time.time() - grant.granted_at + 1.0
    finally:
        owner.stop()


@pytest.mark.asyncio
async def test_an_abort_during_a_poll_is_a_stop_even_if_the_owner_returned(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Review R1-4: a stop that lands while a poll runs ends the turn as a stop.

    The poll reports ``returned`` AFTER the stop fired; the sweep must not ask.
    """
    from local_operator.network.credentials import store as store_mod

    client, store, _grant = _lapsed(mesh)
    _bound(monkeypatch, 30.0)
    asks = _Asks(client, monkeypatch)
    signal = AbortSignal()

    def stop_then_return(self: Any) -> str:
        signal.abort("user stop")
        return "returned"

    monkeypatch.setattr(store_mod._OwnerWait, "poll", stop_then_return, raising=False)
    with pytest.raises(ProviderError) as stopped:
        await asyncio.wait_for(_turn(store, _Provider(), signal=signal), timeout=10.0)
    assert stopped.value.kind == "aborted", stopped.value
    walked = len(asks.at)
    assert walked and walked % 2 == 0 and walked == 2, f"asked after the stop: {asks.at}"


def test_the_waited_time_is_never_overstated() -> None:
    """NIT: the default bound reads as what it is, 7 min 30 s, floored everywhere."""
    from local_operator.network.credentials import messages

    render = getattr(messages, "render_waited")
    assert render(1.5) == "1 s"
    assert render(59.9) == "59 s"
    assert render(120.0) == "2 min"
    assert render(450.0) == "7 min 30 s"
    assert render(449.9) == "7 min 29 s"


# ---------------------------------------------------------------------------
# Review R1-1: the wait holds back ONLY the absent owner's own targets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_rate_limited_fallback_is_retried_on_its_own_clock_not_the_owners(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """R1-1 shape 1 (QA row 7, ``r7b``): the fallback's ``retry-after`` decides the sweep.

    The primary is the borrowed key (owner away, lease over). The configured fallback
    is a keyless provider answering 429 ``retry-after: 3`` once. The loop-back must
    sleep its 3 s and serve through the fallback, not wait the owner's 30 s bound.
    """
    from local_operator.providers import registry

    monkeypatch.setitem(
        registry._BY_ID,  # noqa: SLF001 — the seam the repo's stub_provider fixture uses
        "keylessfb",
        registry.ProviderDefinition(id="keylessfb", name="Keyless", allows_missing_api_key=True),
    )
    _client, store, _grant = _lapsed(mesh)
    _bound(monkeypatch, 30.0)
    settings = {
        "retry": {
            "baseDelayMs": 1,
            "maxRetries": 0,
            "fallbackChains": {
                f"{STUB_PROVIDER}/{MODEL}": [{"provider": "keylessfb", "model": "fb-1"}]
            },
        }
    }
    provider = _KeylessLimitedOnce(retry_after_s=3)
    started = time.monotonic()
    events = await asyncio.wait_for(_turn(store, provider, settings=settings), timeout=60.0)
    assert _ended(events), events
    served_at = provider.served_at[0][0] - started
    assert 2.5 <= served_at < 3.0 + 2.0, f"served at {served_at:.2f}s, not on the fallback's 3 s"
    assert [bearer for _at, bearer in provider.served_at] == [""], provider.served_at


@pytest.mark.asyncio
async def test_a_pin_excluded_primary_is_not_held_behind_the_absent_owner(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """R1-1 shape 2 (QA row 7', ``r7_pinned``): an unwalked target is asked at once.

    The session is pinned to the borrowed fallback, so the walk skips the primary (a
    local key that would serve). The borrowed target is owner-away; the sweep must
    still reach the primary on today's backoff, not after the owner's bound.
    """
    _client, _store, _grant = _lapsed(mesh)
    client = MeshCredentialClient.for_this_device(mesh.b.root)
    assert client is not None
    store = MeshAwareAuthStore(_OpenAIKey(), mesh=client, config_dir=mesh.b.root)
    _bound(monkeypatch, 30.0)
    settings = {
        "retry": {
            "baseDelayMs": 1,
            "maxRetries": 0,
            "fallbackChains": {"openai/gpt-5": [{"provider": STUB_PROVIDER, "model": MODEL}]},
        }
    }
    route = FailoverRouteState(
        active=FallbackTarget(f"{STUB_PROVIDER}/{MODEL}"),
        primary_retry_at_ms=int(time.time() * 1000) + 600_000,
    )
    provider = _Provider()
    started = time.monotonic()
    events = await asyncio.wait_for(
        _turn(
            store,
            provider,
            settings=settings,
            model=ModelSpec(provider="openai", model_id="gpt-5"),
            route_state=route,
        ),
        timeout=60.0,
    )
    assert _ended(events), events
    assert time.monotonic() - started < 3.0, "the primary waited for another device's owner"
    assert provider.seen == ["local-fallback-key"], provider.seen


@pytest.mark.asyncio
async def test_a_mixed_revisit_still_serves_an_owner_that_came_back(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """R1-1 guard: the mixed case keeps the owner-return path. The owner comes back
    during the fallback's own backoff, and the sweep (primary first) is served ONCE
    by the owner's fresh grant."""
    from local_operator.providers import registry

    monkeypatch.setitem(
        registry._BY_ID,  # noqa: SLF001 — the seam the repo's stub_provider fixture uses
        "keyless",
        registry.ProviderDefinition(id="keyless", name="Keyless", allows_missing_api_key=True),
    )
    client, store, grant = _lapsed(mesh)
    _bound(monkeypatch, 30.0)
    settings = {
        "retry": {
            "baseDelayMs": 1,
            "maxRetries": 0,
            "fallbackChains": {f"{STUB_PROVIDER}/{MODEL}": [{"provider": "keyless", "model": "k"}]},
        }
    }
    provider = _KeylessLimitedOnce(retry_after_s=3)
    task = asyncio.create_task(_turn(store, provider, settings=settings))
    await asyncio.sleep(0.5)
    owner = _restart_owner(mesh, monkeypatch)
    try:
        events = await asyncio.wait_for(task, timeout=30.0)
        assert _ended(events), events
        fresh = client.grants.peek(STUB_PROVIDER, SESSION)
        assert fresh is not None and fresh.granted_at > grant.granted_at
        assert [bearer for _at, bearer in provider.served_at] == [fresh.access_token]
    finally:
        owner.stop()


# ---------------------------------------------------------------------------
# N: over-reach guards (pass on both trees)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_process_never_leased_by_the_owner_fails_fast(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """N1: no lease from this owner in this process -> today's sentence, at once."""
    _client, store, _grant = _lapsed(mesh)
    effective._receipts.clear()
    _bound(monkeypatch, 30.0)
    started = time.monotonic()
    with pytest.raises(ProviderError) as failed:
        await asyncio.wait_for(_turn(store, _Provider()), timeout=10.0)
    assert time.monotonic() - started < 2.0
    assert "waited" not in str(failed.value), failed.value
    hint = getattr(store, "owner_wait_hint", None)
    assert hint is None or hint(STUB_PROVIDER) is None


@pytest.mark.asyncio
async def test_an_owner_answer_never_waits(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """N2: the owner is up and says no: that answer is the outcome, at once."""
    effective._receipts.clear()
    _share(mesh)
    _pull(mesh)
    client = MeshCredentialClient.for_this_device(mesh.b.root)
    assert client is not None
    store = MeshAwareAuthStore(_NoLocalLogins(), mesh=client, config_dir=mesh.b.root)
    assert _ended(await _turn(store, _Provider()))
    # The owner's real CLI, as the held-offline cells revoke (it acts on the owner's
    # own config through the fixture's environment).
    assert _lop_network("credential", "revoke", STUB_PROVIDER, "--from", mesh.b.identity.name) == 0
    _bound(monkeypatch, 30.0)
    started = time.monotonic()
    with pytest.raises(ProviderError):
        await asyncio.wait_for(_turn(store, _Provider()), timeout=10.0)
    assert time.monotonic() - started < 2.0
    hint = getattr(store, "owner_wait_hint", None)
    assert hint is None or hint(STUB_PROVIDER) is None


@pytest.mark.asyncio
async def test_held_inside_the_lease_is_served_without_waiting(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """N3: PR-A unchanged: inside the lease the held grant serves, no wait."""
    effective._receipts.clear()
    _share(mesh)
    _pull(mesh)
    client = MeshCredentialClient.for_this_device(mesh.b.root)
    assert client is not None
    store = MeshAwareAuthStore(_NoLocalLogins(), mesh=client, config_dir=mesh.b.root)
    assert _ended(await _turn(store, _Provider()))
    _stop_owner(mesh)
    _bound(monkeypatch, 30.0)
    started = time.monotonic()
    provider = _Provider()
    assert _ended(await asyncio.wait_for(_turn(store, provider), timeout=10.0))
    assert time.monotonic() - started < 2.0
    assert len(provider.seen) == 1


@pytest.mark.asyncio
async def test_families_and_isolated_calls_never_wait(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """N4 (review R1-2): device-bound, GitHub and MCP keys never wait; nor does isolation.

    Each family key is given EVERY other condition the real lapsed provider key
    meets: a heard ``owner_offline`` (T1), the same holder answer and the same
    lapsed ``owner_unavailable`` receipt (T3-T5). T2 is then the only condition that
    can refuse, so this cell fails if T2 is removed. The real key is the control.
    """
    client, store, _grant = _lapsed(mesh)
    _bound(monkeypatch, 30.0)
    owner = client.owner_of(STUB_PROVIDER)
    # The silence is HEARD once (as a turn's first ask would), which is what turns
    # the lapsed borrow receipt into ``owner_unavailable`` with an expired lease.
    heard = client.request_grant_sync(STUB_PROVIDER, session_id=SESSION)
    assert isinstance(heard, BrokerError) and heard.code == "owner_offline", heard
    lapsed = effective.effective_access(STUB_PROVIDER, root=mesh.b.root, mesh=client)
    assert (lapsed.state, lapsed.owner) == ("owner_unavailable", owner), lapsed
    family = ("kimi", "github", "mcp:https://mcp.example.test/sse")
    for key in (STUB_PROVIDER, *family):
        store._refusals[key] = BrokerError(code="owner_offline", key=key, owner_device=owner)
    monkeypatch.setattr(client, "should_borrow", lambda key: True)
    monkeypatch.setattr(effective, "effective_access", lambda *a, **k: lapsed)
    hint = getattr(store, "owner_wait_hint", None)
    if hint is not None:
        assert hint(STUB_PROVIDER) is not None, "control: the forced conditions must admit"
        for key in family:
            assert hint(key) is None, key
    monkeypatch.undo()
    _bound(monkeypatch, 30.0)
    # Isolation (decorative calls: naming, summaries) disables retry, so no wait.
    started = time.monotonic()
    with pytest.raises(ProviderError):
        await asyncio.wait_for(_turn(store, _Provider(), isolated=True), timeout=10.0)
    assert time.monotonic() - started < 2.0


@pytest.mark.asyncio
async def test_a_configured_fallback_still_serves_without_a_wait(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """N5: a fallback the operator configured is walked first, exactly as today."""

    class _FallbackKey(_NoLocalLogins):
        async def get_api_key(self, *args: Any, **_k: Any) -> Any:
            return "local-fallback-key" if args and args[0] == "openai" else None

    _client, _store, _grant = _lapsed(mesh)
    client = MeshCredentialClient.for_this_device(mesh.b.root)
    assert client is not None
    store = MeshAwareAuthStore(_FallbackKey(), mesh=client, config_dir=mesh.b.root)
    _bound(monkeypatch, 30.0)
    settings = {
        "retry": {
            "baseDelayMs": 1,
            "maxRetries": 0,
            "fallbackChains": {f"{STUB_PROVIDER}/{MODEL}": ["openai/gpt-5"]},
        }
    }
    provider = _Provider()
    started = time.monotonic()
    events = await asyncio.wait_for(_turn(store, provider, settings=settings), timeout=10.0)
    assert _ended(events), events
    assert time.monotonic() - started < 2.0
    assert provider.seen == ["local-fallback-key"], provider.seen


@pytest.mark.asyncio
async def test_a_policy_refusal_heard_elsewhere_blocks_the_wait(
    mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """N6: a durable non-transport refusal (another process heard it) means no wait."""
    client, store, _grant = _lapsed(mesh)
    other = MeshCredentialClient.for_this_device(mesh.b.root)
    assert other is not None
    other.state.observe(STUB_PROVIDER, "consent_withdrawn", retry_after_ms=300_000)
    other.state.save()
    _bound(monkeypatch, 30.0)
    started = time.monotonic()
    with pytest.raises(ProviderError):
        await asyncio.wait_for(_turn(store, _Provider()), timeout=10.0)
    assert time.monotonic() - started < 2.0
    hint = getattr(store, "owner_wait_hint", None)
    assert hint is None or hint(STUB_PROVIDER) is None


def test_no_thread_is_left_behind_by_the_module(mesh: Any) -> None:  # noqa: F811
    """Importing and constructing never starts a thread of its own."""
    before = {thread.ident for thread in threading.enumerate()}
    MeshAwareAuthStore(_NoLocalLogins(), mesh=None, config_dir=mesh.b.root)
    assert {thread.ident for thread in threading.enumerate()} - before == set()
