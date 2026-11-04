"""A 401 after a detached early refresh waits for THAT exchange to persist (#3337).

An issuer that revokes the old access token at rotation answers the request
sent with it after the early-wait budget with a 401, while the detached
exchange still owns the refresh lock and is still reading the rotation. The 401
recovery used to wait only ``LOCK_ACQUIRE_TIMEOUT_S`` for that lock; when the
token endpoint was slower than that, the 401 reached the SDK's authorization
branch on a healthy grant. These tests drive the real SDK/httpx2 flow against a
real loopback issuer and resource, with the real cross-process flock, and
shrink only ``LOCK_ACQUIRE_TIMEOUT_S`` so the "slower than the lock wait" case
costs seconds instead of the production 15.

"Reached the SDK's authorization branch" is observed directly: the SDK's
metadata-discovery request builder is replaced with one that records the call
and raises, so a test can tell it apart from every other failure.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import time
from typing import Any

import pytest
from mcp.client.auth import oauth2 as sdk_oauth2
from mcp.client.streamable_http import create_mcp_http_client

from local_operator.mcp import auth as auth_mod
from tests.unit.mcp.rotating_oauth_server import FakeTokenEndpoint
from tests.unit.mcp.test_auth import FakeAuthStore
from tests.unit.mcp.test_oauth_inflight_refresh_lead import _connected_provider
from tests.unit.mcp.test_oauth_refresh_durability import (  # noqa: F401
    SERVER_URL,
    _cfg,
    _endpoints_for,
    _isolated_lock_dir,
)

#: The shrunk ordinary lock wait. Every "slow" endpoint below answers after
#: SLOW_S, past this bound and well inside the exchange's own persist deadline
#: (``REFRESH_HTTP_TIMEOUT_S + REFRESH_LATE_RESPONSE_GRACE_S``, unchanged).
LOCK_S = 1.0
SLOW_S = 1.8


class _AuthorizationBranchEntered(Exception):
    """Raised where the SDK starts its full authorization (metadata discovery)."""


@pytest.fixture
def auth_branch(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    entered: list[str] = []

    def _record(url: str) -> Any:
        entered.append(url)
        raise _AuthorizationBranchEntered(url)

    monkeypatch.setattr(sdk_oauth2, "create_oauth_metadata_request", _record)
    monkeypatch.setattr(auth_mod, "LOCK_ACQUIRE_TIMEOUT_S", LOCK_S)
    return entered


async def _hour_grant(endpoint: FakeTokenEndpoint, *, leaving: Any = None) -> Any:
    """A connected one-hour grant with 45 s left: inside the full 60 s lead."""
    provider, storage, store = await _connected_provider(endpoint, remaining=45, leaving=leaving)
    tokens = await storage.get_tokens()
    assert tokens is not None
    tokens.expires_in = 3600
    await storage.set_tokens(tokens)
    store.rows[0].data["tokens_obtained_at"] = time.time() + 45 - 3600
    provider.context.current_tokens = tokens
    provider.context.token_expiry_time = storage.stored_token_expiry()
    return provider, storage, store


async def _sibling(store: FakeAuthStore, endpoint: FakeTokenEndpoint) -> Any:
    """A second connected provider (another session) over the SAME stored grant."""
    provider = auth_mod.build_oauth_provider(
        SERVER_URL,
        _cfg(),
        store=store,
        interactive=False,
        endpoints=_endpoints_for(endpoint.token_endpoint),
    )
    await provider._initialize()
    return provider


class _RevokingResource:
    """A loopback resource that accepts ONLY the issuer's current access token."""

    def __init__(self, endpoint: FakeTokenEndpoint) -> None:
        self.endpoint = endpoint
        self.bearers: list[str] = []
        self._server: asyncio.Server | None = None
        self.url = ""

    async def __aenter__(self) -> _RevokingResource:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self._server.sockets[0].getsockname()[1]}/resource"
        return self

    async def __aexit__(self, *exc: Any) -> None:
        assert self._server is not None
        self._server.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            bearer = ""
            for line in head.decode("latin-1").split("\r\n")[1:]:
                name, _, value = line.partition(":")
                if name.strip().lower() == "authorization":
                    bearer = value.strip().removeprefix("Bearer ")
            self.bearers.append(bearer)
            if bearer == self.endpoint.access_token:
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    b'Content-Length: 11\r\nConnection: close\r\n\r\n{"ok":true}'
                )
            else:
                writer.write(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
        finally:
            writer.close()

    async def get(self, provider: Any) -> int:
        async with create_mcp_http_client(auth=provider) as client:
            return (await client.get(self.url)).status_code


async def _assert_lock_free() -> None:
    async with auth_mod._oauth_refresh_lock(SERVER_URL) as lock:
        assert lock, "the exchange must release the refresh lock once it has persisted"


@pytest.mark.asyncio
@pytest.mark.parametrize("delay", [0.6, SLOW_S])
async def test_401_adopts_the_detached_rotation(auth_branch: list[str], delay: float) -> None:
    # 0.6 s is the positive control (inside the lock wait, green before #3337);
    # SLOW_S is the case that fell through to authorization before it.
    async with FakeTokenEndpoint(response_delay_s=delay) as endpoint:
        provider, storage, _ = await _hour_grant(endpoint)
        async with _RevokingResource(endpoint) as resource:
            start = time.monotonic()
            status = await resource.get(provider)
            elapsed = time.monotonic() - start
        assert status == 200
        assert resource.bearers == ["access-0", "access-1"]
        assert auth_branch == []
        assert elapsed < delay + 1.0
        assert len(endpoint.requests) == 1
        assert endpoint.reuse_attempts == 0
        assert not endpoint.revoked
        assert (await storage.get_tokens()).refresh_token == "refresh-1"
        assert not storage.send_unconfirmed()
        await _assert_lock_free()


@pytest.mark.asyncio
async def test_sibling_and_concurrent_requests_share_one_exchange(auth_branch: list[str]) -> None:
    async with FakeTokenEndpoint(response_delay_s=SLOW_S) as endpoint:
        provider, storage, store = await _hour_grant(endpoint)
        sibling = await _sibling(store, endpoint)
        async with _RevokingResource(endpoint) as resource:
            first = asyncio.create_task(resource.get(provider))
            # Let the first provider's exchange start and own the lock, so the
            # other requests can only meet it detached, never start their own.
            await asyncio.wait_for(endpoint.rotation_applied.wait(), 5)
            statuses = await asyncio.gather(first, resource.get(provider), resource.get(sibling))
        assert statuses == [200, 200, 200]
        # Every request ends on the one rotation. How many first spent a 401 on
        # access-0 depends on scheduling (a second request on the SAME provider
        # queues on the SDK's context lock and may only start after the
        # rotation landed), so only the invariant is pinned.
        assert resource.bearers.count("access-1") == 3
        assert set(resource.bearers) <= {"access-0", "access-1"}
        assert resource.bearers[0] == "access-0"
        assert auth_branch == []
        assert len(endpoint.requests) == 1
        assert endpoint.reuse_attempts == 0
        assert not endpoint.revoked
        assert (await storage.get_tokens()).refresh_token == "refresh-1"


@pytest.mark.asyncio
async def test_cancelled_401_wait_leaves_the_exchange_to_persist(auth_branch: list[str]) -> None:
    async with FakeTokenEndpoint(response_delay_s=SLOW_S) as endpoint:
        provider, storage, _ = await _hour_grant(endpoint)
        async with _RevokingResource(endpoint) as resource:
            request = asyncio.create_task(resource.get(provider))
            deadline = time.monotonic() + 5
            while resource.bearers != ["access-0"] and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert resource.bearers == ["access-0"], "the 401 must have been served"
            await asyncio.sleep(0.1)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert await auth_mod.drain_refresh_exchanges(timeout_s=5)
        assert auth_branch == []
        assert len(endpoint.requests) == 1
        assert endpoint.reuse_attempts == 0
        assert (await storage.get_tokens()).refresh_token == "refresh-1"
        assert not storage.send_unconfirmed()
        await _assert_lock_free()


@pytest.mark.asyncio
async def test_teardown_during_the_wait_presents_nothing(auth_branch: list[str]) -> None:
    leaving = False
    async with FakeTokenEndpoint(response_delay_s=SLOW_S) as endpoint:
        provider, storage, _ = await _hour_grant(endpoint, leaving=lambda: leaving)
        async with _RevokingResource(endpoint) as resource:
            request = asyncio.create_task(resource.get(provider))
            await asyncio.wait_for(endpoint.rotation_applied.wait(), 5)
            await asyncio.sleep(0.4)
            leaving = True
            with pytest.raises(Exception):  # noqa: B017 — whatever the SDK makes of the 401
                await request
        assert await auth_mod.drain_refresh_exchanges(timeout_s=5)
        # The leaving session neither re-POSTs nor adopts; the detached exchange
        # still persists the rotation for the next session.
        assert provider.context.current_tokens.refresh_token is None
        assert len(endpoint.requests) == 1
        assert endpoint.reuse_attempts == 0
        assert (await storage.get_tokens()).refresh_token == "refresh-1"


class _SlowDeadGrant(FakeTokenEndpoint):
    """Rejects the refresh with ``invalid_grant`` — a grant really revoked server-side."""

    def _decide(self, form: dict[str, str]) -> tuple[int, dict[str, Any]]:
        self.requests.append(form)
        return 400, {"error": "invalid_grant"}


@pytest.mark.asyncio
async def test_slow_dead_grant_still_reaches_authorization(auth_branch: list[str]) -> None:
    # Negative control: waiting for the exchange must not hide a dead grant. The
    # 401 reaches the authorization branch on the exchange's verdict (the
    # tombstone it wrote), with no second POST of the rejected token.
    async with _SlowDeadGrant(response_delay_s=SLOW_S) as endpoint:
        provider, storage, _ = await _hour_grant(endpoint)
        # The grant was revoked server-side, access token included: without
        # this the still-valid access-0 is served and no 401 ever happens.
        endpoint.access_token = "revoked-server-side"
        async with _RevokingResource(endpoint) as resource:
            with pytest.raises(_AuthorizationBranchEntered):
                await resource.get(provider)
        assert resource.bearers == ["access-0"]
        assert len(auth_branch) == 1
        assert len(endpoint.requests) == 1
        assert storage.grant_is_dead()
        assert not storage.send_unconfirmed()
        await _assert_lock_free()


@pytest.mark.asyncio
async def test_never_confirmed_exchange_keeps_its_refusal(
    auth_branch: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Negative control: a SENT exchange whose answer never arrives (the read
    # grace runs out first) leaves the marker armed, and the 401 that waited for
    # it gets the existing unconfirmed refusal: no replay, no silent SDK
    # authorization, no guessed grant death.
    monkeypatch.setattr(auth_mod, "REFRESH_LATE_RESPONSE_GRACE_S", 0.8)
    async with FakeTokenEndpoint(response_delay_s=2.0) as endpoint:
        provider, storage, _ = await _hour_grant(endpoint)
        async with _RevokingResource(endpoint) as resource:
            with pytest.raises(auth_mod.McpRefreshUnconfirmedError):
                await resource.get(provider)
        assert auth_mod.REFRESH_CONTENTION.pop(SERVER_URL) == auth_mod.REFRESH_REFUSAL_UNCONFIRMED
        assert resource.bearers == ["access-0"]
        assert auth_branch == []
        assert len(endpoint.requests) == 1
        assert endpoint.reuse_attempts == 0
        assert storage.send_unconfirmed("refresh-0")
        assert not storage.grant_is_dead()
        await _assert_lock_free()


@pytest.mark.asyncio
async def test_foreign_lock_holder_costs_only_the_ordinary_wait(auth_branch: list[str]) -> None:
    # Another PROCESS's exchange is invisible to this loop: only the ordinary
    # lock bound applies, and the contended path presents nothing.
    async with FakeTokenEndpoint() as endpoint:
        provider, storage, _ = await _hour_grant(endpoint)
        endpoint.access_token = "access-revoked-elsewhere"
        fd = os.open(auth_mod._oauth_refresh_lock_path(SERVER_URL), os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            async with _RevokingResource(endpoint) as resource:
                start = time.monotonic()
                with pytest.raises(_AuthorizationBranchEntered):
                    await resource.get(provider)
                elapsed = time.monotonic() - start
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        assert LOCK_S <= elapsed < LOCK_S + 1.0
        assert endpoint.requests == []
        assert not storage.send_unconfirmed()


@pytest.mark.asyncio
async def test_persist_wait_is_per_server_and_cut_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auth_mod, "LOCK_ACQUIRE_TIMEOUT_S", LOCK_S)
    loop = asyncio.get_running_loop()
    parked = asyncio.Event()

    async def _stuck() -> Any:
        await parked.wait()

    other = asyncio.create_task(_stuck())
    ours = asyncio.create_task(_stuck())
    try:
        auth_mod._detach_refresh_exchange(
            other, "https://other.test/mcp", persist_deadline=loop.time() + 60
        )
        auth_mod._detach_refresh_exchange(
            ours, SERVER_URL, persist_deadline=loop.time() + LOCK_S + 0.3
        )
        start = time.monotonic()
        await auth_mod._await_detached_persist("https://unrelated.test/mcp")
        assert time.monotonic() - start < 0.1, "another server's exchange is not waited for"
        await auth_mod._await_detached_persist(SERVER_URL)
        # The wait stops one lock bound short of the exchange's deadline: the
        # caller's ordinary acquire covers the rest, so the total never exceeds
        # the deadline itself.
        assert 0.2 <= time.monotonic() - start < 0.6
        assert not ours.done(), "the wait must never cancel the exchange"
    finally:
        parked.set()
        await asyncio.gather(other, ours)
    assert auth_mod._DETACHED_REFRESH_EXCHANGES.isdisjoint({other, ours})
