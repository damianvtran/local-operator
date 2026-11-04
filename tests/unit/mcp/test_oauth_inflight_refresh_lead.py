"""Connected OAuth providers refresh before expiry without losing usable grants.

The SDK/httpx2 auth flow is driven by the same client factory as the MCP
transport. Both token and resource requests use loopback HTTP servers, so the
assertions record actual POSTs and bearer headers without reaching .test DNS.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcp.client.streamable_http import create_mcp_http_client

from local_operator.mcp import auth as auth_mod
from tests.unit.mcp.rotating_oauth_server import FakeTokenEndpoint
from tests.unit.mcp.test_auth import FakeAuthStore

# Reuse the per-test cross-process lock fixture (pytest discovers the import).
from tests.unit.mcp.test_oauth_refresh_durability import (  # noqa: F401
    SERVER_URL,
    _cfg,
    _endpoints_for,
    _isolated_lock_dir,
    _seed_expired_grant,
)


async def _connected_provider(
    endpoint: FakeTokenEndpoint, *, remaining: float, leaving: Any = None
) -> tuple[Any, Any, FakeAuthStore]:
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)
    store.rows[0].data["tokens_obtained_at"] = time.time() + remaining - 60
    provider = auth_mod.build_oauth_provider(
        SERVER_URL,
        _cfg(),
        store=store,
        interactive=False,
        endpoints=_endpoints_for(endpoint.token_endpoint),
        leaving=leaving,
    )
    # Model a long-lived CONNECTED provider: no startup freshness check runs
    # between loading these tokens and making the resource request.
    await provider._initialize()
    return provider, storage, store


async def _resource_request(provider: Any) -> list[str | None]:
    headers: list[str | None] = []

    async def resource(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            received = {}
            for line in head.decode("latin-1").split("\r\n")[1:]:
                name, _, value = line.partition(":")
                received[name.lower()] = value.strip()
            headers.append(received.get("authorization"))
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b'Content-Length: 11\r\nConnection: close\r\n\r\n{"ok":true}'
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(resource, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        async with create_mcp_http_client(auth=provider) as client:
            response = await client.get(f"http://127.0.0.1:{port}/resource")
        assert response.status_code == 200
        assert response.json() == {"ok": True}
        return headers
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining,posts,access", [(30, 1, "access-1"), (120, 0, "access-0")])
async def test_connected_refresh_lead(remaining: float, posts: int, access: str) -> None:
    async with FakeTokenEndpoint() as endpoint:
        provider, storage, _ = await _connected_provider(endpoint, remaining=remaining)
        headers = await _resource_request(provider)
        assert len(endpoint.requests) == posts
        assert headers == [f"Bearer {access}"]
        assert not endpoint.revoked
        assert endpoint.reuse_attempts == 0
        assert not storage.send_unconfirmed("refresh-0")


# Every non-success outcome is exercised through the SDK/client request flow,
# not just the coordinator. Otherwise an unguarded SDK POST can look harmless
# when inspecting only the coordinator's return value.
FAILURES = (
    "contended",
    "overran",
    "failed",
    "unreachable",
    "unsent",
    "unacknowledged",
    "unattributed",
    "dead",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", FAILURES)
@pytest.mark.parametrize("remaining", [30, -30])
async def test_refresh_failure_respects_actual_expiry(
    monkeypatch: pytest.MonkeyPatch, outcome: str, remaining: float
) -> None:
    async with FakeTokenEndpoint() as endpoint:
        provider, storage, store = await _connected_provider(endpoint, remaining=remaining)
        attempts = []
        if outcome == "unacknowledged":
            storage.mark_send_unconfirmed("refresh-0")
        marker = store.rows[0].data.get(auth_mod.GRANT_UNCONFIRMED_SEND_KEY)

        async def failed_refresh(*args: Any, **kwargs: Any) -> str:
            assert kwargs["lock"], "coordination must hold the real cross-process lock"
            attempts.append(outcome)
            return outcome

        monkeypatch.setattr(auth_mod, "_refresh_oauth_token_locked", failed_refresh)
        if remaining > 0:
            assert await _resource_request(provider) == ["Bearer access-0"]
            assert provider.context.current_tokens.refresh_token == "refresh-0"
            assert auth_mod.REFRESH_CONTENTION.pop(SERVER_URL) is None
        elif outcome == "dead":
            # Existing dead-grant disposition: strip the refresh token and let
            # the SDK send without expired access; a real 401 would reauthorize.
            assert await _resource_request(provider) == [None]
            assert provider.context.current_tokens.refresh_token is None
        else:
            error = (
                auth_mod.McpRefreshUnconfirmedError
                if outcome == "unacknowledged"
                else auth_mod.McpRefreshContendedError
            )
            with pytest.raises(error):
                await _resource_request(provider)
            assert provider.context.current_tokens.refresh_token is None
            auth_mod.REFRESH_CONTENTION.pop(SERVER_URL)
        assert attempts == [outcome]
        assert endpoint.requests == [], "the SDK must never POST outside coordination"
        assert store.rows[0].data.get(auth_mod.GRANT_UNCONFIRMED_SEND_KEY) == marker


@pytest.mark.asyncio
async def test_peer_rotation_inside_lead_is_adopted_without_post() -> None:
    from mcp.shared.auth import OAuthToken

    async with FakeTokenEndpoint() as endpoint:
        provider, storage, _ = await _connected_provider(endpoint, remaining=30)
        # The new token is ALSO inside the lead. A peer has already paid for
        # this rotation, so freshness alone must not cause another token POST.
        await storage.set_tokens(
            OAuthToken(access_token="peer-access", refresh_token="peer-refresh", expires_in=30)
        )
        assert await _resource_request(provider) == ["Bearer peer-access"]
        assert provider.context.current_tokens.refresh_token == "peer-refresh"
        assert endpoint.requests == []


@pytest.mark.asyncio
async def test_leaving_inside_lead_sends_valid_access_without_refresh() -> None:
    async with FakeTokenEndpoint() as endpoint:
        provider, storage, _ = await _connected_provider(
            endpoint, remaining=30, leaving=lambda: True
        )
        assert await _resource_request(provider) == ["Bearer access-0"]
        assert provider.context.current_tokens.refresh_token == "refresh-0"
        assert endpoint.requests == []
        assert not storage.send_unconfirmed()


@pytest.mark.asyncio
async def test_outside_lead_does_not_lock_or_reread(monkeypatch: pytest.MonkeyPatch) -> None:
    async with FakeTokenEndpoint() as endpoint:
        provider, storage, _ = await _connected_provider(endpoint, remaining=120)

        def unexpected(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("fresh connected tokens must not lock or re-read the store")

        monkeypatch.setattr(auth_mod, "_oauth_refresh_lock", unexpected)
        monkeypatch.setattr(provider.context.storage, "get_tokens", unexpected)
        monkeypatch.setattr(provider._refresh_coord_storage, "stored_token_expiry", unexpected)
        assert await _resource_request(provider) == ["Bearer access-0"]
        assert endpoint.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", [30, -30])
async def test_contended_lock_never_spends_a_grant(
    monkeypatch: pytest.MonkeyPatch, remaining: float
) -> None:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def contended(*args: Any, **kwargs: Any) -> AsyncIterator[bool]:
        # Exercise a real cancellable WAIT rather than an instantaneous refusal.
        await asyncio.sleep(1.5)
        yield False

    async with FakeTokenEndpoint() as endpoint:
        provider, _, _ = await _connected_provider(endpoint, remaining=remaining)
        monkeypatch.setattr(auth_mod, "_oauth_refresh_lock", contended)
        if remaining > 0:
            assert await _resource_request(provider) == ["Bearer access-0"]
        else:
            with pytest.raises(auth_mod.McpRefreshContendedError) as exc:
                await _resource_request(provider)
            assert exc.value.reason_code == auth_mod.REFRESH_REFUSAL_LOCK
            auth_mod.REFRESH_CONTENTION.pop(SERVER_URL)
        assert endpoint.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", [30, -30])
async def test_unconfirmed_marker_never_replays_a_refresh_token(remaining: float) -> None:
    async with FakeTokenEndpoint() as endpoint:
        provider, storage, store = await _connected_provider(endpoint, remaining=remaining)
        storage.mark_send_unconfirmed("refresh-0")
        marker = store.rows[0].data[auth_mod.GRANT_UNCONFIRMED_SEND_KEY].copy()
        if remaining > 0:
            # Repeated requests may still use ACCESS but must never re-present
            # the marked REFRESH token or clear the evidence protecting it.
            assert await _resource_request(provider) == ["Bearer access-0"]
            assert await _resource_request(provider) == ["Bearer access-0"]
        else:
            with pytest.raises(auth_mod.McpRefreshUnconfirmedError):
                await _resource_request(provider)
            auth_mod.REFRESH_CONTENTION.pop(SERVER_URL)
        assert endpoint.requests == []
        assert store.rows[0].data[auth_mod.GRANT_UNCONFIRMED_SEND_KEY] == marker


@pytest.mark.asyncio
async def test_early_endpoint_failure_keeps_access_and_durable_send_marker() -> None:
    # A bare 503 is ambiguous, even without a committed rotation. Keep the
    # existing conservative marker policy: usable access proceeds but refresh
    # may never replay the token whose exchange was not acknowledged.
    async with FakeTokenEndpoint(
        refuse_before_rotation=(503, "temporarily_unavailable")
    ) as endpoint:
        provider, storage, store = await _connected_provider(endpoint, remaining=30)
        assert await _resource_request(provider) == ["Bearer access-0"]
        assert len(endpoint.requests) == 1
        marker = store.rows[0].data[auth_mod.GRANT_UNCONFIRMED_SEND_KEY].copy()
        assert storage.send_unconfirmed("refresh-0")
        assert await _resource_request(provider) == ["Bearer access-0"]
        assert len(endpoint.requests) == 1
        assert store.rows[0].data[auth_mod.GRANT_UNCONFIRMED_SEND_KEY] == marker
        assert not endpoint.revoked
        assert endpoint.reuse_attempts == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", [30, -30])
async def test_unexpected_coordination_exception_respects_actual_expiry(
    monkeypatch: pytest.MonkeyPatch, remaining: float
) -> None:
    async with FakeTokenEndpoint() as endpoint:
        provider, _, _ = await _connected_provider(endpoint, remaining=remaining)

        async def failed_refresh(*args: Any, **kwargs: Any) -> Any:
            raise OSError("local exchange failure")

        monkeypatch.setattr(auth_mod, "_refresh_oauth_token_locked", failed_refresh)
        if remaining > 0:
            assert await _resource_request(provider) == ["Bearer access-0"]
            assert provider.context.current_tokens.refresh_token == "refresh-0"
        else:
            with pytest.raises(auth_mod.McpRefreshContendedError) as exc:
                await _resource_request(provider)
            assert exc.value.reason_code == auth_mod.REFRESH_REFUSAL_UNATTRIBUTED
            assert provider.context.current_tokens.refresh_token is None
            auth_mod.REFRESH_CONTENTION.pop(SERVER_URL)
        assert endpoint.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("marked", [False, True])
async def test_expiry_between_coordination_and_sdk_never_posts_unlocked(
    monkeypatch: pytest.MonkeyPatch, marked: bool
) -> None:
    async with FakeTokenEndpoint() as endpoint:
        provider, storage, _ = await _connected_provider(endpoint, remaining=30)
        if marked:
            storage.mark_send_unconfirmed("refresh-0")
        coordinate = provider._coordinate_inflight_refresh

        async def expiry_before_sdk() -> None:
            if marked:
                await coordinate()
            # Simulate the deadline passing while the SDK waits for its lock.
            provider.context.token_expiry_time = time.time() - 1

        monkeypatch.setattr(provider, "_coordinate_inflight_refresh", expiry_before_sdk)
        with pytest.raises(auth_mod.McpRefreshContendedError):
            await _resource_request(provider)
        assert endpoint.requests == []
        assert storage.send_unconfirmed("refresh-0") is marked
        assert provider.context.current_tokens.refresh_token is None
        auth_mod.REFRESH_CONTENTION.pop(SERVER_URL)
