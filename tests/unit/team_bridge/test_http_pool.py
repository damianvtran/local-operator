"""The Team Bridge hub connection pool (``team_bridge/http_pool.py``).

bridge-instant-load: hub reads reuse one long-lived ``httpx.AsyncClient`` on a
loop that OWNS a pool (the ``lop serve`` lifespan), instead of a TLS handshake
per request. Each module invariant is pinned here: reuse (client AND real TCP
socket), no credential on the shared client, no cookie replay, redirects not
followed, per-loop isolation, the per-request fallback on an un-owned loop, and
the lifespan's close.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import http_pool
from local_operator.team_bridge.client import BridgeRefusal, TeamBridgeClient
from tests.unit.team_bridge.test_client import FakeRow, FakeStore, _ok


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[httpx.AsyncClient]:
    """Every client ``http_pool`` builds, in order (the real builder still runs)."""

    seen: list[httpx.AsyncClient] = []
    real = http_pool.build_hub_client

    def spy(transport):
        created = real(transport)
        seen.append(created)
        return created

    monkeypatch.setattr(http_pool, "build_hub_client", spy)
    return seen


def _client(transport: httpx.AsyncBaseTransport | None, store=None, base="https://hub.example/v1"):
    return TeamBridgeClient(None, base_url=base, store=store or FakeStore(), transport=transport)


@pytest.mark.asyncio
async def test_an_owned_loop_reuses_one_client_across_hub_calls(built):
    transport = httpx.MockTransport(lambda request: _ok({"available": True}))
    async with http_pool.owned_hub_pool():
        client = _client(transport)
        assert await client.status() == {"available": True}
        assert await client.status() == {"available": True}
        assert len(built) == 1, "two hub calls on an owned loop must share one client"
        assert not built[0].is_closed, "the pooled client outlives the request"
    assert built[0].is_closed, "leaving the owner closes the pool"


@pytest.mark.asyncio
async def test_an_unowned_loop_keeps_the_per_request_client(built):
    transport = httpx.MockTransport(lambda request: _ok({"available": True}))
    client = _client(transport)
    await client.status()
    await client.status()
    assert len(built) == 2
    assert all(http.is_closed for http in built), "a per-request client closes with its request"


@pytest.mark.asyncio
async def test_the_bearer_rides_each_request_never_the_shared_client(built):
    """Re-sign-in / account change is safe because the client holds no token."""

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return _ok({"available": True})

    store = FakeStore([FakeRow(1, data={"access": "token-1", "account_id": "acc_1"})])
    transport = httpx.MockTransport(handler)
    async with http_pool.owned_hub_pool():
        client = _client(transport, store=store)
        await client.status()
        # A re-sign-in rewrites the row's token; the SAME pooled client must
        # carry the new one on the next request, with nothing to reset.
        store.rows[0].data["access"] = "token-2"
        await client.status()
        (shared,) = built
        assert "authorization" not in {name.lower() for name in shared.headers}
    assert seen == ["Bearer token-1", "Bearer token-2"]


@pytest.mark.asyncio
async def test_a_hub_set_cookie_is_never_replayed(built):
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("cookie"))
        response = _ok({"available": True})
        response.headers["set-cookie"] = "session=abc; Path=/"
        return response

    async with http_pool.owned_hub_pool():
        client = _client(httpx.MockTransport(handler))
        await client.status()
        await client.status()
        (shared,) = built
        assert len(shared.cookies) == 0, "the shared jar must never store a hub cookie"
    assert seen == [None, None]


@pytest.mark.asyncio
async def test_a_redirect_is_not_followed_by_the_pooled_client(built):
    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://elsewhere.example/steal"})

    async with http_pool.owned_hub_pool():
        client = _client(httpx.MockTransport(handler))
        with pytest.raises(BridgeRefusal):
            await client.status()
        assert built[0].follow_redirects is False
    assert len(hits) == 1 and "elsewhere" not in hits[0]


@pytest.mark.asyncio
async def test_each_injected_transport_and_base_gets_its_own_client(built):
    one = httpx.MockTransport(lambda request: _ok({"available": True}))
    two = httpx.MockTransport(lambda request: _ok({"available": True}))
    async with http_pool.owned_hub_pool():
        await _client(one).status()
        await _client(two).status()
        await _client(one, base="https://other-hub.example/v1").status()
        await _client(one).status()
    assert len(built) == 3


def test_a_pool_never_crosses_event_loops(built):
    """A client bound to one loop must not serve a request on another."""

    transport = httpx.MockTransport(lambda request: _ok({"available": True}))

    async def owned_then_hold() -> httpx.AsyncClient:
        http_pool.bind_hub_pool()
        await _client(transport).status()
        return built[-1]

    loop_a = asyncio.new_event_loop()
    try:
        pooled = loop_a.run_until_complete(owned_then_hold())

        async def unowned() -> None:
            await _client(transport).status()

        asyncio.run(unowned())
        assert len(built) == 2 and built[1] is not pooled
        assert built[1].is_closed and not pooled.is_closed
    finally:
        loop_a.run_until_complete(http_pool.unbind_hub_pool())
        loop_a.close()
    assert pooled.is_closed


async def _counting_server() -> tuple[asyncio.AbstractServer, list[int], int]:
    """A tiny keep-alive HTTP/1.1 server counting accepted TCP connections."""

    accepted: list[int] = []
    body = b'{"msg":"ok","result":{"available":true}}'

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        accepted.append(1)
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                if not head:
                    break
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, accepted, port


@pytest.mark.asyncio
async def test_pooled_hub_calls_reuse_one_tcp_connection():
    """The measurable win: one socket for two reads (a real server, not a mock)."""

    server, accepted, port = await _counting_server()
    base = f"http://127.0.0.1:{port}/v1"
    try:
        async with http_pool.owned_hub_pool():
            client = _client(None, base=base)
            await client.status()
            await client.status()
        assert len(accepted) == 1, "the pooled client must keep the connection alive"

        accepted.clear()
        client = _client(None, base=base)
        await client.status()
        await client.status()
        assert len(accepted) == 2, "control: the per-request shape opens one per call"
    finally:
        server.close()
        await server.wait_closed()
