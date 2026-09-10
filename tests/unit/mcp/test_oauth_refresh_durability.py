"""Durability of one MCP OAuth refresh exchange across cancellation and peers.

``_refresh_oauth_token_locked`` spends a rotating refresh token, and for a
provider running reuse detection (Notion) the WS response is the ONLY copy of
the rotation: presenting the spent token again returns ``invalid_grant`` AND
revokes the whole family. So an exchange whose result is discarded does not
merely lose an optimisation — it leaves a spent token on disk whose next use
logs out every session.

Three ways that happened, each reproduced here against the real machinery: the
real ``_refresh_oauth_token_locked`` / ``ensure_mcp_oauth_fresh``, a real
``McpTokenStorage`` over the credential row, the real cross-process flock, and
a real HTTP token endpoint (see ``rotating_oauth_server``) that rotates on
every exchange and revokes the family on a replay:

(a) cancelling the awaiting task while the POST was in flight — routine, since
    every sidebar switch disposes a manager and cancels its in-flight connects;
(b) a response that arrived after the total timeout;
(c) a peer persisting its own fresh rotation while our POST was in flight, then
    having it overwritten by our tombstone write (a live grant reading dead).

Each test states the before-fix failure in its docstring so the failing case
stays on record rather than only in a review comment.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from local_operator.mcp import auth as auth_mod
from local_operator.mcp.auth import McpTokenStorage
from tests.unit.mcp.rotating_oauth_server import FakeTokenEndpoint
from tests.unit.mcp.test_auth import FakeAuthStore

#: A reserved-by-RFC ``.test`` name: nothing here may reach a real host, and an
#: accidental attempt should fail loudly rather than succeed against one.
SERVER_URL = "https://mcp.rotation.test/mcp"
CLIENT_ID = "rotation-client"


def _cfg() -> Any:
    from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

    return MCPHttpServerConfig(url=SERVER_URL, auth=MCPAuthConfig(type="oauth"))


def _endpoints_for(token_endpoint: str) -> Any:
    """Discovered endpoints whose token endpoint is the local fixture."""
    from mcp.shared.auth import OAuthMetadata

    from local_operator.mcp.auth import DiscoveredOAuthEndpoints

    return DiscoveredOAuthEndpoints(
        oauth_metadata=OAuthMetadata.model_validate(
            {
                "issuer": "https://issuer.rotation.test",
                "authorization_endpoint": "https://issuer.rotation.test/authorize",
                "token_endpoint": token_endpoint,
            }
        )
    )


def _stub_discovery(monkeypatch: pytest.MonkeyPatch, token_endpoint: str) -> None:
    """Answer metadata discovery with the fixture's endpoint instead of HTTP.

    Discovery is not what these tests are about, and letting it hit the network
    would make them depend on DNS for ``.test``.
    """
    auth_mod._DISCOVERED_ENDPOINTS_CACHE.clear()

    async def _discovery(server_url: str) -> Any:
        return _endpoints_for(token_endpoint)

    monkeypatch.setattr(auth_mod, "discover_oauth_endpoints", _discovery)


async def _seed_expired_grant(
    store: FakeAuthStore, *, access: str = "access-0", refresh: str = "refresh-0"
) -> McpTokenStorage:
    """Store a grant old enough that the proactive refresh actually runs.

    Without an EXPIRED access token ``ensure_mcp_oauth_fresh`` returns before
    reaching the lock and the test would silently exercise nothing.
    """
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    storage = McpTokenStorage(SERVER_URL, store)
    await storage.set_client_info(OAuthClientInformationFull(client_id=CLIENT_ID))
    await storage.set_tokens(OAuthToken(access_token=access, refresh_token=refresh, expires_in=60))
    # Age it past its lifetime; expiry is computed from this stamp.
    store.rows[0].data["tokens_obtained_at"] = time.time() - 600
    return storage


async def _until(predicate: Any, timeout_s: float = 5.0) -> Any:
    """Poll an async predicate; return its last value when it turns truthy."""
    deadline = time.monotonic() + timeout_s
    value = await predicate()
    while not value and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        value = await predicate()
    return value


@pytest.mark.asyncio
async def test_cancelling_the_connect_mid_post_keeps_the_rotated_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) A cancelled POST must not discard the rotation it already caused.

    Before the fix this fails: the cancellation is delivered straight into the
    in-flight ``client.post`` inside ``asyncio.timeout``, the response is
    discarded, and the store is left holding the SPENT ``refresh-0`` — the
    state whose next POST is a reuse-detection double-spend. The server has
    already rotated to ``refresh-1`` and revoked ``access-0``, so nothing but
    that response could ever have carried the new token.
    """
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)

    async with FakeTokenEndpoint(response_delay_s=0.25) as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        task = asyncio.ensure_future(
            auth_mod.ensure_mcp_oauth_fresh(SERVER_URL, _cfg(), store=store)
        )
        # Cancel only once the exchange has LANDED server-side: the window this
        # test is about is "rotated but response still unwritten".
        await asyncio.wait_for(endpoint.rotation_applied.wait(), 5)
        assert endpoint.rotation_count == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        tokens = await storage.get_tokens()

    assert tokens is not None
    assert tokens.refresh_token == "refresh-1", (
        "the rotation the server performed must survive the cancellation; a "
        "spent token left as the only stored one is the reuse-detection "
        "double-spend this subsystem exists to prevent"
    )
    assert tokens.access_token == "access-1"
    assert endpoint.reuse_attempts == 0


@pytest.mark.asyncio
async def test_a_response_past_the_timeout_is_still_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) A response that arrives after the total timeout must still persist.

    Before the fix this fails: the ``asyncio.timeout`` overrun cancels the
    request, classifies the refresh ``"failed"``, and throws away a rotation the
    server has already performed. The bound is patched down so the test does
    not spend the real ten seconds; the behaviour under test is unchanged by
    that, because the module reads the constant at call time.
    """
    monkeypatch.setattr(auth_mod, "REFRESH_HTTP_TIMEOUT_S", 0.2)
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)

    async with FakeTokenEndpoint(response_delay_s=0.6) as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        # Returns "failed" at the bound; the exchange is still in flight.
        await auth_mod.ensure_mcp_oauth_fresh(SERVER_URL, _cfg(), store=store)

        async def _rotated() -> bool:
            tokens = await storage.get_tokens()
            return tokens is not None and tokens.refresh_token == "refresh-1"

        persisted = await _until(_rotated)

    assert persisted, (
        "a rotation that lands after the timeout must be persisted rather than "
        "discarded: the server has spent refresh-0 and nothing else holds "
        "refresh-1"
    )
    assert endpoint.reuse_attempts == 0


@pytest.mark.asyncio
async def test_tombstone_skipped_when_a_peer_rotated_the_grant_mid_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) A tombstone must not overwrite a rotation a peer just persisted.

    The documented window at ``GRANT_DEAD_AT_KEY``: ``mark_grant_dead`` reads
    the payload, adds the marker, and writes the WHOLE snapshot back. A sibling
    that persisted a fresh rotation inside that window had its live grant
    rewritten as dead (or, if it wrote a moment later, its token erased by our
    pre-rotation snapshot) — the user's cure was an interactive login.

    Before the fix this fails: the marker lands on the peer's fresh grant, so
    ``grant_is_dead()`` reads True for a token that works.

    The real flock is held here, because the write it guards is the whole point:
    a mocked lock would make the test a statement about the mock.
    """
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store, refresh="spent-0")

    async def _peer_persists_its_rotation() -> None:
        """What a sibling process does while our POST is in flight."""
        from mcp.shared.auth import OAuthToken

        await storage.set_tokens(
            OAuthToken(access_token="peer-access", refresh_token="peer-1", expires_in=3600)
        )

    async with FakeTokenEndpoint(
        access_token="peer-access", refresh_token="peer-1", on_reject=_peer_persists_its_rotation
    ) as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        async with auth_mod._oauth_refresh_lock(SERVER_URL):
            outcome = await auth_mod._refresh_oauth_token_locked(
                SERVER_URL, storage, _endpoints_for(endpoint.token_endpoint)
            )

    assert outcome == "dead"
    tokens = await storage.get_tokens()
    assert tokens is not None
    assert tokens.refresh_token == "peer-1", "the peer's rotation must not be erased"
    assert tokens.access_token == "peer-access"
    assert storage.grant_is_dead() is False, (
        "a live grant must not read dead: the tombstone would suppress refresh "
        "on a working token until an interactive login"
    )


@pytest.mark.asyncio
async def test_tombstone_written_when_grant_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive half of the compare-and-skip: an UNCHANGED grant is still
    tombstoned.

    The check must not cost the marker its purpose. When the stored payload
    still holds the refresh token the server rejected, the rejection is about
    THIS grant, so it is recorded — that is what stops every later boot from
    re-presenting the token and (on a reuse-detecting provider) revoking the
    family again.
    """
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store, refresh="spent-0")

    # The endpoint's current token is something else, so our presentation is
    # rejected as a replay and nothing is rotated: the store is untouched, which
    # is exactly the state the marker is for.
    async with FakeTokenEndpoint(refresh_token="elsewhere-0") as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        async with auth_mod._oauth_refresh_lock(SERVER_URL):
            outcome = await auth_mod._refresh_oauth_token_locked(
                SERVER_URL, storage, _endpoints_for(endpoint.token_endpoint)
            )

    assert outcome == "dead"
    assert storage.grant_is_dead() is True
