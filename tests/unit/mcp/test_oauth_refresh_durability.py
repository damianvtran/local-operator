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
import logging
import time
from pathlib import Path
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


@pytest.fixture(autouse=True)
def _isolated_lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep this module's refresh lock out of the operator's real config dir.

    ``_oauth_refresh_lock_path`` resolves under ``config_dir()``, so without this
    every test here creates — and briefly flocks — a file in
    ``~/.local-operator``, and two xdist workers running this module at once
    contend on the SAME path. The lock-hold assertions below would then read a
    sibling worker's exchange as this test's, which is exactly the cross-test
    coupling that makes a concurrency test lie. A per-test directory also means
    a leaked lock can never outlive the test that took it.
    """
    from local_operator.paths import CONFIG_DIR_ENV

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "cfg"))


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

    The cancellation is delivered IMMEDIATELY now — teardown is no longer
    blocked on the exchange — so the rotation lands a moment after the connect
    raises, persisted by the detached exchange, which holds the refresh lock
    until it does. Polling for it is therefore part of the contract, not a
    concession: nothing else can recover the token, and the lock is what stops
    a sibling from presenting the spent one while that happens.
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

        async def _rotated() -> bool:
            tokens = await storage.get_tokens()
            return tokens is not None and tokens.refresh_token == "refresh-1"

        persisted = await _until(_rotated)
        tokens = await storage.get_tokens()

    assert persisted, (
        "the rotation the server performed must survive the cancellation; a "
        "spent token left as the only stored one is the reuse-detection "
        "double-spend this subsystem exists to prevent"
    )
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
        # The caller acquires and HANDS OVER the lock, exactly as the real call
        # sites do: the exchange owns the release, so it stays held until the
        # store write is done.
        async with auth_mod._oauth_refresh_lock(SERVER_URL) as lock:
            outcome = await auth_mod._refresh_oauth_token_locked(
                SERVER_URL, storage, _endpoints_for(endpoint.token_endpoint), lock=lock
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
        # The caller acquires and HANDS OVER the lock, exactly as the real call
        # sites do: the exchange owns the release, so it stays held until the
        # store write is done.
        async with auth_mod._oauth_refresh_lock(SERVER_URL) as lock:
            outcome = await auth_mod._refresh_oauth_token_locked(
                SERVER_URL, storage, _endpoints_for(endpoint.token_endpoint), lock=lock
            )

    assert outcome == "dead"
    assert storage.grant_is_dead() is True


# --------------------------------------------------------------------------
# The write FRESHNESS discipline (review B1): a detached exchange's late
# success must not resurrect or erase grant state a better-informed writer
# established while it was in flight.
# --------------------------------------------------------------------------


def _lock_is_held(lock_path: Path) -> bool:
    """Whether a FOREIGN open file description can take the refresh lock.

    A same-process re-flock is useless as evidence — ``flock`` is per open file
    description — so this opens its OWN descriptor and makes one non-blocking
    exclusive attempt. ``True`` means somebody else holds it.
    """
    import os

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        return not auth_mod._try_lock_exclusive(fd)
    finally:
        os.close(fd)


@pytest.mark.asyncio
async def test_a_late_success_cannot_clear_a_newer_tombstone(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A response that lands after a tombstone must keep the grant DEAD.

    The reviewer's B1(a): the exchange persisted unconditionally and
    ``set_tokens`` cleared the dead-grant marker unconditionally, so a late HTTP
    200 rewrote a revoked family's token as ALIVE — every later boot re-POSTed
    it (the storm the marker exists to stop) and ``grant_marker()`` moved, so a
    fresh grant appeared to exist.

    The rotation is still kept — it really happened, and support should be able
    to see it — but the marker is not cleared, so nothing can ever present it.
    """
    caplog.set_level(logging.INFO, logger="local_operator.mcp.auth")
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)

    async with FakeTokenEndpoint(response_delay_s=0.3) as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        exchange = asyncio.ensure_future(
            auth_mod._perform_refresh_exchange(
                SERVER_URL, storage, _endpoints_for(endpoint.token_endpoint)
            )
        )
        await asyncio.wait_for(endpoint.rotation_applied.wait(), 5)
        # A better-informed writer lands first: this is the tombstone a PEER
        # writes when it presents the token we are spending right now.
        assert storage.mark_grant_dead(rejected_refresh_token="refresh-0") is True
        assert await _until(lambda: _resolved(exchange)) is True
        outcome = await exchange

    assert outcome == "refreshed"
    tokens = await storage.get_tokens()
    assert tokens is not None
    assert tokens.refresh_token == "refresh-1", "the rotation the server performed is kept"
    assert storage.grant_is_dead() is True, (
        "the late success cleared a newer tombstone: the revoked family reads "
        "ALIVE again and every later boot re-POSTs it"
    )
    assert any(
        "store_refresh_result" in record.getMessage() for record in caplog.records
    ), "the kept-marker write must name its writer"


@pytest.mark.asyncio
async def test_a_late_success_cannot_overwrite_a_newer_rotation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A late response must not erase a rotation a peer persisted first.

    The other half of B1: the write is conditioned on the row still holding the
    refresh token THIS exchange presented, so a response computed from a grant
    somebody has since replaced is dropped rather than written over the newer
    state. Before the fix the late write clobbered the peer's token.
    """
    from mcp.shared.auth import OAuthToken

    caplog.set_level(logging.INFO, logger="local_operator.mcp.auth")
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)

    async with FakeTokenEndpoint(response_delay_s=0.3) as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        exchange = asyncio.ensure_future(
            auth_mod._perform_refresh_exchange(
                SERVER_URL, storage, _endpoints_for(endpoint.token_endpoint)
            )
        )
        await asyncio.wait_for(endpoint.rotation_applied.wait(), 5)
        peer = McpTokenStorage(SERVER_URL, store)
        await peer.set_tokens(
            OAuthToken(access_token="peer-access", refresh_token="refresh-9", expires_in=3600)
        )
        assert await _until(lambda: _resolved(exchange)) is True
        outcome = await exchange

    assert outcome == "refreshed"  # the store holds something usable; the caller re-reads
    tokens = await storage.get_tokens()
    assert tokens is not None
    assert tokens.refresh_token == "refresh-9", "the peer's rotation was overwritten"
    assert tokens.access_token == "peer-access"
    assert any(
        "store_refresh_result" in record.getMessage() and "NOT persisted" in record.getMessage()
        for record in caplog.records
    ), "a dropped rotation must be logged at INFO, naming its writer"


async def _resolved(task: "asyncio.Task[Any]") -> bool:
    """Poll helper: has this exchange task finished?"""
    return task.done()


@pytest.mark.asyncio
async def test_two_concurrent_slow_exchanges_never_revoke_the_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two sessions refreshing the same grant must not double-spend it.

    The reviewer's B1(b) and M2 measured together: with two slow exchanges, the
    replayed token triggered reuse detection (``reuse_attempts == 1``, family
    revoked) and the row was left holding a token of that revoked family reading
    ALIVE. Here the second attempt cannot even POST: the exchange owns the lock
    for the whole request and the write, so the sibling's bounded acquire gives
    up and takes the contended path.
    """
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)
    # Small enough that a refusal is quick, large enough that the holder (whose
    # response takes 0.4s) is still working when the sibling gives up.
    monkeypatch.setattr(auth_mod, "LOCK_ACQUIRE_TIMEOUT_S", 0.1)
    endpoints = None

    async with FakeTokenEndpoint(response_delay_s=0.4) as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        endpoints = _endpoints_for(endpoint.token_endpoint)

        async def attempt() -> Any:
            # The real call shape at both refresh sites: acquire, then hand the
            # lock to the exchange so IT owns the release.
            async with auth_mod._oauth_refresh_lock(SERVER_URL) as lock:
                if not lock:
                    return "contended"
                return await auth_mod._refresh_oauth_token_locked(
                    SERVER_URL, storage, endpoints, lock=lock
                )

        outcomes = await asyncio.gather(attempt(), attempt())

    assert endpoint.reuse_attempts == 0, (
        "the spent token was presented a second time: reuse detection revokes " "the whole family"
    )
    assert endpoint.revoked is False
    assert endpoint.rotation_count == 1, "exactly one exchange consumed the grant"
    assert sorted(outcomes, key=str) == ["contended", "refreshed"]
    tokens = await storage.get_tokens()
    assert tokens is not None
    assert tokens.refresh_token == "refresh-1"
    assert storage.grant_is_dead() is False


@pytest.mark.asyncio
async def test_the_lock_is_held_when_the_awaiter_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the connect must NOT free the lock while the POST is in flight.

    The reviewer's M2, measured: the lock used to be released the instant the
    cancellation was delivered (they measured the flock FREE at the cancel
    instant), so a sibling acquired it, re-read the still-old row and re-presented
    the spent token — the family-revoking request. Now the exchange owns the lock
    until it has persisted, and a sibling cannot get in.
    """
    store = FakeAuthStore()
    await _seed_expired_grant(store)
    lock_path = auth_mod._oauth_refresh_lock_path(SERVER_URL)

    async with FakeTokenEndpoint(response_delay_s=0.4) as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        task = asyncio.ensure_future(
            auth_mod.ensure_mcp_oauth_fresh(SERVER_URL, _cfg(), store=store)
        )
        await asyncio.wait_for(endpoint.rotation_applied.wait(), 5)
        assert _lock_is_held(lock_path) is True, "the exchange should hold the lock"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The instant the awaiter is gone.
        held = _lock_is_held(lock_path)
        assert held is True, (
            "the lock was freed while the POST was still in flight: a sibling can "
            "acquire it and re-present the spent token"
        )
        assert endpoint.reuse_attempts == 0

        async def _free() -> bool:
            return not _lock_is_held(lock_path)

        assert await _until(_free) is True, "the exchange never released the lock"


@pytest.mark.asyncio
async def test_the_lock_is_held_when_the_awaiter_overruns_its_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The budget bounds the WAIT, not the lock — and teardown is not delayed.

    The two halves of the reviewer's M2/minor pair. The timeout arm used to
    return with no wait at all while the lock was released by the awaiter's
    context manager; the cancel arm waited out the remainder of the budget
    (measured: 1.97s of a patched 2s budget blocked teardown) for a guarantee it
    did not deliver. Now the exchange owns the lock for both, and neither path
    waits.
    """
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)
    lock_path = auth_mod._oauth_refresh_lock_path(SERVER_URL)
    monkeypatch.setattr(auth_mod, "REFRESH_HTTP_TIMEOUT_S", 0.1)

    async with FakeTokenEndpoint(response_delay_s=0.6) as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        started = time.monotonic()
        await auth_mod.ensure_mcp_oauth_fresh(SERVER_URL, _cfg(), store=store)
        elapsed = time.monotonic() - started
        assert (
            elapsed < 0.5
        ), "the awaiter waited for the exchange instead of returning at its budget"
        assert (
            _lock_is_held(lock_path) is True
        ), "the over-budget exchange must keep the lock it is still POSTing under"
        assert endpoint.reuse_attempts == 0

        async def _rotated() -> bool:
            tokens = await storage.get_tokens()
            return tokens is not None and tokens.refresh_token == "refresh-1"

        assert await _until(_rotated) is True, "a rotation that lands must still be persisted"

        async def _free() -> bool:
            return not _lock_is_held(lock_path)

        assert await _until(_free) is True


@pytest.mark.asyncio
async def test_a_sent_but_unacknowledged_exchange_blocks_the_next_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request that was SENT and never answered must not be re-presented.

    The state the whole marker exists for: after a read timeout the server may
    well have rotated and revoked, so the next refresh must take an honest
    non-POST path — a fresh sign-in — rather than spending the token again. The
    claim is measured at the wire: exactly ONE request ever reaches the
    endpoint.
    """
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)
    # ONLY the read grace is shrunk: the request connects and is written (the
    # budget is untouched at 10s), so this is httpx's post-send ReadTimeout —
    # the suspect shape — rather than the connect's own budget overrunning.
    monkeypatch.setattr(auth_mod, "REFRESH_LATE_RESPONSE_GRACE_S", 0.15)

    async with FakeTokenEndpoint(response_delay_s=1.0) as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        endpoints = _endpoints_for(endpoint.token_endpoint)

        first = await auth_mod._refresh_oauth_token_locked(SERVER_URL, storage, endpoints)

        assert first == "unacknowledged"
        assert len(endpoint.requests) == 1
        assert (
            storage.send_unconfirmed() is True
        ), "the presented token must be recorded as possibly spent"

        # The next refresh — the one that would replay the token — refuses, and
        # costs no request at all.
        async with auth_mod._oauth_refresh_lock(SERVER_URL) as lock:
            second = await auth_mod._refresh_oauth_token_locked(
                SERVER_URL, storage, endpoints, lock=lock
            )

        assert second == "unacknowledged"
        assert len(endpoint.requests) == 1, "the spent token was presented again"
        assert endpoint.reuse_attempts == 0


@pytest.mark.asyncio
async def test_a_peer_rotation_clears_a_stale_send_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marker keyed to a token nobody holds any more must not suppress anything.

    Self-healing, both halves: the marker records a DIGEST of the presented
    token, so a peer's rotation makes it stale ("ignore and clear it"), and the
    TTL bounds a marker nothing ever resolved. Without the staleness test a
    healthy new grant would be refused a refresh until the user re-authed —
    a false positive paid for with a browser visit.
    """
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)
    storage.mark_send_unconfirmed("refresh-0")
    assert storage.send_unconfirmed() is True

    # A peer rotates the grant in the shared row: the marker no longer describes
    # anything that can be presented.
    from mcp.shared.auth import OAuthToken

    peer = McpTokenStorage(SERVER_URL, store)
    await peer.set_tokens(
        OAuthToken(access_token="peer-access", refresh_token="refresh-2", expires_in=60)
    )
    store.rows[0].data["tokens_obtained_at"] = time.time() - 600

    assert storage.send_unconfirmed() is False
    assert (
        auth_mod.GRANT_UNCONFIRMED_SEND_KEY not in store.rows[0].data
    ), "the stale marker was believed and left in place"

    # And the peer's token refreshes normally: nothing was suppressed.
    async with FakeTokenEndpoint(refresh_token="refresh-2") as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        async with auth_mod._oauth_refresh_lock(SERVER_URL) as lock:
            outcome = await auth_mod._refresh_oauth_token_locked(
                SERVER_URL, storage, _endpoints_for(endpoint.token_endpoint), lock=lock
            )

    assert outcome == "refreshed"
    assert len(endpoint.requests) == 1
    assert endpoint.reuse_attempts == 0
    tokens = await storage.get_tokens()
    assert tokens is not None and tokens.refresh_token == "refresh-1"


@pytest.mark.asyncio
async def test_a_connect_phase_failure_leaves_no_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing was written, so nothing may be suspect.

    httpx distinguishes the connect phase (refused, DNS, TLS handshake, no pool
    slot) from a failure after the request was written, and the marker follows
    that distinction exactly: a refused connection never presented the token, so
    the next attempt is an ordinary transient retry. Arming the marker here would
    cost the user an interactive sign-in for a network blip.
    """
    import httpx

    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)
    calls = {"n": 0}

    def _refused(request: "httpx.Request") -> "httpx.Response":
        calls["n"] += 1
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(_refused)
    real_client = httpx.AsyncClient

    def patched_client(*args: Any, **kwargs: Any) -> Any:
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    outcome = await auth_mod._refresh_oauth_token_locked(
        SERVER_URL, storage, _endpoints_for("http://127.0.0.1:1/token")
    )

    assert calls["n"] == 1
    assert outcome == "failed"
    assert auth_mod.GRANT_UNCONFIRMED_SEND_KEY not in store.rows[0].data
    assert storage.send_unconfirmed() is False


@pytest.mark.asyncio
async def test_a_post_send_failure_keeps_the_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror image: a read timeout CAN have been written, so it is suspect.

    Same taxonomy, opposite conclusion. ``ReadTimeout`` happens after the
    request is on the wire, so the token may have been spent and the next
    refresh must refuse rather than replay it.
    """
    import httpx

    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)

    def _timed_out(request: "httpx.Request") -> "httpx.Response":
        raise httpx.ReadTimeout("no response", request=request)

    transport = httpx.MockTransport(_timed_out)
    real_client = httpx.AsyncClient

    def patched_client(*args: Any, **kwargs: Any) -> Any:
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    outcome = await auth_mod._refresh_oauth_token_locked(
        SERVER_URL, storage, _endpoints_for("http://127.0.0.1:1/token")
    )

    assert outcome == "unacknowledged"
    assert auth_mod.GRANT_UNCONFIRMED_SEND_KEY in store.rows[0].data
    assert storage.send_unconfirmed() is True


@pytest.mark.asyncio
async def test_an_unresolved_marker_expires(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The marker is BOUNDED: it cannot suppress a server's refresh forever.

    The stated trade-off made testable. A marker armed in the crash-between-arm-
    and-wire window describes a token nothing ever presented, so after the TTL
    the refresh path presents it again — exactly the pre-marker behaviour, which
    is the floor this can degrade to.
    """
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)
    monkeypatch.setattr(auth_mod, "UNCONFIRMED_SEND_TTL_S", 0.0)

    storage.mark_send_unconfirmed("refresh-0")

    assert (
        storage.send_unconfirmed() is False
    ), "an expired marker must not keep suppressing the refresh path"
    assert auth_mod.GRANT_UNCONFIRMED_SEND_KEY not in store.rows[0].data
