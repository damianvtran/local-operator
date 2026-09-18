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
    # Its OWN outcome, not "failed": nothing was presented, so nothing could
    # have been refused. The split is what lets the user-visible text say "could
    # not be reached" instead of reporting a rejection by a server that was
    # never contacted (review round 2, minor 1).
    assert outcome == "unreachable"
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


@pytest.mark.asyncio
async def test_a_5xx_after_the_rotation_keeps_the_marker_and_blocks_the_next_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An answer that proves NOTHING about our token must not clear the marker.

    Review round 2, minor 2. Every non-200 used to resolve the send marker, on
    the premise that "the server answered, so it either rotated-and-told-us or
    refused without rotating". The first half of that is false for a provider
    that COMMITS the rotation and then fails the response (a proxy in front of a
    rotating issuer is enough): the row is left holding a spent token, the
    marker is gone, and the next connect re-presents it — the reuse-detecting
    POST this whole mechanism exists to prevent, with the family as the cost.

    The fix is a rule, not a status-code tweak: clear only for an answer that
    settles the question (a parsed 200, or a parsed ``invalid_grant``). The
    fixture here rotates and THEN answers 500, so both halves are measured —
    the rotation really happened, and the marker survives it.
    """
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)

    async with FakeTokenEndpoint(commit_then_fail_status=500) as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        endpoints = _endpoints_for(endpoint.token_endpoint)

        first = await auth_mod._refresh_oauth_token_locked(SERVER_URL, storage, endpoints)

        assert first == "failed"
        assert endpoint.rotation_count == 1, "the fixture must have committed the rotation"
        assert endpoint.reuse_attempts == 0
        assert (
            storage.send_unconfirmed() is True
        ), "a 5xx is an unknown outcome: the presented token may already be spent"

        # The next refresh is the one that would replay the possibly-spent token.
        async with auth_mod._oauth_refresh_lock(SERVER_URL) as lock:
            second = await auth_mod._refresh_oauth_token_locked(
                SERVER_URL, storage, endpoints, lock=lock
            )

        assert second == "unacknowledged"
    assert len(endpoint.requests) == 1, "the possibly-spent token was presented again"
    assert endpoint.reuse_attempts == 0


@pytest.mark.asyncio
async def test_a_4xx_that_is_not_invalid_grant_also_keeps_the_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule is "proves it was not consumed", not "is a 5xx".

    A 403 from a bot filter is a refusal we cannot read as a statement about
    our token — it says nothing about whether the exchange was performed — so it
    takes the same conservative path. The counter-case (a definitive 400
    ``invalid_grant`` clears the marker, because the tombstone that follows IS
    the recorded outcome) is covered by the tombstone tests above.
    """
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)

    async with FakeTokenEndpoint(commit_then_fail_status=403) as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        outcome = await auth_mod._refresh_oauth_token_locked(
            SERVER_URL, storage, _endpoints_for(endpoint.token_endpoint)
        )

    assert outcome == "failed"
    assert storage.send_unconfirmed() is True
    assert endpoint.reuse_attempts == 0


@pytest.mark.asyncio
async def test_a_logout_during_an_in_flight_exchange_is_not_undone(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A removed row stays removed: the two writers now agree (QA round 2, Q6).

    ``store_refresh_result`` used to re-create a row it found absent, reasoning
    that "absence is not evidence that somebody replaced the grant", while its
    sibling ``mark_grant_dead`` declined to write on the same absent row for the
    opposite reason. So a ``/mcp logout`` landing inside the detached-exchange
    grace was silently undone by a refresh that was already on the wire, and the
    re-created row holds a live rotation — the next boot re-authorizes a server
    the user asked us to forget.

    An absent row is now treated as the deliberate removal it is (the
    three-valued ``has_stored_row`` keeps "the store could not be read"
    distinct), and the drop is logged at INFO naming the writer, so support can
    read the loss out of the log.
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
        # The user asks us to forget this server while the response is in flight.
        assert storage.clear() is True
        assert store.rows == []
        assert await _until(lambda: _resolved(exchange)) is True
        outcome = await exchange

    assert outcome == "refreshed"  # the exchange itself completed and rotated
    assert store.rows == [], "a logout was silently undone by an in-flight refresh"
    assert await storage.get_tokens() is None
    assert any(
        "store_refresh_result" in record.getMessage()
        and "removed while this exchange was in flight" in record.getMessage()
        for record in caplog.records
    ), "a dropped rotation must be logged at INFO, naming its writer and the reason"


@pytest.mark.asyncio
async def test_a_marker_write_never_re_creates_a_removed_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A removal that beat the marker write stays a removal (review round 3, M1).

    ``mark_send_unconfirmed`` used ``self._read() or {}``, so a ``/mcp logout``
    landing between the exchange's locked re-read and the write-ahead arming was
    silently undone: the upsert re-created the row keyed on ``project_id``
    (probed before the fix: ``rows == []`` -> one row of
    ``grant_refresh_unconfirmed`` + ``project_id``, and the next
    ``send_unconfirmed()`` then cleared the marker and left the empty row
    behind). The impact is bounded — no grant comes back, so the wording and
    ``server_has_stored_grant`` are unaffected — but it contradicts the symmetry
    this delta states ("two writers racing a removal must give the same answer"):
    ``store_refresh_result`` and ``mark_grant_dead`` both declined on a
    definitively removed row and this third writer did not.

    The rule is narrow, and the second half pins that: an UNREADABLE store is
    not evidence of removal, so the marker is still written. Suppressing a
    healthy refresh over a transient store error would cost the user a browser
    visit for nothing.
    """
    caplog.set_level(logging.INFO, logger="local_operator.mcp.auth")
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)
    assert storage.clear() is True
    assert store.rows == []

    storage.mark_send_unconfirmed("refresh-0")

    assert store.rows == [], "the marker write re-created the row a logout removed"
    assert storage.send_unconfirmed() is False
    assert any(
        "mark_send_unconfirmed" in record.getMessage() for record in caplog.records
    ), "a refusal to arm the marker must be logged, naming its writer"

    # The sibling mutator answers the same question the same way — including on
    # the call that carries no rejected token, which was the other shape that
    # re-created the row.
    assert storage.mark_grant_dead() is False
    assert store.rows == []
    assert storage.grant_is_dead() is False

    class _UnreadableStore(FakeAuthStore):
        def list_credentials(self, provider: Any = None, include_disabled: bool = False) -> Any:
            raise RuntimeError("store unavailable")

    unreadable = _UnreadableStore()
    blind = McpTokenStorage(SERVER_URL, unreadable)
    blind.mark_send_unconfirmed("refresh-0")
    assert unreadable.rows != [], "an unreadable store must not read as a removal"


@pytest.mark.asyncio
async def test_an_empty_row_is_unsent_so_no_server_is_blamed() -> None:
    """Nothing to present means no request at all (review round 3, M2).

    This shape returned ``"failed"``, which the manager composes as the ENDPOINT
    text — "the server returned no token" — a claim that is untrue about the
    WIRE (no request was made) and about the SERVER (it was never asked). Probed
    before the fix: ``client_info`` present, no stored tokens -> outcome
    ``"failed"`` -> the endpoint wording on the card.
    """
    from mcp.shared.auth import OAuthClientInformationFull

    from local_operator.mcp.auth import REFRESH_REFUSAL_UNSENT, McpRefreshContendedError
    from local_operator.mcp.manager import McpManager

    store = FakeAuthStore()
    storage = McpTokenStorage(SERVER_URL, store)
    await storage.set_client_info(OAuthClientInformationFull(client_id=CLIENT_ID))

    # The endpoint points at a closed port: had this shape made a request, the
    # exchange would have failed rather than reported an outcome.
    outcome = await auth_mod._refresh_oauth_token_locked(
        SERVER_URL, storage, _endpoints_for("http://127.0.0.1:1/token")
    )

    assert outcome == "unsent"
    text = McpManager._auth_failure_text(
        "notion", McpRefreshContendedError(SERVER_URL, reason_code=REFRESH_REFUSAL_UNSENT)
    )
    assert text == "no stored token to send"
    assert "server" not in text


@pytest.mark.asyncio
async def test_a_first_grant_still_creates_its_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The removal rule must not stop the ordinary creation path.

    ``store_refresh_result`` never creates a grant — a refresh can only run
    against a row that already held the token it presented — so the only writer
    that may create one is ``set_tokens``, which is what a completed interactive
    login calls. Pinned here because the Q6 fix is a change to a WRITE path, and
    the cheap way to get it wrong is to make absence fatal everywhere.
    """
    from mcp.shared.auth import OAuthToken

    store = FakeAuthStore()
    storage = McpTokenStorage(SERVER_URL, store)
    assert await storage.get_tokens() is None

    await storage.set_tokens(OAuthToken(access_token="acc", refresh_token="ref", expires_in=60))

    tokens = await storage.get_tokens()
    assert tokens is not None and tokens.refresh_token == "ref"


# ---------------------------------------------------------------------------
# Teardown: the credential store must outlive the exchange, and the loss of a
# rotation must be readable in the log.
# ---------------------------------------------------------------------------
#
# The tests above cancel a connect and let the detached exchange finish while the
# process lives on. This section is about the other half of every runtime exit:
# the session's dispose hooks, which close the credential store — and which used
# to close it BEFORE the MCP teardown ran, so an exchange still in flight got its
# HTTP 200, tried to persist the rotation through a closed SQLite connection,
# had the write swallowed at DEBUG, and was told it had succeeded. The row kept
# the SPENT token with the write-ahead marker still armed, so every later connect
# refused to refresh for an hour and told the user to re-authenticate.
#
# Three claims, and each is asserted against the real objects rather than a
# double: a real ``AuthStore`` (SQLite, so a closed handle genuinely refuses the
# write), the real ``drain_refresh_exchanges``, and the real provider built by
# ``build_oauth_provider`` — the same call the manager makes per connect.


async def _cancellation_was_logged(caplog: pytest.LogCaptureFixture) -> bool:
    """Whether any cancelled detached exchange has reported itself yet.

    The settle callback runs from the loop's done-callback queue, so it lands a
    tick after the task resolves: polling is the honest way to read it, and
    polling for the MESSAGE (not for a sleep long enough) is what keeps the
    assertions below from passing on an empty log.
    """
    return any("CANCELLED before any answer" in record.getMessage() for record in caplog.records)


#: A task left pending by a loop that closed under it can never be collected
#: cleanly, and letting it go would print "Task was destroyed but it is pending!"
#: into the suite's output — noise about the loop, not about the drain the test
#: below is about. One strong reference for the life of the process.
_ABANDONED_TASKS: list[Any] = []


def _cancellation_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Every settle line a cancelled detached exchange has emitted.

    The LINE is the behaviour under test in the arms below — what the log says
    after the fact is the whole point of d2 — so the filter lives in one place
    rather than being restated, differently, by each arm.
    """
    return [
        record.getMessage()
        for record in caplog.records
        if "CANCELLED before any answer" in record.getMessage()
    ]


async def _cancel_pending_like_a_loop_teardown() -> bool:
    """Cancel every detached exchange, exactly as ``asyncio.run`` does on exit.

    ``asyncio.runners._cancel_all_tasks`` cancels everything the loop still has
    pending when ``amain()`` returns, and a detached exchange IS pending then —
    the connect that started it was cancelled seconds earlier and its await was
    dropped on purpose. Returns whether the registry held anything, so a test
    cannot mistake "nothing was cancelled" for "the line did not happen".
    """
    pending = [task for task in list(auth_mod._DETACHED_REFRESH_EXCHANGES) if not task.done()]
    for pending_task in pending:
        pending_task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    return bool(pending)


async def _seed_real_grant(
    store: Any,
    *,
    access: str = "access-0",
    refresh: str = "refresh-0",
    age_s: float = 600.0,
    lifetime_s: int = 60,
) -> McpTokenStorage:
    """Install an EXPIRED grant in a real ``AuthStore`` and return its storage.

    Written through ``McpTokenStorage`` (production's own funnel) and then
    re-upserted to backdate ``tokens_obtained_at``, because expiry is computed
    from that stamp: without it the proactive refresh returns before reaching the
    lock and the test would silently exercise nothing.
    """
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    from local_operator.mcp.auth import MCP_OAUTH_PROVIDER

    storage = McpTokenStorage(SERVER_URL, store)
    await storage.set_client_info(OAuthClientInformationFull(client_id=CLIENT_ID))
    await storage.set_tokens(
        OAuthToken(access_token=access, refresh_token=refresh, expires_in=lifetime_s)
    )
    rows = store.list_credentials(MCP_OAUTH_PROVIDER)
    assert rows, "the grant was not stored"
    payload = {k: v for k, v in rows[0].data.items() if k != "type"}
    payload["project_id"] = SERVER_URL
    payload["tokens_obtained_at"] = time.time() - age_s
    store.upsert_credential(MCP_OAUTH_PROVIDER, payload)
    return storage


async def _abandon_a_refresh_mid_post(
    store: Any, monkeypatch: pytest.MonkeyPatch, endpoint: FakeTokenEndpoint
) -> None:
    """Start a real refresh and cancel the connect once the server has rotated.

    The connect is cancelled rather than the exchange, so the exchange is
    DETACHED exactly as a session exit detaches it: it still owns the refresh
    lock, its response is still on the wire, and the rotation exists only in the
    answer it has not read yet.
    """
    _stub_discovery(monkeypatch, endpoint.token_endpoint)
    task = asyncio.ensure_future(auth_mod.ensure_mcp_oauth_fresh(SERVER_URL, _cfg(), store=store))
    await asyncio.wait_for(endpoint.rotation_applied.wait(), 5)
    assert endpoint.rotation_count == 1, "the exchange never reached the server"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_the_teardown_drain_persists_the_rotation_before_the_store_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """L1, both ways round, with the ORDER OF CLOSE AND DRAIN as the only difference.

    Arm 1 is the shipped order — drain first, then close — and asserts the
    outcome the operator actually feels: the row holds the ROTATED refresh token
    (``refresh-1``) and no send marker, so the next boot connects instead of
    refusing.

    Arm 2 is the pre-fix order — close first, then drain — and asserts the loss,
    because that is what makes arm 1 evidence rather than decoration: the drain
    had nothing to write into, the row kept the spent ``refresh-0``, and the
    marker stayed armed. Without this arm a green arm 1 could equally mean "the
    drain is unnecessary"; with it, the ordering is the load-bearing change.

    ``AuthStore`` is real, so the closed store is not simulated: SQLite raises and
    ``_write`` reports the drop, which is also asserted here (``_write``'s return
    value is the caller-visible half of the d3 fix).
    """
    from local_operator.providers.auth_store import AuthStore

    delay_s = 0.3

    # --- Arm 1: the shipped order (drain, then close) -----------------------
    store = AuthStore(tmp_path / "auth.db")
    storage = await _seed_real_grant(store)
    async with FakeTokenEndpoint(response_delay_s=delay_s) as endpoint:
        await _abandon_a_refresh_mid_post(store, monkeypatch, endpoint)
        drained = await auth_mod.drain_refresh_exchanges(2.0)
        assert drained is True, "the drain must see the detached exchange and wait for it"
        tokens = await storage.get_tokens()
        payload = storage._read() or {}
        store.close()

    assert tokens is not None
    assert tokens.refresh_token == "refresh-1", (
        "the rotation the server performed must be persisted while the store is "
        f"still open; the row holds {tokens.refresh_token!r}, a spent token"
    )
    assert tokens.access_token == "access-1"
    assert (
        auth_mod.GRANT_UNCONFIRMED_SEND_KEY not in payload
    ), "the exchange got its answer, so the write-ahead marker must be resolved"
    assert endpoint.reuse_attempts == 0

    # --- Arm 2: the pre-fix order (close, then drain) -----------------------
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="local_operator.mcp.auth"):
        doomed = AuthStore(tmp_path / "auth.db")
        await _seed_real_grant(doomed)
        async with FakeTokenEndpoint(response_delay_s=delay_s) as endpoint2:
            await _abandon_a_refresh_mid_post(doomed, monkeypatch, endpoint2)
            doomed.close()  # exactly what the dispose hook did before the fix
            await auth_mod.drain_refresh_exchanges(2.0)

    # A fresh handle, standing in for the next boot's process — the closed store
    # cannot be asked, which is the point of the arm rather than an inconvenience.
    reader = AuthStore(tmp_path / "auth.db")
    try:
        next_boot = await McpTokenStorage(SERVER_URL, reader).get_tokens()
        next_payload = McpTokenStorage(SERVER_URL, reader)._read() or {}
    finally:
        reader.close()

    assert next_boot is not None and next_boot.refresh_token == "refresh-0", (
        "closing the store first is what loses the rotation — this arm exists to "
        "prove the ordering, not to bless the outcome"
    )
    assert (
        auth_mod.GRANT_UNCONFIRMED_SEND_KEY in next_payload
    ), "and the marker stays armed, which is what makes the next connect refuse"
    assert any(
        "was NOT persisted" in record.getMessage() for record in caplog.records
    ), "a dropped rotation must say so at INFO rather than only at DEBUG"


@pytest.mark.asyncio
async def test_a_drain_that_has_not_yielded_never_reports_nothing_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R1-1/Q5: the empty verdict must not come from a snapshot taken before a yield.

    ``disconnect_all`` cancels its in-flight connects and reaches the drain with no
    ``await`` in between whenever ``self._connections`` is empty, so a cancellation
    it has delivered has not even been SCHEDULED when the drain starts. A first
    snapshot taken then sees an empty registry and returns ``True`` — read by every
    caller as "nothing in flight" — while the cancelled connect registers its
    detached exchange one tick later, and the caller closing the store on that
    verdict is the incident this PR exists to remove.

    The shape below is exactly that interleaving: rotate at the issuer, cancel the
    connect, call the drain with NO intervening await, then close the store on the
    verdict the way the dispose chain does. Before the fix the close landed before
    the exchange was registered, its write was swallowed, and the row kept the
    spent ``refresh-0`` with its marker armed; after it, the drain observes the
    exchange, waits for it, and the rotation is in the row when the store closes.
    """
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(tmp_path / "auth.db")
    await _seed_real_grant(store)
    async with FakeTokenEndpoint(response_delay_s=0.3) as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        task = asyncio.ensure_future(
            auth_mod.ensure_mcp_oauth_fresh(SERVER_URL, _cfg(), store=store)
        )
        await asyncio.wait_for(endpoint.rotation_applied.wait(), 5)
        assert endpoint.rotation_count == 1, "the exchange never reached the server"
        with caplog.at_level(logging.INFO, logger="local_operator.mcp.auth"):
            # No await between the cancel and the drain — the production shape.
            task.cancel()
            drained = await auth_mod.drain_refresh_exchanges(2.0)
            assert drained is True, "a completed drain must still report True"
            # Acting on the verdict, exactly as the dispose chain does: the store
            # closes on the strength of it.
            store.close()
        await asyncio.gather(task, return_exceptions=True)

    assert not [
        exchange for exchange in list(auth_mod._DETACHED_REFRESH_EXCHANGES) if not exchange.done()
    ], "the drain must have waited for the exchange, not returned before it registered"

    reader = AuthStore(tmp_path / "auth.db")
    try:
        storage = McpTokenStorage(SERVER_URL, reader)
        next_boot = await storage.get_tokens()
        next_payload = storage._read() or {}
    finally:
        reader.close()

    assert next_boot is not None and next_boot.refresh_token == "refresh-1", (
        "a True verdict from the drain means the rotation was persisted BEFORE the "
        "store closed; the row holds "
        f"{getattr(next_boot, 'refresh_token', None)!r} instead, which is the loss "
        "the empty first snapshot used to cause"
    )
    assert auth_mod.GRANT_UNCONFIRMED_SEND_KEY not in next_payload
    assert endpoint.reuse_attempts == 0


@pytest.mark.asyncio
async def test_disconnect_all_does_not_report_nothing_in_flight_from_a_pre_yield_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """R1-1, seen from the CALLER: the manager's own teardown is that shape.

    The test above drives the drain's contract; this one drives the caller the
    reviewer and QA reproduced with, because the empty first snapshot is a property
    of HOW ``disconnect_all`` reaches the drain, not of the drain alone:

    * a RECONNECT is in flight with its refresh POST already on the wire and the
      server already rotated — the failure the incident describes;
    * ``_reconnect`` tears the old connection down BEFORE it connects, so
      ``self._connections`` is empty (the "its connection already went" shape) and
      ``disconnect_all`` therefore has NO ``await`` between cancelling the
      reconnect and calling the drain;
    * so the cancellation it has just delivered has not even been scheduled when
      the drain begins.

    Pre-fix the drain read an empty registry, returned ``True`` — "nothing in
    flight" — and the teardown walked on past an exchange that registered one tick
    later. The assertion is that the caller cannot return while an exchange it
    cancelled is still unobserved, and that the rotation is in the row by then.
    """
    from local_operator.mcp.manager import McpManager
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(tmp_path / "auth.db")
    await _seed_real_grant(store)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    async with FakeTokenEndpoint(response_delay_s=0.3) as endpoint:
        _stub_discovery(monkeypatch, endpoint.token_endpoint)
        manager = McpManager(str(workspace), auth_store=store)
        manager._configs["dd"] = _cfg()
        manager._connect_futures["dd"] = asyncio.get_running_loop().create_future()
        manager._schedule_reconnect("dd")
        await asyncio.wait_for(endpoint.rotation_applied.wait(), 10)
        assert endpoint.rotation_count == 1, "the reconnect's exchange never reached the server"
        assert not manager._connections, (
            "this test is about the shape with no connection to await: the reconnect "
            "tore the old one down before it refreshed, so ``if names:`` gives the "
            "teardown no suspension point"
        )
        assert [
            task for task in manager._pending_reconnects.values() if not task.done()
        ], "the reconnect must still be the live task the teardown cancels"

        with caplog.at_level(logging.INFO, logger="local_operator.mcp.auth"):
            await manager.disconnect_all()

        # Read the row SYNCHRONOUSLY, with no await between the verdict and the
        # assertion. A read that yielded first would let the un-drained exchange
        # land its rotation behind it and hide exactly the defect: the drain said
        # "nothing in flight" and the teardown walked on with a live POST still
        # unwinding. (The registry is not the assertion here: with the exchange
        # not yet registered it is empty, so a "no live exchange" check passes
        # vacuously — the row is what the caller actually acts on.)
        # ``_read`` is the module's own SYNCHRONOUS reader (the row's shape is its
        # business, not this test's): what matters is that no await sits between
        # the dispose returning and this assertion.
        payload = McpTokenStorage(SERVER_URL, store)._read() or {}
        stored = payload.get("tokens") or {}
        assert stored.get("refresh_token") == "refresh-1", (
            "the rotation the server performed during the teardown must be in the row "
            f"by the time disconnect_all returns, not the spent token: {payload!r}"
        )
        assert auth_mod.GRANT_UNCONFIRMED_SEND_KEY not in stored, (
            "the exchange got its answer before the store could close under it, so "
            "the write-ahead marker must be resolved"
        )
        assert endpoint.reuse_attempts == 0
    store.close()


@pytest.mark.asyncio
async def test_a_detached_exchange_names_what_is_observable_about_its_request(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """L2/d2: the line must distinguish "handed to the client" from "never built".

    This is the question the 14-hour incident could not answer — the exchange's
    settle callback returned SILENTLY for a cancelled task, so a window with 36
    user-visible connect failures contained zero exchange-outcome lines.

    What the line may say is bounded by what can be OBSERVED, and the three arms
    below are that boundary rather than a claim about the wire:

    * **arm 1 — the request went out, the answer never arrived** (real endpoint,
      real POST): the marker stays armed, because a token that left the machine
      may have been spent;
    * **arm 2 — cancelled during the CONNECT phase** (a transport that blocks
      inside ``handle_async_request``, driven through the real
      ``httpx.AsyncClient``, its hooks and redirect handling included): the marker
      is ALREADY armed when the transport is entered, so the line reports a
      hand-off and this class is explicitly NOT removed by arming in the hook. An
      earlier revision asserted the opposite for this state with a stand-in client
      that never reached a transport — a belief rather than a measurement
      (reviewer round 1, R1-2) — which is why the stand-in is gone and the real
      client is here. The FAIL case is stated rather than asserted: httpx exposes
      no seam that reports "the bytes are on the wire", so a cancellation between
      the hook and the socket cannot be told from one inside it, and this test
      does not pretend otherwise;
    * **arm 3 — cancelled BEFORE any request was built** (held at the exchange's
      own pre-flight store read, so the client is never handed anything): nothing
      is armed and the line says so. That direction IS conclusive, and it is the
      one the refusal policy needs.
    """
    import httpx

    # --- Arm 1: the request went out, the answer never arrived --------------
    store = FakeAuthStore()
    storage = await _seed_expired_grant(store)

    async with FakeTokenEndpoint(response_delay_s=0.5) as endpoint:
        with caplog.at_level(logging.INFO, logger="local_operator.mcp.auth"):
            task = asyncio.ensure_future(
                auth_mod._refresh_oauth_token_locked(
                    SERVER_URL, storage, _endpoints_for(endpoint.token_endpoint)
                )
            )
            await asyncio.wait_for(endpoint.rotation_applied.wait(), 5)
            # The connect goes first, which DETACHES the exchange — it is
            # shielded, so it keeps running with the response still on the wire.
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert (
                await _cancel_pending_like_a_loop_teardown()
            ), "the exchange must be registered as detached: nothing else holds it"
            assert await _until(
                lambda: _cancellation_was_logged(caplog)
            ), "a cancelled detached exchange must log before the loop moves on"

    handed_over = _cancellation_lines(caplog)
    assert handed_over, "a cancelled detached exchange must not be silent any more"
    assert "its request had already entered the sending pipeline" in handed_over[0], handed_over[0]
    assert auth_mod.GRANT_UNCONFIRMED_SEND_KEY in (storage._read() or {}), (
        "a request that reached the wire may have spent the token, so the marker "
        "must stay armed — this fix changes what is LOGGED, never the refusal"
    )

    # --- Arm 2: cancelled inside the transport, before it wrote anything -----
    # The real client, the real request hooks, a transport that never writes.
    caplog.clear()
    connect_store = FakeAuthStore()
    connect_storage = await _seed_expired_grant(connect_store)
    reached_the_transport = asyncio.Event()
    #: What the row said AT the instant the transport was entered: this is the
    #: measurement the old comment got wrong — the hook runs BEFORE it.
    armed_when_the_transport_was_entered: list[bool] = []

    async def _blocking_transport(request: httpx.Request) -> httpx.Response:
        armed_when_the_transport_was_entered.append(
            auth_mod.GRANT_UNCONFIRMED_SEND_KEY in (connect_storage._read() or {})
        )
        reached_the_transport.set()
        await asyncio.Event().wait()  # no bytes, no answer: a connect-phase hold
        raise AssertionError("a cancelled connect must never be answered")

    class _RealClientWithABlockingTransport(httpx.AsyncClient):
        """httpx's own client whose transport blocks before writing a byte.

        A SUBCLASS, not a stand-in: what is under test is where httpx runs the
        request hook relative to the transport, and only the real implementation
        can answer that. ``MockTransport`` is httpx's own transport seam, so the
        handler below runs exactly where ``handle_async_request`` does — after
        the hook, before anything is written.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.MockTransport(_blocking_transport)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _RealClientWithABlockingTransport)
    with caplog.at_level(logging.INFO, logger="local_operator.mcp.auth"):
        task = asyncio.ensure_future(
            auth_mod._refresh_oauth_token_locked(
                SERVER_URL, connect_storage, _endpoints_for("http://127.0.0.1:9/token")
            )
        )
        await asyncio.wait_for(reached_the_transport.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await _cancel_pending_like_a_loop_teardown()
        assert await _until(lambda: _cancellation_was_logged(caplog))
    monkeypatch.undo()

    assert armed_when_the_transport_was_entered == [True], (
        "the request hook arms the marker BEFORE the transport is entered, so a "
        "cancellation during the connect phase still quarantines the grant: that "
        "class is NOT removed by arming in the hook, and the test that used to "
        "assert it was asserting a stand-in"
    )
    assert auth_mod.GRANT_UNCONFIRMED_SEND_KEY in (connect_storage._read() or {}), (
        "the connect phase is not spared: an armed marker here is the conservative "
        "direction and is the honest state of this change"
    )
    cancelled_in_the_connect = _cancellation_lines(caplog)
    assert cancelled_in_the_connect and (
        "had already entered the sending pipeline" in cancelled_in_the_connect[0]
    ), cancelled_in_the_connect
    assert "not evidence it reached the wire" in cancelled_in_the_connect[0], (
        "the line must not claim the request reached the wire: " + cancelled_in_the_connect[0]
    )

    # --- Arm 3: cancelled before any request was built -----------------------
    # The other half of the flag, and the half that IS conclusive. The exchange is
    # held at its own pre-flight store read — a real await it must pass before it
    # can build a request — with the real client (arm 2's patch is undone) never
    # handed anything, so no marker must exist and the line must say `never`.
    caplog.clear()
    unbuilt_store = FakeAuthStore()
    unbuilt_storage = await _seed_expired_grant(unbuilt_store)
    reached_the_preflight = asyncio.Event()
    release_the_preflight = asyncio.Event()
    held_get_tokens = unbuilt_storage.get_tokens

    async def _get_tokens_after_the_gate() -> Any:
        reached_the_preflight.set()
        await release_the_preflight.wait()
        return await held_get_tokens()

    monkeypatch.setattr(unbuilt_storage, "get_tokens", _get_tokens_after_the_gate)
    with caplog.at_level(logging.INFO, logger="local_operator.mcp.auth"):
        task = asyncio.ensure_future(
            auth_mod._refresh_oauth_token_locked(
                SERVER_URL, unbuilt_storage, _endpoints_for("http://127.0.0.1:9/token")
            )
        )
        await asyncio.wait_for(reached_the_preflight.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await _cancel_pending_like_a_loop_teardown()
        assert await _until(lambda: _cancellation_was_logged(caplog))
    monkeypatch.undo()

    never_built = _cancellation_lines(caplog)
    assert never_built and "never entered the sending pipeline" in never_built[0], never_built
    assert auth_mod.GRANT_UNCONFIRMED_SEND_KEY not in (unbuilt_storage._read() or {}), (
        "a request that was never built must leave the grant untouched rather than "
        "quarantined for an hour — that is the half of the flag that is conclusive"
    )


def test_a_registry_entry_left_by_a_closed_loop_does_not_poison_a_later_drain(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """R1-4: an entry whose loop closed can never run, so it must not be waited on.

    The registry is process-wide and entries normally leave it from their own done
    callback. A loop that closes WITHOUT cancelling what it had pending — the
    hand-driven ones in ``session/runtime``, and any future one — leaves its entry
    behind forever, and every later drain in the process then pays the FULL bound
    and logs a loss that no exchange is carrying. That false loss line lands in the
    one log support reads to COUNT real ones, which is why the entry is dropped (at
    the only point it can be recognised: its own callback will never run) instead
    of being counted.

    A SYNC test on purpose: the shape needs a second loop, one that really ran the
    task and then closed with it still pending, while the drain runs on this
    thread's own loop.

    The bound is the discriminator, not a stopwatch: at 0.2s an undiscarded entry
    makes this drain time out and return ``False`` with an overrun line, and the
    discard makes it return ``True`` immediately with none.
    """
    stale_loop = asyncio.new_event_loop()
    run_until_abandoned = asyncio.Event()

    # ``Any`` rather than ``RefreshOutcome``: the task is abandoned before it can
    # return one, and the promise the registry makes is about the tasks that DO
    # run. A coroutine the loop actually started, so this is the shape a loop that
    # goes down without cancelling leaves behind, not an unstarted stub.
    async def _parked_forever() -> Any:
        await run_until_abandoned.wait()

    try:
        stale_task = stale_loop.create_task(_parked_forever())
        stale_loop.run_until_complete(asyncio.sleep(0))
        assert not stale_task.done(), "the abandoned task must still be pending"
        auth_mod._detach_refresh_exchange(stale_task, "https://stale.test/mcp")
    finally:
        # Closed with its task still pending: exactly what a hand-driven loop does
        # when it goes down without cancelling.
        stale_loop.close()
    _ABANDONED_TASKS.append(stale_task)

    assert stale_task in auth_mod._DETACHED_REFRESH_EXCHANGES
    assert stale_task.get_loop().is_closed(), "the entry's loop must really be closed"

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="local_operator.mcp.auth"):
        drained = asyncio.run(auth_mod.drain_refresh_exchanges(0.2))

    assert drained is True, (
        "a task abandoned by a closed loop can never complete, so waiting it out "
        "would be a false loss on every later teardown"
    )
    assert stale_task not in auth_mod._DETACHED_REFRESH_EXCHANGES
    assert not any(
        "outlived the" in record.getMessage() for record in caplog.records
    ), "an abandoned entry must not be reported as a rotation lost at exit"
    assert any(
        "abandoned by a closed event loop" in record.getMessage() for record in caplog.records
    ), "dropping an entry must say why, so a real one is not mistaken for noise"


@pytest.mark.asyncio
async def test_the_leaving_gate_keeps_a_teardown_refresh_off_the_wire(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A session on its way out must not be the party that spends the token.

    This is the companion the store-reordering requires rather than a nicety:
    the SDK runs the SAME auth flow for its session-terminate DELETE, so once the
    credential store outlives the teardown, an ungated teardown would POST a
    refresh token from a process that has already given up on persisting the
    answer. Today that path only fails closed because the store is shut, which is
    the defect this PR removes — so removing it without this gate would trade a
    lost rotation for a POST from a dying process.

    Driven the way httpx drives it for that DELETE: an ``AsyncClient`` with the
    provider as its ``auth``, one request, an expired stored grant. The ONLY
    difference between the two arms is the predicate, which is what makes the
    second arm evidence that the first is not vacuous: with it, zero token POSTs
    reach the authorization server; without it, one does.
    """
    import contextlib

    from mcp.client.streamable_http import create_mcp_http_client

    from local_operator.mcp.auth import build_oauth_provider

    async def _token_posts_for(leaving: Any) -> list[dict[str, str]]:
        store = FakeAuthStore()
        await _seed_expired_grant(store)
        async with FakeTokenEndpoint() as endpoint:
            provider = build_oauth_provider(
                SERVER_URL,
                _cfg(),
                store=store,
                interactive=False,
                endpoints=_endpoints_for(endpoint.token_endpoint),
                leaving=leaving,
            )
            # The SAME client factory the manager wires a real connection with,
            # so the flow is driven by the transport that runs the terminate
            # DELETE rather than by a look-alike.
            client = create_mcp_http_client(auth=provider)
            try:
                with contextlib.suppress(Exception):
                    await client.request("DELETE", endpoint.token_endpoint)
            finally:
                await client.aclose()
            return [form for form in endpoint.requests if form.get("grant_type") == "refresh_token"]

    with caplog.at_level(logging.INFO, logger="local_operator.mcp.auth"):
        suppressed = await _token_posts_for(lambda: True)
        allowed = await _token_posts_for(None)

    assert suppressed == [], "a leaving session must make no token POST at all"
    assert any(
        "refresh suppressed" in record.getMessage() and "tearing down" in record.getMessage()
        for record in caplog.records
    ), "the suppression must be readable in the log, not inferred from silence"
    assert len(allowed) == 1, (
        "the same flow WITHOUT the gate must still post — otherwise this test "
        "would pass for any reason at all"
    )
