"""Real-transport end-to-end proof of refresh durability across sessions.

``tests/unit/mcp/test_oauth_refresh_durability.py`` reproduces each defect
against the real machinery in isolation. This stage is the other half: the REAL
MCP streamable-HTTP transport, the REAL ``OAuthClientProvider`` wiring and the
REAL credential store, against a LOCAL authorization server that behaves like
Notion — it rotates the refresh token on every exchange, invalidates the access
token that went with it, and runs REFRESH-TOKEN REUSE DETECTION (presenting
anything other than the current refresh token returns ``invalid_grant`` AND
revokes the whole family).

That last property is what makes the defect fleet-wide rather than local, so the
two claims here are stated in the terms the incident was described in:

1. a session whose refresh is CANCELLED mid-flight (every sidebar switch
   disposes a manager and cancels its in-flight connects) must not lose the
   rotation that exchange performed — the store must hold the NEW refresh
   token, and the next session must be able to connect WITHOUT posting a spent
   one and killing the family;
2. a session whose access token the server has revoked underneath it (a
   sibling's rotation) must recover from its 401 by refreshing UNDER THE LOCK,
   not by handing the 401 to the SDK's authorization branch.

Before the corresponding fixes, (1) leaves the spent token stored and the next
session's refresh POST revokes the family — ``"MCP authorization failed; run
/mcp reauth"`` for every later session — and (2) ends in a transport-mangled
``CancelledError`` ("Cancelled via cancel scope") with no token POST and no
recovery: the 401 is handed to the SDK's authorization branch, whose browser
refusal the anyio cancel scope swallows, so the session neither heals nor
reports an actionable auth requirement. Both are asserted as outcomes here
rather than as call counts, so the stage fails if the recovery merely looks
plausible.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qsl

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig
from local_operator.mcp.manager import McpManager
from local_operator.providers.auth_store import AuthStore
from tests.e2e.test_desktop_radient import serve

pytestmark = pytest.mark.e2e


class RotatingIssuer:
    """A local authorization server AND MCP endpoint that rotates and revokes.

    Two behaviours, both taken from Notion's observed behaviour rather than from
    RFC 6749 (which permits, but does not require, either): every successful
    refresh issues a NEW refresh token and invalidates the access token that
    went with the old one, and a refresh presenting anything but the current
    refresh token is treated as a REPLAY — ``invalid_grant`` plus revocation of
    the entire family, so every later exchange fails too.

    ``post_delay_s`` holds the RESPONSE back after the rotation has been applied,
    which is the window these tests cancel inside.
    """

    def __init__(self) -> None:
        self.origin = ""
        self.access_token = "access-0"
        self.refresh_token = "refresh-0"
        self.revoked = False
        self.rotations = 0
        self.token_posts = 0
        self.reuse_attempts = 0
        self.post_delay_s = 0.0
        #: Lifetime of the access token this issuer hands out. Short tokens are
        #: how a client ends up holding a token that dies between connecting and
        #: disconnecting, which is the state in which the SDK's session-terminate
        #: DELETE enters the auth flow — the path the teardown gate exists for.
        self.access_lifetime_s = 3600
        self.rotated = asyncio.Event()
        self.app = FastAPI()
        self._wire()

    def rotate_now(self) -> None:
        """Rotate without a client request: what a sibling process does."""
        self._rotate()

    def revoke_access_tokens(self) -> None:
        """Invalidate the access tokens issued so far, WITHOUT spending the
        refresh token.

        The state a 401-recovery exists for: our stored refresh token is still
        the current one, so nothing looks wrong in memory, but the bearer token
        every session is carrying has stopped being accepted. A sibling's
        rotation does this as a side effect; a server can also do it on its own.
        """
        self.access_token = f"revoked-{self.rotations}"

    def _rotate(self) -> None:
        self.rotations += 1
        self.access_token = f"access-{self.rotations}"
        self.refresh_token = f"refresh-{self.rotations}"
        self.rotated.set()

    def _wire(self) -> None:
        app = self.app

        @app.get("/.well-known/oauth-protected-resource")
        @app.get("/.well-known/oauth-protected-resource/mcp")
        async def protected_resource() -> dict[str, Any]:
            return {"resource": self.origin + "/mcp", "authorization_servers": [self.origin]}

        @app.get("/.well-known/oauth-authorization-server")
        async def authorization_server() -> dict[str, Any]:
            return {
                "issuer": self.origin,
                "authorization_endpoint": self.origin + "/authorize",
                "token_endpoint": self.origin + "/token",
                "registration_endpoint": self.origin + "/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": ["none"],
            }

        @app.post("/register")
        async def register(request: Request) -> JSONResponse:
            body = await request.json()
            return JSONResponse(
                {**body, "client_id": "e2e-rotation", "token_endpoint_auth_method": "none"},
                status_code=201,
            )

        @app.post("/token")
        async def token(request: Request) -> JSONResponse:
            self.token_posts += 1
            form = dict(parse_qsl((await request.body()).decode("utf-8")))
            if form.get("grant_type") != "refresh_token":
                return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)
            if form.get("refresh_token") != self.refresh_token:
                self.reuse_attempts += 1
                self.revoked = True
                return JSONResponse({"error": "invalid_grant"}, status_code=400)
            self._rotate()
            if self.post_delay_s:
                # AFTER the rotation: a client that walks away now has spent the
                # token it presented and has no way to learn the new one.
                await asyncio.sleep(self.post_delay_s)
            return JSONResponse(
                {
                    "access_token": self.access_token,
                    "token_type": "Bearer",
                    "expires_in": self.access_lifetime_s,
                    "refresh_token": self.refresh_token,
                    "scope": "fixture",
                }
            )

        @app.api_route("/mcp", methods=["GET", "POST", "DELETE"])
        async def mcp(request: Request) -> Response:
            if request.headers.get("authorization") != "Bearer " + self.access_token:
                return JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                    headers={
                        "WWW-Authenticate": (
                            f'Bearer resource_metadata="'
                            f'{self.origin}/.well-known/oauth-protected-resource"'
                        )
                    },
                )
            if request.method != "POST":
                return Response(status_code=405)
            body = await request.json()
            if "id" not in body:
                return Response(status_code=202)
            method = body.get("method")
            if method == "initialize":
                result: dict[str, Any] = {
                    "protocolVersion": body.get("params", {}).get("protocolVersion", "2025-03-26"),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "rotation-fixture", "version": "1"},
                }
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "fixture",
                            "description": "Fixture tool",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    ]
                }
            elif method == "tools/call":
                result = {"content": [{"type": "text", "text": "ok"}], "isError": False}
            else:
                result = {}
            return JSONResponse({"jsonrpc": "2.0", "id": body["id"], "result": result})


async def _seed_grant(
    store: AuthStore,
    url: str,
    *,
    access: str,
    refresh: str,
    age_s: float = 0.0,
    lifetime_s: int = 3600,
) -> None:
    """Install a stored grant, optionally aged past its lifetime.

    Goes through ``McpTokenStorage`` for the write — the same funnel production
    uses — and re-upserts the row to backdate ``tokens_obtained_at``, because
    expiry is computed from that stamp and a fresh stamp would let the
    proactive refresh skip the exchange these tests are about.
    """
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    from local_operator.mcp.auth import MCP_OAUTH_PROVIDER, McpTokenStorage

    storage = McpTokenStorage(url, store)
    await storage.set_client_info(OAuthClientInformationFull(client_id="e2e-rotation"))
    await storage.set_tokens(
        OAuthToken(access_token=access, refresh_token=refresh, expires_in=lifetime_s)
    )
    if not age_s:
        return
    rows = store.list_credentials(MCP_OAUTH_PROVIDER)
    assert rows, "the grant was not stored"
    payload = {k: v for k, v in rows[0].data.items() if k != "type"}
    payload["project_id"] = url
    payload["tokens_obtained_at"] = time.time() - age_s
    store.upsert_credential(MCP_OAUTH_PROVIDER, payload)


def _stored_tokens(store: AuthStore, url: str) -> dict[str, Any]:
    from local_operator.mcp.auth import MCP_OAUTH_PROVIDER

    for row in store.list_credentials(MCP_OAUTH_PROVIDER):
        if row.data.get("project_id") == url or row.data.get("identity_key") == url:
            return dict(row.data.get("tokens") or {})
    return {}


@pytest.mark.asyncio
async def test_a_cancelled_refresh_keeps_its_rotation_and_the_next_session_connects(
    headless_tui_env, tmp_path, monkeypatch
) -> None:
    """One cancelled refresh must not cost the fleet its grant.

    Phase 1 is the routine teardown: a connect is cancelled while its refresh
    POST is in flight, and the server has ALREADY rotated. Phase 2 is the next
    session, which is where the damage shows — if the store kept the spent
    token, its proactive refresh posts that token, the server treats it as a
    replay, and the family dies (``auth-required`` for every later session).

    So the assertions are the two outcomes a user would feel: the store holds
    the NEW refresh token, and the later session connects with ZERO token POSTs
    and zero reuse attempts.
    """
    store = AuthStore(headless_tui_env / "auth.db")
    issuer = RotatingIssuer()
    try:
        async with serve(issuer.app) as origin:
            issuer.origin = origin
            url = origin + "/mcp"
            cfg = MCPHttpServerConfig(url=url, auth=MCPAuthConfig(type="oauth"))
            workspace = tmp_path / "ws"
            workspace.mkdir()
            await _seed_grant(
                store,
                url,
                access="access-0",
                refresh="refresh-0",
                age_s=600,
                lifetime_s=60,
            )

            # --- Phase 1: cancel the connect while its refresh POST is in
            # flight, after the server has applied the rotation.
            issuer.post_delay_s = 0.6
            manager = McpManager(str(workspace), auth_store=store)
            task = asyncio.ensure_future(manager._connect_server("dd", cfg))
            await asyncio.wait_for(issuer.rotated.wait(), 10)
            assert issuer.rotations == 1, "the exchange never reached the server"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            tokens = _stored_tokens(store, url)
            # The cancellation is delivered IMMEDIATELY now — teardown no longer
            # waits out the refresh budget — and the rotation lands a moment
            # later, persisted by the detached exchange, which still owns the
            # refresh lock until it has. Polling is part of the contract rather
            # than a concession: nothing else can recover that token.
            deadline = time.monotonic() + 5
            while tokens.get("refresh_token") != "refresh-1" and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
                tokens = _stored_tokens(store, url)
            assert tokens.get("refresh_token") == "refresh-1", (
                "the cancelled exchange lost the server's rotation; the store is "
                f"holding a spent token: {tokens!r}"
            )
            assert issuer.reuse_attempts == 0

            # --- Phase 2: a later session. It must connect from the rotated
            # grant without posting anything, and without a reuse detection.
            issuer.post_delay_s = 0.0
            posts_before = issuer.token_posts
            later = McpManager(str(workspace), auth_store=store)
            conn = await later._connect_server("dd", cfg)
            assert conn is not None
            assert conn.tools, "the session connected without tools"
            assert issuer.token_posts == posts_before, (
                "a later session re-posted a refresh token for an already-rotated " "grant"
            )
            assert issuer.reuse_attempts == 0, (
                "the stale refresh token reached the server: the family is revoked "
                "and every later session needs a human re-grant"
            )
            assert isinstance(conn.config, MCPHttpServerConfig)
            assert conn.config.url == url
    finally:
        store.close()


@pytest.mark.asyncio
async def test_a_401_on_a_revoked_token_recovers_by_refreshing_under_the_lock(
    headless_tui_env, tmp_path
) -> None:
    """A 401 must be recovered by a locked refresh, not by a browser grant.

    The state: our access token is still VALID locally (unexpired), and it is
    the same one the shared store holds — so there is no peer rotation to adopt.
    The server has simply stopped accepting it, so the request comes back 401.
    The refresh token we hold is still the current one, so the correct recovery
    is to spend it under the refresh lock and retry.

    Handing that 401 to the SDK instead ends in the SDK's authorization branch,
    which a non-interactive connect turns into ``McpAuthRequiredError`` — an
    actionable-looking error telling the user to run ``/mcp reauth`` for a grant
    that only needed a refresh. So the assertion is the user-visible outcome:
    the tool call succeeds, the grant is reported connected, and the store holds
    the ROTATED token.
    """
    store = AuthStore(headless_tui_env / "auth.db")
    issuer = RotatingIssuer()
    try:
        async with serve(issuer.app) as origin:
            issuer.origin = origin
            url = origin + "/mcp"
            cfg = MCPHttpServerConfig(url=url, auth=MCPAuthConfig(type="oauth"))
            workspace = tmp_path / "ws"
            workspace.mkdir()
            # Unexpired on purpose: this token is valid in memory and revoked
            # server-side, which is exactly the state a 401-recovery exists for.
            await _seed_grant(store, url, access="access-0", refresh="refresh-0")

            manager = McpManager(str(workspace), auth_store=store)
            conn = await manager._connect_server("dd", cfg)
            assert (
                issuer.token_posts == 0
            ), "a fresh, unexpired grant must connect without any token POST"

            # The access tokens this session is carrying stop being accepted,
            # while the stored refresh token stays the current one.
            issuer.revoke_access_tokens()
            posts_before = issuer.token_posts

            result = await conn.live_session.call_tool("fixture", {}, read_timeout_seconds=30)
            assert result.is_error is False
            assert (
                issuer.token_posts == posts_before + 1
            ), "the 401 did not trigger the locked refresh"
            assert issuer.reuse_attempts == 0
            tokens = _stored_tokens(store, url)
            assert (
                tokens.get("refresh_token") == "refresh-1"
            ), f"the rotated grant was not persisted: {tokens!r}"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# The session's own exit: the credential store must outlive the teardown that
# still needs it, and the teardown must not be the party that spends a grant.
# ---------------------------------------------------------------------------
#
# Everything above cancels a connect and lets the exchange finish while the
# process lives on. These tests are about the path the operator actually feels —
# a runtime boot whose IDLE EXIT lands while a refresh POST is on the wire — and
# they drive the real dispose hooks the composition root registers rather than
# calling the drain by hand, because the ORDER of those hooks was the defect: the
# store's close was registered during ``create_session`` and ``manager.disconnect_all``
# only later, so the store closed first, the rotation write was swallowed at
# DEBUG, ``store_refresh_result`` still said it had succeeded, and the row kept the
# SPENT refresh token with a live ``grant_refresh_unconfirmed`` marker. Every
# later connect then refused to refresh for up to an hour and told the user to run
# ``/mcp reauth``.


class _DisposingSession:
    """The session surface the dispose hooks need, with ``Session``'s contract.

    Two lists — ordinary hooks, then late ones — run in exactly that order, which
    is what ``local_operator.session.session.Session.dispose`` does (and
    ``tests/unit/test_session_factory.py`` pins the real class against that
    shape, including the registration this file relies on). A double is honest
    here because the claim is about the ORDER the composition root's hooks ran
    in, not about Session's own teardown of a conversation, a transcript and a job
    group; the hooks themselves are the production ``attach_auth_dispose`` and
    ``attach_mcp_dispose``.
    """

    def __init__(self) -> None:
        self._hooks: list[Any] = []
        self._final_hooks: list[Any] = []
        self.mcp_manager: Any = None
        # ``attach_mcp_dispose`` installs these two as the manager's sinks; the
        # real Session owns them, so the double has to exist with them.
        self._on_mcp_incident = lambda *args, **kwargs: None
        self._on_mcp_recovery = lambda *args, **kwargs: None

    def add_dispose_hook(self, hook: Any, *, last: bool = False) -> None:
        (self._final_hooks if last else self._hooks).append(hook)

    async def dispose(self) -> None:
        for hook in [*self._hooks, *self._final_hooks]:
            outcome = hook()
            if inspect.isawaitable(outcome):
                await outcome


def _read_stored_tokens(db_path: Any, url: str) -> dict[str, Any]:
    """Read one server's stored grant through a FRESH handle over the file.

    What the next runtime process does, and the only way to look at the row after
    a dispose that closed the session's store (the store closing is part of the
    behaviour under test, so the test cannot hold it open to peek).
    """
    reader = AuthStore(db_path)
    try:
        return _stored_tokens(reader, url)
    finally:
        reader.close()


def _dispose_via_the_real_hooks(store: AuthStore, manager: McpManager) -> _DisposingSession:
    """Register the production dispose hooks for ``session.dispose()`` to run."""
    from local_operator.session_factory import attach_auth_dispose, attach_mcp_dispose

    session = _DisposingSession()
    # Cast at the seam: the double mirrors the two registration methods the
    # composition root calls, and ``Session`` itself is what the hooks are typed
    # against. The contract it mirrors is pinned against the real class in
    # ``tests/unit/test_session_factory.py``.
    attach_auth_dispose(cast("Session", session), store)
    attach_mcp_dispose(cast("Session", session), manager)
    return session


@pytest.mark.asyncio
async def test_a_refresh_in_flight_at_idle_exit_survives_to_the_successor_boot(
    headless_tui_env, tmp_path, caplog
) -> None:
    """The path the operator feels: boot, idle-exit mid-refresh, successor boot.

    The exit lands while the refresh POST is on the wire and the server has
    already rotated, which is the shape measured on this host (132 idle-exits in
    14.6 h, one boot every ~5.3 minutes). The assertions are the two outcomes a
    user would notice: the successor connects with NO token POST and no reuse
    attempt — i.e. no refusal, no ``/mcp reauth`` — and it does so off the
    rotation persisted by the session that died.

    The successor is a NEW ``AuthStore`` over the same file, which is what a new
    runtime process is: the boundary that matters is the file, not the object.
    """
    from local_operator.mcp.auth import GRANT_UNCONFIRMED_SEND_KEY

    store = AuthStore(headless_tui_env / "auth.db")
    issuer = RotatingIssuer()
    try:
        async with serve(issuer.app) as origin:
            issuer.origin = origin
            url = origin + "/mcp"
            cfg = MCPHttpServerConfig(url=url, auth=MCPAuthConfig(type="oauth"))
            workspace = tmp_path / "ws"
            workspace.mkdir()
            await _seed_grant(
                store, url, access="access-0", refresh="refresh-0", age_s=600, lifetime_s=60
            )

            # --- boot 1: the refresh is in flight when the exit lands -------
            issuer.post_delay_s = 0.6
            manager = McpManager(str(workspace), auth_store=store)
            task = asyncio.ensure_future(manager._connect_server("dd", cfg))
            await asyncio.wait_for(issuer.rotated.wait(), 10)
            assert issuer.rotations == 1, "the exchange never reached the server"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            with caplog.at_level(logging.INFO, logger="local_operator.mcp.auth"):
                session = _dispose_via_the_real_hooks(store, manager)
                await session.dispose()

            # Read through a FRESH handle over the same file: the session's store
            # is closed by the time dispose returns, and this is how the next
            # process sees the row.
            tokens = _read_stored_tokens(headless_tui_env / "auth.db", url)
            assert tokens.get("refresh_token") == "refresh-1", (
                "the rotation the server performed during the exit must be in the "
                f"row, not the spent token the session presented: {tokens!r}"
            )
            assert GRANT_UNCONFIRMED_SEND_KEY not in tokens, (
                "the exchange got its answer before the store closed, so the "
                "write-ahead marker must be resolved — an armed marker is what "
                "makes the next boot refuse for up to an hour"
            )

            # --- boot 2: a fresh process over the same credential file -----
            issuer.post_delay_s = 0.0
            successor_store = AuthStore(headless_tui_env / "auth.db")
            try:
                posts_before = issuer.token_posts
                successor = McpManager(str(workspace), auth_store=successor_store)
                conn = await successor._connect_server("dd", cfg)
                assert conn.tools, "the successor connected without tools"
                assert issuer.token_posts == posts_before, (
                    "the successor re-posted a refresh token for a grant that was "
                    "already rotated, which is the reuse-detection POST"
                )
                assert issuer.reuse_attempts == 0
                await successor.disconnect_all()
            finally:
                successor_store.close()

        assert not any(
            "never acknowledged" in record.getMessage() or "reauth" in record.getMessage()
            for record in caplog.records
        ), "the successor must connect without a refusal and without telling the user to re-auth"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_a_drain_that_runs_out_of_time_reports_the_loss_it_could_not_stop(
    headless_tui_env, tmp_path, monkeypatch, caplog
) -> None:
    """When the bound is hit, the loss stops being silent — that is the d2 half.

    The drain is a CUT, so an authorization server slow enough can outlive it and
    the rotation really is lost. What must not happen is losing it quietly: the
    measured incident was 36 user-visible connect failures across 14.4 h with
    ZERO exchange-outcome lines in the log, because the settle callback returned
    silently for a cancelled exchange and the failed write sat at DEBUG. Both new
    lines are asserted here — the bound being hit, and the per-exchange line that
    says whether httpx was handed the request (the question the incident could
    not answer, and the one any decision about the send path turns on — see
    ``RefreshSendState`` for why the line names the hand-off rather than the
    wire).
    """
    auth_mod = pytest.importorskip("local_operator.mcp.auth")

    store = AuthStore(headless_tui_env / "auth.db")
    issuer = RotatingIssuer()
    try:
        async with serve(issuer.app) as origin:
            issuer.origin = origin
            url = origin + "/mcp"
            cfg = MCPHttpServerConfig(url=url, auth=MCPAuthConfig(type="oauth"))
            workspace = tmp_path / "ws"
            workspace.mkdir()
            await _seed_grant(
                store, url, access="access-0", refresh="refresh-0", age_s=600, lifetime_s=60
            )
            # The bound is patched down instead of the server being held for the
            # default couple of seconds: the behaviour under test is the same,
            # because the module reads the constant at call time.
            monkeypatch.setattr(auth_mod, "REFRESH_DRAIN_TIMEOUT_S", 0.2)
            issuer.post_delay_s = 3.0

            manager = McpManager(str(workspace), auth_store=store)
            task = asyncio.ensure_future(manager._connect_server("dd", cfg))
            await asyncio.wait_for(issuer.rotated.wait(), 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            with caplog.at_level(logging.INFO, logger="local_operator.mcp.auth"):
                session = _dispose_via_the_real_hooks(store, manager)
                await session.dispose()
                held_the_bound = any(
                    "outlived the 0.2s teardown drain" in record.getMessage()
                    for record in caplog.records
                )
                assert held_the_bound, "giving up on the bound must say so"

                # ...and then the loop goes down, cancelling what is left, which
                # is what ``asyncio.run(amain())`` does to every pending task.
                pending = [
                    exchange
                    for exchange in list(auth_mod._DETACHED_REFRESH_EXCHANGES)
                    if not exchange.done()
                ]
                assert pending, "the exchange must still be registered, or nothing was cancelled"
                for exchange in pending:
                    exchange.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                assert any(
                    "CANCELLED before any answer" in record.getMessage()
                    and "had already entered the sending pipeline" in record.getMessage()
                    for record in caplog.records
                ), "a lost exchange must name what is known about its request: that is d2"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_attaching_boots_post_at_most_once(headless_tui_env, tmp_path) -> None:
    """The attach rate must not regress: this PR changes WHO posts, not HOW OFTEN.

    Three successive boots against one store: with a fresh stored token none of
    them may post, and with an expired one exactly ONE may — the other two see
    the rotation the first persisted and spend nothing. Pinned because the fix
    touches the arm/dispatch path every exchange goes through, and a change that
    quietly turned one boot into three POSTs would be a family-revocation risk
    with no other symptom.
    """
    store = AuthStore(headless_tui_env / "auth.db")
    issuer = RotatingIssuer()
    try:
        async with serve(issuer.app) as origin:
            issuer.origin = origin
            url = origin + "/mcp"
            cfg = MCPHttpServerConfig(url=url, auth=MCPAuthConfig(type="oauth"))
            workspace = tmp_path / "ws"
            workspace.mkdir()

            async def _one_boot() -> McpManager:
                manager = McpManager(str(workspace), auth_store=store)
                conn = await manager._connect_server("dd", cfg)
                assert conn.tools, "the boot connected without tools"
                return manager

            # --- fresh grant: no boot may post ------------------------------
            await _seed_grant(
                store,
                url,
                access="access-0",
                refresh="refresh-0",
                age_s=1.0,
                lifetime_s=3600,
            )
            posts_before = issuer.token_posts
            for _ in range(3):
                manager = await _one_boot()
                await manager.disconnect_all()
            assert (
                issuer.token_posts == posts_before == 0
            ), "a fresh stored token must connect three times without a single POST"

            # --- expired grant: exactly one of the three posts --------------
            await _seed_grant(
                store,
                url,
                access=issuer.access_token,
                refresh=issuer.refresh_token,
                # Aged PAST a 3600 s lifetime: a merely old token would still be
                # inside REFRESH_SKEW_S and no refresh would run at all.
                age_s=4000,
                lifetime_s=3600,
            )
            posts_before = issuer.token_posts
            for _ in range(3):
                manager = await _one_boot()
                await manager.disconnect_all()
            assert issuer.token_posts == posts_before + 1, (
                "an expired grant must collapse three attaching boots into ONE "
                "rotation, which is what the under-lock peer-freshness re-read buys"
            )
            assert issuer.reuse_attempts == 0
    finally:
        store.close()


if TYPE_CHECKING:
    from local_operator.session.session import Session
