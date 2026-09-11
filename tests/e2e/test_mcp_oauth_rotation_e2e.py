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
import time
from typing import Any
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
                    "expires_in": 3600,
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
