"""Real HTTP MCP + OAuth fixture. Consent is simulated, not a browser/user proof."""

import asyncio
import os
import secrets
import socket
from urllib.parse import urlencode

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from local_operator.mcp.manager import McpConnectionError, McpManager
from local_operator.server.app import app
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from tests.e2e.harness import ScriptedStream, build_session
from tests.e2e.test_desktop_controls import request_id, until
from tests.e2e.test_desktop_radient import serve

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_mcp_oauth_controls(headless_tui_env, workspace, monkeypatch):
    from local_operator.mcp import auth
    from local_operator.providers.auth_store import AuthStore

    root = headless_tui_env
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", secrets.token_hex(32))
    (root / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    issuer = FastAPI()
    access = secrets.token_hex(32)
    origin = ""
    grants = []
    consent = asyncio.Event()
    allow_consent = True

    @issuer.get("/.well-known/oauth-protected-resource")
    async def resource():
        return {
            "resource": origin + "/mcp",
            "authorization_servers": [origin],
            "scopes_supported": ["fixture"],
        }

    @issuer.get("/.well-known/oauth-authorization-server")
    async def metadata():
        return {
            "issuer": origin,
            "authorization_endpoint": origin + "/authorize",
            "token_endpoint": origin + "/token",
            "registration_endpoint": origin + "/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none"],
            "scopes_supported": ["fixture"],
        }

    @issuer.post("/register")
    async def register(request: Request):
        body = await request.json()
        return JSONResponse(
            {**body, "client_id": "desktop-fixture", "token_endpoint_auth_method": "none"},
            status_code=201,
        )

    @issuer.get("/authorize")
    async def authorize(request: Request):
        args = request.query_params
        return RedirectResponse(
            args["redirect_uri"] + "?" + urlencode({"state": args["state"], "code": "fixture-code"})
        )

    @issuer.post("/token")
    async def token():
        grants.append(True)
        return {
            "access_token": access,
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "fixture-refresh",
            "scope": "fixture",
        }

    @issuer.api_route("/mcp", methods=["GET", "POST", "DELETE"])
    async def mcp(request: Request):
        if request.headers.get("authorization") != "Bearer " + access:
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={
                    "WWW-Authenticate": (
                        f"Bearer "
                        f'resource_metadata="{origin}/.well-known/oauth-protected-resource"'
                    )
                },
            )
        if request.method != "POST":
            return Response(status_code=405)
        body = await request.json()
        if "id" not in body:
            return Response(status_code=202)
        result = (
            {
                "protocolVersion": body.get("params", {}).get("protocolVersion", "2025-03-26"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "oauth-fixture", "version": "1"},
            }
            if body["method"] == "initialize"
            else (
                {
                    "tools": [
                        {
                            "name": "fixture",
                            "description": "Fixture tool",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    ]
                }
                if body["method"] == "tools/list"
                else {}
            )
        )
        return {"jsonrpc": "2.0", "id": body["id"], "result": result}

    async def fixture_consent(url):
        # Exercise the real callback listener with a fixture authorization
        # redirect. No real browser is opened and no external grant is asserted.
        assert url.startswith(origin + "/authorize")
        consent.set()
        if allow_consent:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url)
                assert response.status_code == 200
        return True

    monkeypatch.setattr(auth, "open_browser_quietly", fixture_consent)
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    callback_port = probe.getsockname()[1]
    probe.close()
    monkeypatch.setattr(auth, "DEFAULT_CALLBACK_PORT", callback_port)
    runtime = handle = manager = None
    store = AuthStore(root / "auth.db")
    try:
        async with serve(issuer) as issuer_url, serve(app) as desktop_url:
            origin = issuer_url
            async with httpx.AsyncClient(
                base_url=desktop_url,
                headers={"Authorization": "Bearer " + os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"]},
                timeout=30,
            ) as client:
                created = await client.post(
                    "/v1/desktop/sessions", json={"request_id": request_id(), "cwd": str(workspace)}
                )
                sid = created.json()["result"]["session_id"]
                session = build_session(root / "sessions" / sid, ScriptedStream([]), cwd=workspace)
                manager = McpManager(str(workspace), auth_store=store)
                session.mcp_manager = manager
                handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
                runtime = RuntimeServer(handle, kind="daemon")
                await runtime.start_in_process()
                (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))
                route = f"/v1/desktop/sessions/{sid}/mcp"

                async def control(**body):
                    result = await client.post(route, json=body)
                    assert result.status_code == 200, result.status_code
                    assert access not in result.text
                    return result.json()["result"]["data"]

                await control(action="add", name="oauth-fixture", url=origin + "/mcp", oauth=True)
                with pytest.raises((auth.McpAuthRequiredError, McpConnectionError)):
                    await manager.wait_for_connection("oauth-fixture")
                assert not consent.is_set() and not grants
                op = await control(action="login", name="oauth-fixture")
                await asyncio.wait_for(consent.wait(), 20)
                await until(lambda: not handle._desktop_mcp.running)
                result = await control(action="status", operation_id=op["id"])
                assert result["status"] == "complete", result
                assert manager.get_connection_status("oauth-fixture") == "connected"
                assert len(manager.get_server_tools("oauth-fixture")) == 1
                assert grants and store.list_credentials()
                consent.clear()
                allow_consent = False
                reauth = await control(action="reauth", name="oauth-fixture", confirmed=True)
                await asyncio.wait_for(consent.wait(), 20)
                cancelled = await control(action="cancel", operation_id=reauth["id"])
                assert cancelled["status"] == "cancelled" and cancelled["credential_removed"]
                assert all(not row.data.get("tokens") for row in store.list_credentials())
                allow_consent = True
                login_again = await control(action="login", name="oauth-fixture")
                await until(lambda: not handle._desktop_mcp.running)
                assert (await control(action="status", operation_id=login_again["id"]))[
                    "status"
                ] == "complete"
                logout = await control(action="logout", name="oauth-fixture", confirmed=True)
                await until(lambda: not handle._desktop_mcp.running)
                assert (await control(action="status", operation_id=logout["id"]))[
                    "credential_removed"
                ]
                assert all(not row.data.get("tokens") for row in store.list_credentials())
                print(
                    (
                        "Real local HTTP MCP/OAuth fixture: login/status200 discovers one "
                        "tool and stores grant; logout200 deletes grant; reauth/cancel200 "
                        "releases callback and leaves no grant; no browser or external "
                        "credential proof claimed"
                    )
                )
    finally:
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()
        if manager is not None:
            await manager.disconnect_all()
        store.close()
