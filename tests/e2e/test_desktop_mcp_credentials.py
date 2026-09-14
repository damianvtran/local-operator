"""The real desktop HTTP -> owner RPC -> encrypted store -> HTTP MCP path."""

import asyncio
import json
import os
import secrets

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from local_operator.mcp.config import load_all_mcp_configs
from local_operator.mcp.manager import McpManager
from local_operator.secrets import access
from local_operator.server.app import app
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session
from tests.e2e.test_desktop_controls import request_id
from tests.e2e.test_desktop_radient import serve

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_encrypted_mcp_http_rpc_restart(headless_tui_env, workspace, monkeypatch):
    root = headless_tui_env
    for key in list(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", secrets.token_hex(32))
    monkeypatch.setattr("local_operator.secrets.client.ensure_broker", lambda *a, **kw: False)
    (root / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    key = secrets.token_hex(32)
    issuer = FastAPI()
    pings = []

    @issuer.api_route("/mcp", methods=["GET", "POST", "DELETE"])
    async def mcp(request: Request):
        if request.headers.get("authorization") != "Bearer " + key:
            # Echo the rejected header like a badly behaved third-party server.
            return JSONResponse({"error": request.headers.get("authorization")}, status_code=401)
        if request.method != "POST":
            return Response(status_code=405)
        body = await request.json()
        if "id" not in body:
            return Response(status_code=202)
        if body["method"] == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "key-fixture", "version": "1"},
            }
        elif body["method"] == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "ping",
                        "description": "Synthetic ping",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            }
        else:
            pings.append(True)
            result = {"content": [{"type": "text", "text": "pong"}]}
        return {"jsonrpc": "2.0", "id": body["id"], "result": result}

    runtime = handle = manager = None
    seen_chain: set[int] = set()
    try:
        async with serve(issuer) as issuer_url, serve(app) as desktop_url:
            (root / "mcp.json").write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "api": {
                                "type": "http",
                                "url": issuer_url + "/mcp",
                                "auth": {"type": "apikey"},
                                "headers": {"Authorization": "Bearer ${HUBSPOT_TOKEN}"},
                            }
                        }
                    }
                )
            )
            async with httpx.AsyncClient(
                base_url=desktop_url,
                headers={"Authorization": "Bearer " + os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"]},
                timeout=30,
            ) as client:
                created = await client.post(
                    "/v1/desktop/sessions", json={"request_id": request_id(), "cwd": str(workspace)}
                )
                assert created.status_code == 200
                sid = created.json()["result"]["session_id"]
                session = build_session(root / "sessions" / sid, ScriptedStream([]), cwd=workspace)
                manager = McpManager(
                    workspace,
                    secret_base=root,
                    register_secret=session.variables.register_redaction,
                )
                manager._configs, manager._sources = load_all_mcp_configs(workspace)
                session.mcp_manager = manager
                handle = ServingSessionHandle(
                    session, asyncio.get_running_loop(), cwd=str(workspace)
                )
                runtime = RuntimeServer(handle, kind="daemon")
                await runtime.start_in_process()
                (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))
                route = f"/v1/desktop/sessions/{sid}/mcp"
                store_route = route + "/credentials"
                listed = await client.get(route)
                assert (
                    listed.json()["result"]["data"]["servers"][0]["secret_refs"][0]["id"]
                    == "HUBSPOT_TOKEN"
                )
                payload = {"name": "api", "values": {"HUBSPOT_TOKEN": key}, "confirmed_replace": []}
                refused = await client.post(
                    store_route, json=payload, headers={"Authorization": ""}
                )
                assert refused.status_code == 401
                refused = await client.post(
                    store_route, json=payload, headers={"Origin": "https://untrusted.invalid"}
                )
                assert refused.status_code == 403
                invalid = await client.post(
                    store_route, json={**payload, "values": {"Authorization": key}}
                )
                assert invalid.json()["result"]["data"]["code"] == "invalid_target"
                for bad in (
                    {**payload, "extra": key},
                    {**payload, "values": {"HUBSPOT_TOKEN": {"bad": key}}},
                    {**payload, "values": {"HUBSPOT_TOKEN": ""}},
                ):
                    response = await client.post(store_route, json=bad)
                    assert response.status_code == 422 and key not in response.text
                invalid = await client.post(store_route, json={**payload, "name": "unknown"})
                assert invalid.json()["result"]["data"]["code"] == "invalid_target"
                saved = await client.post(store_route, json=payload)
                assert saved.status_code == 200 and key not in saved.text
                assert saved.json()["result"]["data"]["code"] == "saved"
                assert access.open_store(root).get("HUBSPOT_TOKEN") == key.encode()
                refused = await client.post(store_route, json=payload)
                assert refused.json()["result"]["data"]["code"] == "replace_confirmation_required"
                connected = await client.post(route, json={"action": "connect", "name": "api"})
                assert connected.status_code == 200 and key not in connected.text
                assert connected.json()["result"]["data"]["servers"][0]["status"] == "connected"
                assert len(manager.get_server_tools("api")) == 1
                assert not session.variables.credential_env()
                # Fresh manager = no in-memory credential cache. Auth survives
                # restart solely through the encrypted persisted reference ID.
                await manager.disconnect_all()
                manager = McpManager(workspace, secret_base=root)
                manager._configs, manager._sources = load_all_mcp_configs(workspace)
                connection = await manager._connect_server("api", manager._configs["api"])
                session = connection.session
                assert session is not None
                result = await session.call_tool("ping", {})
                assert "pong" in str(result) and pings
                # The desktop app's own startup constructs a CredentialManager
                # (`server/app.py:113`), which CREATES an empty `credentials.env`.
                # That is pre-existing behaviour unrelated to this path, and it is
                # recorded rather than changed here: what this change must
                # guarantee is that no VALUE of ours lands in the file — the
                # legacy store is read-only for MCP references and a new key is an
                # encrypted-store write.
                legacy = root / "credentials.env"
                if legacy.exists():
                    assert key not in legacy.read_text()
                # A WRONG stored value must fail without echoing itself back. This
                # fixture answers a bad header with the rejected header IN THE
                # ERROR BODY, which is the shape a real third-party server can
                # have, so the assertion covers the error text the user sees AND
                # the exception it is chained to (BI-1's "HTTP error body echo").
                wrong = secrets.token_hex(32)
                replaced = await client.post(
                    store_route,
                    json={
                        "name": "api",
                        "values": {"HUBSPOT_TOKEN": wrong},
                        "confirmed_replace": ["HUBSPOT_TOKEN"],
                    },
                )
                assert replaced.json()["result"]["data"]["code"] == "saved"
                failure = await client.post(route, json={"action": "connect", "name": "api"})
                assert failure.status_code == 200
                assert wrong not in failure.text
                try:
                    await asyncio.wait_for(manager.wait_for_connection("api"), 5)
                except Exception as exc:
                    chain, text = exc, ""
                    while chain is not None and id(chain) not in seen_chain:
                        seen_chain.add(id(chain))
                        text += f"{chain!r} {chain.args}"
                        chain = chain.__cause__ or chain.__context__
                    assert wrong not in text, "the credential reached the exception chain"
                else:
                    raise AssertionError("a wrong key must not connect")
                # Restore, so the restart leg below proves the healed path.
                restored = await client.post(
                    store_route,
                    json={
                        "name": "api",
                        "values": {"HUBSPOT_TOKEN": key},
                        "confirmed_replace": ["HUBSPOT_TOKEN"],
                    },
                )
                assert restored.json()["result"]["data"]["code"] == "saved"
                # Scan raw persisted artifacts, including SQLite, config and
                # transcripts/receipts. Assertion output never includes a value.
                for path in root.rglob("*"):
                    if path.is_file():
                        assert key.encode() not in path.read_bytes(), str(path.relative_to(root))
    finally:
        if manager is not None:
            await manager.disconnect_all()
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()
