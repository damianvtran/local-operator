"""Actual loopback HTTP to a labelled fake Radient upstream, not real account proof."""

import asyncio
import secrets
import socket
import time
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from local_operator.server.app import app
from tests.e2e.test_desktop_controls import request_id, until

pytestmark = pytest.mark.e2e


@asynccontextmanager
async def serve(application):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(application, log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await until(lambda: server.started)
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 30)
        listener.close()


@pytest.mark.asyncio
async def test_radient_proxy_real_http(headless_tui_env, monkeypatch):
    from local_operator.providers.oauth import radient as oauth
    from local_operator.server.routes import desktop_radient

    token, refresh, access, app_key = [secrets.token_hex(32) for _ in range(4)]
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    (headless_tui_env / "config.yml").write_text("version: 0.0.0\nvalues: {}\n")
    fake = FastAPI()
    calls = []
    refreshes = []

    @fake.post("/token")
    async def rotate(request: Request):
        body = await request.json()
        assert body["refresh_token"] == refresh
        refreshes.append(True)
        return {"access_token": access, "refresh_token": refresh, "expires_in": 3600}

    @fake.api_route("/v1/{path:path}", methods=["GET", "POST", "PATCH", "DELETE"])
    async def upstream(path: str, request: Request):
        calls.append((request.method, path, dict(request.query_params)))
        if path != "prices" and request.headers.get("authorization") != "Bearer " + access:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if path.startswith("tenants/wrong/"):
            return JSONResponse({"error": "wrong tenant", "access_token": access}, status_code=403)
        if path == "agents/redirect":
            return JSONResponse({}, status_code=302, headers={"Location": "https://example.org"})
        return {
            "status": 200,
            "result": {
                "path": path,
                "api_key": app_key if path.endswith("applications") or path == "provision" else "",
                "access_token": access,
                "refresh_token": refresh,
                "balance": 123,
                "items": [],
            },
        }

    async with serve(fake) as upstream_url, serve(app) as desktop_url:
        monkeypatch.setattr(desktop_radient, "base_url", lambda: upstream_url + "/v1")
        monkeypatch.setattr(oauth, "TOKEN_URL", upstream_url + "/token")
        async with httpx.AsyncClient(base_url=desktop_url, timeout=30) as client:
            route = "/v1/desktop/radient"
            assert (await client.post(route, json={"operation": "account"})).status_code == 401
            client.headers["Authorization"] = "Bearer " + token
            assert (
                await client.post(route, json={"operation": "account"}, headers={"Origin": "null"})
            ).status_code == 403
            assert (await client.post(route, json={"operation": "account"})).status_code == 409
            await client.get("/v1/auth/status")
            store = app.state.desktop_auth.store
            store.upsert_credential(
                "radient",
                {
                    "type": "oauth",
                    "access": "expired-fixture",
                    "refresh": refresh,
                    "expires": int(time.time() * 1000) - 1000,
                },
            )

            requests = [
                {"operation": "account"},
                {"operation": "prices"},
                {"operation": "credits", "tenant_id": "fixture"},
                {
                    "operation": "usage",
                    "tenant_id": "fixture",
                    "query": {"rollup": "daily", "provider": "test"},
                },
                {"operation": "agents.list", "query": {"page": 1, "per_page": 10}},
                {"operation": "account.agents", "account_id": "fixture"},
                {"operation": "agents.get", "agent_id": "fixture"},
                *[
                    {
                        "operation": "agents." + operation,
                        "agent_id": "fixture",
                        "request_id": request_id(),
                        "confirmed": True,
                    }
                    for operation in (
                        "like",
                        "unlike",
                        "liked",
                        "like_count",
                        "favourite",
                        "unfavourite",
                        "favourited",
                        "favourite_count",
                        "download_count",
                        "delete",
                    )
                ],
                {
                    "operation": "agents.create",
                    "request_id": request_id(),
                    "payload": {"name": "Fixture", "version": "1"},
                },
                {
                    "operation": "agents.update",
                    "agent_id": "fixture",
                    "request_id": request_id(),
                    "payload": {"description": "Updated"},
                },
                {"operation": "comments.list", "agent_id": "fixture"},
                *[
                    {
                        "operation": "comments." + operation,
                        "agent_id": "fixture",
                        "comment_id": "fixture",
                        "request_id": request_id(),
                        "confirmed": True,
                        "payload": {"text": "Fixture"} if operation != "delete" else {},
                    }
                    for operation in ("create", "update", "delete")
                ],
                {"operation": "provision", "request_id": request_id()},
                {
                    "operation": "application.create",
                    "tenant_id": "fixture",
                    "request_id": request_id(),
                    "payload": {"name": "Desktop fixture"},
                },
            ]
            for body in requests:
                response = await client.post(route, json=body)
                assert response.status_code == 200, (body["operation"], response.status_code)
                assert all(
                    value not in response.text for value in (access, refresh, app_key, token)
                )
            assert len(refreshes) == 1
            assert any(
                row.credential_type == "api_key" and row.data.get("key") == app_key
                for row in store.list_credentials("radient")
            )
            assert len(calls) == len(requests)
            replay = await client.post(route, json=requests[-1])
            assert replay.json()["result"]["replayed"] and len(calls) == len(requests)
            wrong = await client.post(route, json={"operation": "credits", "tenant_id": "wrong"})
            assert wrong.status_code == 403 and access not in wrong.text
            redirect = await client.post(
                route, json={"operation": "agents.get", "agent_id": "redirect"}
            )
            assert redirect.status_code == 502
            before = len(calls)
            for body in (
                {"operation": "account", "url": "https://example.org"},
                {"operation": "credits", "tenant_id": "../secret"},
                {"operation": "agents.delete", "agent_id": "fixture", "request_id": request_id()},
                {"operation": "usage", "tenant_id": "fixture", "query": {"token": "bad"}},
                {
                    "operation": "application.create",
                    "tenant_id": "fixture",
                    "request_id": request_id(),
                    "payload": {"name": "x", "api_key": "bad"},
                },
            ):
                assert (await client.post(route, json=body)).status_code == 422
            assert len(calls) == before
            accounts = (await client.get("/v1/auth/status")).json()["result"]["accounts"]
            key_account = next(row for row in accounts if row["type"] == "api_key")
            removed = await client.delete("/v1/auth/accounts/" + str(key_account["id"]))
            assert removed.status_code == 200
            assert store.get_credential(key_account["id"]) is None
            assert await store.get_oauth_access("radient") is not None
            assert (
                await client.delete("/v1/auth/accounts/" + str(key_account["id"]))
            ).status_code == 404
            print(
                (
                    f"Fake Radient upstream (no real third-party credentials): "
                    f"{len(requests)} supported operations HTTP200; real AuthStore refresh"
                    f" once, provisioned API key stored centrally, no credentials in "
                    f"responses"
                )
            )
            print(
                (
                    "Proxy failures: token401, Origin403, signed-out409, wrong tenant403,"
                    " redirect502, invalid path/payload/query/unconfirmed mutation422; "
                    "mutation retry replayed without upstream duplication"
                )
            )
