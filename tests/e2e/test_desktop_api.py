"""Exercise the assembled FastAPI lifespan over a real loopback HTTP socket.

ASGI adapter tests cannot catch startup isolation or middleware registration
mistakes. This test needs no provider credential and shuts the test server down
through its normal lifecycle; it never replaces a developer's running service.
"""

import asyncio
import secrets
import socket
from pathlib import Path

import httpx
import pytest
import uvicorn

from local_operator.config import ConfigManager
from local_operator.providers.auth_store import AuthStore
from local_operator.server.app import app


@pytest.mark.asyncio
async def test_desktop_controls_over_real_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "desktop-config"
    monkeypatch.setenv("HOME", str(tmp_path / "desktop-home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(1000):
            if server.started:
                break
            if serving.done():
                await serving
            await asyncio.sleep(0)
        assert server.started
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
            public = await client.get("/v1/capabilities")
            print("GET /v1/capabilities", public.status_code, public.json())
            assert public.json()["result"]["desktop_available"]
            missing = await client.get("/v1/settings")
            print("GET /v1/settings without bearer", missing.status_code, missing.json())
            assert missing.status_code == 401
            client.headers["Authorization"] = f"Bearer {token}"
            forbidden = await client.get(
                "/v1/auth/providers", headers={"Origin": "https://evil.example"}
            )
            print("GET /v1/auth/providers foreign origin", forbidden.status_code)
            assert forbidden.status_code == 403
            registry = await client.get("/v1/settings")
            assert registry.status_code == 200
            print(
                "GET /v1/settings",
                registry.status_code,
                "registered rows",
                len(registry.json()["result"]["settings"]),
            )
            saved = await client.patch("/v1/settings/hosting", json={"value": "openrouter"})
            print("PATCH /v1/settings/hosting", saved.status_code, saved.json()["result"]["value"])
            assert saved.status_code == 200
            assert ConfigManager(root).get_config_value("hosting") == "openrouter"
            invalid = await client.patch("/v1/settings/retry.maxRetries", json={"value": 1.5})
            print("PATCH fractional integer", invalid.status_code, invalid.json())
            assert invalid.status_code == 422
            secret = "synthetic-key-for-http-smoke"
            key = await client.put("/v1/auth/providers/openrouter/key", json={"value": secret})
            print("PUT /v1/auth/providers/openrouter/key", key.status_code, key.json())
            assert key.status_code == 200 and secret not in key.text
            with_store = AuthStore(root / "auth.db")
            try:
                assert with_store.list_credentials("openrouter")[0].data["key"] == secret
                print("AuthStore readback: same isolated root; key persisted (value withheld)")
            finally:
                with_store.close()
            accounts = await client.get("/v1/auth/status")
            print("GET /v1/auth/status", accounts.status_code, accounts.json())
            assert secret not in accounts.text
            assert accounts.headers["cache-control"] == "no-store"
            assert not (tmp_path / "desktop-home" / ".local-operator" / "config.yml").exists()
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, timeout=20)
        listener.close()
