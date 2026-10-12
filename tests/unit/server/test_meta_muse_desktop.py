"""Real desktop HTTP login-operation contract driven by the Muse protocol fixture."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from local_operator.providers.oauth import meta_muse
from tests.unit.server.test_desktop_controls import desktop as _desktop_fixture
from tests.unit.server.test_desktop_controls import wait_for_state

desktop = _desktop_fixture
KEY = "LLM|1234|fixture_desktop_key"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "denied", "inactive", "cancelled"])
async def test_meta_device_operation_contract(desktop, monkeypatch, outcome):
    client, app = desktop
    pending = asyncio.Event()
    original = meta_muse.login_meta_muse

    async def handle(request):
        if request.url == meta_muse.AUTHORIZATION_URL:
            return httpx.Response(
                200,
                json={
                    "device_code": "private-fixture-device",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "https://auth.meta.com/oauth/device/",
                    "expires_in": 600,
                    "interval": 1,
                },
            )
        if request.url == meta_muse.TOKEN_URL:
            if outcome == "cancelled":
                await pending.wait()
            if outcome == "denied":
                return httpx.Response(400, json={"error": "access_denied"})
            return httpx.Response(200, json={"access_token": "fixture-account-token"})
        return httpx.Response(
            200,
            json={
                "api_key": KEY,
                "user_email": "fixture@example.test",
                "is_subs_active": outcome != "inactive",
                "require_payment": False,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as protocol:

        async def login(callbacks, *, signal=None, **_kwargs):
            return await original(callbacks, signal=signal, http_client=protocol)

        monkeypatch.setattr(meta_muse, "login_meta_muse", login)
        started = await client.post("/v1/auth/login", json={"provider": "meta-muse"})
        assert started.status_code == 200
        op = started.json()["result"]
        assert op["user_code"] == "ABCD-EFGH"
        assert op["auth_url"] == "https://auth.meta.com/oauth/device/"
        assert "unsupported by Meta" in op["instructions"]
        assert "Contributor" in op["instructions"]
        assert "private-fixture-device" not in json.dumps(op)
        assert KEY not in json.dumps(op)
        if outcome == "cancelled":
            response = await client.delete(f"/v1/auth/operations/{op['id']}")
            assert response.status_code == 200
            settled = response.json()["result"]
        else:
            settled = await wait_for_state(client, op["id"], "succeeded", "failed")
        assert settled["state"] == (
            "succeeded"
            if outcome == "success"
            else "cancelled" if outcome == "cancelled" else "failed"
        )
        rows = app.state.desktop_auth.store.list_credentials("meta-muse")
        assert len(rows) == (1 if outcome == "success" else 0)
        if rows:
            assert rows[0].credential_type == "oauth"
            assert "access" not in rows[0].data
            assert "access_secret_ref" in rows[0].data
        assert KEY not in json.dumps(settled)
        refused = await client.put("/v1/auth/providers/meta-muse/key", json={"value": KEY})
        assert refused.status_code == 422
