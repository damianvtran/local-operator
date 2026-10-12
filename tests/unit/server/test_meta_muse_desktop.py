"""Real desktop HTTP login-operation contract driven by the Muse protocol fixture."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from local_operator.providers.oauth import meta_muse
from tests.unit.server.test_desktop_controls import desktop as _desktop_fixture

desktop = _desktop_fixture
KEY = "LLM|1234|fixture_desktop_key"


async def wait_for_muse_state(client, operation_id: str):
    """Real elapsed budget: store/default writes may complete in worker threads.

    Yielding 1000 loop turns can finish before a healthy background write even
    starts. Poll the same HTTP contract within a monotonic wall-clock bound,
    retaining a last-state diagnostic rather than waiving a slow success.
    """
    last_state = "not observed"
    try:
        async with asyncio.timeout(30):
            while True:
                data = (await client.get(f"/v1/auth/operations/{operation_id}")).json()["result"]
                last_state = data["state"]
                if last_state in ("succeeded", "failed"):
                    return data
                await asyncio.sleep(0.01)
    except TimeoutError:
        pytest.fail(
            f"Muse login did not settle within 30 elapsed seconds; last state: {last_state}"
        )


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

        async def login(callbacks, *, signal=None, open_browser=None, **_kwargs):
            assert open_browser is not None  # renderer-owned opening must reach the device flow
            return await original(
                callbacks, signal=signal, http_client=protocol, open_browser=open_browser
            )

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
            settled = await wait_for_muse_state(client, op["id"])
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
