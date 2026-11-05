"""TR1 desktop wire against the real router/client and a mocked HTTP hub.

No live account or network is used. The upstream records actual httpx requests;
the OAuth seam is a row store, so owner checks and dispatch are not stubbed.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from local_operator.config import ConfigManager
from local_operator.server.routes import capabilities, desktop_team_bridge
from local_operator.team_bridge import client as bridge

pytestmark = pytest.mark.asyncio
TOKEN = "thread-read-test-desktop-token"
PATH = "/v1/desktop/team-bridge/conversations/bcv_abc12345/messages/bcmsg_abc12345/read"
BODY = {"expected_account_id": "acc_1", "tenant_id": "tnt_1", "seq": 17}
POST_STATE = {
    "conversation_id": "bcv_abc12345",
    "root_id": "bcmsg_abc12345",
    "thread_last_read_seq": 17,
    "thread_unread_count": 2,
}


class Credentials:
    """Only the credential row surface; all transport/auth logic stays real."""

    def __init__(self):
        self.rows = [
            SimpleNamespace(
                id=1,
                provider="radient",
                credential_type="oauth",
                data={"access": "mock-hub-bearer", "account_id": "acc_1"},
            )
        ]
        self.refreshes = 0

    def list_credentials(self, provider=None, include_disabled=False):
        return self.rows

    def get_credential(self, credential_id):
        return next((row for row in self.rows if row.id == credential_id), None)

    async def ensure_oauth_fresh_or_raise(self, credential_id, *, force=False):
        self.refreshes += 1
        row = self.get_credential(credential_id)
        return dict(row.data) if row else None


@pytest_asyncio.fixture
async def proxy(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", "https://desktop.example")
    monkeypatch.setattr(bridge, "org_oauth_destination_allowed", lambda base: True)
    store = Credentials()
    seen = []
    hub = {
        "status": {
            "api_version": "1",
            "available": True,
            "tenants": [{"tenant_id": "tnt_1", "eligible": True, "endpoint": None}],
            "features": {"team_bridge_thread_reads": 1},
        },
        "result": POST_STATE,
        "error": None,
    }

    def upstream(request):
        seen.append(request)
        if request.url.path == "/v1/messaging/status":
            return httpx.Response(200, json={"result": hub["status"]})
        if hub["error"] is not None:
            status, code = hub["error"]
            return httpx.Response(status, json={"code": code, "error": "refused"})
        return httpx.Response(200, json={"result": hub["result"]})

    discipline = bridge.StatusDiscipline(
        tmp_path / "config",
        base_url="https://hub.example/v1",
        transport=httpx.MockTransport(upstream),
        store=store,
        read_only=True,
    )
    monkeypatch.setattr(bridge, "get_status_discipline", lambda *args, **kwargs: discipline)
    monkeypatch.setattr(desktop_team_bridge, "_conversation_client", lambda cm: discipline._client)
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path / "config")
    app.include_router(desktop_team_bridge.router)
    app.include_router(capabilities.router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield SimpleNamespace(client=client, discipline=discipline, seen=seen, hub=hub, store=store)


async def test_thread_read_exact_wire_and_post_state(proxy):
    response = await proxy.client.post(PATH, json=BODY)
    assert response.status_code == 200, response.text
    assert response.json()["result"] == POST_STATE
    assert len(proxy.seen) == 1
    request = proxy.seen[0]
    assert request.method == "POST"
    assert request.url.path == (
        "/v1/messaging/conversations/bcv_abc12345/messages/bcmsg_abc12345/read"
    )
    assert dict(request.url.params) == {}
    assert json.loads(request.content) == {"tenant_id": "tnt_1", "seq": 17}
    assert request.headers["authorization"] == "Bearer mock-hub-bearer"
    print(f"PROXY {response.status_code} {response.text}")
    print(f"UPSTREAM {request.method} {request.url} {request.content.decode()}")


@pytest.mark.parametrize(
    "body",
    [
        {"tenant_id": "tnt_1", "seq": 17},
        {**BODY, "expected_account_id": "bad/owner"},
        {**BODY, "tenant_id": " "},
        {key: value for key, value in BODY.items() if key != "tenant_id"},
        {key: value for key, value in BODY.items() if key != "seq"},
        *[{**BODY, "seq": seq} for seq in (-1, True, "17", 1.5, None)],
        {**BODY, "muted": True},
        {**BODY, "notify": "mute"},
        {**BODY, "actor_endpoint_id": "bep_other"},
    ],
)
async def test_thread_read_invalid_body_never_dispatches(proxy, body):
    response = await proxy.client.post(PATH, json=body)
    assert response.status_code == 422, response.text
    assert proxy.seen == []


@pytest.mark.parametrize(
    "path",
    [
        PATH.replace("bcv_abc12345", "invalid"),
        PATH.replace("bcmsg_abc12345", "invalid"),
        PATH + "?scope=account",
        PATH + "?tenant_id=tnt_other",
    ],
)
async def test_thread_read_invalid_path_or_query_never_dispatches(proxy, path):
    response = await proxy.client.post(path, json=BODY)
    assert response.status_code == 422, response.text
    assert proxy.seen == []


async def test_thread_read_signed_out_never_dispatches(proxy):
    proxy.store.rows = []
    response = await proxy.client.post(PATH, json=BODY)
    assert response.status_code == 401, response.text
    assert proxy.seen == []


async def test_thread_read_wrong_owner_never_dispatches(proxy):
    response = await proxy.client.post(PATH, json={**BODY, "expected_account_id": "acc_2"})
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "account_mismatch"
    assert proxy.seen == []
    print(f"WRONG OWNER {response.status_code} {response.text}")


@pytest.mark.parametrize(
    "headers,status", [({"Authorization": ""}, 401), ({"Origin": "https://evil.example"}, 403)]
)
async def test_thread_read_preserves_desktop_auth_and_origin(proxy, headers, status):
    response = await proxy.client.post(PATH, json=BODY, headers=headers)
    assert response.status_code == status, response.text
    assert proxy.seen == []
    print(f"DESKTOP NEGATIVE {response.status_code} {response.text}")


@pytest.mark.parametrize(
    "status,code",
    [(404, "conversation_not_found"), (404, "message_not_found"), (422, "invalid_tenant")],
)
async def test_thread_read_preserves_upstream_membership_and_root_refusals(proxy, status, code):
    proxy.hub["error"] = (status, code)
    response = await proxy.client.post(PATH, json=BODY)
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    assert len(proxy.seen) == 1
    print(f"HUB NEGATIVE {response.status_code} {response.text}")


@pytest.mark.parametrize("version", [None, 0, True, "1", 1, 2])
async def test_capability_is_only_forwarded_from_fresh_current_owner_status(proxy, version):
    def flags(response):
        assert response.status_code == 200, response.text
        return response.json()["result"]["features"]

    cold = flags(await proxy.client.get("/v1/capabilities"))
    assert "team_bridge_thread_reads" not in cold
    assert proxy.seen == []
    proxy.hub["status"]["features"] = (
        {} if version is None else {"team_bridge_thread_reads": version}
    )
    await proxy.discipline.check("surface_open")
    assert len(proxy.seen) == 1
    refreshes = proxy.store.refreshes
    actual = flags(await proxy.client.get("/v1/capabilities"))
    if type(version) is int and version >= 1:
        assert actual["team_bridge_thread_reads"] == version
    else:
        assert "team_bridge_thread_reads" not in actual
    assert len(proxy.seen) == 1
    assert proxy.store.refreshes == refreshes, "negotiation must not refresh credentials"
    for headers in ({"Authorization": ""}, {"Origin": "https://evil.example"}):
        assert "team_bridge_thread_reads" not in flags(
            await proxy.client.get("/v1/capabilities", headers=headers)
        )
    proxy.store.rows[0].data["account_id"] = "acc_2"
    assert "team_bridge_thread_reads" not in flags(await proxy.client.get("/v1/capabilities"))
    proxy.store.rows[0].data["account_id"] = "acc_1"
    proxy.discipline._states["acc_1"].last_attempt -= bridge.STATUS_CACHE_TTL_S
    assert "team_bridge_thread_reads" not in flags(await proxy.client.get("/v1/capabilities"))
    assert len(proxy.seen) == 1


async def test_failed_status_does_not_advertise_last_good_capability(proxy):
    from dataclasses import replace

    await proxy.discipline.check("surface_open")
    state = proxy.discipline._states["acc_1"]
    state.check = replace(
        state.check,
        state=bridge.STATE_UNAVAILABLE,
        error=bridge.BridgeRefusal("bridge_unreachable", "mock failure", http_status=503),
    )
    response = await proxy.client.get("/v1/capabilities")
    assert response.status_code == 200, response.text
    assert "team_bridge_thread_reads" not in response.json()["result"]["features"]
    assert len(proxy.seen) == 1


async def test_history_and_replies_preserve_thread_unread_and_replier_fields(proxy):
    root = {
        "id": "bcmsg_abc12345",
        "seq": 1,
        "thread_unread_count": 2,
        "thread_last_read_seq": 17,
        "reply_count": 3,
        "reply_participants": [{"display_label": "A"}],
        "reply_participant_ids": [{"account_id": "acc_1", "kind": "human"}],
    }
    base = "/v1/desktop/team-bridge/conversations/bcv_abc12345/messages"
    for suffix, result in (
        ("", {"messages": [root], "has_more": False}),
        ("/bcmsg_abc12345/replies", {"root": root, "replies": [], "has_more": False}),
    ):
        proxy.hub["result"] = result
        params = {"tenant_id": "tnt_1"}
        if suffix:
            params["expected_account_id"] = "acc_1"
        response = await proxy.client.get(base + suffix, params=params)
        assert response.status_code == 200, response.text
        assert response.json()["result"] == result
