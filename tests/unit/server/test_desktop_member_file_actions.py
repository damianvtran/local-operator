"""Isolated ASGI router tests with synthetic main provider and mock hub only.

These tests explicitly install the unregistered router in a test sub-app. They
are not production registration, trusted-main proof, or a live hub/socket test.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.server.routes import desktop_member_file_actions as route
from local_operator.team_bridge import client as client_module
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.file_action_orchestrator import HumanFileActionOrchestrator, TrustedMainView
from local_operator.team_bridge.file_action_registry import InertFileActionRegistry

pytestmark = pytest.mark.asyncio
PREFIX = "/v1/desktop/member-file-actions"
WINDOW = "window_1"
FILE = "batt_aaaaaaaa"
TOKEN = "test-desktop-token"


@dataclass
class Row:
    id: int = 1
    provider: str = "radient"
    credential_type: str = "oauth"
    disabled_cause: str | None = None
    updated_at: int = 42
    data: dict[str, Any] = field(default_factory=lambda: {"access": "test-upstream-secret", "account_id": "acc_1", "expires": 4102444800000})


class Store:
    db_path = Path("test-auth.db")
    def __init__(self):
        self.row = Row()
        self.incarnation = "a" * 32
        self.refreshes = 0
    def list_credentials(self, _provider):
        return [self.row]
    def get_credential(self, credential_id):
        return self.row if credential_id == 1 else None
    def native_lifecycle_snapshot(self, _path, _provider):
        return [(1, 42, self.incarnation, None, "oauth", "acc_1", None, None, None)]
    async def ensure_oauth_fresh_or_raise(self, *args, **kwargs):
        self.refreshes += 1
        raise AssertionError("refresh is forbidden")


def page():
    return {"msg": "ok", "result": {"messages": [{"id": "bcmsg_aaaaaaaa", "conversation_id": "bcv_aaaaaaaa",
        "files": [{"id": FILE, "mime": "application/octet-stream", "download_only": True,
                   "state": "bound", "class": "other", "size": 0, "expires_at": "2099-01-01T00:00:00Z"}]}]}}


@pytest_asyncio.fixture
async def api(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("CMUX_") or key.startswith("LOP_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    monkeypatch.setattr(client_module, "org_oauth_destination_allowed", lambda _: True)
    state = {"view": TrustedMainView(WINDOW, 3, "tnt_alpha", "bcv_aaaaaaaa", "bcmsg_aaaaaaaa"), "status": 200}
    calls, store = [], Store()
    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/messages"):
            return httpx.Response(state["status"], json=page())
        return httpx.Response(state["status"], content=b"", headers={"content-type": "application/octet-stream"})
    bridge = TeamBridgeClient(base_url="https://hub.example/v1", store=store, transport=httpx.MockTransport(handler))
    registry = InertFileActionRegistry()
    # Only the TEST installs a synthetic provider. Production dependency is 503.
    actions = HumanFileActionOrchestrator(bridge, registry, lambda window: state["view"] if window == WINDOW else None)
    app = FastAPI()
    app.include_router(route.router)
    app.dependency_overrides[route.member_file_orchestrator] = lambda: actions
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://desktop.invalid",
                           headers={"Authorization": f"Bearer {TOKEN}"}) as client:
        yield client, actions, registry, store, calls, state, app


async def commission(client):
    response = await client.post(PREFIX + "/commission", json={"window_id": WINDOW})
    assert response.status_code == 200
    return response.json()["context"]


async def begin(client, context):
    response = await client.post(PREFIX + "/begin", json={"window_id": WINDOW, "context": context, "file_id": FILE})
    assert response.status_code == 200
    return response.json()["operation"]


async def test_requires_desktop_and_unprovided_provider_fails_closed(api):
    client, actions, registry, store, calls, state, app = api
    denied = await client.post(PREFIX + "/commission", json={"window_id": WINDOW}, headers={"Authorization": "Bearer wrong"})
    assert denied.status_code == 401 and calls == []
    del app.dependency_overrides[route.member_file_orchestrator]
    missing = await client.post(PREFIX + "/commission", json={"window_id": WINDOW})
    assert missing.status_code == 503 and calls == []


async def test_full_bound_action_and_headers_zero_byte_replay(api):
    client, actions, registry, store, calls, state, app = api
    context = await commission(client)
    operation = await begin(client, context)
    payload = {"window_id": WINDOW, "operation": operation}
    content = await client.post(PREFIX + "/bytes", json=payload)
    assert content.status_code == 200 and content.content == b""
    assert content.headers["content-type"] == "application/octet-stream"
    assert content.headers["content-disposition"] == 'attachment; filename="attachment.bin"'
    assert content.headers["cache-control"] == "no-store" and content.headers["x-content-type-options"] == "nosniff"
    assert (await client.post(PREFIX + "/verify", json=payload)).status_code == 204
    assert (await client.post(PREFIX + "/bytes", json=payload)).status_code == 404
    assert (await client.post(PREFIX + "/revoke", json={"window_id": WINDOW, "context": context})).status_code == 204
    assert (await client.post(PREFIX + "/verify", json=payload)).status_code == 404
    assert len(calls) == 2 and store.refreshes == 0
    assert dict(calls[0].url.params) == {"tenant_id": "tnt_alpha", "limit": "50"}
    assert dict(calls[1].url.params) == {"tenant_id": "tnt_alpha"}


@pytest.mark.parametrize("extra", [
    {"account_id": "acc_other"}, {"incarnation": "x"}, {"tenant_id": "tnt_other"},
    {"endpoint_id": "endpoint"}, {"actor_endpoint_id": "endpoint"}, {"purpose": "context"},
    {"url": "https://hub.example"}, {"path": "../../file"}, {"generation": 99},
])
async def test_rejects_untrusted_scope_and_agent_fields_without_echo(api, extra):
    client, actions, registry, store, calls, state, app = api
    response = await client.post(PREFIX + "/commission", json={"window_id": WINDOW, **extra})
    assert response.status_code == 422 and calls == []
    assert all(str(value) not in response.text for value in extra.values())
    assert "test-upstream-secret" not in response.text


async def test_rejects_query_duplicate_json_and_injected_file_id(api):
    client, actions, registry, store, calls, state, app = api
    response = await client.post(PREFIX + "/commission?endpoint_id=evil", json={"window_id": WINDOW})
    assert response.status_code == 422
    response = await client.post(PREFIX + "/commission", content='{"window_id":"window_1","window_id":"window_2"}', headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    response = await client.post(PREFIX + "/begin", json={"window_id": WINDOW, "context": "a" * 43, "file_id": "../../secret"})
    assert response.status_code == 422 and calls == []


async def test_wrong_window_tenant_invalidation_and_cancel(api):
    client, actions, registry, store, calls, state, app = api
    context = await commission(client)
    bad = await client.post(PREFIX + "/begin", json={"window_id": "window_2", "context": context, "file_id": FILE})
    assert bad.status_code == 404
    operation = await begin(client, context)
    state["view"] = replace(state["view"], tenant_id="tnt_other")
    refused = await client.post(PREFIX + "/bytes", json={"window_id": WINDOW, "operation": operation})
    assert refused.status_code == 404 and len(calls) == 1
    state["view"] = replace(state["view"], tenant_id="tnt_alpha")
    assert (await client.post(PREFIX + "/cancel", json={"window_id": WINDOW, "operation": operation})).status_code == 204
    actions.invalidate(WINDOW, 3)
    assert (await client.post(PREFIX + "/begin", json={"window_id": WINDOW, "context": context, "file_id": FILE})).status_code == 404
    assert store.refreshes == 0


async def test_history_and_byte_401_are_closed_no_refresh(api):
    client, actions, registry, store, calls, state, app = api
    state["status"] = 401
    response = await client.post(PREFIX + "/commission", json={"window_id": WINDOW})
    assert response.status_code == 401 and response.json() == {"error": "reauthentication_required"}
    assert len(calls) == 1 and store.refreshes == 0
    state["status"] = 200
    context = await commission(client)
    operation = await begin(client, context)
    state["status"] = 401
    response = await client.post(PREFIX + "/bytes", json={"window_id": WINDOW, "operation": operation})
    assert response.status_code == 401 and response.json() == {"error": "reauthentication_required"}
    assert len(calls) == 3 and store.refreshes == 0
    assert "test-upstream-secret" not in response.text
