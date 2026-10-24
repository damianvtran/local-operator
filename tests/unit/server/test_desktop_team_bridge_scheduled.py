"""Scheduled CRUD through actual loopback HTTP, with an isolated hub fixture.

Only the credential store and hub's domain are fixtures: desktop bearer checks,
routing, request validation, OAuth resolution/refresh, HTTP transport, refusal
mapping and response envelopes all execute their production implementations.
This is proxy proof, not proof of the server's Mongo worker or native UI.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from local_operator.server.dependencies import get_config_manager
from local_operator.server.routes import desktop_team_bridge as routes
from local_operator.team_bridge import client as bridge
from tests.unit.team_bridge.test_client import FakeStore

PREFIX = "/v1/desktop/team-bridge/scheduled"
SID = "bsch_" + "a" * 32
CID = "bcv_" + "b" * 32
CREATE = {
    "tenant_id": "org-a",
    "conversation_id": CID,
    "client_ref": "schedule-1",
    "body": "unsent private fixture body",
    "send_at": "2030-04-05T12:30:00Z",
    "mentions": [{"kind": "human", "account_id": "acc-2"}],
}


@pytest.fixture
def proxy(tmp_path, monkeypatch):
    calls = []
    state: dict[str, Any] = {"row": None, "error": None, "malformed": False}

    class Hub(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass  # The body and bearer must never appear in fixture logs either.

        def dispatch(self):
            payload = self.rfile.read(int(self.headers.get("content-length", 0)))
            body: Any = json.loads(payload) if payload else None
            out: dict[str, Any]
            calls.append((self.command, self.path, body, self.headers.get("authorization")))
            code = 200
            if state["error"]:
                code, error = state["error"]
                out = {
                    "code": error,
                    "error": "Fixture refusal",
                    "details": {"field": "send_at", "rule": "outside_window"},
                }
            elif state["malformed"]:
                out = {"msg": "Missing read result"}
                if state["malformed"] == "list-result":
                    out["result"] = []
            elif self.command == "POST":
                state["row"] = {
                    "id": SID,
                    "tenant_id": body["tenant_id"],
                    "conversation_id": body["conversation_id"],
                    "body": body["body"],
                    "send_at": body["send_at"],
                    "status": "scheduled",
                    "created_at": "2030-04-05T10:00:00Z",
                }
                code, out = 201, {"msg": "Message scheduled", "result": state["row"]}
            elif self.command == "PATCH":
                state["row"].update(body)
                out = {"msg": "Message updated", "result": state["row"]}
            elif self.command == "DELETE":
                state["row"] = None
                out = {"msg": "Scheduled message canceled"}
            else:
                out = {
                    "msg": "Schedules",
                    "result": {"scheduled": [state["row"]] if state["row"] else []},
                }
            encoded = (
                b"not json" if state["malformed"] == "invalid-json" else json.dumps(out).encode()
            )
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        do_GET = do_POST = do_PATCH = do_DELETE = dispatch

    hub = ThreadingHTTPServer(("127.0.0.1", 0), Hub)
    hub_thread = threading.Thread(target=hub.serve_forever, daemon=True)
    hub_thread.start()
    store = FakeStore()
    real_client = bridge.TeamBridgeClient
    monkeypatch.setattr(bridge, "org_oauth_destination_allowed", lambda _base: True)
    monkeypatch.setattr(
        bridge,
        "TeamBridgeClient",
        lambda config_dir: real_client(
            config_dir, base_url=f"http://127.0.0.1:{hub.server_port}/v1", store=store
        ),
    )
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "fixture-desktop-capability")
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_config_manager] = lambda: SimpleNamespace(config_dir=tmp_path)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started, "loopback proxy did not start"
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            headers={"authorization": "Bearer fixture-desktop-capability"},
        ) as client:
            yield client, calls, state, store
    finally:
        server.should_exit = True
        thread.join(10)
        listener.close()
        hub.shutdown()
        hub.server_close()
        hub_thread.join(10)
        assert not thread.is_alive()
        assert not hub_thread.is_alive()


def test_real_http_create_list_edit_cancel(proxy):
    client, calls, state, _ = proxy
    created = client.post(PREFIX, json=CREATE)
    assert created.status_code == 200, created.text
    assert created.json()["result"]["id"] == SID
    assert created.json()["result"]["body"] == CREATE["body"]
    assert calls[-1][:3] == ("POST", "/v1/messaging/scheduled", CREATE)
    assert calls[-1][3] == "Bearer token-1"
    listed = client.get(PREFIX, params={"tenant_id": "org-a"})
    assert listed.json()["result"]["scheduled"] == [state["row"]]
    patch = {"body": "edited fixture body", "send_at": "2030-04-05T14:30:00+02:00"}
    edited = client.patch(f"{PREFIX}/{SID}?tenant_id=org-a", json=patch)
    assert edited.status_code == 200, edited.text
    assert edited.json()["result"]["body"] == patch["body"]
    assert calls[-1][:3] == ("PATCH", f"/v1/messaging/scheduled/{SID}?tenant_id=org-a", patch)
    canceled = client.delete(f"{PREFIX}/{SID}?tenant_id=org-a")
    assert canceled.status_code == 200, canceled.text
    assert canceled.json()["result"] == {}
    assert state["row"] is None
    assert client.get(PREFIX + "?tenant_id=org-a").json()["result"] == {"scheduled": []}


@pytest.mark.parametrize(
    "change",
    [
        {"send_at": "tomorrow"},
        {"send_at": "2030-02-30T12:00:00Z"},
        {"send_at": "2030-04-05T12:30:00"},
        {"send_at": 123},
        {"send_at": "2030-04-05T12:30:00+00:99"},
        {"conversation_id": "../bad"},
        {"tenant_id": ""},
        {"client_ref": ""},
        {"body": ""},
        {"body": 42},
        {"sendAt": "typo"},
        {"mentions": [{"kind": "human", "account_id": "x", "extra": "bad"}]},
        # A thread root is held to the canonical message-id grammar: a
        # malformed id must be refused HERE, not by the hub's 400.
        {"thread_root_id": "not-a-message-id"},
        {"thread_root_id": "bcmsg_ABC12345"},
        {"thread_root_id": ""},
        {"thread_root_id": 7},
    ],
)
def test_invalid_create_never_reaches_hub(proxy, change):
    client, calls, _, _ = proxy
    response = client.post(PREFIX, json=CREATE | change)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["code"] == "invalid_request"
    assert CREATE["body"] not in response.text
    assert calls == []


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("GET", PREFIX, None),
        ("GET", PREFIX + "?tenant_id=org-a&tenant_id=org-b", None),
        ("GET", PREFIX + "?tenant_id=org-a&scope=account", None),
        ("PATCH", PREFIX + "/bad?tenant_id=org-a", {"body": "edit"}),
        ("DELETE", PREFIX + "/bad?tenant_id=org-a", None),
        ("PATCH", PREFIX + f"/{SID}?tenant_id=org-a", {}),
        ("PATCH", PREFIX + f"/{SID}?tenant_id=org-a", {"body": None}),
        ("PATCH", PREFIX + f"/{SID}?tenant_id=org-a", {"conversation_id": CID}),
        ("PATCH", PREFIX + f"/{SID}?tenant_id=org-a", {"send_at": "not-a-date"}),
        ("DELETE", PREFIX + f"/{SID}", None),
        ("POST", PREFIX + "?tenant_id=org-a", CREATE),
    ],
)
def test_invalid_selector_and_edit_never_reach_hub(proxy, method, path, body):
    client, calls, _, _ = proxy
    response = client.request(method, path, json=body)
    assert response.status_code == 422, response.text
    assert calls == []


def test_desktop_auth_negative_control(proxy):
    client, calls, _, _ = proxy
    response = client.get(PREFIX + "?tenant_id=org-a", headers={"authorization": "Bearer wrong"})
    assert response.status_code == 401
    assert calls == []


@pytest.mark.parametrize(
    "status,code",
    [
        (403, "tenant_forbidden"),
        (404, "message_not_found"),
        (409, "invalid_transition"),
        (400, "invalid_request"),
    ],
)
@pytest.mark.parametrize("method", ["GET", "POST", "PATCH", "DELETE"])
def test_server_refusals_survive_proxy(proxy, status, code, method):
    client, _, state, _ = proxy
    state["error"] = status, code
    path = PREFIX if method in {"GET", "POST"} else PREFIX + "/" + SID
    if method != "POST":
        path += "?tenant_id=wrong-tenant"
    response = client.request(
        method,
        path,
        json=CREATE if method == "POST" else {"body": "edit"} if method == "PATCH" else None,
    )
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    assert response.json()["detail"]["details"] == {"field": "send_at", "rule": "outside_window"}


def test_upstream_auth_refresh_once_then_signin(proxy):
    client, calls, state, store = proxy
    state["error"] = 401, "unauthorized"
    response = client.get(PREFIX + "?tenant_id=org-a")
    assert response.status_code == 401, response.text
    assert response.json()["detail"]["code"] == "bridge_signin_required"
    assert len(calls) == 2
    assert any(force for _, force in store.refresh_calls)


@pytest.mark.parametrize("resource", ["scheduled", "conversations"])
@pytest.mark.parametrize("malformed", ["missing-result", "list-result", "invalid-json"])
def test_invalid_upstream_success_is_bad_gateway(proxy, resource, malformed):
    client, calls, state, _ = proxy
    state["malformed"] = malformed
    response = client.get(f"/v1/desktop/team-bridge/{resource}?tenant_id=org-a")
    assert len(calls) == 1
    assert response.json()["detail"]["code"] == "bridge_invalid_response"
    assert response.status_code == 502, response.text


def test_a_scheduled_reply_forwards_its_thread_root(proxy):
    """A scheduled message can be a thread reply, on the SAME wire field name.

    Before this, ``ScheduledCreateIn`` was ``_Strict`` and had no root field, so
    a scheduled reply was not expressible through the desktop route at all —
    the client could carry ``thread_root_id`` but nothing upstream of it ever
    set one.
    """

    client, calls, _state, _ = proxy
    root = "bcmsg_root0001"
    payload = CREATE | {"thread_root_id": root}

    created = client.post(PREFIX, json=payload)

    assert created.status_code == 200, created.text
    assert calls[-1][:3] == ("POST", "/v1/messaging/scheduled", payload)
    # The wire body carries the root, under the same name ``channel_post`` uses.
    assert calls[-1][2]["thread_root_id"] == root


def test_a_schedule_without_a_thread_root_sends_no_such_field(proxy):
    """Absent stays absent: the hub's defaults must not be overwritten with a
    null the model never supplied (``exclude_unset`` is the reason)."""

    client, calls, _state, _ = proxy

    created = client.post(PREFIX, json=CREATE)

    assert created.status_code == 200, created.text
    assert "thread_root_id" not in calls[-1][2]
