"""H4 mounted desktop HTTP → real bridge client → synthetic loopback HTTP hub.

Both ends listen on disposable sockets. Only credential rows and hub semantics
are synthetic; routing, auth/origin/owner guards and HTTP serialization are real.
The tests prove transport, never live membership authorization or cascade effects.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from local_operator.config import ConfigManager
from local_operator.server.routes import capabilities, desktop_team_bridge
from local_operator.team_bridge import client as bridge
from tests.unit.server.test_desktop_thread_reads import Credentials

TOKEN = "h4-synthetic-desktop-token"
CAPS = ("team_bridge_endpoint_member_removal", "team_bridge_leave_withdraws_agent")
BASE = "/v1/desktop/team-bridge/conversations/bcv_abc12345"
MEMBER = BASE + "/members/acc_2"
OWNER = {"expected_account_id": "acc_1", "tenant_id": "tnt_1"}
ROSTER = {
    "conversation_id": "bcv_abc12345",
    "can_manage_members": False,
    "members": [
        {"account_id": "acc_2", "kind": "human", "can_remove": False},
        {
            "account_id": "acc_2",
            "kind": "agent",
            "can_remove": True,
            "agent": {"endpoint_id": "bep_agent2"},
        },
    ],
}


@contextmanager
def listening(app):
    """Own the socket/thread, including failed-start teardown; never a user PID."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", lifespan="off", access_log=False)
    )
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started, "loopback server failed preflight"
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        assert not thread.is_alive(), "owned loopback server did not stop"


@pytest.fixture
def wire(tmp_path):
    root = tmp_path
    patch = pytest.MonkeyPatch()
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            patch.delenv(name)
    patch.setenv("HOME", str(root))
    patch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root / "config"))
    patch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    patch.setenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", "https://desktop.example")
    patch.setattr(bridge, "org_oauth_destination_allowed", lambda base: True)
    state = SimpleNamespace(seen=[], error=None, status={}, result=ROSTER)
    hub = FastAPI()

    @hub.api_route("/v1/{rest:path}", methods=["GET", "POST", "DELETE"])
    async def upstream(rest: str, request: Request):
        body = await request.body()
        state.seen.append(
            {
                "method": request.method,
                "path": request.url.path,
                "query": list(request.query_params.multi_items()),
                "body": body.decode(),
                "auth_ok": request.headers.get("authorization") == "Bearer mock-hub-bearer",
            }
        )
        if rest == "messaging/status" and state.error is None:
            return {"result": state.status}
        if state.error:
            status, code = state.error
            return JSONResponse({"error": "synthetic refusal", "code": code}, status_code=status)
        return {"result": state.result}

    try:
        with listening(hub) as hub_url:
            core = FastAPI()
            core.state.config_manager = ConfigManager(root / "config")
            core.include_router(desktop_team_bridge.router)
            core.include_router(capabilities.router)
            with listening(core) as core_url:
                state.hub_url = hub_url
                state.core_url = core_url
                state.root = root
                yield state
    finally:
        patch.undo()


@pytest.fixture
def proxy(wire, monkeypatch):
    wire.seen.clear()
    wire.error = None
    wire.result = ROSTER
    wire.status = {
        "api_version": "1",
        "available": True,
        "tenants": [{"tenant_id": "tnt_1", "eligible": True, "endpoint": None}],
        "features": dict.fromkeys(CAPS, 1),
    }
    store = Credentials()
    discipline = bridge.StatusDiscipline(
        wire.root / "config", base_url=wire.hub_url + "/v1", store=store, read_only=True
    )
    monkeypatch.setattr(bridge, "get_status_discipline", lambda *a, **kw: discipline)
    monkeypatch.setattr(desktop_team_bridge, "_conversation_client", lambda cm: discipline._client)
    with httpx.Client(
        base_url=wire.core_url, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=10
    ) as client:
        yield SimpleNamespace(client=client, wire=wire, store=store, discipline=discipline)


def test_endpoint_and_legacy_human_delete_wire_and_echo(proxy):
    for endpoint in (None, "bep_agent2"):
        params = {**OWNER}
        if endpoint is not None:
            params["endpoint_id"] = endpoint
        response = proxy.client.delete(MEMBER, params=params)
        assert response.status_code == 200, response.text
        assert response.json()["result"] == ROSTER
        got = proxy.wire.seen[-1]
        assert got == {
            "method": "DELETE",
            "path": "/v1/messaging/conversations/bcv_abc12345/members/acc_2",
            "query": [("tenant_id", "tnt_1")]
            + ([] if endpoint is None else [("endpoint_id", endpoint)]),
            "body": "",
            "auth_ok": True,
        }
        print("DELETE", response.status_code, response.text, json.dumps(got))


def test_leave_shape_and_false_roster_projections(proxy):
    for method, path in (("POST", BASE + "/leave"), ("GET", BASE)):
        response = proxy.client.request(
            method, path, params=OWNER if method == "POST" else {"tenant_id": "tnt_1"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["result"] == ROSTER
        if method == "POST":
            assert proxy.wire.seen[-1] == {
                "method": "POST",
                "path": "/v1/messaging/conversations/bcv_abc12345/leave",
                "query": [("tenant_id", "tnt_1")],
                "body": "",
                "auth_ok": True,
            }
        print("LEAVE/READ", response.status_code, response.text)


@pytest.mark.parametrize("selector", ["", " ", "../evil", "bep/x", "bep%2Fx", "x" * 129, "💥"])
def test_malformed_endpoint_never_falls_back_to_human(proxy, selector):
    response = proxy.client.delete(MEMBER, params={**OWNER, "endpoint_id": selector})
    assert response.status_code == 422, response.text
    assert not proxy.wire.seen


@pytest.mark.parametrize(
    "query",
    [
        "endpoint_id=bep_a&endpoint_id=bep_b",
        "tenant_id=tnt_1&tenant_id=tnt_2",
        "tenant_id=",
        "expected_account_id=acc_1&expected_account_id=acc_2",
        "scope=account",
        "actor_endpoint_id=bep_actor",
        "unknown=1",
    ],
)
def test_ambiguous_or_unknown_queries_never_dispatch(proxy, query):
    response = proxy.client.delete(MEMBER + "?" + httpx.QueryParams(OWNER).__str__() + "&" + query)
    assert response.status_code == 422, response.text
    assert not proxy.wire.seen


@pytest.mark.parametrize(
    "params",
    [
        {"endpoint_id": "bep_a"},
        {"tenant_id": "tnt_1", "endpoint_id": "bep_a"},
        {"expected_account_id": "acc_1", "endpoint_id": "bep_a"},
        {**OWNER, "expected_account_id": "acc_2", "endpoint_id": "bep_a"},
    ],
)
def test_endpoint_owner_receipt_required_and_pinned(proxy, params):
    response = proxy.client.delete(MEMBER, params=params)
    assert response.status_code == (
        409 if params.get("expected_account_id") == "acc_2" else 422
    ), response.text
    assert not proxy.wire.seen


@pytest.mark.parametrize(
    "headers,status",
    [
        ({"Authorization": ""}, 401),
        ({"Authorization": "Bearer wrong"}, 401),
        ({"Origin": "https://evil.example"}, 403),
    ],
)
def test_desktop_auth_origin_negatives(proxy, headers, status):
    response = proxy.client.delete(
        MEMBER, params={**OWNER, "endpoint_id": "bep_a"}, headers=headers
    )
    assert response.status_code == status, response.text
    assert not proxy.wire.seen
    print("LOCAL NEGATIVE", response.status_code, response.text)


@pytest.mark.parametrize(
    "status,code",
    [
        (403, "not_a_member"),
        (404, "member_not_found"),
        (404, "conversation_not_found"),
        (422, "invalid_tenant"),
        (409, "invalid_transition"),
        (429, "rate_limited"),
        (503, "messaging_unavailable"),
    ],
)
def test_upstream_refusal_projection(proxy, status, code):
    proxy.wire.error = (status, code)
    for path, method in ((MEMBER, "DELETE"), (BASE + "/leave", "POST")):
        response = proxy.client.request(
            method,
            path,
            params={**OWNER, **({"endpoint_id": "bep_a"} if method == "DELETE" else {})},
        )
        assert response.status_code == status, response.text
        assert response.json()["detail"]["code"] == code
        print("UPSTREAM NEGATIVE", response.status_code, response.text)


@pytest.mark.parametrize("version", [None, False, 0, "1", 1.5, 2, 1])
def test_capabilities_exact_supported_version_and_zero_poll_io(proxy, version):
    def capabilities_now(headers=None):
        response = proxy.client.get("/v1/capabilities", headers=headers)
        assert response.status_code == 200, response.text
        return response.json()["result"]["features"]

    assert not any(key in capabilities_now() for key in CAPS)
    assert not proxy.wire.seen
    proxy.wire.status["features"] = {} if version is None else dict.fromkeys(CAPS, version)
    response = proxy.client.get("/v1/desktop/team-bridge/status")
    assert response.status_code == 200, response.text
    assert len(proxy.wire.seen) == 1
    refreshes = proxy.store.refreshes
    features = capabilities_now()
    for key in CAPS:
        if type(version) is int and version == 1:
            assert features[key] == 1
        else:
            assert key not in features
    for headers in (
        {"Authorization": ""},
        {"Authorization": "Bearer wrong"},
        {"Origin": "https://evil.example"},
    ):
        assert not any(key in capabilities_now(headers) for key in CAPS)
    assert len(proxy.wire.seen) == 1
    assert proxy.store.refreshes == refreshes
    proxy.store.rows[0].data["account_id"] = "acc_2"
    assert not any(key in capabilities_now() for key in CAPS)
    proxy.store.rows[0].data["account_id"] = "acc_1"
    proxy.discipline._states["acc_1"].last_attempt -= bridge.STATUS_CACHE_TTL_S
    assert not any(key in capabilities_now() for key in CAPS)
    proxy.wire.error = (503, "messaging_unavailable")
    proxy.client.get("/v1/desktop/team-bridge/status")
    assert not any(key in capabilities_now() for key in CAPS)
    print("CAPABILITIES", repr(version), {key: features.get(key) for key in CAPS})


def test_known_signed_out_retained_row_withholds_prior_success(proxy, monkeypatch):
    response = proxy.client.get("/v1/desktop/team-bridge/status")
    assert response.status_code == 200, response.text
    assert all(
        key in proxy.client.get("/v1/capabilities").json()["result"]["features"] for key in CAPS
    )

    async def failed_refresh(*args, **kwargs):
        return None

    monkeypatch.setattr(proxy.store, "ensure_oauth_fresh_or_raise", failed_refresh)
    response = proxy.client.get("/v1/desktop/team-bridge/status")
    assert response.json()["result"]["state"] == "signed_out", response.text
    assert proxy.store.rows[0].data["account_id"] == "acc_1", "row deliberately retained"
    features = proxy.client.get("/v1/capabilities").json()["result"]["features"]
    assert not any(key in features for key in CAPS)
    assert len(proxy.wire.seen) == 1


def test_h4_capabilities_are_independent_and_server_permission_false_survives(proxy):
    proxy.wire.status["features"] = {CAPS[0]: 1, CAPS[1]: False}
    proxy.client.get("/v1/desktop/team-bridge/status").raise_for_status()
    features = proxy.client.get("/v1/capabilities").json()["result"]["features"]
    assert features[CAPS[0]] == 1
    assert CAPS[1] not in features
    # Authority is not inferred from these capabilities, kind, or creator role.
    response = proxy.client.get(BASE, params={"tenant_id": "tnt_1"})
    assert response.json()["result"]["can_manage_members"] is False
    assert response.json()["result"]["members"][0]["can_remove"] is False


def test_offline_status_withholds_last_success(proxy):
    proxy.client.get("/v1/desktop/team-bridge/status").raise_for_status()
    assert all(
        key in proxy.client.get("/v1/capabilities").json()["result"]["features"] for key in CAPS
    )
    proxy.discipline._states["acc_1"].last_attempt -= bridge.STATUS_CACHE_TTL_S
    # A bound but non-listening owned socket is unreachable without racing
    # another session for a port or sending a request beyond loopback.
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        proxy.discipline._client._base_url = f"http://127.0.0.1:{unavailable.getsockname()[1]}/v1"
        response = proxy.client.get("/v1/desktop/team-bridge/status")
        assert response.status_code == 200, response.text
        assert response.json()["result"]["state"] == "unavailable", response.text
        features = proxy.client.get("/v1/capabilities").json()["result"]["features"]
        assert not any(key in features for key in CAPS)
    assert len(proxy.wire.seen) == 1
