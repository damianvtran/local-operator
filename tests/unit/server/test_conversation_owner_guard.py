"""Captured DATA-owner receipts through real ASGI routes, client and auth.db.

Only the remote hub is simulated: requests traverse desktop authentication,
Pydantic, route forwarding, real credential selection and TeamBridgeClient.
Synthetic OAuth credentials live in a disposable SQLite AuthStore, never the
operator's store. The remote ledger records actual dispatch side effects; its
membership refusal is deliberately separate from the local account receipt.
This is not evidence of a deployed Radient service or a renderer integration.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from local_operator.env import DEFAULT_RADIENT_API_BASE_URL
from local_operator.providers.auth_store import AuthStore
from local_operator.server.routes import desktop_team_bridge as routes
from local_operator.team_bridge.client import (
    BridgeRefusal,
    BridgeSignInRequired,
    TeamBridgeClient,
)

pytestmark = pytest.mark.asyncio
PREFIX = "/v1/desktop/team-bridge/conversations"
MEMBER = {"account_id": "acc_guest", "kind": "human", "display_label": "Guest"}
# Every write named by the capability, including the same send on retry and the
# read endpoint's notification-policy variant. Threads/reactions are NOT here.
CASES = [
    pytest.param("POST", "", {"kind": "dm", "members": [MEMBER]}, False, id="create"),
    pytest.param(
        "POST", "/bcv_01234567/messages", {"body": "hello", "client_ref": "ref_1"}, False, id="send"
    ),
    pytest.param(
        "POST",
        "/bcv_01234567/messages",
        {"body": "hello", "client_ref": "retry_1"},
        False,
        id="retry",
    ),
    pytest.param("POST", "/bcv_01234567/join", {}, False, id="join"),
    pytest.param("POST", "/bcv_01234567/read", {"seq": 5}, False, id="read"),
    pytest.param("POST", "/bcv_01234567/read", {"notify": "mute"}, False, id="notify"),
    pytest.param(
        "POST",
        "/bcv_01234567/members/acc_agent/consent",
        {"decision": "accept", "consent_version": "v1"},
        False,
        id="consent",
    ),
    pytest.param(
        "POST",
        "/bcv_01234567/messages/bcmsg_01234567/admit",
        {"verdict": "allow", "body_hash": "hash"},
        False,
        id="admit",
    ),
    pytest.param("POST", "/bcv_01234567/archive", None, True, id="archive"),
    pytest.param("POST", "/bcv_01234567/unarchive", None, True, id="unarchive"),
    pytest.param("POST", "/bcv_01234567/leave", None, True, id="leave"),
    pytest.param("POST", "/bcv_01234567/members", [MEMBER], True, id="add-members"),
    pytest.param("DELETE", "/bcv_01234567/members/acc_guest", None, True, id="remove-member"),
]


@pytest_asyncio.fixture
async def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "synthetic-desktop-token")
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path, config_overrides={})
    calls: list[httpx.Request] = []
    effects: list[tuple[str, str]] = []

    def login(owner):
        # Real logout/login against the disposable database, not current_access
        # patched to return a canned account. Separate request clients re-resolve.
        for row in store.list_credentials("radient"):
            store.delete_credential(row.id)
        return store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": owner,
                "org_id": "org_default",
                "access": f"synthetic-{owner}",
                "expires": 9999999999999,
            },
        )

    def upstream(request):
        calls.append(request)
        body = json.loads(request.content) if request.content else {}
        tenant = body.get("tenant_id") or request.url.params.get("tenant_id")
        # A second organization is VALID for the same account. OAuth org_id is
        # not the authority for target membership; only this upstream policy is.
        if tenant == "org_forbidden":
            return httpx.Response(403, json={"code": "forbidden", "error": "No membership"})
        effects.append((request.method, request.url.path))
        return httpx.Response(200, json={"msg": "ok", "result": {"ok": True}})

    def wire():
        return TeamBridgeClient(
            config_dir=tmp_path,
            base_url=DEFAULT_RADIENT_API_BASE_URL,
            store=store,
            transport=httpx.MockTransport(upstream),
        )

    monkeypatch.setattr(routes, "_conversation_client", lambda config: wire())
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_config_manager] = lambda: SimpleNamespace(
        config_dir=tmp_path
    )
    login("acc_a")
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://desktop",
            headers={"Authorization": "Bearer " + os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"]},
        ) as http:
            yield SimpleNamespace(
                http=http, store=store, login=login, calls=calls, effects=effects, wire=wire
            )
    finally:
        store.close()


def request_args(body, query, *, owner="acc_a", tenant="org_other", guarded=True):
    receipt = {"tenant_id": tenant} if tenant is not None else {}
    if guarded:
        receipt["expected_account_id"] = owner
    if query:
        return {"params": receipt, **({"json": body} if body is not None else {})}
    return {"json": {**body, **receipt}}


@pytest.mark.parametrize("method,path,body,query", CASES)
async def test_captured_owner_switch_refuses_before_dispatch(rig, method, path, body, query):
    args = request_args(body, query)
    rig.login("acc_b")  # DATA was captured by A; credentials now belong to B.
    response = await rig.http.request(method, PREFIX + path, **args)
    print(
        json.dumps(
            {
                "case": path or "create",
                "status": response.status_code,
                "response": response.json(),
                "outbound": len(rig.calls),
                "mutations": len(rig.effects),
            }
        )
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "account_mismatch"
    assert rig.calls == []
    assert rig.effects == []


@pytest.mark.parametrize("method,path,body,query", CASES)
async def test_matching_owner_allows_second_org_and_strips_receipt(rig, method, path, body, query):
    response = await rig.http.request(method, PREFIX + path, **request_args(body, query))
    assert response.status_code == 200, response.text
    print(
        json.dumps(
            {
                "case": path or "create",
                "status": response.status_code,
                "response": response.json(),
                "outbound": len(rig.calls),
                "mutations": len(rig.effects),
                "target": "org_other",
            }
        )
    )
    assert len(rig.effects) == len(rig.calls) == 1
    request = rig.calls[0]
    assert request.headers["authorization"] == "Bearer synthetic-acc_a"
    assert "expected_account_id" not in request.url.params
    assert b"expected_account_id" not in request.content
    upstream_body = json.loads(request.content) if request.content else {}
    assert (upstream_body.get("tenant_id") or request.url.params.get("tenant_id")) == "org_other"


@pytest.mark.parametrize("method,path,body,query", CASES)
@pytest.mark.parametrize("tenant", [None, "", "   "])
async def test_guard_requires_explicit_nonempty_tenant(rig, method, path, body, query, tenant):
    response = await rig.http.request(
        method, PREFIX + path, **request_args(body, query, tenant=tenant)
    )
    assert response.status_code == 422, response.text
    assert rig.calls == rig.effects == []


@pytest.mark.parametrize("method,path,body,query", CASES)
async def test_legacy_receipt_omission_remains_compatible(rig, method, path, body, query):
    response = await rig.http.request(
        method, PREFIX + path, **request_args(body, query, tenant=None, guarded=False)
    )
    assert response.status_code == 200, response.text
    assert len(rig.effects) == 1


async def test_desktop_auth_and_upstream_membership_are_independent_guards(rig):
    args = request_args({"body": "hello", "client_ref": "ref_1"}, False)
    response = await rig.http.post(
        PREFIX + "/bcv_01234567/messages", headers={"Authorization": "Bearer wrong"}, **args
    )
    assert response.status_code in (401, 403)
    assert rig.calls == rig.effects == []
    args["json"]["tenant_id"] = "org_forbidden"
    response = await rig.http.post(PREFIX + "/bcv_01234567/messages", **args)
    assert response.status_code == 403
    assert len(rig.calls) == 1
    assert rig.effects == []


async def test_send_retry_rechecks_owner_without_dispatching_retry(rig):
    args = request_args({"body": "hello", "client_ref": "same-ref"}, False)
    assert (await rig.http.post(PREFIX + "/bcv_01234567/messages", **args)).status_code == 200
    rig.login("acc_b")
    assert (await rig.http.post(PREFIX + "/bcv_01234567/messages", **args)).status_code == 409
    assert len(rig.calls) == len(rig.effects) == 1


async def test_supplied_snapshot_is_compared_and_never_reresolved(rig):
    wire = rig.wire()
    captured = await wire.current_access()
    rig.login("acc_b")
    assert await wire._guarded_access("acc_a", access=captured) is captured
    await wire.conversation_send(
        "bcv_01234567",
        client_ref="ref_1",
        text="hello",
        tenant_id="org_other",
        expected_account_id="acc_a",
        access=captured,
    )
    assert rig.calls[0].headers["authorization"] == "Bearer synthetic-acc_a"
    with pytest.raises(BridgeRefusal) as failure:
        await wire.conversation_send(
            "bcv_01234567",
            client_ref="ref_2",
            text="hello",
            tenant_id="org_other",
            expected_account_id="acc_b",
            access=captured,
        )
    assert failure.value.code == "account_mismatch"
    assert len(rig.calls) == 1


async def test_capability_names_the_guard_version():
    from local_operator.server.features import static_feature_flags

    assert static_feature_flags()["team_bridge_conversation_owner_guard"] == 1


async def test_notify_wrapper_also_checks_owner(rig):
    rig.login("acc_b")
    with pytest.raises(BridgeRefusal) as failure:
        await rig.wire().conversation_notify(
            "bcv_01234567",
            notify="mute",
            tenant_id="org_other",
            expected_account_id="acc_a",
        )
    assert failure.value.code == "account_mismatch"
    assert rig.calls == rig.effects == []


async def test_legacy_send_preserves_supplied_snapshot(rig):
    wire = rig.wire()
    captured = await wire.current_access()
    rig.login("acc_b")
    await wire.conversation_leave("bcv_01234567", access=captured)
    assert rig.calls[0].headers["authorization"] == "Bearer synthetic-acc_a"


async def test_refresh_cannot_retry_under_replacement_account(rig):
    calls = []

    def reject_and_switch(request):
        calls.append(request)
        rig.login("acc_b")
        return httpx.Response(401, json={"code": "unauthorized"})

    wire = rig.wire()
    wire._transport = httpx.MockTransport(reject_and_switch)
    with pytest.raises(BridgeSignInRequired):
        await wire.conversation_send(
            "bcv_01234567",
            client_ref="ref_1",
            text="hello",
            tenant_id="org_other",
            expected_account_id="acc_a",
        )
    assert len(calls) == 1
    assert calls[0].headers["authorization"] == "Bearer synthetic-acc_a"
    assert rig.effects == []
