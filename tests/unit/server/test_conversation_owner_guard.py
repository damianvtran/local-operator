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
    # Irreversible, so it carries the same captured-owner receipt as every other write.
    pytest.param("DELETE", "/bcv_01234567", None, True, id="delete"),
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


def request_args(body, query, *, owner="acc_a", tenant: str | None = "org_other", guarded=True):
    # None deliberately omits the target: invalid for guarded requests, but a
    # supported legacy shape. Keep it distinct from the empty/blank negatives.
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


# -- threads and reactions: REQUIRED receipts (team_bridge_threads/_reactions) --
#
# These routes were held closed in the desktop UI until they carried this guard,
# and the two capabilities are advertised only because they do. Unlike CASES
# above, the receipt is REQUIRED here (no legacy omission), so each case below is
# driven through the same real router -> client -> disposable AuthStore path,
# with only the hub simulated. `reaction` covers PUT (body) and DELETE (query);
# `replies` is the thread read, whose answer would otherwise fill the renderer's
# thread cache with the wrong account's thread.

REACTION_PATH = "/bcv_01234567/messages/bcmsg_01234567/reactions/%F0%9F%91%8D"
REPLIES_PATH = "/bcv_01234567/messages/bcmsg_01234567/replies"
GUARDED_THREAD_CASES = [
    pytest.param("PUT", REACTION_PATH, False, id="react-add"),
    pytest.param("DELETE", REACTION_PATH, True, id="react-remove"),
    pytest.param("GET", REPLIES_PATH, True, id="replies"),
]


def guarded_args(query, *, owner="acc_a", tenant="org_other"):
    receipt = {"tenant_id": tenant, "expected_account_id": owner}
    return {"params": receipt} if query else {"json": receipt}


@pytest.mark.parametrize("method,path,query", GUARDED_THREAD_CASES)
async def test_thread_and_reaction_owner_mismatch_refuses_before_dispatch(rig, method, path, query):
    """Captured as A, signed in as B: coded refusal, zero hub requests."""
    rig.login("acc_b")
    response = await rig.http.request(method, PREFIX + path, **guarded_args(query))
    print(
        json.dumps(
            {
                "case": method + " " + path,
                "status": response.status_code,
                "response": response.json(),
                "outbound": len(rig.calls),
            }
        )
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "account_mismatch"
    assert rig.calls == rig.effects == []


@pytest.mark.parametrize("method,path,query", GUARDED_THREAD_CASES)
async def test_thread_and_reaction_switch_mid_request_refuses(
    rig, monkeypatch, method, path, query
):
    """The account changes AFTER the route accepted the request, BEFORE dispatch.

    The receipt is valid when the route reads it (A is signed in), then the
    machine signs in as B while the request is in flight -- the delayed click or
    retry the UI hold existed for. The guard runs against the bearer snapshot
    the dispatch would use, so the request is refused rather than sent as B.
    """

    real_wire = rig.wire

    def wire_then_switch():
        client = real_wire()
        rig.login("acc_b")  # the switch lands between route validation and dispatch
        return client

    monkeypatch.setattr(routes, "_conversation_client", lambda config: wire_then_switch())
    response = await rig.http.request(method, PREFIX + path, **guarded_args(query))
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "account_mismatch"
    assert rig.calls == rig.effects == []


@pytest.mark.parametrize("method,path,query", GUARDED_THREAD_CASES)
async def test_thread_and_reaction_matched_owner_dispatches_and_strips_receipt(
    rig, method, path, query
):
    """Matched A: exactly one hub request as A, to the target org, no receipt upstream."""
    response = await rig.http.request(method, PREFIX + path, **guarded_args(query))
    assert response.status_code == 200, response.text
    assert len(rig.calls) == len(rig.effects) == 1
    request = rig.calls[0]
    assert rig.effects[0] == (method, request.url.path)
    assert request.headers["authorization"] == "Bearer synthetic-acc_a"
    assert "expected_account_id" not in request.url.params
    assert b"expected_account_id" not in request.content
    upstream_body = json.loads(request.content) if request.content else {}
    assert (upstream_body.get("tenant_id") or request.url.params.get("tenant_id")) == "org_other"


@pytest.mark.parametrize("method,path,query", GUARDED_THREAD_CASES)
@pytest.mark.parametrize(
    "receipt",
    [
        pytest.param({}, id="none"),
        pytest.param({"tenant_id": "org_other"}, id="tenant-only"),
        pytest.param({"expected_account_id": "acc_a"}, id="owner-only"),
        pytest.param({"tenant_id": "  ", "expected_account_id": "acc_a"}, id="blank-tenant"),
    ],
)
async def test_thread_and_reaction_receipt_is_required(rig, method, path, query, receipt):
    """No legacy omission on this plane: an unguarded call never reaches the hub."""
    args = {"params": receipt} if query else {"json": receipt}
    response = await rig.http.request(method, PREFIX + path, **args)
    assert response.status_code == 422, response.text
    assert rig.calls == rig.effects == []


async def test_reaction_retry_after_switch_is_refused(rig):
    """A react that succeeded as A, retried after B signs in, never reacts as B."""
    args = guarded_args(False)
    assert (await rig.http.put(PREFIX + REACTION_PATH, **args)).status_code == 200
    rig.login("acc_b")
    assert (await rig.http.put(PREFIX + REACTION_PATH, **args)).status_code == 409
    assert len(rig.calls) == len(rig.effects) == 1


# -- the threaded send: a thread MUTATION, so the receipt is REQUIRED ----
#
# Round-1 review R1 found `POST .../messages` still dispatched a reply with
# `thread_root_id` after an account switch, because the send DTO inherited the
# legacy OPTIONAL receipt. `team_bridge_threads` advertises the thread surface,
# so the mutation half is required here exactly as the read half is; an ordinary
# top-level send (no `thread_root_id`) keeps the legacy shape.
THREADED_SEND = {
    "body": "reply in thread",
    "client_ref": "ref_t1",
    "thread_root_id": "bcmsg_01234567",
}


@pytest.mark.parametrize(
    "receipt",
    [
        pytest.param({}, id="none"),
        pytest.param({"tenant_id": "org_other"}, id="tenant-only"),
        pytest.param({"expected_account_id": "acc_a"}, id="owner-only"),
        pytest.param({"tenant_id": "  ", "expected_account_id": "acc_a"}, id="blank-tenant"),
    ],
)
async def test_threaded_reply_send_requires_receipt(rig, receipt):
    """A pre-fix client cannot mutate the thread surface: 422 before any I/O."""
    response = await rig.http.post(
        PREFIX + "/bcv_01234567/messages", json={**THREADED_SEND, **receipt}
    )
    assert response.status_code == 422, response.text
    assert rig.calls == rig.effects == []


async def test_threaded_reply_after_switch_is_refused(rig):
    """Composed as A, sent after B signs in: never posted as B."""
    args = {"json": {**THREADED_SEND, "tenant_id": "org_other", "expected_account_id": "acc_a"}}
    assert (await rig.http.post(PREFIX + "/bcv_01234567/messages", **args)).status_code == 200
    rig.login("acc_b")
    assert (await rig.http.post(PREFIX + "/bcv_01234567/messages", **args)).status_code == 409
    assert len(rig.calls) == len(rig.effects) == 1


async def test_ordinary_top_level_send_keeps_legacy_receipt_omission(rig):
    """The thread guard must not tighten the ordinary send an old renderer makes."""
    response = await rig.http.post(
        PREFIX + "/bcv_01234567/messages", json={"body": "hi", "client_ref": "ref_plain"}
    )
    assert response.status_code == 200, response.text
    assert len(rig.effects) == 1


async def test_thread_and_reaction_capabilities_ship_with_their_guard():
    """The keys PROMISE the required receipt; they must not outlive it.

    Two halves. This one is a STATIC contract check — the DTO field and the two
    allowed-name sets plus each handler's own source calling the guard helper —
    because those are the things that can be checked without a request. The
    behavioural proof that a receipt-less call is refused (422) and a mismatched
    one refuses to dispatch (409) is `test_thread_and_reaction_receipt_is_required`
    and the mismatch cases above, which drive the real router. Neither half alone
    is sufficient: this one catches a handler or DTO that stopped consulting the
    guard, the end-to-end ones catch a guard that stopped refusing.
    """

    import inspect

    from local_operator.server.features import static_feature_flags

    flags = static_feature_flags()
    assert flags["team_bridge_threads"] == 1
    assert flags["team_bridge_reactions"] == 1
    assert routes.ConversationReactIn.model_fields["expected_account_id"].is_required()
    assert "expected_account_id" in routes._REPLIES_QUERY
    assert "expected_account_id" in routes._REACTION_DELETE_QUERY
    # A threaded reply is a thread MUTATION, so the send DTO refuses one without
    # the receipt (round-1 review R1: it used to dispatch receipt-less).
    with pytest.raises(ValueError):
        routes.ConversationSendIn(client_ref="ref_1", body="hi", thread_root_id="bcmsg_01234567")
    # The routes that keep the promise must actually call the guard helper; a
    # handler that stopped consulting it would otherwise only be caught by the
    # end-to-end cases, and only for the shapes they cover.
    for handler in (routes.conversation_replies, routes.conversation_react_remove):
        assert "_required_owner_query(" in inspect.getsource(handler), handler.__name__
    # ...and keep forwarding the receipt to the client, which is where the
    # atomic compare against the dispatch snapshot happens.
    from local_operator.team_bridge.client import TeamBridgeClient

    for method in ("conversation_replies", "conversation_react", "conversation_send"):
        assert (
            "expected_account_id" in inspect.signature(getattr(TeamBridgeClient, method)).parameters
        ), method
