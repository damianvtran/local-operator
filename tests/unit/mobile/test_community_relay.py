"""Relay web's ``/api/community/*`` door over the REAL daemon app (``build_app``),
with only the hub faked (``tests/unit/community/fakehub.py``).

Proves the RFC §7.2 relay rules: the daemon's existing gate comes first, the
route set is exactly the member allowlist (no staff operation over relay), the
switch keeps it dark, refusals carry codes and facts only, and a rejected or
oversized body is never echoed.
"""

from __future__ import annotations

import base64

import httpx
import pytest
from starlette.routing import Route
from starlette.testclient import TestClient

from local_operator.community import COMMUNITY_ENV
from local_operator.community.member_ops import MEMBER_OPERATIONS
from local_operator.mobile import community as relay
from local_operator.mobile.daemon import MobileDaemon, build_app
from tests.unit.community.fakehub import KEY, PNG, FakeHub, FakeStore, make_client

BASE = "/api/community/communities"
SECRET = "SECRET-DRAFT-5d10"


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setenv(COMMUNITY_ENV, "1")
    hub = FakeHub()
    store = FakeStore()
    monkeypatch.setattr(
        relay, "_client", lambda expected: make_client(hub, store, expected_account_id=expected)
    )
    app = build_app(MobileDaemon(port=0, password="pw123"))
    client = TestClient(app, follow_redirects=False)
    return client, hub, app


def _login(client: TestClient) -> None:
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)


def test_the_gate_comes_first(rig) -> None:
    client, hub, _ = rig
    assert client.get(BASE).status_code == 401
    assert client.post(BASE + "/local-operator/join", json={}).status_code == 401
    assert hub.sent == []


def test_the_relay_route_set_is_exactly_the_member_allowlist(rig) -> None:
    _, _, app = rig
    mounted = sorted(
        (sorted(r.methods - {"HEAD"})[0], r.path)
        for r in app.routes
        if isinstance(r, Route) and r.path.startswith(relay.PREFIX)
    )
    expected = sorted(
        (op.method, relay.PREFIX + relay._starlette_path(op.path)) for op in MEMBER_OPERATIONS
    )
    assert mounted == expected


def test_dark_by_default(rig, monkeypatch) -> None:
    client, hub, _ = rig
    _login(client)
    monkeypatch.delenv(COMMUNITY_ENV)
    response = client.get(BASE)
    assert response.status_code == 404
    assert response.json() == {"error": "community_disabled", "code": "community_disabled"}
    assert hub.sent == []


def test_member_operations_round_trip(rig) -> None:
    client, hub, _ = rig
    _login(client)
    listed = client.get(BASE)
    assert listed.status_code == 200 and listed.json()["featured"][0]["id"] == "local-operator"
    assert listed.headers["cache-control"] == "no-store"
    joined = client.post(
        BASE + "/local-operator/join",
        json={"guidelines_version": "2026-11-01", "accept_guidelines": True},
    )
    assert joined.status_code == 200 and joined.json()["membership"]["status"] == "active"
    image = {"content_type": "image/png", "data_base64": base64.b64encode(PNG).decode()}
    posted = client.post(
        BASE + "/local-operator/conversations/conv_general/messages",
        json={"text": "hi", "images": [image]},
    )
    assert posted.status_code == 200 and posted.json()["state"] == "published"
    assert PNG in hub.sent[-1].body
    assert (
        client.post(BASE + f"/local-operator/submissions/{KEY}/cancel").json()["state"]
        == "cancelled"
    )
    assert client.delete(BASE + "/local-operator/messages/msg_1").json() == {"deleted": True}
    assert client.get(BASE + "/local-operator/events").json()["next"] == "e2"
    assert [s.path for s in hub.sent][-3:] == [
        f"/community/v1/communities/local-operator/submissions/{KEY}:cancel",
        "/community/v1/communities/local-operator/messages/msg_1",
        "/community/v1/communities/local-operator/events",
    ]


def test_refusals_carry_codes_and_facts_only(rig) -> None:
    client, hub, _ = rig
    _login(client)
    hub.override = lambda s: httpx.Response(
        503, json={"message": SECRET, "code": "moderation_unavailable", "retry_after": 30}
    )
    response = client.post(
        BASE + "/local-operator/conversations/conv_general/messages",
        json={"text": "x", "idempotency_key": KEY},
    )
    assert response.status_code == 503
    assert response.json() == {
        "error": "moderation_unavailable",
        "code": "moderation_unavailable",
        "retry_after": 30,
        "idempotency_key": KEY,
    }
    hub.override = lambda s: httpx.Response(
        429, headers={"Retry-After": "7"}, json={"message": SECRET}
    )
    limited = client.get(BASE + "/local-operator/channels")
    assert limited.status_code == 429 and limited.headers["retry-after"] == "7"
    assert SECRET not in limited.text


def test_rejected_and_oversized_bodies_are_never_echoed(rig) -> None:
    client, hub, _ = rig
    _login(client)
    path = BASE + "/local-operator/conversations/conv_general/messages"
    for body in ({"text": SECRET, "extra": SECRET}, {"text": SECRET * 400}):
        response = client.post(path, json=body)
        assert response.status_code == 422
        assert response.json() == {"error": "invalid_request", "code": "invalid_request"}
    bad_json = client.post(
        path, content=b"{" + SECRET.encode(), headers={"content-type": "application/json"}
    )
    assert bad_json.status_code == 422 and SECRET not in bad_json.text
    huge = client.post(path, content=b"x" * (relay.MAX_BODY_BYTES + 1))
    assert huge.status_code == 413 and huge.json()["code"] == "payload_too_large"
    assert (
        client.get(BASE + "/local-operator/channels", params={"filter": "staff"}).status_code == 422
    )
    assert hub.sent == []


def test_signin_required_is_never_a_401_on_the_relay(rig, monkeypatch) -> None:
    """R1-M1: relay web reads ANY 401 as a dead relay cookie (wipes drafts, reloads).

    A signed-out Radient account over a LIVE relay session must answer 409 with
    the code, so relay web branches on ``community_signin_required`` instead.
    """

    client, hub, _ = rig
    _login(client)
    signed_out = FakeStore(rows=[])
    monkeypatch.setattr(
        relay,
        "_client",
        lambda expected: make_client(hub, signed_out, expected_account_id=expected),
    )
    response = client.get(BASE)
    assert response.status_code == 409
    assert response.json() == {
        "error": "community_signin_required",
        "code": "community_signin_required",
    }
    assert hub.sent == []
    # The relay gate's own 401 is untouched: the session cookie is still good.
    assert client.get("/api/sessions").status_code != 401


def _every_refusal() -> list:
    """One instance of EVERY ``CommunityError`` type, plus a bare 401 base."""

    from local_operator.community import errors

    special = {
        errors.CommunityGuidelinesChanged: lambda: errors.CommunityGuidelinesChanged("v2"),
        errors.CommunityRefused: lambda: errors.CommunityRefused("spam", "ref_1"),
        errors.CommunityRateLimited: lambda: errors.CommunityRateLimited(5, "sender"),
        errors.CommunityModerationUnavailable: lambda: errors.CommunityModerationUnavailable(30),
    }
    kinds = [getattr(errors, name) for name in errors.__all__]
    made = [special[k]() if k in special else k() for k in kinds]
    # A type added later with 401 is covered by the status rule, not the list.
    made.append(errors.CommunityError("future_signin_state", http_status=401))
    return made


_BODIES = {
    "join": {"guidelines_version": "v1", "accept_guidelines": True},
    "submit": {"text": "x"},
    "edit": {"text": "x"},
    "react": {"emoji": "+1"},
}


def test_no_relay_route_answers_401_for_any_refusal(rig, monkeypatch) -> None:
    """The sweep behind R1-M1: every member route x every refusal type, never 401."""

    from local_operator.community import errors, member_ops

    client, hub, _ = rig
    _login(client)
    refusals = _every_refusal()
    # Every exported refusal type is in the table (so a new one cannot skip it).
    assert {type(e) for e in refusals} >= {getattr(errors, n) for n in errors.__all__}
    current: list[errors.CommunityError] = []

    async def refuse(*_args, **_kwargs):
        raise current[0]

    monkeypatch.setattr(member_ops, "run", refuse)
    params = {
        "community_id": "local-operator",
        "conversation_id": "conv_general",
        "idempotency_key": KEY,
        "message_id": "msg_1",
    }
    checked = 0
    for op in MEMBER_OPERATIONS:
        path = relay.PREFIX + op.path.format(**params)
        for error in refusals:
            current[:] = [error]
            response = client.request(op.method, path, json=_BODIES.get(op.name))
            assert response.status_code != 401, (op.name, error.code)
            assert response.json()["code"] == error.code, (op.name, error.code)
            checked += 1
    assert checked == len(MEMBER_OPERATIONS) * len(refusals)
    assert hub.sent == []
