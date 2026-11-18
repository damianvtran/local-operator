"""``/v1/desktop/community/*`` through the real router, desktop auth and the app's
own 422 scrubber, with only the hub faked (``tests/unit/community/fakehub.py``).

What is proven here, per the RFC §7.2 door contract: the route table IS the
member allowlist; the switch keeps it dark; each route reaches its contract
path; refusals keep their facts (and nothing of the hub's text); a rejected
body never comes back in a 422; and the router keeps httpx off the boot path.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from local_operator.community import COMMUNITY_ENV
from local_operator.community.member_ops import MEMBER_OPERATIONS
from local_operator.server.routes import desktop_community as routes
from tests.unit.community.fakehub import KEY, PNG, FakeHub, FakeStore, make_client

pytestmark = pytest.mark.asyncio

BASE = "/v1/desktop/community/communities"
SECRET = "SECRET-DRAFT-91c2"


@pytest_asyncio.fixture
async def rig(tmp_path, monkeypatch):
    from local_operator.server.app import desktop_validation_error

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "synthetic-desktop-token")
    monkeypatch.setenv(COMMUNITY_ENV, "1")
    hub = FakeHub()
    store = FakeStore()
    built: list[str | None] = []

    def wire(config_manager, expected_account_id):
        built.append(expected_account_id)
        return make_client(hub, store, expected_account_id=expected_account_id)

    monkeypatch.setattr(routes, "_client", wire)
    app = FastAPI()
    app.include_router(routes.router)
    # The production 422 handler, not a lookalike: the scrubbing under test is
    # the app's (`server/app.py` `desktop_validation_error`).
    app.add_exception_handler(RequestValidationError, desktop_validation_error)
    app.dependency_overrides[routes.get_config_manager] = lambda: SimpleNamespace(
        config_dir=tmp_path
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://desktop",
        headers={"Authorization": "Bearer " + os.environ["LOCAL_OPERATOR_DESKTOP_TOKEN"]},
    ) as http:
        yield SimpleNamespace(http=http, hub=hub, store=store, built=built)


async def test_the_route_table_is_exactly_the_member_allowlist() -> None:
    registered = sorted(
        (sorted(r.methods)[0], r.path) for r in routes.router.routes  # type: ignore[attr-defined]
    )
    expected = sorted((op.method, routes.PREFIX + op.path) for op in MEMBER_OPERATIONS)
    assert registered == expected
    # No staff/moderation operation of the contract has a local door.
    staff = ("role", "sanction", "case", "kill", "oversight", "lock", "archive", "reassign")
    assert not [p for _, p in registered if any(word in p for word in staff)]


async def test_the_router_keeps_httpx_off_the_boot_path(tmp_path: Path) -> None:
    """`lop serve` imports every router at start-up; the client stack must not."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "from local_operator.server.routes import desktop_community; "
            "from local_operator.mobile import community; "
            "assert 'httpx' not in sys.modules, 'community put httpx on the boot path'",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])},
    )
    assert result.returncode == 0, result.stderr


async def test_dark_by_default_and_nothing_leaves(rig, monkeypatch) -> None:
    monkeypatch.delenv(COMMUNITY_ENV)
    response = await rig.http.get(BASE)
    assert response.status_code == 404
    assert response.json()["detail"] == {"code": "community_disabled"}
    assert rig.hub.sent == []


async def test_the_desktop_bearer_is_required(rig) -> None:
    response = await rig.http.get(BASE, headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401
    assert rig.hub.sent == []


@pytest.mark.parametrize(
    ("method", "path", "json", "upstream"),
    [
        ("GET", "", None, ("GET", "/communities")),
        (
            "GET",
            "/local-operator/bootstrap",
            None,
            ("GET", "/communities/local-operator/bootstrap"),
        ),
        (
            "POST",
            "/local-operator/join",
            {"guidelines_version": "2026-11-01", "accept_guidelines": True},
            ("POST", "/communities/local-operator/join"),
        ),
        ("POST", "/local-operator/leave", None, ("POST", "/communities/local-operator/leave")),
        (
            "GET",
            "/local-operator/channels?filter=public",
            None,
            ("GET", "/communities/local-operator/directory/channels"),
        ),
        (
            "GET",
            "/local-operator/conversations/conv_general/messages?cursor=c1",
            None,
            ("GET", "/communities/local-operator/conversations/conv_general/messages"),
        ),
        (
            "POST",
            "/local-operator/conversations/conv_general/messages",
            {"text": "hi", "idempotency_key": KEY},
            ("POST", "/communities/local-operator/conversations/conv_general/submissions"),
        ),
        (
            "GET",
            f"/local-operator/submissions/{KEY}",
            None,
            ("GET", f"/communities/local-operator/submissions/{KEY}"),
        ),
        (
            "POST",
            f"/local-operator/submissions/{KEY}/cancel",
            None,
            ("POST", f"/communities/local-operator/submissions/{KEY}:cancel"),
        ),
        (
            "POST",
            "/local-operator/messages/msg_1/edit",
            {"text": "edited"},
            ("POST", "/communities/local-operator/messages/msg_1:edit"),
        ),
        (
            "DELETE",
            "/local-operator/messages/msg_1",
            None,
            ("DELETE", "/communities/local-operator/messages/msg_1"),
        ),
        (
            "POST",
            "/local-operator/messages/msg_1/reactions",
            {"emoji": "👍"},
            ("POST", "/communities/local-operator/messages/msg_1/reactions"),
        ),
        (
            "GET",
            "/local-operator/events?after=e1",
            None,
            ("GET", "/communities/local-operator/events"),
        ),
    ],
)
async def test_each_member_operation_reaches_its_contract_path(rig, method, path, json, upstream):
    response = await rig.http.request(method, BASE + path, json=json)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == 200
    (sent,) = rig.hub.sent
    assert (sent.method, sent.path) == (upstream[0], "/community/v1" + upstream[1])


async def test_submit_with_an_image_streams_it_upstream(rig) -> None:
    image = {"content_type": "image/png", "data_base64": base64.b64encode(PNG).decode()}
    response = await rig.http.post(
        BASE + "/local-operator/conversations/conv_general/messages",
        json={"text": "look", "images": [image]},
    )
    assert response.status_code == 200, response.text
    (sent,) = rig.hub.sent
    assert PNG in sent.body and b'filename="image-1.png"' in sent.body
    assert len(sent.headers["idempotency-key"]) == 32


async def test_refusals_keep_their_facts_and_a_rate_limit_its_header(rig) -> None:
    rig.hub.override = lambda s: httpx.Response(
        422,
        json={
            "status": "refused",
            "category": "harassment_or_abuse",
            "reference": "MOD-7K2Q-9XDF",
            "message": SECRET,
        },
    )
    path = BASE + "/local-operator/conversations/conv_general/messages"
    refused = await rig.http.post(path, json={"text": "x", "idempotency_key": KEY})
    assert refused.status_code == 422
    assert refused.json()["detail"] == {
        "code": "refused",
        "category": "harassment_or_abuse",
        "reference": "MOD-7K2Q-9XDF",
        "idempotency_key": KEY,
    }
    assert SECRET not in refused.text
    rig.hub.override = lambda s: httpx.Response(
        429, json={"message": "slow", "retry_after": 12, "scope": "account_burst"}
    )
    limited = await rig.http.post(path, json={"text": "x"})
    assert limited.status_code == 429 and limited.headers["retry-after"] == "12"
    detail = limited.json()["detail"]
    assert detail["retry_after"] == 12 and detail["scope"] == "account_burst"
    # The door minted the attempt's key and hands it back for a status read.
    assert len(detail["idempotency_key"]) == 32


async def test_join_guidelines_changed_and_explicit_acceptance(rig) -> None:
    path = BASE + "/local-operator/join"
    for body in (
        {"guidelines_version": "v1"},
        {"guidelines_version": "v1", "accept_guidelines": False},
    ):
        assert (await rig.http.post(path, json=body)).status_code == 422
    assert rig.hub.sent == []
    rig.hub.override = lambda s: httpx.Response(
        409, json={"message": "x", "code": "guidelines_changed", "guidelines_version": "v2"}
    )
    changed = await rig.http.post(
        path, json={"guidelines_version": "v1", "accept_guidelines": True}
    )
    assert changed.status_code == 409
    assert changed.json()["detail"] == {"code": "guidelines_changed", "guidelines_version": "v2"}


async def test_a_rejected_body_never_comes_back_in_the_422(rig) -> None:
    path = BASE + "/local-operator/conversations/conv_general/messages"
    for body in (
        {"text": SECRET, "unexpected": SECRET},
        {"text": SECRET * 400},  # over the 4000-character limit
        {"text": SECRET, "images": [{"content_type": "image/gif", "data_base64": SECRET}]},
    ):
        response = await rig.http.post(path, json=body)
        assert response.status_code == 422
        assert SECRET not in response.text
        assert response.json() == {"detail": "The request has invalid fields."}
    assert rig.hub.sent == []


async def test_the_owner_guard_refuses_before_any_request(rig) -> None:
    mismatch = await rig.http.get(BASE, params={"expected_account_id": "acc_other"})
    assert mismatch.status_code == 409
    assert mismatch.json()["detail"] == {"code": "account_mismatch"}
    assert rig.hub.sent == []
    ok = await rig.http.get(
        BASE + "/local-operator/channels", params={"expected_account_id": "acc_1", "q": "gen"}
    )
    assert ok.status_code == 200, ok.text
    assert rig.built[-1] == "acc_1"
    assert rig.hub.sent[-1].params == {"q": "gen"}


async def test_sign_in_and_unreachable_states(rig) -> None:
    rig.store.rows.clear()
    signin = await rig.http.get(BASE)
    assert signin.status_code == 401
    assert signin.json()["detail"] == {"code": "community_signin_required"}


async def test_the_capability_key_follows_the_switch(monkeypatch) -> None:
    """`features.community` is advertised exactly when the routes would serve."""

    from local_operator.server.features import feature_flags

    monkeypatch.delenv(COMMUNITY_ENV, raising=False)
    assert "community" not in feature_flags()
    monkeypatch.setenv(COMMUNITY_ENV, "0")
    assert "community" not in feature_flags()
    monkeypatch.setenv(COMMUNITY_ENV, "1")
    assert feature_flags()["community"] == 1
