"""Desktop typing-indicator routes through the real ASGI router, client and auth.db.

Only the hub is simulated, at its contract boundary (agent-server §18): each
request traverses desktop authentication, Pydantic, the route, real credential
selection from a disposable SQLite AuthStore, and TeamBridgeClient. ``calls``
records every request that actually left for the hub, which is how "refused
before any upstream byte" and "share_typing off makes no hub call" are proven.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Callable

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from local_operator.env import DEFAULT_RADIENT_API_BASE_URL
from local_operator.providers.auth_store import AuthStore
from local_operator.server.routes import desktop_team_bridge as routes
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.settings import TeamBridgeSettings, record_share_typing

pytestmark = pytest.mark.asyncio

PATH = "/v1/desktop/team-bridge/conversations/bcv_01234567/typing"
ROOT = "bcmsg_01234567"
RECEIPT = {"tenant_id": "org_1", "expected_account_id": "acc_a"}
CADENCE = {"ttl_ms": 8000, "resend_after_ms": 3000}


@pytest_asyncio.fixture
async def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "synthetic-desktop-token")
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path, config_overrides={})
    calls: list[httpx.Request] = []
    state: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        "answer": lambda request: httpx.Response(200, json={"msg": "ok", "result": CADENCE})
    }

    def login(owner: str) -> None:
        for row in store.list_credentials("radient"):
            store.delete_credential(row.id)
        store.upsert_credential(
            "radient",
            {
                "type": "oauth",
                "account_id": owner,
                "org_id": "org_default",
                "access": f"synthetic-{owner}",
                "expires": 9999999999999,
            },
        )

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return state["answer"](request)

    def wire() -> TeamBridgeClient:
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

            def answer(fn: Callable[[httpx.Request], httpx.Response]) -> None:
                state["answer"] = fn

            yield SimpleNamespace(
                http=http, calls=calls, login=login, answer=answer, config_dir=tmp_path
            )
    finally:
        store.close()


# -- happy paths -----------------------------------------------------------------


async def test_put_forwards_body_without_the_receipt_and_answers_cadence(rig):
    response = await rig.http.put(PATH, json={**RECEIPT, "thread_root_id": ROOT})
    assert response.status_code == 200, response.text
    assert response.json()["result"] == CADENCE
    (sent,) = rig.calls
    assert sent.method == "PUT"
    assert sent.url.path.endswith("/messaging/conversations/bcv_01234567/typing")
    assert json.loads(sent.content) == {"tenant_id": "org_1", "thread_root_id": ROOT}


async def test_get_passes_the_hub_json_through(rig):
    view: dict[str, Any] = {
        "enabled": True,
        "poll_after_ms": 3000,
        "message_seq": 7,
        "truncated": False,
        "typers": [
            {"account_id": "acc_x", "kind": "human", "thread_root_id": ROOT, "expires_in_ms": 5000}
        ],
    }
    rig.answer(lambda r: httpx.Response(200, json={"msg": "ok", "result": view}))
    response = await rig.http.get(PATH, params={"tenant_id": "org_1"})
    assert response.status_code == 200, response.text
    assert response.json()["result"] == view
    assert dict(rig.calls[0].url.params) == {"tenant_id": "org_1"}


async def test_delete_is_204_with_fields_in_the_upstream_query(rig):
    rig.answer(lambda r: httpx.Response(204))
    response = await rig.http.delete(PATH, params={**RECEIPT, "thread_root_id": ROOT})
    assert response.status_code == 204, response.text
    assert not response.content
    (sent,) = rig.calls
    assert sent.method == "DELETE"
    assert dict(sent.url.params) == {"tenant_id": "org_1", "thread_root_id": ROOT}


# -- owner guard ---------------------------------------------------------------------


@pytest.mark.parametrize("method", ["PUT", "DELETE"])
async def test_owner_mismatch_refuses_before_any_upstream_byte(rig, method):
    rig.login("acc_b")  # captured as A, signed in as B
    args = {"json": RECEIPT} if method == "PUT" else {"params": RECEIPT}
    response = await rig.http.request(method, PATH, **args)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "account_mismatch"
    assert rig.calls == []


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"tenant_id": "org_1"}, id="no-receipt"),
        pytest.param({"expected_account_id": "acc_a"}, id="no-tenant"),
        pytest.param({**RECEIPT, "tenant_id": "  "}, id="blank-tenant"),
        pytest.param({**RECEIPT, "thread_root_id": "../x"}, id="bad-root"),
        pytest.param({**RECEIPT, "endpoint_id": "bep_1"}, id="agent-endpoint"),
    ],
)
async def test_put_refuses_an_unguarded_or_malformed_body(rig, body):
    response = await rig.http.put(PATH, json=body)
    assert response.status_code == 422, response.text
    assert rig.calls == []


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({"tenant_id": "org_1"}, id="no-receipt"),
        pytest.param({**RECEIPT, "thread_root_id": "../x"}, id="bad-root"),
        pytest.param({**RECEIPT, "endpoint_id": "bep_1"}, id="unknown-name"),
    ],
)
async def test_delete_refuses_an_unguarded_or_malformed_query(rig, params):
    response = await rig.http.delete(PATH, params=params)
    assert response.status_code == 422, response.text
    assert rig.calls == []


async def test_get_requires_a_tenant_and_a_closed_query(rig):
    assert (await rig.http.get(PATH)).status_code == 422
    assert (await rig.http.get(PATH, params={"tenant_id": "org_1", "x": "1"})).status_code == 422
    assert rig.calls == []


# -- share_typing ------------------------------------------------------------------


async def test_share_typing_off_answers_shared_false_without_calling_the_hub(rig):
    record_share_typing(rig.config_dir, False)
    response = await rig.http.put(PATH, json=RECEIPT)
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"shared": False}
    assert rig.calls == []


async def test_share_typing_off_still_lets_a_stop_through(rig):
    """Removing state is always safe; an opt-out mid-draft must clear its entry."""
    record_share_typing(rig.config_dir, False)
    rig.answer(lambda r: httpx.Response(204))
    response = await rig.http.delete(PATH, params=RECEIPT)
    assert response.status_code == 204
    assert len(rig.calls) == 1


# -- hub error shapes ----------------------------------------------------------------


@pytest.mark.parametrize("status", [404, 405])
@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE"])
async def test_an_old_hub_is_outdated_hub_409(rig, status, method):
    rig.answer(lambda r: httpx.Response(status, json={"message": "Not Found"}))
    args: dict[str, Any] = {
        "GET": {"params": {"tenant_id": "org_1"}},
        "PUT": {"json": RECEIPT},
        "DELETE": {"params": RECEIPT},
    }[method]
    response = await rig.http.request(method, PATH, **args)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "outdated_hub"


async def test_a_coded_not_found_passes_through_as_404(rig):
    rig.answer(
        lambda r: httpx.Response(
            404, json={"code": "conversation_not_found", "error": "Conversation not found"}
        )
    )
    response = await rig.http.get(PATH, params={"tenant_id": "org_1"})
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "conversation_not_found"


async def test_a_503_passes_through(rig):
    rig.answer(lambda r: httpx.Response(503, json={"code": "bridge_disabled", "error": "dark"}))
    response = await rig.http.put(PATH, json=RECEIPT)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "bridge_disabled"


async def test_a_429_carries_retry_after(rig):
    rig.answer(
        lambda r: httpx.Response(
            429, json={"code": "rate_limited", "error": "slow"}, headers={"Retry-After": "11"}
        )
    )
    response = await rig.http.get(PATH, params={"tenant_id": "org_1"})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "11"
    assert response.json()["detail"]["retry_after_seconds"] == 11


# -- settings round trip --------------------------------------------------------------


async def test_share_typing_only_settings_put_is_local_and_round_trips(rig, monkeypatch):
    async def no_hub(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a share_typing-only write must not reach the hub")

    monkeypatch.setattr(TeamBridgeClient, "put_settings", no_hub)
    url = "/v1/desktop/team-bridge/settings"
    response = await rig.http.put(url, json={"expected_account_id": "acc_a", "share_typing": False})
    assert response.status_code == 200, response.text
    assert response.json()["result"] == {"share_typing": False}
    from local_operator.config import ConfigManager

    assert TeamBridgeSettings.from_config(ConfigManager(rig.config_dir)).share_typing is False

    async def hub_settings(self: Any, **kwargs: Any) -> dict[str, Any]:
        return {"endpoint": None}

    monkeypatch.setattr(TeamBridgeClient, "settings", hub_settings)
    got = await rig.http.get(url)
    assert got.status_code == 200, got.text
    assert got.json()["result"]["share_typing"] is False

    back = await rig.http.put(url, json={"expected_account_id": "acc_a", "share_typing": True})
    assert back.json()["result"] == {"share_typing": True}
    assert rig.calls == []


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"expected_account_id": "acc_a"}, id="neither-state-nor-pref"),
        pytest.param(
            {"expected_account_id": "acc_a", "share_typing": False, "label": "x"},
            id="pref-with-hub-field",
        ),
    ],
)
async def test_a_stateless_settings_put_is_only_the_preference(rig, body):
    response = await rig.http.put("/v1/desktop/team-bridge/settings", json=body)
    assert response.status_code == 422, response.text


async def test_share_typing_defaults_on_and_fails_closed_on_junk():
    assert TeamBridgeSettings.from_values({}).share_typing is True
    assert TeamBridgeSettings.from_values({"team_bridge": {"share_typing": True}}).share_typing
    for junk in (False, None, "yes", 1):
        values = {"team_bridge": {"share_typing": junk}}
        assert TeamBridgeSettings.from_values(values).share_typing is False, junk


async def test_capability_is_advertised_with_its_guard():
    from local_operator.server.features import static_feature_flags

    assert static_feature_flags()["team_bridge_typing"] == 1
    assert routes.ConversationTypingIn.model_fields["expected_account_id"].is_required()
    assert routes.ConversationTypingIn.model_fields["tenant_id"].is_required()
    assert "expected_account_id" in routes._TYPING_DELETE_QUERY
