"""The five agent-messages desktop routes: state shapes, refusals, and the gate.

The bridge CLIENT is faked here on purpose — its own behaviour (auth refresh,
refusal mapping, body rules) is exercised byte-for-byte against a scripted
transport in ``tests/unit/team_bridge``. These tests pin the ROUTE contract the
UI lane consumes: what a success answers, how each refusal class maps to a
status and a ``code``, and that the consent-conflict path invalidates the shared
status cache.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import desktop_team_bridge
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import (
    STATE_READY,
    STATE_SIGNED_OUT,
    AcceptedText,
    BridgeRefusal,
    BridgeSignInRequired,
    StatusCheck,
)

pytestmark = pytest.mark.asyncio

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakeClient:
    """Scripted per-method behaviour; records every constructed instance."""

    behavior: dict[str, Any] = {}
    instances: list["FakeClient"] = []

    def __init__(self, config_dir: Any = None, **kwargs: Any) -> None:
        self.config_dir = config_dir
        FakeClient.instances.append(self)

    def _answer(self, name: str) -> Any:
        value = FakeClient.behavior.get(name, {})
        if isinstance(value, Exception):
            raise value
        return value

    async def status(self, **kwargs: Any) -> dict[str, Any]:
        return self._answer("status")

    async def settings(self, *, tenant_id: str | None = None) -> dict[str, Any]:
        return self._answer("settings")

    async def put_settings(self, **kwargs: Any) -> dict[str, Any]:
        FakeClient.behavior["last_put_kwargs"] = kwargs
        return self._answer("put_settings")

    async def allowlist(self, *, tenant_id: str | None = None) -> dict[str, Any]:
        return self._answer("allowlist")

    async def put_allowlist(
        self,
        entries: Any,
        *,
        tenant_id: str | None = None,
        expected_account_id: str | None = None,
    ) -> dict[str, Any]:
        FakeClient.behavior["last_entries"] = list(entries)
        FakeClient.behavior["last_allowlist_kwargs"] = {
            "tenant_id": tenant_id,
            "expected_account_id": expected_account_id,
        }
        return self._answer("put_allowlist")


class FakeDiscipline:
    def __init__(self, check: StatusCheck) -> None:
        self.check_result = check
        self.boundaries: list[str] = []
        self.invalidated = 0

    async def check(self, boundary: Any) -> StatusCheck:
        self.boundaries.append(str(boundary))
        # Mirror the real discipline: the answer carries the boundary that
        # produced it (a cache hit re-labels the same payload).
        return replace(self.check_result, boundary=str(boundary))

    def invalidate(self) -> None:
        self.invalidated += 1


def _status_check(**overrides: Any) -> StatusCheck:
    base: dict[str, Any] = dict(
        state=STATE_READY,
        boundary="surface_open",
        payload={
            "api_version": "1",
            "available": True,
            "tenants": [
                {
                    "tenant_id": "tnt_1",
                    "tenant_name": "Org",
                    "eligible": True,
                    "endpoint": None,
                    "counts": {"unread": 0, "requests_pending": 0, "blocked": 0},
                }
            ],
            "consent": {"version": "bridge_consent_v1", "text": "the served text"},
        },
        from_cache=False,
        fetched_at=1.0,
        checked_at=1.0,
        error=None,
        account_id="acc_1",
        # The served consent is only displayable when it IS the accepted
        # accession (review round 1, R3): the fixture's payload and its
        # accepted record agree, as a real answer's do after acceptance.
        texts={
            "consent": AcceptedText(
                version="bridge_consent_v1",
                text="the served text",
                sha256="x",
                accepted_at=1.0,
            )
        },
    )
    base.update(overrides)
    return StatusCheck(**base)


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    root = tmp_path / ".local-operator"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    FakeClient.behavior = {}
    FakeClient.instances = []
    discipline = FakeDiscipline(_status_check())
    monkeypatch.setattr(tb_client, "TeamBridgeClient", FakeClient)
    monkeypatch.setattr(
        tb_client,
        "get_status_discipline",
        lambda config_dir=None, base_url=None: discipline,
    )
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.include_router(desktop_team_bridge.router)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost", headers=AUTH
    ) as client:
        yield client, root, discipline


async def test_status_composes_state_local_config_consent_and_the_screening_receipt(api):
    client, _root, discipline = api
    response = await client.get("/v1/desktop/team-bridge/status?boundary=app_start")
    assert response.status_code == 200
    result = response.json()["result"]

    assert result["state"] == STATE_READY
    assert result["boundary"] == "app_start"
    assert result["from_cache"] is False
    # The §5.6 reading the renderer binds to (computed once, backend-side).
    assert result["capability"] == 1
    assert result["surface_state"] == "B"
    assert result["tenant_id"] == "tnt_1"
    assert result["visibility"] == {
        "settings": True,
        "inbox": False,
        "account_door": False,
        "onboarding": True,
    }
    assert result["read_only"] is False
    assert result["can_receive"] is False
    assert result["can_decide"] is False
    # The captured-account guard receipt (R6) and the refused-slot record (R3).
    assert result["consent_account_guard"] is True
    assert result["state_note"] is None
    assert result["texts_rejected"] == []
    # The local keys the renderer composes its gate from.
    assert result["local"] == {"enabled": False, "onboarding": "", "session": ""}
    # The consent renders exactly as served, with its source named.
    assert result["consent"] == {
        "version": "bridge_consent_v1",
        "text": "the served text",
        "source": "served",
    }
    # Screening is an invariant receipt, never a setting (§8, §13.9).
    assert result["screening"] == {"required": True}
    assert result["error"] is None
    assert discipline.boundaries == ["app_start"]


async def test_status_reports_signed_out_as_a_state_not_an_error(api, monkeypatch):
    client, _root, discipline = api
    discipline.check_result = _status_check(
        state=STATE_SIGNED_OUT, payload=None, account_id=None, fetched_at=None
    )
    response = await client.get("/v1/desktop/team-bridge/status")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["state"] == STATE_SIGNED_OUT
    assert result["consent"] is None


async def test_status_carries_a_refusal_as_data(api):
    client, _root, discipline = api
    discipline.check_result = _status_check(
        state="unavailable",
        error=BridgeRefusal("messaging_unavailable", "not launched", http_status=503),
    )
    response = await client.get("/v1/desktop/team-bridge/status")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["state"] == "unavailable"
    assert result["error"]["code"] == "messaging_unavailable"


async def test_settings_get_passes_the_payload_through(api):
    client, _root, _discipline = api
    payload = {"endpoint": None, "settings": {"tenant_id": "tnt_1", "eligible": True}}
    FakeClient.behavior["settings"] = payload
    response = await client.get("/v1/desktop/team-bridge/settings?tenant_id=tnt_1")
    assert response.status_code == 200
    assert response.json()["result"] == payload


async def test_sign_in_required_maps_to_a_401_with_its_own_code(api):
    client, _root, _discipline = api
    FakeClient.behavior["settings"] = BridgeSignInRequired(
        "Sign in to Radient to use agent messages."
    )
    response = await client.get("/v1/desktop/team-bridge/settings")
    assert response.status_code == 401
    assert response.json()["detail"] == {
        "code": "bridge_signin_required",
        "message": "Sign in to Radient to use agent messages.",
    }


async def test_put_settings_requires_state_in_the_body(api):
    client, _root, _discipline = api
    response = await client.put("/v1/desktop/team-bridge/settings", json={"tenant_id": "tnt_1"})
    assert response.status_code == 422


async def test_put_settings_requires_the_captured_account_guard(api):
    """R6: an unguarded settings write is not a shape this plane accepts.

    Without the captured account the dialog-replay guard is impossible to
    enforce, so the desktop body REFUSES the request outright (422, empty
    body semantics) and no bridge client is ever constructed.
    """

    client, _root, _discipline = api
    response = await client.put(
        "/v1/desktop/team-bridge/settings",
        json={"state": "paused"},
    )
    assert response.status_code == 422
    assert FakeClient.instances == []


async def test_put_allowlist_requires_the_captured_account_guard(api):
    client, _root, _discipline = api
    response = await client.put(
        "/v1/desktop/team-bridge/allowlist",
        json={"entries": []},
    )
    assert response.status_code == 422
    assert FakeClient.instances == []


async def test_a_stale_captured_account_maps_to_account_mismatch(api):
    """R6/Q1: the client's guard refusal surfaces as a refresh-status signal."""

    client, _root, discipline = api
    FakeClient.behavior["put_settings"] = BridgeRefusal(
        "account_mismatch", "prepared for a different account", http_status=409
    )
    response = await client.put(
        "/v1/desktop/team-bridge/settings",
        json={"state": "paused", "expected_account_id": "acc_a"},
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "account_mismatch"
    # The staleness signal: re-read status before the user tries again.
    assert detail["refresh_status"] is True
    assert discipline.invalidated == 1


async def test_the_captured_account_reaches_the_client_verbatim(api):
    client, _root, _discipline = api
    FakeClient.behavior["put_settings"] = {"endpoint": None}
    response = await client.put(
        "/v1/desktop/team-bridge/settings",
        json={"state": "paused", "expected_account_id": "acc_7"},
    )
    assert response.status_code == 200
    assert FakeClient.behavior["last_put_kwargs"]["expected_account_id"] == "acc_7"


async def test_put_settings_local_precondition_failures_are_422_client_bugs(api):
    client, _root, _discipline = api
    FakeClient.behavior["put_settings"] = ValueError(
        "a pause or opt-out must be sent alone (RFC §5.2)"
    )
    response = await client.put(
        "/v1/desktop/team-bridge/settings",
        json={"state": "paused", "label": "My agent", "expected_account_id": "acc_1"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"


async def test_put_settings_consent_conflict_flags_refresh_and_invalidates_the_cache(api):
    client, _root, discipline = api
    FakeClient.behavior["put_settings"] = BridgeRefusal(
        "consent_version_unknown", "the consent version moved", http_status=409
    )
    response = await client.put(
        "/v1/desktop/team-bridge/settings",
        json={
            "state": "active",
            "consent_version": "bridge_consent_v1",
            "expected_account_id": "acc_1",
        },
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "consent_version_unknown"
    # §5.2: refresh status and re-render — the cached consent must not be served.
    assert detail["refresh_status"] is True
    assert discipline.invalidated == 1


async def test_put_settings_refusal_carries_details_and_retry_after(api):
    client, _root, _discipline = api
    FakeClient.behavior["put_settings"] = BridgeRefusal(
        "rate_limited",
        "slow down",
        http_status=429,
        details={"retry_after_seconds": 7},
        retry_after_seconds=7,
    )
    response = await client.put(
        "/v1/desktop/team-bridge/settings",
        json={"state": "paused", "expected_account_id": "acc_1"},
    )
    assert response.status_code == 429
    detail = response.json()["detail"]
    assert detail["retry_after_seconds"] == 7
    assert detail["details"] == {"retry_after_seconds": 7}


async def test_allowlist_get_composes_entries_and_the_view(api):
    client, _root, _discipline = api
    FakeClient.behavior["allowlist"] = {
        "allowlist": {
            "tenant_id": "tnt_1",
            "entries": [
                {"direction": "send", "effect": "allow", "target": {"endpoint_id": "bep_1"}}
            ],
        }
    }
    response = await client.get("/v1/desktop/team-bridge/allowlist")
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["allowlist"]["tenant_id"] == "tnt_1"
    assert result["view"]["directions"]["send"] == {
        "mode": "listed_only",
        "allow": 1,
        "deny": 0,
    }
    assert result["view"]["cues"]["removing_allow_does_not_stop_open_thread"] is True


async def test_put_allowlist_rejects_more_than_two_hundred_entries(api):
    client, _root, _discipline = api
    entries = [
        {"direction": "send", "effect": "allow", "target": {"endpoint_id": f"bep_{i}"}}
        for i in range(201)
    ]
    response = await client.put(
        "/v1/desktop/team-bridge/allowlist",
        json={"entries": entries, "expected_account_id": "acc_1"},
    )
    assert response.status_code == 422


async def test_put_allowlist_forwards_the_replace_set(api):
    client, _root, _discipline = api
    entry = {"direction": "receive", "effect": "deny", "target": {"account_id": "acc_9"}}
    FakeClient.behavior["put_allowlist"] = {"allowlist": {"tenant_id": "tnt_1", "entries": [entry]}}
    response = await client.put(
        "/v1/desktop/team-bridge/allowlist",
        json={"entries": [entry], "tenant_id": "tnt_1", "expected_account_id": "acc_1"},
    )
    assert response.status_code == 200
    assert FakeClient.behavior["last_entries"] == [entry]
    result = response.json()["result"]
    assert result["view"]["directions"]["receive"]["mode"] == "class_default"


async def test_every_route_requires_the_desktop_bearer(tmp_path: Path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(name)
    root = tmp_path / ".local-operator"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setattr(tb_client, "TeamBridgeClient", FakeClient)
    app = FastAPI()
    app.state.config_manager = ConfigManager(root)
    app.include_router(desktop_team_bridge.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer wrong-token"},
    ) as client:
        for method, path in [
            ("GET", "/v1/desktop/team-bridge/status"),
            ("GET", "/v1/desktop/team-bridge/settings"),
            ("PUT", "/v1/desktop/team-bridge/settings"),
            ("GET", "/v1/desktop/team-bridge/allowlist"),
            ("PUT", "/v1/desktop/team-bridge/allowlist"),
        ]:
            response = await client.request(method, path, json={"state": "paused"})
            assert response.status_code == 401, (method, path)
