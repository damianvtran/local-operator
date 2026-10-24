"""UI identity mutations traverse the real route, store, resolver and bridge client.

Only the external authority/credential store is scripted. A warm status cache is
essential: a cold-only test misses the user-visible five-minute reconciliation gap.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from local_operator import settings_io
from local_operator.aida import state as aida_state
from local_operator.config import ConfigManager
from local_operator.server.routes import settings
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import Boundary
from local_operator.team_bridge.store import Store
from tests.unit.team_bridge.test_client import FakeRow, FakeStore, _ok
from tests.unit.team_bridge.test_cos_name import _echoing
from tests.unit.team_bridge.test_discipline import (
    _committed_owner,
    _Counter,
    _discipline,
)

pytestmark = pytest.mark.asyncio


def _rig(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: dict[str, Any] | None = None,
    store: FakeStore | None = None,
):
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "fixture-desktop")
    manager = ConfigManager(tmp_path)
    settings_io.write_setting(manager, settings_io.BY_KEY["team_bridge.session"], "sess_old")
    aida_state.write_state(tmp_path, {"session_id": "sess_aida"})
    _, owner = _committed_owner(tmp_path, session="sess_old")
    counter = _Counter(lambda request: _ok(payload if payload is not None else _echoing(None)))
    discipline = _discipline(counter, tmp_path, store=store)
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)
    monkeypatch.setattr(tb_client, "get_status_discipline", lambda root: discipline)
    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_config_manager] = lambda: manager
    return (
        manager,
        owner,
        counter,
        discipline,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://ui.test",
            headers={"Authorization": "Bearer fixture-desktop"},
        ),
    )


async def test_warm_cache_ui_designation_reconciles_before_reply(tmp_path, monkeypatch):
    manager, owner, counter, discipline, ui = _rig(tmp_path, monkeypatch)
    await discipline.check(Boundary.APP_START)
    assert [r.method for r in counter.requests] == ["GET"]
    async with ui:
        response = await ui.patch("/v1/settings/team_bridge.session", json={"value": "sess_aida"})
    assert response.status_code == 200
    assert (
        settings_io.read_setting(manager, settings_io.BY_KEY["team_bridge.session"]) == "sess_aida"
    )
    assert Store(tmp_path).designated_session_for(owner) == "sess_aida"
    assert [r.method for r in counter.requests] == ["GET", "GET", "PUT"]
    assert json.loads(counter.requests[-1].content) == {
        "state": "active",
        "tenant_id": "tnt_1",
        "agent_name": "Aida",
    }
    # The first fresh answer predates the PUT. Do not manufacture a server echo
    # or spend a second GET to make a just-published name appear confirmed.
    check = await discipline.check(Boundary.SURFACE_OPEN)
    assert check.from_cache
    assert check.payload is not None
    assert check.payload["tenants"][0]["endpoint"]["agent_name"] is None
    assert len(counter.requests) == 3


async def test_rename_reset_clear_and_noop_use_canonical_identity(tmp_path, monkeypatch):
    manager, owner, counter, discipline, ui = _rig(tmp_path, monkeypatch)
    await discipline.check(Boundary.APP_START)
    async with ui:
        response = await ui.patch("/v1/settings/team_bridge.session", json={"value": "sess_aida"})
        assert response.status_code == 200
        response = await ui.patch("/v1/settings/aida.name", json={"value": "  Friday   Jones "})
        assert response.status_code == 200
        assert json.loads(counter.requests[-1].content)["agent_name"] == "Friday Jones"
        count = len(counter.requests)
        # Canonically unchanged writes and resetting an unrelated setting are
        # not an escape hatch from the normal status floor.
        response = await ui.patch("/v1/settings/aida.name", json={"value": "Friday Jones"})
        assert response.status_code == 200
        assert (await ui.post("/v1/settings/team_bridge.enabled/reset")).status_code == 200
        assert len(counter.requests) == count
        assert (await ui.post("/v1/settings/aida.name/reset")).status_code == 200
        assert json.loads(counter.requests[-1].content)["agent_name"] == "Aida"
        count = len(counter.requests)
        assert (await ui.post("/v1/settings/team_bridge.session/reset")).status_code == 200
        assert Store(tmp_path).designated_session_for(owner) is None
        assert [r.method for r in counter.requests[count:]] == ["GET"]
        count = len(counter.requests)
        response = await ui.patch("/v1/settings/team_bridge.session", json={"value": ""})
        assert response.status_code == 200
        assert len(counter.requests) == count


@pytest.mark.parametrize(
    "guard",
    [
        "paused",
        "disabled",
        "ineligible",
        "unsupported",
        "old-hub",
        "unowned",
        "wrong-account",
        "wrong-endpoint",
        "non-aida",
    ],
)
async def test_mutation_preserves_publication_guards(tmp_path, monkeypatch, guard):
    payload = _echoing(None)
    tenant = payload["tenants"][0]
    if guard in {"paused", "disabled"}:
        tenant["endpoint"]["state"] = guard
    elif guard == "ineligible":
        tenant["eligible"] = False
    elif guard == "unsupported":
        payload["api_version"] = "2"
    elif guard == "old-hub":
        del tenant["endpoint"]["agent_name"]
    elif guard == "wrong-endpoint":
        tenant["endpoint"]["id"] = "bep_other"
    manager, owner, counter, discipline, ui = _rig(tmp_path, monkeypatch, payload=payload)
    if guard == "unowned":
        monkeypatch.setattr(Store, "active_owner", lambda self: None)
    elif guard == "wrong-account":
        from dataclasses import replace

        monkeypatch.setattr(
            Store, "active_owner", lambda self: replace(owner, account_id="acc_other")
        )
    await discipline.check(Boundary.APP_START)
    async with ui:
        value = "sess_other" if guard == "non-aida" else "sess_aida"
        response = await ui.patch("/v1/settings/team_bridge.session", json={"value": value})
    assert response.status_code == 200
    assert all(request.method == "GET" for request in counter.requests)
    assert Store(tmp_path).committed_owner_count() == 1  # never enroll/re-consent


@pytest.mark.parametrize("failure", ["invalid", "write"])
async def test_failed_settings_write_never_reconciles(tmp_path, monkeypatch, failure):
    manager, owner, counter, discipline, ui = _rig(tmp_path, monkeypatch)
    await discipline.check(Boundary.APP_START)
    if failure == "write":

        def refuse(*args, **kwargs):
            raise OSError("private-sentinel-never-echo")

        monkeypatch.setattr(settings_io, "write_setting", refuse)
    async with ui:
        response = await ui.patch(
            (
                "/v1/settings/aida.name"
                if failure == "invalid"
                else "/v1/settings/team_bridge.session"
            ),
            json={"value": "" if failure == "invalid" else "sess_aida"},
        )
    assert response.status_code == (422 if failure == "invalid" else 503)
    assert len(counter.requests) == 1
    assert Store(tmp_path).designated_session_for(owner) == "sess_old"
    assert "private-sentinel" not in response.text


async def test_queued_edits_coalesce_latest_without_holding_settings_lock(tmp_path, monkeypatch):
    manager, owner, counter, discipline, ui = _rig(tmp_path, monkeypatch)
    await discipline.check(Boundary.APP_START)
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold_first(request):
        if not entered.is_set():
            entered.set()
            await release.wait()
        return counter(request)

    discipline._client._transport = httpx.MockTransport(hold_first)
    discipline.invalidate()
    older_read = asyncio.create_task(discipline.check(Boundary.SURFACE_OPEN))
    await entered.wait()
    written = {value: asyncio.Event() for value in ("sess_aida", "Friday", "Saturday")}
    write = settings_io.write_setting

    def recording_write(manager, setting, value, **kwargs):
        write(manager, setting, value, **kwargs)
        if value in written:
            written[value].set()

    monkeypatch.setattr(settings_io, "write_setting", recording_write)
    async with ui:
        pending = []
        for key, value in (
            ("team_bridge.session", "sess_aida"),
            ("aida.name", "Friday"),
            ("aida.name", "Saturday"),
        ):
            pending.append(
                asyncio.create_task(ui.patch(f"/v1/settings/{key}", json={"value": value}))
            )
            await written[value].wait()
        # No network under the global settings lock: unrelated saves finish
        # while the bridge request is deliberately still blocked.
        assert (await ui.post("/v1/settings/team_bridge.enabled/reset")).status_code == 200
        release.set()
        responses = await asyncio.gather(*pending)
        await older_read
    assert all(response.status_code == 200 for response in responses)
    assert Store(tmp_path).designated_session_for(owner) == "sess_aida"
    assert [r.method for r in counter.requests] == ["GET", "GET", "PUT", "GET"]
    published = json.loads(next(r.content for r in counter.requests if r.method == "PUT"))
    assert published["agent_name"] == "Saturday"


@pytest.mark.parametrize("failure", ["timeout", "refusal", "unavailable"])
async def test_bridge_failure_does_not_fail_persisted_edit(tmp_path, monkeypatch, caplog, failure):
    manager, owner, counter, discipline, ui = _rig(tmp_path, monkeypatch)
    await discipline.check(Boundary.APP_START)
    cancelled = asyncio.Event()

    async def failing(request):
        if failure == "timeout":
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        if failure == "refusal" and request.method == "GET":
            return _ok(_echoing(None))
        return httpx.Response(
            503, json={"error": "private-sentinel", "code": "service_unavailable"}
        )

    discipline._client._transport = httpx.MockTransport(failing)
    monkeypatch.setattr(settings, "_IDENTITY_RECONCILE_TIMEOUT_S", 0.01)
    async with ui:
        response = await ui.patch("/v1/settings/team_bridge.session", json={"value": "sess_aida"})
    assert response.status_code == 200
    assert (
        settings_io.read_setting(manager, settings_io.BY_KEY["team_bridge.session"]) == "sess_aida"
    )
    assert "deferred" in caplog.text
    assert "private-sentinel" not in caplog.text + response.text
    if failure == "timeout":
        assert cancelled.is_set()


@pytest.mark.parametrize("change", ["account-switch", "owner-revoked"])
async def test_authority_changed_during_status_read_never_publishes(tmp_path, monkeypatch, change):
    # Keep the typed fake at its injection seam instead of reaching through the
    # client's production AuthStore annotation to mutate test-only attributes.
    store = FakeStore()
    manager, owner, counter, discipline, ui = _rig(tmp_path, monkeypatch, store=store)
    await discipline.check(Boundary.APP_START)

    def changed_while_awaiting(request):
        assert request.method == "GET"  # any subsequent PUT fails this control
        if change == "account-switch":
            store.rows = [FakeRow(2, data={"access": "fixture-other", "account_id": "acc_other"})]
        else:
            monkeypatch.setattr(Store, "active_owner", lambda self: None)
        return counter(request)

    discipline._client._transport = httpx.MockTransport(changed_while_awaiting)
    async with ui:
        response = await ui.patch("/v1/settings/team_bridge.session", json={"value": "sess_aida"})
    assert response.status_code == 200
    assert [r.method for r in counter.requests] == ["GET", "GET"]


async def test_unauthorized_mutation_never_writes_or_fetches(tmp_path, monkeypatch):
    manager, owner, counter, discipline, ui = _rig(tmp_path, monkeypatch)
    await discipline.check(Boundary.APP_START)
    async with ui:
        response = await ui.patch(
            "/v1/settings/team_bridge.session",
            json={"value": "sess_aida"},
            headers={"Authorization": "invalid"},
        )
    assert response.status_code == 401
    assert len(counter.requests) == 1
    assert Store(tmp_path).designated_session_for(owner) == "sess_old"
