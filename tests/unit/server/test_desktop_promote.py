"""``POST /v1/desktop/sessions/{id}/promote`` — the promotion contract (S4).

Why the route rather than the composite: the composite's tests prove the copy,
but only the wire can prove the three things a client depends on — that the
source is reached through the SESSION (the account-only profile routes cannot see
a conversation row), that ``request_id`` makes a retry replay instead of copying
twice, and that the failure statuses are the ladder's (404 unknown session or
name, 409 a refusal, 422 a malformed body).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.agents import AgentEditFields
from local_operator.config import ConfigManager
from local_operator.server.routes import capabilities, desktop_catalogues
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.teams import TeamEditFields

TOKEN = "desktop-promote-token"


@pytest_asyncio.fixture
async def promote_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)

    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    pool = DesktopSessions(tmp_path)
    app.state.desktop_sessions = pool
    app.include_router(desktop_catalogues.router)
    app.include_router(capabilities.router)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        session_id = await pool.create(str(workspace))
        yield client, session_id, pool
    await pool.close()


async def _seed_role(bridge: Any, name: str, prompt: str = "Check releases.\n") -> Any:
    registry = bridge.remote.agent_registry
    row = registry.create_agent(
        AgentEditFields(name=name, tags=["role"], description=f"{name} when needed"),
        scope="session",
    )
    registry.set_agent_system_prompt(row.id, prompt)
    return row


def _request(session_id: str, **fields: Any) -> dict[str, Any]:
    body = {"request_id": str(uuid.uuid4()), "kind": "agent", "name": "release-check"}
    body.update(fields)
    return body


@pytest.mark.asyncio
async def test_promote_answers_the_new_account_detail_row(promote_api) -> None:
    client, session_id, pool = promote_api
    async with pool.session(session_id) as bridge:
        source = await _seed_role(bridge, "release-check")
        response = await client.post(
            f"/v1/desktop/sessions/{session_id}/promote",
            json=_request(session_id),
        )
        payload = response.json()["result"]
        # The ACCOUNT row, read from the account store: the conversation still
        # holds its own copy, so a session-first read would have been the wrong row.
        account = bridge.remote.agent_registry.global_registry.get_agent_by_name("release-check")
        session_row = bridge.remote.agent_registry.get_agent_by_name("release-check")
        assert account is not None and session_row is not None
        assert account.id != source.id == session_row.id
        assert bridge.remote.agent_registry.get_agent_system_prompt(account.id) == (
            "Check releases.\n"
        )

    assert response.status_code == 200
    assert payload["name"] == "release-check"
    assert payload["scope"] == "global"
    assert payload["agent_id"] == account.id
    assert payload["promoted"]["kind"] == "agent"
    assert payload["promoted"]["created"] == [
        {"kind": "agent", "name": "release-check", "id": account.id}
    ]
    assert payload["promoted"]["expanded"] == []
    assert payload["promoted"]["replaced"] == []
    assert payload["promoted"]["authorized_by"] == "surface:desktop"
    assert payload["promoted"]["provenance_ids"] == [account.id]


@pytest.mark.asyncio
async def test_a_team_promotion_answers_the_team_row_and_names_its_members(promote_api) -> None:
    client, session_id, pool = promote_api
    async with pool.session(session_id) as bridge:
        await _seed_role(bridge, "helper", prompt="Help.\n")
        teams = bridge.remote.team_registry
        teams.create_team(
            TeamEditFields(
                name="pod",
                manager="helper",
                members=[{"role": "helper", "count": 2}],
                instructions="Collaborate.",
                project="Ship S4.",
            ),
            scope="session",
        )
        response = await client.post(
            f"/v1/desktop/sessions/{session_id}/promote",
            json=_request(session_id, kind="team", name="pod"),
        )
        account_team = teams.global_registry.get_team_by_name("pod")
        assert account_team is not None
        assert teams.global_registry.get_team_by_name("pod").project == "Ship S4."

    assert response.status_code == 200
    payload = response.json()["result"]
    assert payload["name"] == "pod"
    assert payload["scope"] == "global"
    assert payload["manager"] == "helper"
    assert payload["definition_revision"]
    assert payload["promoted"]["expanded"] == ["helper"]
    assert {row["name"] for row in payload["promoted"]["created"]} == {"helper", "pod"}


@pytest.mark.asyncio
async def test_an_unknown_name_is_404(promote_api) -> None:
    client, session_id, _pool = promote_api

    response = await client.post(
        f"/v1/desktop/sessions/{session_id}/promote", json=_request(session_id)
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_an_unknown_session_is_404(promote_api) -> None:
    client, _session_id, _pool = promote_api
    missing = str(uuid.uuid4())

    response = await client.post(f"/v1/desktop/sessions/{missing}/promote", json=_request(missing))

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_a_collision_is_409_and_confirm_replace_overwrites(promote_api) -> None:
    client, session_id, pool = promote_api
    async with pool.session(session_id) as bridge:
        await _seed_role(bridge, "release-check")
        first = await client.post(
            f"/v1/desktop/sessions/{session_id}/promote", json=_request(session_id)
        )
        second = await client.post(
            f"/v1/desktop/sessions/{session_id}/promote", json=_request(session_id)
        )
        replaced = await client.post(
            f"/v1/desktop/sessions/{session_id}/promote",
            json=_request(session_id, confirm_replace=True),
        )
        account = bridge.remote.agent_registry.global_registry.get_agent_by_name("release-check")

    assert first.status_code == 200
    assert second.status_code == 409
    assert "would overwrite it" in second.json()["detail"]
    assert replaced.status_code == 200
    assert replaced.json()["result"]["promoted"]["replaced"] == ["release-check"]
    assert replaced.json()["result"]["agent_id"] == account.id


@pytest.mark.asyncio
async def test_a_malformed_body_is_422(promote_api) -> None:
    client, session_id, _pool = promote_api

    for body in (
        {"kind": "agent", "name": "release-check"},  # no request_id
        {"request_id": str(uuid.uuid4()), "name": "release-check"},  # no kind
        {"request_id": str(uuid.uuid4()), "kind": "role", "name": "x"},  # bad kind
        {"request_id": str(uuid.uuid4()), "kind": "agent", "name": ""},  # empty name
    ):
        response = await client.post(f"/v1/desktop/sessions/{session_id}/promote", json=body)
        assert response.status_code == 422, body


@pytest.mark.asyncio
async def test_a_retried_request_id_replays_instead_of_copying_twice(promote_api) -> None:
    client, session_id, pool = promote_api
    body = _request(session_id)
    async with pool.session(session_id) as bridge:
        await _seed_role(bridge, "release-check")
        first = await client.post(f"/v1/desktop/sessions/{session_id}/promote", json=body)
        again = await client.post(f"/v1/desktop/sessions/{session_id}/promote", json=body)
        names = [row.name for row in bridge.remote.agent_registry.global_registry.list_agents()]

    assert first.status_code == again.status_code == 200
    # A replay answers the SAME result plus the receipt's own mark, so the copy
    # happened once and the second request was answered rather than re-run.
    replayed = dict(again.json()["result"])
    assert replayed.pop("replayed") is True
    assert replayed == first.json()["result"]
    assert names.count("release-check") == 1


@pytest.mark.asyncio
async def test_capabilities_advertises_profile_promote(promote_api) -> None:
    client, _session_id, _pool = promote_api

    response = await client.get("/v1/capabilities")

    features = response.json()["result"]["features"]
    assert features["profile_promote"] == 1
    # Its own key, not a bump: a renderer that predates it keeps every row it
    # already parses and simply offers no promote action.
    assert features["profile_scope"] == 1
