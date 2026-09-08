"""Canonical catalogue and authoring parity without a runtime or legacy chat row."""

import os
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.agent_profiles import resolve_profile_or_specialist
from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.resume import SessionRow, read_session_attachment
from local_operator.server.routes import (
    capabilities,
    desktop_profiles,
    desktop_sessions,
)
from local_operator.session.catalog import CatalogEntry, load_catalog
from local_operator.teams import TeamRegistry

pytestmark = pytest.mark.asyncio


def mutation(**fields):
    return {"request_id": str(uuid.uuid4()), **fields}


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "canonical-profile-test")
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.include_router(desktop_profiles.router)
    app.include_router(desktop_sessions.router)
    app.include_router(capabilities.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": "Bearer canonical-profile-test"},
    ) as client:
        yield client, tmp_path
    if hasattr(app.state, "desktop_sessions"):
        await app.state.desktop_sessions.close()


async def test_catalogue_is_authenticated_and_never_allocates(api):
    client, root = api
    denied = await client.get("/v1/desktop/profiles", headers={"Authorization": "Bearer wrong"})
    assert denied.status_code == 401
    response = await client.get("/v1/desktop/profiles")
    assert response.status_code == 200, response.text
    rows = response.json()["result"]["profiles"]
    assert any(row["name"] == "reviewer" and row["source"] == "builtin" for row in rows)
    assert all("instructions" not in row for row in rows)
    assert not list((root / "sessions").glob("*"))
    features = (await client.get("/v1/capabilities")).json()["result"]["features"]
    assert features["session_catalogue"] == 2


async def test_install_edit_preserves_policy_and_provenance(api):
    client, root = api
    installed = await client.post("/v1/desktop/profiles/install", json=mutation(name="reviewer"))
    assert installed.status_code == 200, installed.text
    before = installed.json()["result"]
    edited = await client.patch(
        "/v1/desktop/profiles/reviewer", json=mutation(instructions="Review carefully.")
    )
    assert edited.status_code == 200, edited.text
    current = edited.json()["result"]
    assert current["source"] == "installed"
    assert current["tools"] == before["tools"]
    assert current["delegate"] == before["delegate"]
    assert "instructions" in current["divergent_fields"]
    again = await client.post("/v1/desktop/profiles/install", json=mutation(name="reviewer"))
    assert again.json()["result"]["instructions"] == "Review carefully."
    kind, profile, _, _ = resolve_profile_or_specialist("reviewer", registry=AgentRegistry(root))
    assert kind == "role" and profile is not None and profile.instructions == "Review carefully."


async def test_specialist_metadata_and_independent_extension(api):
    client, root = api
    payload = mutation(
        name="researcher",
        kind="specialist",
        description="Research contracts",
        instructions="Use primary sources.",
    )
    created = await client.post("/v1/desktop/profiles", json=payload)
    assert created.status_code == 200, created.text
    row = created.json()["result"]
    assert row["kind"] == "specialist" and row["agent_id"]
    assert row["description"] == "Research contracts"
    retry = await client.post("/v1/desktop/profiles", json=payload)
    assert retry.json()["result"]["agent_id"] == row["agent_id"]
    assert len(AgentRegistry(root).list_agents()) == 1
    conflict = await client.patch("/v1/desktop/profiles/researcher", json=mutation(kind="role"))
    assert conflict.status_code == 409
    extended = await client.post(
        "/v1/desktop/profiles",
        json=mutation(
            name="research-copy",
            kind=row["kind"],
            description=row["description"],
            instructions=row["instructions"],
        ),
    )
    assert extended.status_code == 200, extended.text
    await client.patch(
        "/v1/desktop/profiles/researcher", json=mutation(instructions="Changed source.")
    )
    copy = (await client.get("/v1/desktop/profiles/research-copy")).json()["result"]
    assert copy["instructions"] == "Use primary sources."


async def test_team_patch_preserves_unloaded_briefs(api):
    client, root = api
    created = await client.post(
        "/v1/desktop/teams",
        json=mutation(
            name="audit",
            manager="manager",
            members=[{"role": "reviewer", "count": 2, "kind": "agent"}],
            instructions="Coordinate reviews.",
            project="Canonical chat.",
        ),
    )
    assert created.status_code == 200, created.text
    listed = (await client.get("/v1/desktop/teams")).json()["result"]["teams"]
    assert "instructions" not in listed[0]
    edited = await client.patch(
        "/v1/desktop/teams/audit", json=mutation(description="Audit contracts")
    )
    assert edited.status_code == 200, edited.text
    team = TeamRegistry(root).get_team_by_name("audit")
    assert team is not None
    assert team.instructions == "Coordinate reviews." and team.project == "Canonical chat."
    await client.patch("/v1/desktop/teams/audit", json=mutation(instructions=""))
    team = TeamRegistry(root).get_team_by_name("audit")
    assert team is not None
    assert team.instructions == "" and team.project == "Canonical chat."
    invalid = await client.patch(
        "/v1/desktop/teams/audit", json=mutation(members=[{"role": "reviewer", "count": 17}])
    )
    assert invalid.status_code == 422


async def test_many_chats_same_profile_and_replayed_creation(api):
    client, root = api
    body = mutation(cwd=str(root), target={"kind": "agent", "name": "reviewer"})
    response = await client.post("/v1/desktop/sessions", json=body)
    assert response.status_code == 200, response.text
    first = response.json()["result"]
    repeat = (await client.post("/v1/desktop/sessions", json=body)).json()["result"]
    assert repeat["session_id"] == first["session_id"] and repeat["replayed"]
    second = (
        await client.post(
            "/v1/desktop/sessions", json=mutation(cwd=str(root), target=body["target"])
        )
    ).json()["result"]
    assert second["session_id"] != first["session_id"]
    for item in (first, second):
        attachment = read_session_attachment(root / "sessions" / item["session_id"])
        assert attachment is not None
        assert attachment.agent == "reviewer" and attachment.team == ""
        assert item["binding"] == {"agent": "reviewer", "team": None}
    rows = (await client.get("/v1/desktop/sessions")).json()["result"]["sessions"]
    catalog = load_catalog(root)
    assert [row["id"] for row in rows] == [entry.id for entry in catalog]
    assert all(row["status"]["code"] == "recent" and not row["active"] for row in rows)
    assert {row["id"] for row in rows} == {first["session_id"], second["session_id"]}


async def test_unresolved_owner_rejects_before_model_work_and_allows_same_id_retry():
    from typing import Any, cast

    from tests.unit.session.runtime.test_owned import make_handle

    handle, session = make_handle()
    cast(Any, session)._unresolved_agent = "deleted-profile"
    with pytest.raises(ValueError, match="could not be restored"):
        await handle.prompt("Verify the contract", command_id="same-admission")
    assert session.prompt_calls == []
    assert not handle.has_admitted_command("same-admission")
    cast(Any, session)._unresolved_agent = ""
    session.prompt_release.set()
    await handle.prompt("Verify the contract", command_id="same-admission", wait_complete=True)
    assert session.prompt_calls == ["Verify the contract"]


async def test_invalid_target_or_cwd_never_publishes_session(api):
    client, root = api
    for fields in (
        {"cwd": str(root), "target": {"kind": "agent", "name": "missing"}},
        {"cwd": str(root / "absent"), "target": {"kind": "agent", "name": "reviewer"}},
    ):
        result = await client.post("/v1/desktop/sessions", json=mutation(**fields))
        assert result.status_code in (404, 409), result.text
    assert not list((root / "sessions").glob("*"))


async def test_failed_binding_write_does_not_publish_false_success(api, monkeypatch):
    client, root = api
    monkeypatch.setattr(
        "local_operator.server.utils.desktop_sessions.write_session_attachment",
        lambda *a, **kw: None,
    )
    result = await client.post(
        "/v1/desktop/sessions",
        json=mutation(cwd=str(root), target={"kind": "agent", "name": "reviewer"}),
    )
    assert result.status_code == 409
    assert not list((root / "sessions").glob("*/desktop.json"))
    assert load_catalog(root) == []


@pytest.mark.parametrize(
    "live,pending,unseen,kind,expected",
    [
        ("busy", None, True, "error", "busy"),
        ("busy", "approval", True, "error", "approval"),
        ("wedged", None, True, "completed", "wedged"),
        ("", None, False, "error", "error"),
        ("", None, False, "interrupted", "interrupted"),
        ("", None, False, "completed", "complete"),
    ],
)
async def test_shared_status_precedence(live, pending, unseen, kind, expected):
    entry = CatalogEntry(
        SessionRow("123456789abc", 1, "test", live_state=live, pending=pending),
        unseen,
        kind,
        "token",
        "anchor",
    )
    assert entry.status_code == expected
    assert entry.active == (entry.rank[0] <= 2)
    assert CatalogEntry(SessionRow("123456789abc", 1, "unknown")).status_code == "recent"
