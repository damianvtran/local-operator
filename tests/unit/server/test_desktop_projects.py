"""``/v1/desktop/projects*`` on the wire: CRUD, links, milestones, refusals.

An isolated ``HOME``/config root and a sub-app carrying only the routers under
test, mirroring ``test_desktop_profiles``: these assertions are about what a
renderer is told (status codes, machine codes, the composed view's shape), not
about the store — that has its own suite.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.server.routes import capabilities, desktop_projects

pytestmark = pytest.mark.asyncio

TOKEN = "projects-desktop-token"
SESSION_A = "4e92693767fa"


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("CMUX_") or name.startswith("LOP_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    app.include_router(desktop_projects.router)
    app.include_router(capabilities.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, tmp_path


async def test_empty_listing_and_the_capability_gate(api) -> None:
    client, _root = api
    listed = await client.get("/v1/desktop/projects")
    assert listed.status_code == 200
    assert listed.json()["result"] == {"projects": []}

    caps = await client.get("/v1/capabilities")
    assert caps.status_code == 200
    assert caps.json()["result"]["features"]["projects"] == 1


async def test_create_read_patch_delete_round_trip(api) -> None:
    client, root = api
    created = await client.post(
        "/v1/desktop/projects",
        json={"name": "payments-migration", "description": "Payments", "tags": ["q4"]},
    )
    assert created.status_code == 200
    summary = created.json()["result"]
    assert summary["status"] == "active" and summary["progress_stale"] is True
    assert summary["sessions"] == 0
    project_id = summary["id"]

    # The store is the row on disk, not a cache in the route.
    on_disk = json.loads((root / "projects" / f"{project_id}.json").read_text())
    assert on_disk["name"] == "payments-migration" and on_disk["schema"] == 1

    by_name = await client.get("/v1/desktop/projects/payments-migration")
    assert by_name.status_code == 200
    assert by_name.json()["result"]["project"]["id"] == project_id

    patched = await client.patch(
        f"/v1/desktop/projects/{project_id}",
        json={"status": "done", "progress": "cutover done"},
    )
    assert patched.status_code == 200
    result = patched.json()["result"]
    assert result["status"] == "done"
    # status -> done with completed_at omitted stamps today (the one convenience).
    assert result["completed_at"] is not None
    assert result["progress_stale"] is False

    detail = await client.get(f"/v1/desktop/projects/{project_id}")
    payload = detail.json()["result"]
    assert payload["project"]["progress_stale"] is False
    assert payload["project"]["progress_updated_at"] is not None
    assert payload["links"] == []

    refused = await client.request(
        "DELETE", f"/v1/desktop/projects/{project_id}", json={"confirm": "not-the-name"}
    )
    assert refused.status_code == 422
    assert refused.json()["detail"]["code"] == "project_confirm_mismatch"

    deleted = await client.request(
        "DELETE", f"/v1/desktop/projects/{project_id}", json={"confirm": "Payments-Migration"}
    )
    assert deleted.status_code == 200 and deleted.json()["result"] == {"deleted": True}
    missing = await client.get(f"/v1/desktop/projects/{project_id}")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "project_not_found"


async def test_a_taken_name_is_a_409_and_invalid_values_are_422(api) -> None:
    client, _root = api
    await client.post("/v1/desktop/projects", json={"name": "alpha"})
    duplicate = await client.post("/v1/desktop/projects", json={"name": "ALPHA"})
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "project_name_exists"

    created = await client.post("/v1/desktop/projects", json={"name": "beta"})
    beta = created.json()["result"]["id"]
    for body in (
        {"tags": ["Q4!"]},
        {"estimate": 0},
        {"start_date": "2026-10-10", "target_date": "2026-10-01"},
        {"status": "sideways"},
        {"milestones": [{"name": "x"}, {"name": "X"}]},
    ):
        bad = await client.patch(f"/v1/desktop/projects/{beta}", json=body)
        assert bad.status_code == 422, body
    shape = await client.post("/v1/desktop/projects", json={"name": "x" * 65})
    assert shape.status_code == 422


async def test_milestone_routes_report_derived_status_and_refuse_unknowns(api) -> None:
    client, _root = api
    created = await client.post("/v1/desktop/projects", json={"name": "alpha"})
    project_id = created.json()["result"]["id"]

    added = await client.post(
        f"/v1/desktop/projects/{project_id}/milestones",
        json={"name": "beta cut", "target_date": "2026-10-01"},
    )
    assert added.status_code == 200
    (milestone,) = added.json()["result"]["milestones"]
    assert milestone == {
        "name": "beta cut",
        "target_date": "2026-10-01",
        "completed_at": None,
        "status": "upcoming",
    }

    overdue = await client.post(
        f"/v1/desktop/projects/{project_id}/milestones",
        json={"name": "audit", "target_date": "2020-01-01"},
    )
    overdue_row = next(m for m in overdue.json()["result"]["milestones"] if m["name"] == "audit")
    assert overdue_row["status"] == "overdue"

    completed = await client.post(
        f"/v1/desktop/projects/{project_id}/milestones",
        json={"name": "Beta Cut", "completed": True},
    )
    # Add-or-update is keyed by name, CASE-INSENSITIVELY — and the stored
    # spelling is the one the row was created with: renaming is remove + add.
    done_row = next(
        m for m in completed.json()["result"]["milestones"] if m["name"].casefold() == "beta cut"
    )
    assert done_row["status"] == "completed" and done_row["completed_at"] is not None

    removed = await client.delete(f"/v1/desktop/projects/{project_id}/milestones/beta%20cut")
    assert removed.status_code == 200
    assert [m["name"] for m in removed.json()["result"]["milestones"]] == ["audit"]

    missing = await client.delete(f"/v1/desktop/projects/{project_id}/milestones/ghost")
    assert missing.status_code == 422
    assert "no milestone named 'ghost'" in missing.json()["detail"]["message"]


async def test_links_round_trip_and_the_view_counts_live_sessions(api) -> None:
    client, root = api
    created = await client.post("/v1/desktop/projects", json={"name": "alpha"})
    project_id = created.json()["result"]["id"]

    # A live-looking record for the session we are about to link: pid = ours.
    run_dir = root / "run" / "mobile"
    run_dir.mkdir(parents=True)
    (run_dir / f"{os.getpid()}.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "kind": "tui",
                "session_id": SESSION_A,
                "conversation_name": "x",
                "cwd": str(root),
                "model_label": "m",
                "control_port": 1,
                "control_key": "0" * 16,
                "heartbeat_at": time.time(),
                "busy": False,
            }
        )
    )

    linked = await client.post(
        f"/v1/desktop/projects/{project_id}/links", json={"session_id": SESSION_A}
    )
    assert linked.status_code == 200
    assert linked.json()["result"]["sessions"] == 1
    assert linked.json()["result"]["live_sessions"] == 1

    detail = await client.get(f"/v1/desktop/projects/{project_id}")
    (link,) = detail.json()["result"]["links"]
    assert link["session_id"] == SESSION_A
    assert link["runtime"]["state"] == "live"
    assert link["exists"] is False and link["todos"] is None and link["subagents"] is None

    bad = await client.post(f"/v1/desktop/projects/{project_id}/links", json={"session_id": "nope"})
    assert bad.status_code == 422

    unlinked = await client.delete(f"/v1/desktop/projects/{project_id}/links/{SESSION_A}")
    assert unlinked.status_code == 200
    assert unlinked.json()["result"]["sessions"] == 0


async def test_the_listing_is_sorted_by_status_then_freshness(api) -> None:
    client, _root = api
    for name, status in (
        ("archived-one", "archived"),
        ("active-old", "active"),
        ("paused", "paused"),
    ):
        await client.post("/v1/desktop/projects", json={"name": name, "status": status})
    # A later edit makes one active row fresher than the other.
    await client.patch("/v1/desktop/projects/active-old", json={"description": "touched"})
    listed = await client.get("/v1/desktop/projects")
    rows = [row["name"] for row in listed.json()["result"]["projects"]]
    assert rows == ["active-old", "paused", "archived-one"]


async def test_a_row_from_a_newer_build_is_readable_but_refuses_mutation(api) -> None:
    client, root = api
    created = await client.post("/v1/desktop/projects", json={"name": "alpha"})
    project_id = created.json()["result"]["id"]
    path = root / "projects" / f"{project_id}.json"
    payload = json.loads(path.read_text())
    payload["schema"] = 2
    payload["future_field"] = True
    path.write_text(json.dumps(payload))

    read = await client.get(f"/v1/desktop/projects/{project_id}")
    assert read.status_code == 200  # reads stay lenient

    write = await client.patch(f"/v1/desktop/projects/{project_id}", json={"description": "x"})
    assert write.status_code == 409
    assert write.json()["detail"]["code"] == "project_schema_newer"
    assert "update this build" in write.json()["detail"]["message"]


async def test_lock_contention_answers_503(api, monkeypatch) -> None:
    from local_operator.projects import ProjectRegistry, ProjectRegistryLockTimeout

    client, _root = api
    created = await client.post("/v1/desktop/projects", json={"name": "alpha"})
    project_id = created.json()["result"]["id"]

    def refuse(self, *args: Any, **kwargs: Any):
        raise ProjectRegistryLockTimeout("Timed out waiting for the projects registry lock")

    monkeypatch.setattr(ProjectRegistry, "update_project", refuse)
    busy = await client.patch(f"/v1/desktop/projects/{project_id}", json={"description": "x"})
    assert busy.status_code == 503
    assert busy.json()["detail"]["code"] == "project_store_busy"


async def test_requests_without_the_desktop_token_are_refused(api) -> None:
    client, _root = api
    anonymous = await client.get(
        "/v1/desktop/projects",
        headers={"Authorization": f"Bearer {uuid4().hex}"},
    )
    assert anonymous.status_code in (401, 403)
