"""``/v1/desktop/projects*`` on the wire: CRUD, links, milestones, refusals.

An isolated ``HOME``/config root and a sub-app carrying only the routers under
test, mirroring ``test_desktop_profiles``: these assertions are about what a
renderer is told (status codes, machine codes, the composed view's shape), not
about the store — that has its own suite.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.projects import PROJECT_SCHEMA
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
    # 2 = the CRUD surface plus the derived search/timeline reads (additive).
    assert caps.json()["result"]["features"]["projects"] == 2


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
    assert on_disk["name"] == "payments-migration" and on_disk["schema"] == PROJECT_SCHEMA

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
    payload["schema"] = PROJECT_SCHEMA + 1
    payload["future_field"] = True
    # A linked session, so the unlink route reaches the mutation (and therefore
    # the guard) rather than stopping at its membership check.
    payload["sessions"] = [SESSION_A]
    path.write_text(json.dumps(payload))

    read = await client.get(f"/v1/desktop/projects/{project_id}")
    assert read.status_code == 200  # reads stay lenient

    write = await client.patch(f"/v1/desktop/projects/{project_id}", json={"description": "x"})
    assert write.status_code == 409
    assert write.json()["detail"]["code"] == "project_schema_newer"
    assert "update this build" in write.json()["detail"]["message"]

    # EVERY mutating route answers the same way. These three used to fall
    # through to `errors()`'s generic RuntimeError arm — the guard subclasses
    # RuntimeError — and told the client `503 runtime_unreachable`, a reconnect
    # remedy that can never work (QA round 1, Q1; agent review M1).
    for response in (
        await client.post(
            f"/v1/desktop/projects/{project_id}/links", json={"session_id": "abcdef012345"}
        ),
        await client.delete(f"/v1/desktop/projects/{project_id}/links/{SESSION_A}"),
        await client.request(
            "DELETE", f"/v1/desktop/projects/{project_id}", json={"confirm": "alpha"}
        ),
    ):
        assert response.status_code == 409, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "project_schema_newer", detail
        assert "update this build" in detail["message"]

    # ... and none of them wrote: the row still exists, still at the newer schema.
    still_there = json.loads(path.read_text())
    assert still_there["schema"] == PROJECT_SCHEMA + 1 and still_there["sessions"] == [SESSION_A]


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


async def test_owner_team_and_title_flow_through_listing_and_detail(api) -> None:
    client, _root = api
    created = await client.post("/v1/desktop/projects", json={"name": "gamma"})
    project_id = created.json()["result"]["id"]

    patched = await client.patch(
        f"/v1/desktop/projects/{project_id}",
        json={"owner": "Damian", "team": "Platform", "title": "Gamma Stream"},
    )
    assert patched.status_code == 200
    result = patched.json()["result"]
    assert (result["owner"], result["team"], result["title"]) == (
        "Damian",
        "Platform",
        "Gamma Stream",
    )

    listed = (await client.get("/v1/desktop/projects")).json()["result"]["projects"][0]
    assert listed["owner"] == "Damian" and listed["title"] == "Gamma Stream"

    detail = (await client.get(f"/v1/desktop/projects/{project_id}")).json()["result"]
    assert detail["project"]["owner"] == "Damian"
    assert detail["project"]["title"] == "Gamma Stream"
    assert detail["project"]["updates"] == []

    cleared = await client.patch(f"/v1/desktop/projects/{project_id}", json={"owner": ""})
    assert cleared.json()["result"]["owner"] is None


async def test_a_progress_write_appends_history_and_settled_rows_never_read_stale(api) -> None:
    client, root = api
    created = await client.post("/v1/desktop/projects", json={"name": "delta"})
    project_id = created.json()["result"]["id"]
    # Never reported: stale by construction while the first honest line is owed.
    assert created.json()["result"]["progress_stale"] is True

    patched = await client.patch(
        f"/v1/desktop/projects/{project_id}", json={"progress": "first", "status": "done"}
    )
    assert patched.json()["result"]["progress_stale"] is False
    row = json.loads((root / "projects" / f"{project_id}.json").read_text())
    assert [update["text"] for update in row["updates"]] == ["first"]
    assert row["updates"][0]["by"] == "operator"
    assert row["updates"][0]["at"].endswith("Z")

    # A settled row never reads stale even once its one line ages out.
    aged = json.loads((root / "projects" / f"{project_id}.json").read_text())
    aged["progress_updated_at"] = 0.0
    (root / "projects" / f"{project_id}.json").write_text(json.dumps(aged))
    summary = (await client.get("/v1/desktop/projects")).json()["result"]["projects"][0]
    assert summary["progress_stale"] is False
    detail = (await client.get(f"/v1/desktop/projects/{project_id}")).json()["result"]
    assert detail["project"]["progress_stale"] is False
    # The history survives the aging — it is the log, not the freshness pair.
    assert [update["text"] for update in detail["project"]["updates"]] == ["first"]


async def test_a_row_written_before_the_new_fields_reads_with_nulls(api) -> None:
    client, root = api
    created = await client.post("/v1/desktop/projects", json={"name": "legacy"})
    project_id = created.json()["result"]["id"]
    path = root / "projects" / f"{project_id}.json"
    payload = json.loads(path.read_text())
    for key in ("owner", "team", "title", "updates"):
        payload.pop(key, None)
    path.write_text(json.dumps(payload))

    detail = await client.get(f"/v1/desktop/projects/{project_id}")
    assert detail.status_code == 200
    project = detail.json()["result"]["project"]
    assert project["owner"] is None and project["team"] is None and project["title"] is None
    assert project["updates"] == []
    listed = (await client.get("/v1/desktop/projects")).json()["result"]["projects"][0]
    assert listed["owner"] is None and listed["title"] is None


def test_status_rank_covers_the_lifecycle_in_order() -> None:
    from local_operator.server.models.desktop_projects import STATUS_RANK

    ordered = [key for key, _ in sorted(STATUS_RANK.items(), key=lambda kv: kv[1])]
    assert ordered == [
        "planning",
        "active",
        "qa",
        "validation",
        "paused",
        "done",
        "archived",
    ]


async def test_the_wire_counts_work_only_and_carries_the_filing_and_refresh(api) -> None:
    """Schema 2 on the wire: the summary's ``sessions``/``live_sessions`` are the
    WORK set (a filing gets its own count), the view carries the ids and each
    link's ``role``, and the refreshed pair crosses while ``progress_stale``
    keeps the content clock's truth (a refresh never clears the badge)."""
    from local_operator.projects import ProjectEdit, ProjectRegistry

    client, root = api
    session_b = "abcdef012345"
    created = await client.post("/v1/desktop/projects", json={"name": "alpha"})
    project_id = created.json()["result"]["id"]

    store = ProjectRegistry(root)
    store.link_session(project_id, SESSION_A)
    store.link_session(project_id, session_b, role="coordination")
    store.update_project(project_id, ProjectEdit(progress="still true"), reporter="operator")
    path = root / "projects" / f"{project_id}.json"
    payload = json.loads(path.read_text())
    payload["progress_updated_at"] = time.time() - 5 * 3600
    path.write_text(json.dumps(payload))
    store = ProjectRegistry(root)  # a fresh reader sees the backdate
    store.refresh_project(project_id, reporter=session_b)

    listed = (await client.get("/v1/desktop/projects")).json()["result"]["projects"]
    summary = next(row for row in listed if row["id"] == project_id)
    assert summary["sessions"] == 1 and summary["live_sessions"] == 0
    assert summary["coordination_sessions"] == 1
    assert summary["progress_stale"] is True
    assert summary["progress_refreshed_at"] is not None
    assert summary["progress_refreshed_by"] == session_b

    detail = (await client.get(f"/v1/desktop/projects/{project_id}")).json()["result"]
    assert detail["project"]["sessions"] == [SESSION_A]
    assert detail["project"]["coordination_sessions"] == [session_b]
    links = {row["session_id"]: row for row in detail["links"]}
    assert links[SESSION_A]["role"] == "work"
    filed = links[session_b]
    assert filed["role"] == "coordination"
    assert filed["runtime"] is None and filed["subagents"] is None and filed["todos"] is None


async def test_search_ranks_echoes_and_stays_ahead_of_a_project_named_search(api) -> None:
    client, _root = api
    await client.post(
        "/v1/desktop/projects", json={"name": "alpha", "description": "payment pipeline"}
    )
    gamma = (await client.post("/v1/desktop/projects", json={"name": "gamma"})).json()["result"][
        "id"
    ]
    await client.patch(
        f"/v1/desktop/projects/{gamma}", json={"title": "Improve ADM Classifier Throughput"}
    )
    # A project NAMED "search" must not swallow the static sibling: the sibling
    # is declared before ``/{key}``, which is the load-bearing order.
    await client.post("/v1/desktop/projects", json={"name": "search"})

    answer = await client.get("/v1/desktop/projects/search", params={"q": "Throughput classifer"})
    assert answer.status_code == 200
    result = answer.json()["result"]
    assert result["query"] == "Throughput classifer"  # the echo the client checks
    assert result["count"] == len(result["projects"]) == 1
    (hit,) = result["projects"]
    assert hit["id"] == gamma and hit["fields"] == ["name"]
    assert hit["name"] == "Improve ADM Classifier Throughput"  # the display name
    assert hit["score"] == 16.0  # 2 tokens x 8; the phrase is not contiguous


async def test_search_bounds_echo_and_the_empty_query_pass_through(api) -> None:
    client, _root = api
    await client.post("/v1/desktop/projects", json={"name": "alpha"})
    await client.post("/v1/desktop/projects", json={"name": "beta"})

    too_long = await client.get("/v1/desktop/projects/search", params={"q": "x" * 257})
    assert too_long.status_code == 422
    low = await client.get("/v1/desktop/projects/search", params={"limit": 0})
    assert low.status_code == 422
    high = await client.get("/v1/desktop/projects/search", params={"limit": 201})
    assert high.status_code == 422
    edge = await client.get("/v1/desktop/projects/search", params={"q": "x", "limit": 200})
    assert edge.status_code == 200

    empty = (await client.get("/v1/desktop/projects/search", params={"q": "  "})).json()["result"]
    listing = (await client.get("/v1/desktop/projects")).json()["result"]["projects"]
    # The empty box is not a search: every row, in the LISTING's own order.
    assert [hit["id"] for hit in empty["projects"]] == [row["id"] for row in listing]
    assert empty["query"] == "  "
    assert all(hit["score"] == 0.0 and hit["fields"] == [] for hit in empty["projects"])


async def test_search_limit_truncates_after_ranking(api) -> None:
    client, _root = api
    for index in range(4):
        await client.post(
            "/v1/desktop/projects", json={"name": f"proj-{index}", "description": "keyword"}
        )
    result = (
        await client.get("/v1/desktop/projects/search", params={"q": "keyword", "limit": 2})
    ).json()["result"]
    assert result["count"] == len(result["projects"]) == 2


async def test_timeline_is_one_document_with_derived_statuses(api) -> None:
    client, _root = api
    alpha = (await client.post("/v1/desktop/projects", json={"name": "alpha"})).json()["result"][
        "id"
    ]
    beta = (await client.post("/v1/desktop/projects", json={"name": "beta"})).json()["result"]["id"]
    for body in (
        {"name": "beta cut", "target_date": "2099-01-01"},
        {"name": "audit", "target_date": "2020-01-01"},
        {"name": "ship", "completed": True},
    ):
        added = await client.post(f"/v1/desktop/projects/{alpha}/milestones", json=body)
        assert added.status_code == 200

    doc = (await client.get("/v1/desktop/projects/timeline")).json()["result"]
    entries = {entry["id"]: entry for entry in doc["projects"]}
    assert set(entries) == {alpha, beta}
    assert entries[beta]["milestones"] == []  # every row appears, empty list included
    statuses = {m["name"]: m["status"] for m in entries[alpha]["milestones"]}
    assert statuses == {"beta cut": "upcoming", "audit": "overdue", "ship": "completed"}

    listing = (await client.get("/v1/desktop/projects")).json()["result"]["projects"]
    assert [entry["id"] for entry in doc["projects"]] == [row["id"] for row in listing]

    # The same derived status the milestone routes answer with: one rule.
    detail = (await client.get(f"/v1/desktop/projects/{alpha}")).json()["result"]["project"]
    detail_status = {m["name"]: m["status"] for m in detail["milestones"]}
    assert detail_status["audit"] == statuses["audit"] == "overdue"


async def test_the_two_new_reads_run_off_the_event_loop(api, monkeypatch) -> None:
    """Structural, not timing: both handlers hand store work to ``to_thread``.

    Records the thread inside the real call each route makes and asserts it is
    never the loop thread — the same shape as the repo's maintenance-callback
    spy; it fails deterministically the day a handler drops its
    ``asyncio.to_thread``.
    """

    client, _root = api
    await client.post("/v1/desktop/projects", json={"name": "alpha", "description": "keyword"})

    loop_thread = threading.get_ident()
    seen: list[int] = []
    real_search = desktop_projects.search_projects
    real_entry = desktop_projects.project_timeline_entry

    def spy_search(rows, query, *, limit=None):
        seen.append(threading.get_ident())
        return real_search(rows, query, limit=limit)

    def spy_entry(project):
        seen.append(threading.get_ident())
        return real_entry(project)

    monkeypatch.setattr(desktop_projects, "search_projects", spy_search)
    monkeypatch.setattr(desktop_projects, "project_timeline_entry", spy_entry)

    search = await client.get("/v1/desktop/projects/search", params={"q": "keyword"})
    timeline = await client.get("/v1/desktop/projects/timeline")
    assert search.status_code == 200 and timeline.status_code == 200
    assert seen, "the spies never ran — the routes stopped calling them"
    assert all(ident != loop_thread for ident in seen)
