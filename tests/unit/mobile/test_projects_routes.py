"""``/api/projects*`` on the wire: CRUD, links, milestones, refusals.

An isolated ``HOME``/config root (the autouse ``isolate_environment`` fixture)
and the real ``build_app`` over a ``MobileDaemon``, mirroring
``test_daemon.py``: these assertions are about what the phone is told (status
codes, machine codes, the composed view's shape), not about the store — that
has its own suite. The desktop route tests
(``tests/unit/server/test_desktop_projects.py``) cover the same matrix over the
same store; where a cell matters on both surfaces it is asserted here against
the same expected values, because "identical in spirit" is only true while it
is tested.

Nothing here touches a real session: the one live-runtime row is a synthetic
record under this test's own config root, keyed to a session id that exists
nowhere else.
"""

from __future__ import annotations

import json
import os
import time

import pytest
from starlette.testclient import TestClient

from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.paths import config_dir
from local_operator.projects import PROJECT_SCHEMA

SESSION_A = "4e92693767fa"


def _client() -> TestClient:
    """A logged-in client over the real daemon app, under the test's HOME."""
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


def _write_live_record(session_id: str = SESSION_A) -> None:
    """A runtime record for a session that reads ``live`` — this test's own pid,
    so liveness is genuinely true rather than a state name we wrote down."""
    run_dir = config_dir() / "run" / "mobile"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{os.getpid()}.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "kind": "tui",
                "session_id": session_id,
                "conversation_name": "x",
                "cwd": str(config_dir()),
                "model_label": "m",
                "control_port": 1,
                "control_key": "0" * 16,
                "heartbeat_at": time.time(),
                "busy": False,
            }
        )
    )


def test_the_gate_holds_and_login_unlocks_the_surface() -> None:
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.get("/api/projects").status_code == 401
    assert client.post("/api/projects", json={"name": "alpha"}).status_code == 401

    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    listed = client.get("/api/projects")
    assert listed.status_code == 200
    assert listed.json() == {"projects": []}


def test_create_read_patch_delete_round_trip() -> None:
    client = _client()
    created = client.post(
        "/api/projects",
        json={"name": "payments-migration", "description": "Payments", "tags": ["q4"]},
    )
    assert created.status_code == 200
    summary = created.json()["project"]
    assert summary["status"] == "active" and summary["progress_stale"] is True
    assert summary["sessions"] == 0 and summary["milestones_total"] == 0
    project_id = summary["id"]

    # The row on disk is the store's, not a route cache.
    on_disk = json.loads((config_dir() / "projects" / f"{project_id}.json").read_text())
    assert on_disk["name"] == "payments-migration" and on_disk["schema"] == PROJECT_SCHEMA

    by_name = client.get("/api/projects/Payments-Migration")
    assert by_name.status_code == 200
    assert by_name.json()["project"]["id"] == project_id

    patched = client.patch(
        f"/api/projects/{project_id}",
        json={"status": "done", "progress": "cutover done"},
    )
    assert patched.status_code == 200
    result = patched.json()["project"]
    assert result["status"] == "done"
    # status -> done with completed_at omitted stamps today (the one convenience).
    assert result["completed_at"] is not None
    assert result["progress_stale"] is False

    detail = client.get(f"/api/projects/{project_id}")
    payload = detail.json()
    assert payload["project"]["progress_reported_by"] == "operator"
    assert payload["links"] == []

    refused = client.request(
        "DELETE", f"/api/projects/{project_id}", json={"confirm": "not-the-name"}
    )
    assert refused.status_code == 422
    assert refused.json()["code"] == "project_confirm_mismatch"

    deleted = client.request(
        "DELETE", f"/api/projects/{project_id}", json={"confirm": "Payments-Migration"}
    )
    assert deleted.status_code == 200 and deleted.json() == {"ok": True, "deleted": True}
    missing = client.get(f"/api/projects/{project_id}")
    assert missing.status_code == 404
    assert missing.json()["code"] == "project_not_found"


def test_a_taken_name_is_a_409_and_invalid_values_are_422() -> None:
    client = _client()
    assert client.post("/api/projects", json={"name": "alpha"}).status_code == 200
    duplicate = client.post("/api/projects", json={"name": "ALPHA"})
    assert duplicate.status_code == 409
    assert duplicate.json()["code"] == "project_name_exists"

    beta = client.post("/api/projects", json={"name": "beta"}).json()["project"]["id"]
    for body in (
        {"tags": ["Q4!"]},
        {"estimate": 0},
        {"start_date": "2026-10-10", "target_date": "2026-10-01"},
        {"status": "sideways"},
        {"nmae": "typo"},
    ):
        bad = client.patch(f"/api/projects/{beta}", json=body)
        assert bad.status_code == 422, body
        assert bad.json()["code"] == "project_invalid"

    shape = client.post("/api/projects", json={"name": "x" * 65})
    assert shape.status_code == 422
    nameless = client.post("/api/projects", json={"description": "no name"})
    assert nameless.status_code == 422

    # A body that is not a JSON object is a 400, the daemon's body-shaped
    # refusal — distinct from a well-formed body the store rejects, and in the
    # daemon's own sentence (the five pre-existing handlers', verbatim).
    not_an_object = client.post(
        "/api/projects", content=b"[]", headers={"content-type": "application/json"}
    )
    assert not_an_object.status_code == 400
    assert not_an_object.json()["error"] == "request body must be an object"
    assert (
        client.request(
            "DELETE",
            f"/api/projects/{beta}",
            content=b"not json",
            headers={"content-type": "application/json"},
        ).status_code
        == 400
    )


def test_milestone_routes_report_derived_status_and_refuse_unknowns() -> None:
    client = _client()
    project_id = client.post("/api/projects", json={"name": "alpha"}).json()["project"]["id"]

    added = client.post(
        f"/api/projects/{project_id}/milestones",
        json={"name": "beta cut", "target_date": "2026-10-01"},
    )
    assert added.status_code == 200
    (milestone,) = added.json()["project"]["milestones"]
    assert milestone == {
        "name": "beta cut",
        "target_date": "2026-10-01",
        "completed_at": None,
        "status": "upcoming",
    }

    overdue = client.post(
        f"/api/projects/{project_id}/milestones",
        json={"name": "audit", "target_date": "2020-01-01"},
    )
    overdue_row = next(m for m in overdue.json()["project"]["milestones"] if m["name"] == "audit")
    assert overdue_row["status"] == "overdue"

    # The toggle: completed=True stamps today; False clears it again.
    toggled = client.post(
        f"/api/projects/{project_id}/milestones", json={"name": "Beta Cut", "completed": True}
    )
    done_row = next(
        m for m in toggled.json()["project"]["milestones"] if m["name"].casefold() == "beta cut"
    )
    assert done_row["status"] == "completed" and done_row["completed_at"] is not None
    cleared = client.post(
        f"/api/projects/{project_id}/milestones", json={"name": "beta cut", "completed": False}
    )
    cleared_row = next(
        m for m in cleared.json()["project"]["milestones"] if m["name"] == "beta cut"
    )
    assert cleared_row["completed_at"] is None and cleared_row["status"] == "upcoming"

    removed = client.delete(f"/api/projects/{project_id}/milestones/beta%20cut")
    assert removed.status_code == 200
    assert [m["name"] for m in removed.json()["project"]["milestones"]] == ["audit"]

    missing = client.delete(f"/api/projects/{project_id}/milestones/ghost")
    assert missing.status_code == 422
    assert "no milestone named 'ghost'" in missing.json()["error"]


def test_links_round_trip_and_the_view_counts_live_sessions() -> None:
    client = _client()
    project_id = client.post("/api/projects", json={"name": "alpha"}).json()["project"]["id"]
    _write_live_record()

    linked = client.post(f"/api/projects/{project_id}/links", json={"session_id": SESSION_A})
    assert linked.status_code == 200
    assert linked.json()["project"]["sessions"] == 1
    assert linked.json()["project"]["live_sessions"] == 1

    detail = client.get(f"/api/projects/{project_id}").json()
    (link,) = detail["links"]
    assert link["session_id"] == SESSION_A
    assert link["runtime"]["state"] == "live"
    assert link["exists"] is False and link["todos"] is None and link["subagents"] is None

    bad = client.post(f"/api/projects/{project_id}/links", json={"session_id": "nope"})
    assert bad.status_code == 422

    unlinked = client.delete(f"/api/projects/{project_id}/links/{SESSION_A}")
    assert unlinked.status_code == 200
    assert unlinked.json()["project"]["sessions"] == 0


def test_the_listing_is_sorted_by_status_then_freshness() -> None:
    client = _client()
    for name, status in (
        ("archived-one", "archived"),
        ("active-old", "active"),
        ("paused", "paused"),
    ):
        made = client.post("/api/projects", json={"name": name, "status": status})
        assert made.status_code == 200
    # A later edit makes one active row fresher than the other.
    patched = client.patch("/api/projects/active-old", json={"description": "touched"})
    assert patched.status_code == 200
    rows = [row["name"] for row in client.get("/api/projects").json()["projects"]]
    assert rows == ["active-old", "paused", "archived-one"]


def test_an_unknown_key_is_a_404_naming_the_closest() -> None:
    client = _client()
    client.post("/api/projects", json={"name": "payments-migration"})
    client.post("/api/projects", json={"name": "payments-cutover"})
    missing = client.get("/api/projects/payments")
    assert missing.status_code == 404
    body = missing.json()
    assert body["code"] == "project_not_found"
    # Up to two prefix-matches, in the store's own sorted-by-name order: the
    # remedy names real rows rather than restating the key that failed.
    assert "closest: payments-cutover, payments-migration" in body["error"]


def test_a_row_from_a_newer_build_is_readable_but_refuses_mutation() -> None:
    client = _client()
    project_id = client.post("/api/projects", json={"name": "alpha"}).json()["project"]["id"]
    path = config_dir() / "projects" / f"{project_id}.json"
    payload = json.loads(path.read_text())
    payload["schema"] = PROJECT_SCHEMA + 1
    payload["future_field"] = True
    payload["sessions"] = [SESSION_A]
    path.write_text(json.dumps(payload))

    assert client.get(f"/api/projects/{project_id}").status_code == 200  # reads stay lenient

    for response in (
        client.patch(f"/api/projects/{project_id}", json={"description": "x"}),
        client.post(
            f"/api/projects/{project_id}/milestones", json={"name": "m", "completed": True}
        ),
        client.post(f"/api/projects/{project_id}/links", json={"session_id": "abcdef012345"}),
        client.delete(f"/api/projects/{project_id}/links/{SESSION_A}"),
        client.request("DELETE", f"/api/projects/{project_id}", json={"confirm": "alpha"}),
    ):
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "project_schema_newer"

    # ... and none of them wrote: the row still exists, still at the newer schema.
    still_there = json.loads(path.read_text())
    assert still_there["schema"] == PROJECT_SCHEMA + 1 and still_there["sessions"] == [SESSION_A]


def test_lock_contention_answers_503(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.projects import ProjectRegistry, ProjectRegistryLockTimeout

    client = _client()
    project_id = client.post("/api/projects", json={"name": "alpha"}).json()["project"]["id"]

    def refuse(self, *args: object, **kwargs: object):
        raise ProjectRegistryLockTimeout("Timed out waiting for the projects registry lock")

    monkeypatch.setattr(ProjectRegistry, "update_project", refuse)
    busy = client.patch(f"/api/projects/{project_id}", json={"description": "x"})
    assert busy.status_code == 503
    assert busy.json()["code"] == "project_store_busy"


def test_an_invalid_value_reads_as_the_stores_own_sentence() -> None:
    """Design round 1, D4: the 422 body carries prose, not a validator dump.

    ``readable_error`` renders pydantic as ``<field>: Value error, <sentence>``;
    the phone's refusal boundary drops that wrapper so the reader meets the
    store's own sentence — the way the 409 case reads ("project 'x' already
    exists"). The machine code is what a client keys on; the sentence is for
    the person holding the phone.
    """
    client = _client()
    bad = client.post("/api/projects", json={"name": "bad name"})
    assert bad.status_code == 422
    body = bad.json()
    assert body["code"] == "project_invalid"
    assert "Value error" not in body["error"]
    assert body["error"].startswith("project name must be")


def test_the_request_key_tuples_are_the_desktop_models_fields() -> None:
    """Round 1, [m]3: the phone's accepted-key tuples are DERIVED from the
    desktop request models, never a hand copy that drifts when a model grows.

    With the derivation in place this pin exists for the revert: a literal
    copy that matches today's fields passes until a model changes, and then
    this test fails — which is the drift the round-1 review found.
    """
    from local_operator.mobile import projects as mobile_projects
    from local_operator.server.models.desktop_projects import (
        LinkMutation,
        MilestoneMutation,
        ProjectCreate,
        ProjectDelete,
        ProjectPatch,
    )

    assert mobile_projects._CREATE_FIELDS == tuple(ProjectCreate.model_fields)
    assert mobile_projects._DELETE_FIELDS == tuple(ProjectDelete.model_fields)
    assert mobile_projects._LINK_FIELDS == tuple(LinkMutation.model_fields)
    assert mobile_projects._PATCH_FIELDS == tuple(ProjectPatch.model_fields)
    assert mobile_projects._MILESTONE_FIELDS == tuple(MilestoneMutation.model_fields)


def test_the_summary_and_view_shapes_are_the_desktop_wire_models() -> None:
    """The phone's payloads ARE the desktop's models — same keys, same derived
    fields. A second builder would pass its own tests and fail this one."""
    client = _client()
    created = client.post("/api/projects", json={"name": "alpha"}).json()["project"]
    from local_operator.server.models.desktop_projects import (
        ProjectSummary,
        ProjectView,
    )

    assert set(created) == set(ProjectSummary.model_fields)
    detail = client.get("/api/projects/alpha").json()
    assert set(detail["project"]) == set(ProjectView.model_fields)
