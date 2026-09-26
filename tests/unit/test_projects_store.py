"""The project store: model edges, the write guard, and the composed view.

One file per project under ``<root>/projects/``, mutated only under a store-wide
lock and published atomically — these tests pin the edges the design names
(§V2.A.1's validation table, §V2.A.4's write guard, §V2.A.3's refresh
amendment) plus the parts a future refactor is most likely to break silently:
the no-op suppression, the derived link direction, and ``null``-means-unknown in
the view payload.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.projects import (
    MILESTONES_MAX,
    PROJECT_PROGRESS_STALE_S,
    PROJECT_SCHEMA,
    MilestoneEdit,
    Project,
    ProjectEdit,
    ProjectMilestone,
    ProjectNameConflictError,
    ProjectRegistry,
    ProjectRegistryLockTimeout,
    ProjectSchemaGuardError,
    build_project_view,
    milestone_status,
    progress_is_stale,
    scan_runtime_states,
    validate_project_name,
)

SESSION_A = "4e92693767fa"
SESSION_B = "abcdef012345"


@pytest.fixture()
def store(tmp_path: Path) -> ProjectRegistry:
    return ProjectRegistry(tmp_path)


def create(store: ProjectRegistry, name: str = "payments-migration", **fields) -> Project:
    return store.create_project(ProjectEdit(name=name, **fields), sessions=[SESSION_A])


# -- model edges ------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    ["", " has space", "-leading", "way-too-long" + "x" * 80, "sla/sh", "ünïcode"],
)
def test_name_grammar_matches_the_team_rule(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_project_name(bad)


def test_caps_and_shapes_are_enforced_at_the_model(store: ProjectRegistry) -> None:
    with pytest.raises(ValueError):
        create(store, description="x" * 241)
    with pytest.raises(ValueError):
        create(store, progress="x" * 1001)
    with pytest.raises(ValueError):
        create(store, tags=["q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12"])
    with pytest.raises(ValueError):
        create(store, tags=["Q4!"])
    with pytest.raises(ValueError):
        create(store, sessions=["not-hex-at-all"])
    with pytest.raises(ValueError):
        create(store, estimate=0)
    with pytest.raises(ValueError):
        create(store, estimate=1000.5)
    with pytest.raises(ValueError):
        create(store, estimate_unit="weeks")
    with pytest.raises(ValueError):
        create(store, start_date="2026-1-2")


def test_an_inverted_range_is_refused_only_when_both_dates_are_set(
    store: ProjectRegistry,
) -> None:
    create(store, name="ok", start_date="2026-09-20", target_date="2026-10-15")
    create(store, name="open-end", start_date="2026-09-20")  # target TBD: legal
    with pytest.raises(ValueError):
        create(store, name="bad", start_date="2026-10-15", target_date="2026-09-20")

    project = store.get_project_by_name("ok")
    assert project is not None
    # Clearing one side is how "TBD" is expressed; the update path must allow it.
    store.update_project(project.id, ProjectEdit(target_date=""))
    cleared = store.get_project_by_name("ok")
    assert cleared is not None and cleared.target_date is None


def test_milestones_are_capped_and_case_insensitively_unique(store: ProjectRegistry) -> None:
    project = create(store)
    with pytest.raises(ValueError):
        store.update_project(
            project.id,
            ProjectEdit(
                milestones=[ProjectMilestone(name=f"m{i}") for i in range(MILESTONES_MAX + 1)]
            ),
        )
    store.update_project(project.id, ProjectEdit(milestones=[ProjectMilestone(name="Beta cut")]))
    with pytest.raises(ValueError):
        store.update_project(
            project.id,
            ProjectEdit(
                milestones=[ProjectMilestone(name="Beta cut"), ProjectMilestone(name="beta CUT")]
            ),
        )
    with pytest.raises(ValueError):
        store.update_project(project.id, ProjectEdit(milestones=[ProjectMilestone(name="x" * 81)]))


def test_milestone_status_is_derived_from_dates_never_stored() -> None:
    assert milestone_status(ProjectMilestone(name="m", completed_at="2026-09-01")) == "completed"
    assert milestone_status(ProjectMilestone(name="m", target_date="2000-01-01")) == "overdue"
    assert milestone_status(ProjectMilestone(name="m", target_date="2999-01-01")) == "upcoming"
    assert milestone_status(ProjectMilestone(name="m")) == "upcoming"


# -- storage -----------------------------------------------------------------


def test_a_row_is_one_json_file_named_by_its_id(store: ProjectRegistry, tmp_path: Path) -> None:
    project = create(store, description="Payments migration", tags=["q4"])
    path = tmp_path / "projects" / f"{project.id}.json"
    assert path.is_file()
    payload = json.loads(path.read_text())
    assert payload["schema"] == PROJECT_SCHEMA
    assert payload["name"] == "payments-migration"
    assert payload["sessions"] == [SESSION_A]
    # The store's own bookkeeping is dot-prefixed and never a row.
    assert sorted(p.name for p in (tmp_path / "projects").iterdir()) == [
        ".lock",
        f"{project.id}.json",
    ]
    # No temp files survive a publish.
    assert not [
        p for p in (tmp_path / "projects").iterdir() if p.name.startswith(".") and p.name != ".lock"
    ]


def test_names_are_unique_case_insensitively(store: ProjectRegistry) -> None:
    create(store, name="Payments-Migration")
    with pytest.raises(ProjectNameConflictError):
        create(store, name="payments-migration")


def test_a_corrupt_row_does_not_hide_the_valid_ones(store: ProjectRegistry, tmp_path: Path) -> None:
    create(store, name="good")
    (tmp_path / "projects" / "deadbeef.json").write_text("{ not json")
    (tmp_path / "projects" / "note.txt").write_text("not a row")
    listed = ProjectRegistry(tmp_path).list_projects()
    assert [project.name for project in listed] == ["good"]


def test_a_foreign_row_whose_id_disagrees_with_its_name_is_skipped(
    store: ProjectRegistry, tmp_path: Path
) -> None:
    project = create(store, name="real")
    payload = json.loads((tmp_path / "projects" / f"{project.id}.json").read_text())
    payload["id"] = "0" * 32
    (tmp_path / "projects" / "1111.json").write_text(json.dumps(payload))
    listed = ProjectRegistry(tmp_path).list_projects()
    assert [item.name for item in listed] == ["real"]


def test_another_writers_row_appears_without_waiting_for_the_refresh_interval(
    tmp_path: Path,
) -> None:
    reader = ProjectRegistry(tmp_path, refresh_interval=60.0)
    assert reader.list_projects() == []
    other = ProjectRegistry(tmp_path)
    create(other, name="landed")
    # The directory mtime check is what makes this immediate; the interval alone
    # would hide a fresh row for a minute.
    assert [project.name for project in reader.list_projects()] == ["landed"]


def test_missing_registry_root_is_not_created_by_a_read(tmp_path: Path) -> None:
    ProjectRegistry(tmp_path).list_projects()
    assert not (tmp_path / "projects").exists()


def test_the_store_lock_times_out_with_guidance(store: ProjectRegistry, monkeypatch) -> None:
    from local_operator import projects as module

    monkeypatch.setattr(module, "_LOCK_TIMEOUT_S", 0.01)
    monkeypatch.setattr(module, "_try_lock_exclusive", lambda fd: False)
    with pytest.raises(ProjectRegistryLockTimeout) as excinfo:
        create(store)
    assert "retry" in str(excinfo.value)


# -- the write guard ---------------------------------------------------------


def test_a_newer_schema_is_readable_but_never_mutable(
    store: ProjectRegistry, tmp_path: Path
) -> None:
    project = create(store, name="from-the-future")
    path = tmp_path / "projects" / f"{project.id}.json"
    payload = json.loads(path.read_text())
    payload["schema"] = PROJECT_SCHEMA + 1
    payload["unknown_future_field"] = {"nested": True}
    path.write_text(json.dumps(payload))

    reader = ProjectRegistry(tmp_path)
    loaded = reader.get_project(project.id)
    assert loaded.schema_version == PROJECT_SCHEMA + 1  # reads stay lenient

    for mutate in (
        lambda: reader.update_project(project.id, ProjectEdit(description="x")),
        lambda: reader.link_session(project.id, SESSION_B),
        lambda: reader.unlink_session(project.id, SESSION_A),
        lambda: reader.set_milestone(project.id, MilestoneEdit(name="m")),
    ):
        with pytest.raises(ProjectSchemaGuardError) as excinfo:
            mutate()
        assert "newer local-operator" in str(excinfo.value)

    # REMOVAL is guarded too — an extension of §V2.A.4's rule to the delete
    # path, stated in the method: the row holds fields this build cannot
    # render, so a delete would destroy them unseen.
    with pytest.raises(ProjectSchemaGuardError):
        reader.delete_project(project.id)
    assert reader.get_project(project.id).schema_version == PROJECT_SCHEMA + 1


# -- update semantics --------------------------------------------------------


def test_progress_writes_stamp_the_reporter_and_a_no_op_writes_nothing(
    store: ProjectRegistry, tmp_path: Path
) -> None:
    project = create(store)
    path = tmp_path / "projects" / f"{project.id}.json"

    first = store.update_project(
        project.id, ProjectEdit(progress="2026-09-26 dashboard cutover done"), reporter=SESSION_A
    )
    assert first.changed and not first.refreshed
    assert first.project.progress_reported_by == SESSION_A
    assert not progress_is_stale(first.project)

    before = path.stat().st_mtime_ns
    again = store.update_project(
        project.id, ProjectEdit(progress="2026-09-26 dashboard cutover done"), reporter=SESSION_A
    )
    assert not again.changed and not again.refreshed
    assert path.stat().st_mtime_ns == before  # nothing written at all


def test_an_identical_line_on_a_stale_record_refreshes_instead_of_suppressing(
    store: ProjectRegistry, tmp_path: Path
) -> None:
    project = create(store)
    store.update_project(project.id, ProjectEdit(progress="still true"), reporter=SESSION_A)
    path = tmp_path / "projects" / f"{project.id}.json"
    payload = json.loads(path.read_text())
    payload["progress_updated_at"] = time.time() - PROJECT_PROGRESS_STALE_S - 60
    path.write_text(json.dumps(payload))

    outcome = store.update_project(
        project.id, ProjectEdit(progress="still true"), reporter=SESSION_B
    )
    assert outcome.changed and outcome.refreshed
    assert outcome.project.progress == "still true"  # the text is untouched
    assert outcome.project.progress_reported_by == SESSION_B  # the refresh is attributed


def test_clearing_progress_clears_the_freshness_pair(store: ProjectRegistry) -> None:
    project = create(store)
    store.update_project(project.id, ProjectEdit(progress="done"), reporter=SESSION_A)
    cleared = store.update_project(project.id, ProjectEdit(progress=""), reporter=SESSION_A)
    assert cleared.project.progress == ""
    assert cleared.project.progress_updated_at is None
    assert cleared.project.progress_reported_by == ""
    assert progress_is_stale(cleared.project)


def test_status_done_stamps_completion_only_on_the_transition(store: ProjectRegistry) -> None:
    project = create(store)
    done = store.update_project(project.id, ProjectEdit(status="done"))
    assert done.project.completed_at is not None
    stamped = done.project.completed_at

    # Staying done does not move the date.
    store.update_project(project.id, ProjectEdit(status="done"))
    stamped_row = store.get_project(project.id)
    assert stamped_row is not None and stamped_row.completed_at == stamped

    # Moving away leaves it; coming back stamps afresh.
    paused = store.update_project(project.id, ProjectEdit(status="paused"))
    assert paused.project.completed_at == stamped
    reopen = store.update_project(project.id, ProjectEdit(status="done"))
    assert reopen.project.completed_at is not None

    # An explicit clear is honoured even on the transition.
    create(store, name="second", status="done", completed_at="")
    second = store.get_project_by_name("second")
    assert second is not None and second.completed_at is None
    explicit = store.update_project(second.id, ProjectEdit(status="done", completed_at=""))
    assert explicit.project.completed_at is None


def test_descriptions_and_tags_replace_rather_than_merge(store: ProjectRegistry) -> None:
    project = create(store, tags=["q4", "core"])
    updated = store.update_project(project.id, ProjectEdit(tags=["q4"]))
    assert updated.project.tags == ["q4"]
    cleared = store.update_project(project.id, ProjectEdit(description=""))
    assert cleared.project.description == ""


def test_unknown_ids_raise_key_error(store: ProjectRegistry) -> None:
    with pytest.raises(KeyError):
        store.get_project("0" * 32)
    with pytest.raises(KeyError):
        store.update_project("0" * 32, ProjectEdit(description="x"))


# -- links -------------------------------------------------------------------


def test_links_are_stored_once_and_derived_backwards(store: ProjectRegistry) -> None:
    project = create(store)
    linked, added = store.link_session(project.id, SESSION_B)
    assert added and linked.sessions == [SESSION_A, SESSION_B]
    again, added_again = store.link_session(project.id, SESSION_B)
    assert not added_again and again.sessions == [SESSION_A, SESSION_B]

    assert [item.name for item in store.projects_for_session(SESSION_B)] == [project.name]
    assert store.projects_for_session("0" * 12) == []

    removed, unlinked = store.unlink_session(project.id, SESSION_B)
    assert unlinked and removed.sessions == [SESSION_A]
    _same, unlinked_again = store.unlink_session(project.id, SESSION_B)
    assert not unlinked_again


def test_the_link_cap_refuses_and_names_unlink(store: ProjectRegistry) -> None:
    project = create(store)
    for index in range(63):
        store.link_session(project.id, f"{index:012x}")
    with pytest.raises(ValueError) as excinfo:
        store.link_session(project.id, "ffffffffffff")
    assert "unlink" in str(excinfo.value)


# -- milestones --------------------------------------------------------------


def test_the_surgical_milestone_op_adds_updates_completes_and_removes(
    store: ProjectRegistry,
) -> None:
    project = create(store)
    added, action = store.set_milestone(
        project.id, MilestoneEdit(name="Beta Cut", target_date="2026-10-01")
    )
    assert action == "added" and [m.name for m in added.milestones] == ["Beta Cut"]

    # Case-insensitive hit, `completed=True` stamps today.
    completed, action = store.set_milestone(
        project.id, MilestoneEdit(name="beta cut", completed=True)
    )
    assert action == "updated"
    assert completed.milestones[0].completed_at is not None

    # Nothing supplied on an existing milestone is the truthful no-change.
    _same, action = store.set_milestone(project.id, MilestoneEdit(name="BETA CUT"))
    assert action == "unchanged"

    cleared, action = store.set_milestone(
        project.id, MilestoneEdit(name="beta cut", completed=False)
    )
    assert action == "updated" and cleared.milestones[0].completed_at is None

    with_date, action = store.set_milestone(
        project.id, MilestoneEdit(name="beta cut", target_date="2026-11-01")
    )
    assert with_date.milestones[0].target_date == "2026-11-01"
    without_date, action = store.set_milestone(
        project.id, MilestoneEdit(name="beta cut", target_date="")
    )
    assert without_date.milestones[0].target_date is None

    removed, action = store.set_milestone(project.id, MilestoneEdit(name="Beta Cut", remove=True))
    assert action == "removed" and removed.milestones == []
    with pytest.raises(KeyError):
        store.set_milestone(project.id, MilestoneEdit(name="Beta Cut", remove=True))


# -- the composed view -------------------------------------------------------


def _seed_session(
    root: Path, session_id: str, *, title: str | None, jobs: list[dict[str, Any]]
) -> Path:
    session_dir = root / "sessions" / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "created_at.json").write_text("1700000000.0")
    if title is not None:
        (session_dir / "title.json").write_text(json.dumps({"text": title, "user_set": True}))
    if jobs is not None:
        (session_dir / "subagent-roster.v1.json").write_text(
            json.dumps({"version": 1, "jobs": jobs})
        )
    return session_dir


def test_the_view_reports_unknown_as_null_never_zero(
    store: ProjectRegistry, tmp_path: Path
) -> None:
    project = create(store)
    view = build_project_view(project, config_dir=tmp_path)
    (row,) = view["sessions"]
    assert row["session_id"] == SESSION_A
    assert row["exists"] is False
    assert row["title"] is None
    assert row["runtime"] == {
        "state": "stopped",
        "busy": None,
        "heartbeat_age_s": None,
        "pid": None,
    }
    assert row["subagents"] is None
    assert row["todos"] is None
    assert view["progress_stale"] is True


def test_the_view_reads_title_roster_and_the_newest_todo_snapshot(
    store: ProjectRegistry, tmp_path: Path
) -> None:
    project = create(store)
    session_dir = _seed_session(
        tmp_path,
        SESSION_A,
        title="Payments session",
        jobs=[
            {"id": "j1", "status": "running", "label": "reviewer"},
            {"id": "j2", "status": "completed", "label": "scout"},
        ],
    )
    rows = [
        {
            "id": "a",
            "ts": 1.0,
            "type": "custom",
            "payload": {"custom_type": "todo_snapshot", "details": {"items": []}},
        },
        {
            "id": "b",
            "ts": 2.0,
            "type": "custom",
            "payload": {
                "custom_type": "todo_snapshot",
                "details": {
                    "items": [
                        {
                            "name": "Todos",
                            "items": [
                                {"text": "one", "status": "pending"},
                                {"text": "two", "status": "done"},
                            ],
                        }
                    ]
                },
            },
        },
    ]
    (session_dir / "transcript.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    view = build_project_view(project, config_dir=tmp_path)
    (row,) = view["sessions"]
    assert row["exists"] is True
    assert row["title"] == "Payments session"
    assert row["created_at"] == 1700000000.0
    assert row["subagents"] == {"running": 1, "settled": 1, "names": ["reviewer", "scout"]}
    assert row["todos"] == {"open": 1, "total": 2}


def test_archived_sessions_stay_linked_and_are_marked(
    store: ProjectRegistry, tmp_path: Path
) -> None:
    project = create(store)
    _seed_session(tmp_path, SESSION_A, title=None, jobs=[])
    (tmp_path / "archived-sessions.json").write_text(json.dumps([SESSION_A]))
    (row,) = build_project_view(project, config_dir=tmp_path)["sessions"]
    assert row["archived"] is True


def test_a_live_record_classifies_the_session(store: ProjectRegistry, tmp_path: Path) -> None:
    import os as _os
    import secrets

    project = create(store)
    _seed_session(tmp_path, SESSION_A, title=None, jobs=[])
    run_dir = tmp_path / "run" / "mobile"
    run_dir.mkdir(parents=True)
    record = {
        "pid": _os.getpid(),
        "kind": "tui",
        "session_id": SESSION_A,
        "conversation_name": "x",
        "cwd": str(tmp_path),
        "model_label": "m",
        "control_port": 1,
        "control_key": secrets.token_hex(8),
        "heartbeat_at": time.time(),
        "busy": True,
    }
    (run_dir / f"{_os.getpid()}.json").write_text(json.dumps(record))

    assert scan_runtime_states(tmp_path)[SESSION_A]["state"] == "live"
    (row,) = build_project_view(project, config_dir=tmp_path)["sessions"]
    assert row["runtime"]["state"] == "live"
    assert row["runtime"]["busy"] is True
    assert row["runtime"]["pid"] == _os.getpid()


def test_the_tui_overlay_wins_only_for_the_fields_it_passes(
    store: ProjectRegistry, tmp_path: Path
) -> None:
    project = create(store)
    view = build_project_view(
        project,
        config_dir=tmp_path,
        live={SESSION_A: {"todos": {"open": 0, "total": 0}, "runtime": {"state": "live"}}},
    )
    (row,) = view["sessions"]
    assert row["todos"] == {"open": 0, "total": 0}
    assert row["runtime"] == {"state": "live"}  # overridden wholesale, as documented
    assert row["subagents"] is None  # untouched
