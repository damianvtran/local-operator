"""The project store: model edges, the write guard, and the composed view.

One file per project under ``<root>/projects/``, mutated only under a store-wide
lock and published atomically — these tests pin the edges the design names
(§V2.A.1's validation table, §V2.A.4's write guard, §V2.A.3's refresh
amendment) plus the parts a future refactor is most likely to break silently:
the no-op suppression, the derived link direction, and ``null``-means-unknown in
the view payload.
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.projects import (
    ATTACHMENTS_MAX,
    MILESTONES_MAX,
    PROJECT_PROGRESS_STALE_S,
    PROJECT_SCHEMA,
    UPDATES_MAX,
    MilestoneEdit,
    Project,
    ProjectEdit,
    ProjectMilestone,
    ProjectNameConflictError,
    ProjectRegistry,
    ProjectRegistryLockTimeout,
    ProjectSchemaGuardError,
    build_project_view,
    display_name,
    milestone_status,
    progress_is_stale,
    scan_runtime_states,
    stale_after_s,
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
    # The cap refusal NAMES its remedy, one shape with the link cap's "unlink
    # one first" (agent review round 1, m1 — the guide promises both name it).
    with pytest.raises(ValueError, match="remove one"):
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


def test_the_default_date_basis_is_the_local_today_the_stamp_comes_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One "today" or the badge contradicts the stamp (agent review round 1, n1).

    Pinned by DISCRIMINATION, not by reading the source: the patched moment
    (2000-01-01) is BEFORE the target (2000-06-01), which is before any real
    run's local today. A helper-basis run calls that "upcoming"; a fallback to
    ``date.today()`` calls it "overdue", so the test fails if the two bases
    ever drift apart again. The basis is the OPERATOR'S LOCAL day (UX round 1
    follow-up: the stamp is a human-facing date, and a UTC basis stored
    tomorrow's day for an evening toggle west of Greenwich) — the one-basis
    rule this test exists for is what must not move.
    """

    from local_operator import projects

    monkeypatch.setattr(projects, "_local_today", lambda: datetime.date(2000, 1, 1))
    assert milestone_status(ProjectMilestone(name="m", target_date="2000-06-01")) == "upcoming"
    assert projects._today_iso() == "2000-01-01"


def test_local_today_reads_the_local_clock_not_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    """UX round 1 follow-up: the stamp is the day the OPERATOR saw.

    Reproduces the measured case: a 2026-09-29 20:5x EDT toggle stored
    2026-09-30 (UTC's tomorrow). The fake clock's naive ``now()`` is the
    evening of the 29th while its UTC reading is already the 30th — a UTC
    basis stamps the 30th; the local basis must say 29.
    """
    import datetime as dt

    from local_operator import projects

    class FakeDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return dt.datetime(2026, 9, 29, 20, 55)
            return dt.datetime(2026, 9, 30, 0, 55, tzinfo=dt.timezone.utc)

    monkeypatch.setattr(projects, "datetime", FakeDatetime)
    assert projects._today_iso() == "2026-09-29"


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
    """Refresh ≠ update: the identical line re-checks, it does not re-date.

    The content clock is the one the badge reads, so a refresh must NOT move
    it (the operator's "must not clear it"); the assertion pair is what moves,
    and the checker is attributed separately from the author because the line's
    authorship did not change.
    """
    project = create(store)
    store.update_project(project.id, ProjectEdit(progress="still true"), reporter=SESSION_A)
    path = tmp_path / "projects" / f"{project.id}.json"
    payload = json.loads(path.read_text())
    old_stamp = time.time() - PROJECT_PROGRESS_STALE_S - 60
    payload["progress_updated_at"] = old_stamp
    path.write_text(json.dumps(payload))

    outcome = store.update_project(
        project.id, ProjectEdit(progress="still true"), reporter=SESSION_B
    )
    assert outcome.changed and outcome.refreshed
    refreshed = outcome.project
    assert refreshed.progress == "still true"  # the text is untouched
    assert refreshed.progress_updated_at == old_stamp  # and the CONTENT clock is unmoved
    assert refreshed.progress_reported_by == SESSION_A  # authorship is untouched
    assert refreshed.progress_refreshed_by == SESSION_B  # the CHECK is attributed
    assert refreshed.progress_refreshed_at is not None

    # A whitespace-only variant of the stored line is the same line: refresh,
    # never an append (the dedupe rule is normalized equality).
    path.write_text(json.dumps({**json.loads(path.read_text()), "progress_updated_at": old_stamp}))
    again = store.update_project(
        project.id, ProjectEdit(progress="  still   true\n"), reporter=SESSION_B
    )
    assert again.refreshed and again.project.updates == refreshed.updates
    assert again.project.progress == "still true"  # the stored text is not re-spaced

    # And the next NEW line CLEARS the assertion: it described superseded text.
    moved = store.update_project(project.id, ProjectEdit(progress="now moved"), reporter=SESSION_A)
    assert moved.project.progress_refreshed_at is None
    assert moved.project.progress_refreshed_by == ""
    assert moved.project.progress_updated_at != old_stamp


@pytest.mark.parametrize(
    ("stored", "resent"),
    [
        ("QA 7/7", "QA 0/7"),
        ("merged as c9326d8", "merged as 0b2dc18"),
        ("P&S/DC cut done", "P&SD cut done"),
    ],
)
def test_a_near_identical_resend_is_an_update_not_a_refresh(
    store: ProjectRegistry, stored: str, resent: str
) -> None:
    """No similarity heuristic: tiny edits flip meaning (7/7 vs 0/7; one SHA
    vs another; a typo flip), so anything but exact-normalized equality
    appends and moves the content clock."""
    project = create(store, progress=stored)
    path = store.projects_dir / f"{project.id}.json"
    payload = json.loads(path.read_text())
    payload["progress_updated_at"] = time.time() - PROJECT_PROGRESS_STALE_S - 60
    path.write_text(json.dumps(payload))

    outcome = store.update_project(project.id, ProjectEdit(progress=resent), reporter=SESSION_A)
    assert outcome.changed and not outcome.refreshed
    assert outcome.project.progress == resent
    assert outcome.project.updates[-1].text == resent
    assert outcome.project.progress_refreshed_at is None


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


# -- attribution, title, the history log, and attachments --------------------


def test_owner_team_and_title_round_trip_set_clear_and_absent(store) -> None:
    project = create(store)
    assert project.owner is None and project.team is None and project.title is None
    assert display_name(project) == project.name  # absent falls back to the key

    set_row = store.update_project(
        project.id, ProjectEdit(owner="  Damian Tran ", team="Platform", title=" Q4 Payments ")
    )
    # Trimmed on write, merge-only on read.
    assert (set_row.project.owner, set_row.project.team, set_row.project.title) == (
        "Damian Tran",
        "Platform",
        "Q4 Payments",
    )
    assert display_name(set_row.project) == "Q4 Payments"
    reloaded = store.get_project(project.id)
    assert (reloaded.owner, reloaded.team, reloaded.title) == (
        "Damian Tran",
        "Platform",
        "Q4 Payments",
    )

    merged = store.update_project(project.id, ProjectEdit(progress="moved"))
    assert merged.project.owner == "Damian Tran" and merged.project.title == "Q4 Payments"

    cleared = store.update_project(project.id, ProjectEdit(title="", owner="team ", team=""))
    assert cleared.project.title is None and cleared.project.team is None
    assert cleared.project.owner == "team"
    assert display_name(cleared.project) == cleared.project.name


def test_owner_team_and_title_caps_refuse_with_the_field_name(store) -> None:
    project = create(store)
    with pytest.raises(ValueError) as excinfo:
        store.update_project(project.id, ProjectEdit(owner="x" * 81))
    assert "owner must be at most 80 characters" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        store.update_project(project.id, ProjectEdit(team="x" * 81))
    assert "team must be at most 80 characters" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        store.update_project(project.id, ProjectEdit(title="x" * 81))
    assert "title must be at most 80 characters" in str(excinfo.value)


def test_the_staleness_window_is_four_hours(store) -> None:
    project = create(store, progress="still true")
    stamp = project.progress_updated_at
    assert stamp is not None
    row = store.get_project(project.id)
    assert not progress_is_stale(row, now=stamp + 4 * 3600 - 60)  # 3:59 — fresh
    assert progress_is_stale(row, now=stamp + 4 * 3600 + 60)  # 4:01 — stale
    # The rule is "more than": exactly at the window is still fresh, and the
    # ONE constant is what the boundary is measured against.
    assert not progress_is_stale(row, now=stamp + PROJECT_PROGRESS_STALE_S)
    assert PROJECT_PROGRESS_STALE_S == 4 * 3600


def test_the_staleness_window_reads_the_configured_hours(
    store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``projects.stale_after_hours`` moves the boundary the rule measures."""
    import yaml

    root = tmp_path / "cfg"
    home = tmp_path / "home"
    root.mkdir()
    home.mkdir()
    (root / "config.yml").write_text(
        yaml.safe_dump({"values": {"projects": {"stale_after_hours": 2}}})
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("HOME", str(home))

    project = create(store, progress="still true")
    row = store.get_project(project.id)
    stamp = row.progress_updated_at
    assert stamp is not None
    assert not progress_is_stale(row, now=stamp + 2 * 3600 - 60)
    assert progress_is_stale(row, now=stamp + 2 * 3600 + 60)

    # Removing the key restores the default through the same resolver.
    (root / "config.yml").write_text("")
    assert not progress_is_stale(row, now=stamp + 2 * 3600 + 60)
    assert progress_is_stale(row, now=stamp + 4 * 3600 + 60)


def test_stale_after_s_falls_back_without_a_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An install whose config was never written pays the default, no raise."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "never-written"))
    monkeypatch.setenv("HOME", str(home))

    assert stale_after_s() == PROJECT_PROGRESS_STALE_S


def test_settled_records_never_read_stale(store) -> None:
    project = create(store, progress="settled")
    row = store.get_project(project.id)
    stale_moment = row.progress_updated_at + PROJECT_PROGRESS_STALE_S + 60
    assert progress_is_stale(row, now=stale_moment)  # active + an old report

    for status in ("paused", "done", "archived"):
        settled = store.update_project(project.id, ProjectEdit(status=status)).project
        assert not progress_is_stale(settled, now=stale_moment)
        assert not progress_is_stale(settled)  # and against the real clock


def test_every_new_line_appends_to_the_history(store) -> None:
    project = create(store)
    assert project.updates == []

    first = store.update_project(project.id, ProjectEdit(progress="first"), reporter=SESSION_A)
    (entry,) = first.project.updates
    assert (entry.text, entry.by) == ("first", SESSION_A)
    assert entry.at.endswith("Z") and "T" in entry.at  # ISO-8601 UTC

    second = store.update_project(project.id, ProjectEdit(progress="second"), reporter=SESSION_B)
    assert [update.text for update in second.project.updates] == ["first", "second"]

    # A refresh of the SAME text on a stale record re-stamps freshness but
    # appends nothing (no new report); a clear appends nothing either.
    path = store.projects_dir / f"{project.id}.json"
    payload = json.loads(path.read_text())
    payload["progress_updated_at"] = 0.0
    path.write_text(json.dumps(payload))
    refreshed = store.update_project(project.id, ProjectEdit(progress="second"), reporter=SESSION_A)
    assert refreshed.refreshed
    assert [update.text for update in refreshed.project.updates] == ["first", "second"]
    cleared = store.update_project(project.id, ProjectEdit(progress=""))
    assert [update.text for update in cleared.project.updates] == ["first", "second"]


def test_creating_with_a_progress_line_appends_it(store) -> None:
    project = store.create_project(
        ProjectEdit(name="alpha", progress="created with a line"),
        sessions=[SESSION_A],
        progress_reported_by=SESSION_A,
    )
    assert [(entry.text, entry.by) for entry in project.updates] == [
        ("created with a line", SESSION_A)
    ]


def test_the_history_is_bounded_at_five_hundred_oldest_first_out(store) -> None:
    project = create(store)
    path = store.projects_dir / f"{project.id}.json"
    payload = json.loads(path.read_text())
    payload["updates"] = [
        {
            "at": f"2026-01-01T{i // 3600:02d}:{(i // 60) % 60:02d}:{i % 60:02d}Z",
            "text": f"seed {i}",
        }
        for i in range(UPDATES_MAX)
    ]
    path.write_text(json.dumps(payload))
    outcome = store.update_project(project.id, ProjectEdit(progress="newest"), reporter=SESSION_A)
    texts = [entry.text for entry in outcome.project.updates]
    assert len(texts) == UPDATES_MAX
    assert texts[0] == "seed 1" and texts[-1] == "newest"
    # The bound also holds on a pure reload (the reader keeps the same slice).
    assert len(store.get_project(project.id).updates) == UPDATES_MAX


def test_attachments_are_copied_into_the_store_and_described(store, tmp_path: Path) -> None:
    project = create(store)
    shot = tmp_path / "frame.png"
    shot.write_bytes(b"\x89PNG" + b"x" * 2048)
    log = tmp_path / "out.log"
    log.write_text("all green")

    outcome = store.update_project(
        project.id,
        ProjectEdit(progress="with evidence"),
        reporter=SESSION_A,
        attachments=[shot, str(log)],
    )
    (entry,) = outcome.project.updates
    first, second = entry.attachments
    assert (first.name, first.kind, first.bytes) == ("frame.png", "image", 2052)
    assert first.added_at.endswith("Z")
    assert (second.name, second.kind) == ("out.log", "data")

    # The copy lives under the store root with a fresh unique name; the
    # original location is never referenced (a reaped scratch dir cannot
    # take the evidence with it).
    copied = Path(first.path)
    assert copied.exists()
    assert copied.parent == store.projects_dir / "attachments" / project.id
    assert copied.name != "frame.png" and copied.suffix == ".png"
    shot.unlink()
    assert copied.read_bytes().startswith(b"\x89PNG")


def test_attachment_refusals_name_the_remedy(store, tmp_path: Path) -> None:
    project = create(store)
    store.update_project(project.id, ProjectEdit(progress="baseline"), reporter=SESSION_A)
    shot = tmp_path / "s.png"
    shot.write_bytes(b"x")
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (5 * 1024 * 1024 + 1))

    with pytest.raises(ValueError) as excinfo:
        store.update_project(
            project.id,
            ProjectEdit(progress="many"),
            reporter=SESSION_A,
            attachments=[shot] * (ATTACHMENTS_MAX + 1),
        )
    assert "at most 10 attachments" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        store.update_project(
            project.id, ProjectEdit(progress="big"), reporter=SESSION_A, attachments=[big]
        )
    assert "limited to 5.0 MB each" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        store.update_project(
            project.id,
            ProjectEdit(progress="missing"),
            reporter=SESSION_A,
            attachments=[tmp_path / "nope.png"],
        )
    assert "no file at" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        store.update_project(
            project.id, ProjectEdit(progress="dir"), reporter=SESSION_A, attachments=[tmp_path]
        )
    assert "is not a file" in str(excinfo.value)

    # No NEW line → no entry to carry files: a refresh, a clear and a
    # progress-less update all refuse.
    for fields in (
        ProjectEdit(progress="baseline"),
        ProjectEdit(progress=""),
        ProjectEdit(title="t"),
    ):
        with pytest.raises(ValueError) as excinfo:
            store.update_project(project.id, fields, reporter=SESSION_A, attachments=[shot])
        assert "NEW progress line" in str(excinfo.value)

    assert [entry.text for entry in store.get_project(project.id).updates] == ["baseline"]


def test_malformed_history_reads_as_empty_never_refuses_the_row(store) -> None:
    project = create(store)
    path = store.projects_dir / f"{project.id}.json"
    payload = json.loads(path.read_text())
    payload["updates"] = [
        "junk",
        42,
        {
            "at": 5,
            "text": None,
            "by": "x",
            "attachments": ["junk", {"name": "a.png", "kind": "wat", "bytes": "12"}],
        },
        {"text": "kept"},
    ]
    path.write_text(json.dumps(payload))
    reader = ProjectRegistry(store.config_dir)
    reloaded = reader.get_project(project.id)
    assert reloaded is not None
    assert [entry.text for entry in reloaded.updates] == ["", "kept"]
    malformed = reloaded.updates[0]
    assert malformed.at == "5" and malformed.by == "x"
    (attachment,) = malformed.attachments
    assert (attachment.name, attachment.kind, attachment.bytes) == ("a.png", "data", 12)

    # A wholly non-list history is [] — and the row still loads.
    reloaded_value = json.loads(path.read_text())
    reloaded_value["updates"] = "garbage"
    path.write_text(json.dumps(reloaded_value))
    assert ProjectRegistry(store.config_dir).get_project(project.id).updates == []


def test_a_row_written_before_the_new_fields_loads_as_unknown(store) -> None:
    project = create(store)
    path = store.projects_dir / f"{project.id}.json"
    payload = json.loads(path.read_text())
    for key in ("owner", "team", "title", "updates"):
        payload.pop(key, None)
    path.write_text(json.dumps(payload))

    reloaded = ProjectRegistry(store.config_dir).get_project(project.id)
    assert reloaded is not None
    assert reloaded.owner is None and reloaded.team is None and reloaded.title is None
    assert reloaded.updates == []
    assert display_name(reloaded) == reloaded.name
    # The view payload carries the same unknowns as nulls, never inventing
    # a default string.
    view = build_project_view(reloaded, config_dir=store.config_dir)
    assert view["project"]["owner"] is None
    assert view["project"]["title"] is None
    assert view["project"]["updates"] == []


def test_delete_reclaims_the_projects_attachment_files(store, tmp_path) -> None:
    project = create(store)
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"x" * 64)
    store.update_project(
        project.id, ProjectEdit(progress="with evidence"), reporter=SESSION_A, attachments=[shot]
    )
    attached = Path(store.get_project(project.id).updates[-1].attachments[0].path)
    assert attached.exists()

    store.delete_project(project.id)
    assert not (store.projects_dir / "attachments" / project.id).exists()
    assert not attached.exists()


def test_history_eviction_reclaims_the_evicted_entrys_files(store) -> None:
    project = create(store)
    folder = store.projects_dir / "attachments" / project.id
    folder.mkdir(parents=True)
    seeds = []
    for index in range(UPDATES_MAX):
        path = folder / f"seed-{index}.bin"
        path.write_bytes(b"y")
        seeds.append(path)
    row = json.loads((store.projects_dir / f"{project.id}.json").read_text())
    row["updates"] = [
        {
            "at": f"2026-01-01T00:00:{index % 60:02d}Z",
            "text": f"seed {index}",
            "by": "",
            "attachments": [
                {
                    "name": seeds[index].name,
                    "kind": "data",
                    "path": str(seeds[index]),
                    "bytes": 1,
                }
            ],
        }
        for index in range(UPDATES_MAX)
    ]
    (store.projects_dir / f"{project.id}.json").write_text(json.dumps(row))

    reader = ProjectRegistry(store.config_dir)
    outcome = reader.update_project(project.id, ProjectEdit(progress="newest"), reporter=SESSION_A)
    assert len(outcome.project.updates) == UPDATES_MAX
    assert outcome.project.updates[-1].text == "newest"
    assert not seeds[0].exists()  # evicted oldest-first -> its file reclaimed
    assert seeds[-1].exists()  # every entry still in the log keeps its file


def test_eviction_never_deletes_files_outside_the_store(store, tmp_path) -> None:
    # A hand-edited row must not be able to turn the reclaim into an
    # arbitrary-file delete: only paths under the store's attachments root
    # are ever unlinked (agent review round 1, F1's safety rule).
    project = create(store)
    outside = tmp_path / "keep.txt"
    outside.write_text("keep me")
    row = json.loads((store.projects_dir / f"{project.id}.json").read_text())
    row["updates"] = [
        {
            "at": "2026-01-01T00:00:00Z",
            "text": "outside path",
            "by": "",
            "attachments": [{"name": "keep.txt", "kind": "data", "path": str(outside), "bytes": 7}],
        },
        *[
            {"at": f"2026-01-02T00:00:{index % 60:02d}Z", "text": f"filler {index}", "by": ""}
            for index in range(UPDATES_MAX - 1)
        ],
    ]
    (store.projects_dir / f"{project.id}.json").write_text(json.dumps(row))

    reader = ProjectRegistry(store.config_dir)
    outcome = reader.update_project(project.id, ProjectEdit(progress="newest"), reporter=SESSION_A)
    assert outcome.project.updates[-1].text == "newest"  # the eviction ran
    assert not outcome.project.updates[0].attachments  # ... past the outside path
    assert outside.exists()


def test_a_refused_update_leaves_no_copied_files(store, tmp_path) -> None:
    alpha = create(store)
    create(store, name="beta")
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"z")

    with pytest.raises(ProjectNameConflictError):
        store.update_project(
            alpha.id,
            ProjectEdit(name="beta", progress="rename me"),
            reporter=SESSION_A,
            attachments=[shot],
        )

    folder = store.projects_dir / "attachments" / alpha.id
    assert not (folder.exists() and any(folder.iterdir()))
    assert store.get_project(alpha.id).updates == []


def test_a_midcopy_failure_reclaims_the_copies_already_made(store, tmp_path, monkeypatch) -> None:
    import local_operator.projects as projects_module

    project = create(store)
    first = tmp_path / "first.png"
    first.write_bytes(b"1")
    second = tmp_path / "second.png"
    second.write_bytes(b"2")
    real = projects_module.shutil.copyfile
    calls = {"count": 0}

    def flaky(source, destination):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("disk full")
        return real(source, destination)

    monkeypatch.setattr(projects_module.shutil, "copyfile", flaky)
    with pytest.raises(OSError):
        store.update_project(
            project.id,
            ProjectEdit(progress="two files"),
            reporter=SESSION_A,
            attachments=[first, second],
        )

    folder = store.projects_dir / "attachments" / project.id
    assert not (folder.exists() and any(folder.iterdir()))
    assert store.get_project(project.id).updates == []


def test_a_post_replace_failure_keeps_the_rows_files(store, tmp_path, monkeypatch) -> None:
    """F3: once the row replace lands, the file policy mirrors the row policy.

    ``_save_project_locked`` keeps the just-replaced bytes when the follow-up
    directory fsync fails; the take-back must not run past that point, or the
    live row would point at a deleted file.
    """
    import local_operator.projects as projects_module

    project = create(store)
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"x" * 32)
    calls = {"count": 0}

    def flaky(path):
        calls["count"] += 1
        raise OSError("EIO: simulated directory fsync failure")

    monkeypatch.setattr(projects_module, "_fsync_dir", flaky)
    with pytest.raises(OSError):
        store.update_project(
            project.id,
            ProjectEdit(progress="newest"),
            reporter=SESSION_A,
            attachments=[shot],
        )

    # The simulated failure fired on the post-replace directory fsync...
    assert calls["count"] >= 1
    # ...and the row that replace landed must keep its files.
    payload = json.loads((store.projects_dir / f"{project.id}.json").read_text())
    stored_paths = [
        attachment["path"]
        for entry in payload["updates"]
        for attachment in entry.get("attachments") or []
    ]
    assert stored_paths
    assert all(Path(path).exists() for path in stored_paths)
    # The store converges to the landed row on a fresh read.
    assert store.get_project(project.id).updates[-1].text == "newest"


def test_the_status_vocabulary_is_one_source_and_round_trips(store) -> None:
    from typing import get_args

    from local_operator.projects import (
        PROJECT_LIVE_STATUSES,
        PROJECT_STATUSES,
        ProjectStatus,
    )

    # ONE copy: the model literal, the tool's vocabulary and the refusal text
    # all derive from PROJECT_STATUSES (and the tool's list is this tuple).
    assert tuple(get_args(ProjectStatus)) == PROJECT_STATUSES
    assert PROJECT_LIVE_STATUSES == {"planning", "active", "qa", "validation"}
    for status in PROJECT_STATUSES:
        project = create(store, name=f"st-{status}", status=status)
        assert store.get_project(project.id).status == status


@pytest.mark.parametrize("status", ["planning", "active", "qa", "validation"])
def test_in_flight_statuses_can_read_stale(store, status) -> None:
    project = create(store, name=f"live-{status}", status=status, progress="older line")
    row = store.get_project(project.id)
    stale_moment = row.progress_updated_at + PROJECT_PROGRESS_STALE_S + 60
    assert progress_is_stale(row, now=stale_moment)


def test_never_reported_is_stale_for_every_in_flight_status(store) -> None:
    for status in ("planning", "active", "qa", "validation"):
        project = create(store, name=f"nr-{status}", status=status)
        assert progress_is_stale(project), status
    for status in ("paused", "done", "archived"):
        project = create(store, name=f"nr-settled-{status}", status=status)
        assert not progress_is_stale(project), status


def test_lifecycle_rows_written_before_the_extension_still_load(store) -> None:
    # Back-compat: every status the OLD vocabulary could write is still valid,
    # and a fresh reader loads each row unchanged.
    project = create(store)
    row = json.loads((store.projects_dir / f"{project.id}.json").read_text())
    for legacy in ("active", "paused", "done", "archived"):
        row["status"] = legacy
        (store.projects_dir / f"{project.id}.json").write_text(json.dumps(row))
        reader = ProjectRegistry(store.config_dir)
        assert reader.get_project(project.id).status == legacy


def test_done_needs_complete_milestones_or_force_done(store) -> None:
    project = create(
        store,
        milestones=[
            ProjectMilestone(name="beta cut"),
            ProjectMilestone(name="gamma review", completed_at="2026-01-01"),
        ],
    )
    with pytest.raises(ValueError) as excinfo:
        store.update_project(project.id, ProjectEdit(status="done"), reporter=SESSION_A)
    message = str(excinfo.value)
    assert "cannot set status 'done'" in message
    assert "'beta cut'" in message  # the incomplete one is named
    assert "'gamma review'" not in message  # the complete one is not
    assert "force_done=true" in message  # and the escape hatch is in the sentence
    assert store.get_project(project.id).status != "done"

    forced = store.update_project(
        project.id, ProjectEdit(status="done"), reporter=SESSION_A, force_done=True
    )
    assert forced.project.status == "done"
    assert forced.project.completed_at is not None  # the stamp still applies


def test_done_is_allowed_when_the_plan_is_complete_or_empty(store) -> None:
    complete = create(
        store,
        name="complete-plan",
        milestones=[ProjectMilestone(name="one", completed_at="2026-01-01")],
    )
    assert store.update_project(complete.id, ProjectEdit(status="done")).project.status == "done"
    empty = create(store, name="plan-less")
    assert store.update_project(empty.id, ProjectEdit(status="done")).project.status == "done"


def test_a_same_call_replace_to_a_complete_list_passes_the_gate(store) -> None:
    project = create(store, milestones=[ProjectMilestone(name="open")])
    outcome = store.update_project(
        project.id,
        ProjectEdit(
            status="done",
            milestones=[ProjectMilestone(name="closed", completed_at="2026-01-01")],
        ),
        reporter=SESSION_A,
    )
    assert outcome.project.status == "done"


def test_create_refuses_done_with_incomplete_milestones(store) -> None:
    with pytest.raises(ValueError) as excinfo:
        store.create_project(
            ProjectEdit(
                name="born-done", status="done", milestones=[ProjectMilestone(name="open")]
            ),
            sessions=[SESSION_A],
        )
    assert "force_done=true" in str(excinfo.value)
    made = store.create_project(
        ProjectEdit(
            name="born-done-forced",
            status="done",
            milestones=[ProjectMilestone(name="open")],
        ),
        sessions=[SESSION_A],
        force_done=True,
    )
    assert made.status == "done"


def test_a_row_with_an_unknown_status_loads_with_a_warning(store, caplog) -> None:
    """QA round 1, Q1: a row from a NEWER build must LOAD — the word is
    preserved and every surface renders it (the board's leading column, a
    ``[? word]`` chip) — while the WRITE path stays strict."""
    import logging

    project = create(store, name="future-row", status="qa")
    path = store.projects_dir / f"{project.id}.json"
    row = json.loads(path.read_text())
    row["status"] = "shipped"
    path.write_text(json.dumps(row))

    with caplog.at_level(logging.WARNING, logger="local_operator.projects"):
        reader = ProjectRegistry(store.config_dir)
        names = [candidate.name for candidate in reader.list_projects()]
    assert "future-row" in names  # loaded, not dropped
    found = reader.get_project_by_name("future-row")
    assert found is not None and found.status == "shipped"  # preserved verbatim
    assert any("unknown status 'shipped'" in record.message for record in caplog.records)

    # Writes stay strict: the edit vocabulary refuses the word before any lock.
    # (`ProjectEdit`'s status is the Literal, so the refusal is at construction;
    # the kwargs form is the deliberate type violation this test pins.)
    with pytest.raises(ValueError):
        ProjectEdit.model_validate({"status": "shipped"})


# -- schema 2: the role split (P1) -------------------------------------------

# The extra imports for the acceptance block live here (the module's top block
# predates the split); pytest imports the module once, so the placement is
# cosmetic.
from local_operator.projects import (  # noqa: E402
    SESSIONS_MAX,
    stale_after_s,
    refreshed_age_text,
    refreshed_note,
    stale_projects_fingerprint,
    stale_projects_for_session,
)


def _age_row(store: ProjectRegistry, project: Project, age_s: float) -> float:
    """Backdate a row's content clock on disk; returns the stamp written.

    In-place content writes leave the directory mtime alone, and the store's
    refresh is directory-gated — so a reader that must SEE the backdate (the
    production shape: a fresh registry, like a new process) goes through
    :func:`_reloaded` after this call. Writers reload under the store lock and
    see it without help.
    """
    path = store.projects_dir / f"{project.id}.json"
    payload = json.loads(path.read_text())
    payload["progress_updated_at"] = time.time() - age_s
    path.write_text(json.dumps(payload))
    return float(payload["progress_updated_at"])


def _reloaded(store: ProjectRegistry) -> ProjectRegistry:
    return ProjectRegistry(store.config_dir)


def test_coordination_links_are_a_separate_list_with_the_same_caps(store) -> None:
    # The one representation rule: an id sits in at most one list, because a
    # reader of `sessions` must never have to filter a filing out by hand.
    with pytest.raises(ValueError):
        store.create_project(
            ProjectEdit(name="dup"), sessions=[SESSION_A], coordination_sessions=[SESSION_A]
        )
    # The union is capped by the single-list discipline (moves never grow it).
    ids = [f"{index:012x}" for index in range(SESSIONS_MAX + 1)]
    with pytest.raises(ValueError):
        store.create_project(
            ProjectEdit(name="over"), sessions=ids[:40], coordination_sessions=ids[40:]
        )
    with pytest.raises(ValueError):
        store.create_project(ProjectEdit(name="bad-id"), coordination_sessions=["not-hex"])


def test_link_role_moves_between_lists_and_unlink_targets_either(store) -> None:
    project = create(store)  # SESSION_A working
    updated, changed = store.link_session(project.id, SESSION_B, role="coordination")
    assert changed
    assert updated.sessions == [SESSION_A] and updated.coordination_sessions == [SESSION_B]
    # Same role again: a no-op that reports so.
    _, again = store.link_session(project.id, SESSION_B, role="coordination")
    assert not again
    # The other role MOVES the id — the one-op re-kind a wrong migration
    # demotion is repaired with.
    moved, changed = store.link_session(project.id, SESSION_B, role="work")
    assert changed
    assert moved.sessions == [SESSION_A, SESSION_B] and moved.coordination_sessions == []
    # Unlink targets the id across EITHER list.
    cleaned, removed = store.unlink_session(project.id, SESSION_A)
    assert removed and cleaned.sessions == [SESSION_B]
    _, removed = store.unlink_session(project.id, SESSION_B)
    assert removed
    _, removed = store.unlink_session(project.id, "ffffffffffff")
    assert not removed
    # A bad role is refused, never silently treated as work.
    with pytest.raises(ValueError):
        store.link_session(project.id, SESSION_A, role="boss")  # type: ignore[arg-type]


def test_projects_for_session_is_membership_of_either_list(store) -> None:
    store.create_project(ProjectEdit(name="filed"), coordination_sessions=[SESSION_B])
    create(store, name="working")
    assert [p.name for p in store.projects_for_session(SESSION_B)] == ["filed"]
    assert [p.name for p in store.projects_for_session(SESSION_A)] == ["working"]


def test_the_completion_check_never_fires_through_a_coordination_link(store) -> None:
    store.create_project(
        ProjectEdit(name="filed", progress="old"), coordination_sessions=[SESSION_B]
    )
    assert stale_projects_for_session(store, SESSION_B) == []
    create(store, name="worked")  # no progress: stale by construction
    assert [p.name for p in stale_projects_for_session(store, SESSION_A)] == ["worked"]


def test_a_refresh_quiets_the_completion_check_for_one_window(store) -> None:
    project = create(store, progress="still true")
    stamp = _age_row(store, project, PROJECT_PROGRESS_STALE_S + 60)
    store = _reloaded(store)
    moment = stamp + PROJECT_PROGRESS_STALE_S + 120
    assert [p.name for p in stale_projects_for_session(store, SESSION_A, now=moment)] == [
        "payments-migration"
    ]

    outcome = store.refresh_project(project.id, reporter=SESSION_A)
    assert outcome.changed and outcome.refreshed
    refreshed = outcome.project
    checked_at = refreshed.progress_refreshed_at
    assert checked_at is not None
    # Inside the window: quiet. The BADGE, however, still reads stale — two
    # clocks under one name, the operator's "must not clear it".
    assert stale_projects_for_session(store, SESSION_A, now=checked_at + 60) == []
    assert progress_is_stale(refreshed, now=checked_at + 60) is True
    # Past the window the reminder is back (the record is still content-stale).
    assert [
        p.name
        for p in stale_projects_for_session(
            store, SESSION_A, now=checked_at + PROJECT_PROGRESS_STALE_S + 60
        )
    ] == ["payments-migration"]


def test_the_fingerprint_carries_the_assertion_and_moves_on_a_refresh(store) -> None:
    project = create(store, progress="x")
    _age_row(store, project, PROJECT_PROGRESS_STALE_S + 60)
    store = _reloaded(store)
    row = store.get_project(project.id)
    before = stale_projects_fingerprint([row])
    assert before == ((row.id, "active", int(row.progress_updated_at or 0), 0),)
    outcome = store.refresh_project(project.id, reporter=SESSION_A)
    after = stale_projects_fingerprint([outcome.project])
    assert after != before
    assert after[0][3] == int(outcome.project.progress_refreshed_at or 0)


def test_refresh_project_no_ops_without_content_or_on_a_settled_record(store) -> None:
    project = create(store)  # no progress: nothing to assert about
    quiet = store.refresh_project(project.id, reporter=SESSION_A)
    assert not quiet.changed and not quiet.refreshed
    closed = store.create_project(
        ProjectEdit(name="closed", status="done", progress="wrapped"), force_done=True
    )
    settled = store.refresh_project(closed.id, reporter=SESSION_A)
    assert not settled.changed  # settled rows are never stale, so never refreshable


def test_the_view_tags_roles_and_strips_liveness_from_filings(store, tmp_path) -> None:
    project = store.create_project(
        ProjectEdit(name="alpha"), sessions=[SESSION_A], coordination_sessions=[SESSION_B]
    )
    view = build_project_view(store.get_project(project.id), config_dir=tmp_path)
    rows = {row["session_id"]: row for row in view["sessions"]}
    work_row = rows[SESSION_A]
    assert work_row["role"] == "work"
    assert {"runtime", "subagents", "todos"} <= set(work_row)
    filed = rows[SESSION_B]
    assert filed["role"] == "coordination"
    # No liveness fact exists on a filing for a renderer to misread.
    assert set(filed) == {"session_id", "role", "exists", "title", "created_at", "archived"}


def test_a_hand_edited_window_degrades_to_the_default(store, tmp_path) -> None:
    """The registry enforces bounds at WRITE time; a hand-edited config.yml
    reaches no validation, so the resolver repeats the shape check and the
    default is always the honest fallback (ruling §3's one-reader rule)."""
    config = tmp_path / "config.yml"
    config.write_text("values:\n  projects:\n    stale_after_hours: 2\n")
    assert stale_after_s(tmp_path) == 2 * 3600.0
    for bad in ("-5", "0", "bogus", "true"):
        config.write_text(f"values:\n  projects:\n    stale_after_hours: {bad}\n")
        assert stale_after_s(tmp_path) == PROJECT_PROGRESS_STALE_S
    config.unlink()
    assert stale_after_s(tmp_path) == PROJECT_PROGRESS_STALE_S


def test_the_derivations_honour_the_configured_window(store, tmp_path) -> None:
    (tmp_path / "config.yml").write_text("values:\n  projects:\n    stale_after_hours: 1\n")
    project = create(store, progress="now")
    stamp = _age_row(store, project, 3601)
    store = _reloaded(store)
    row = store.get_project(project.id)
    moment = stamp + 3601
    # The same row, two verdicts: the shipped window fresh, the configured
    # window stale — resolved once from the config dir the surface holds.
    assert not progress_is_stale(row, now=moment, window=PROJECT_PROGRESS_STALE_S)
    assert progress_is_stale(row, now=moment, window=stale_after_s(tmp_path))
    assert [p.name for p in stale_projects_for_session(store, SESSION_A, now=moment)] == [
        "payments-migration"
    ]
    view = build_project_view(row, config_dir=tmp_path)
    assert view["progress_stale"] is True


def test_the_refreshed_note_shows_only_while_newer_than_the_content() -> None:
    now = 1_800_000_000.0
    updated = now - 5 * 3600
    checked = now - 2 * 3600
    row = Project(
        id="a" * 12,
        name="alpha",
        status="active",
        progress="a line",
        progress_updated_at=updated,
        progress_refreshed_at=checked,
        progress_refreshed_by=SESSION_A,
    )
    day = datetime.datetime.fromtimestamp(updated).date().isoformat()
    assert refreshed_note(row, now=now) == (
        f"refreshed 2h ago by session {SESSION_A} — no new content since {day}"
    )
    assert refreshed_age_text(row, now=now) == "2h"
    # A superseded assertion (hand-edit) paints nothing, and neither does an
    # assertion about no content at all.
    assert (
        refreshed_note(row.model_copy(update={"progress_refreshed_at": updated - 10}), now=now)
        is None
    )
    assert refreshed_note(row.model_copy(update={"progress_updated_at": None}), now=now) is None
    bare = row.model_copy(update={"progress_refreshed_by": ""})
    assert refreshed_note(bare, now=now) == f"refreshed 2h ago — no new content since {day}"
