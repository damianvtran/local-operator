"""A follower's ``jobs`` delta rebuilds only the children that moved.

WHY. A loaded parent (12 stepping lanes plus 240 settled children) sends the
WHOLE roster on every roster tick -- measured at 252 rows / ~410 KB per delta
with 1-12 rows actually changed -- and the viewer rebuilt every row each time
(deep copy, pydantic validation, freeze): ~11 ms p50 per delta on the TUI's own
loop, and 252 fresh objects that told every downstream reader "everything
moved". ``FrontendStateStore._reusable_job`` keeps the canonical row of every
child whose wire body is unchanged; these tests pin the proof it rests on.

Structural assertions only (object identity, call counts, revision tokens), per
AGENTS.md "Timing, flakes": no wall-clock bound appears here.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from local_operator.session import frontend_state as module
from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
    FrontendUpdate,
    JobState,
    SnapshotJobs,
    SnapshotSubagentComms,
)


def _row(job_id: str, **fields: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": job_id,
        "type": "task",
        "status": "completed",
        "label": f"child {job_id}",
        "result_text": "ok " * 20,
        "session_dir": f"/nowhere/{job_id}",
    }
    body.update(fields)
    return body


def _store(count: int = 4) -> FrontendStateStore:
    store = FrontendStateStore(FrontendSessionState(session_id="parent", epoch="owner"))
    store.apply_update(_delta(store, jobs=[_row(f"c{i}") for i in range(count)]))
    return store


def _delta(store: FrontendStateStore, **changes: Any) -> FrontendUpdate:
    extra = {
        key: changes.pop(key)
        for key in ("job_trajectory_appends", "job_trajectory_replacements", "job_todo_updates")
        if key in changes
    }
    return FrontendUpdate(
        epoch="owner", sequence=store._state.sequence + 1, changes=changes, **extra
    )


def _rows(store: FrontendStateStore) -> dict[str, JobState]:
    # CANONICAL rows: ``state`` detaches a fresh copy per read, so identity
    # across two ``state`` reads is vacuous (see test_frontend_row_window.py).
    return {job.id: job for job in store._state.jobs}


def test_an_unchanged_row_is_kept_and_only_the_moved_row_is_rebuilt() -> None:
    store = _store()
    before = _rows(store)
    roster = [_row(f"c{i}") for i in range(4)]
    roster[2] = _row("c2", status="running", latest_details={"progress": "reading"})
    with patch.object(module, "_freeze_job", wraps=module._freeze_job) as freeze:
        store.apply_update(_delta(store, jobs=roster))
    after = _rows(store)
    assert freeze.call_count == 1, "only the one moved row may be rebuilt"
    for job_id in ("c0", "c1", "c3"):
        assert after[job_id] is before[job_id], f"{job_id} did not move and was rebuilt"
    assert after["c2"] is not before["c2"]
    assert after["c2"].status == "running"
    assert after["c2"].latest_details == {"progress": "reading"}
    # Order follows the wire, reused and rebuilt rows alike.
    assert [job.id for job in store._state.jobs] == ["c0", "c1", "c2", "c3"]


def test_a_trajectory_append_rebuilds_its_row_even_with_an_equal_body() -> None:
    """The window is merged from OUTSIDE the body, so an equal body proves nothing."""
    store = _store(2)
    before = _rows(store)
    store.apply_update(
        _delta(
            store,
            jobs=[_row("c0"), _row("c1")],
            job_trajectory_appends={"c1": [{"type": "message_update", "delta": "new"}]},
        )
    )
    after = _rows(store)
    assert after["c0"] is before["c0"]
    assert after["c1"] is not before["c1"]
    assert [row["delta"] for row in after["c1"].trajectory] == ["new"]


def test_a_trajectory_replacement_rebuilds_its_row() -> None:
    store = _store(1)
    store.apply_update(
        _delta(
            store,
            jobs=[_row("c0")],
            job_trajectory_appends={"c0": [{"type": "message_update", "delta": "a"}]},
        )
    )
    before = _rows(store)["c0"]
    store.apply_update(
        _delta(
            store,
            jobs=[_row("c0")],
            job_trajectory_replacements=["c0"],
            job_trajectory_appends={"c0": [{"type": "message_update", "delta": "b"}]},
        )
    )
    after = _rows(store)["c0"]
    assert after is not before
    assert [row["delta"] for row in after.trajectory] == ["b"]


def test_a_todo_update_rebuilds_its_row() -> None:
    store = _store(1)
    before = _rows(store)["c0"]
    plan = [{"name": "Build", "items": [{"text": "x", "status": "pending"}]}]
    store.apply_update(_delta(store, jobs=[_row("c0")], job_todo_updates={"c0": plan}))
    after = _rows(store)["c0"]
    assert after is not before
    assert after.todos is not None and after.todos[0]["name"] == "Build"
    # And a later delta with no todo update keeps the plan it now carries.
    store.apply_update(_delta(store, jobs=[_row("c0")]))
    assert _rows(store)["c0"] is after


def test_a_row_replaced_by_a_local_seed_is_not_vouched_for_by_its_old_body() -> None:
    """``seed_job_trajectory`` swaps the row without a delta; the body record is
    paired with the OLD row by identity, so it must stop vouching."""
    store = _store(1)
    store.seed_job_trajectory("c0", [{"type": "message_update", "delta": "seeded"}])
    seeded = _rows(store)["c0"]
    store.apply_update(_delta(store, jobs=[_row("c0")]))
    after = _rows(store)["c0"]
    assert after is not seeded, "a stale body record vouched for a row it never built"
    # The rebuild still extends the seeded window, exactly as before this change.
    assert [row["delta"] for row in after.trajectory] == ["seeded"]


def test_a_snapshot_install_drops_every_body_record() -> None:
    store = _store(2)
    store.replace(
        FrontendSessionState(
            session_id="parent",
            epoch="owner",
            sequence=store._state.sequence,
            jobs=[JobState(**_row("c0")), JobState(**_row("c1"))],
        )
    )
    installed = _rows(store)
    store.apply_update(_delta(store, jobs=[_row("c0"), _row("c1")]))
    after = _rows(store)
    assert after["c0"] is not installed["c0"]
    assert after["c1"] is not installed["c1"]


def test_a_degraded_delta_drops_every_body_record() -> None:
    store = _store(1)
    before = _rows(store)["c0"]
    store.apply_update(
        FrontendUpdate(
            epoch="owner",
            sequence=store._state.sequence + 1,
            changes={},
            degraded=True,
            degraded_reason="line limit",
        )
    )
    store.apply_update(_delta(store, jobs=[_row("c0")]))
    assert _rows(store)["c0"] is not before


def test_an_invalid_delta_installs_nothing_and_keeps_the_records() -> None:
    store = _store(2)
    before = _rows(store)
    sequence = store._state.sequence
    bad = _row("c1", status={"not": "a string"})
    try:
        store.apply_update(_delta(store, jobs=[_row("c0"), bad]))
    except Exception:  # noqa: BLE001 -- the refusal itself is the pre-existing contract
        pass
    assert store._state.sequence == sequence
    assert _rows(store) == before
    store.apply_update(_delta(store, jobs=[_row("c0"), _row("c1")]))
    after = _rows(store)
    assert after["c0"] is before["c0"] and after["c1"] is before["c1"]


def test_an_explicit_empty_roster_still_clears() -> None:
    store = _store(2)
    store.apply_update(_delta(store, jobs=[]))
    assert list(store._state.jobs) == []
    assert store._follower_job_bodies == {}


def test_the_revision_moves_only_when_the_band_can_see_a_change() -> None:
    store = _store(3)
    first = store.revision()
    store.apply_update(_delta(store, activity_phase="responding"))
    assert store.revision() == first, "a scalar delta cannot change a roster row"
    store.apply_update(_delta(store, jobs=[_row(f"c{i}") for i in range(3)]))
    assert store.revision() == first, "a repeated roster is not a change"
    roster = [_row(f"c{i}") for i in range(3)]
    roster[0] = _row("c0", result_text="different")
    store.apply_update(_delta(store, jobs=roster))
    moved = store.revision()
    assert moved.jobs != first.jobs
    assert moved.lifecycle == first.lifecycle, "text churn is not a start/settle"
    roster[1] = _row("c1", status="failed")
    store.apply_update(_delta(store, jobs=roster))
    settled = store.revision()
    assert settled.lifecycle != moved.lifecycle, "a status change is lifecycle"
    store.apply_update(_delta(store, jobs=[*roster, _row("c9")]))
    assert store.revision().lifecycle != settled.lifecycle, "an arrival is lifecycle"
    plan = [{"name": "p", "items": [{"text": "t", "status": "pending"}]}]
    before_todos = store.revision()
    store.apply_update(_delta(store, todos=plan))
    assert store.revision().todos != before_todos.todos


def test_clone_free_readers_match_the_cloned_state() -> None:
    store = _store(0)
    store.apply_update(
        _delta(
            store,
            jobs=[
                _row("run", status="running"),
                _row("queued", status="running", queued=True),
                _row("done"),
                _row("bash", type="bash", status="running"),
            ],
            attention={"completion_token": "t1", "anchor_id": "a"},
        )
    )
    cloned = store.state
    assert store.running_task_count() == sum(
        1 for job in cloned.jobs if job.type == "task" and job.status == "running" and not job.queued
    )
    assert store.running_task_count() == 1
    copy = store.attention_copy()
    assert copy == dict(cloned.attention)
    copy["completion_token"] = "mutated by a caller"
    assert store._state.attention["completion_token"] == "t1"


def test_the_jobs_facade_redetaches_only_rows_whose_canonical_object_moved() -> None:
    store = _store(3)
    jobs = SnapshotJobs(store._state.jobs)
    roster = [_row(f"c{i}") for i in range(3)]
    roster[1] = _row("c1", result_text="moved")
    store.apply_update(_delta(store, jobs=roster))
    with patch.object(module, "_public_job", wraps=module._public_job) as detach:
        jobs.replace(store._state.jobs)
    assert detach.call_count == 1
    # Callers still get their own detached copy per read, and the moved row's
    # new content.
    first, second = jobs.get("c0"), jobs.get("c0")
    assert first is not None and second is not None and first is not second
    moved = jobs.get("c1")
    assert moved is not None and moved.result_text == "moved"


def test_the_comms_facade_reuses_a_node_only_when_it_reads_no_disk() -> None:
    wired = JobState(**_row("wired"))
    derived = JobState(id="derived", type="task", session_id="abcabcabcabc", label="d")
    comms = SnapshotSubagentComms([wired, derived])
    wired_node, derived_node = comms.node("wired"), comms.node("derived")
    with patch.object(
        SnapshotSubagentComms, "_node_for", wraps=SnapshotSubagentComms._node_for
    ) as build:
        comms.replace([wired, derived])
    assert comms.node("wired") is wired_node
    # The derived row's directory is proven against an origin marker a starting
    # child may not have written yet; it is rebuilt on every frame, as before.
    assert comms.node("derived") is not derived_node
    assert build.call_count == 1
