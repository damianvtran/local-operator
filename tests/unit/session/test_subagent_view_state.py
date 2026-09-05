"""Child detail ownership across canonical snapshots, deltas and fetch races."""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator.session.frontend_state import (
    JOB_TODOS_WIRE_BYTES,
    FrontendSessionState,
    FrontendStateStore,
    FrontendUpdate,
    JobState,
    SnapshotSubagentComms,
    filter_update_trajectories,
    job_todos_wire_value,
    sync_wire_payload,
)


def _todos(text: str) -> list[dict[str, Any]]:
    return [{"name": "Work", "items": [{"text": text, "status": "pending", "reason": ""}]}]


def _job(todos=None, **changes) -> JobState:  # noqa: ANN001, ANN003
    return JobState(id="manager", type="task", session_id="child-session", todos=todos, **changes)


def _state(*jobs: JobState) -> FrontendSessionState:
    return FrontendSessionState(session_id="root", epoch="epoch", jobs=list(jobs))


def test_todo_delta_is_watched_and_does_not_ride_roster_or_attach() -> None:
    owner = FrontendStateStore(_state(_job()))
    follower = FrontendStateStore(owner.state)
    update = owner.mutate(jobs=[_job(_todos("Child work"))])
    assert update is not None
    payload = update.model_dump(mode="json")
    assert "todos" not in payload["changes"]["jobs"][0]
    assert filter_update_trajectories(payload, lambda _: False)["job_todo_updates"] == {}
    watched = filter_update_trajectories(payload, lambda _: True)
    follower.apply_update(FrontendUpdate.model_validate_json(json.dumps(watched)))
    assert follower.state.jobs[0].todos == _todos("Child work")
    snapshot = sync_wire_payload(owner.subscribe(lambda _: None).sync)
    assert snapshot["snapshot"]["jobs"][0]["todos"] is None
    # A status-only roster delta must preserve the already-hydrated plan.
    later = owner.mutate(jobs=[_job(_todos("Child work"), status="completed")])
    assert later is not None and not later.job_todo_updates
    follower.apply_update(FrontendUpdate.model_validate_json(later.model_dump_json()))
    assert follower.state.jobs[0].todos == _todos("Child work")


def test_fetched_todos_and_stream_replacements_have_two_way_watermarks() -> None:
    owner = FrontendStateStore(_state(_job()))
    follower = FrontendStateStore(owner.state)
    older = owner.mutate(jobs=[_job(_todos("Earlier"))])
    newer = owner.mutate(jobs=[_job(_todos("Newer"))])
    assert older is not None and newer is not None
    assert follower.seed_job_todos(
        "manager",
        _todos("Newer"),
        epoch="epoch",
        sequence=newer.sequence,
        session_id="child-session",
    )
    follower.apply_update(FrontendUpdate.model_validate_json(older.model_dump_json()))
    assert follower.state.jobs[0].todos == _todos("Newer")
    follower.apply_update(FrontendUpdate.model_validate_json(newer.model_dump_json()))
    assert not follower.seed_job_todos(
        "manager",
        _todos("Earlier"),
        epoch="epoch",
        sequence=older.sequence,
        session_id="child-session",
    )
    clear = owner.mutate(jobs=[_job([])])
    assert clear is not None
    follower.apply_update(FrontendUpdate.model_validate_json(clear.model_dump_json()))
    assert follower.state.jobs[0].todos == []
    assert not follower.seed_job_todos(
        "manager",
        _todos("Wrong session"),
        epoch="epoch",
        sequence=100,
        session_id="another-session",
    )
    follower.replace(_state(_job()).model_copy(update={"epoch": "new-owner"}))
    assert not follower.seed_job_todos(
        "manager",
        _todos("Old owner"),
        epoch="epoch",
        sequence=100,
        session_id="child-session",
    )
    assert follower.seed_job_todos(
        "manager",
        [],
        epoch="new-owner",
        sequence=0,
        session_id="child-session",
    )


def test_large_child_plans_cannot_amplify_attach_or_selected_frames() -> None:
    large = _todos("x" * (JOB_TODOS_WIRE_BYTES + 1))
    assert job_todos_wire_value(large) is None  # unavailable, never a partial task
    assert job_todos_wire_value([]) == []
    jobs = [JobState(id=f"child-{i}", type="task", todos=large) for i in range(12)]
    owner = FrontendStateStore(_state())
    update = owner.mutate(jobs=jobs)
    assert update is not None
    snapshot = sync_wire_payload(owner.subscribe(lambda _: None).sync)
    assert len(json.dumps(snapshot).encode()) < 1_048_576
    assert all(job["todos"] is None for job in snapshot["snapshot"]["jobs"])
    selected = filter_update_trajectories(
        update.model_dump(mode="json"), lambda key: key == "child-1"
    )
    assert selected["job_todo_updates"] == {"child-1": None}
    assert len(json.dumps(selected).encode()) < 1_048_576
    assert owner.state.jobs[1].todos == large


def test_resumed_manager_alias_resolves_to_one_current_node() -> None:
    comms = SnapshotSubagentComms(
        [
            _job([], attempt_aliases=["old-manager"]),
            JobState(id="leaf", type="task", parent_job_id="manager", session_id="leaf-session"),
        ]
    )
    node = comms.node("old-manager")
    assert node is not None and node.job_id == "manager"
    assert [node.job_id for node in comms.children("old-manager")] == ["leaf"]
    parent = comms.parent("leaf")
    assert parent is not None and parent.job_id == "manager"


@pytest.mark.parametrize("saved", [[], _todos("Previously saved child task")])
@pytest.mark.asyncio
async def test_cold_owner_hydrates_selected_child_plan_and_reconstructs_nested_rows(
    tmp_path, monkeypatch, saved
) -> None:
    import asyncio

    from local_operator.session.runtime.owned import OwnedSessionHandle
    from local_operator.session.transcript import Transcript
    from local_operator.tools.builtin import TODO_STORE
    from local_operator.tui.widgets.subagent_panel import job_elapsed, job_seconds
    from tests.unit.harness.test_comms import ScriptedProvider, make_parent

    root = make_parent(tmp_path, ScriptedProvider())
    child = Transcript(tmp_path / "cold-manager")
    await child.append_custom("todo_snapshot", {"items": saved})
    leaf = Transcript(tmp_path / "cold-leaf")
    root.subagent_comms.restore(
        [
            {
                "job_id": "manager",
                "label": "Manager",
                "session_dir": str(child.directory),
                "outcome": "completed",
            },
            {
                "job_id": "leaf",
                "label": "Leaf",
                "parent_job_id": "manager",
                "session_dir": str(leaf.directory),
                "outcome": "completed",
            },
        ]
    )
    monkeypatch.delitem(TODO_STORE, "cold-manager", raising=False)
    root._frontend_state_store.refresh_jobs(root)
    manager = next(job for job in root.frontend_state.jobs if job.id == "manager")
    assert manager.todos is None  # absent process store is not an empty saved plan
    assert job_seconds(manager) == 0.0
    assert job_elapsed(manager) == ""  # no invented multi-day age or zero-second receipt
    assert (
        next(job for job in root.frontend_state.jobs if job.id == "leaf").parent_job_id == "manager"
    )
    handle = OwnedSessionHandle(root, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        page = await handle.job_trajectory("manager", 0, 100)
        assert page["known"] is True
        assert page["todos"] == saved
        assert page["detail_epoch"] == root.frontend_state.epoch
        assert page["detail_sequence"] == root.frontend_state.sequence
        assert "cold-manager" in TODO_STORE
        root._frontend_state_store.refresh_jobs(root)
        assert next(job for job in root.frontend_state.jobs if job.id == "manager").todos == saved
    finally:
        TODO_STORE.pop("cold-manager", None)
        await root.dispose()


@pytest.mark.asyncio
async def test_live_empty_child_plan_fetch_never_reads_transcript(tmp_path, monkeypatch) -> None:
    import asyncio

    from local_operator.harness.jobs import AsyncJob
    from local_operator.session.runtime import owned
    from local_operator.tools.builtin import TODO_STORE
    from tests.unit.harness.test_comms import ScriptedProvider, make_parent

    root = make_parent(tmp_path / "root", ScriptedProvider())
    child = make_parent(tmp_path / "child", ScriptedProvider())
    job = AsyncJob(id="live-child", type="task", label="Live child", start_time=1.0)
    root.jobs._jobs[job.id] = job
    root.subagent_comms.record_launch(job.id, job.label)
    root.subagent_comms.attach(job.id, child, child._transcript.directory)
    monkeypatch.delitem(TODO_STORE, child.session_id, raising=False)

    def forbid_read(*args):  # noqa: ANN002, ANN202
        raise AssertionError("a live empty plan must not trigger historical disk I/O")

    monkeypatch.setattr(owned, "_read_child_todo_snapshot", forbid_read)
    handle = owned.OwnedSessionHandle(root, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        page = await handle.job_trajectory(job.id, 0, 100)
        assert page["known"] is True
        assert page["todos"] == []
        assert child.session_id not in TODO_STORE
    finally:
        root.subagent_comms.detach(job.id)
        job.status = "completed"
        await child.dispose()
        await root.dispose()


@pytest.mark.asyncio
async def test_parent_checkpoint_does_not_duplicate_child_plans() -> None:
    saved = []

    class Transcript:
        async def append_custom(self, kind, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
            saved.append(payload)

    store = FrontendStateStore(_state(_job(_todos("Private child plan"))))
    await store.checkpoint(Transcript())
    assert saved
    assert saved[-1]["state"]["jobs"][0]["todos"] is None
    assert store.state.jobs[0].todos == _todos("Private child plan")
