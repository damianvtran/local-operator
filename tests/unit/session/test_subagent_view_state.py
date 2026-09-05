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


def test_follower_comms_answers_session_dir_from_wire_then_derives_it(
    tmp_path, monkeypatch
) -> None:
    """The follower's history seam: the full-page view pages durable history
    through ``comms.session_dir_of``, and a ``None`` there is treated as a
    PERMANENT absence by the view, so every answer below is load-bearing."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # The derived answer is only trusted once the directory proves it is this
    # child's session (see ``_derived_dir_belongs_to``), so stamp the marker
    # ``run_subagent`` writes at creation. Written BEFORE the facade is built:
    # ``replace()`` resolves every node's directory eagerly.
    derived_dir = tmp_path / "sessions" / "def456"
    derived_dir.mkdir(parents=True)
    (derived_dir / "origin.json").write_text(
        json.dumps({"origin": "subagent", "label": "worker", "agent": "coder"}),
        encoding="utf-8",
    )
    comms = SnapshotSubagentComms(
        [
            # A current owner stamps the directory on the wire; it wins even
            # when it disagrees with where the session_id would derive to.
            JobState(
                id="wired",
                type="task",
                session_id="abc123",
                session_dir=str(tmp_path / "elsewhere" / "abc123"),
            ),
            # A pre-fix owner sends only the session_id: derive the path the
            # child was created at so the operator's already-running daemon
            # gains history without a restart.
            JobState(
                id="derived",
                type="task",
                session_id="def456",
                label="worker",
                agent_role="coder",
            ),
            # A job with neither (a bash job, a swept child) has nothing to read.
            JobState(id="bare", type="task"),
        ]
    )
    assert comms.session_dir_of("wired") == tmp_path / "elsewhere" / "abc123"
    assert comms.session_dir_of("derived") == derived_dir
    assert comms.session_dir_of("bare") is None
    assert comms.session_dir_of("unknown") is None
    # The node carries the same answer, which is what the todo panel reads.
    node = comms.node("derived")
    assert node is not None and node.session_dir == derived_dir


def test_a_derived_path_is_refused_unless_the_directory_proves_ownership(
    tmp_path, monkeypatch
) -> None:
    """The guess must prove itself; the wire value never has to.

    ``session_id`` is a 48-bit truncated uuid and carries no ownership proof,
    and the derivation reads the PROCESS-GLOBAL ``config_dir()`` rather than
    whichever root a follower attached against — so an unrelated local session
    is directly reachable, no collision required (review round 1, M2; QA hit
    the config-root half of this with a "wrong config root" cell that resolved
    anyway).

    Every case below degrades to ``None``, which the view already renders as
    the "no saved transcript" note — the behaviour followers had before the
    derivation existed. Failing closed can only withhold history; failing open
    renders somebody else's conversation under this child's name.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    sessions = tmp_path / "sessions"

    # 1. Nothing on disk at all: a remote/mobile follower whose children live
    #    on another machine. Must NOT be trusted, and must not create anything.
    absent = JobState(id="absent", type="task", session_id="aaaaaaaaaaaa", label="w")
    assert SnapshotSubagentComms([absent]).session_dir_of("absent") is None
    assert not (sessions / "aaaaaaaaaaaa").exists(), "resolution must not touch disk"

    # 2. A real local session that is the USER'S OWN, not anyone's child. It
    #    has no origin marker, which is precisely how a user session looks.
    user_dir = sessions / "bbbbbbbbbbbb"
    user_dir.mkdir(parents=True)
    (user_dir / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    stranger = JobState(id="stranger", type="task", session_id="bbbbbbbbbbbb", label="w")
    assert SnapshotSubagentComms([stranger]).session_dir_of("stranger") is None

    # 3. A subagent session, but a DIFFERENT child's: the id matched, the
    #    identity does not.
    other_dir = sessions / "cccccccccccc"
    other_dir.mkdir(parents=True)
    (other_dir / "origin.json").write_text(
        json.dumps({"origin": "subagent", "label": "someone-else", "agent": "reviewer"}),
        encoding="utf-8",
    )
    mismatched = JobState(
        id="mismatched", type="task", session_id="cccccccccccc", label="w", agent_role="coder"
    )
    assert SnapshotSubagentComms([mismatched]).session_dir_of("mismatched") is None

    # 4. The real thing: same id, and the marker agrees about who it is.
    mine_dir = sessions / "dddddddddddd"
    mine_dir.mkdir(parents=True)
    (mine_dir / "origin.json").write_text(
        json.dumps({"origin": "subagent", "label": "w", "agent": "coder"}),
        encoding="utf-8",
    )
    mine = JobState(
        id="mine", type="task", session_id="dddddddddddd", label="w", agent_role="coder"
    )
    assert SnapshotSubagentComms([mine]).session_dir_of("mine") == mine_dir

    # 5. A WIRE-supplied directory is the owner's own answer and is trusted
    #    without a marker — the owner knows where it put the child.
    bare = sessions / "eeeeeeeeeeee"
    wired = JobState(
        id="wired", type="task", session_id="ffffffffffff", session_dir=str(bare), label="w"
    )
    assert SnapshotSubagentComms([wired]).session_dir_of("wired") == bare


def test_a_missing_marker_is_not_memoised_as_a_verdict(tmp_path, monkeypatch) -> None:
    """The ownership verdict is cached, so a child that is still STARTING
    must not be pinned as unusable.

    ``session_dir_of`` rides the page's 1 Hz refresh and QA measured this path
    at 1.08 stats/s with 0 page reads, so the marker read is memoised. The
    memo may only hold facts about CONTENT: a directory whose marker is not
    there yet (or briefly unreadable) has to be re-asked, exactly as
    ``resume._session_origin_read`` declines to cache an ``OSError``.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    job = JobState(
        id="starting", type="task", session_id="abcabcabcabc", label="w", agent_role="coder"
    )
    comms = SnapshotSubagentComms([job])
    assert comms.session_dir_of("starting") is None

    directory = tmp_path / "sessions" / "abcabcabcabc"
    directory.mkdir(parents=True)
    (directory / "origin.json").write_text(
        json.dumps({"origin": "subagent", "label": "w", "agent": "coder"}),
        encoding="utf-8",
    )
    # A later frame re-projects the node; the answer must now be the directory.
    comms.replace([job])
    assert comms.session_dir_of("starting") == directory


def test_lineage_stamps_the_child_session_dir_as_a_string() -> None:
    """``_with_lineage`` is where an owner projects its registry onto the
    wire; ``Path`` is not JSON-native so the projection is a string, and a
    node without a directory projects ``None`` rather than ``"None"``."""
    from pathlib import Path
    from types import SimpleNamespace

    from local_operator.session.frontend_state import _with_lineage

    def comms_with(session_dir):  # noqa: ANN001, ANN202
        node = SimpleNamespace(
            job_id="child",
            parent_job_id="root",
            session_id="abc123",
            session_dir=session_dir,
            attempt_aliases=(),
            live=False,
        )
        return SimpleNamespace(node=lambda _job_id: node)

    job = JobState(id="child", type="task")
    stamped = _with_lineage(job, comms_with(Path("/tmp/sessions/abc123")))
    assert stamped.session_dir == "/tmp/sessions/abc123"
    assert stamped.session_id == "abc123"
    # Survives a JSON round trip, which is how a follower receives it.
    wire = JobState.model_validate_json(stamped.model_dump_json())
    assert wire.session_dir == "/tmp/sessions/abc123"
    assert _with_lineage(job, comms_with(None)).session_dir is None
    # An owner from before the field existed sends no key at all; the model
    # must default it rather than reject the frame.
    old = JobState.model_validate({"id": "child", "type": "task", "session_id": "abc123"})
    assert old.session_dir is None


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
