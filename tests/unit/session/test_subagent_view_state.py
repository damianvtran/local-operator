"""Child detail ownership across canonical snapshots, deltas and fetch races."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.comms import SubagentComms
from local_operator.session.frontend_state import (
    JOB_LAUNCH_PROMPT_WIRE_CHARS,
    JOB_LAUNCH_PROMPTS_FRAME_BUDGET_CHARS,
    JOB_LAUNCH_PROMPTS_MAX,
    LAUNCH_PROMPT_ELIDED_PLACEHOLDER,
    JOB_TODOS_WIRE_BYTES,
    FrontendSessionState,
    FrontendStateStore,
    FrontendUpdate,
    JobState,
    SnapshotJobs,
    SnapshotSubagentComms,
    _with_lineage,
    filter_update_trajectories,
    job_todos_wire_value,
    sync_wire_payload,
)
from local_operator.session.transcript import TRANSCRIPT_FILENAME
from local_operator.tui.widgets.subagent_view import strip_control_sequences
from tests.unit.harness.test_comms import FakeChild, FakeJobs, FakeParent


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

    # The PARTIAL state, which a plain absent->complete step walks straight
    # past: ``mark_session_origin`` writes with ``write_text``, which truncates
    # and then writes, so a reader landing between the two sees an empty or
    # half-written file. That window is reachable for a child between
    # ``claim_session`` and its stamp, and caching the parse failure pinned
    # such a child as unusable for the life of the process (round 2, R2-2).
    (directory / "origin.json").write_text("", encoding="utf-8")
    comms.replace([job])
    assert comms.session_dir_of("starting") is None

    # Half-written JSON is the same class of transient, not a decided answer.
    (directory / "origin.json").write_text('{"origin": "suba', encoding="utf-8")
    comms.replace([job])
    assert comms.session_dir_of("starting") is None

    (directory / "origin.json").write_text(
        json.dumps({"origin": "subagent", "label": "w", "agent": "coder"}),
        encoding="utf-8",
    )
    # A later frame re-projects the node; the answer must now be the directory.
    comms.replace([job])
    assert comms.session_dir_of("starting") == directory


def test_a_marker_that_cannot_name_its_child_authorises_nobody(tmp_path, monkeypatch) -> None:
    """Absent identity in the marker means NOT PROVEN, never proven.

    ``resume.backfill_session_origins`` stamps
    ``{"origin": "subagent", "backfilled": true}`` — no label, no agent — over
    the operator's existing store at startup (``cli.py`` calls it on boot), so
    these markers are real rather than hypothetical. Under a "match only the
    fields present" rule such a marker satisfied every node, which handed any
    child a directory it had not earned (round 2, R2-1).

    Refusing genuinely backfilled old sessions is the deliberate trade: a
    marker that cannot say WHICH child it belongs to cannot answer the only
    question being asked, and those sessions degrade to the existing
    "no saved transcript" note rather than showing the wrong conversation.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    sessions = tmp_path / "sessions"

    def resolve(session_id: str, marker: dict[str, Any], label: str, agent_role: str):
        directory = sessions / session_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "origin.json").write_text(json.dumps(marker), encoding="utf-8")
        job = JobState(
            id=f"j-{session_id}",
            type="task",
            session_id=session_id,
            label=label,
            agent_role=agent_role,
        )
        return SnapshotSubagentComms([job]).session_dir_of(f"j-{session_id}")

    backfilled = {"origin": "subagent", "backfilled": True}
    assert resolve("aaaaaaaaaaaa", backfilled, "any-child", "reviewer") is None

    # Label-only: the agent mismatch must still be fatal.
    assert resolve("bbbbbbbbbbbb", {"origin": "subagent", "label": "w"}, "w", "coder") is None
    # Agent-only, mirrored.
    assert resolve("cccccccccccc", {"origin": "subagent", "agent": "coder"}, "w", "coder") is None

    # A node carrying no identity of its own cannot be proven to own anything.
    full = {"origin": "subagent", "label": "w", "agent": "coder"}
    assert resolve("dddddddddddd", full, "", "") is None
    assert resolve("eeeeeeeeeeee", full, "w", "") is None

    # The genuine article still resolves.
    assert resolve("ffffffffffff", full, "w", "coder") == sessions / "ffffffffffff"


def test_the_ownership_memo_is_bounded(tmp_path, monkeypatch) -> None:
    """The memo outlives any single roster, so it needs its own cap.

    ``SubagentComms._records`` is capped at ``MAX_RECORDS``, but that bounds
    the roster at an INSTANT: eviction there does not clear entries here, so
    the key space is every distinct (directory, label, agent) triple the
    process has ever seen. Bounded first-seen-first-out; re-deciding an
    evicted entry costs one stat.
    """
    from local_operator.session import frontend_state as module

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(module, "_DERIVED_OWNERSHIP", {})
    monkeypatch.setattr(module, "_DERIVED_OWNERSHIP_MAX", 8)
    sessions = tmp_path / "sessions"
    for index in range(20):
        session_id = f"s{index:011d}"
        directory = sessions / session_id
        directory.mkdir(parents=True)
        (directory / "origin.json").write_text(
            json.dumps({"origin": "subagent", "label": f"L{index}", "agent": "coder"}),
            encoding="utf-8",
        )
        job = JobState(
            id=f"j{index}",
            type="task",
            session_id=session_id,
            label=f"L{index}",
            agent_role="coder",
        )
        assert SnapshotSubagentComms([job]).session_dir_of(f"j{index}") == directory
    assert len(module._DERIVED_OWNERSHIP) <= 8


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


def _launched_comms(tmp_path: Path, *, resumed: bool = False) -> tuple[SubagentComms, FakeJobs]:
    """A real owner-side comms holding one launched child, optionally resumed.

    The production registry rather than a stand-in node: the fields under test
    are produced by ``record_launch``/the #314 attempt fold, so a hand-rolled
    namespace would pin this test to the shape it invented instead of to the
    shape the owner actually stamps (a specific review criterion on #669).
    """
    jobs = FakeJobs()
    comms = SubagentComms(FakeParent(jobs))  # type: ignore[arg-type]
    session_dir = tmp_path / "child"
    session_dir.mkdir(exist_ok=True)
    (session_dir / TRANSCRIPT_FILENAME).write_text("{}\n", encoding="utf-8")
    jobs.add("attempt-1", status="running")
    comms.record_launch(
        "attempt-1",
        "reviewer",
        prompt="Original task.",
        effective_prompt="[role: reviewer]\nSYSTEM PREAMBLE\nOriginal task.",
        launch_message_id="subagent-launch:attempt-1",
        agent_role="reviewer",
    )
    comms.attach("attempt-1", FakeChild(), session_dir)  # type: ignore[arg-type]
    if not resumed:
        return comms, jobs
    comms.record_outcome("attempt-1", "cancelled")
    comms.detach("attempt-1")
    jobs.jobs["attempt-1"].status = "cancelled"
    jobs.add("attempt-2", status="running")
    comms.record_launch(
        "attempt-2",
        "reviewer",
        prompt="Wrap up.",
        effective_prompt="[role: reviewer]\nSYSTEM PREAMBLE\nWrap up.",
        launch_message_id="subagent-launch:attempt-2",
        agent_role="reviewer",
    )
    comms.attach("attempt-2", FakeChild(), session_dir)  # type: ignore[arg-type]
    return comms, jobs


def test_launch_identity_rides_the_wire_and_rebuilds_the_follower_node(tmp_path) -> None:
    """The fold inputs reach a follower on BOTH frame kinds.

    Without them the follower cannot correlate the durable launch turn with its
    synthetic prompt head and renders the delegated brief twice — the full
    role/team/system preamble, not a short line. Inert until #669 gave followers
    durable history at all, which is why it is fixed here (#669 Q1 / round-1
    item 6).
    """
    comms, _jobs = _launched_comms(tmp_path, resumed=True)
    job = _with_lineage(JobState(id="attempt-2", type="task", prompt="Wrap up."), comms)
    assert job.launch_message_id == "subagent-launch:attempt-2"
    # Every collapsed attempt, not just the newest: reconciling only the current
    # launch leaks the earlier attempt's preamble as a plain user row.
    assert job.launch_prompts == {
        "subagent-launch:attempt-1": "Original task.",
        "subagent-launch:attempt-2": "Wrap up.",
    }

    owner = FrontendStateStore(_state(job))
    snapshot = sync_wire_payload(owner.subscribe(lambda _: None).sync)
    wire_job = snapshot["snapshot"]["jobs"][0]
    # The identity is DERIVABLE from the job id, so it is flagged rather than
    # spelled out — 46.7 B a row was the round-1 blocker. The follower rebuilds
    # the identical string below.
    assert "launch_message_id" not in wire_job
    # No marker rides in its place: the job TYPE already says a task row can
    # have a launch turn, so the follower re-derives without being told.
    assert "launch_id_derived" not in wire_job
    assert wire_job["launch_prompts"] == job.launch_prompts

    # DELTA rebuild, not only the attach snapshot: job rows are re-serialized
    # per update through ``_job_summary``, which never sees ``sync_wire_payload``.
    update = owner.mutate(jobs=[job.model_copy(update={"status": "completed"})])
    assert update is not None
    delta_job = update.model_dump(mode="json")["changes"]["jobs"][0]
    assert "launch_message_id" not in delta_job
    assert "launch_id_derived" not in delta_job
    assert delta_job["launch_prompts"] == job.launch_prompts

    # The production follower facade, fed the wire snapshot as a follower is.
    follower = FrontendStateStore(FrontendSessionState.model_validate(snapshot["snapshot"]))
    rows = list(follower.state.jobs)
    node = SnapshotSubagentComms(rows).node("attempt-2")
    assert node is not None
    assert node.launch_message_id == "subagent-launch:attempt-2"
    assert node.launch_prompts == job.launch_prompts
    # A plain dict the view may index and iterate, never canonical state.
    assert isinstance(node.launch_prompts, dict)
    # The ROW itself carries the flag rather than the string; the app reads the
    # identity off the job first and falls back to the node
    # (``_refresh_subagent_view``), so the node answer above is the one the view
    # actually consumes on a follower.
    follower_job = SnapshotJobs(rows).get("attempt-2")
    assert follower_job is not None
    assert follower_job.launch_message_id == ""


def test_a_never_resumed_child_carries_its_launch_row_without_a_comms_record(tmp_path) -> None:
    """A swept comms record still leaves the job row's own launch identity.

    ``_with_lineage`` stamps nothing when the node is gone, and the view derives
    ``{launch_message_id: prompt}`` itself, so the current launch still folds.
    """
    job = JobState.from_job(
        SimpleNamespace(
            id="j1",
            type="task",
            label="reviewer",
            prompt="Do the thing.",
            launch_message_id="subagent-launch:j1",
        )
    )
    assert job.launch_message_id == "subagent-launch:j1"
    node = SnapshotSubagentComms([job]).node("j1")
    assert node is not None and node.launch_message_id == "subagent-launch:j1"


def test_an_older_owner_without_launch_fields_degrades_rather_than_crashing() -> None:
    """``extra='allow'``: a pre-fix owner simply sends neither field.

    The node still builds. Its identity is DERIVED for a task row, which is
    correct rather than a guess: a v0.49.2 owner minted the launch id with the
    same deterministic ``subagent-launch:<job_id>`` rule, so the derived value
    is exactly what its durable transcript carries — such a follower gains
    reconciliation instead of merely not crashing. A wrong-shaped id could only
    match no durable row, which is the pre-#681 behaviour anyway.

    A BASH row has no launch turn on any version and must stay empty: the type
    is what distinguishes "elided because derivable" from "never had one".
    """
    old = JobState.model_validate({"id": "j1", "type": "task", "label": "reviewer"})
    node = SnapshotSubagentComms([old]).node("j1")
    assert node is not None
    assert node.launch_message_id == "subagent-launch:j1"
    assert node.launch_prompts == {}

    bash = JobState.model_validate({"id": "b1", "type": "bash", "label": "bash: ls"})
    bash_node = SnapshotSubagentComms([bash]).node("b1")
    assert bash_node is not None
    assert bash_node.launch_message_id == ""


def test_launch_prompts_cannot_grow_the_frame_with_resume_depth(tmp_path) -> None:
    """The fourth unbounded-field instance, closed before it ships.

    ``launch_prompts`` is free text keyed by attempt, so a deeply resumed roster
    would push the attach frame past the socket's line limit exactly as
    trajectories, receipts and job text each did.
    """
    huge = {f"subagent-launch:a{index}": "X" * 40_000 for index in range(40)}
    node = SimpleNamespace(
        job_id="j",
        label="reviewer",
        parent_job_id=None,
        session_id=None,
        session_dir=None,
        prompt="p",
        launch_message_id="subagent-launch:a39",
        agent_role="reviewer",
        effort="",
        launch_prompts=huge,
        attempt_aliases=(),
        live=True,
        status="running",
    )
    bounded = _with_lineage(
        JobState(id="j", type="task"), SimpleNamespace(node=lambda _job_id: node)
    ).launch_prompts
    assert len(bounded) == JOB_LAUNCH_PROMPTS_MAX
    assert all(len(text) <= JOB_LAUNCH_PROMPT_WIRE_CHARS + 1 for text in bounded.values())
    # The NEWEST attempts survive: durable history pages from the tail.
    assert "subagent-launch:a39" in bounded
    assert "subagent-launch:a0" not in bounded
    # A clipped prompt is marked, so a reader can tell it from a short brief.
    assert all(text.endswith("…") for text in bounded.values())

    jobs = [
        JobState(id=f"child-{index}", type="task", launch_prompts=bounded) for index in range(200)
    ]
    owner = FrontendStateStore(_state(*jobs))
    snapshot = sync_wire_payload(owner.subscribe(lambda _: None).sync)
    assert len(json.dumps(snapshot).encode()) < 1_048_576
    update = owner.mutate(jobs=[job.model_copy(update={"status": "completed"}) for job in jobs])
    assert update is not None
    assert len(json.dumps(update.model_dump(mode="json")).encode()) < 1_048_576


def test_launch_prompt_text_is_shared_across_the_roster_not_per_row() -> None:
    """A per-row cap alone leaves the FRAME growing with roster depth.

    Measured before the shared budget existed: 200 resumed rows at the per-row
    cap added 392 KB to a 1 MiB frame and 1,000 rows added 1.9 MB, which is the
    same "bounded per row, unbounded in total" defect
    ``JOB_TEXT_FRAME_BUDGET_CHARS`` exists to close one level up.

    Rows past the budget keep their launch IDENTITY and lose only the prior
    attempts' text, so the duplicate-preamble fix still holds at any depth.
    """
    prompts = {
        f"subagent-launch:a{index}": "X" * JOB_LAUNCH_PROMPT_WIRE_CHARS
        for index in range(JOB_LAUNCH_PROMPTS_MAX)
    }
    jobs = [
        JobState(
            id=f"child-{index}",
            type="task",
            launch_message_id=f"subagent-launch:child-{index}",
            launch_prompts=prompts,
        )
        for index in range(1_000)
    ]
    owner = FrontendStateStore(_state(*jobs))
    served = [
        job
        for job in sync_wire_payload(owner.subscribe(lambda _: None).sync)["snapshot"]["jobs"]
        if job.get("launch_prompts")
    ]
    assert served, "the budget starved every row; the fix would not hold anywhere"
    spent = sum(
        len(key) + len(value) for job in served for key, value in job["launch_prompts"].items()
    )
    assert spent <= JOB_LAUNCH_PROMPTS_FRAME_BUDGET_CHARS
    assert len(served) < len(jobs), "the fixture no longer exceeds the shared budget"

    snapshot = sync_wire_payload(owner.subscribe(lambda _: None).sync)["snapshot"]["jobs"]
    # Identity is what suppresses the duplicate, and EVERY row keeps it — as a
    # derivation flag here, since these ids all derive from their job ids.
    assert all("launch_message_id" not in job for job in snapshot)
    follower_rows = FrontendStateStore(
        FrontendSessionState.model_validate(
            {"session_id": "root", "epoch": "epoch", "jobs": snapshot}
        )
    ).state.jobs
    comms = SnapshotSubagentComms(list(follower_rows))
    follower_nodes = [comms.node(f"child-{index}") for index in range(len(snapshot))]
    assert all(node is not None for node in follower_nodes)
    assert [node.launch_message_id for node in follower_nodes if node is not None] == [
        f"subagent-launch:child-{index}" for index in range(len(snapshot))
    ]

    # The delta stream is serialized by its own route and is bounded too.
    fresh = FrontendStateStore(_state())
    fresh.subscribe(lambda _: None)
    update = fresh.mutate(jobs=jobs)
    assert update is not None
    delta_rows = update.model_dump(mode="json")["changes"]["jobs"]
    delta_spent = sum(
        len(key) + len(value)
        for job in delta_rows
        for key, value in (job.get("launch_prompts") or {}).items()
    )
    assert delta_spent <= JOB_LAUNCH_PROMPTS_FRAME_BUDGET_CHARS


def test_a_row_with_no_launch_facts_spends_no_wire_bytes_on_them() -> None:
    """Empty values are omitted, not sent: an absent fact buys nothing.

    At roster scale the empty keys alone measured ~9 KB of the 1 MiB budget and
    pushed the ``ran all year`` class guard over the limit with no information
    in them. Omission is equivalent to sending the defaults — a delta
    revalidates each raw row rather than merging onto the prior one.
    """
    owner = FrontendStateStore(_state(JobState(id="bash-1", type="bash")))
    row = sync_wire_payload(owner.subscribe(lambda _: None).sync)["snapshot"]["jobs"][0]
    assert "launch_message_id" not in row and "launch_prompts" not in row
    update = owner.mutate(jobs=[JobState(id="bash-1", type="bash", status="completed")])
    assert update is not None
    delta_row = update.model_dump(mode="json")["changes"]["jobs"][0]
    assert "launch_message_id" not in delta_row and "launch_prompts" not in delta_row
    # And the follower rebuilt from those rows behaves exactly as before.
    follower = FrontendStateStore(_state())
    follower.apply_update(FrontendUpdate.model_validate(update.model_dump(mode="json")))
    node = SnapshotSubagentComms(list(follower.state.jobs)).node("bash-1")
    # Absence stays absence: a bash job never had a launch turn, and the
    # follower must not invent an identity that matches no durable row.
    assert node is not None and node.launch_message_id == "" and node.launch_prompts == {}


def _resumed_row(job_id: str, *, aliases: list[str], prompts: dict[str, str]) -> JobState:
    """One resumed child: a current launch plus its collapsed predecessors."""
    return JobState(
        id=job_id,
        type="task",
        prompt="Current brief.",
        launch_message_id=f"subagent-launch:{job_id}",
        attempt_aliases=aliases,
        launch_prompts=prompts,
    )


def test_a_starved_row_keeps_its_keys_so_collapsed_attempts_still_reconcile() -> None:
    """Round-1 MAJOR: emptying a starved map re-introduced the preamble leak.

    Reconciliation matches a durable launch row BY KEY. The original bound
    dropped a starved row's whole map, arguing the view re-derives the current
    launch from the job's own ``prompt`` — true, but a COLLAPSED PRIOR attempt
    has no other source, so its durable row fell back to a plain user row
    carrying the full role/team/system preamble. Reproduced independently by
    review and QA on a realistic roster.

    The keys now survive with a placeholder value, which must stay VISIBLE:
    ``SubagentView.show`` drops an entry whose text strips to falsy, so an empty
    placeholder would leak exactly as before.
    """
    filler = [
        _resumed_row(
            f"filler{index}",
            aliases=[],
            prompts={f"subagent-launch:filler{index}-a{a}": "F" * 200 for a in range(8)},
        )
        for index in range(60)
    ]
    target = _resumed_row(
        "target",
        aliases=["target-prior"],
        prompts={
            "subagent-launch:target-prior": "First attempt.",
            "subagent-launch:target": "Current brief.",
        },
    )
    owner = FrontendStateStore(_state(*filler, target))
    rows = sync_wire_payload(owner.subscribe(lambda _: None).sync)["snapshot"]["jobs"]
    wire_target = next(row for row in rows if row["id"] == "target")

    served = wire_target.get("launch_prompts") or {}
    assert served, "the target row was dropped entirely; the fixture no longer starves it"
    assert set(served.values()) == {
        LAUNCH_PROMPT_ELIDED_PLACEHOLDER
    }, "the fixture no longer exercises the starved tier"
    # EVERY key survives — that is what keeps the durable rows reconcilable.
    assert set(served) == {"subagent-launch:target-prior", "subagent-launch:target"}
    assert strip_control_sequences(
        LAUNCH_PROMPT_ELIDED_PLACEHOLDER
    ).strip(), "the placeholder must survive the view's own strip or the entry is dropped"


def test_budget_degradation_is_monotonic_across_the_roster() -> None:
    """A cheap late row must not out-rank an expensive earlier one.

    Deciding each row independently let a small row past the budget render at
    full text while its predecessors were elided, so two children with the same
    history looked different purely by roster position — and a dropped row frees
    no budget, so a starved row could be followed by an unstarved one.
    """
    heavy = [
        _resumed_row(
            f"heavy{index}",
            aliases=[],
            prompts={f"subagent-launch:heavy{index}-a{a}": "H" * 200 for a in range(8)},
        )
        for index in range(200)
    ]
    light = _resumed_row(
        "light",
        aliases=[],
        prompts={"subagent-launch:light-a0": "tiny"},
    )
    owner = FrontendStateStore(_state(*heavy, light))
    rows = sync_wire_payload(owner.subscribe(lambda _: None).sync)["snapshot"]["jobs"]
    tail = next(row for row in rows if row["id"] == "light")
    served = tail.get("launch_prompts") or {}
    assert served, "the tail row lost its keys and can no longer reconcile"
    assert set(served.values()) == {
        LAUNCH_PROMPT_ELIDED_PLACEHOLDER
    }, "a late cheap row rendered at full text after earlier rows were starved"


def test_a_derivable_launch_identity_is_omitted_and_rebuilt_by_the_follower() -> None:
    """Round-1 BLOCKER: the identity was a per-row scalar bounded by nothing.

    At 46.7 B on every task child it cut the maximum attachable roster from 812
    rows on v0.49.2 to 769 — a session that attaches on the shipped release and
    does not attach here. An identity cannot be truncated (a clipped key matches
    no durable row), so it is OMITTED where it is reconstructible instead:
    ``run_subagent`` mints it as ``subagent-launch:<job_id>``, so the follower
    rebuilds the identical string.
    """
    row = _resumed_row("child-1", aliases=[], prompts={})
    owner = FrontendStateStore(_state(row))
    wire = sync_wire_payload(owner.subscribe(lambda _: None).sync)["snapshot"]["jobs"][0]
    assert "launch_message_id" not in wire, "the derivable identity still rides the wire"
    assert "launch_id_derived" not in wire, "a marker field replaced the string it saved"

    follower = FrontendStateStore(
        FrontendSessionState.model_validate(
            {"session_id": "root", "epoch": "epoch", "jobs": [wire]}
        )
    )
    node = SnapshotSubagentComms(list(follower.state.jobs)).node("child-1")
    assert node is not None
    assert node.launch_message_id == "subagent-launch:child-1"

    # A NON-derivable identity is information the job id does not carry — a
    # resumed child rebuilt from a persisted row — and must ride verbatim.
    literal = JobState(id="child-2", type="task", launch_message_id="subagent-launch:older-attempt")
    owner2 = FrontendStateStore(_state(literal))
    wire2 = sync_wire_payload(owner2.subscribe(lambda _: None).sync)["snapshot"]["jobs"][0]
    assert wire2["launch_message_id"] == "subagent-launch:older-attempt"


def test_a_dropped_map_is_rebuilt_from_attempt_aliases() -> None:
    """Even the cheapest tier keeps every collapsed attempt reconcilable.

    When the budget cannot afford a row's KEYS either, the map is dropped — but
    the keys are themselves derivable: an attempt's identity is
    ``subagent-launch:<its job id>`` and ``attempt_aliases`` already lists those
    ids on the same row, bounded at one short id each. The follower rebuilds
    them, so no tier of this bound can resurrect the duplicate.
    """
    heavy = [
        _resumed_row(
            f"heavy{index}",
            aliases=[],
            prompts={f"subagent-launch:heavy{index}-a{a}": "H" * 200 for a in range(8)},
        )
        for index in range(400)
    ]
    target = _resumed_row(
        "target",
        aliases=["prior-1", "prior-2"],
        prompts={
            "subagent-launch:prior-1": "First.",
            "subagent-launch:prior-2": "Second.",
            "subagent-launch:target": "Current brief.",
        },
    )
    owner = FrontendStateStore(_state(*heavy, target))
    rows = sync_wire_payload(owner.subscribe(lambda _: None).sync)["snapshot"]["jobs"]
    follower = FrontendStateStore(
        FrontendSessionState.model_validate({"session_id": "root", "epoch": "epoch", "jobs": rows})
    )
    node = SnapshotSubagentComms(list(follower.state.jobs)).node("target")
    assert node is not None
    # Whatever tier the target landed on, BOTH collapsed attempts reconcile.
    for alias in ("prior-1", "prior-2"):
        assert (
            f"subagent-launch:{alias}" in node.launch_prompts
        ), f"{alias} lost its identity and would render its full preamble"


def test_a_whitespace_only_prompt_never_reaches_the_wire() -> None:
    """Round-1 finding 3: it bought bytes and could never reconcile.

    The view strips the value and drops the entry, so the outcome was already
    correct — but the bytes were spent and the key silently failed to fold.
    """
    row = JobState(
        id="child-1",
        type="task",
        launch_prompts={"subagent-launch:a": "   ", "subagent-launch:b": " real "},
    )
    owner = FrontendStateStore(_state(row))
    wire = sync_wire_payload(owner.subscribe(lambda _: None).sync)["snapshot"]["jobs"][0]
    assert wire.get("launch_prompts") == {"subagent-launch:b": "real"}
