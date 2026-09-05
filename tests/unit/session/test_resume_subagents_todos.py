"""Subagents and todos survive a resume.

THE bug: after quitting and resuming a session, the subagent list, their
statuses, and the todo list all vanished — even though the children's
transcripts were still on disk and could in principle be continued. Every one
of those structures lived only in memory:

* ``AsyncJobManager._jobs`` — the rows the subagent panel and the ``jobs`` tool
  paint;
* ``SubagentComms._records`` — the roster and the ``job_id -> session_dir``
  mapping ``hub op='resume'`` relaunches a child from;
* ``TODO_STORE[session_id]`` — the todo list the panel and the continuation
  guardrail read.

The session now snapshots all three to its transcript as custom entries and
rehydrates them at construction, exactly the way wake schedules and the
conversation title already did. These tests drive a REAL parent session and a
REAL child run through a scripted provider, tear the session down, reopen it on
the same directory, and assert the children and todos come back — and, the
whole point, that a restored child is still resumable.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
    ToolResult,
    Usage,
)
from local_operator.session import session as session_module
from local_operator.session.session import (
    SUBAGENT_ROSTER_CUSTOM_TYPE,
    SUBAGENT_ROSTER_SIDECAR,
    TODO_SNAPSHOT_CUSTOM_TYPE,
    Session,
    _compact_subagent_record,
)
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import TODO_STORE, execute_todo
from tests.unit.session.test_launch_subagent import MODEL, OneShotStream


@pytest.fixture(autouse=True)
def _clean_todo_store():
    """The todo tool's store is a MODULE-GLOBAL keyed by session id, and every
    session in this suite opens on ``tmp_path/"sess"`` — so its id is ``"sess"``
    for all of them, and a list one test leaves would be read by the next (and,
    worse, by ``test_session.py`` sessions sharing that id, whose transcript
    entry counts then include a todo snapshot they never wrote). Clear it on the
    way in and out so each test starts and ends with an empty store.
    """
    TODO_STORE.clear()
    yield
    TODO_STORE.clear()


async def wait_for(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


def _status(session, job_id: str) -> str | None:
    """The status of ``job_id`` on ``session``'s manager, or ``None`` when the
    row is not (yet) present — narrowed so assertions never index ``None``."""
    job = session.jobs.get(job_id)
    return None if job is None else job.status


async def wait_for_settled_delivery(session, stream) -> None:
    """Block until the auto-delivered parent turn a settled child triggers has
    both STARTED and finished.

    A completed subagent is not the end of the parent's activity: settlement
    runs ``Session._on_job_completed``, which spawns a background parent turn
    that feeds the child's result back into the conversation. That turn is a
    real production behaviour, but it is asynchronous and nothing about the
    child's ``completed`` status implies it has run — so a test that merely
    waits on the child's status is waiting on a PROXY for parent quiescence
    that the parent has not actually reached.

    Both halves are load-bearing. Waiting only for "not streaming" can observe
    the gap BEFORE the delivery turn starts and return immediately; waiting
    only for the request count can return while the turn is still streaming.
    Together they pin the turn's whole lifecycle, which is what makes a
    following observation window contain only what the test provoked.

    Without this, a delivery turn that happened to still be in flight published
    its own ``streaming`` and ``history_cursor`` updates into a window the test
    believed held only its own coalesced batch (measured: ``assert 3 == 1`` at
    ~10% locally, and reproducible on demand by delaying the parent stream).
    """
    # The delivery turn's own provider call is the only unambiguous evidence it
    # started; the launch turn is request #1.
    await wait_for(lambda: len(stream.requests) >= 2)
    await wait_for(lambda: not session.is_streaming)
    await wait_for(lambda: not any(not task.done() for task in session._background_tasks))


async def _todo(ctx, call_id: str, args: dict[str, object]) -> None:
    """Drive the todo tool with the full positional signature its guard
    decorator declares (signal / on_update default to ``None``)."""
    await execute_todo(call_id, args, None, None, ctx)


class HangingStream:
    """A provider turn that stays live until the owning job is cancelled."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        async def gen():
            self.started.set()
            assert signal is not None
            await signal.wait()
            yield StreamEndEvent(stop_reason="aborted")

        return gen()


class IdleStream:
    """A parent stream that answers every turn with a single text delta.

    The parent never calls a tool here — the test drives ``_launch_subagent``
    and the todo tool directly — so one text turn is all it needs.
    """

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def _session(tmp_path, stream) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
    )


@pytest.mark.asyncio
async def test_a_completed_subagent_is_restored_and_resumable(tmp_path, monkeypatch):
    """Launch a real child, let it finish, tear down, resume — the child is
    back on the roster AND ``hub op='resume'`` can pick it up."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    parent = _session(tmp_path, OneShotStream())
    await parent.async_init()
    job_id = parent._launch_subagent(label="explore", prompt="do a thing")
    await wait_for(lambda: _status(parent, job_id) == "completed")
    # Let the post-settle roster snapshot land on the task group.
    await asyncio.sleep(0.05)
    session_dir = parent.subagent_comms.session_dir_of(job_id)
    assert session_dir is not None
    await parent.dispose()

    # Resume: a fresh Session over the same transcript directory.
    resumed = _session(tmp_path, IdleStream())
    await resumed.async_init()

    # The panel's data source — the job manager — shows the child again.
    rows = [j for j in resumed.jobs.list() if j.type == "task"]
    assert len(rows) == 1
    assert rows[0].id == job_id
    assert rows[0].label == "explore"
    assert rows[0].status == "completed"
    assert rows[0].restored is True

    # The roster shows it and marks it resumable, and resume actually starts a
    # fresh job against the OLD transcript directory.
    roster = resumed.subagent_comms.roster()
    assert len(roster) == 1
    assert roster[0].resumable is True
    assert resumed.subagent_comms.session_dir_of(job_id) == session_dir

    new_job_id, error = resumed.subagent_comms.resume(job_id, "keep going")
    assert error is None
    assert new_job_id and new_job_id != job_id

    def _is_reconciled() -> bool:
        row = resumed.jobs.get(new_job_id)
        return row is not None and row.logical_id == str(session_dir)

    await wait_for(_is_reconciled)
    rows = [j for j in resumed.jobs.list() if j.type == "task"]
    assert [row.id for row in rows] == [new_job_id]
    assert resumed.jobs.get(job_id) is resumed.jobs.get(new_job_id)
    assert [item.job_id for item in resumed.subagent_comms.roster()] == [new_job_id]
    assert resumed.subagent_comms.session_dir_of(job_id) == session_dir
    await resumed.dispose()


@pytest.mark.asyncio
async def test_live_continuation_persists_one_logical_row_before_restart(tmp_path, monkeypatch):
    """Binding is the durability boundary, not eventual child settlement."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    parent = _session(tmp_path, OneShotStream())
    await parent.async_init()
    old_id = parent._launch_subagent(label="explore", prompt="do a thing")
    await wait_for(lambda: _status(parent, old_id) == "completed")
    await parent._persist_subagent_roster()
    await parent.dispose()

    hanging = HangingStream()
    resumed = _session(tmp_path, hanging)
    await resumed.async_init()
    new_id, error = resumed.subagent_comms.resume(old_id, "keep going")
    assert error is None and new_id is not None
    await hanging.started.wait()
    await wait_for(
        lambda: resumed.jobs.get(new_id) is not None
        and resumed.jobs.get(new_id).logical_id is not None  # type: ignore[union-attr]
    )

    def _binding_is_durable() -> bool:
        sidecar = resumed._transcript.directory / SUBAGENT_ROSTER_SIDECAR
        try:
            details = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            return False
        return [row.get("id") for row in details.get("jobs", [])] == [new_id] and [
            row.get("job_id") for row in details.get("records", [])
        ] == [new_id]

    await wait_for(_binding_is_durable)
    restarted = _session(tmp_path, IdleStream())
    await restarted.async_init()
    assert [row.id for row in restarted.jobs.list() if row.type == "task"] == [new_id]
    assert restarted.jobs.get(old_id) is restarted.jobs.get(new_id)
    assert [item.job_id for item in restarted.subagent_comms.roster()] == [new_id]

    await restarted.dispose()
    await resumed.dispose()


@pytest.mark.asyncio
async def test_live_continuation_preserves_prior_accounting_across_restart(tmp_path, monkeypatch):
    """A folded predecessor's settled usage must live in the authoritative row."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    stream = HangingStream()
    parent = _session(tmp_path, stream)
    await parent.async_init()
    old_id = parent._launch_subagent(label="accounted", prompt="first attempt")
    await stream.started.wait()
    old_row = parent.jobs.get(old_id)
    assert old_row is not None
    old_row.usage = Usage(input_tokens=4, provider="test", model_id="m")
    # ``usage`` is a durable roster field, and this test mutates it DIRECTLY
    # rather than through the relay, so nothing else announces it here. The
    # announcement is what marks the sidecar dirty for the watermark-guarded
    # writer (see AsyncJobManager.note_usage_changed, which also records why
    # no shipped path depends on it).
    parent.jobs.note_usage_changed()
    await parent._persist_subagent_roster()
    await parent.dispose()

    hanging = HangingStream()
    resumed = _session(tmp_path, hanging)
    await resumed.async_init()
    new_id, error = resumed.subagent_comms.resume(old_id, "continue")
    assert error is None and new_id is not None
    await hanging.started.wait()
    # Wait on the ACCOUNTING itself, not on the identity bind that precedes
    # it. ``bind_logical_identity`` sets ``logical_id`` and then folds the
    # predecessor's usage into ``prior_attempt_usage``; asserting on the
    # second after waiting on the first is wait-on-a-proxy-then-assert-on-
    # the-real-thing, and under load the fold has not landed when the
    # identity has (#463: ``assert 0 == 4`` on an otherwise-green commit).
    #
    # This wait depends on the predecessor's usage having reached the sidecar
    # above, which is why the direct mutation is announced. It is NOT the
    # cause of the intermittent #548 CI failure in this file: reverting that
    # announcement leaves this test passing (41/41 measured, including under
    # concurrency). The flake was parent-turn quiescence in
    # test_live_progress_never_schedules_the_roster_writer; see
    # wait_for_settled_delivery.
    await wait_for(
        lambda: sum(item.input_tokens for item in resumed.jobs.accounting_components()) == 4,
        timeout=30.0,
    )
    assert sum(item.input_tokens for item in resumed.jobs.accounting_components()) == 4
    await resumed._persist_subagent_roster()

    restarted = _session(tmp_path, IdleStream())
    await restarted.async_init()
    current = restarted.jobs.get(new_id)
    assert current is not None
    assert [row.id for row in restarted.jobs.list() if row.type == "task"] == [new_id]
    assert restarted.jobs.get(old_id) is current
    assert sum(item.input_tokens for item in restarted.jobs.accounting_components()) == 4
    assert sum(item.input_tokens for item in restarted.jobs.accounting_components()) == 4

    await restarted.dispose()
    await resumed.dispose()


@pytest.mark.asyncio
async def test_usage_mutation_persists_when_the_writer_already_drained(tmp_path, monkeypatch):
    """A running child's usage reaches the sidecar even when the roster writer
    has ALREADY caught up before the mutation.

    ``usage`` is a durable roster field, but the session's writer is
    watermark-guarded: ``_persist_subagent_roster`` flushes only while
    ``written_generation < generation``. A mutation that announces nothing
    therefore leaves the very next persist a no-op and the tokens off disk.

    Draining the writer FIRST is what makes this deterministic: it removes the
    pending flush an unannounced mutation would otherwise ride along on. This
    pins the manager seam's own contract — no shipped path depends on it
    today (see ``AsyncJobManager.note_usage_changed`` for what covers it), and
    this is NOT the cause of the #548 flake.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    stream = HangingStream()
    parent = _session(tmp_path, stream)
    await parent.async_init()
    job_id = parent._launch_subagent(label="accounted", prompt="do a thing")
    await stream.started.wait()

    # The whole point: no pending flush is left for the mutation to ride on.
    await parent._await_subagent_roster_writer()
    assert parent._subagent_roster_written_generation == parent._subagent_roster_generation

    row = parent.jobs.get(job_id)
    assert row is not None
    # Mutate before settlement: afterwards the manager owns a detached bill,
    # and rewriting a display row is deliberately not a financial correction.
    row.usage = Usage(input_tokens=4, provider="test", model_id="m")
    parent.jobs.note_usage_changed()
    await parent._persist_subagent_roster()

    sidecar = parent._transcript.directory / SUBAGENT_ROSTER_SIDECAR
    persisted = json.loads(sidecar.read_text())
    assert [(item.get("usage") or {}).get("input_tokens") for item in persisted["jobs"]] == [4]
    await parent.dispose()

    # And it survives the round trip a resume actually performs.
    resumed = _session(tmp_path, IdleStream())
    await resumed.async_init()
    assert sum(item.input_tokens for item in resumed.jobs.accounting_components()) == 4
    await resumed.dispose()


@pytest.mark.asyncio
async def test_detach_persists_descendant_usage_when_the_writer_already_drained(
    tmp_path, monkeypatch
):
    """``detach_child_manager`` announces its durable write too.

    Same seam contract as the usage mutation above, on the other durable
    accounting field: ``descendant_usage`` is in the roster projection, so
    replacing the live child edge with the detached ledger has to mark the
    sidecar dirty. Without the announcement the drained writer skips the
    persist and the field lands as ``[]`` on disk.

    Also covered incidentally in production (the runner detaches just before
    the row settles, and settling persists), so this pins the seam rather than
    a reachable loss.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    stream = HangingStream()
    parent = _session(tmp_path, stream)
    await parent.async_init()
    job_id = parent._launch_subagent(label="nested", prompt="do a thing")
    await stream.started.wait()

    await parent._await_subagent_roster_writer()
    assert parent._subagent_roster_written_generation == parent._subagent_roster_generation

    parent.jobs.detach_child_manager(job_id, [Usage(input_tokens=7, provider="test", model_id="m")])
    await parent._persist_subagent_roster()

    sidecar = parent._transcript.directory / SUBAGENT_ROSTER_SIDECAR
    persisted = json.loads(sidecar.read_text())
    assert [
        [item.get("input_tokens") for item in (row.get("descendant_usage") or [])]
        for row in persisted["jobs"]
    ] == [[7]]
    await parent.dispose()

    resumed = _session(tmp_path, IdleStream())
    await resumed.async_init()
    assert sum(item.input_tokens for item in resumed.jobs.accounting_components()) == 7
    await resumed.dispose()


@pytest.mark.asyncio
async def test_descendant_accounting_survives_process_resume(tmp_path, monkeypatch):
    """Nested spend lives on the owning row, not an in-process child manager."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    stream = HangingStream()
    parent = _session(tmp_path, stream)
    await parent.async_init()
    job_id = parent._launch_subagent(label="nested", prompt="do nested work")
    await stream.started.wait()
    row = parent.jobs.get(job_id)
    assert row is not None
    row.descendant_usage = [
        Usage(
            input_tokens=5,
            usd_cost=0.125,
            provider="openrouter",
            model_id="routed",
        ),
        Usage(input_tokens=7, provider="test", model_id="m"),
    ]
    # The runner replaces descendant_usage from its child's final ledger during
    # teardown, so seed that owner rather than only its temporary display copy.
    assert row.child_jobs is not None
    row.child_jobs.restore_accounting(row.descendant_usage)
    parent.jobs.note_usage_changed()
    await parent._persist_subagent_roster()
    await parent.dispose()

    resumed = _session(tmp_path, IdleStream())
    await resumed.async_init()
    restored = resumed.jobs.get(job_id)
    assert restored is not None
    assert [
        (item.provider, item.model_id, item.usd_cost) for item in restored.descendant_usage
    ] == [
        ("openrouter", "routed", 0.125),
        ("test", "m", None),
    ]
    assert sum(item.input_tokens for item in resumed.jobs.accounting_components()) >= 12
    await resumed.dispose()


@pytest.mark.asyncio
async def test_role_and_effort_survive_a_resume(tmp_path, monkeypatch):
    """The child's agent ROLE and effort TIER are on the roster allowlist, so a
    restored row still names them.

    Regression guard for the seam between two features that shipped together:
    the roster snapshot is an ALLOWLIST (``_ROSTER_ROW_FIELDS``), so a field
    that is stamped on the live job but not listed works in-process and then
    silently blanks on every resume. The subagent page title and the status
    band both read these two facts off the restored row, so a child a previous
    process launched must come back saying what kind it was and at what effort —
    exactly what the panel/model/usage fields already promise."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    # An explicit tier now fails the launch when nothing is configured for it
    # (``test_pinned_subagent_model``), so the tier under test is configured —
    # to the parent's own selector, so the child still runs on OneShotStream.
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "config.yml").write_text(
        f"values:\n  subagents:\n    models:\n      hi: {MODEL.provider}/{MODEL.model_id}\n"
    )

    parent = _session(tmp_path, OneShotStream())
    await parent.async_init()
    # A non-default role and an explicit tier, so a blanked field would read as
    # the default rather than as the value under test.
    job_id = parent._launch_subagent(label="scan", prompt="look around", agent="scout", effort="hi")
    await wait_for(lambda: _status(parent, job_id) == "completed")
    await asyncio.sleep(0.05)  # let the post-settle roster snapshot land
    await parent.dispose()

    resumed = _session(tmp_path, IdleStream())
    await resumed.async_init()
    rows = [j for j in resumed.jobs.list() if j.type == "task"]
    assert len(rows) == 1
    assert rows[0].agent_role == "scout"
    assert rows[0].effort == "hi"
    await resumed.dispose()


@pytest.mark.asyncio
async def test_a_running_child_restores_as_interrupted(tmp_path, monkeypatch):
    """A child still running when the process exits comes back as
    ``interrupted`` — not a spinning ``running`` its manager can never settle —
    and stays resumable because its transcript survived."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    class HangingStream:
        """A child stream that never ends, so the child is still 'running' at
        teardown — the crash-mid-run case."""

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            async def gen():
                yield StreamTextDelta(delta="working")
                await asyncio.sleep(30)
                yield StreamEndEvent(stop_reason="stop")

            return gen()

    parent = _session(tmp_path, HangingStream())
    await parent.async_init()
    job_id = parent._launch_subagent(label="long", prompt="a long job")
    # Wait until the child has attached (its session_dir is known) so the
    # snapshot names a resumable transcript.
    await wait_for(lambda: parent.subagent_comms.session_dir_of(job_id) is not None)
    await asyncio.sleep(0.05)  # let the post-attach snapshot land
    # dispose cancels the running child; the roster snapshot on disk still
    # holds the row as it was persisted (running) BEFORE the cancel.
    await parent.dispose()

    resumed = _session(tmp_path, IdleStream())
    await resumed.async_init()
    rows = [j for j in resumed.jobs.list() if j.type == "task"]
    assert len(rows) == 1
    assert rows[0].status == "interrupted"
    info = resumed.subagent_comms.roster()[0]
    assert info.status == "interrupted"
    assert info.resumable is True
    await resumed.dispose()


@pytest.mark.asyncio
async def test_todos_survive_a_resume(tmp_path):
    """A todo list built during a turn comes back after a resume, with every
    item and status intact."""
    TODO_STORE.pop((tmp_path / "sess").name, None)
    parent = _session(tmp_path, IdleStream())
    await parent.async_init()
    sid = parent.session_id

    ctx = parent._build_tool_context()
    await _todo(ctx, "t1", {"op": "init", "items": ["alpha", "beta", "gamma"]})
    await _todo(ctx, "t2", {"op": "done", "items": ["alpha"]})
    await _todo(ctx, "t3", {"op": "block", "items": ["beta"], "reason": "waiting on X"})
    # Persist happens at turn end; drive one turn so the snapshot is written.
    await parent.prompt("anything")
    await parent.dispose()
    # Clear the live store so the resume genuinely rehydrates from disk.
    TODO_STORE.pop(sid, None)

    resumed = _session(tmp_path, IdleStream())
    await resumed.async_init()
    # The store is PHASED now (phased-todos change): a flat ``init`` restores as
    # one implicit "Todos" phase, so flatten across phases before checking the
    # items survived. Persistence round-trips the phase structure, not a bare
    # item list — a resume that flattened would lose a multi-phase plan.
    phases = TODO_STORE.get(resumed.session_id) or []
    items = [item for phase in phases for item in phase["items"]]
    by_text = {i["text"]: i for i in items}
    assert by_text["alpha"]["status"] == "done"
    assert by_text["beta"]["status"] == "blocked"
    assert by_text["beta"].get("reason") == "waiting on X"
    assert by_text["gamma"]["status"] == "pending"
    await resumed.dispose()


@pytest.mark.asyncio
async def test_phased_todos_survive_a_resume_with_phase_identity(tmp_path):
    """A MULTI-PHASE plan round-trips through a resume with its phase names and
    per-phase membership intact.

    The persistence path (``todo_snapshot``/``restore_todos``) landed on main
    for a FLAT store; the phased-todos change made the store phased, so this
    pins the integration: a resume must rebuild the phase structure, not
    collapse it into one anonymous list. Without the phase-aware snapshot a
    two-phase plan would come back as a single implicit phase and the panel's
    headers, per-phase progress, and the guardrail's phase-aware fingerprint
    would all differ from what they were before the restart.
    """
    TODO_STORE.pop((tmp_path / "sess").name, None)
    parent = _session(tmp_path, IdleStream())
    await parent.async_init()
    sid = parent.session_id

    ctx = parent._build_tool_context()
    await _todo(
        ctx,
        "p1",
        {
            "op": "init",
            "phases": [
                {"phase": "Foundation", "items": ["scaffold", "wire config"]},
                {"phase": "Verification", "items": ["run gate"]},
            ],
        },
    )
    await _todo(ctx, "p2", {"op": "done", "phase": "Foundation"})
    await parent.prompt("anything")
    await parent.dispose()
    TODO_STORE.pop(sid, None)

    resumed = _session(tmp_path, IdleStream())
    await resumed.async_init()
    phases = TODO_STORE.get(resumed.session_id) or []
    assert [p["name"] for p in phases] == ["Foundation", "Verification"]
    foundation = {i["text"]: i["status"] for i in phases[0]["items"]}
    assert foundation == {"scaffold": "done", "wire config": "done"}
    verification = {i["text"]: i["status"] for i in phases[1]["items"]}
    assert verification == {"run gate": "pending"}
    await resumed.dispose()


@pytest.mark.asyncio
async def test_unchanged_todos_are_not_re_persisted_every_turn(tmp_path):
    """The post-turn snapshot is guarded by a fingerprint, so a turn that did
    not touch the list writes no new snapshot entry."""
    TODO_STORE.pop((tmp_path / "sess").name, None)
    parent = _session(tmp_path, IdleStream())
    await parent.async_init()
    ctx = parent._build_tool_context()
    await _todo(ctx, "t1", {"op": "init", "items": ["one"]})
    await parent.prompt("turn one")

    def _count() -> int:
        return sum(
            1
            for e in parent._transcript.entries()
            if e.type == "custom" and e.payload.get("custom_type") == TODO_SNAPSHOT_CUSTOM_TYPE
        )

    after_first = _count()
    assert after_first >= 1
    await parent.prompt("turn two — no todo change")
    assert _count() == after_first  # unchanged list wrote nothing new
    await parent.dispose()


@pytest.mark.asyncio
async def test_malformed_v1_sidecar_falls_back_to_legacy_roster(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    transcript = Transcript(tmp_path / "sess")
    await transcript.append_custom(
        SUBAGENT_ROSTER_CUSTOM_TYPE,
        {
            "jobs": [],
            "records": [
                {
                    "job_id": "legacy",
                    "label": "legacy",
                    "session_dir": str(tmp_path / "child"),
                    "outcome": "completed",
                }
            ],
        },
    )
    (transcript.directory / SUBAGENT_ROSTER_SIDECAR).write_text(
        json.dumps({"version": 1, "generation": "broken", "jobs": [], "records": []})
    )
    resumed = _session(tmp_path, IdleStream())
    assert [row.job_id for row in resumed.subagent_comms.roster()] == ["legacy"]


@pytest.mark.asyncio
async def test_todo_snapshot_precedes_cancelling_tool_end_subscriber(tmp_path, monkeypatch) -> None:
    """A real todo mutation is durable before async ToolExecutionEnd fan-out."""
    from local_operator.harness.types import ToolExecutionEndEvent

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = _session(tmp_path, IdleStream())
    TODO_STORE[parent.session_id] = [
        {"name": "Nested", "items": [{"text": "survive", "status": "pending"}]}
    ]
    seen: list[str] = []

    async def cancelling_subscriber(event) -> None:  # noqa: ANN001
        if isinstance(event, ToolExecutionEndEvent) and event.tool_name == "todo":
            details = parent._transcript.latest_custom(TODO_SNAPSHOT_CUSTOM_TYPE)
            assert details is not None
            seen.append(details["items"][0]["items"][0]["text"])
            raise asyncio.CancelledError

    parent.subscribe(cancelling_subscriber)
    event = ToolExecutionEndEvent(
        tool_call_id="todo-1",
        tool_name="todo",
        result=ToolResult(tool_call_id="todo-1", content=[TextContent(text="ok")]),
    )
    with pytest.raises(asyncio.CancelledError):
        # This loop fragment is the exact production ordering under test.
        await parent._maybe_persist_todos()
        await parent._emit(event)
    assert seen == ["survive"]


@pytest.mark.asyncio
async def test_child_detail_notification_follows_durable_append(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    notifications: list[str] = []
    comms = type(
        "Comms",
        (),
        {"notify_detail_persisted": lambda self, job_id: notifications.append(job_id)},
    )()
    child = Session(
        model=MODEL,
        stream_fn=IdleStream(),
        tools=[],
        transcript=Transcript(tmp_path / "child"),
        system_blocks_provider=lambda: ["stable"],
        job_id="nested",
        subagent_comms=comms,
    )
    original_messages = child._persist_new_messages
    original_todos = child._maybe_persist_todos
    TODO_STORE[child.session_id] = [
        {
            "name": "Nested",
            "items": [{"text": "persisted after error", "status": "pending"}],
        }
    ]

    async def checked_messages(messages):  # noqa: ANN001, ANN202
        assert notifications == []
        await original_messages(messages)
        assert notifications == []

    async def checked_todos() -> None:
        assert notifications == []
        await original_todos()
        assert notifications == []
        details = child._transcript.latest_custom(TODO_SNAPSHOT_CUSTOM_TYPE)
        assert details is not None
        assert details["items"][0]["items"][0]["text"] == "persisted after error"

    monkeypatch.setattr(child, "_persist_new_messages", checked_messages)
    monkeypatch.setattr(child, "_maybe_persist_todos", checked_todos)
    await child.prompt("persist this child turn")
    assert notifications == ["nested"]
    assert len(child._transcript.build_llm_history()) > 1


@pytest.mark.asyncio
async def test_empty_roster_final_flush_persists_todos_without_delay(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = _session(tmp_path, IdleStream())
    TODO_STORE[parent.session_id] = [
        {"name": "Work", "items": [{"text": "persist me", "status": "pending"}]}
    ]
    loop = asyncio.get_running_loop()
    started = loop.time()
    await parent._final_persist_snapshots()
    assert loop.time() - started < 0.5
    assert parent._subagent_roster_written_generation == parent._subagent_roster_generation
    details = parent._transcript.latest_custom(TODO_SNAPSHOT_CUSTOM_TYPE)
    assert details is not None
    assert details["items"][0]["items"][0]["text"] == "persist me"


@pytest.mark.asyncio
async def test_final_roster_persist_waits_for_active_writer(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = _session(tmp_path, IdleStream())
    parent.jobs.restore([])
    parent._subagent_roster_generation = 1
    entered = asyncio.Event()
    release = asyncio.Event()
    writes: list[int] = []

    async def blocked_write() -> None:
        entered.set()
        await release.wait()
        writes.append(parent._subagent_roster_generation)
        parent._subagent_roster_written_generation = parent._subagent_roster_generation
        if parent._subagent_roster_writer is asyncio.current_task():
            parent._subagent_roster_writer = None

    writer = asyncio.create_task(blocked_write())
    parent._subagent_roster_writer = writer
    await entered.wait()
    final = asyncio.create_task(parent._final_persist_snapshots())
    await asyncio.sleep(0)
    assert not final.done()
    release.set()
    await final
    assert writes == [2]
    assert parent._subagent_roster_written_generation == 2


@pytest.mark.asyncio
async def test_snapshot_entry_is_written_for_a_launched_child(tmp_path, monkeypatch):
    """A launched child writes a roster snapshot custom entry to the
    transcript — the durable record the resume reads."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = _session(tmp_path, OneShotStream())
    await parent.async_init()
    job_id = parent._launch_subagent(label="x", prompt="p")
    await wait_for(lambda: _status(parent, job_id) == "completed")
    await asyncio.sleep(0.05)
    entries = [
        e
        for e in parent._transcript.entries()
        if e.type == "custom" and e.payload.get("custom_type") == SUBAGENT_ROSTER_CUSTOM_TYPE
    ]
    assert entries
    sidecar = parent._transcript.directory / SUBAGENT_ROSTER_SIDECAR
    assert sidecar.exists()
    first_size = sidecar.stat().st_size
    transcript_size = parent._transcript.path.stat().st_size
    for _ in range(25):
        parent._schedule_subagent_persist()
    await asyncio.sleep(0.1)
    # Generation digits may add a byte, but repeated transitions replace one
    # bounded file and never append another full roster to the transcript.
    assert sidecar.stat().st_size <= first_size + 4
    assert parent._transcript.path.stat().st_size == transcript_size
    latest = entries[-1].payload["details"]
    # Job rows carry the manager's own ``id`` key; comms records carry job_id.
    row = next(r for r in latest["jobs"] if r["id"] == job_id)
    assert any(r["job_id"] == job_id for r in latest["records"])
    # The snapshot is a SLIM projection: the unbounded fields a heavy child
    # carries (its full reply, its verbatim prompt, its trajectory, its live
    # output tail) must not be written — they are recoverable from the child's
    # own transcript, and re-appending them on every roster move is the
    # O(children^2) footprint the projection exists to prevent (review R1 M1).
    for heavy in ("result_text", "prompt", "trajectory", "output_tail", "latest_details"):
        assert heavy not in row, f"{heavy} must not be in the roster snapshot"
    # But the fields the panel actually paints ARE kept.
    for kept in ("id", "label", "status", "start_time"):
        assert kept in row
    await parent.dispose()


@pytest.mark.asyncio
async def test_live_progress_never_schedules_the_roster_writer(tmp_path, monkeypatch):
    """Transient activity publishes to frontends without touching durability."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = OneShotStream()
    parent = _session(tmp_path, stream)
    await parent.async_init()
    job_id = parent._launch_subagent(label="x", prompt="p")
    await wait_for(lambda: _status(parent, job_id) == "completed")
    # The observation window below must contain ONLY the progress batch this
    # test provokes, so the parent has to be genuinely idle first — see
    # wait_for_settled_delivery for the turn that otherwise publishes into it.
    await wait_for_settled_delivery(parent, stream)
    await parent._await_subagent_roster_writer()
    generation = parent._subagent_roster_generation
    updates = []
    subscription = parent.subscribe_frontend(updates.append)
    updates.clear()  # discard the join-time source reconciliation

    report = parent.jobs._progress_fn(job_id)
    for index in range(60):
        report(f"boundary {index}")
    assert updates == []  # publication waits for the one 50 ms coalescer
    assert parent._subagent_roster_generation == generation
    assert parent._subagent_roster_writer is None
    # The COALESCE is the property under test: 60 progress reports produce one
    # publication, not 60. Its deadline lives in this wait — a slower host makes
    # the wait take longer, never makes it publish twice.
    #
    # A second wall-clock assertion on the same elapsed time used to follow, and
    # it measured the HOST rather than the code: `wait_for` already raises at the
    # same 0.25 s bound, so it could only fire when the loop was descheduled
    # between satisfying the predicate and reading the clock. On a contended
    # machine that is routine — it failed 10/40 on an unmodified tree here,
    # every one of them latency-shaped — so it reported scheduler pressure as a
    # coalescing regression. Batching is asserted by the count below.
    await wait_for(lambda: bool(updates), timeout=0.25)
    assert len(updates) == 1
    assert updates[0].changes["jobs"][0]["latest_details"] == {"progress": "boundary 59"}
    subscription.unsubscribe()

    # Settlement remains durable even though progress is not. A fresh task must
    # advance the generation at registration and again at its terminal outcome.
    before_second = parent._subagent_roster_generation
    second = parent._launch_subagent(label="y", prompt="q")
    await wait_for(lambda: _status(parent, second) == "completed")
    await parent._await_subagent_roster_writer()
    assert parent._subagent_roster_generation > before_second
    await parent.dispose()


@pytest.mark.asyncio
async def test_redundant_roster_events_skip_the_sidecar_write(tmp_path, monkeypatch):
    """A roster event that does not change the persisted projection writes no
    sidecar, but the generation watermark still advances.

    Defensive/manual persist signals can still repeat without a projection
    change. Transient progress never reaches this hook; this test guards the
    fingerprint fallback for the remaining redundant durable notifications.
    Each write is a full fsync + os.replace + directory-fsync; firing one per
    identical event is the disk-write volume that tripped the macOS throttle on
    a long delegating session. The guard collapses those, and the watermark must
    still advance so teardown's writer drain does not loop chasing a generation
    the guard will always skip."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = _session(tmp_path, OneShotStream())
    await parent.async_init()
    job_id = parent._launch_subagent(label="x", prompt="p")
    await wait_for(lambda: _status(parent, job_id) == "completed")

    writes = {"n": 0}
    real_write = session_module._write_roster_sidecar

    def _counting_write(path, payload):
        writes["n"] += 1
        return real_write(path, payload)

    monkeypatch.setattr(session_module, "_write_roster_sidecar", _counting_write)

    # Flush any pending writer from the launch/settle so the counter starts from
    # a settled baseline the guard has already fingerprinted.
    await parent._persist_subagent_roster()
    baseline_writes = writes["n"]
    baseline_generation = parent._subagent_roster_written_generation

    # Fire a burst of roster events that change nothing in the projection.
    for _ in range(20):
        parent._schedule_subagent_persist()
    await parent._await_subagent_roster_writer()

    # No redundant write happened, yet the watermark tracked every bumped
    # generation (otherwise the writer drain would spin).
    assert writes["n"] == baseline_writes
    assert parent._subagent_roster_written_generation > baseline_generation
    assert parent._subagent_roster_written_generation == parent._subagent_roster_generation

    # A REAL change (a second child) must still write.
    second = parent._launch_subagent(label="y", prompt="q")
    await wait_for(lambda: _status(parent, second) == "completed")
    await parent._persist_subagent_roster()
    assert writes["n"] > baseline_writes

    await parent.dispose()


@pytest.mark.asyncio
async def test_compact_record_caps_oversized_text_but_keeps_short_and_fields():
    """The durable record projection bounds the unbounded text tails (#308)
    while preserving short values verbatim and every resume-critical field.

    The cap is what keeps a 30 KB ``result_text`` from bloating the sidecar; the
    verbatim short case is what keeps the roster panel's ``cancelled while
    parked`` one-word detection and the settled-summary display intact. The
    session_dir/outcome/job_id fields are the resume basis and must survive
    untouched."""
    long_record = {
        "job_id": "job1",
        "label": "child1",
        "session_dir": "/sessions/child1",
        "outcome": "completed",
        "prompt": "p" * 10_000,
        "result_text": "R" * 30_000,
        "error_text": "E" * 30_000,
        "attempt_aliases": ["old1"],
        "paused": False,
    }
    compact = _compact_subagent_record(long_record)
    cap = session_module._SUBAGENT_SUMMARY_CHARS
    assert len(compact["result_text"]) == cap
    assert len(compact["error_text"]) == cap
    assert len(compact["prompt"]) == cap
    # Resume-critical fields pass through byte-for-byte.
    assert compact["job_id"] == "job1"
    assert compact["session_dir"] == "/sessions/child1"
    assert compact["outcome"] == "completed"
    assert compact["attempt_aliases"] == ["old1"]

    short_record = {
        "job_id": "job2",
        "session_dir": "/sessions/child2",
        "outcome": "cancelled",
        "result_text": "cancelled",  # the parked-cancel one-word detail
        "error_text": None,
    }
    short_compact = _compact_subagent_record(short_record)
    assert short_compact["result_text"] == "cancelled"  # verbatim, not clipped
    assert short_compact["error_text"] is None  # None survives, not "" (a regression)


@pytest.mark.asyncio
async def test_capped_snapshot_round_trips_through_restore(tmp_path, monkeypatch):
    """A capped durable snapshot restores without error and stays resumable.

    Caps live only in the DURABLE projection; a restore reads those capped rows
    and must still list every child with its session_dir intact so
    ``hub op='resume'`` can relaunch it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = _session(tmp_path, OneShotStream())
    await parent.async_init()
    job_id = parent._launch_subagent(label="heavy", prompt="p")
    await wait_for(lambda: _status(parent, job_id) == "completed")
    # Simulate a heavy settled child: a large result_text on the live record.
    record = parent.subagent_comms._records[job_id]
    record.result_text = "R" * 30_000
    await parent._persist_subagent_roster()
    await parent.dispose()

    resumed = _session(tmp_path, IdleStream())
    await resumed.async_init()
    roster = resumed.subagent_comms.roster()
    assert [row.job_id for row in roster] == [job_id]
    # The restored record carries a capped-but-present result and its resume dir.
    restored = resumed.subagent_comms.snapshot()[0]
    assert restored["session_dir"]
    assert 0 < len(restored["result_text"]) <= session_module._SUBAGENT_SUMMARY_CHARS
    await resumed.dispose()
