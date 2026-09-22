"""A live child cut off mid-flight must read as CUT OFF, not as a clean finish.

``interrupted`` is a RESTORE-only status: it names a ``task`` row that was
``running`` when the process exited, rehydrated from the persisted roster on the
next resume (``harness/jobs.py``), and it never arises from a live transition.
So before this change a child the loop cut off mid-flight reached its parent with
NO vocabulary at all: ``SubagentEndEvent`` carried ``status`` and no cause,
``AsyncJob.cut_off_cause`` existed on the row but nothing wrote it from a live
settle, and the relay copied only ``event.error``. The parent saw a child that
"finished" — the reported "stopping without committing".

The vocabulary it *should* use already existed (``AgentEndEvent.cut_off`` /
``cut_off_cause``, ``incidents.CUT_OFF_CAUSES``, and the panel's "cut off" row);
the writer was what was missing. These tests pin the writer on every surface the
parent reads, and they pin the NEGATIVE arm — a clean completion carries no
cause — because a fix that labelled every child cut off would pass a test that
only asserted "a cause is present".
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.comms import SubagentComms
from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.subagent import _make_relay, _publish_terminal_outcome
from local_operator.harness.types import (
    AbortSignal,
    AgentEndEvent,
    AgentEvent,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    SubagentEndEvent,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

#: The token the loop stamps on a cut-off run. Spelled as a literal — rather
#: than imported from ``incidents`` — so this file's base failure is the missing
#: WRITER, not a missing import: every assertion below is about whether the
#: cause reaches a surface, and the vocabulary test lives in
#: ``tests/unit/harness/test_loop.py``.
CUT_CAUSE = "continuation-limit"


async def _absorb(_event: Any) -> None:
    return None


class _FakeJobs:
    """The slice of ``AsyncJobManager`` comms reads (row lookup only)."""

    def __init__(self) -> None:
        self.rows: dict[str, Any] = {}

    def get(self, job_id: str, **_kwargs):
        return self.rows.get(job_id)


class _FakeParent:
    """A parent session as comms sees it: a job manager, and nothing else here."""

    def __init__(self) -> None:
        self.jobs = _FakeJobs()


def _comms() -> SubagentComms:
    return SubagentComms(_FakeParent())  # type: ignore[arg-type]


def _relay(collector: dict[str, Any]) -> Any:
    job = SimpleNamespace(trajectory=[])
    return _make_relay(
        "job-1",
        "child",
        job,
        AsyncJobManager(),
        _absorb,
        lambda _progress: None,
        collector,
    )


def test_the_relay_carries_the_childs_own_classification() -> None:
    """The child has ALREADY decided; the parent must not re-decide from ``error``.

    ``Session._classify_cut_off`` derives the involuntary verdict from the frame
    the child emitted, so the relay's job is to carry it. Re-deriving it here
    from ``error`` would answer the same question in a second place — the drift
    this taxonomy exists to remove.
    """
    final: dict[str, Any] = {"text": "", "error": None}
    relay = _relay(final)

    asyncio.run(
        relay(
            AgentEndEvent(
                messages=[],
                aborted=False,
                error="the turn was cut off (continuation-limit)",
                cut_off="the turn reached its continuation limit with work still queued",
                cut_off_cause=CUT_CAUSE,
            )
        )
    )

    assert final["error"]
    assert final["cut_off_cause"] == CUT_CAUSE
    assert final["cut_off"], "the rendered sentence must ride along"


def test_the_relay_reports_no_cause_for_a_clean_child_end() -> None:
    """The negative arm on the relay: a normal child end stays bare."""
    final: dict[str, Any] = {"text": "", "error": None}
    relay = _relay(final)

    asyncio.run(relay(AgentEndEvent(messages=[], aborted=False)))

    assert final["error"] is None
    assert final.get("cut_off_cause", "") == ""
    assert final.get("cut_off", "") == ""


@pytest.mark.asyncio
async def test_publishing_a_cut_off_child_stamps_the_row_the_event_and_the_record() -> None:
    """One terminal fact, three surfaces — all of them must carry the cause.

    The job ROW is what the panel and ``jobs.list()`` read (and it is the word
    "cut off" the panel already renders), the EVENT is what the parent's stream
    sees, and the comms RECORD is the durable half that outlives the swept row.
    A writer that stamped only the event would leave ``hub op='list'`` unable to
    say why a child stopped once its row aged out.
    """
    comms = _comms()
    comms.record_launch("job-1", "child", prompt="go")
    job = SimpleNamespace(status="running", error_text=None, result_text=None, cut_off_cause="")
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    await _publish_terminal_outcome(
        comms,
        emit,
        job=job,
        job_id="job-1",
        label="child",
        status="failed",
        error_text="the turn was cut off",
        cut_off_cause=CUT_CAUSE,
        cut_off="the turn reached its continuation limit with work still queued",
    )

    assert job.cut_off_cause == CUT_CAUSE
    [end] = [e for e in events if isinstance(e, SubagentEndEvent)]
    assert end.cut_off_cause == CUT_CAUSE
    assert end.cut_off
    assert end.status == "failed"
    stored = comms._record("job-1")
    assert stored is not None and stored.cut_off_cause == CUT_CAUSE


@pytest.mark.asyncio
async def test_publishing_a_clean_completion_carries_no_cause() -> None:
    """The discriminating negative for the writer.

    A writer that stamped a cause unconditionally — or that copied a stale one
    off the child — would make every successful child read as cut off.
    """
    comms = _comms()
    comms.record_launch("job-2", "child", prompt="go")
    job = SimpleNamespace(status="running", error_text=None, result_text=None, cut_off_cause="")
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    await _publish_terminal_outcome(
        comms,
        emit,
        job=job,
        job_id="job-2",
        label="child",
        status="completed",
        result_text="all done",
    )

    assert job.cut_off_cause == ""
    [end] = [e for e in events if isinstance(e, SubagentEndEvent)]
    assert end.cut_off_cause == ""
    assert end.cut_off == ""
    stored = comms._record("job-2")
    assert stored is not None and stored.cut_off_cause == ""


def test_the_restored_row_word_is_cut_off_while_a_clean_stop_keeps_its_own() -> None:
    """The panel already renders the cause — no UI change is needed for a RESTORE.

    A restored row carries ``interrupted`` (the only word the roster has for a
    run that never settled), and ``status_glyph(..., cut_off=True)`` is the
    existing branch that splits that word from the deliberate-stop one. Pinning
    it here proves the row a parent reads after a restart says "cut off" once
    ``restored_job_row`` stops relabelling a recorded cause as ``owner-lost``.

    What this does NOT cover, deliberately: a child the loop cut off LIVE now
    settles ``failed`` with the cause on its row, and the panel renders that row
    as "failed". Extending the ``cut off`` word to a failed row is a real
    user-visible change and belongs to a design round, not to this fix — the
    cause on the row is what such a round would render, and it is on the row.
    """
    from local_operator.tui.widgets.subagent_panel import status_glyph

    _glyph, word, _tone = status_glyph("interrupted", cut_off=True)
    assert word == "cut off"
    # The control: a deliberate stop keeps its own word.
    _glyph, deliberate, _tone = status_glyph("interrupted")
    assert deliberate == "interrupted"


@pytest.mark.asyncio
async def test_the_roster_reports_why_a_settled_child_stopped() -> None:
    """``hub op='list'`` must be able to say WHY, after the job row is swept."""
    comms = _comms()
    comms.record_launch("job-3", "child", prompt="go")
    comms.record_outcome("job-3", "failed", error_text="cut off", cut_off_cause=CUT_CAUSE)

    [row] = [r for r in comms.roster() if r.job_id == "job-3"]
    assert row.cut_off_cause == CUT_CAUSE
    assert row.status == "failed"


class _AttachedChild:
    """The child handle ``comms.attach`` stores; nothing here calls into it.

    Signature-compatible with ``ChildSession`` so the type checker agrees this
    is a stand-in for the protocol rather than an accidental one.
    """

    def subscribe(self, handler: Callable[[AgentEvent], Any]) -> Callable[[], None]:
        return lambda: None

    def queue_aside(self, thunk: Callable[[], Any]) -> None:
        return None

    def steer_message(self, message: Message) -> None:
        return None


def test_the_cause_survives_a_snapshot_restore_round_trip() -> None:
    """The durable half: the record outlives the row, so the cause must persist.

    Without this, a parent reconnecting to a resumed session sees a child that
    stopped for no recorded reason — exactly the state the cause exists to
    remove.
    """
    comms = _comms()
    comms.record_launch("job-4", "child", prompt="go")
    comms.attach("job-4", cast(Any, _AttachedChild()), Path("/tmp/child-4"))
    comms.record_outcome("job-4", "failed", cut_off_cause=CUT_CAUSE)

    payload = comms.snapshot()
    assert payload[0]["cut_off_cause"] == CUT_CAUSE

    restored = _comms()
    restored.restore(payload)
    record = restored._record("job-4")
    assert record is not None and record.cut_off_cause == CUT_CAUSE


def test_restored_rows_prefer_a_recorded_cause_over_owner_lost() -> None:
    """A row restored after a restart must not relabel a known cause.

    ``owner-lost`` is the blanket answer for "the process under it ended"; a
    child the loop itself cut off has a SPECIFIC recorded cause, and relabelling
    it would send a reader looking for a dead runtime that never died.
    """
    from local_operator.harness.jobs import AsyncJob
    from local_operator.session.restored_rows import restored_job_row

    job = AsyncJob.model_construct(
        id="job-5", type="task", label="child", status="running", restored=False, cut_off_cause=""
    )
    record = {
        "job_id": "job-5",
        "label": "child",
        "outcome": None,
        "session_dir": None,
        "cut_off_cause": CUT_CAUSE,
    }
    row = restored_job_row(job, record)
    assert row.cut_off_cause == CUT_CAUSE
    assert row.status == "interrupted"

    # The negative arm: a record with no cause keeps today's blanket answer.
    bare = dict(record, cut_off_cause="")
    assert restored_job_row(job, bare).cut_off_cause == "owner-lost"


@pytest.mark.asyncio
async def test_a_child_cut_off_by_its_own_loop_settles_named_and_resumable(
    tmp_path, monkeypatch
) -> None:
    """END TO END: the reported scenario, through the production launch path.

    A child whose loop hits the continuation guard stops mid-flight. The parent
    must see ``failed`` WITH the cause on the event and the row, and the roster
    must offer it as resumable with that cause — never as a completed child that
    simply stopped without committing.

    The guard is tripped the way a parent trips it in production: by sending
    notes faster than the child consumes them (``Session._drain_asides``, on the
    even poll, so one note lands on each yield boundary — the inner loop's own
    inflight drain takes the odd ones).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    class _OneTurn:
        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            async def gen():
                yield StreamTextDelta(delta="working")
                yield StreamEndEvent(stop_reason="stop")

            return gen()

    transcript = Transcript(tmp_path / "sess")
    parent = Session(
        model=MODEL,
        stream_fn=_OneTurn(),
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable"],
    )

    polls = {"n": 0}

    async def flooding_asides(self: Session):
        polls["n"] += 1
        if polls["n"] % 2 == 0 and polls["n"] <= 60:
            return [Message.user(f"parent note {polls['n'] // 2}")]
        return []

    monkeypatch.setattr(Session, "_drain_asides", flooding_asides)

    events: list[AgentEvent] = []
    parent.subscribe(events.append)
    job_id = parent._launch_subagent(label="stuck", prompt="go do a thing")

    async def _settled() -> bool:
        row = parent.jobs.get(job_id)
        return row is not None and row.status != "running"

    for _ in range(4000):
        if await _settled():
            break
        await asyncio.sleep(0.01)
    else:  # pragma: no cover - a wedge, not a slow machine
        raise AssertionError("the child never settled")

    row = parent.jobs.get(job_id)
    assert row is not None
    [end] = [e for e in events if isinstance(e, SubagentEndEvent)]
    assert end.status == "failed", end.status
    assert end.cut_off_cause == CUT_CAUSE, (end.cut_off_cause, end.error_text)
    assert end.cut_off
    assert row.cut_off_cause == CUT_CAUSE

    [child_row] = [r for r in parent.subagent_comms.roster() if r.job_id == job_id]
    assert child_row.cut_off_cause == CUT_CAUSE
    assert child_row.status == "failed"
    assert child_row.resumable

    await parent.dispose()


@pytest.mark.asyncio
async def test_a_cleanly_finishing_child_carries_no_cause_end_to_end(tmp_path, monkeypatch) -> None:
    """The end-to-end negative arm.

    A child that simply answers must settle ``completed`` with no cause on the
    event or the row — otherwise every successful delegation would read as cut
    off, which is worse than the bug this change fixes.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    class _OneTurn:
        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            async def gen():
                yield StreamTextDelta(delta="child did the work")
                yield StreamEndEvent(stop_reason="stop")

            return gen()

    parent = Session(
        model=MODEL,
        stream_fn=_OneTurn(),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
    )
    events: list[AgentEvent] = []
    parent.subscribe(events.append)
    job_id = parent._launch_subagent(label="fine", prompt="just answer")

    async def _settled() -> bool:
        row = parent.jobs.get(job_id)
        return row is not None and row.status != "running"

    for _ in range(4000):
        if await _settled():
            break
        await asyncio.sleep(0.01)
    else:  # pragma: no cover
        raise AssertionError("the child never settled")

    row = parent.jobs.get(job_id)
    assert row is not None
    [end] = [e for e in events if isinstance(e, SubagentEndEvent)]
    assert end.status == "completed"
    assert end.cut_off_cause == ""
    assert end.cut_off == ""
    assert row.cut_off_cause == ""

    await parent.dispose()


def test_a_start_event_still_validates_without_the_new_fields() -> None:
    """Additivity: an old viewer keeps validating an end event it never knew.

    ``AgentEvent`` is ``extra="allow"`` and the fields default to ``""``, which
    is also what an OLD child runtime produces — that is the whole backwards
    compatibility story, and the reason no ``PROTOCOL_VERSION`` bump is needed.
    """
    from local_operator.harness.types import SubagentStartEvent as _Start

    assert SubagentEndEvent(job_id="j", label="l", status="completed").cut_off_cause == ""
    assert SubagentEndEvent(job_id="j", label="l", status="completed").cut_off == ""
    assert _Start(job_id="j", label="l", agent_id="a", model="m").type == "subagent_start"
