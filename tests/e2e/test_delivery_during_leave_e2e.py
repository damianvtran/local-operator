"""A settled child's result must survive an exit that has already committed.

THE OPERATOR'S REPORT (session ``a81ceec0982b``, 2026-09-24 22:22): a turn
completed, its work was done and merged, and the operator still got a red
"Stopped with an error" card plus a ``[session incident]`` telling the next turn
"the runtime was cut off before this turn produced a result ... do not assume the
request completed" — which then cost a turn re-verifying finished work.

The ordering, read off the durable rows: the finishing turn's ``finally`` flushed
the nine results it had deferred during that turn, which opened ONE delivery turn
(one ``attention_started``, nine ``job_result`` rows in 26 ms). The runtime had
already committed to leaving, so the disposal that followed aborted that turn
before its first provider call — a ``kind=error cause=disposed`` row 650 ms after
the operator's own honest ``complete``.

WHY THE TURN COULD OPEN AT ALL: ``begin_drain``/``begin_retire`` refuse ``prompt``
and ``receive_peer_message`` — the two paths a CLIENT reaches. A job result is
harness-initiated through ``Session._deliver_job_results``, so nothing refused it
and "invariant (i), no new work after the commit" was silently false for the
third arrival.

These cells drive the REAL ordering over a real ``Session`` behind the real
``ServingSessionHandle``: a turn is in flight, children settle into it, the
departure latch is taken, and only THEN does the turn end and flush. The first
cell asserts the whole story the operator reads — no error row, no incident row,
one run for the work turn, the results durable across the exit, and the
successor's next real turn carrying them. The second is the NEGATIVE CONTROL the
fix cannot ship without: the same script WITHOUT the latch must still deliver the
batch as exactly one turn. Consult the latch on the wrong side and every child's
report in the fleet is held instead of delivered, which is the bug the deferral
path exists to fix.

Isolation: ``headless_tui_env`` redirects the config dir and the root conftest
redirects ``HOME``; no ``CMUX_*`` variable is touched or read, and every id here
is synthetic.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.jobs import JOB_RESULT_MESSAGE_TYPE, AsyncJob
from local_operator.harness.types import AbortSignal, ChatRequest, StreamEvent
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from tests.e2e.harness import (
    ScriptedStream,
    _user_row_texts,
    build_session,
    last_user_text,
    text_turn,
)
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

#: How long the cells will wait for something the production loop does on its
#: own. Generous relative to the real cost (a scripted turn is milliseconds in
#: the driver's own loop) and short enough that a genuinely stuck step reports
#: through the assertion rather than through the watchdog.
_STEP_TIMEOUT_S = 30.0


class _ParkedWorkTurn(ScriptedStream):
    """The FIRST provider call parks; every later call is scripted as usual.

    The cell is an ORDERING question — the child has to settle while the parent
    turn is genuinely in flight (that is what defers it) and the departure latch
    has to be taken before that turn ends (that is what the incident measured) —
    so holding the first call open is what makes both facts arranged rather than
    hoped for. ``entered`` proves the turn reached the provider, which is the same
    evidence ``session.is_streaming`` gives and the only moment the fixture can
    act from.
    """

    #: The request whose call is parked. Matched on CONTENT, not on call index:
    #: a session may spend a provider call naming its conversation before the
    #: turn proper, and a cell gated on "call 1" would then park that call and
    #: test an ordering nobody arranged (``last_user_text`` is the harness's own
    #: reader for "what is this call asking").
    park_marker = "delegate two children"

    def __init__(self, turns: Sequence[Sequence[StreamEvent]]) -> None:
        super().__init__(turns)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    def __call__(
        self, request: ChatRequest, signal: AbortSignal | None = None
    ) -> AsyncIterator[StreamEvent]:
        inner = super().__call__(request, signal)
        park = self.park_marker in last_user_text(request)

        async def gen() -> AsyncIterator[StreamEvent]:
            if park:
                self.entered.set()
                await self.release.wait()
            async for event in inner:
                yield event

        return gen()


def _settled(job_id: str) -> AsyncJob:
    """A child that has settled, as the manager hands it to the settle hook."""
    return AsyncJob(
        id=job_id,
        type="task",
        status="completed",
        label=job_id,
        start_time=1.0,
        result_text=f"{job_id} done",
    )


def _lines(directory: Path) -> list[dict[str, Any]]:
    """Every readable transcript row. A half-written tail is skipped, not fatal."""
    path = directory / "transcript.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue  # the writer may still be appending
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _rows(directory: Path, custom_type: str) -> list[dict[str, Any]]:
    """The rows of one custom type, read from DISK — the store the successor sees."""
    return [
        dict(row.get("payload", {}).get("details") or {})
        for row in _lines(directory)
        if row.get("type") == "custom"
        and isinstance(row.get("payload"), dict)
        and row["payload"].get("custom_type") == custom_type
    ]


def _run_rows(directory: Path) -> list[dict[str, Any]]:
    """``attention_started`` rows: one per RUN this conversation opened."""
    return _rows(directory, "attention_started")


def _completion_rows(directory: Path) -> list[dict[str, Any]]:
    """``completion_attention`` rows: the outcomes the operator's card reads."""
    return [row for row in _rows(directory, "completion_attention") if row.get("kind")]


def _job_result_rows(directory: Path) -> list[dict[str, Any]]:
    """The durable ``job_result`` message rows, in transcript order."""
    return [
        dict(row.get("payload", {}).get("details") or {})
        for row in _lines(directory)
        if row.get("type") == "message"
        and isinstance(row.get("payload"), dict)
        and row["payload"].get("custom_type") == JOB_RESULT_MESSAGE_TYPE
    ]


def _incident_rows(directory: Path) -> list[dict[str, Any]]:
    """``session_incident`` rows — what the next turn is handed as its context."""
    return [
        dict(row.get("payload", {}).get("details") or {})
        for row in _lines(directory)
        if row.get("type") == "message"
        and isinstance(row.get("payload"), dict)
        and row["payload"].get("custom_type") == "session_incident"
    ]


async def _work_turn_with_deferred_children(
    config: Path, directory: Path, *, latch: bool
) -> tuple[Session, _ParkedWorkTurn, asyncio.Task[Any]]:
    """The incident's shape, arranged: a live turn, two deferred children, a latch.

    ``latch`` is the only difference between the cell and its negative control, so
    the two run the same script over the same ordering and differ in exactly the
    fact under test.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stream = _ParkedWorkTurn([text_turn("working"), text_turn("delivery")])
    session = build_session(directory, stream)
    handle = ServingSessionHandle(
        session,
        asyncio.get_running_loop(),
        cwd=str(directory),
        install_gates=False,
        config_dir=config,
    )
    task = asyncio.ensure_future(session.prompt("delegate two children"))
    await asyncio.wait_for(stream.entered.wait(), _STEP_TIMEOUT_S)
    assert session.is_streaming, "the work turn never reached the provider"

    # The children settle DURING the turn, which is the deferral the incident's
    # nine-row backlog came from.
    await session._on_job_completed("qa-r2", "the QA round finished", _settled("qa-r2"))
    await session._on_job_completed("rev-r6", "the review round finished", _settled("rev-r6"))
    assert session._deferred_job_results, "precondition: the batch is deferred"

    if latch:
        # THE DEPARTURE, committed while the work turn is still in flight and
        # therefore before the flush that follows it.
        assert handle.begin_drain("runtime-retired", "(0.62.31 → 0.62.32)") is True
        assert session._leaving_deliveries is True, "the latch has to reach the session"
        # A result that lands AFTER the latch: held AT ARRIVAL, so its row is
        # written by the holding arm WITH the marker (U6). The two settles
        # above journaled their rows at settle time -- before the latch
        # existed -- and keep the delivered shape; that contrast is the point
        # of the extra arrival, and it is what keeps the marker's end-to-end
        # coverage after the early write moved the marker's line.
        await session._on_job_completed("late-r7", "the late report", _settled("late-r7"))

    stream.release.set()
    await asyncio.wait_for(asyncio.shield(task), timeout=_STEP_TIMEOUT_S)
    return session, stream, task


@pytest.mark.asyncio
async def test_a_settled_batch_survives_an_exit_that_already_committed(
    headless_tui_env: Path,
) -> None:
    """The incident, end to end: the rows live, the conversation owes no error.

    Four things are asserted, and each is a surface the operator reads: the run
    count (no run was opened for the delivery), the outcome rows (no ``error``),
    the incident rows (the next turn is not told finished work was cut off), and
    the successor's own provider request (the results reach the model).
    """
    config = headless_tui_env
    directory = config / "delivery-while-leaving"

    with bounded(120, "a settled batch on a runtime that has committed to leaving"):
        session, stream, _task = await _work_turn_with_deferred_children(
            config, directory, latch=True
        )
        assert len(stream.requests) == 1, "the held batch must not have bought a turn"
        assert [row.get("job_id") for row in _job_result_rows(directory)] == [
            "qa-r2",
            "rev-r6",
            "late-r7",
        ], (
            "every result is durable: the two mid-turn settles journal at settle "
            "time, and the post-latch arrival is written by the holding arm"
        )
        # ...and the exit runs, over the real disposal rung the drain leads to.
        await session.dispose()

    runs = _run_rows(directory)
    assert len(runs) == 1, f"only the work turn may have opened a run: {runs!r}"
    assert [row.get("kind") for row in _completion_rows(directory)] == [
        "complete"
    ], f"the conversation owes exactly one honest outcome: {_completion_rows(directory)!r}"
    assert (
        _incident_rows(directory) == []
    ), "the next turn must not be told its finished work was cut off"
    held = _job_result_rows(directory)
    assert [row.get("job_id") for row in held] == [
        "qa-r2",
        "rev-r6",
        "late-r7",
    ], f"all three results must be durable, in settle order, after the exit: {held!r}"

    # THE SUCCESSOR: a real boot over the same directory, running a real turn.
    # This is the half that makes "durable" mean "the model sees it".
    successor_stream = ScriptedStream([text_turn("carrying on")])
    successor = build_session(directory, successor_stream)
    await successor.async_init()
    try:
        await successor.prompt("continue")
        assert successor_stream.requests, "the successor never called the provider"
        sent = "\n".join(
            text for request in successor_stream.requests for text in _user_row_texts(request)
        )
        assert "the QA round finished" in sent, sent[-2000:]
        assert "the review round finished" in sent, sent[-2000:]
        assert "the late report" in sent, sent[-2000:]
        assert (
            "[session incident]" not in sent
        ), "a successor that re-verifies finished work is the wasted turn this fixes"
    finally:
        await successor.dispose()
    # U7, ON THE REAL FLOW (review round 2): the held marker is written once and
    # cleared nowhere, so the notice is painted on every later replay — and the
    # first wording claimed "no turn has read it yet", which after THIS turn is a
    # lie. The successor has now read and answered the reports; the notice is
    # recomposed here from the DURABLE rows by the production decision, and it must
    # still be a fact about the arrival rather than a claim about now.
    #
    # TWO ROW SHAPES, and this scenario produces both on purpose: ``late-r7``
    # was held at arrival and carries the marker; ``qa-r2``/``rev-r6``
    # journaled their rows at settle time, before the latch existed, and keep
    # the delivered shape -- the operator watched those arrive, so the marker's
    # first clause ("held when it arrived") would be false on them. The
    # recomposition contract below is asserted on the marked row.
    from local_operator.harness.rows import HELD_DELIVERY_NOTICE, held_delivery_notice

    by_id = {row.get("job_id"): row for row in _job_result_rows(directory)}
    for job_id in ("qa-r2", "rev-r6"):
        assert (
            held_delivery_notice(by_id[job_id]) is None
        ), f"{job_id} arrived while the turn was up and keeps its delivered shape"
    assert by_id["late-r7"].get("held") is True, "held at arrival, the marker must ride it"
    decided = held_delivery_notice(by_id["late-r7"])
    assert decided is not None, "the row is still marked held, which is now harmless"
    text, _severity = decided
    assert "no turn ran for it at that point" in text
    assert (
        "has read it yet" not in text
    ), "after the answering turn, the notice must not still claim nobody read it"
    assert "no turn has read it yet" not in HELD_DELIVERY_NOTICE


@pytest.mark.asyncio
async def test_a_settled_batch_still_opens_one_turn_without_the_latch(
    headless_tui_env: Path,
) -> None:
    """THE NEGATIVE CONTROL: off the latch, the batch is delivered as ONE turn.

    Identical script, no departure. If the gate were consulted anywhere but the
    latch, this cell would report fleet-wide silent under-delivery instead of a
    delivered batch — so it is also what stops the fix from being written on the
    wrong side of the condition.
    """
    config = headless_tui_env
    directory = config / "delivery-without-a-departure"

    with bounded(120, "a settled batch with no departure"):
        session, stream, task = await _work_turn_with_deferred_children(
            config, directory, latch=False
        )
        assert session._leaving_deliveries is False, "precondition: no latch is taken"
        # The delivery turn is spawned by the flush; give the loop until it has
        # spent its one provider call.
        deadline = time.monotonic() + _STEP_TIMEOUT_S
        while len(stream.requests) < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        await asyncio.wait_for(task, timeout=_STEP_TIMEOUT_S)
        await session.dispose()

    assert (
        len(stream.requests) == 2
    ), f"one work turn plus ONE batched delivery turn: {len(stream.requests)}"
    delivered = "\n".join(text for text in _user_row_texts(stream.requests[1]))
    assert (
        "the QA round finished" in delivered and "the review round finished" in delivered
    ), f"both results ride the same turn: {delivered[-2000:]}"
    assert len(_run_rows(directory)) == 2, "the delivery turn opened its own run"
    # ONE ROW PER RESULT, re-asserted AFTER the delivery turn ran. The rows
    # above are written at settle time; the delivery turn then hands those same
    # messages back as its initials, and the pipeline's incoming-journal loop
    # must not journal them a second time (the QA probe on this PR measured two
    # rows becoming FOUR, same ids, until the loop learned to skip a durable
    # row). The latch cell cannot see this -- its batch is held, not delivered
    # -- so this is the only cell that runs the loop over pre-durable initials.
    rows = _job_result_rows(directory)
    assert [row.get("job_id") for row in rows] == ["qa-r2", "rev-r6"], (
        "one row per result after the delivery turn: the loop must skip rows "
        f"already made durable at settle time (found {[r.get('job_id') for r in rows]})"
    )


# --- 664a234ec561 (2026-09-28): the exit that arms NO latch --------------------
#
# The same handoff as a81ceec0982b, through the door neither departure rung
# covers: an exec/print-mode run ending NORMALLY, then its process disposing.
# Nothing arms ``begin_drain``/``begin_retire`` on that path, so the batch a
# last turn deferred was flushed into a delivery run that the disposal aborted
# 13 ms after the completed turn's own honest marker — a second
# ``error | cause=disposed`` row that superseded ``complete`` for every
# latest-wins reader and made the successor journal a ``[session incident]``
# about work that had finished. The fleet shows the tail
# ``complete → attention_started → error|disposed`` in >=5 exec sessions across
# three days, every one of them with ZERO provider round-trips in the aborted
# run.
#
# The cells below pin both halves of the fix, from the ordering 664a had:
# a disposal that catches a live run publishes a cut-off ONLY when the run has
# evidence (a provider round-trip was dispatched, or a person's/peer's prompt
# opened it); otherwise the run settles with an ``eligible:False`` marker and
# NO verdict. And the one-shot host arms the same departure pair before its
# dispose, so the doomed turn is held instead of opened.


def _park_runs_before_dispatch(
    session: Session, *, except_token: str | None
) -> tuple[asyncio.Event, asyncio.Event]:
    """Park the first run whose token is not ``except_token`` BEFORE dispatch.

    ``_prepare_system_blocks`` runs inside ``_run_turn`` after the run's abort
    signal exists and before any request is built or handed to the provider,
    so parking here is exactly the 664a state: the run is ADMITTED — its
    token is minted, its ``attention_started`` row is written, ``_turn_task``
    is live — and it has spent no provider round-trip. ``except_token`` lets
    the WORK run through untouched so the cell can park the DELIVERY run
    alone.
    """
    parked = asyncio.Event()
    release = asyncio.Event()
    original = session._prepare_system_blocks
    fired = False

    async def gated(*args: Any, **kwargs: Any) -> Any:
        nonlocal fired
        if (
            not fired
            and session._attention_run_token is not None
            and session._attention_run_token != except_token
        ):
            fired = True
            parked.set()
            await release.wait()
        return await original(*args, **kwargs)

    session._prepare_system_blocks = gated  # type: ignore[method-assign]
    return parked, release


class _ParkedDeliveryTurn(_ParkedWorkTurn):
    """``_ParkedWorkTurn`` plus an optional park on the DELIVERY turn's call.

    With ``hold_delivery`` the delivery request is parked INSIDE the provider
    call (its round-trip was dispatched — the negative control's shape);
    without it the delivery run is only ever exercised pre-dispatch.
    """

    #: Every batch row's text — the LAST row is what ``last_user_text`` returns,
    #: so matching only the first one never fired (the batch carries both
    #: children, and the review row is the tail). Annotated because cells
    #: override it per-case with shorter tuples (the peer and mid-work cells
    #: park on a single marker); the inferred ``tuple[str, str]`` would refuse
    #: those assignments.
    delivery_markers: tuple[str, ...] = ("the QA round finished", "the review round finished")

    def __init__(self, turns: Sequence[Sequence[StreamEvent]], *, hold_delivery: bool) -> None:
        super().__init__(turns)
        self.hold_delivery = hold_delivery
        self.delivery_entered = asyncio.Event()

    def __call__(
        self, request: ChatRequest, signal: AbortSignal | None = None
    ) -> AsyncIterator[StreamEvent]:
        base = super().__call__(request, signal)
        park_delivery = self.hold_delivery and any(
            marker in last_user_text(request) for marker in self.delivery_markers
        )

        async def gen() -> AsyncIterator[StreamEvent]:
            if park_delivery:
                self.delivery_entered.set()
                # Released only by the abort cancelling the pump, like the work
                # turn's park: the cell never lets a parked delivery turn finish.
                await asyncio.Event().wait()
            async for event in base:
                yield event

        return gen()


async def _work_turn_with_deferred_children_unlatched(
    config: Path,
    directory: Path,
    stream: _ParkedDeliveryTurn,
    *,
    gate_flushed_run: bool,
) -> tuple[Session, tuple[asyncio.Event, asyncio.Event] | None]:
    """The 664a script: a live work turn, two deferred children, NO latch.

    No latch is the exec condition — nothing arms the departure pair here.
    With ``gate_flushed_run`` the helper parks whatever run the flush spawns
    BEFORE dispatch (the record cell's admitted-but-zero-work state); without
    it the flushed run is left to reach the provider on its own (the negative
    control's shape). The gate has to be installed before the work turn is
    released, because the flush — and the spawn it causes — run in that turn's
    own tail.
    """
    directory.mkdir(parents=True, exist_ok=True)
    session = build_session(directory, stream)
    task = asyncio.ensure_future(session.prompt("delegate two children"))
    await asyncio.wait_for(stream.entered.wait(), _STEP_TIMEOUT_S)
    assert session.is_streaming, "the work turn never reached the provider"
    gate: tuple[asyncio.Event, asyncio.Event] | None = None
    if gate_flushed_run:
        work_token = session._attention_run_token
        assert work_token, "the work run must have minted its token by admission"
        gate = _park_runs_before_dispatch(session, except_token=work_token)

    await session._on_job_completed("qa-r2", "the QA round finished", _settled("qa-r2"))
    await session._on_job_completed("rev-r6", "the review round finished", _settled("rev-r6"))
    assert session._deferred_job_results, "precondition: the batch is deferred"

    stream.release.set()
    await asyncio.wait_for(asyncio.shield(task), timeout=_STEP_TIMEOUT_S)
    return session, gate


@pytest.mark.asyncio
async def test_a_zero_work_disposal_settles_without_a_verdict(
    headless_tui_env: Path,
) -> None:
    """THE 664a RECORD CELL: an admitted, zero-work run leaves NO verdict.

    The first cell of this file required a departure latch to keep the batch
    from opening a turn at all. This one is the door that latch cannot reach:
    the delivery run IS admitted (its own ``attention_started`` exists) and the
    disposal meets it pre-dispatch. Nothing was spent and no person asked, so
    the exit settles the run — an ``eligible:False`` marker, no store row, no
    successor narration — instead of a ``disposed`` error that supersedes the
    work turn's honest ``complete``.
    """
    config = headless_tui_env
    directory = config / "zero-work-disposal"

    with bounded(120, "a zero-work delivery run caught by the disposal"):
        stream = _ParkedDeliveryTurn(
            [text_turn("working"), text_turn("delivery")], hold_delivery=False
        )
        session, gate = await _work_turn_with_deferred_children_unlatched(
            config, directory, stream, gate_flushed_run=True
        )
        assert gate is not None, "the record cell must park the flushed run"
        parked, release = gate
        # The flush spawned the delivery task; it is ADMITTED (its own run
        # token, its own attention_started row, a live _turn_task) and parked
        # exactly where 664a's run was when the disposal arrived.
        await asyncio.wait_for(parked.wait(), _STEP_TIMEOUT_S)
        delivery_token = session._attention_run_token
        assert delivery_token is not None
        assert (
            session._turn_task is not None and not session._turn_task.done()
        ), "the gate must park a LIVE run, not a finished one"

        dispose_task = asyncio.ensure_future(session.dispose())
        while not (session._signal is not None and session._signal.aborted):
            # The disposal decides and aborts BEFORE awaiting the run; release
            # the park only after the abort fired so the run unwinds through
            # the pre-aborted fast path (zero provider statements executed).
            await asyncio.sleep(0.005)
        release.set()
        await asyncio.wait_for(dispose_task, timeout=_STEP_TIMEOUT_S)

    runs = _run_rows(directory)
    assert len(runs) == 2, f"the work run and the admitted delivery run: {runs!r}"
    assert runs[-1].get("token") == delivery_token
    assert [row.get("kind") for row in _completion_rows(directory)] == [
        "complete"
    ], f"no verdict may replace the completed turn's own: {_completion_rows(directory)!r}"
    from local_operator.session.attention import AttentionStore, conversation_identity

    state = AttentionStore().state(conversation_identity(directory))
    assert state.get("kind") == "complete", f"the store's latest row stays complete: {state!r}"

    markers = [
        row
        for row in _rows(directory, "completion_attention")
        if row.get("token") == delivery_token
    ]
    assert len(markers) == 1, f"one settle marker for the delivery run: {markers!r}"
    marker = markers[0]
    assert marker.get("eligible") is False, marker
    assert not marker.get("kind") and not marker.get(
        "cause"
    ), "the settlement asserts NOTHING about how the run ended"
    assert (
        _incident_rows(directory) == []
    ), "the next turn must not be told a zero-work delivery run was cut off"

    # THE SUCCESSOR: a real boot over the same directory. It must narrate
    # nothing AND still carry the two durable results.
    successor_stream = ScriptedStream([text_turn("carrying on")])
    successor = build_session(directory, successor_stream)
    await successor.async_init()
    try:
        await successor.prompt("continue")
        assert successor_stream.requests, "the successor never called the provider"
        sent = "\n".join(
            text for request in successor_stream.requests for text in _user_row_texts(request)
        )
        assert "the QA round finished" in sent, sent[-2000:]
        assert "the review round finished" in sent, sent[-2000:]
        assert "[session incident]" not in sent, sent[-2000:]
    finally:
        await successor.dispose()


@pytest.mark.asyncio
async def test_a_noted_serving_cause_still_settles_a_zero_evidence_run(
    headless_tui_env: Path,
) -> None:
    """THE SERVING DOOR: a cause noted before the handover must not dodge the settle.

    ``ServingSessionHandle.dispose()`` notes the retirement cause whenever
    ``disposal_cuts_a_turn()`` is true — the same condition the disposal's own
    settle decision uses — so the synthesis arrives holding BOTH a noted cause
    AND an armed settle intent. Review round 1 (MAJOR-1) measured what that
    combination did: the synthesis took the publish branch, the armed intent
    suppressed that publish, and the run ended settled by NOBODY — no verdict,
    no ``eligible:False`` marker — which the successor reclassified as "the
    cause could not be determined" about a turn that spent nothing. With the
    intent forcing the settle branch, the serving door writes the marker and
    the successor narrates nothing.
    """
    config = headless_tui_env
    directory = config / "serving-note-settles"

    with bounded(120, "a noted serving cause over a zero-evidence run"):
        stream = _ParkedDeliveryTurn(
            [text_turn("working"), text_turn("delivery")], hold_delivery=False
        )
        session, gate = await _work_turn_with_deferred_children_unlatched(
            config, directory, stream, gate_flushed_run=True
        )
        assert gate is not None, "the cell must park the flushed run"
        parked, release = gate
        await asyncio.wait_for(parked.wait(), _STEP_TIMEOUT_S)
        delivery_token = session._attention_run_token
        assert delivery_token is not None
        assert session._turn_task is not None and not session._turn_task.done()

        # Exactly ``ServingSessionHandle._note_retirement_cut_off()``'s call and
        # gate (serving.py:1735-1739): note only while a live turn is being cut.
        assert session.disposal_cuts_a_turn(), "the serving rung notes under this gate"
        session.note_cut_off("runtime-shutdown")

        dispose_task = asyncio.ensure_future(session.dispose())
        while not (session._signal is not None and session._signal.aborted):
            await asyncio.sleep(0.005)
        release.set()
        await asyncio.wait_for(dispose_task, timeout=_STEP_TIMEOUT_S)

    assert (
        session._attention_run_settled is True
    ), "the run must not be left open for the successor to reclassify"
    rows = _completion_rows(directory)
    assert [row.get("kind") for row in rows] == [
        "complete"
    ], f"the noted cause must not become a verdict for a zero-evidence run: {rows!r}"
    markers = [
        row
        for row in _rows(directory, "completion_attention")
        if row.get("token") == delivery_token
    ]
    assert len(markers) == 1, f"one settle marker for the delivery run: {markers!r}"
    assert markers[0].get("eligible") is False, markers[0]
    from local_operator.session.attention import AttentionStore, conversation_identity

    state = AttentionStore().state(conversation_identity(directory))
    assert state.get("kind") == "complete", f"the store stays on the completed turn: {state!r}"
    assert _incident_rows(directory) == []

    # THE SUCCESSOR must narrate nothing: the "cause could not be determined"
    # reclassification is exactly what the missing marker used to produce.
    successor_stream = ScriptedStream([text_turn("carrying on")])
    successor = build_session(directory, successor_stream)
    await successor.async_init()
    try:
        await successor.prompt("continue")
        sent = "\n".join(
            text for request in successor_stream.requests for text in _user_row_texts(request)
        )
        assert "[session incident]" not in sent, sent[-2000:]
    finally:
        await successor.dispose()


@pytest.mark.asyncio
async def test_a_post_completion_delivery_cut_mid_request_settles_without_a_verdict(
    headless_tui_env: Path,
) -> None:
    """THE CASE-3 CELL (v3, 2026-09-30): cut AFTER dispatch — still no verdict.

    This replaced the v2 control ``…_still_errors`` for the delivery shape,
    because that shape IS the case-3 misclassification: a delivery run admitted
    6-14 ms after the completed turn's output was delivered, parked INSIDE its
    provider call (its request WAS dispatched — the fact v2's split keyed on),
    and cut by the one-shot host's own disposal. v2 could only publish
    ``error|disposed`` for it, which masked the completion on nine exec
    sessions (da0d927a7986, cc42557b5571, e4e0dfa96358, ...). The disposal
    now reads the run's POST-COMPLETION provenance (admitted after a settled
    ``complete``) and its PRODUCTION (nothing persisted — the empty tail
    cannot arm it) and settles the run with no verdict at all: ``eligible:
    False``, the completion stands, the successor narrates nothing. The
    honest dispatched-run pin moved to
    ``test_a_mid_work_cut_after_its_first_request_still_errors`` (no settled
    success before it) and to the unit matrix, so this reversal removes no
    guard.
    """
    config = headless_tui_env
    directory = config / "delivery-cut-after-dispatch"

    with bounded(120, "a post-completion delivery run cut mid-request"):
        stream = _ParkedDeliveryTurn(
            [text_turn("working"), text_turn("delivery")], hold_delivery=True
        )
        session, gate = await _work_turn_with_deferred_children_unlatched(
            config, directory, stream, gate_flushed_run=False
        )
        assert gate is None, "this cell must let the flushed run reach the provider"
        await asyncio.wait_for(stream.delivery_entered.wait(), _STEP_TIMEOUT_S)
        assert session.is_streaming, "the delivery turn never reached the provider"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not session._attention_run_request_dispatched:
            await asyncio.sleep(0.005)
        inputs = (
            getattr(session, "_attention_run_request_dispatched", False),
            getattr(session, "_attention_run_after_settled_success", None),
            getattr(session, "_attention_run_produced", None),
        )

        delivery_token = session._attention_run_token
        await session.dispose()

    assert delivery_token is not None
    rows = _completion_rows(directory)
    # THE CLASS EVIDENCE IS CHECKED FIRST, so an unfixed tree fails HERE —
    # with the false row — rather than on a probe attribute.
    assert [row.get("kind") for row in rows] == ["complete"], rows
    # …and the decision inputs the disposal read (``None`` on an unfixed tree).
    assert inputs == (True, True, False), inputs
    markers = [
        row
        for row in _rows(directory, "completion_attention")
        if row.get("token") == delivery_token
    ]
    assert len(markers) == 1, f"one settle marker for the delivery run: {markers!r}"
    assert markers[0].get("eligible") is False, markers[0]
    assert not markers[0].get("kind"), "the settlement asserts no verdict"
    from local_operator.session.attention import AttentionStore, conversation_identity

    state = AttentionStore().state(conversation_identity(directory))
    assert state.get("kind") == "complete", f"the store stays on the completed turn: {state!r}"
    assert _incident_rows(directory) == []

    successor_stream = ScriptedStream([text_turn("carrying on")])
    successor = build_session(directory, successor_stream)
    await successor.async_init()
    try:
        await successor.prompt("continue")
        sent = "\n".join(
            text for request in successor_stream.requests for text in _user_row_texts(request)
        )
        assert "[session incident]" not in sent, sent[-2000:]
    finally:
        await successor.dispose()


@pytest.mark.asyncio
async def test_a_mid_work_cut_after_its_first_request_still_errors(
    headless_tui_env: Path,
) -> None:
    """HONESTY (v3): dispatched work with NO settled success before it errors.

    The v3 arm is scoped by TWO facts the disposal reads (a settled
    ``complete`` behind the run, and nothing produced). This cell removes the
    first — a fresh conversation, its only run already INSIDE its provider
    call — so the arm must not fire and the honest ``error | cause=disposed``
    stands, with the successor narration intact. Without this pin the new arm
    could be widened into "any dispatched disposal settles silently", which
    would hide a genuine mid-work death.
    """
    config = headless_tui_env
    directory = config / "mid-work-cut-after-dispatch"
    directory.mkdir(parents=True, exist_ok=True)
    stream = _ParkedDeliveryTurn([text_turn("working")], hold_delivery=True)
    # The only call is this run's, and it must hold INSIDE the provider: the
    # "dispatched" fact is exactly what the arm must NOT override here.
    stream.delivery_markers = ("do the thing",)
    session = build_session(directory, stream)

    with bounded(120, "a mid-work run cut after its first request"):
        task = asyncio.ensure_future(session.prompt("do the thing"))
        await asyncio.wait_for(stream.delivery_entered.wait(), _STEP_TIMEOUT_S)
        assert session.is_streaming, "the run never reached the provider"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not session._attention_run_request_dispatched:
            await asyncio.sleep(0.005)
        assert session._attention_run_request_dispatched, "precondition: dispatched"
        assert getattr(session, "_attention_run_after_settled_success", None) in (False, None)

        await session.dispose()
        await asyncio.wait_for(asyncio.shield(task), timeout=_STEP_TIMEOUT_S)

    rows = _completion_rows(directory)
    assert [row.get("kind") for row in rows] == ["error"], rows
    assert rows[-1].get("cause") == "disposed", rows[-1]

    successor_stream = ScriptedStream([text_turn("carrying on")])
    successor = build_session(directory, successor_stream)
    await successor.async_init()
    try:
        await successor.prompt("continue")
        sent = "\n".join(
            text for request in successor_stream.requests for text in _user_row_texts(request)
        )
        assert "[session incident]" in sent, sent[-2000:]
    finally:
        await successor.dispose()


@pytest.mark.asyncio
async def test_a_peer_arrival_cut_after_dispatch_closes_neutrally(
    headless_tui_env: Path,
) -> None:
    """THE PEER DOOR (v3): a carried post-completion run gets ``closed``.

    Session 23fc556c3799's shape at the dispatch boundary: the work turn's
    output was delivered, a peer ask was admitted right after, and the one-shot
    disposal cut it while its request was in flight. Carried provenance makes
    the neutral record a ``closed`` receipt rather than silence (someone is
    waiting on it); the settle intent suppresses the abort tail, so exactly one
    outcome row exists for the run.
    """
    config = headless_tui_env
    directory = config / "peer-dispatched-closure"
    directory.mkdir(parents=True, exist_ok=True)
    stream = _ParkedWorkTurn([text_turn("warm up answer"), text_turn("peer answer")])
    stream.park_marker = "peer asks a question"
    session = build_session(directory, stream)

    with bounded(120, "a peer arrival cut after dispatch"):
        await session.prompt("warm up")
        message = session._peer_custom_message(
            "peer asks a question",
            {"pid": 999999, "conversation_name": "peer", "model_label": "m"},
        )
        task = asyncio.ensure_future(session._prompt_messages([message], carried_prompt=True))
        await asyncio.wait_for(stream.entered.wait(), _STEP_TIMEOUT_S)
        assert session.is_streaming, "the peer run never reached the provider"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not session._attention_run_request_dispatched:
            await asyncio.sleep(0.005)
        inputs = (
            getattr(session, "_attention_run_request_dispatched", False),
            getattr(session, "_attention_run_after_settled_success", None),
            getattr(session, "_attention_run_carried_prompt", None),
        )

        await session.dispose()
        await asyncio.wait_for(asyncio.shield(task), timeout=_STEP_TIMEOUT_S)

    rows = _completion_rows(directory)
    # CLASS EVIDENCE FIRST (see the sibling cell): the carried peer run must
    # close, not error, once it followed a settled completion.
    assert [row.get("kind") for row in rows] == ["complete", "closed"], rows
    assert rows[-1].get("cause") == "disposed", rows[-1]
    assert inputs == (True, True, True), inputs


@pytest.mark.asyncio
async def test_a_declared_one_shot_turn_holds_its_final_batch_inside_the_lock(
    headless_tui_env: Path,
) -> None:
    """THE v3 PREVENTION CELL (B): the hold lives in the TURN, not the host.

    The case-3 admission happens at ``_turn_lock``'s release — the very next
    loop turns after ``session.prompt()`` returns — and the host's own arming
    (``run_print_mode``'s, and its predecessor here: anything after the prompt
    returns) lands ~100-160 ms too late for it. This cell is the 664a script
    with the ONE-SHOT DECLARATION: the session arms its departure pair in the
    turn pipeline's ``finally``, before the deferred batch is delivered and
    before the lock frees — so the flush HOLDS the batch inline and no
    delivery task ever spawns. Asserted both ways: no task reaches the
    provider within a bounded beat, the latch is armed by the turn itself, and
    the batch stays durable for the successor. Before the fix the spawn
    happens in the same ``finally`` (nothing armed) and the run it opens
    reaches the provider — this cell's negative is cell
    ``test_a_post_completion_delivery_cut_mid_request_settles_without_a_verdict``,
    which drives that admitted run through the disposal.
    """
    config = headless_tui_env
    directory = config / "one-shot-declared-batch-hold"
    directory.mkdir(parents=True, exist_ok=True)
    stream = _ParkedDeliveryTurn([text_turn("working"), text_turn("delivery")], hold_delivery=True)
    session = build_session(directory, stream)
    declare = getattr(session, "declare_one_shot_exit", None)
    if callable(declare):
        declare()

    with bounded(120, "a declared one-shot turn holds its final batch"):
        task = asyncio.ensure_future(session.prompt("delegate two children"))
        await asyncio.wait_for(stream.entered.wait(), _STEP_TIMEOUT_S)
        assert session.is_streaming, "the work turn never reached the provider"
        assert (
            not session._leaving_deliveries
        ), "precondition: the latch arms in the turn's OWN tail, not at entry"
        await session._on_job_completed("qa-r2", "the QA round finished", _settled("qa-r2"))
        await session._on_job_completed("rev-r6", "the review round finished", _settled("rev-r6"))
        assert session._deferred_job_results, "precondition: the batch is deferred"

        stream.release.set()
        await asyncio.wait_for(asyncio.shield(task), timeout=_STEP_TIMEOUT_S)
        # THE DISCRIMINATOR: with the declaration, the flush held the batch
        # inside the lock and NO delivery task exists to be admitted at the
        # release. Give the loop a bounded beat to prove one did not appear.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(asyncio.ensure_future(stream.delivery_entered.wait())),
                timeout=3.0,
            )
        assert session._leaving_deliveries, "the turn itself armed the departure pair"
        await session.dispose()

    assert (
        len(stream.requests) == 1
    ), f"the held batch must not buy a provider call: {len(stream.requests)}"
    assert len(_run_rows(directory)) == 1, "only the work turn may have opened a run"
    assert [row.get("kind") for row in _completion_rows(directory)] == ["complete"]
    assert _incident_rows(directory) == []
    assert [row.get("job_id") for row in _job_result_rows(directory)] == ["qa-r2", "rev-r6"]


@pytest.mark.asyncio
async def test_a_typed_prompt_cut_with_zero_round_trips_closes_neutrally(
    headless_tui_env: Path,
) -> None:
    """A person's zero-work run closes as ``closed``, not as an error.

    INTENTIONAL REVERSAL, with the receipts (2026-09-29 directive): this cell
    shipped in v1 (PR #1731) pinning the opposite — a typed prompt cut with
    zero round-trips published ``error|disposed``. The operator reopened the
    class on session 23fc556c3799: on the installed build, a run cut after the
    COMPLETED turn's output was delivered still rendered "Stopped with an
    error" on both surfaces, and the directive is that any disposal which
    catches a run that spent nothing (no provider round-trip) must render
    neutrally instead. A typed prompt is the same zero-work shape, so it takes
    the same neutral record: ``closed``, cause preserved, no incident.
    The v1 negative control survives WHERE IT MUST: a run that DID dispatch
    still errors (``test_a_run_cut_after_its_first_request_still_errors``).
    """
    config = headless_tui_env
    directory = config / "typed-prompt-cut-pre-dispatch"
    stream = _ParkedWorkTurn([text_turn("working")])

    with bounded(120, "a typed prompt cut before dispatch"):
        directory.mkdir(parents=True, exist_ok=True)
        session = build_session(directory, stream)
        parked, release = _park_runs_before_dispatch(session, except_token=None)
        task = asyncio.ensure_future(session.prompt("do the thing"))
        await asyncio.wait_for(parked.wait(), _STEP_TIMEOUT_S)
        assert session._turn_task is not None and not session._turn_task.done()

        dispose_task = asyncio.ensure_future(session.dispose())
        while not (session._signal is not None and session._signal.aborted):
            await asyncio.sleep(0.005)
        release.set()
        await asyncio.wait_for(dispose_task, timeout=_STEP_TIMEOUT_S)
        await asyncio.wait_for(asyncio.shield(task), timeout=_STEP_TIMEOUT_S)

    rows = _completion_rows(directory)
    assert [row.get("kind") for row in rows] == ["closed"], rows
    assert rows[-1].get("cause") == "disposed", rows[-1]
    assert stream.exhausted_at is None, "the run must not have consumed a scripted turn"


@pytest.mark.asyncio
async def test_a_peer_prompt_cut_with_zero_round_trips_closes_neutrally(
    headless_tui_env: Path,
) -> None:
    """The peer arm carries provenance, not just ``prompt()`` — closed, not error.

    Both peer arms of ``receive_peer_message`` spawn
    ``_prompt_messages([message], carried_prompt=True)``, and review round 1
    (MAJOR-2) caught the keyword never being forwarded to the pipeline, which
    sent every peer-opened run down the settle arm silently. The forwarding is
    pinned here still — the run must CARRY the peer's provenance — but the
    verdict it earns is the v2 neutral one (``closed``): the run spent no
    provider round-trip, so the disposal must not claim it was cut off. See
    ``test_a_typed_prompt_cut_with_zero_round_trips_closes_neutrally`` for the
    intentional reversal of the v1 pin.
    """
    config = headless_tui_env
    directory = config / "peer-prompt-cut-pre-dispatch"
    stream = _ParkedWorkTurn([text_turn("working")])

    with bounded(120, "a peer-opened run cut before dispatch"):
        directory.mkdir(parents=True, exist_ok=True)
        session = build_session(directory, stream)
        parked, release = _park_runs_before_dispatch(session, except_token=None)
        message = session._peer_custom_message(
            "peer asks a question",
            {"pid": 999999, "conversation_name": "peer", "model_label": "m"},
        )
        task = asyncio.ensure_future(session._prompt_messages([message], carried_prompt=True))
        await asyncio.wait_for(parked.wait(), _STEP_TIMEOUT_S)
        assert session._turn_task is not None and not session._turn_task.done()
        assert (
            session._attention_run_carried_prompt is True
        ), "the run must carry the peer's provenance (review round 1, MAJOR-2)"

        dispose_task = asyncio.ensure_future(session.dispose())
        while not (session._signal is not None and session._signal.aborted):
            await asyncio.sleep(0.005)
        release.set()
        await asyncio.wait_for(dispose_task, timeout=_STEP_TIMEOUT_S)
        await asyncio.wait_for(asyncio.shield(task), timeout=_STEP_TIMEOUT_S)

    rows = _completion_rows(directory)
    assert [row.get("kind") for row in rows] == ["closed"], rows
    assert rows[-1].get("cause") == "disposed", rows[-1]
    assert stream.exhausted_at is None, "the run must not have consumed a scripted turn"


@pytest.mark.asyncio
async def test_a_completed_turn_then_a_peer_prompt_disposal_closes_neutrally(
    headless_tui_env: Path,
) -> None:
    """THE 23fc RECORD CELL: completion delivered, then the disposal.

    Session 23fc556c3799 (2026-09-29, the operator's re-report): the turn
    completed at 12:05:25.975 and its output was delivered; the peer closeout
    opened a run 16 s later, and the disposal aborted that zero-round-trip run
    and published ``error|disposed`` (row 46912) — which superseded the
    completion for latest-wins readers and put "Stopped with an error" on a
    finished turn. The directive: a dispose that catches a run which spent
    nothing renders NEUTRALLY. The run carried a peer's ask, so it publishes
    ``closed`` (cause preserved), the completion is never masked by an error,
    and the successor journals no cut-off card.
    """
    config = headless_tui_env
    directory = config / "completed-then-peer-closed"
    stream = _ParkedWorkTurn([text_turn("first"), text_turn("peer reply")])

    with bounded(120, "a peer-opened run cut after a delivered completion"):
        directory.mkdir(parents=True, exist_ok=True)
        session = build_session(directory, stream)
        # The first turn COMPLETES — its honest ``complete`` row is the one the
        # disposal must not mask. The peer run is arranged only afterwards.
        await session.prompt("first")
        assert stream.requests, "the first turn must reach the provider"
        assert [row.get("kind") for row in _completion_rows(directory)] == [
            "complete"
        ], "precondition: the completed turn's own row"

        # The 23fc ordering: the peer's ask is delivered and admitted, then the
        # exit arrives before the run spends its first round-trip.
        parked, release = _park_runs_before_dispatch(session, except_token=None)
        message = session._peer_custom_message(
            "peer closeout after the turn",
            {"pid": 999999, "conversation_name": "peer", "model_label": "m"},
        )
        task = asyncio.ensure_future(session._prompt_messages([message], carried_prompt=True))
        await asyncio.wait_for(parked.wait(), _STEP_TIMEOUT_S)
        assert session._attention_run_carried_prompt is True
        assert session._turn_task is not None and not session._turn_task.done()

        dispose_task = asyncio.ensure_future(session.dispose())
        while not (session._signal is not None and session._signal.aborted):
            await asyncio.sleep(0.005)
        release.set()
        await asyncio.wait_for(dispose_task, timeout=_STEP_TIMEOUT_S)
        await asyncio.wait_for(asyncio.shield(task), timeout=_STEP_TIMEOUT_S)

    rows = _completion_rows(directory)
    assert [row.get("kind") for row in rows] == ["complete", "closed"], rows
    assert not [row for row in rows if row.get("kind") == "error"], rows
    assert rows[-1].get("cause") == "disposed", rows[-1]
    # LATEST-WINS, THE OPERATOR'S SYMPTOM: the store's newest row is the closed
    # record — never the ``error|disposed`` that used to supersede the result.
    from local_operator.session.attention import AttentionStore, conversation_identity

    state = AttentionStore().state(conversation_identity(directory))
    assert state.get("kind") == "closed", state
    assert (
        _incident_rows(directory) == []
    ), "the successor must not be told a closed run was cut off"

    # THE SUCCESSOR: a real boot. Its next turn gets no cut-off card and no
    # re-verification instruction about the finished work.
    successor_stream = ScriptedStream([text_turn("carrying on")])
    successor = build_session(directory, successor_stream)
    await successor.async_init()
    try:
        await successor.prompt("continue")
        assert successor_stream.requests, "the successor never called the provider"
        sent = "\n".join(
            text for request in successor_stream.requests for text in _user_row_texts(request)
        )
        assert "[session incident]" not in sent, sent[-2000:]
    finally:
        await successor.dispose()


@pytest.mark.asyncio
async def test_run_print_mode_holds_a_deferred_batch_on_its_way_out(
    headless_tui_env: Path,
) -> None:
    """THE HOST CELL: the one-shot exit arms the departure pair itself.

    ``run_print_mode`` is how exec ends, and it is the door 664a went through:
    nothing on that path armed a latch, so the batch the last turn flushed was
    spawned into a delivery task and the following dispose aborted the turn it
    opened. With the pair armed BEFORE the dispose (and before the
    ``before_dispose`` hook that gives exec's own close() its window), the
    slipped spawn is held at admission: no request, no run, no verdict — and
    the results stay durable for whoever opens next.
    """
    from local_operator.headless_print import run_print_mode

    config = headless_tui_env
    directory = config / "run-print-mode-holds-the-batch"
    directory.mkdir(parents=True, exist_ok=True)
    stream = _ParkedDeliveryTurn([text_turn("working"), text_turn("delivery")], hold_delivery=True)
    session = build_session(directory, stream)

    async def hook() -> None:
        # The window exec's own close() gives a spawned task between the
        # commit and the disposal: wait for it to reach its decision — either
        # the delivery run dispatched (the pre-v3 shape; the park fires) or
        # the held task finished without opening a turn (the v1 shape).
        # With the v3 session-side arming (2026-09-30) there is a THIRD shape
        # and it is the shipped one: the batch was held INSIDE the last turn's
        # lock, so no delivery task exists at all and there is nothing to wait
        # for. The one-beat wait below is kept anyway, so a regression that
        # spawns again surfaces HERE rather than racing the assertions below.
        pending = [t for t in session._background_tasks if not t.done()]
        waiter = asyncio.ensure_future(stream.delivery_entered.wait())
        try:
            if pending:
                await asyncio.wait([*pending, waiter], return_when=asyncio.FIRST_COMPLETED)
            else:
                try:
                    await asyncio.wait_for(asyncio.shield(waiter), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
        finally:
            if not waiter.done():
                waiter.cancel()

    run_task = asyncio.ensure_future(
        run_print_mode(session, ["delegate two children"], before_dispose=hook)
    )
    with bounded(120, "run_print_mode holds a deferred batch on its way out"):
        await asyncio.wait_for(stream.entered.wait(), _STEP_TIMEOUT_S)
        await session._on_job_completed("qa-r2", "the QA round finished", _settled("qa-r2"))
        await session._on_job_completed("rev-r6", "the review round finished", _settled("rev-r6"))
        assert session._deferred_job_results, "precondition: the batch is deferred"
        stream.release.set()
        code = await asyncio.wait_for(run_task, timeout=_STEP_TIMEOUT_S)
        assert code == 0

    assert (
        len(stream.requests) == 1
    ), f"the held batch must not have bought a provider call: {len(stream.requests)} request(s)"
    runs = _run_rows(directory)
    assert len(runs) == 1, f"only the work turn may have opened a run: {runs!r}"
    assert [row.get("kind") for row in _completion_rows(directory)] == [
        "complete"
    ], f"no verdict for the held delivery: {_completion_rows(directory)!r}"
    assert (
        _incident_rows(directory) == []
    ), "the next turn must not be told the held batch was cut off"
    rows = _job_result_rows(directory)
    assert [row.get("job_id") for row in rows] == ["qa-r2", "rev-r6"], rows

    # THE SUCCESSOR still receives the batch — a hold must never be a drop.
    successor_stream = ScriptedStream([text_turn("carrying on")])
    successor = build_session(directory, successor_stream)
    await successor.async_init()
    try:
        await successor.prompt("continue")
        sent = "\n".join(
            text for request in successor_stream.requests for text in _user_row_texts(request)
        )
        assert "the QA round finished" in sent, sent[-2000:]
        assert "the review round finished" in sent, sent[-2000:]
        assert "[session incident]" not in sent, sent[-2000:]
    finally:
        await successor.dispose()
