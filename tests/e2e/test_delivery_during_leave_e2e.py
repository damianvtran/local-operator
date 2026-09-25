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
        assert [row.get("job_id") for row in _job_result_rows(directory)] == ["qa-r2", "rev-r6"]
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
    ], f"both results must be durable, in settle order, after the exit: {held!r}"

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
        assert (
            "[session incident]" not in sent
        ), "a successor that re-verifies finished work is the wasted turn this fixes"
    finally:
        await successor.dispose()
    # U7, ON THE REAL FLOW (review round 2): the held marker is written once and
    # cleared nowhere, so the notice is painted on every later replay — and the
    # first wording claimed "no turn has read it yet", which after THIS turn is a
    # lie. The successor has now read and answered both reports; the notice is
    # recomposed here from the DURABLE rows by the production decision, and it must
    # still be a fact about the arrival rather than a claim about now.
    from local_operator.harness.rows import HELD_DELIVERY_NOTICE, held_delivery_notice

    for row in _job_result_rows(directory):
        decided = held_delivery_notice(row)
        assert decided is not None, "the row is still marked held, which is now harmless"
        text, _severity = decided
        assert "no turn ran for it at that point" in text
        assert "has read it yet" not in text, (
            "after the answering turn, the notice must not still claim nobody read it"
        )
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
