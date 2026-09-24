"""An accepted prompt that races a turn the queue did not open must LAND.

QA on PR #1528 (finding Q1-5) measured a silent drop on the desktop send path:
a ``POST /messages`` whose admission raced a background job's result-delivery
turn was refused with ``TurnInFlight`` AFTER the owner had accepted it (the
route answered 503), and the client's retry with the SAME request id was then
answered "admitted" while the message reached no transcript — 2 of 19 race runs,
on both arms. Two defects compose it, and each has its own cell here:

* the drain handed an accepted prompt to ``Session.prompt`` while a wake-style
  turn held the session's turn lock, and ``Session.prompt`` refuses outright
  rather than queueing — so an admitted prompt failed (the "prompt failed after
  admission … TurnInFlight" log line);
* the refusal parked the id as ``prompt-transfer``, and a same-id PROMPT retry
  then read that parked state as "already admitted".

The interleaving is FORCED, not timed: a real ``Session`` behind the production
``ServingSessionHandle`` opens a job-result delivery turn exactly as a settled
child does (``_deliver_job_results``), whose provider call parks on an event this
file owns, so the prompt provably arrives while that turn holds the lock.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import StreamEvent
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.transcript import Transcript
from tests.e2e.harness import ScriptedStream, build_session, text_turn


class _GatedStream(ScriptedStream):
    """``ScriptedStream`` whose FIRST call parks until ``release`` is set.

    The first call is the delivery turn's, so parking it holds the session's
    turn lock for exactly as long as the test needs; ``entered`` is the event
    that says the lock is held (a turn only calls the provider under it).
    """

    def __init__(self, turns: Sequence[Sequence[StreamEvent]]) -> None:
        super().__init__(turns)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    def __call__(self, request: Any, signal: Any = None) -> AsyncIterator[StreamEvent]:
        inner = super().__call__(request, signal)
        first = len(self.requests) == 1

        async def gen() -> AsyncIterator[StreamEvent]:
            if first:
                self.entered.set()
                await self.release.wait()
            async for event in inner:
                yield event

        return gen()


async def _deliver_a_job_result_and_hold_it(tmp_path: Path) -> tuple[Any, Any, _GatedStream]:
    # Spare turns beyond the two a cell needs: a first real prompt also spends a
    # provider call on naming the conversation, and a short tape would answer
    # the user turn from the wrong script.
    stream = _GatedStream([text_turn(f"reply {i}") for i in range(4)])
    session = build_session(tmp_path / "sess", stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    # The production opener for a settled child's result (one idle-time turn,
    # ``Session._prompt_messages`` under ``_turn_lock``), not a stand-in.
    session._deliver_job_results([("job-1", "child 1 done", None)])
    await asyncio.wait_for(stream.entered.wait(), 5)
    assert session._turn_lock.locked(), "the delivery turn must hold the lock"
    return session, handle, stream


def _landed(directory: Path, command_id: str) -> bool:
    """Read the durable index from DISK, the authority a restart would read."""
    return Transcript(directory).has_admitted_command(command_id)


@pytest.mark.asyncio
async def test_a_prompt_accepted_during_a_delivery_turn_waits_and_lands(tmp_path: Path) -> None:
    """The admitted prompt waits for the delivery turn instead of failing.

    On the unfixed drain the prompt's receipt raises ``TurnInFlight`` — the 503
    the desktop route answered — the instant the drain reaches it.
    """
    session, handle, stream = await _deliver_a_job_result_and_hold_it(tmp_path)
    try:
        receipt = asyncio.ensure_future(handle.prompt("race hi", command_id="race-1"))
        # Let the drain reach the head while the lock is still held — the
        # forced interleaving. A refusal would complete the receipt here.
        for _ in range(20):
            await asyncio.sleep(0)
        assert not receipt.done() or receipt.exception() is None, receipt.exception()
        stream.release.set()
        assert await asyncio.wait_for(receipt, 10) == "prompt admitted"
        assert _landed(tmp_path / "sess", "race-1")
    finally:
        stream.release.set()
        await handle.dispose()


@pytest.mark.asyncio
async def test_an_admitted_receipt_is_never_given_for_a_message_that_did_not_land(
    tmp_path: Path,
) -> None:
    """The invariant itself: "admitted" for an id implies a durable row.

    Driven through the sequence QA measured — the first attempt fails while the
    delivery turn holds the lock, and the client retries the SAME id. Whatever
    the first attempt answers, every receipt that says admitted must be backed
    by the transcript once the session is idle. On the unfixed tree the retry
    answered "already admitted" and the row never existed.
    """
    session, handle, stream = await _deliver_a_job_result_and_hold_it(tmp_path)
    receipts: list[str] = []
    try:
        first = asyncio.ensure_future(handle.prompt("race hi", command_id="race-2"))
        for _ in range(20):
            await asyncio.sleep(0)
        if first.done() and first.exception() is not None:
            # QA's sequence: the client saw a failure, the session went idle,
            # and the client retried the same request id.
            receipts.append(f"failed: {first.exception()}")
            stream.release.set()
            await asyncio.wait_for(_idle(session, handle), 10)
            receipts.append(await handle.prompt("race hi", command_id="race-2"))
        else:
            stream.release.set()
            receipts.append(await asyncio.wait_for(first, 10))
        await asyncio.wait_for(_idle(session, handle), 10)
        admitted = [r for r in receipts if r in ("prompt admitted", "already admitted")]
        assert admitted, receipts
        assert _landed(tmp_path / "sess", "race-2"), f"reported {receipts} but nothing landed"
    finally:
        stream.release.set()
        await handle.dispose()


async def _idle(session: Any, handle: ServingSessionHandle) -> None:
    """Bounded by the caller's ``wait_for``; polls the handle's own busy view."""
    while handle.is_busy() or session.is_streaming:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_the_boot_inbox_drain_still_steers_instead_of_waiting(tmp_path: Path) -> None:
    """The one caller that must NOT wait: the boot drain of the spool.

    ``process._drain_inbox_into`` runs before the control socket listens, so a
    prompt that waited out a wake turn there would keep the runtime unreachable
    for that whole turn. It opts out (``wait_for_turn=False``) and answers the
    refusal by steering the owner's row into the turn in flight — the path QA
    round 1 (Q-1) built. Pinned against a real ``Session`` so the opt-out is
    proved to reach the drain: without it the drain call below would park until
    ``release`` is set and ``wait_for`` would time out.
    """
    from local_operator.session.runtime import process as child_mod
    from local_operator.session.runtime.inbox import (
        SOURCE_USER,
        InboxLine,
        append_inbox,
    )

    session, handle, stream = await _deliver_a_job_result_and_hold_it(tmp_path)
    try:
        append_inbox(
            tmp_path / "sess",
            InboxLine(
                text="spooled hi",
                sender={},
                mode="mailbox",
                wake=True,
                source=SOURCE_USER,
                command_id="spool-1",
            ),
        )
        assert await asyncio.wait_for(child_mod._drain_inbox_into(handle), 5) == 1
        assert [getattr(m, "id", None) for m in session.queued_steering()] == ["spool-1"]
    finally:
        stream.release.set()
        await handle.dispose()


@pytest.mark.asyncio
async def test_a_headless_exec_turn_waits_for_a_turn_it_did_not_open(tmp_path: Path) -> None:
    """``lop exec`` (and ``--resume``) submits through the same queue.

    The sighting beside Q1-5: ``lop exec --resume`` failed with "session is
    already streaming; use steer() to inject mid-turn" when the session was busy
    with a turn exec did not open — a resume catch-up, a job-result delivery.
    ``run_headless_prompt`` must now wait that turn out and complete its own,
    not report failure.
    """
    session, handle, stream = await _deliver_a_job_result_and_hold_it(tmp_path)
    try:
        run = asyncio.ensure_future(handle.run_headless_prompt("exec hi"))
        for _ in range(20):
            await asyncio.sleep(0)
        assert not run.done(), f"exec gave up while the delivery turn ran: {run.result()}"
        stream.release.set()
        assert await asyncio.wait_for(run, 10) is True, handle.last_prompt_failure
    finally:
        stream.release.set()
        await handle.dispose()
