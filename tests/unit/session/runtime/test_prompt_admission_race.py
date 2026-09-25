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
import json
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import StreamEvent
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.transcript import Transcript
from tests.e2e.harness import ScriptedStream, build_session, text_turn, tool_call_turn


class _GatedStream(ScriptedStream):
    """``ScriptedStream`` whose FIRST call parks until ``release`` is set.

    The first call is the delivery turn's, so parking it holds the session's
    turn lock for exactly as long as the test needs; ``entered`` is the event
    that says the lock is held (a turn only calls the provider under it).
    """

    def __init__(
        self,
        turns: Sequence[Sequence[StreamEvent]],
        *,
        gate_call: int = 1,
        fail: Callable[[Any], bool] | None = None,
        error: str = "probe boom",
    ) -> None:
        super().__init__(turns)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        #: The provider call the gate parks; call 1 is the delivery turn's first
        #: request, and nothing else can be running then.
        self.gate_call = gate_call
        #: Which LATER request raises. A predicate rather than a call index
        #: because calls interleave with the handle's conversation-naming call,
        #: so an index pins the wrong request the moment naming moves; this
        #: picks the request by what it carries.
        self.fail = fail
        self.error = error

    def __call__(self, request: Any, signal: Any = None) -> AsyncIterator[StreamEvent]:
        inner = super().__call__(request, signal)
        index = len(self.requests)

        async def gen() -> AsyncIterator[StreamEvent]:
            if index == self.gate_call:
                self.entered.set()
                await self.release.wait()
            if index != self.gate_call and self.fail is not None and self.fail(request):
                # A provider failure, which ``Session`` reports as an EVENT
                # rather than as an exception (``AgentEndEvent.error``).
                raise RuntimeError(self.error)
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


def _request_text(request: Any) -> str:
    """Everything the provider was sent, as one string (for predicates)."""
    parts: list[str] = []
    for message in request.messages:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(getattr(part, "text", part)) for part in content)
    return "\n".join(parts)


def _last_role(request: Any) -> str:
    return str(getattr(request.messages[-1], "role", "") or "")


def _is_delivery_followup(request: Any) -> bool:
    """The delivery turn's post-tool request, and not the queued command's.

    Keyed on the tool RESULT the request ends at, not on a provider-call index:
    both turns' follow-ups end at one, so the id-bearing text is what tells them
    apart — the queued command's own requests carry its prompt text, the delivery
    turn's do not. (An index would silently pin the wrong request as soon as the
    handle's conversation-naming call moves; agent review round 2, MINOR-1.)
    """
    return _last_role(request) == "tool" and "exec hi" not in _request_text(request)


def _is_own_followup(request: Any) -> bool:
    """The queued command's SECOND generation: its own tool result, then failure.

    Deliberately the second call rather than the first: this is the shape a
    per-turn attribution with the wrong generation comparison would report as
    clean, which is what the cell below is here to catch.
    """
    return _last_role(request) == "tool" and "exec hi" in _request_text(request)


def _rig_tape() -> list[list[StreamEvent]]:
    """The provider script, indexed by CALL, in the order this rig issues them.

    Measured on the rig (agent review round 2, MINOR-1), and the middle entry is
    the one worth naming: the handle's conversation-naming call is issued when
    the QUEUED command is enqueued and lands BETWEEN the delivery turn's two
    calls, so a tape that ignored it would serve the delivery turn's follow-up
    from the queued command's script and the cells would pass by alignment rather
    than by design.

    1. the delivery turn's first request — parked by the gate, answered with a
       ``todo`` call so that turn completes a REAL tool boundary;
    2. the naming call (raises nothing; its turn is irrelevant here);
    3. the delivery turn's follow-up — the request that FAILS;
    4. the queued command's first request — answered with a ``glob`` call, a
       different tool name from the delivery turn's, so a boundary note can be
       attributed to the turn that produced it;
    5. the queued command's follow-up — served a reply, or made to fail by the
       cell's own predicate.
    """
    return [
        tool_call_turn(
            text="delivery",
            tool_name="todo",
            tool_call_id="call-delivery",
            arguments={"op": "view"},
        ),
        text_turn("unused: the handle's naming call"),
        text_turn("unused: the delivery follow-up raises"),
        tool_call_turn(
            text="exec",
            tool_name="glob",
            tool_call_id="call-exec",
            arguments={"pattern": "*.jsonl"},
        ),
        text_turn("unused: the queued command's follow-up"),
    ]


def _exec_tools(tmp_path: Path) -> list[Any]:
    """Two REGISTERED tools, deliberately distinct names.

    A planning failure ("tool not found") emits no ``ToolExecutionEndEvent`` at
    all (``harness/loop.py`` withholds it for a call that never started), so an
    invented tool name cannot reach the boundary note; and two names are what
    make ``boundaries`` say WHICH turn was noted rather than only how many.
    """
    from local_operator.harness.types import ToolContext
    from local_operator.tools.registry import create_tools

    return create_tools(ToolContext(cwd=str(tmp_path)), enabled=["todo", "glob"])


async def _deliver_a_failing_job_result_and_hold_it(
    tmp_path: Path, *, fail: Any = None
) -> tuple[Any, Any, _GatedStream, list[str]]:
    """A delivery turn that holds the lock, completes a REAL tool, then FAILS.

    The gate parks the delivery turn's first provider call; the turn is then
    answered with a ``todo`` call, completes that boundary, and its follow-up
    request raises — so the waited-out turn ends with ``AgentEndEvent(error=...)``
    after a real ``ToolExecutionEndEvent``. That is the shape agent review round
    1's MAJOR-1 probe exercised.
    """
    stream = _GatedStream(_rig_tape(), gate_call=1, fail=fail or _is_delivery_followup)
    session = build_session(tmp_path / "sess", stream, tools=_exec_tools(tmp_path))
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    # Spy on the boundary note: the second half of the finding is WHICH turn's
    # tool ends reach it, and a spy measures exactly that.
    boundaries: list[str] = []
    inner_boundary = handle._note_turn_boundary

    def note(name: str) -> None:
        boundaries.append(name)
        inner_boundary(name)

    handle._note_turn_boundary = note  # type: ignore[method-assign]
    session._deliver_job_results([("job-1", "child 1 done", None)])
    await asyncio.wait_for(stream.entered.wait(), 5)
    assert session._turn_lock.locked(), "the delivery turn must hold the lock"
    return session, handle, stream, boundaries


@pytest.mark.asyncio
async def test_a_turn_that_failed_during_the_wait_is_not_this_commands_failure(
    tmp_path: Path,
) -> None:
    """Agent review round 1, MAJOR-1: the waited-out turn's end is not ours.

    The drain waits out a turn it did not open, and its outcome probe was
    installed before that wait — so the OTHER turn's terminal event set this
    command's failure flag. A queued ``lop exec`` turn that ran, landed and
    ended clean was reported failed (``run_headless_prompt`` -> False, with an
    empty ``last_prompt_failure``, so nothing even named the cause), and the
    same subscription fed the other turn's tool boundary into this command's
    journal note.

    Both halves are asserted, because both come from the one subscription: a fix
    that only re-armed the failure flag would leave the boundary.
    """
    session, handle, stream, boundaries = await _deliver_a_failing_job_result_and_hold_it(tmp_path)
    try:
        run = asyncio.ensure_future(handle.run_headless_prompt("exec hi"))
        # Let the drain reach the head while the delivery turn still runs: the
        # forced interleaving, not a race.
        for _ in range(20):
            await asyncio.sleep(0)
        assert not run.done(), "the exec turn must still be queued behind the delivery turn"
        stream.release.set()
        assert await asyncio.wait_for(run, 10) is True, (
            "the delivery turn's failure was reported as this command's: "
            f"last_prompt_failure={handle.last_prompt_failure!r}"
        )
        assert handle.last_prompt_failure == ""
        assert _user_rows(tmp_path / "sess", "exec hi") == 1, "the exec row must land exactly once"
        # OUR OWN boundary, and only it. Both turns complete a tool call, so the
        # list says WHICH was noted rather than only how many: the delivery
        # turn's ``todo`` reaching this note is the finding, and the unfixed head
        # notes both names. Measured there, the ``todo`` note lands at
        # provider-call index 2 — after the waited-out turn's tool call and while
        # this command was still waiting at the head of the queue.
        assert boundaries == [
            "glob"
        ], f"the waited-out turn's tool boundary was noted against this command: {boundaries}"
    finally:
        stream.release.set()
        await handle.dispose()


@pytest.mark.asyncio
async def test_this_commands_own_failure_is_still_reported(tmp_path: Path) -> None:
    """The other direction, so the gate above cannot be an over-correction.

    The failure is moved onto the QUEUED command's own turn, and that turn is
    genuinely MULTI-GENERATION — it completes a ``glob`` boundary and its
    SECOND request fails — which is the shape a per-turn attribution with the
    wrong generation comparison would report as clean. ``False`` is what the
    goal loop's continuation reads before deciding whether to iterate again.
    """
    session, handle, stream, boundaries = await _deliver_a_failing_job_result_and_hold_it(
        tmp_path, fail=_is_own_followup
    )
    try:
        run = asyncio.ensure_future(handle.run_headless_prompt("exec hi"))
        for _ in range(20):
            await asyncio.sleep(0)
        assert not run.done(), "the exec turn must still be queued behind the delivery turn"
        stream.release.set()
        assert (
            await asyncio.wait_for(run, 10) is False
        ), "this command's own failed turn was reported as a success"
        # Its own boundary DID fire (so the verdict above is about a turn that
        # really ran), and the delivery turn's ``todo`` boundary is still absent.
        assert boundaries == ["glob"], boundaries
        # The row still lands: a failing turn is reported, never silently
        # dropped. ``last_prompt_failure`` is deliberately NOT asserted — that
        # string is only set on the RAISING path (the drain's ``except`` arm);
        # an event-reported provider failure leaves it empty, which is
        # pre-existing behaviour this PR does not change.
        assert _user_rows(tmp_path / "sess", "exec hi") == 1
    finally:
        stream.release.set()
        await handle.dispose()


def _user_rows(directory: Path, text: str) -> int:
    """How many durable USER rows carry ``text`` — the transcript, not a stub."""
    rows = 0
    for line in (directory / "transcript.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = entry.get("payload") or {}
        if payload.get("role") != "user":
            continue
        content = payload.get("content") or []
        if any(isinstance(part, dict) and part.get("text") == text for part in content):
            rows += 1
    return rows


@pytest.mark.asyncio
async def test_a_cancelled_admission_awaiter_does_not_break_the_running_turn(
    tmp_path: Path,
) -> None:
    """Agent review round 2, MAJOR-1: a cancelled awaiter must not raise on the bus.

    ``admitted`` is awaited BARE at the handle's own receipt (``serving.py:2886``)
    and by a duplicate's ``await existing.admitted`` (``2703``), so cancelling any
    awaiter — a desktop client disconnect, a request timeout, a shed task, the
    same-id retry this PR targets — cancels the SHARED future. The outcome gate
    then read ``.exception()`` on it, which RAISES ``CancelledError``; that is a
    ``BaseException``, so ``Session._emit``'s per-handler ``except Exception``
    cannot contain it and it leaves the fan-out into whichever turn was
    emitting. Measured consequence: the RUNNING turn is truncated at its first
    event, the prompt drain is cancelled, and nothing is reported anywhere.

    Both halves are asserted, because either alone would leave the other open: the
    running turn must survive, and the queued command must still land.
    """
    from local_operator.harness.types import AgentEndEvent

    session, handle, stream = await _deliver_a_job_result_and_hold_it(tmp_path)
    ends: list[AgentEndEvent] = []

    def observe(event: Any) -> None:
        if isinstance(event, AgentEndEvent):
            ends.append(event)

    unsubscribe = session.subscribe(observe)
    try:
        waiter = asyncio.ensure_future(handle.prompt("race hi", command_id="cancel-1"))
        for _ in range(20):
            await asyncio.sleep(0)
        assert not waiter.done(), "the prompt must still be queued behind the delivery turn"
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        # The delivery turn is still running and still emitting: from here on its
        # events are the ones that reached the gate.
        stream.release.set()
        deadline = asyncio.get_running_loop().time() + 10
        while not ends and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert ends, "the running turn was truncated: its end event never arrived"
        drain = handle._prompt_drain_task
        assert drain is not None and not drain.cancelled(), "the prompt drain was cancelled"
        # And the cancelled caller's message is NOT silently lost: the drain runs
        # it once the delivery turn releases the lock.
        deadline = asyncio.get_running_loop().time() + 10
        while _user_rows(tmp_path / "sess", "race hi") != 1 and (
            asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.01)
        assert (
            _user_rows(tmp_path / "sess", "race hi") == 1
        ), "the cancelled caller's message never landed"
    finally:
        stream.release.set()
        unsubscribe()
        await handle.dispose()


@pytest.mark.asyncio
async def test_a_cancelled_duplicate_retry_leaves_the_first_admission_intact(
    tmp_path: Path,
) -> None:
    """The other door to the same state: a cancelled SAME-ID retry.

    A duplicate send with an id the queue already holds awaits the FIRST
    command's future. Cancelling that duplicate (the client gives up) cancels the
    shared future unless it is shielded, and the first caller's own receipt then
    raises ``CancelledError`` for a message that was admitted — the retry path
    this PR exists to make safe, walking into the same hole from the other side.
    """
    session, handle, stream = await _deliver_a_job_result_and_hold_it(tmp_path)
    try:
        first = asyncio.ensure_future(handle.prompt("race hi", command_id="dup-1"))
        for _ in range(20):
            await asyncio.sleep(0)
        assert not first.done()
        duplicate = asyncio.ensure_future(handle.prompt("race hi", command_id="dup-1"))
        for _ in range(20):
            await asyncio.sleep(0)
        duplicate.cancel()
        with pytest.raises(asyncio.CancelledError):
            await duplicate
        command = handle._prompt_commands.get("dup-1")
        assert (
            command is not None and not command.admitted.cancelled()
        ), "the duplicate's cancellation destroyed the first caller's admission"
        stream.release.set()
        assert await asyncio.wait_for(first, 10) == "prompt admitted"
        assert _user_rows(tmp_path / "sess", "race hi") == 1, "the message must land exactly once"
    finally:
        stream.release.set()
        await handle.dispose()


@pytest.mark.asyncio
async def test_a_cancelled_admission_future_reads_as_not_ours(tmp_path: Path) -> None:
    """The gate's own half, isolated from the shields.

    The two shields below remove the handle's OWN route to a cancelled admission
    future, and this cell is why the ORDER of the gate's two checks still
    matters: ``Future.exception()`` RAISES ``CancelledError`` on a cancelled
    future, a ``BaseException`` that ``Session._emit``'s ``except Exception``
    cannot contain, so the next event on the bus takes the running turn and the
    drain down with it. Constructed here by cancelling the future the way a
    foreign awaiter would (``asyncio.wait_for``, a gather that cancels siblings,
    any later caller), because "another holder destroyed it" is the state a
    shared future has to survive rather than one this handle can prevent.
    """
    from local_operator.harness.types import AgentEndEvent

    session, handle, stream = await _deliver_a_job_result_and_hold_it(tmp_path)
    ends: list[AgentEndEvent] = []

    def observe(event: Any) -> None:
        if isinstance(event, AgentEndEvent):
            ends.append(event)

    unsubscribe = session.subscribe(observe)
    waiter: asyncio.Future[str] | None = None
    try:
        waiter = asyncio.ensure_future(handle.prompt("race hi", command_id="cancel-2"))
        for _ in range(20):
            await asyncio.sleep(0)
        assert (
            waiter is not None and not waiter.done()
        ), "the prompt must still be queued behind the delivery turn"
        command = handle._prompt_commands["cancel-2"]
        command.admitted.cancel()
        stream.release.set()
        # The running turn must still finish; on the unfixed gate its first event
        # after the cancellation raises into the fan-out and truncates it.
        deadline = asyncio.get_running_loop().time() + 10
        while not ends and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert ends, "a cancelled admission future truncated the running turn"
        drain = handle._prompt_drain_task
        assert drain is not None and not drain.cancelled(), "the prompt drain was cancelled"
        deadline = asyncio.get_running_loop().time() + 10
        while _user_rows(tmp_path / "sess", "race hi") != 1 and (
            asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.01)
        assert _user_rows(tmp_path / "sess", "race hi") == 1, "the message never landed"
    finally:
        stream.release.set()
        if waiter is not None:
            waiter.cancel()
        unsubscribe()
        await handle.dispose()
