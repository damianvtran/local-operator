"""End-to-end: an aside adopted mid-turn waits for the turn's safe boundary.

Operator decision (option A, decision rows D1-D6): an aside's "Add to
conversation" that arrives while the main session is mid-turn is QUEUED and
applied at the turn's next safe point instead of being refused. The invariant
this file pins is the one that made the old refusal necessary: nothing may
splice into a live tool batch, so the provider only ever sees a message list it
accepts.

Assembled like ``test_steer_wake_e2e.py``: the real ``Session``, the real agent
loop and transcript, a tool parked on an ``asyncio.Event`` so the adopt lands
while the batch is provably open, and only the provider scripted.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    AgentTool,
    HistoryDeltaEvent,
    Message,
    TextContent,
    ToolResult,
)
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    dispose_quietly,
    text_turn,
    tool_call_turn,
)
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

BOUND_S = 90.0

QUESTION = "ASIDE-Q: why sed and not edit?"
ANSWER = "ASIDE-A: the file is generated."


def _message_rows(session: Any) -> list[tuple[str, str]]:
    """``(role, first text or tool id)`` for every persisted message row, in order."""
    rows: list[tuple[str, str]] = []
    for entry in session._transcript.entries():
        if entry.type != "message" or entry.payload.get("kind") != "message":
            continue
        payload = entry.payload
        role = str(payload.get("role", ""))
        if role == "tool":
            rows.append((role, str(payload.get("tool_call_id", ""))))
            continue
        calls = payload.get("tool_calls") or []
        if calls:
            rows.append((role, "call:" + str(calls[0].get("id", ""))))
            continue
        texts = [str(block.get("text") or "") for block in payload.get("content", [])]
        rows.append((role, next((text for text in texts if text), "")))
    return rows


def _request_shape(request: Any) -> list[tuple[str, str]]:
    shape: list[tuple[str, str]] = []
    for message in request.messages:
        role = str(getattr(message, "role", ""))
        if role == "tool":
            shape.append((role, str(getattr(message, "tool_call_id", ""))))
        elif getattr(message, "tool_calls", None):
            shape.append((role, "call:" + message.tool_calls[0].id))
        else:
            shape.append((role, str(getattr(message, "text", "") or "")))
    return shape


@pytest.mark.asyncio
async def test_an_adopt_during_a_parked_tool_lands_after_the_closed_batch(
    headless_tui_env: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The headline cell: adopt while the tool is parked, release, inspect.

    * the adopt answers ``queued`` and writes nothing while the batch is open;
    * the persisted order is tool call -> tool result -> the turn's answer ->
      the adopted user/assistant pair, all after the turn has ended;
    * the NEXT provider request carries the pair AFTER the closed batch;
    * the pair is announced once as a ``history_delta`` emitted after
      ``agent_end``;
    * no "spliced" repair fired anywhere (the repair is the symptom of exactly
      the splice this change must never make).
    """
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute(
        tool_call_id: str, args: dict[str, Any], signal: Any, on_update: Any, context: Any
    ) -> ToolResult:
        started.set()
        await release.wait()
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="echo", content=[TextContent(text="ok")]
        )

    echo = AgentTool(
        name="echo",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        execute=execute,
    )
    stream = ScriptedStream(
        [
            tool_call_turn(text="working", tool_name="echo", tool_call_id="c1", arguments={}),
            text_turn("TURN-DONE"),
            text_turn("NEXT-ANSWER"),
        ]
    )
    session = build_session(
        headless_tui_env / "sessions" / "adopt-queue", stream, cwd=headless_tui_env, tools=[echo]
    )
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    pair = [Message.user(QUESTION), Message.assistant(ANSWER)]
    try:
        with caplog.at_level(logging.WARNING), bounded(BOUND_S, "a queued aside adopt"):
            task = asyncio.ensure_future(session.prompt("start"))
            await asyncio.wait_for(started.wait(), BOUND_S)

            # The batch is OPEN: the live list ends in the unanswered call.
            tail = session._context.messages[-1]
            assert isinstance(tail, Message) and tail.tool_calls[0].id == "c1"
            assert await session.adopt_aside(pair) == "queued"
            assert not session._transcript.has_entry(pair[-1].id)
            assert all(m.id != pair[0].id for m in session._context.messages)

            release.set()
            await asyncio.wait_for(task, BOUND_S)

            rows = _message_rows(session)
            assert rows == [
                ("user", "start"),
                ("assistant", "call:c1"),
                ("tool", "c1"),
                ("assistant", "TURN-DONE"),
                ("user", QUESTION),
                ("assistant", ANSWER),
            ], rows
            assert session._pending_shell_records == []

            end_at = next(i for i, e in enumerate(events) if isinstance(e, AgentEndEvent))
            deltas = [(i, e) for i, e in enumerate(events) if isinstance(e, HistoryDeltaEvent)]
            assert len(deltas) == 1, deltas
            delta_at, delta = deltas[0]
            assert delta_at > end_at
            assert [m.id for m in delta.messages] == [m.id for m in pair]

            # The next provider request carries the pair after the closed batch.
            await session.prompt("and next?")
            assert len(stream.requests) == 3
            shape = _request_shape(stream.requests[2])
            assert shape[-7:] == [
                ("user", "start"),
                ("assistant", "call:c1"),
                ("tool", "c1"),
                ("assistant", "TURN-DONE"),
                ("user", QUESTION),
                ("assistant", ANSWER),
                ("user", "and next?"),
            ], shape
            assert stream.exhausted_at is None
        spliced = [r for r in caplog.records if "spliced" in r.getMessage()]
        assert spliced == [], [r.getMessage() for r in spliced]
    finally:
        release.set()
        await dispose_quietly(session)


def _echo_parked() -> tuple[AgentTool, asyncio.Event, asyncio.Event]:
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute(
        tool_call_id: str, args: dict[str, Any], signal: Any, on_update: Any, context: Any
    ) -> ToolResult:
        started.set()
        await release.wait()
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="echo", content=[TextContent(text="ok")]
        )

    tool = AgentTool(
        name="echo",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    return tool, started, release


@pytest.mark.asyncio
async def test_a_cancelled_turn_keeps_bang_receipts_and_reports_the_unlanded_adopt(
    headless_tui_env: Path,
) -> None:
    """D7, the reviewer's R1 probe as a cell.

    The prompt task is cancelled while the tool is parked (no abort: its
    control socket went away), which leaves the tail open. The queued adopt
    cannot land on that tail; a later ``! echo one`` must still become durable
    exactly as it did before adopts shared the queue, and dispose must report
    what never landed instead of dropping it silently.
    """
    from local_operator.harness.types import NoticeEvent
    from local_operator.session.transcript import Transcript

    echo, started, release = _echo_parked()
    stream = ScriptedStream(
        [tool_call_turn(text="working", tool_name="echo", tool_call_id="c1", arguments={})]
    )
    directory = headless_tui_env / "sessions" / "adopt-cancel"
    session = build_session(directory, stream, cwd=headless_tui_env, tools=[echo])
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    pair = [Message.user(QUESTION), Message.assistant(ANSWER)]
    try:
        with bounded(BOUND_S, "a cancelled turn with a queued adopt"):
            task = asyncio.ensure_future(session.prompt("start"))
            await asyncio.wait_for(started.wait(), BOUND_S)
            assert await session.adopt_aside(pair) == "queued"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            receipt = ToolResult(
                tool_call_id="shell-1", tool_name="bash", content=[TextContent(text="one")]
            )
            await session.record_shell("echo one", receipt)
            await session.dispose()
    finally:
        release.set()
        await dispose_quietly(session)

    reopened = Transcript(directory)
    assert reopened.has_entry("shell:shell-1:result"), "the bang receipt must stay durable"
    # The receipt's own closed call/result pair now ends the list, so the held
    # adoption lands right after it rather than being lost or reported.
    rows = _message_rows(session)
    assert rows[-5:] == [
        ("user", "! echo one"),
        ("assistant", "call:shell-1"),
        ("tool", "shell-1"),
        ("user", QUESTION),
        ("assistant", ANSWER),
    ], rows
    assert not [e for e in events if isinstance(e, NoticeEvent) and e.kind == "error"]


def _unanswered_calls(request: Any) -> list[str]:
    """Call ids in ``request`` NOT answered by the tool rows directly after them.

    The wire rule the provider enforces: each assistant tool call's result must
    come straight after it. Measured on the request the session actually built
    (after ``_pair_spliced_tool_results``), not on the transcript.
    """
    messages = list(request.messages)
    missing: list[str] = []
    for index, message in enumerate(messages):
        calls = getattr(message, "tool_calls", None) or []
        if getattr(message, "role", "") != "assistant" or not calls:
            continue
        answered: set[str] = set()
        after = index + 1
        while after < len(messages) and getattr(messages[after], "role", "") == "tool":
            answered.add(str(messages[after].tool_call_id))
            after += 1
        missing.extend(call.id for call in calls if call.id not in answered)
    return missing


async def _cancel_bang_then_prompt(directory: Path, cwd: Path, *, adopt: bool) -> Any:
    """Cancel a turn mid-tool, run ``! echo one``, then prompt; return that request."""
    echo, started, release = _echo_parked()
    stream = ScriptedStream(
        [
            tool_call_turn(text="working", tool_name="echo", tool_call_id="c1", arguments={}),
            text_turn("NEXT"),
        ]
    )
    session = build_session(directory, stream, cwd=cwd, tools=[echo])
    try:
        task = asyncio.ensure_future(session.prompt("start"))
        await asyncio.wait_for(started.wait(), BOUND_S)
        if adopt:
            assert (
                await session.adopt_aside([Message.user(QUESTION), Message.assistant(ANSWER)])
                == "queued"
            )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        receipt = ToolResult(
            tool_call_id="shell-1", tool_name="bash", content=[TextContent(text="one")]
        )
        await session.record_shell("echo one", receipt)
        await session.prompt("next")
        return stream.requests[-1]
    finally:
        release.set()
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_the_adopt_adds_no_pairing_defect_to_the_next_request(
    headless_tui_env: Path,
) -> None:
    """The request after cell (a) obeys the wire rule as well as the bang-only one.

    A cancelled turn leaves ``c1`` unanswered in the middle of history, and a
    later bang receipt persists after it — that shape is ALREADY on origin/main
    with no adopt involved (the bang-only control below reproduces it; measured
    on ``aa01373ae`` too). The pin here is that the adopt adds nothing to it:
    the pair lands after the closed shell batch, and the set of unanswered calls
    in the provider request is identical with and without it.
    """
    with bounded(BOUND_S, "the next request after a cancelled turn"):
        control = await _cancel_bang_then_prompt(
            headless_tui_env / "sessions" / "pair-control", headless_tui_env, adopt=False
        )
        adopted = await _cancel_bang_then_prompt(
            headless_tui_env / "sessions" / "pair-adopt", headless_tui_env, adopt=True
        )

    assert _unanswered_calls(adopted) == _unanswered_calls(control) == ["c1"]
    control_shape = _request_shape(control)
    adopted_shape = _request_shape(adopted)
    # Identical apart from the pair, which sits after the shell batch's result
    # and before the new prompt — never between a call and its result.
    assert adopted_shape[:-1] == control_shape[:-1] + [
        ("user", QUESTION),
        ("assistant", ANSWER),
    ], adopted_shape
    assert adopted_shape[-1] == control_shape[-1] == ("user", "next")


@pytest.mark.asyncio
async def test_a_cancelled_turn_reports_an_adopt_that_never_lands(
    headless_tui_env: Path,
) -> None:
    """D7's terminal case: nothing closes the tail before dispose.

    The adopt cannot land on the open tail and dispose is the last boundary, so
    it is reported — one error notice for live frontends and a durable
    session-incident row for one that reconnects (review R6) — never dropped.
    """
    from local_operator.harness.message_types import SESSION_INCIDENT_MESSAGE_TYPE
    from local_operator.harness.types import NoticeEvent
    from local_operator.session.transcript import Transcript

    echo, started, release = _echo_parked()
    stream = ScriptedStream(
        [tool_call_turn(text="working", tool_name="echo", tool_call_id="c1", arguments={})]
    )
    directory = headless_tui_env / "sessions" / "adopt-stranded"
    session = build_session(directory, stream, cwd=headless_tui_env, tools=[echo])
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    pair = [Message.user(QUESTION), Message.assistant(ANSWER)]
    try:
        with bounded(BOUND_S, "a stranded queued adopt"):
            task = asyncio.ensure_future(session.prompt("start"))
            await asyncio.wait_for(started.wait(), BOUND_S)
            assert await session.adopt_aside(pair) == "queued"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await session.dispose()
    finally:
        release.set()
        await dispose_quietly(session)

    reopened = Transcript(directory)
    assert not reopened.has_entry(pair[-1].id)
    notices = [e for e in events if isinstance(e, NoticeEvent) and e.kind == "error"]
    assert len(notices) == 1 and "1 aside(s)" in notices[0].text, notices
    incidents = [
        e
        for e in reopened.entries()
        if e.payload.get("custom_type") == SESSION_INCIDENT_MESSAGE_TYPE
        and pair[-1].id in str(e.payload.get("details", {}).get("text", ""))
    ]
    assert len(incidents) == 1


@pytest.mark.asyncio
async def test_an_adopt_after_the_last_in_pipeline_flush_lands_before_the_lock_frees(
    headless_tui_env: Path,
) -> None:
    """D8, the reviewer's R2 probe as a cell.

    The pipeline's own flush is followed by more awaits before the lock is
    released (the patience flush among them). An adopt arriving there sees the
    lock held and queues; it must be durable by the time the lock is free,
    not stranded until some later turn.
    """
    stream = ScriptedStream([text_turn("DONE")])
    session = build_session(
        headless_tui_env / "sessions" / "adopt-window", stream, cwd=headless_tui_env
    )
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    gate, inside = asyncio.Event(), asyncio.Event()
    original = session._flush_patience_armed_after

    async def slow() -> None:
        inside.set()
        await gate.wait()
        await original()

    session._flush_patience_armed_after = slow  # type: ignore[method-assign]
    pair = [Message.user(QUESTION), Message.assistant(ANSWER)]
    try:
        with bounded(BOUND_S, "an adopt in the post-flush window"):
            task = asyncio.ensure_future(session.prompt("start"))
            await asyncio.wait_for(inside.wait(), BOUND_S)
            assert session._turn_lock.locked()
            assert await session.adopt_aside(pair) == "queued"
            gate.set()
            await asyncio.wait_for(task, BOUND_S)

            assert not session._turn_lock.locked()
            assert session._transcript.has_entry(pair[-1].id)
            assert session._pending_shell_records == []
            deltas = [e for e in events if isinstance(e, HistoryDeltaEvent)]
            assert [[m.id for m in d.messages] for d in deltas] == [[m.id for m in pair]]
    finally:
        gate.set()
        await dispose_quietly(session)
