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
