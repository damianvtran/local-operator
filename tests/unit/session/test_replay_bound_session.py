"""The bound through a real session, a real transcript and a real compaction.

QA's attack on this change is information loss, so the evidence has to be the
whole chain rather than the transformation in isolation:

* a long session whose tool result is far over the bound still hands the model
  an actionable row (the head it acts on, the tail it judges by, and a marker
  that states how much is missing rather than hiding it);
* the DURABLE transcript keeps every byte — asserted against the file on disk
  and against ``Transcript.build_llm_history()``, not against the in-memory
  list the bound happened to copy;
* the replay marker never reaches that file, before or after a compaction pass;
* compaction does not run ahead of the first request of a turn, which is what
  makes "the bound costs nothing on the critical path" a fact rather than a
  hope.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator.compaction import api as compaction_api
from local_operator.compaction.api import CompactionSettings
from local_operator.harness.replay_bound import DEFAULT_REPLAY_BOUND_CHARS
from local_operator.harness.types import (
    AgentEvent,
    AgentTool,
    ChatRequest,
    CompactionEndEvent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
    TurnStartEvent,
)
from local_operator.session.protocol import CompactionOutcome
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="ctx-probe", context_window=1_000_000)

#: ~90k chars: comfortably over the 8 KiB bound, and the shape real output has.
BIG_TEXT = "".join(f"row {index}: value-{index}\n" for index in range(6000))

#: Kept small so a couple of turns leave history outside the kept window, which
#: is what gives ``find_cut_point`` something to cut when a pass is forced.
KEEP_RECENT = 40


class ScriptedStream:
    """Replays per-call event scripts; records every request it is handed."""

    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self.turns = turns
        self.requests: list[ChatRequest] = []
        self.compactions_seen: list[int] = []

    def __call__(self, request: ChatRequest, signal: Any):
        self.requests.append(request)
        self.compactions_seen.append(self.compactions_before_request())
        turn = self.turns[min(len(self.requests) - 1, len(self.turns) - 1)]

        async def gen():
            for event in turn:
                yield event

        return gen()

    #: Set by the test to the live event list, so the stream fn can record how
    #: many compaction passes had already finished when each request went out.
    events: list[AgentEvent] | None = None

    def compactions_before_request(self) -> int:
        return sum(1 for event in (self.events or []) if isinstance(event, CompactionEndEvent))


def dump_tool() -> AgentTool:
    """A tool whose result is over the bound and is NOT capped by the tools layer.

    The real leak shape: nothing on the execution path imposes a size on a
    custom tool's output, so the row it writes is whatever the tool returned.
    """

    async def execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="dump",
            content=[TextContent(text=BIG_TEXT)],
        )

    return AgentTool(
        name="dump",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


def make_session(tmp_path, stream, tools=None, **kwargs) -> Session:
    return Session(
        model=kwargs.pop("model", MODEL),
        stream_fn=stream,
        tools=tools or [],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=kwargs.pop(
            "compaction_settings", CompactionSettings(keep_recent_tokens=KEEP_RECENT)
        ),
        **kwargs,
    )


def text_at(message: Message, index: int = 0) -> str:
    """The text of ``message``'s block ``index``, narrowed off the content union."""
    block = message.content[index]
    assert isinstance(block, TextContent), f"block {index} is not text"
    return block.text


def tool_call_turn(call_id: str = "c1", name: str = "dump") -> list[StreamEvent]:
    return [
        StreamToolCallDelta(index=0, id=call_id, name=name, argument_delta="{}"),
        StreamEndEvent(stop_reason="toolUse"),
    ]


def reply_turn() -> list[StreamEvent]:
    return [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]


async def run_two_turns(tmp_path, stream: ScriptedStream) -> Session:
    """Turn 1 produces a huge tool row; turn 2 is the request we assert on."""
    session = make_session(tmp_path, stream, tools=[dump_tool()])
    await session.prompt("dump everything")
    await session.prompt("now use what you dumped")
    return session


@pytest.mark.asyncio
async def test_the_second_request_is_bounded_and_the_model_keeps_the_ends(tmp_path):
    stream = ScriptedStream([tool_call_turn(), reply_turn(), reply_turn()])
    session = await run_two_turns(tmp_path, stream)

    second = stream.requests[1]
    row = next(m for m in second.messages if m.role == "tool" and m.tool_name == "dump")
    text = text_at(row)

    assert len(text) <= DEFAULT_REPLAY_BOUND_CHARS
    assert "elided on replay" in text, "the loss is stated, never silent"
    # The ends are what a model acts on: the first row says what came back, the
    # last says how it ended. Both survive, so "act on the truncated material"
    # is a claim the request supports.
    assert text.startswith("row 0: value-0\n")
    assert text.rstrip().endswith("row 5999: value-5999")
    assert str(len(BIG_TEXT)) in text or "characters elided" in text
    # The elided count is the real one, not a placeholder.
    elided = int(text.split("[... ")[1].split(" characters")[0])
    assert 0 < elided < len(BIG_TEXT)
    assert row.tool_call_id == "c1", "the call/result pairing the provider validates survives"
    await session.dispose()


@pytest.mark.asyncio
async def test_the_durable_transcript_keeps_every_byte(tmp_path):
    """The stored transcript is the durable record; the bound is a wire concern."""
    stream = ScriptedStream([tool_call_turn(), reply_turn(), reply_turn()])
    session = await run_two_turns(tmp_path, stream)
    directory = session._transcript.directory
    await session.dispose()

    raw = (directory / "transcript.jsonl").read_text()
    # The marker has no newline in it, so a raw-bytes search is valid for it;
    # the row's own text is JSON-escaped on disk (newlines as ``\n``), so the
    # full-text claim is taken through the decoder instead of the file bytes.
    assert "elided on replay" not in raw, "the replay marker must never reach the durable record"
    stored = [
        "".join(block.get("text", "") for block in json.loads(line)["payload"]["content"])
        for line in raw.splitlines()
        if json.loads(line).get("type") == "message"
        and json.loads(line)["payload"].get("role") == "tool"
    ]
    assert stored == [BIG_TEXT], "the full row is on disk, byte for byte"

    replayed = [
        message
        for message in Transcript(directory).build_llm_history()
        if isinstance(message, Message) and message.role == "tool" and message.tool_name == "dump"
    ]
    assert replayed and text_at(replayed[0]) == BIG_TEXT


@pytest.mark.asyncio
async def test_compaction_never_runs_before_the_first_request_of_a_turn(tmp_path, monkeypatch):
    """The pass is post-turn work; nothing about it may delay a request."""
    stream = ScriptedStream([tool_call_turn(), reply_turn(), reply_turn()])
    session = make_session(tmp_path, stream, tools=[dump_tool()])
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    stream.events = events

    await session.prompt("dump everything")
    await session.prompt("now use what you dumped")

    assert stream.compactions_seen[:2] == [
        0,
        0,
    ], "no compaction pass had finished when either of the first two requests went out"
    first_turn = next(
        index for index, event in enumerate(events) if isinstance(event, TurnStartEvent)
    )
    assert not [
        event for event in events[:first_turn] if isinstance(event, CompactionEndEvent)
    ], "nothing may compact ahead of the turn's first request"

    # And the pass does run once the turn is over, when the gate says so.
    pin_measured_context(monkeypatch, 900_000)
    await session.prompt("one more")
    assert [
        e for e in events if isinstance(e, CompactionEndEvent)
    ], "a forced pass must still run — the ordering claim is not a claim that it never runs"
    await session.dispose()


@pytest.mark.asyncio
async def test_the_request_after_a_compaction_is_still_bounded(tmp_path, monkeypatch):
    """The bound survives the pass that rewrites history, and the pass does not
    write the elided text into the durable record."""
    stream = ScriptedStream([tool_call_turn(), reply_turn(), reply_turn()])
    session = make_session(tmp_path, stream, tools=[dump_tool()])
    await session.prompt("dump everything")
    pin_measured_context(monkeypatch, 900_000)

    plan = await session._plan_compaction(respect_threshold=True)
    assert not isinstance(plan, CompactionOutcome), f"expected a plan, got {plan!r}"
    outcome = await session._run_compaction(plan, reason="context-window")
    assert getattr(outcome, "ran", True)

    await session.prompt("carry on")
    last = stream.requests[-1]
    oversized = [
        m
        for m in last.messages
        if m.role == "tool" and len(text_at(m)) > DEFAULT_REPLAY_BOUND_CHARS
    ]
    assert not oversized, "every replayed tool row is within the bound after a pass"
    directory = session._transcript.directory
    await session.dispose()
    assert "elided on replay" not in (directory / "transcript.jsonl").read_text()


@pytest.mark.asyncio
async def test_a_bounded_request_still_serializes_to_a_provider_body(tmp_path):
    """The end of the chain: the bounded request builds a valid OpenAI body, so
    the pairing and the JSON survive the elision rather than merely the fields."""
    from local_operator.providers.clients import OpenAICompatClient

    stream = ScriptedStream([tool_call_turn(), reply_turn(), reply_turn()])
    session = await run_two_turns(tmp_path, stream)
    body = OpenAICompatClient("https://compat.example/v1")._build_body(stream.requests[1])
    rows = [m for m in body["messages"] if m.get("role") == "tool"]
    assert rows and all(
        len(json.dumps(row.get("content"))) <= DEFAULT_REPLAY_BOUND_CHARS + 512 for row in rows
    )
    await session.dispose()


def pin_measured_context(monkeypatch, tokens: int) -> None:
    """Pin both rulers the gate consults, so the trigger is a number not a size."""
    monkeypatch.setattr(compaction_api, "messages_tokens_upper_bound", lambda messages: tokens)
    monkeypatch.setattr(compaction_api, "estimate_messages_tokens", lambda messages: tokens)
