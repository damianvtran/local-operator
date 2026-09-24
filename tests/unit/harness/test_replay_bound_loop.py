"""The bound as the loop actually applies it — on the request, not on history.

The unit tests prove the transformation. These prove the two things only the
loop's own seam can: that the request carried to the provider is the BOUNDED
view, and that the history the session keeps is not.

The negative control at the end is the reason this file exists rather than a
single assertion: with the bound switched off at its one call site, the same
request goes out carrying the full 8 MB-shaped row, which is what makes the
positive assertion evidence about the bound instead of about the fixture.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator.harness import loop as loop_module
from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.replay_bound import DEFAULT_REPLAY_BOUND_CHARS
from local_operator.harness.types import (
    ChatRequest,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    TextContent,
    ToolCall,
)
from local_operator.providers.clients import (
    OpenAICompatClient,
    _replayable_tool_arguments_json,
)

MODEL = ModelSpec(provider="test", model_id="test-model", context_window=200_000)


def lines(count: int, prefix: str = "output") -> str:
    return "".join(f"{prefix} {index}: value-{index}\n" for index in range(count))


class ScriptedStream:
    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self.turns = turns
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: Any):
        self.requests.append(request)
        turn = self.turns[len(self.requests) - 1]

        async def gen():
            for event in turn:
                yield event

        return gen()


def make_config(stream_fn: Any, **kwargs: Any) -> LoopConfig:
    defaults: dict[str, Any] = dict(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=stream_fn,
    )
    defaults.update(kwargs)
    return LoopConfig(**defaults)


def text_row(messages: Any) -> Message:
    """The tool row in ``messages``, narrowed.

    ``LoopContext.messages`` is ``list[AgentMessage]`` — the union with
    ``CustomMessage`` — so a tool row is only reachable behind a narrowing
    check, and the request's own list is the same type by construction.
    """
    row = next(m for m in messages if isinstance(m, Message) and m.role == "tool")
    assert isinstance(row.content[0], TextContent), "a tool row carries text here"
    return row


def call_row(messages: Any) -> Message:
    """The assistant row carrying a tool call in ``messages``, narrowed."""
    return next(m for m in messages if isinstance(m, Message) and m.tool_calls)


def block_text(message: Message) -> str:
    """The text of a row's first block, narrowed off the content union."""
    block = message.content[0]
    assert isinstance(block, TextContent)
    return block.text


class Fixture:
    """A history carrying one oversized result and one oversized call."""

    def __init__(self) -> None:
        self.result_text = lines(4000)  # ~100 KB, the shape that leaked
        self.argument_text = lines(4000)
        self.call = Message(
            role="assistant",
            content=[TextContent(text="writing the file")],
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="write",
                    arguments={"path": "big.py", "content": self.argument_text},
                    raw_arguments=json.dumps({"path": "big.py", "content": self.argument_text}),
                )
            ],
        )
        self.result = Message(
            role="tool",
            content=[TextContent(text=self.result_text)],
            tool_call_id="c1",
            tool_name="write",
        )
        self.history = [Message.user("go"), self.call, self.result]

    def context(self) -> LoopContext:
        return LoopContext(system_blocks=["sys"], tools=[], messages=list(self.history))


async def run_one_turn(context: LoopContext) -> ScriptedStream:
    stream = ScriptedStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    await AgentLoop().run_to_end(
        [Message.user("answer using the file you just wrote")],
        context,
        make_config(stream),
        None,
    )
    return stream


@pytest.mark.asyncio
async def test_the_request_goes_out_bounded():
    fixture = Fixture()
    context = fixture.context()

    stream = await run_one_turn(context)
    sent = stream.requests[0]

    result = text_row(sent.messages)
    call = call_row(sent.messages).tool_calls[0]

    assert len(block_text(result)) <= DEFAULT_REPLAY_BOUND_CHARS
    assert "elided on replay" in block_text(result)
    assert len(call.arguments["content"]) <= DEFAULT_REPLAY_BOUND_CHARS
    assert len(json.loads(_replayable_tool_arguments_json(call))["content"]) <= (
        DEFAULT_REPLAY_BOUND_CHARS
    )
    assert call.name == "write" and call.id == "c1"
    assert result.tool_call_id == "c1", "the pairing the provider validates is intact"


@pytest.mark.asyncio
async def test_the_session_history_is_not_the_thing_that_was_bounded():
    """The durable record keeps every byte; only the request was shortened."""
    fixture = Fixture()
    context = fixture.context()

    await run_one_turn(context)

    kept_result = text_row(context.messages)
    kept_call = call_row(context.messages).tool_calls[0]
    assert kept_result is fixture.result
    assert block_text(kept_result) == fixture.result_text
    assert kept_call is fixture.call.tool_calls[0]
    assert kept_call.arguments["content"] == fixture.argument_text
    assert kept_call.raw_arguments == json.dumps(
        {"path": "big.py", "content": fixture.argument_text}
    )


@pytest.mark.asyncio
async def test_the_bounded_request_is_still_a_valid_wire_body():
    """Bounding must not produce a body the provider would reject: the OpenAI
    wire shape is built from the bounded request and its arguments must parse."""
    fixture = Fixture()
    stream = await run_one_turn(fixture.context())

    body = OpenAICompatClient("https://compat.example/v1")._build_body(stream.requests[0])

    call = next(message for message in body["messages"] if message.get("tool_calls"))["tool_calls"][
        0
    ]
    assert json.loads(call["function"]["arguments"])["path"] == "big.py"
    assert len(json.loads(call["function"]["arguments"])["content"]) <= DEFAULT_REPLAY_BOUND_CHARS


@pytest.mark.asyncio
async def test_with_the_bound_switched_off_the_full_row_goes_out(monkeypatch):
    """The negative control: this is what the request looks like without the
    bound, so the assertions above are about the bound and not about a fixture
    that was already small."""
    monkeypatch.setattr(loop_module, "bound_replay_payloads", lambda messages, **kwargs: messages)
    fixture = Fixture()
    stream = await run_one_turn(fixture.context())

    result = text_row(stream.requests[0].messages)
    assert block_text(result) == fixture.result_text
    assert len(block_text(result)) > DEFAULT_REPLAY_BOUND_CHARS * 10
