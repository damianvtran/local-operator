"""AgentLoop tests: full turn shape, tool pairing on abort/length, steering
interrupts, validation errors back to the model, gates, follow-ups."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

import pytest

from local_operator.harness.loop import AgentLoop, LoopContext, validate_tool_arguments
from local_operator.harness.types import (
    AbortSignal,
    AgentEndEvent,
    AgentTool,
    AgentToolUpdate,
    Aside,
    ChatRequest,
    CustomMessage,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolCallComposeEvent,
    ToolContext,
    ToolResult,
)
from local_operator.providers.failover import ProviderError

MODEL = ModelSpec(provider="test", model_id="m")


class ScriptedStream:
    """Fake stream_fn: replays a per-call script of StreamEvents and records
    every ChatRequest it receives."""

    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self.turns = turns
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        turn = self.turns[len(self.requests) - 1]

        async def gen():
            for event in turn:
                yield event

        return gen()


def echo_tool(
    executed: list[str],
    name: str = "echo",
    concurrency: Literal["shared", "exclusive"] = "shared",
    delay: float = 0.0,
) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        if delay:
            await asyncio.sleep(delay)
        executed.append(name)
        return ToolResult(
            tool_call_id=tool_call_id, tool_name=name, content=[TextContent(text=f"ok:{args}")]
        )

    return AgentTool(
        name=name,
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        concurrency=concurrency,
        execute=execute,
    )


def make_config(stream_fn, **kwargs) -> LoopConfig:
    defaults: dict[str, Any] = dict(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=stream_fn,
    )
    defaults.update(kwargs)
    return LoopConfig(**defaults)


def tool_call_delta(index: int, *, id: str | None = None, name: str | None = None, args: str = ""):
    return StreamToolCallDelta(index=index, id=id, name=name, argument_delta=args)


@pytest.mark.asyncio
async def test_full_turn_text_tool_text():
    """Stream emits text, then a tool call (deltas split), then more text;
    after execution the model stops. Assert event order and final messages."""
    executed: list[str] = []
    stream = ScriptedStream(
        [
            [
                StreamTextDelta(delta="Hello "),
                tool_call_delta(0, id="call_1", name="echo"),
                tool_call_delta(0, args='{"text":'),
                tool_call_delta(0, args='"hi"}'),
                StreamTextDelta(delta="world"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="Done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(system_blocks=["sys"], tools=[echo_tool(executed)])
    config = make_config(stream)
    loop = AgentLoop()

    events = []
    async for event in loop.run([Message.user("go")], context, config, None):
        events.append(event)

    types = [e.type for e in events]
    assert types == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_update",
        # The model announces the call it is composing as soon as the tool's
        # NAME is known — long before the call exists for a large argument. A UI
        # with nothing between `message_end` and `tool_execution_start` has
        # nothing to paint while the arguments stream, which is the frozen frame
        # this event was added to remove.
        "tool_call_compose",
        "message_update",
        # The second is the end-of-stream flush: whatever the throttle swallowed
        # is announced before the call becomes an execution, so the row never
        # under-reports what the model actually wrote.
        "tool_call_compose",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "turn_end",
        "turn_start",
        "message_start",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert executed == ["echo"]

    final = events[-1]
    assert isinstance(final, AgentEndEvent)
    assert not final.aborted
    assert len(final.messages) == 3  # assistant1, tool result, assistant2

    assistant1 = final.messages[0]
    assert isinstance(assistant1, Message)
    assert assistant1.text == "Hello world"
    assert len(assistant1.tool_calls) == 1
    assert assistant1.tool_calls[0].name == "echo"
    assert assistant1.tool_calls[0].arguments == {"text": "hi"}
    assert assistant1.tool_calls[0].raw_arguments == '{"text":"hi"}'

    tool_msg = final.messages[1]
    assert isinstance(tool_msg, Message)
    assert tool_msg.role == "tool"
    assert tool_msg.tool_call_id == assistant1.tool_calls[0].id

    last = context.messages[-1]
    assert isinstance(last, Message)
    assert last.text == "Done"
    # System blocks and converted messages reached the provider.
    assert stream.requests[0].system_blocks == ["sys"]
    assert stream.requests[0].messages[0].text == "go"


@pytest.mark.asyncio
async def test_abort_pairs_dangling_tool_calls():
    """stop_reason 'aborted': every dangling tool call gets a synthetic
    is_error result; tools are NOT executed; agent_end is aborted."""
    executed: list[str] = []
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="echo", args='{"text":"a"}'),
                tool_call_delta(1, id="c2", name="echo", args='{"text":"b"}'),
                StreamEndEvent(stop_reason="aborted"),
            ]
        ]
    )
    context = LoopContext(tools=[echo_tool(executed)])
    loop = AgentLoop()

    events = []
    async for event in loop.run([Message.user("go")], context, make_config(stream), None):
        events.append(event)

    assert executed == []
    end = events[-1]
    assert isinstance(end, AgentEndEvent)
    assert end.aborted is True
    assert [e.type for e in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        # Two calls were being composed when the abort landed. The announcements
        # stay in the stream: they are what the UI already painted, and a turn
        # that stops has to reconcile those rows rather than pretend they never
        # appeared.
        "tool_call_compose",
        "tool_call_compose",
        "message_end",
        "turn_end",
        "agent_end",
    ]

    assistant = next(
        m for m in context.messages if isinstance(m, Message) and m.role == "assistant"
    )
    call_ids = {c.id for c in assistant.tool_calls}
    tool_messages = [m for m in context.messages if isinstance(m, Message) and m.role == "tool"]
    assert {m.tool_call_id for m in tool_messages} == call_ids
    assert all(m.is_error and m.text == "aborted" for m in tool_messages)


@pytest.mark.asyncio
async def test_length_pairs_but_does_not_execute():
    """stop_reason 'length': placeholders pair the dangling call (they go back
    to the model), but the tool is NEVER executed."""
    executed: list[str] = []
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="echo", args="{}"),
                StreamEndEvent(stop_reason="length"),
            ],
            # The placeholder result goes back to the model; it stops cleanly.
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[echo_tool(executed)])
    loop = AgentLoop()

    events = []
    async for event in loop.run([Message.user("go")], context, make_config(stream), None):
        events.append(event)

    assert executed == []
    end = events[-1]
    assert isinstance(end, AgentEndEvent)
    assert end.aborted is False
    tool_messages = [m for m in context.messages if isinstance(m, Message) and m.role == "tool"]
    assert len(tool_messages) == 1 and tool_messages[0].is_error


@pytest.mark.asyncio
async def test_invalid_arguments_go_back_to_model():
    """Validation failure never raises; the model receives an is_error result
    and may recover on the next turn."""
    executed: list[str] = []
    stream = ScriptedStream(
        [
            [
                # echo.text is typed string; 42 violates the schema.
                tool_call_delta(0, id="c1", name="echo", args='{"text": 42}'),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="fixed"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[echo_tool(executed)])
    loop = AgentLoop()

    messages = await loop.run_to_end([Message.user("go")], context, make_config(stream), None)

    assert executed == []
    tool_result = next(m for m in messages if isinstance(m, Message) and m.role == "tool")
    assert tool_result.is_error
    assert "Invalid arguments" in tool_result.text
    # The error result was sent on the second model call.
    assert any(m.role == "tool" and m.is_error for m in stream.requests[1].messages)


@pytest.mark.asyncio
async def test_unknown_tool_reports_error_result():
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="ghost", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[])
    messages = await AgentLoop().run_to_end(
        [Message.user("go")], context, make_config(stream), None
    )
    tool_result = next(m for m in messages if isinstance(m, Message) and m.role == "tool")
    assert tool_result.is_error and "Tool not found" in tool_result.text


@pytest.mark.asyncio
async def test_fallback_tool_resolution():
    executed: list[str] = []

    def resolver(name: str):
        return echo_tool(executed, name=name) if name == "deferred" else None

    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="deferred", args='{"text":"x"}'),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[])
    config = make_config(stream, resolve_fallback_tool=resolver)
    await AgentLoop().run_to_end([Message.user("go")], context, config, None)
    assert executed == ["deferred"]


@pytest.mark.asyncio
async def test_steering_interrupts_between_exclusive_calls():
    """interrupt_mode='immediate': after the first batch slot, a queued
    steering message skips the remaining calls with synthetic results."""
    executed: list[str] = []
    steering_flag = {"queued": False}

    async def slow_execute(tool_call_id, args, signal, on_update, context):
        executed.append("a")
        steering_flag["queued"] = True  # steering arrives while batch runs
        return ToolResult(tool_call_id=tool_call_id, tool_name="a", content=[TextContent(text="a")])

    tool_a = AgentTool(
        name="a", parameters={"type": "object"}, concurrency="exclusive", execute=slow_execute
    )
    tool_b = echo_tool(executed, name="b", concurrency="exclusive")

    drained = {"done": False}

    async def get_steering():
        if steering_flag["queued"] and not drained["done"]:
            drained["done"] = True
            return [Message.user("stop that")]
        return []

    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="a", args="{}"),
                tool_call_delta(1, id="c2", name="b", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[tool_a, tool_b])
    config = make_config(
        stream,
        interrupt_mode="immediate",
        get_steering_messages=get_steering,
        has_steering_messages=lambda: steering_flag["queued"] and not drained["done"],
    )

    messages = await AgentLoop().run_to_end([Message.user("go")], context, config, None)

    assert executed == ["a"]  # b never ran
    results = [m for m in messages if isinstance(m, Message) and m.role == "tool"]
    assert len(results) == 2
    skipped = next(m for m in results if m.tool_call_id == "c2")
    assert skipped.is_error and "skipped" in skipped.text.lower()
    # The steering message entered context and reached the second model call.
    assert any(
        isinstance(m, Message) and m.text == "stop that" for m in stream.requests[1].messages
    )


@pytest.mark.asyncio
async def test_follow_up_reenters_outer_loop():
    """A follow-up at the yield boundary re-enters the outer loop."""
    follow_ups = [[Message.user("one more thing")], []]
    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="first"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="second"), StreamEndEvent(stop_reason="stop")],
        ]
    )

    async def get_follow_ups():
        return follow_ups.pop(0) if follow_ups else []

    context = LoopContext(tools=[])
    config = make_config(stream, get_follow_up_messages=get_follow_ups)
    messages = await AgentLoop().run_to_end([Message.user("go")], context, config, None)

    assert len(stream.requests) == 2
    texts = [m.text for m in messages if isinstance(m, Message) and m.role == "assistant"]
    assert texts == ["first", "second"]
    assert any(isinstance(m, Message) and m.text == "one more thing" for m in context.messages)


@pytest.mark.asyncio
async def test_aside_thunks_commit_and_none_dropped():
    """Aside thunks are invoked at injection; None results are dropped;
    on_commit fires when a custom payload reaches the model."""
    committed: list[str] = []
    aside_msg = CustomMessage(
        custom_type="note",
        attribution="user",
        details={"n": 1},
        on_commit=lambda: committed.append("kept"),
    )

    aside_calls: list[list[Aside]] = [[lambda: aside_msg, lambda: None], []]

    async def get_asides():
        return aside_calls.pop(0) if aside_calls else []

    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="a"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="b"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[])
    config = make_config(stream, get_aside_messages=get_asides)
    await AgentLoop().run_to_end([Message.user("go")], context, config, None)

    assert committed == ["kept"]
    assert aside_msg in context.messages
    assert len(stream.requests) == 2


@pytest.mark.asyncio
async def test_before_model_call_gate_stops_run():
    stream = ScriptedStream([])
    context = LoopContext(tools=[])
    config = make_config(stream, before_model_call=lambda: False)

    events = []
    async for event in AgentLoop().run([Message.user("go")], context, config, None):
        events.append(event)

    assert stream.requests == []
    assert isinstance(events[-1], AgentEndEvent)
    assert events[-1].aborted is True
    assert "gate" in (events[-1].error or "")


@pytest.mark.asyncio
async def test_tool_execution_update_events():
    """on_update partial results surface as tool_execution_update events."""

    async def execute(tool_call_id, args, signal, on_update, context):
        on_update(AgentToolUpdate(content=[TextContent(text="partial")]))
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="u", content=[TextContent(text="fin")]
        )

    tool = AgentTool(name="u", parameters={"type": "object"}, execute=execute)
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="u", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[tool])
    events = []
    async for event in AgentLoop().run([Message.user("go")], context, make_config(stream), None):
        events.append(event)

    updates = [e for e in events if e.type == "tool_execution_update"]
    assert len(updates) == 1
    assert updates[0].partial_result.content[0].text == "partial"


@pytest.mark.asyncio
async def test_approval_denied_returns_error_result():
    """A denied approval produces an is_error ToolResult, never a raise."""

    async def deny(name: str, summary: str) -> bool:
        return False

    executed: list[str] = []
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="echo", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(
        tools=[echo_tool(executed)], tool_context=ToolContext(request_approval=deny)
    )
    messages = await AgentLoop().run_to_end(
        [Message.user("go")], context, make_config(stream), None
    )
    assert executed == []
    tool_result = next(m for m in messages if isinstance(m, Message) and m.role == "tool")
    assert tool_result.is_error and "approval" in tool_result.text.lower()


@pytest.mark.asyncio
async def test_shared_calls_run_in_parallel():
    """Two shared tools overlap in time (gather), both results arrive."""
    order: list[str] = []

    def make_execute(tag: str):
        async def execute(tool_call_id, args, signal, on_update, context):
            order.append(f"{tag}-start")
            await asyncio.sleep(0.01)
            order.append(f"{tag}-end")
            return ToolResult(
                tool_call_id=tool_call_id, tool_name=tag, content=[TextContent(text=tag)]
            )

        return execute

    tool_x = AgentTool(name="x", parameters={"type": "object"}, execute=make_execute("x"))
    tool_y = AgentTool(name="y", parameters={"type": "object"}, execute=make_execute("y"))
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="x", args="{}"),
                tool_call_delta(1, id="c2", name="y", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[tool_x, tool_y])
    messages = await AgentLoop().run_to_end(
        [Message.user("go")], context, make_config(stream), None
    )

    results = [m for m in messages if isinstance(m, Message) and m.role == "tool"]
    assert {m.tool_name for m in results} == {"x", "y"}
    # Both started before either finished -> parallel.
    assert order[:2] == ["x-start", "y-start"]


def test_validate_tool_arguments():
    async def noop(tool_call_id, args, signal, on_update, context):
        return ToolResult(tool_call_id=tool_call_id, content=[])

    tool = AgentTool(
        name="t",
        parameters={
            "type": "object",
            "properties": {"n": {"type": "integer"}, "s": {"type": "string"}},
            "required": ["n"],
        },
        execute=noop,
    )
    assert validate_tool_arguments(tool, {"n": 3}) == []
    missing = validate_tool_arguments(tool, {})
    assert any("required" in e for e in missing)
    wrong = validate_tool_arguments(tool, {"n": "nope"})
    assert any("'n'" in e for e in wrong)
    bad_json = validate_tool_arguments(tool, {}, raw_arguments="{broken")
    assert any("JSON" in e for e in bad_json)


@pytest.mark.asyncio
async def test_exec_tier_prompts_exactly_once_per_call():
    """The loop is the single write/exec gate: one approval prompt per call,
    fired after tool_execution_start. A second prompt at the tool level was
    the defect this pins (the user answered twice per action)."""
    prompts: list[tuple[str, str]] = []

    async def approve(name: str, summary: str) -> bool:
        prompts.append((name, summary))
        return True

    executed: list[str] = []
    tool = echo_tool(executed)
    tool = AgentTool(
        name=tool.name,
        parameters=tool.parameters,
        execute=tool.execute,
        approval_tier="exec",
    )
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="echo", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[tool], tool_context=ToolContext(request_approval=approve))
    await AgentLoop().run_to_end([Message.user("go")], context, make_config(stream), None)
    assert executed == ["echo"]
    assert len(prompts) == 1
    assert prompts[0][0] == "echo"


@pytest.mark.parametrize(
    "exc, wants_traceback",
    [
        (ProviderError(400, "`temperature` is deprecated for this model."), False),
        (RuntimeError("dict changed size during iteration"), True),
    ],
)
@pytest.mark.asyncio
async def test_rendered_provider_errors_are_logged_without_a_stack(exc, wants_traceback, caplog):
    """A provider's 400 is an answer, not a defect. The loop hands it to the UI
    as `error`, which the TUI prints as one "× HTTP 400: ..." line; logging the
    same failure a second time as a traceback painted forty lines of stack over
    the interface. A genuine bug keeps its stack — the frames are the only clue
    there is."""

    def boom(request, signal):
        async def gen():
            raise exc
            yield  # pragma: no cover - generator marker

        return gen()

    caplog.set_level(logging.WARNING, logger="local_operator.harness.loop")
    events = [
        e
        async for e in AgentLoop().run([Message.user("go")], LoopContext(), make_config(boom), None)
    ]

    end = next(e for e in events if isinstance(e, AgentEndEvent))
    assert str(exc) in (end.error or ""), "the caller must still receive the message"
    records = [r for r in caplog.records if r.message.startswith("model stream failed")]
    assert len(records) == 1
    assert bool(records[0].exc_info) is wants_traceback
    assert str(exc) in records[0].getMessage()


@pytest.mark.asyncio
async def test_a_composing_call_reports_its_final_size():
    """Whatever the throttle swallowed is flushed when the stream ends.

    Arguments commonly land in one burst inside a single throttle window, so
    without a flush the row's size reports a fraction of the call — or, when the
    whole payload arrives faster than one window, never displays a size at all.
    It matters most on an aborted turn, where the frozen row is what is left on
    screen.
    """
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="echo"),
                StreamToolCallDelta(index=0, argument_delta='{"text": "' + "x" * 4000 + '"}'),
                StreamEndEvent(stop_reason="tool_calls"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    executed: list[str] = []
    context = LoopContext(system_blocks=["sys"], tools=[echo_tool(executed)])
    loop = AgentLoop()

    composes = [
        event
        async for event in loop.run([Message.user("go")], context, make_config(stream), None)
        if isinstance(event, ToolCallComposeEvent)
    ]
    assert composes, "the call must be announced while it is being composed"
    assert composes[-1].argument_bytes == 4012
    # One id for the whole call: a key that changes mid-stream mounts a second row.
    assert len({event.tool_call_id for event in composes}) == 1


@pytest.mark.asyncio
async def test_a_late_call_id_does_not_change_the_compose_key():
    """The key is latched on the first announcement.

    An OpenAI-compatible endpoint may send the tool NAME before the call id.
    Recomputing the key each time changed it mid-stream, and the UI — which keys
    its rows by it — mounted a second row and then marked the abandoned one
    interrupted, for a call that had in fact succeeded.
    """
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, name="echo"),
                StreamToolCallDelta(index=0, argument_delta='{"text": '),
                StreamToolCallDelta(index=0, id="call_late"),
                StreamToolCallDelta(index=0, argument_delta='"hi"}'),
                StreamEndEvent(stop_reason="tool_calls"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    executed: list[str] = []
    context = LoopContext(system_blocks=["sys"], tools=[echo_tool(executed)])
    loop = AgentLoop()

    composes = [
        event
        async for event in loop.run([Message.user("go")], context, make_config(stream), None)
        if isinstance(event, ToolCallComposeEvent)
    ]
    assert len({event.tool_call_id for event in composes}) == 1
