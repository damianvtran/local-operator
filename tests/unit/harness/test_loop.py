"""AgentLoop tests: full turn shape, tool pairing on abort/length, steering
interrupts, validation errors back to the model, gates, follow-ups."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import Counter
from typing import Any, Literal

import httpx
import pytest

import local_operator.harness.loop as loop_module
from local_operator.harness.loop import (
    ABORT_DRAIN_TIMEOUT_S,
    MAX_CONNECTIVITY_CONTINUATIONS,
    STEERING_INTERRUPT_POLL_S,
    AgentLoop,
    LoopContext,
    _consume_claim,
    _get_before_timeout,
    validate_tool_arguments,
)
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
    MessageStartEvent,
    ModelSpec,
    NoticeEvent,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsageEvent,
    TextContent,
    ToolCallComposeEvent,
    ToolContext,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
    TurnEndEvent,
    TurnStartEvent,
    Usage,
)
from local_operator.providers.failover import (
    ProviderError,
    _mark_mid_stream_connectivity,
    wrap_transport_error,
)

MODEL = ModelSpec(provider="test", model_id="m")


class _ControlledDeadline:
    """Keep the abort drain live until a test-controlled parking boundary."""

    def __init__(self, parked: asyncio.Event) -> None:
        self.parked = parked

    def __sub__(self, _now: float) -> float:
        return -1.0 if self.parked.is_set() else 60.0


class ControlledDrainBudget:
    """Replace the numeric drain budget without adding a production test seam."""

    def __init__(self, parked: asyncio.Event) -> None:
        self.parked = parked

    def __radd__(self, _now: float) -> _ControlledDeadline:
        return _ControlledDeadline(self.parked)


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
async def test_a_tool_only_turn_carries_no_empty_text_block() -> None:
    """A turn that calls a tool and says nothing must keep ``content == []``.

    Anthropic rejects a message carrying an empty text block outright:

        HTTP 400: messages: text content blocks must be non-empty

    and it is the NEXT request of the turn that fails, so the run dies after
    the tool has already run. "Call the tool, say nothing" is an ordinary
    model turn — it is what most tool steps look like — so this is not an
    edge case.

    The assembly used to live inside the delta loop, where it was simply
    unreachable when no text delta arrived. Hoisting it out of the loop (to
    stop the per-delta rebuild being quadratic in response length) made it run
    unconditionally, which turned "no text" into an empty text block rather
    than no content at all. The guard is what keeps the optimization from
    changing what a silent turn sends.
    """

    executed: list[str] = []
    stream = ScriptedStream(
        [
            # No StreamTextDelta at all: tool call only.
            [
                tool_call_delta(0, id="call_1", name="echo"),
                tool_call_delta(0, args='{"text":"hi"}'),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(system_blocks=["sys"], tools=[echo_tool(executed)])
    loop = AgentLoop()

    events = []
    async for event in loop.run([Message.user("go")], context, make_config(stream), None):
        events.append(event)

    final = events[-1]
    assert isinstance(final, AgentEndEvent)
    assistant = final.messages[0]
    assert isinstance(assistant, Message)
    assert assistant.content == [], "a silent turn must not carry a text block"
    assert len(assistant.tool_calls) == 1
    assert executed == ["echo"]

    # The regression is only observable on the SECOND request, which is the one
    # that carries the first assistant message back to the provider.
    replayed = stream.requests[1].messages
    assert not any(
        getattr(block, "text", None) == ""
        for message in replayed
        for block in (message.content or [])
    ), "an empty text block reached the provider"


@pytest.mark.asyncio
async def test_a_tool_result_is_redacted_before_it_reaches_the_model() -> None:
    """The ``read`` path the reviewer reproduced: a credential written by one
    tool and read by another must not survive into the tool message."""
    from local_operator.variables import VariableStore

    secret = "leaked-by-read-before-the-fix"
    store = VariableStore(cwd="/tmp", env={})
    store.store_credential("LO_TEST_TOKEN", secret, "command")

    async def read_execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="read",
            content=[TextContent(text=secret)],
        )

    read_tool = AgentTool(
        name="read",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=read_execute,
    )
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="read", args='{"path": "/tmp/secret"}'),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[read_tool])
    config = make_config(stream, redact_tool_result=store.redact)
    loop = AgentLoop()

    events = []
    async for event in loop.run([Message.user("go")], context, config, None):
        events.append(event)

    tool_msg = next(m for m in events[-1].messages if isinstance(m, Message) and m.role == "tool")
    assert secret not in tool_msg.text
    assert tool_msg.text == "[redacted]"


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
async def test_courtesy_steering_waits_for_the_running_tool():
    """has_urgent_steering_messages separates may-interrupt steering from
    courtesy injections: with it answering False, a queued message rides the
    queue PAST the running interruptible tool and is delivered at the next
    boundary instead of cancelling the call."""
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    steering_flag = {"queued": False, "drained": False}

    async def blocking_execute(tool_call_id, args, signal, on_update, context):
        tool_started.set()
        await release_tool.wait()
        return ToolResult(tool_call_id=tool_call_id, tool_name="a", content=[TextContent(text="a")])

    tool_a = AgentTool(
        name="a",
        parameters={"type": "object"},
        interruptible=True,
        execute=blocking_execute,
    )

    async def get_steering():
        if steering_flag["queued"] and not steering_flag["drained"]:
            steering_flag["drained"] = True
            return [Message.user("scheduled wake")]
        return []

    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="a", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[tool_a])
    config = make_config(
        stream,
        interrupt_mode="immediate",
        get_steering_messages=get_steering,
        has_steering_messages=lambda: steering_flag["queued"] and not steering_flag["drained"],
        has_urgent_steering_messages=lambda: False,  # nothing here may interrupt
    )
    loop = AgentLoop()
    events: list[Any] = []

    async def run() -> None:
        async for event in loop.run([Message.user("go")], context, config, None):
            events.append(event)

    task = asyncio.ensure_future(run())
    await asyncio.wait_for(tool_started.wait(), timeout=5)
    steering_flag["queued"] = True
    # A poll cycle passes with the tool still running: no cancellation.
    await asyncio.sleep(STEERING_INTERRUPT_POLL_S * 3)
    assert not task.done(), "courtesy steering cancelled the running tool"
    release_tool.set()
    await asyncio.wait_for(task, timeout=5)

    # The tool produced its REAL result (no synthetic skip), and the steering
    # message was delivered into the follow-up model call.
    tool_messages = [m for m in context.messages if isinstance(m, Message) and m.role == "tool"]
    assert len(tool_messages) == 1 and not tool_messages[0].is_error
    assert any(
        isinstance(m, Message) and m.text == "scheduled wake" for m in stream.requests[1].messages
    )


@pytest.mark.asyncio
async def test_urgent_steering_still_interrupts_a_running_tool():
    """The urgent peek answering True keeps the immediate semantics: the
    running interruptible tool is cancelled and the steer jumps the queue."""
    tool_started = asyncio.Event()
    outcome: dict[str, str] = {}
    drained = {"done": False}

    async def get_steering():
        if drained["done"]:
            return []
        drained["done"] = True
        return [Message.user("stop that")]

    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="block", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(
        tools=[_blocking_tool("block", tool_started, interruptible=True, outcome=outcome)]
    )
    config = make_config(
        stream,
        interrupt_mode="immediate",
        get_steering_messages=get_steering,
        has_steering_messages=lambda: True,
        has_urgent_steering_messages=lambda: True,
    )

    messages = await AgentLoop().run_to_end([Message.user("go")], context, config, None)

    assert outcome == {"block": "cancelled"}
    results = [m for m in messages if isinstance(m, Message) and m.role == "tool"]
    assert results[0].is_error and "skipped" in results[0].text.lower()


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
async def test_todo_reminder_follow_up_reenters_and_stays_invisible():
    """The todo continuation guardrail, at the loop's own boundary.

    The session hands ``get_follow_up_messages`` a ``CustomMessage`` when todos
    are still open (``Session._todo_continuation``); the loop must re-enter for
    it — a turn that ended here is the bug the guardrail exists to fix — and it
    must do so without emitting an event or reporting the reminder in the run's
    messages, which is what keeps the nudge out of the transcript and off the
    user's screen. The real session renderer is pinned in
    ``tests/unit/session/test_todo_guardrail.py``; this stands in for it.
    """
    from local_operator.tools.builtin import TODO_REMINDER_MESSAGE_TYPE

    reminder = CustomMessage(
        custom_type=TODO_REMINDER_MESSAGE_TYPE,
        attribution="system",
        details={"text": "<system-reminder>still open: ship it</system-reminder>"},
    )
    follow_ups: list[list[Any]] = [[reminder], []]
    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="answered in prose"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="back to work"), StreamEndEvent(stop_reason="stop")],
        ]
    )

    async def get_follow_ups():
        return follow_ups.pop(0) if follow_ups else []

    def convert(messages):
        # Mirrors the session's allow-list: a custom reminder renders as a user
        # turn. Without this the loop would re-enter with nothing to react to.
        out = []
        for message in messages:
            if isinstance(message, Message):
                out.append(message)
            elif message.custom_type == TODO_REMINDER_MESSAGE_TYPE:
                out.append(Message.user(message.details["text"]))
        return out

    context = LoopContext(tools=[])
    config = make_config(stream, convert_to_llm=convert, get_follow_up_messages=get_follow_ups)

    events = []
    async for event in AgentLoop().run([Message.user("go")], context, config, None):
        events.append(event)

    # Re-entered: a second provider request happened at all.
    assert len(stream.requests) == 2
    assert reminder in context.messages
    assert any("still open: ship it" in m.text for m in stream.requests[1].messages)
    # Invisible: no event carries it, and the run never reports it as a message
    # the host should persist.
    end = events[-1]
    assert isinstance(end, AgentEndEvent)
    assert reminder not in end.messages
    assert not any(getattr(event, "message", None) is reminder for event in events)


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


class TestABrokenApprovalGateIsNotAUserRefusal:
    """The reported defect: two bash calls read ``User denied approv… ✕ 0.0s``
    in a session whose band said ``! auto-approve``.

    The gate had raised — a widget failing mid-render inside the TUI's
    callback — and ``except Exception: approved = False`` attributed our bug to
    the user as a deliberate refusal. Two things were wrong with that. It blames
    the user, so the report that comes back is "it denied my command" and nobody
    looks for the exception; and it turns a crash in a SECURITY gate into a quiet
    policy outcome, when a gate that could not run has not decided anything.
    """

    def _stream(self) -> ScriptedStream:
        return ScriptedStream(
            [
                [
                    tool_call_delta(0, id="c1", name="echo", args="{}"),
                    StreamEndEvent(stop_reason="toolUse"),
                ],
                [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
            ]
        )

    async def _run(self, gate: Any) -> tuple[ToolResult, list[str]]:
        executed: list[str] = []
        result: list[ToolResult] = []
        context = LoopContext(
            tools=[echo_tool(executed)], tool_context=ToolContext(request_approval=gate)
        )
        async for event in AgentLoop().run(
            [Message.user("go")], context, make_config(self._stream()), None
        ):
            if isinstance(event, ToolExecutionEndEvent):
                result.append(event.result)
        return result[0], executed

    @pytest.mark.asyncio
    async def test_the_failure_names_itself_and_the_exception(self) -> None:
        async def broken(name: str, summary: str) -> bool:
            raise AttributeError("'NoneType' object has no attribute 'update'")

        result, executed = await self._run(broken)
        assert executed == [], "a gate that cannot answer has granted nothing"
        assert result.is_error
        text = result.text
        assert "denied" not in text.lower(), "the user did not deny anything"
        assert text.startswith("Approval gate failed for 'echo'")
        assert "AttributeError: 'NoneType' object has no attribute 'update'" in text
        assert "not a refusal by the user" in text

    @pytest.mark.asyncio
    async def test_the_first_line_alone_carries_the_diagnosis(self) -> None:
        """The TUI labels a failed card with ``_first_line(result_text)`` and
        then clips it, which is how the whole story reached the owner as
        ``User denied approv…``. So line one has to be the diagnosis, not a
        preamble to it."""

        async def broken(name: str, summary: str) -> bool:
            raise RuntimeError("widget exploded")

        result, _ = await self._run(broken)
        first = result.text.splitlines()[0]
        assert first == "Approval gate failed for 'echo' — the call was not run."
        # And the two outcomes are still distinguishable clipped to a narrow card.
        assert first[:14] != "User denied ap"[:14]

    @pytest.mark.asyncio
    async def test_an_argumentless_exception_still_says_what_it_was(self) -> None:
        """``str(TimeoutError())`` is the empty string — the same defect as a
        ``ProviderError`` that renders ``HTTP 404:`` and nothing else."""

        async def broken(name: str, summary: str) -> bool:
            raise TimeoutError()

        result, _ = await self._run(broken)
        assert "TimeoutError" in result.text

    @pytest.mark.asyncio
    async def test_a_host_can_tell_the_two_apart_without_reading_prose(self) -> None:
        """``__synthetic`` cannot: a refusal and a broken gate are the same shape
        and opposite meanings."""

        async def broken(name: str, summary: str) -> bool:
            raise RuntimeError("boom")

        async def deny(name: str, summary: str) -> bool:
            return False

        failed, _ = await self._run(broken)
        refused, _ = await self._run(deny)
        assert failed.details == {"__synthetic": True, "__approval_gate_failed": True}
        assert refused.details == {"__synthetic": True}

    @pytest.mark.asyncio
    async def test_the_exception_is_findable_in_the_log_with_its_stack(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """At ERROR, not WARNING: a fault inside a security gate is not weather.
        The stack is the only thing there is to diagnose from."""

        async def broken(name: str, summary: str) -> bool:
            raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR, logger="local_operator.harness.loop"):
            await self._run(broken)
        record = next(r for r in caplog.records if "approval gate raised" in r.getMessage())
        assert record.levelno == logging.ERROR
        assert record.exc_info is not None
        assert "the call was NOT run" in record.getMessage()

    @pytest.mark.asyncio
    async def test_a_real_refusal_is_untouched(self) -> None:
        """The other half must keep saying exactly what it said: the user's own
        decision is not a fault and must not be dressed up as one."""

        async def deny(name: str, summary: str) -> bool:
            return False

        result, executed = await self._run(deny)
        assert executed == []
        assert result.text == "User denied approval for 'echo'."

    @pytest.mark.asyncio
    async def test_a_broken_gate_does_not_abort_the_turn(self) -> None:
        """Judgement call, pinned: the failure is loud but it is still a RESULT.
        Every call has to come back paired (see ``_execute_batch``), and the
        sibling handler already answers a raising TOOL with an error result
        rather than a dead turn — so trading a misleading result for an aborted
        turn would be a different bug, not a fix."""

        async def broken(name: str, summary: str) -> bool:
            raise RuntimeError("boom")

        context = LoopContext(
            tools=[echo_tool([])], tool_context=ToolContext(request_approval=broken)
        )
        messages = await AgentLoop().run_to_end(
            [Message.user("go")], context, make_config(self._stream()), None
        )
        # The turn completed: the tool result was paired and the model answered.
        assert any(
            isinstance(m, Message) and m.role == "assistant" and m.text == "ok" for m in messages
        )


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


class TestSystemBlocksAreReadAtEveryCall:
    """Live session instructions reach the next model step of the same run."""

    @staticmethod
    def _two_call_stream() -> ScriptedStream:
        return ScriptedStream(
            [
                [
                    tool_call_delta(0, id="c1", name="echo", args="{}"),
                    StreamEndEvent(stop_reason="toolUse"),
                ],
                [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
            ]
        )

    @pytest.mark.asyncio
    async def test_a_goal_change_between_calls_lands_on_the_next_one(self):
        live = [["stable", "goal: old"]]
        stream = self._two_call_stream()

        async def execute(tool_call_id, args, signal, on_update, context):
            # The user runs /goal while this tool is in flight.
            live[0] = ["stable", "goal: new"]
            return ToolResult(
                tool_call_id=tool_call_id, tool_name="echo", content=[TextContent(text="ok")]
            )

        tool = AgentTool(name="echo", parameters={"type": "object"}, execute=execute)
        context = LoopContext(system_blocks=["stable", "goal: old"], tools=[tool])
        config = make_config(stream, get_system_blocks=lambda _model: live[0])

        async for _ in AgentLoop().run([Message.user("go")], context, config, None):
            pass

        assert [request.system_blocks for request in stream.requests] == [
            ["stable", "goal: old"],
            ["stable", "goal: new"],
        ]

    @pytest.mark.asyncio
    async def test_an_async_block_resolver_is_supported(self):
        stream = self._two_call_stream()
        executed: list[str] = []
        context = LoopContext(system_blocks=["snapshot"], tools=[echo_tool(executed)])

        async def resolve(_model: ModelSpec) -> list[str]:
            await asyncio.sleep(0)
            return ["live"]

        async for _ in AgentLoop().run(
            [Message.user("go")], context, make_config(stream, get_system_blocks=resolve), None
        ):
            pass

        assert [request.system_blocks for request in stream.requests] == [["live"], ["live"]]

    @pytest.mark.asyncio
    async def test_goal_change_on_step_start_reaches_the_not_yet_started_call(self):
        live = [["goal: old"]]
        stream = ScriptedStream(
            [[StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]]
        )
        context = LoopContext(system_blocks=["goal: old"])
        events = AgentLoop().run(
            [Message.user("go")],
            context,
            make_config(stream, get_system_blocks=lambda _model: live[0]),
            None,
        )

        async for event in events:
            if event.type == "turn_start":
                live[0] = ["goal: new"]

        assert stream.requests[0].system_blocks == ["goal: new"]

    @pytest.mark.asyncio
    async def test_model_is_resolved_after_an_async_block_refresh(self):
        new = ModelSpec(provider="test", model_id="new")
        current = [MODEL]
        stream = ScriptedStream(
            [[StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]]
        )
        context = LoopContext(system_blocks=["snapshot"])

        async def blocks(model: ModelSpec) -> list[str]:
            await asyncio.sleep(0)
            current[0] = new
            return [f"Model: {model.model_id}"]

        async for _ in AgentLoop().run(
            [Message.user("go")],
            context,
            make_config(stream, get_system_blocks=blocks, get_model=lambda: current[0]),
            None,
        ):
            pass

        # A change DURING an async block build belongs to the next call; this
        # request stays internally consistent instead of pairing new-model wire
        # options with an old-model environment block.
        assert stream.requests[0].model == MODEL
        assert stream.requests[0].system_blocks == ["Model: m"]

    @pytest.mark.asyncio
    async def test_a_broken_block_resolver_uses_the_snapshot(self, caplog):
        stream = self._two_call_stream()
        executed: list[str] = []
        context = LoopContext(system_blocks=["snapshot"], tools=[echo_tool(executed)])

        def broken(_model: ModelSpec) -> list[str]:
            raise RuntimeError("block resolver exploded")

        with caplog.at_level(logging.ERROR):
            async for _ in AgentLoop().run(
                [Message.user("go")], context, make_config(stream, get_system_blocks=broken), None
            ):
                pass

        assert [request.system_blocks for request in stream.requests] == [
            ["snapshot"],
            ["snapshot"],
        ]
        assert "block resolver exploded" in caplog.text


class TestTheModelIsReadAtEveryCall:
    """A model switched mid-run reaches the run's NEXT provider call.

    A run is a chain of provider calls with tool batches between them, and
    ``LoopConfig.model`` is bound once when the host builds the config. On its
    own that pinned every call of a run to the model it started on, so a user
    switching model while the agent worked saw the switch ignored until their
    next message. ``get_model`` is asked once per call instead.
    """

    @staticmethod
    def _two_call_stream() -> ScriptedStream:
        return ScriptedStream(
            [
                [
                    tool_call_delta(0, id="c1", name="echo", args="{}"),
                    StreamEndEvent(stop_reason="toolUse"),
                ],
                [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
            ]
        )

    @staticmethod
    def _labels(stream: ScriptedStream) -> list[str]:
        return [f"{r.model.provider}/{r.model.model_id}" for r in stream.requests]

    @pytest.mark.asyncio
    async def test_a_switch_between_calls_lands_on_the_next_one(self):
        """The point of the feature: the second call of the SAME run switches."""
        new = ModelSpec(provider="test", model_id="new")
        current = [MODEL]
        stream = self._two_call_stream()

        async def execute(tool_call_id, args, signal, on_update, context):
            # The user runs /model while this tool is in flight.
            current[0] = new
            return ToolResult(
                tool_call_id=tool_call_id, tool_name="echo", content=[TextContent(text="ok")]
            )

        tool = AgentTool(name="echo", parameters={"type": "object"}, execute=execute)
        context = LoopContext(system_blocks=["sys"], tools=[tool])
        config = make_config(stream, get_model=lambda: current[0])

        async for _ in AgentLoop().run([Message.user("go")], context, config, None):
            pass

        assert self._labels(stream) == ["test/m", "test/new"]

    @pytest.mark.asyncio
    async def test_a_config_without_a_resolver_still_runs(self):
        """Every embedder and test double builds a LoopConfig with no ``get_model``."""
        stream = self._two_call_stream()
        executed: list[str] = []
        context = LoopContext(system_blocks=["sys"], tools=[echo_tool(executed)])

        async for _ in AgentLoop().run([Message.user("go")], context, make_config(stream), None):
            pass

        assert self._labels(stream) == ["test/m", "test/m"]

    @pytest.mark.asyncio
    async def test_a_broken_resolver_does_not_lose_the_turn(self, caplog):
        """A host accessor that raises is a host bug, not a reason to bin the work.

        Discriminating: it asserts the run COMPLETED on the snapshot model, so a
        resolver that raised into the loop would fail this rather than be
        silently tolerated by a test that only checked the label.
        """
        stream = self._two_call_stream()
        executed: list[str] = []
        context = LoopContext(system_blocks=["sys"], tools=[echo_tool(executed)])

        def broken() -> ModelSpec:
            raise RuntimeError("host accessor exploded")

        config = make_config(stream, get_model=broken)
        ends = []
        with caplog.at_level(logging.ERROR):
            async for event in AgentLoop().run([Message.user("go")], context, config, None):
                if isinstance(event, AgentEndEvent):
                    ends.append(event)

        assert self._labels(stream) == ["test/m", "test/m"]
        assert ends and ends[-1].error is None and not ends[-1].aborted
        assert "host accessor exploded" in caplog.text

    @pytest.mark.asyncio
    async def test_a_resolver_returning_none_falls_back_to_the_snapshot(self):
        """``None`` means "nothing better to say", not "call a model of None"."""
        stream = self._two_call_stream()
        executed: list[str] = []
        context = LoopContext(system_blocks=["sys"], tools=[echo_tool(executed)])
        config = make_config(stream, get_model=lambda: None)

        async for _ in AgentLoop().run([Message.user("go")], context, config, None):
            pass

        assert self._labels(stream) == ["test/m", "test/m"]


class TestTheContextHintIsStampedPerRequestByTheLoop:
    """``ChatRequest.context_tokens_hint`` is stamped by the LOOP, per call.

    The hint picks the Anthropic prompt-cache TTL, and the loop is the owner
    of the conversation its calls belong to. Two contracts (review F8/F9):
    the host's ``get_context_tokens_hint`` is only the cross-turn SEED, and
    once a call in this run reports ``Usage.context_tokens`` that count wins
    for the rest of the run — a subagent is one turn for its whole life, and
    a tool loop crosses the threshold long before the host's figure moves.
    """

    @staticmethod
    def _three_call_stream(reported: list[int | None]) -> ScriptedStream:
        def turn(context: int | None, more: bool) -> list[StreamEvent]:
            events: list[StreamEvent] = []
            if more:
                events.append(tool_call_delta(0, id="c", name="echo", args="{}"))
            if context is not None:
                events.append(StreamUsageEvent(usage=Usage(input_tokens=1, context_tokens=context)))
            events.append(StreamEndEvent(stop_reason="toolUse" if more else "stop"))
            return events

        return ScriptedStream(
            [turn(reported[0], True), turn(reported[1], True), turn(reported[2], False)]
        )

    @staticmethod
    def _hints(stream: ScriptedStream) -> list[int | None]:
        return [r.context_tokens_hint for r in stream.requests]

    async def _run(self, stream: ScriptedStream, **kwargs: Any) -> None:
        executed: list[str] = []
        context = LoopContext(system_blocks=["sys"], tools=[echo_tool(executed)])
        async for _ in AgentLoop().run(
            [Message.user("go")], context, make_config(stream, **kwargs), None
        ):
            pass

    @pytest.mark.asyncio
    async def test_the_seed_covers_the_first_call_and_in_run_counts_take_over(self):
        """Call 1 carries the host's seed; call N+1 carries what call N reported."""
        stream = self._three_call_stream([160_000, 170_000, 180_000])

        await self._run(stream, get_context_tokens_hint=lambda: 140_000)

        assert self._hints(stream) == [140_000, 160_000, 170_000]

    @pytest.mark.asyncio
    async def test_without_a_seed_the_first_call_is_unstamped(self):
        """No host callback (a subagent's first request, a bare embedder): the
        client's own estimate decides call 1, and the run still learns its
        size from call 1's report — the case the turn-boundary-only update
        never covered."""
        stream = self._three_call_stream([160_000, 170_000, 180_000])

        await self._run(stream)

        assert self._hints(stream) == [None, 160_000, 170_000]

    @pytest.mark.asyncio
    async def test_a_call_that_reports_no_count_keeps_the_last_one(self):
        """A wire that omits ``context_tokens`` must not blank the hint —
        that would send a large context out at 5m by the byte estimate."""
        stream = self._three_call_stream([160_000, None, 180_000])

        await self._run(stream, get_context_tokens_hint=lambda: None)

        assert self._hints(stream) == [None, 160_000, 160_000]


# ---------------------------------------------------------------------------
# Immediate abort: Esc must halt the turn NOW, not at the next natural boundary
# ---------------------------------------------------------------------------


def _blocking_tool(
    name: str,
    started: asyncio.Event,
    *,
    interruptible: bool,
    outcome: dict[str, str],
) -> AgentTool:
    """A tool that parks forever and records how its run ended.

    ``outcome`` is written by the tool itself, so a test can tell "the abort
    cancelled it" apart from "the tool finished on its own and the loop merely
    stopped waiting" — a distinction the loop's own events cannot make.
    """

    async def execute(tool_call_id, args, signal, on_update, context):
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            outcome[name] = "cancelled"
            raise
        outcome[name] = "completed"
        return ToolResult(
            tool_call_id=tool_call_id, tool_name=name, content=[TextContent(text="late")]
        )

    return AgentTool(
        name=name,
        parameters={"type": "object", "properties": {}},
        interruptible=interruptible,
        execute=execute,
    )


@pytest.mark.parametrize("interruptible", [False, True])
@pytest.mark.asyncio
async def test_abort_cancels_a_running_tool_whatever_its_interruptible_flag(interruptible):
    """Esc stops a tool mid-run even when it never opted into interruption.

    ``interruptible`` means "steering may redirect this", which is a different
    and weaker permission than "the user may stop this". Before this, a batch
    of non-interruptible calls ignored the abort completely and the turn ended
    only when the slowest one finished; the user's stop appeared to do nothing.
    """
    started = asyncio.Event()
    outcome: dict[str, str] = {}
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="block", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(
        tools=[_blocking_tool("block", started, interruptible=interruptible, outcome=outcome)]
    )
    config = make_config(stream, interrupt_mode="immediate", has_steering_messages=lambda: False)
    signal = AbortSignal()
    loop = AgentLoop()

    events: list[Any] = []

    async def run() -> None:
        async for event in loop.run([Message.user("go")], context, config, signal):
            events.append(event)

    task = asyncio.ensure_future(run())
    await asyncio.wait_for(started.wait(), timeout=5)
    signal.abort("interrupted")
    # Generous relative to the ~0s the fix achieves, but far below the 30 s the
    # tool would otherwise run: the assertion is "promptly", not a stopwatch.
    await asyncio.wait_for(task, timeout=5)

    assert outcome == {"block": "cancelled"}
    end = events[-1]
    assert isinstance(end, AgentEndEvent)
    assert end.aborted is True
    # Still paired: an abort must not leave a tool_use without its tool_result,
    # or the next request is rejected outright by the provider.
    tool_messages = [m for m in context.messages if isinstance(m, Message) and m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["c1"]
    assert all(m.is_error for m in tool_messages)


@pytest.mark.asyncio
async def test_abort_cancels_every_tool_in_a_parallel_batch():
    """One press stops the whole batch, not just the call that noticed first."""
    started_a, started_b = asyncio.Event(), asyncio.Event()
    outcome: dict[str, str] = {}
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="a", args="{}"),
                tool_call_delta(1, id="c2", name="b", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(
        tools=[
            _blocking_tool("a", started_a, interruptible=False, outcome=outcome),
            _blocking_tool("b", started_b, interruptible=True, outcome=outcome),
        ]
    )
    config = make_config(stream, interrupt_mode="immediate", has_steering_messages=lambda: False)
    signal = AbortSignal()
    loop = AgentLoop()

    async def run() -> None:
        async for _ in loop.run([Message.user("go")], context, config, signal):
            pass

    task = asyncio.ensure_future(run())
    await asyncio.wait_for(asyncio.gather(started_a.wait(), started_b.wait()), timeout=5)
    signal.abort("interrupted")
    await asyncio.wait_for(task, timeout=5)

    assert outcome == {"a": "cancelled", "b": "cancelled"}
    tool_messages = [m for m in context.messages if isinstance(m, Message) and m.role == "tool"]
    assert {m.tool_call_id for m in tool_messages} == {"c1", "c2"}


@pytest.mark.asyncio
async def test_abort_cuts_a_stalled_model_stream():
    """A model that has gone quiet is dropped on abort, not waited out.

    The provider stream sits in an ``await`` between tokens, so an abort used
    to take effect only when the NEXT event arrived. On a stalled or
    slow-reasoning stream that is seconds of a UI painting a turn the user has
    already stopped.
    """
    reached_second_token = False
    first_token = asyncio.Event()

    def stream_fn(request: ChatRequest, signal: AbortSignal | None):
        async def gen():
            nonlocal reached_second_token
            yield StreamTextDelta(delta="thinking")
            first_token.set()
            await asyncio.sleep(30)  # the model goes quiet
            reached_second_token = True
            yield StreamTextDelta(delta="never")
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    context = LoopContext(tools=[])
    config = make_config(stream_fn, interrupt_mode="immediate", has_steering_messages=lambda: False)
    signal = AbortSignal()
    loop = AgentLoop()

    events: list[Any] = []

    async def run() -> None:
        async for event in loop.run([Message.user("go")], context, config, signal):
            events.append(event)

    task = asyncio.ensure_future(run())
    await asyncio.wait_for(first_token.wait(), timeout=5)
    signal.abort("interrupted")
    await asyncio.wait_for(task, timeout=5)

    assert not reached_second_token, "the stalled stream was waited out instead of dropped"
    end = events[-1]
    assert isinstance(end, AgentEndEvent)
    assert end.aborted is True
    # The text produced BEFORE the abort survives: a stop keeps what was said.
    assistant = next(
        m for m in context.messages if isinstance(m, Message) and m.role == "assistant"
    )
    assert assistant.text == "thinking"


@pytest.mark.asyncio
async def test_an_unaborted_stream_is_untouched_by_the_abort_wrapper():
    """The abort-aware pull must not drop, reorder or duplicate events."""
    stream = ScriptedStream(
        [
            [
                StreamTextDelta(delta="a"),
                StreamTextDelta(delta="b"),
                tool_call_delta(0, id="c1", name="echo", args='{"text":"x"}'),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    executed: list[str] = []
    context = LoopContext(tools=[echo_tool(executed)])
    signal = AbortSignal()  # present but never fired
    loop = AgentLoop()

    events = []
    async for event in loop.run([Message.user("go")], context, make_config(stream), signal):
        events.append(event)

    assert executed == ["echo"]
    end = events[-1]
    assert isinstance(end, AgentEndEvent)
    assert end.aborted is False
    assistants = [m for m in context.messages if isinstance(m, Message) and m.role == "assistant"]
    assert assistants[0].text == "ab"
    assert assistants[-1].text == "done"


@pytest.mark.asyncio
async def test_a_provider_failure_still_surfaces_through_the_abort_wrapper():
    """The stream wrapper must not swallow errors.

    It drains the provider on a pump task, so a failure now reaches the loop as
    that task's exception rather than as a raise from the ``async for``. If it
    were dropped, a dead provider would look like a clean empty turn — the loop
    would report success and the user would see a turn that did nothing.
    """

    def stream_fn(request: ChatRequest, signal: AbortSignal | None):
        async def gen():
            yield StreamTextDelta(delta="partial")
            raise ProviderError(500, "upstream exploded", retryable=False)

        return gen()

    context = LoopContext(tools=[])
    signal = AbortSignal()  # present but never fired
    loop = AgentLoop()

    events = []
    async for event in loop.run([Message.user("go")], context, make_config(stream_fn), signal):
        events.append(event)

    end = events[-1]
    assert isinstance(end, AgentEndEvent)
    assert end.aborted is False, "a provider failure is an error, not a user abort"
    assert end.error is not None and "upstream exploded" in end.error


@pytest.mark.asyncio
async def test_a_signal_aborted_before_the_run_does_not_wedge_the_turn():
    """Review round 1, B1. An abort that has ALREADY fired must end the run.

    The stream is drained by a pump task, and ``ensure_future`` only SCHEDULES
    it — so a cancel landing before the body runs executed no statement inside
    it, ``finally`` included. Waking the consumer from that ``finally`` left
    the drain parked on a notification nobody would ever send, and the turn
    never ended: `is_streaming` stayed True and every later prompt was
    rejected, from the very keypress this feature is about.

    Reachable without touching internals: `Session._emit` awaits every handler,
    so an Esc during `turn_end` delivery lands after the loop's post-batch
    abort check and before the next model call.
    """
    stream = ScriptedStream([[StreamTextDelta(delta="hi"), StreamEndEvent(stop_reason="stop")]])
    context = LoopContext(tools=[])
    signal = AbortSignal()
    signal.abort("interrupted")  # fired BEFORE the run starts
    loop = AgentLoop()

    events = []

    async def run() -> None:
        async for event in loop.run([Message.user("go")], context, make_config(stream), signal):
            events.append(event)

    # The bug was an infinite hang; the timeout IS the assertion.
    await asyncio.wait_for(run(), timeout=5)

    end = events[-1]
    assert isinstance(end, AgentEndEvent)
    assert end.aborted is True


@pytest.mark.parametrize("interruptible", [False, True])
@pytest.mark.asyncio
async def test_an_aborted_tool_still_emits_its_end_event(interruptible):
    """Review round 1, B2. Every start event needs its end event.

    The batch-wide abort watcher cancels the runner coroutine from OUTSIDE.
    ``interruptible_runner``'s handler only caught its INNER task's
    cancellation, so an outer cancel unwound past ``park`` — which is what
    emits ``tool_execution_end``. The backfill kept the wire legal, but it
    emits no events, so every consumer other than the TUI was left with a tool
    that never finished: the API server holds the record IN_PROGRESS forever
    and never publishes a TOOL_END. It hits bash, eval, wait, hub, ask, web
    search and every MCP tool, which set ``interruptible=True``.
    """
    started = asyncio.Event()
    outcome: dict[str, str] = {}
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="block", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(
        tools=[_blocking_tool("block", started, interruptible=interruptible, outcome=outcome)]
    )
    config = make_config(stream, interrupt_mode="immediate", has_steering_messages=lambda: False)
    signal = AbortSignal()
    loop = AgentLoop()

    events: list[Any] = []

    async def run() -> None:
        async for event in loop.run([Message.user("go")], context, config, signal):
            events.append(event)

    task = asyncio.ensure_future(run())
    await asyncio.wait_for(started.wait(), timeout=5)
    signal.abort("interrupted")
    await asyncio.wait_for(task, timeout=10)

    starts = [e for e in events if isinstance(e, ToolExecutionStartEvent)]
    ends = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert len(starts) == 1
    assert len(ends) == len(starts), "a started tool card was never settled"
    assert {e.tool_call_id for e in ends} == {e.tool_call_id for e in starts}


@pytest.mark.asyncio
async def test_a_slow_tool_unwind_does_not_hold_the_turn_open():
    """Review round 1, M1. ``ABORT_DRAIN_TIMEOUT_S`` must actually bound.

    A cancelled tool is entitled to unwind — bash kills its process group — but
    not to hold the turn while it does. The first implementation bounded the
    wrong wait: the drain sat until every task had settled, so the ``finally``
    budget had nothing left to bound and a six-second unwind still cost six
    seconds. That is this feature's own bug, moved from the tool body into its
    cleanup.
    """
    started = asyncio.Event()
    unwind = 6.0

    async def execute(tool_call_id, args, signal, on_update, context) -> ToolResult:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(unwind)  # a process group refusing to die
            raise
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="stubborn", content=[TextContent(text="late")]
        )

    tool = AgentTool(
        name="stubborn",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="stubborn", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[tool])
    config = make_config(stream, interrupt_mode="immediate", has_steering_messages=lambda: False)
    signal = AbortSignal()
    loop = AgentLoop()

    async def run() -> None:
        async for _ in loop.run([Message.user("go")], context, config, signal):
            pass

    task = asyncio.ensure_future(run())
    await asyncio.wait_for(started.wait(), timeout=5)
    began = time.monotonic()
    signal.abort("interrupted")
    await asyncio.wait_for(task, timeout=unwind + 10)
    elapsed = time.monotonic() - began

    assert elapsed < unwind - 1, (
        f"the turn waited {elapsed:.1f}s for a {unwind}s unwind despite a "
        f"{ABORT_DRAIN_TIMEOUT_S}s budget"
    )
    # Still paired, even though the tool never parked its own result.
    tool_messages = [m for m in context.messages if isinstance(m, Message) and m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["c1"]


@pytest.mark.asyncio
async def test_a_slow_unwind_still_reports_the_tool_as_ENDED():
    """Review round 2, R2. The drain timeout must not swallow the end event.

    A tool whose cleanup outruns ``ABORT_DRAIN_TIMEOUT_S`` had its start event
    announced and no end event ever: the backfill repaired the RESULTS so the
    wire stayed legal, but emitted nothing, so a consumer holding execution
    records by id kept that one IN_PROGRESS forever and never published a
    TOOL_END on its stream. The TUI survives it by retiring orphaned cards at
    the turn boundary; the API server does not.

    That is the same damage ``interruptible_runner``'s cancellation handler
    exists to prevent — it closes the fast path, and this closes the slow one.
    """
    started = asyncio.Event()

    async def execute(tool_call_id, args, signal, on_update, context) -> ToolResult:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(ABORT_DRAIN_TIMEOUT_S + 2)  # outruns the budget
            raise
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="stubborn", content=[TextContent(text="late")]
        )

    tool = AgentTool(
        name="stubborn",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="stubborn", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[tool])
    config = make_config(stream, interrupt_mode="immediate", has_steering_messages=lambda: False)
    signal = AbortSignal()
    loop = AgentLoop()

    events: list[Any] = []

    async def run() -> None:
        async for event in loop.run([Message.user("go")], context, config, signal):
            events.append(event)

    task = asyncio.ensure_future(run())
    await asyncio.wait_for(started.wait(), timeout=5)
    signal.abort("interrupted")
    await asyncio.wait_for(task, timeout=ABORT_DRAIN_TIMEOUT_S + 10)

    starts = [e for e in events if isinstance(e, ToolExecutionStartEvent)]
    ends = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert len(starts) == 1, "the tool never announced its start"
    assert len(ends) == 1, (
        "the tool was announced as started and never as ended, so a consumer "
        "holding it by id keeps that execution IN_PROGRESS forever"
    )
    assert ends[0].tool_call_id == starts[0].tool_call_id
    assert ends[0].is_error, "an aborted tool must not be reported as a clean success"


@pytest.mark.asyncio
async def test_a_late_parking_tool_is_not_robbed_of_its_end_event_mid_backfill(monkeypatch):
    """The final flush must emit an end parked after the drain has expired.

    Two calls make the boundary observable: c1 parks immediately on abort, while
    c2 cannot finish cancellation until the consumer is suspended on c1's end.
    At that exact yield, c2 parks synchronously and the controlled budget expires.
    The backfill sees c2's filled slot and owes no synthetic end, leaving the
    final queue flush as the only path that can emit c2's real end event.
    """
    started = asyncio.Event()
    parked = asyncio.Event()
    allow_park = asyncio.Event()
    started_count = 0
    monkeypatch.setattr(loop_module, "ABORT_DRAIN_TIMEOUT_S", ControlledDrainBudget(parked))

    async def parks_immediately(tool_call_id, args, signal, on_update, context) -> ToolResult:
        nonlocal started_count
        started_count += 1
        if started_count == 2:
            started.set()
        await asyncio.sleep(30)
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="first", content=[TextContent(text="late")]
        )

    async def parks_at_flush_boundary(tool_call_id, args, signal, on_update, context) -> ToolResult:
        nonlocal started_count
        started_count += 1
        if started_count == 2:
            started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.CancelledError):
                await allow_park.wait()
            # No await may separate this signal from the runner's synchronous
            # park: the consumer must resume only after c2 has filled its slot
            # and queued the end event that solely the final flush can emit.
            parked.set()
            raise
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="second", content=[TextContent(text="late")]
        )

    tools = [
        AgentTool(
            name="first",
            parameters={"type": "object", "properties": {}},
            execute=parks_immediately,
        ),
        AgentTool(
            name="second",
            parameters={"type": "object", "properties": {}},
            execute=parks_at_flush_boundary,
        ),
    ]
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="first", args="{}"),
                tool_call_delta(1, id="c2", name="second", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=tools)
    config = make_config(stream, interrupt_mode="immediate", has_steering_messages=lambda: False)
    signal = AbortSignal()
    loop = AgentLoop()
    events: list[Any] = []

    async def run() -> None:
        async for event in loop.run([Message.user("go")], context, config, signal):
            events.append(event)
            if isinstance(event, ToolExecutionEndEvent) and event.tool_call_id == "c1":
                # Every yielded event reaches an awaiting consumer, matching the
                # TUI paint/API write shape rather than draining synchronously.
                allow_park.set()
                await asyncio.wait_for(parked.wait(), timeout=5)
            await asyncio.sleep(0)

    task = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), timeout=5)
    signal.abort("interrupted")
    await asyncio.wait_for(task, timeout=15)

    started_ids = [e.tool_call_id for e in events if isinstance(e, ToolExecutionStartEvent)]
    ended_ids = [e.tool_call_id for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert sorted(started_ids) == ["c1", "c2"]
    for call_id in started_ids:
        assert ended_ids.count(call_id) == 1, (
            f"{call_id} was announced as started and ended {ended_ids.count(call_id)} "
            f"times; every started call needs exactly one end (ends={ended_ids})"
        )

    tool_messages = [m for m in context.messages if isinstance(m, Message) and m.role == "tool"]
    assert sorted(str(m.tool_call_id) for m in tool_messages) == ["c1", "c2"]


@pytest.mark.asyncio
async def test_a_call_that_never_started_is_never_announced_as_ended():
    """Review round 4, R4-1. The flush must share the backfill's own guard.

    A planning failure — an unknown tool name, a duplicate call id — is parked
    up front by ``park()``, which queues an end event for a call that never
    emitted a START. That was harmless only because nothing drained the queue
    afterwards; the flush added for R3-1 reads it and announces the end of a
    call no consumer ever saw begin, which is the mirror image of the bug the
    flush exists to fix and is what the backfill a few lines below explicitly
    refuses to do.

    Reachable from a hallucinated tool name alone: no abort, no timing window.
    """
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="no_such_tool", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[])
    config = make_config(stream, interrupt_mode="immediate", has_steering_messages=lambda: False)
    loop = AgentLoop()

    events: list[Any] = []
    async for event in loop.run([Message.user("go")], context, config, None):
        events.append(event)

    started_ids = {e.tool_call_id for e in events if isinstance(e, ToolExecutionStartEvent)}
    ended_ids = [e.tool_call_id for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert started_ids == set(), "a call with no resolvable tool must never start"
    assert [
        c for c in ended_ids if c not in started_ids
    ] == [], f"end event(s) {ended_ids} announced for calls that never started"
    # The wire is still paired: the failure's result reaches the model.
    tool_messages = [m for m in context.messages if isinstance(m, Message) and m.role == "tool"]
    assert [str(m.tool_call_id) for m in tool_messages] == ["c1"]


@pytest.mark.asyncio
async def test_a_duplicate_call_id_does_not_suppress_the_real_calls_end_event():
    """Review round 5, R5-1. The flush's suppression must not be keyed by id.

    Call ids are NOT unique within a batch: a model can emit two calls with one
    id, and the loop keeps the first and turns the second into a planning
    failure. That leaves one id owned by two slots — one that STARTED and one
    that never did — so a suppression set keyed by id cannot tell them apart
    and swallows the genuine end event along with the parked one. The started
    call then keeps its start forever with no end, which is the same consumer
    damage the flush exists to prevent, arrived at from the other side.

    Counting per id is what makes it exact: swallow as many as are owed, no
    more. This asserts the invariant that survives either implementation —
    every call that STARTED gets exactly one end.
    """
    started = asyncio.Event()

    async def slow(tool_call_id, args, signal, on_update, context) -> ToolResult:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(ABORT_DRAIN_TIMEOUT_S + 0.3)
            raise
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="slow", content=[TextContent(text="late")]
        )

    tool = AgentTool(
        name="slow",
        parameters={"type": "object", "properties": {}},
        execute=slow,
    )
    stream = ScriptedStream(
        [
            [
                # One id, twice: the second is dropped to a planning failure.
                tool_call_delta(0, id="dup", name="slow", args="{}"),
                tool_call_delta(1, id="dup", name="slow", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[tool])
    config = make_config(stream, interrupt_mode="immediate", has_steering_messages=lambda: False)
    signal = AbortSignal()
    loop = AgentLoop()

    events: list[Any] = []

    async def run() -> None:
        async for event in loop.run([Message.user("go")], context, config, signal):
            events.append(event)
            await asyncio.sleep(0.05)

    task = asyncio.ensure_future(run())
    await asyncio.wait_for(started.wait(), timeout=5)
    signal.abort("interrupted")
    await asyncio.wait_for(task, timeout=ABORT_DRAIN_TIMEOUT_S + 15)

    started_ids = [e.tool_call_id for e in events if isinstance(e, ToolExecutionStartEvent)]
    ended_ids = [e.tool_call_id for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert started_ids == ["dup"], "exactly one of the colliding calls should run"
    assert ended_ids.count("dup") >= 1, (
        "the started call's end event was suppressed along with the duplicate's, "
        "leaving a start with no end"
    )
    # The duplicate never started, so it must not add an end of its own.
    assert ended_ids.count("dup") == len(
        started_ids
    ), f"expected one end per started call, got ends={ended_ids} starts={started_ids}"


@pytest.mark.asyncio
async def test_a_call_that_never_ran_still_reports_itself_to_the_operator():
    """Review round 6, R6-3. Suppressing the end event must not mean silence.

    `park` withholds the end event for a call that never started, which is
    right — but the headless renderer printed `✗ <name> failed` off exactly
    that event, so withholding it alone turned a visible diagnostic into
    nothing at all: an operator watching a hallucinated tool name saw the run
    simply produce no output about it.

    A notice is the honest carrier. It reports the failure without claiming a
    lifecycle that never began, and the model is unaffected either way because
    it still receives the parked `tool_result`.
    """
    executed: list[str] = []
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="echo", args='{"text":"hi"}'),
                tool_call_delta(1, id="c2", name="no_such_tool", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[echo_tool(executed)])
    config = make_config(stream, interrupt_mode="immediate", has_steering_messages=lambda: False)
    loop = AgentLoop()

    events: list[Any] = []
    async for event in loop.run([Message.user("go")], context, config, None):
        events.append(event)

    notices = [e.text for e in events if isinstance(e, NoticeEvent)]
    assert any(
        "no_such_tool" in text for text in notices
    ), f"the unresolvable call produced no operator-visible diagnostic: {notices}"
    # Still no orphan end event, which is what R4-1/R5-1 fixed.
    started_ids = {e.tool_call_id for e in events if isinstance(e, ToolExecutionStartEvent)}
    ended_ids = [e.tool_call_id for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert [c for c in ended_ids if c not in started_ids] == []
    # And the model still gets both results, so the wire stays legal.
    tool_messages = [m for m in context.messages if isinstance(m, Message) and m.role == "tool"]
    assert sorted(str(m.tool_call_id) for m in tool_messages) == ["c1", "c2"]


def test_the_flush_suppression_counts_rather_than_matching():
    """Review round 6/7, R6-2 and R7-3. Pin the mechanism R5-1 was about.

    Exercises the REAL helper rather than a retyped copy of its logic. The
    behavioural tests cannot reach this branch: the source-side guard in `park`
    removes the collision it defends against (R6-1), so the code path is
    unreachable and a revert to a plain `set` leaves every other test green.
    A test that reproduced the rule inline had the same blind spot as the code
    it was meant to guard (R7-3) — this calls the function the loop calls.

    The rule: with N queued ends carrying one id and K owed suppression,
    exactly N-K survive. Matching by id instead of counting swallows both, and
    the one it must not swallow is the started call's genuine end.
    """
    # Two slots, one id: one started, one a duplicate that never did, so
    # exactly one suppression is owed.
    claimed: Counter[str] = Counter(["dup"])

    survived = [call_id for call_id in ("dup", "dup") if not _consume_claim(claimed, call_id)]

    assert survived == ["dup"], (
        "counting must swallow only what is owed; matching by id swallows the "
        "started call's genuine end event too, which is R5-1"
    )
    # And the claim is spent, not standing: a later end for the same id lives.
    assert _consume_claim(claimed, "dup") is False
    # An id nobody claimed is never suppressed.
    assert _consume_claim(Counter(), "other") is False


@pytest.mark.asyncio
async def test_wait_for_can_lose_an_item_delivered_at_its_timeout_boundary(monkeypatch):
    """Pin the production race that requires an explicit getter/timer race."""

    class ControlledTimeout:
        def __init__(self) -> None:
            self.armed = asyncio.Event()
            self.task: asyncio.Task[Any] | None = None
            self.expiring = False

        async def __aenter__(self):
            self.task = asyncio.current_task()
            self.armed.set()
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            if self.expiring and exc_type is asyncio.CancelledError:
                raise TimeoutError from exc
            return None

        def fire(self) -> None:
            self.expiring = True
            assert self.task is not None
            self.task.cancel()

    queue: asyncio.Queue[object] = asyncio.Queue()
    item = object()
    getter = asyncio.create_task(queue.get())
    await asyncio.sleep(0)  # Register the getter before arranging the boundary.
    controlled = ControlledTimeout()
    monkeypatch.setattr(asyncio.timeouts, "timeout", lambda _delay: controlled)
    waiting = asyncio.create_task(asyncio.wait_for(getter, timeout=30))
    await controlled.armed.wait()

    # Queue delivery runs first and dequeues the item; timeout cancellation then
    # wins before wait_for observes that result, reproducing the lost-item race.
    queue.put_nowait(item)
    asyncio.get_running_loop().call_soon(controlled.fire)

    with pytest.raises(TimeoutError):
        await waiting
    assert queue.empty()
    assert getter.done() and not getter.cancelled()
    assert getter.result() is item


@pytest.mark.asyncio
async def test_queue_get_wins_when_it_ties_the_timeout():
    queue = asyncio.Queue()
    item = object()
    queue.put_nowait(item)

    assert await _get_before_timeout(queue, timeout=0) is item
    assert queue.empty()


@pytest.mark.asyncio
async def test_queue_timeout_cancels_and_awaits_its_pending_getter():
    queue = asyncio.Queue()
    current = asyncio.current_task()
    tasks_before = {task for task in asyncio.all_tasks() if task is not current}

    with pytest.raises(TimeoutError):
        await _get_before_timeout(queue, timeout=0)

    tasks_after = {task for task in asyncio.all_tasks() if task is not current}
    assert tasks_after == tasks_before
    item = object()
    queue.put_nowait(item)
    await asyncio.sleep(0)
    assert queue.get_nowait() is item


@pytest.mark.asyncio
async def test_caller_cancellation_leaves_a_dequeued_item_in_the_queue():
    """A cancelled caller must not consume an item and drop it.

    The helper's whole purpose is that an item it takes off the queue is either
    returned or put back. The first version held that only for the TIMEOUT
    path: its cleanup cancelled and joined both contestants unconditionally, so
    a caller cancelled in the same turn a delivery landed discarded the item the
    getter had already dequeued. That is a silent divergence from the
    ``wait_for`` this replaced, which never touches an already-completed getter
    and so leaves the item in the queue (R1-1, agent review round 1).

    Unreachable as damage from today's sole caller — the abort drain is already
    unwinding and abandons the queue — but the divergence is exactly the class
    of bug this helper exists to prevent, so the invariant gets a test rather
    than an argument about reachability.

    Deterministic, not probabilistic: the loop below advances the event loop one
    turn at a time and cancels only once the getter has provably dequeued the
    item (``queue.empty()``), which is the precise state the old cleanup
    mishandled. Against the pre-fix helper this fails 200/200; after, 200/200
    preserve the item.
    """
    queue: asyncio.Queue[object] = asyncio.Queue()
    item = object()
    current = asyncio.current_task()
    tasks_before = {task for task in asyncio.all_tasks() if task is not current}

    # A timeout far past any scheduling jitter: the timer must never be the
    # reason this call ends, or the test would prove the already-covered path.
    caller = asyncio.create_task(_get_before_timeout(queue, timeout=30))
    await asyncio.sleep(0)  # Let the helper create and register its contestants.
    await asyncio.sleep(0)
    queue.put_nowait(item)

    # Spin until the getter has taken the item but the caller has not yet
    # resumed to receive it. Bounded so a helper that never dequeues fails here
    # rather than hanging.
    for _ in range(50):
        if queue.empty():
            break
        await asyncio.sleep(0)
    else:  # pragma: no cover - a helper that never dequeues is already broken
        pytest.fail("the getter never dequeued the item")
    assert not caller.done(), "caller resumed before it could be cancelled mid-delivery"

    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    # The item is back, and it is THE item: a reclaim that put back something
    # else would satisfy a bare length check.
    assert queue.qsize() == 1
    assert queue.get_nowait() is item

    # And the reclaim did not come at the cost of the property the timeout path
    # already had: no contestant survives this call.
    tasks_after = {task for task in asyncio.all_tasks() if task is not current}
    assert tasks_after == tasks_before


# ---------------------------------------------------------------------------
# Refusal terminal: the model said no, and the run must end saying WHY
# ---------------------------------------------------------------------------


class TestRefusalEndsTheRunVisibly:
    """``stop_reason="refusal"`` used to fall through to the clean-stop path:
    the loop treated it as a finished answer and the frame showed an empty
    turn. It must end the run the way an error does — dangling calls paired,
    no further model calls — while carrying the provider's refusal message
    on the ``agent_end`` so a UI can show it.
    """

    @pytest.mark.asyncio
    async def test_refusal_ends_the_run_with_the_providers_message(self) -> None:
        stream = ScriptedStream(
            [
                [
                    StreamEndEvent(
                        stop_reason="refusal",
                        error="model refused: I can't help with that. [finish_reason=stop]",
                    )
                ],
                # A second scripted turn that must NEVER run: a refusal is
                # terminal, and reaching this script means the loop fed the
                # refusal back for another call.
                [StreamTextDelta(delta="unreachable"), StreamEndEvent(stop_reason="stop")],
            ]
        )
        context = LoopContext()
        loop = AgentLoop()

        events = []
        async for event in loop.run([Message.user("go")], context, make_config(stream), None):
            events.append(event)

        assert len(stream.requests) == 1
        end = events[-1]
        assert isinstance(end, AgentEndEvent)
        assert end.aborted is False
        assert end.error is not None and "I can't help with that." in end.error

    @pytest.mark.asyncio
    async def test_refusal_pairs_dangling_calls_without_executing(self) -> None:
        """A stream can be cut by a filter mid-tool-call; the composed call must
        be paired (the wire stays legal) and must not execute."""
        executed: list[str] = []
        stream = ScriptedStream(
            [
                [
                    tool_call_delta(0, id="c1", name="echo", args='{"text":"a"}'),
                    StreamEndEvent(stop_reason="refusal", error="model refused [marker]"),
                ]
            ]
        )
        context = LoopContext(tools=[echo_tool(executed)])
        loop = AgentLoop()

        events = []
        async for event in loop.run([Message.user("go")], context, make_config(stream), None):
            events.append(event)

        assert executed == []
        tool_messages = [m for m in context.messages if isinstance(m, Message) and m.role == "tool"]
        assert [m.tool_call_id for m in tool_messages] == ["c1"]
        assert all(m.is_error for m in tool_messages)

    @pytest.mark.asyncio
    async def test_refusal_message_survives_onto_the_assistant_message(self) -> None:
        """The agent_end dies with the run; a resumed session replays the
        transcript's messages, so the refusal text must be stored on the
        assistant message it explains (under ``provider_payload``, the
        established home for harness bookkeeping that wire clients never
        replay)."""
        stream = ScriptedStream(
            [[StreamEndEvent(stop_reason="refusal", error="model refused: no. [m]")]]
        )
        context = LoopContext()
        loop = AgentLoop()
        async for _ in loop.run([Message.user("go")], context, make_config(stream), None):
            pass

        assistant = next(
            m for m in context.messages if isinstance(m, Message) and m.role == "assistant"
        )
        assert assistant.stop_reason == "refusal"
        assert (assistant.provider_payload or {}).get("refusal") == "model refused: no. [m]"

    @pytest.mark.asyncio
    async def test_a_bare_refusal_end_still_says_something(self) -> None:
        """A wire client that emits ``refusal`` with no message must not
        reintroduce the silent failure this stop_reason exists to fix."""
        stream = ScriptedStream([[StreamEndEvent(stop_reason="refusal")]])
        context = LoopContext()
        loop = AgentLoop()

        events = []
        async for event in loop.run([Message.user("go")], context, make_config(stream), None):
            events.append(event)

        end = events[-1]
        assert isinstance(end, AgentEndEvent)
        assert end.error is not None and "refused" in end.error


# ---------------------------------------------------------------------------
# Empty truncation: the silent-turn failure (session f3c058d1)
# ---------------------------------------------------------------------------


def _laddered_model() -> ModelSpec:
    return ModelSpec(
        provider="test",
        model_id="m",
        reasoning_efforts=("low", "medium", "high"),
        reasoning_effort="high",
    )


@pytest.mark.asyncio
async def test_empty_length_truncation_retries_at_lower_effort():
    """A length stop with NO text and NO calls is a silent turn: the loop
    retries one rung lower instead of ending with nothing on screen."""
    stream = ScriptedStream(
        [
            [StreamEndEvent(stop_reason="length")],  # spent the budget thinking
            [StreamTextDelta(delta="here is the answer"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext()
    loop = AgentLoop()
    events = []
    async for event in loop.run(
        [Message.user("go")], context, make_config(stream, model=_laddered_model()), None
    ):
        events.append(event)

    assert len(stream.requests) == 2
    assert stream.requests[0].model.reasoning_effort == "high"
    assert stream.requests[1].model.reasoning_effort == "medium"
    notices = [e for e in events if isinstance(e, NoticeEvent)]
    assert any("retrying at effort medium" in n.text for n in notices)
    end = events[-1]
    assert isinstance(end, AgentEndEvent) and not end.aborted
    # The empty assistant message must not survive into the context: it is an
    # illegal wire block and teaches the retry to say nothing.
    assert all(
        not (m.role == "assistant" and not m.text and not m.tool_calls)
        for m in context.messages
        if isinstance(m, Message)
    )


@pytest.mark.asyncio
async def test_empty_length_truncation_ends_with_a_notice_when_no_lower_rung():
    """Retries are bounded; when they are spent the turn ends, but the user
    sees WHY instead of minutes of thinking followed by silence."""
    stream = ScriptedStream(
        [
            [StreamEndEvent(stop_reason="length")],
            [StreamEndEvent(stop_reason="length")],
            [StreamEndEvent(stop_reason="length")],
        ]
    )
    context = LoopContext()
    loop = AgentLoop()
    events = []
    async for event in loop.run(
        [Message.user("go")], context, make_config(stream, model=_laddered_model()), None
    ):
        events.append(event)

    # initial + two lower-rung retries; the third empty truncation ends it.
    assert len(stream.requests) == 3
    efforts = [r.model.reasoning_effort for r in stream.requests]
    assert efforts == ["high", "medium", "low"]
    notices = [e for e in events if isinstance(e, NoticeEvent)]
    assert any("no visible output" in n.text for n in notices)
    end = events[-1]
    assert isinstance(end, AgentEndEvent)


@pytest.mark.asyncio
async def test_text_only_length_truncation_is_not_retried():
    """A truncation that DID produce visible text is an ordinary truncation:
    the turn ends, no effort step, no retry."""
    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="partial"), StreamEndEvent(stop_reason="length")],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext()
    loop = AgentLoop()
    events = []
    async for event in loop.run(
        [Message.user("go")], context, make_config(stream, model=_laddered_model()), None
    ):
        events.append(event)

    assert len(stream.requests) == 1
    assert not [e for e in events if isinstance(e, NoticeEvent)]


@pytest.mark.asyncio
async def test_empty_length_retry_clamps_the_resolved_model_too():
    """Review F1: the production session supplies ``get_model`` (its resolver
    ignores the loop's ``config.model`` mutation), so the retreat must clamp
    the RESOLVED spec or the retry goes back out at the same silent rung."""
    host_model = _laddered_model()  # always returns "high"
    stream = ScriptedStream(
        [
            [StreamEndEvent(stop_reason="length")],
            [StreamTextDelta(delta="answered"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext()
    loop = AgentLoop()
    config = make_config(stream, model=host_model, get_model=lambda: host_model)
    events = [e async for e in loop.run([Message.user("go")], context, config, None)]

    assert len(stream.requests) == 2
    assert stream.requests[0].model.reasoning_effort == "high"
    assert stream.requests[1].model.reasoning_effort == "medium"
    assert isinstance(events[-1], AgentEndEvent)


@pytest.mark.asyncio
async def test_a_hard_cancel_keeps_the_text_that_already_streamed() -> None:
    """A cancelled turn must still carry the partial answer it received.

    The assistant message is assembled once when the turn ends, because
    rebuilding it on every delta was quadratic in response length. A hard
    cancel never reaches that assembly, so it assembles in the
    ``CancelledError`` handler instead — without which a consumer still holding
    the message from ``MessageStartEvent`` would find it empty where it
    previously held every delta up to the cut.

    Asserted through the real ``_model_turn`` rather than by calling the
    handler, because the point is that the cancellation path reaches it.
    """
    started: dict[str, Message] = {}

    async def parked_stream(request: ChatRequest, signal: AbortSignal | None):
        yield StreamTextDelta(delta="partial ")
        yield StreamTextDelta(delta="answer")
        await asyncio.sleep(60)  # park mid-stream; the cancel lands here
        yield StreamEndEvent(stop_reason="stop")

    loop = AgentLoop()
    context = LoopContext(tools=[])
    config = make_config(parked_stream)

    async def drive() -> None:
        async for event in loop._model_turn(context, config, None):
            # Narrow to Message: the field is the AgentMessage union, and only
            # a real assistant Message carries the text this asserts on.
            if isinstance(event, MessageStartEvent) and isinstance(event.message, Message):
                started["message"] = event.message

    task = asyncio.create_task(drive())
    # Let both deltas land before cutting, so there is text to lose.
    for _ in range(100):
        await asyncio.sleep(0.01)
        if started.get("message") is not None:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (
        started["message"].text == "partial answer"
    ), "a hard cancel dropped the text that had already streamed"


# ---------------------------------------------------------------------------
# A pending fork rides the interrupt poll, but is NOT a steer
# ---------------------------------------------------------------------------
#
# The asymmetry pinned here is the subtlest correctness point in `/fork`, and
# both halves of it fail silently:
#
# - Too weak, and a fork requested during a ten-minute `wait` waits out the
#   whole tool before the branch is taken.
# - Too strong, and the fork SKIPS the parent's remaining tool calls — the
#   parent's own turn is damaged by a command that was supposed to leave it
#   completely alone.


@pytest.mark.asyncio
async def test_a_pending_fork_interrupts_an_interruptible_tool():
    """A fork reaches its boundary promptly instead of waiting out the tool.

    Same mechanism steering uses (the tool is re-runnable by construction,
    which is what ``interruptible=True`` means), reached through the fork
    predicate alone — no steering message is queued anywhere in this test.
    """
    tool_started = asyncio.Event()
    outcome: dict[str, str] = {}

    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="block", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(
        tools=[_blocking_tool("block", tool_started, interruptible=True, outcome=outcome)]
    )
    config = make_config(
        stream,
        interrupt_mode="immediate",
        has_pending_fork=lambda: True,
    )

    messages = await AgentLoop().run_to_end([Message.user("go")], context, config, None)

    assert outcome == {"block": "cancelled"}
    results = [m for m in messages if isinstance(m, Message) and m.role == "tool"]
    assert results[0].is_error


@pytest.mark.asyncio
async def test_a_pending_fork_does_not_skip_the_rest_of_the_batch():
    """T7. THE test. A fork must not damage the parent's turn.

    The control is ``test_steering_interrupts_between_exclusive_calls``, which
    is this exact shape with a steering message instead of a fork and asserts
    the opposite: there, ``b`` never runs and gets a SKIPPED result. Here every
    planned call must still execute and produce a real result, because a fork
    has not redirected the work — it is a copy of the conversation, taken to the
    side, and the parent was told to carry on.
    """
    executed: list[str] = []
    tool_a = echo_tool(executed, name="a", concurrency="exclusive")
    tool_b = echo_tool(executed, name="b", concurrency="exclusive")

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
        # Pending for the WHOLE run, which is the hostile case: every batch-slot
        # check sees it and none of them may skip anything.
        has_pending_fork=lambda: True,
    )

    messages = await AgentLoop().run_to_end([Message.user("go")], context, config, None)

    assert executed == ["a", "b"], "a pending fork skipped the parent's remaining tool calls"
    results = [m for m in messages if isinstance(m, Message) and m.role == "tool"]
    assert len(results) == 2
    assert not any(result.is_error for result in results)
    assert not any("skipped" in result.text.lower() for result in results)


@pytest.mark.asyncio
async def test_a_pending_fork_injects_nothing_into_the_parents_context():
    """T8. A fork is not a message.

    A steer becomes a user turn the parent's model is given; a fork must leave
    the conversation byte-identical to a run that never forked at all. Asserted
    against a CONTROL run rather than against a hand-written expectation, so the
    test cannot drift from what the loop actually produces.
    """

    def build() -> tuple[ScriptedStream, LoopContext, list[str]]:
        executed: list[str] = []
        stream = ScriptedStream(
            [
                [
                    tool_call_delta(0, id="c1", name="echo", args="{}"),
                    StreamEndEvent(stop_reason="toolUse"),
                ],
                [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
            ]
        )
        return stream, LoopContext(tools=[echo_tool(executed)]), executed

    control_stream, control_context, _ = build()
    await AgentLoop().run_to_end(
        [Message.user("go")],
        control_context,
        make_config(control_stream, interrupt_mode="immediate"),
        None,
    )

    fork_stream, fork_context, _ = build()
    await AgentLoop().run_to_end(
        [Message.user("go")],
        fork_context,
        make_config(fork_stream, interrupt_mode="immediate", has_pending_fork=lambda: True),
        None,
    )

    def texts(context: LoopContext) -> list[tuple[str, str]]:
        return [(m.role, m.text) for m in context.messages if isinstance(m, Message)]

    assert texts(fork_context) == texts(control_context)


@pytest.mark.asyncio
async def test_a_raising_fork_predicate_never_breaks_the_turn():
    """A host callback that raises costs an interrupt, never the run.

    Same posture the steering peek has: the fork is a convenience on top of a
    turn the user is waiting for, and a bug in the host's predicate must not be
    able to take that turn down.
    """
    executed: list[str] = []

    def boom() -> bool:
        raise RuntimeError("host bug")

    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="c1", name="echo", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[echo_tool(executed)])
    config = make_config(stream, interrupt_mode="immediate", has_pending_fork=boom)

    messages = await AgentLoop().run_to_end([Message.user("go")], context, config, None)

    assert executed == ["echo"]
    assert any(isinstance(m, Message) and m.text == "done" for m in messages)


# ---------------------------------------------------------------------------
# Mid-stream connectivity loss: the laptop closed at home, opened at work
# ---------------------------------------------------------------------------


class FailingStream:
    """Fake stream_fn whose Nth call dies PART WAY THROUGH the answer.

    The failure is raised after the deltas have already been yielded, which is
    the only shape that matters here: with nothing forwarded the provider layer
    retries in place and the loop never sees a failure at all.
    """

    def __init__(
        self, failures: list[BaseException | None], *, partial: str = "The answer is "
    ) -> None:
        self.failures = failures
        self.partial = partial
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        index = len(self.requests)
        self.requests.append(request)
        failure = self.failures[index] if index < len(self.failures) else None

        async def gen():
            if failure is not None:
                yield StreamTextDelta(delta=self.partial)
                raise failure
            yield StreamTextDelta(delta="42.")
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    def history(self, index: int) -> list[str]:
        """``role:text`` of the messages the Nth request carried."""
        return [f"{m.role}:{m.text}" for m in self.requests[index].messages]


#: The exceptions a REAL severed socket raises mid-answer, captured from a live
#: provider whose TCP connection was killed with SO_LINGER{1,0} half way through
#: an SSE body. This tuple is the fixture population these tests must exercise,
#: and writing it as exception OBJECTS rather than as message strings is the
#: point: an earlier round of these tests built every fixture from a hand-typed
#: ``ConnectError`` string and then asserted its own premise
#: (``assert error.connectivity_loss``), so the suite stayed green while the
#: shipped binary did not continue a single real mid-stream cut. A fixture that
#: cannot fail cannot catch a classifier that never matches.
#:
#: An EMPTY message is deliberately first: ``httpx.ReadError('')`` is the single
#: most common shape of a mid-body RST, and it is precisely the one a
#: text-matching classifier cannot see.
_REAL_MID_STREAM_EXCEPTIONS: tuple[Exception, ...] = (
    httpx.ReadError(""),
    httpx.ReadError("[Errno 54] Connection reset by peer"),
    httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
    # What a SILENT drop (lid closed, no RST) produces: the stall watchdog in
    # clients.py constructs exactly this after STREAM_READ_TIMEOUT_S.
    httpx.ReadTimeout("stream stalled: no data for 180s"),
    httpx.WriteError("[Errno 32] Broken pipe"),
)


def _offline(exc: Exception | None = None) -> ProviderError:
    """A mid-stream cut as the DRIVER hands it to the loop.

    Built by pushing a real ``httpx`` exception through the same two functions
    the live path uses — ``wrap_transport_error`` then the driver's
    ``forwarded_any`` marking — instead of asserting a hand-written string is
    classified the way the test wants. Nothing here asserts the premise: if the
    classifier stops recognising a severed socket, the fixture stops carrying
    the flag and the behavioural tests below fail, which is how it should have
    been able to catch Q1.

    Defaults to the empty-message ``ReadError`` observed against a real killed
    connection, so the DEFAULT fixture is the hardest case rather than the
    easiest.
    """
    error = wrap_transport_error(exc if exc is not None else httpx.ReadError(""))
    _mark_mid_stream_connectivity(error)
    return error


def _offline_preconnect() -> ProviderError:
    """A PRE-connection connectivity loss (DNS/route), for the arm that owns it.

    Kept distinct from :func:`_offline` because the two are classified by
    different predicates for different reasons — see
    ``is_mid_stream_connectivity_loss``.
    """
    return wrap_transport_error(
        httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")
    )


@pytest.mark.parametrize("exc", _REAL_MID_STREAM_EXCEPTIONS, ids=lambda e: type(e).__name__)
def test_every_real_mid_stream_exception_is_continuable(exc: Exception) -> None:
    """THE Q1 REGRESSION GUARD, at the classifier boundary.

    Each of these is an exception a real severed socket raises, and every one of
    them classified ``False`` before this fix — so the loop's continuation
    branch, which is reachable ONLY after deltas were forwarded, had a live
    population of exactly the failures it could not recognise.

    Asserted on objects httpx itself constructs, so this cannot be satisfied by
    a marker string that merely looks like one.
    """
    wrapped = wrap_transport_error(exc)
    # Before the driver marks it, a mid-stream shape is an ORDINARY transient:
    # seen pre-connect it must keep its fast retry and its fallback walk.
    assert not wrapped.connectivity_loss
    _mark_mid_stream_connectivity(wrapped)
    assert wrapped.connectivity_loss, f"{type(exc).__name__} must be continuable mid-stream"


def test_a_provider_that_answered_cannot_claim_the_machine_is_offline() -> None:
    """R6: an in-band error chunk has no status, so its TEXT alone used to be
    enough to route a provider's own upstream trouble down the continue path.

    Provenance is what separates them: only ``wrap_transport_error`` — our own
    client observing a socket die — sets ``transport``.
    """
    spoof = ProviderError(
        None,
        "upstream_error: upstream: network is unreachable at edge",
        retryable=True,
    )
    assert not spoof.transport
    _mark_mid_stream_connectivity(spoof)
    assert not spoof.connectivity_loss
    # And a provider that answered with a STATUS is never continuable either.
    for status in (500, 503, 429, 401, 400):
        answered = ProviderError(status, "boom", retryable=True)
        _mark_mid_stream_connectivity(answered)
        assert not answered.connectivity_loss


@pytest.mark.asyncio
async def test_connectivity_loss_after_deltas_continues_the_turn() -> None:
    """THE REPORTED BUG: the network changed mid-answer and killed the session.

    The provider layer cannot retry this — the deltas are already on the user's
    screen — so before this fix the run ended with `stop_reason="error"`. Now
    the partial answer is committed as history and the turn continues, so the
    user sees ONE uninterrupted answer.
    """
    stream = FailingStream([_offline(), None])
    config = make_config(stream)

    events = []
    async for event in AgentLoop().run([Message.user("go")], LoopContext(), config, None):
        events.append(event)

    # The run did NOT end in an error: that is the whole fix.
    ends = [e for e in events if isinstance(e, AgentEndEvent)]
    assert len(ends) == 1
    assert ends[0].error is None
    assert ends[0].aborted is False

    # The user sees the partial answer and its continuation, each exactly once.
    on_screen = "".join(e.delta for e in events if e.type == "message_update")
    assert on_screen == "The answer is 42."

    # A visible notice explains the seam rather than the text silently jumping.
    notices = [e.text for e in events if isinstance(e, NoticeEvent)]
    assert any("network connection lost" in text for text in notices)

    # THE NO-DUPLICATION INVARIANT, asserted structurally: the retry carries the
    # partial answer as HISTORY, so the model writes the remainder instead of
    # re-streaming what was already read.
    history = stream.history(1)
    assert history[:2] == ["user:go", "assistant:The answer is "]

    # And it asks for a CONTINUATION rather than leaving the partial answer as a
    # trailing assistant turn. That shape is load-bearing, not cosmetic: a
    # trailing assistant message is a "prefill", which current Claude models
    # reject with HTTP 400 ("Prefilling assistant messages is not supported for
    # this model") — so the default model of this harness would turn a
    # recoverable blip into a hard failure. Ending on a user turn also keeps the
    # role alternation Anthropic documents.
    assert len(history) == 3
    assert history[2].startswith("user:")
    assert "cut off" in history[2]

    # The partial text keeps its trailing space, so the transcript is exactly
    # what the user read. Legal only because the assistant turn is not final.
    assert stream.requests[1].messages[-2].text == "The answer is "
    assert stream.requests[1].messages[-1].role == "user"


@pytest.mark.asyncio
async def test_connectivity_loss_does_not_duplicate_the_partial_text() -> None:
    """The continuation must never re-render text already in the transcript.

    Asserted on the FINAL assembled messages as well as on the stream, because
    a duplicate that only shows up in the persisted transcript is still a
    duplicate the user reads on reload.
    """
    stream = FailingStream([_offline(), None])
    config = make_config(stream)

    messages = await AgentLoop().run_to_end([Message.user("go")], LoopContext(), config, None)

    assistant_text = "".join(
        m.text for m in messages if isinstance(m, Message) and m.role == "assistant"
    )
    assert assistant_text == "The answer is 42."
    assert assistant_text.count("The answer is ") == 1


@pytest.mark.asyncio
async def test_mid_stream_5xx_after_deltas_still_ends_the_run() -> None:
    """A PROVIDER failure mid-answer keeps the old terminal behaviour.

    The distinction the fix rests on: an offline machine means nothing was wrong
    with the request, so re-asking is the entire fix. A 500 means the provider
    DID answer — replaying that turn would re-bill it and paper over a failure
    the user needs to see.
    """
    boom = ProviderError(500, "internal server error", retryable=True)
    assert not boom.connectivity_loss
    stream = FailingStream([boom, None])
    config = make_config(stream)

    events = []
    async for event in AgentLoop().run([Message.user("go")], LoopContext(), config, None):
        events.append(event)

    ends = [e for e in events if isinstance(e, AgentEndEvent)]
    assert len(ends) == 1
    assert ends[0].error is not None
    assert "internal server error" in ends[0].error
    # It never retried: one request only.
    assert len(stream.requests) == 1


@pytest.mark.asyncio
async def test_connectivity_continuation_budget_surfaces_a_bounded_error() -> None:
    """A genuinely dead network must END the run, not retry forever.

    Every attempt fails offline, so the run exhausts
    MAX_CONNECTIVITY_CONTINUATIONS and then surfaces the provider's own
    diagnostic error — bounded, named, and not a hang.
    """
    failures: list[BaseException | None] = [_offline() for _ in range(20)]
    stream = FailingStream(failures)
    config = make_config(stream)

    events = []
    async for event in AgentLoop().run([Message.user("go")], LoopContext(), config, None):
        events.append(event)

    ends = [e for e in events if isinstance(e, AgentEndEvent)]
    assert len(ends) == 1
    assert ends[0].error is not None
    # The provider layer's own diagnostic survives to the frame: the run ends
    # NAMED (here the severed-socket class), never as a bare hang or an empty
    # string — which is what `wrap_transport_error` keeps the class name for.
    assert "ReadError" in ends[0].error
    # Bounded: the initial attempt plus exactly the continuation budget.
    assert len(stream.requests) == MAX_CONNECTIVITY_CONTINUATIONS + 1
    # And the budget is SPENT VISIBLY: one notice per continuation, each naming
    # its position, rather than three identical claims of a reconnection.
    notices = [e.text for e in events if isinstance(e, NoticeEvent)]
    assert notices == [
        f"network connection lost mid-response — retrying ({n}/{MAX_CONNECTIVITY_CONTINUATIONS})"
        for n in range(1, MAX_CONNECTIVITY_CONTINUATIONS + 1)
    ]


@pytest.mark.asyncio
async def test_connectivity_continuation_drops_truncated_tool_calls() -> None:
    """A tool call still streaming when the socket died is DROPPED, not run.

    Its arguments are truncated JSON: executing it would run a call the model
    never finished asking for. Dropping it also keeps the wire legal — no
    tool_use block means no unmatched tool_result.
    """
    executed: list[str] = []

    class _PartialCallStream:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            index = len(self.requests)
            self.requests.append(request)

            async def gen():
                if index == 0:
                    yield StreamTextDelta(delta="reading it ")
                    # Arguments cut mid-JSON by the network going away.
                    yield tool_call_delta(0, id="c1", name="echo", args='{"text": "hal')
                    raise _offline()
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")

            return gen()

    stream = _PartialCallStream()
    config = make_config(stream)

    messages = await AgentLoop().run_to_end(
        [Message.user("go")], LoopContext(tools=[echo_tool(executed)]), config, None
    )

    # The half-dictated call never ran.
    assert executed == []
    # And no assistant message carries it, so the wire stays legal.
    assert all(
        not m.tool_calls for m in messages if isinstance(m, Message) and m.role == "assistant"
    )
    assert any(isinstance(m, Message) and m.text == "done" for m in messages)


@pytest.mark.asyncio
async def test_connectivity_continuation_keeps_a_COMPLETE_tool_call() -> None:
    """A call whose arguments FINISHED arriving before the cut is not truncated.

    The sibling test above drops a half-dictated call, which is right. Dropping
    a complete one is not: the tool never runs, the model must re-derive it, and
    the continuation prompt then describes a turn whose own record of having
    asked for a tool has been deleted from the history it is being asked to
    continue.

    The surviving call is PAIRED rather than executed — the network died before
    the loop could run it, and an unmatched ``tool_use`` block is a 400 on the
    Anthropic wire — so the model sees the call it made did not run and may
    re-issue it.
    """
    executed: list[str] = []

    class _CompleteCallStream:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            index = len(self.requests)
            self.requests.append(request)

            async def gen():
                if index == 0:
                    yield StreamTextDelta(delta="Let me check that. ")
                    # Arguments complete: valid JSON, fully arrived...
                    yield tool_call_delta(0, id="c1", name="echo", args='{"text": "hello"}')
                    # ...and only THEN the socket dies.
                    raise _offline()
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")

            return gen()

    stream = _CompleteCallStream()
    config = make_config(stream)
    messages = await AgentLoop().run_to_end(
        [Message.user("go")], LoopContext(tools=[echo_tool(executed)]), config, None
    )

    # It is not EXECUTED — the turn was abandoned, not completed.
    assert executed == []
    # But it survives in history, so the model's own record is intact.
    retry_history = stream.requests[1].messages
    carried = [m for m in retry_history if isinstance(m, Message) and m.tool_calls]
    assert len(carried) == 1
    assert carried[0].tool_calls[0].name == "echo"
    assert carried[0].tool_calls[0].arguments == {"text": "hello"}
    # And it is PAIRED, so the wire stays legal.
    paired = [
        m
        for m in retry_history
        if isinstance(m, Message)
        and m.role == "tool"
        and m.tool_call_id == carried[0].tool_calls[0].id
    ]
    assert len(paired) == 1
    assert any(isinstance(m, Message) and m.text == "done" for m in messages)


@pytest.mark.asyncio
async def test_connectivity_continuation_closes_the_turn_it_abandons() -> None:
    """Every TurnStart gets a TurnEnd, including on the continuation path.

    ``TurnEndEvent`` is what drives a front end's per-turn reconciliation — in
    the TUI, ``_retire_live_tool_cards``, the only routine path that settles a
    row for a call that never ran. Without it the abandoned turn's composing
    spinner animates forever and the working line goes on announcing "composing
    a call" while the continued turn streams prose.
    """

    class _ComposingThenCut:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            index = len(self.requests)
            self.requests.append(request)

            async def gen():
                if index == 0:
                    yield StreamTextDelta(delta="one moment ")
                    yield tool_call_delta(0, id="c1", name="echo", args='{"text": "hal')
                    raise _offline()
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")

            return gen()

    events = []
    async for event in AgentLoop().run(
        [Message.user("go")],
        LoopContext(tools=[echo_tool([])]),
        make_config(_ComposingThenCut()),
        None,
    ):
        events.append(event)

    starts = [e for e in events if isinstance(e, TurnStartEvent)]
    ends = [e for e in events if isinstance(e, TurnEndEvent)]
    assert len(starts) == 2
    assert len(ends) == len(starts), "an abandoned turn must still be closed"
    # And the close lands BEFORE the continuation's TurnStart, so a front end
    # reconciles the dead turn's rows before the next one opens its own.
    order = [type(e).__name__ for e in events if isinstance(e, (TurnStartEvent, TurnEndEvent))]
    assert order == ["TurnStartEvent", "TurnEndEvent", "TurnStartEvent", "TurnEndEvent"]


@pytest.mark.asyncio
async def test_connectivity_continuation_prompt_is_persisted() -> None:
    """The prompt reaches the TRANSCRIPT, not just the live request.

    ``new_messages`` is what ``AgentEndEvent`` hands the host to persist. With
    the prompt appended only to the live context, a resumed session read the
    partial answer glued straight to its continuation with no record that a
    network interruption sat between them — and a run that continued more than
    once persisted a run of consecutive assistant messages that no longer
    explains itself, which compaction then summarises.
    """
    stream = FailingStream([_offline(), None])
    messages = await AgentLoop().run_to_end(
        [Message.user("go")], LoopContext(), make_config(stream), None
    )

    roles = [m.role for m in messages if isinstance(m, Message)]
    assert roles == ["assistant", "user", "assistant"], "no consecutive assistant runs"
    prompts = [
        m
        for m in messages
        if isinstance(m, Message) and m.text == loop_module.CONNECTIVITY_CONTINUATION_PROMPT
    ]
    assert len(prompts) == 1


@pytest.mark.asyncio
async def test_a_whitespace_only_partial_is_dropped_not_referenced() -> None:
    """R4: the guard must agree with the SERIALIZER about what counts as text.

    Both wire builders drop an assistant turn whose text is whitespace-only
    (``_is_empty_assistant``: "Whitespace-only text counts as empty"). A model
    that emits a newline before the cut therefore had its turn vanish from the
    request while the prompt went on telling it to continue "the partial text
    above" — text no wire carried. Aligning the guard routes this to the
    drop-and-re-ask arm instead.
    """
    stream = FailingStream([_offline(), None], partial="\n")
    await AgentLoop().run_to_end([Message.user("go")], LoopContext(), make_config(stream), None)

    retry = stream.requests[1].messages
    assert not any(isinstance(m, Message) and m.role == "assistant" for m in retry)
    # No dangling reference to text the model cannot see: the original question
    # is simply re-asked.
    assert not any(
        isinstance(m, Message) and m.text == loop_module.CONNECTIVITY_CONTINUATION_PROMPT
        for m in retry
    )
    assert [m.role for m in retry if isinstance(m, Message)] == ["user"]


@pytest.mark.asyncio
async def test_connectivity_loss_before_any_delta_is_unchanged() -> None:
    """The pre-first-token case belongs to the PROVIDER layer and must stay
    there: nothing was forwarded, so the driver retries in place and the loop
    never sees a failed turn at all. Asserted here so the harness fix cannot
    quietly start intercepting a case it does not own."""

    class _EmptyThenOk:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.requests.append(request)

            async def gen():
                # No delta before the end: the provider layer's own retry would
                # have handled a failure here, so the loop sees a clean turn.
                yield StreamTextDelta(delta="42.")
                yield StreamEndEvent(stop_reason="stop")

            return gen()

    stream = _EmptyThenOk()
    config = make_config(stream)

    events = []
    async for event in AgentLoop().run([Message.user("go")], LoopContext(), config, None):
        events.append(event)

    assert len(stream.requests) == 1
    assert [e.text for e in events if isinstance(e, NoticeEvent)] == []


@pytest.mark.asyncio
async def test_connectivity_continuation_request_is_wire_legal_for_anthropic() -> None:
    """The continuation must SERIALIZE legally, not merely look right in the loop.

    Two Anthropic rules make the obvious implementation — leave the partial
    answer as the last message and re-send — a hard 400, which would convert a
    recoverable network blip into a dead run on this harness's own default
    model:

    * a trailing assistant message is a PREFILL, and current Claude models
      answer "Prefilling assistant messages is not supported for this model";
    * a final assistant message may not end in whitespace, and an interrupted
      delta ("The answer is ") is precisely how one is produced.

    Asserted through the REAL client body builder rather than by re-reading the
    loop's own list, because the serializer is where the rule actually bites.
    """
    from local_operator.providers.clients import AnthropicClient

    stream = FailingStream([_offline(), None])
    config = make_config(stream)
    await AgentLoop().run_to_end([Message.user("go")], LoopContext(), config, None)

    retry_request = stream.requests[1]
    body = AnthropicClient("https://api.anthropic.com")._build_body(
        ChatRequest(
            model=ModelSpec(provider="anthropic", model_id="claude-opus-5"),
            messages=retry_request.messages,
        )
    )
    roles = [entry["role"] for entry in body["messages"]]

    # Not a prefill: the request ends on a user turn.
    assert roles[-1] == "user"
    # Roles alternate, which is what Anthropic documents for its turn model.
    assert roles == ["user", "assistant", "user"]
    # The one assistant turn is not final, so its trailing space is legal and
    # the transcript can stay byte-identical to what was displayed.
    assistant_text = body["messages"][1]["content"][0]["text"]
    assert assistant_text == "The answer is "
