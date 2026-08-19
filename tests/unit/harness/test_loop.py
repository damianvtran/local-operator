"""AgentLoop tests: full turn shape, tool pairing on abort/length, steering
interrupts, validation errors back to the model, gates, follow-ups."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, Literal

import pytest

from local_operator.harness.loop import (
    ABORT_DRAIN_TIMEOUT_S,
    AgentLoop,
    LoopContext,
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
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolCallComposeEvent,
    ToolContext,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
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
