"""Tool-call fault classification, measured through the REAL AgentLoop.

The rate arithmetic is trivial; the CLASSIFICATION is the claim, so every case
here drives an actual turn and asserts what the loop reported through
``LoopConfig.record_tool_call`` rather than unit-testing a helper.

The split that matters is whether a fault is the MODEL's: only those three
classes feed the benchmarking figure, and a misclassification would either
inflate the number (crediting a denied call) or defame the model (blaming it
for an HTTP 500). Each class is therefore pinned at its own source.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamToolCallDelta,
    TextContent,
    ToolContext,
    ToolResult,
)

MODEL = ModelSpec(provider="test", model_id="m")


class _Scripted:
    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self.turns = turns
        self.n = 0

    def __call__(self, request, signal: AbortSignal | None):
        turn = self.turns[self.n]
        self.n += 1

        async def gen():
            for event in turn:
                yield event

        return gen()


def _ok_tool(
    name: str = "read", concurrency: Literal["shared", "exclusive"] = "shared"
) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id, tool_name=name, content=[TextContent(text="ok")]
        )

    return AgentTool(
        name=name,
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        concurrency=concurrency,
        execute=execute,
    )


def _raising_tool(name: str = "web_fetch") -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        raise RuntimeError("HTTP 500 from upstream")

    return AgentTool(
        name=name,
        parameters={"type": "object", "properties": {"url": {"type": "string"}}},
        execute=execute,
    )


def _calls(*specs: tuple[int, str, str, str]) -> list[StreamEvent]:
    out: list[StreamEvent] = []
    for index, cid, name, args in specs:
        out.append(StreamToolCallDelta(index=index, id=cid, name=name, argument_delta=""))
        out.append(StreamToolCallDelta(index=index, id=cid, name=None, argument_delta=args))
    return out


async def _run(
    turn: list[StreamEvent], tools: list[AgentTool], **context_kwargs: Any
) -> list[tuple[str, str, str, float]]:
    recorded: list[tuple[str, str, str, float]] = []
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=_Scripted([turn, [StreamEndEvent(stop_reason="stop")]]),
        record_tool_call=lambda name, origin, fault, ms: recorded.append((name, origin, fault, ms)),
    )
    context = LoopContext(
        system_blocks=["sys"],
        tools=tools,
        tool_context=ToolContext(session_id="s1", **context_kwargs),
    )
    async for _ in AgentLoop().run([], context, config):
        pass
    return recorded


def _faults(recorded) -> dict[str, str]:
    return {name: fault for name, _origin, fault, _ms in recorded}


@pytest.mark.asyncio
async def test_a_clean_call_records_no_fault():
    recorded = await _run(
        _calls((0, "c1", "read", '{"path":"a"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [_ok_tool()],
    )
    assert recorded == [("read", "model", "", recorded[0][3])]
    assert recorded[0][3] >= 0, "a dispatched call reports its duration"


@pytest.mark.asyncio
async def test_a_hallucinated_tool_name_is_a_model_fault():
    """The model named a tool that does not exist. Its NAME is retained."""
    recorded = await _run(
        _calls((0, "c1", "reed_file", '{"path":"a"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [_ok_tool()],
    )
    assert _faults(recorded) == {"reed_file": "unknown_tool"}


@pytest.mark.asyncio
async def test_a_schema_violation_is_a_model_fault():
    """``path`` is declared a string; the model sent an int."""
    recorded = await _run(
        _calls((0, "c1", "read", '{"path":123}')) + [StreamEndEvent(stop_reason="toolUse")],
        [_ok_tool()],
    )
    assert _faults(recorded) == {"read": "invalid_arguments"}


@pytest.mark.asyncio
async def test_a_duplicate_call_id_is_a_model_fault():
    """One id emitted twice: the first wins, the twin is a malformed emission.

    Classified by its ``details`` MARKER and not by its text \u2014 a duplicate is
    parked with ``tool is None`` exactly like an unknown tool, so the two are
    indistinguishable structurally at ``park``.
    """
    recorded = await _run(
        _calls((0, "c1", "read", '{"path":"a"}'), (1, "c1", "read", '{"path":"b"}'))
        + [StreamEndEvent(stop_reason="toolUse")],
        [_ok_tool()],
    )
    faults = sorted(fault for _n, _o, fault, _ms in recorded)
    assert faults == ["", "duplicate_id"]


@pytest.mark.asyncio
async def test_a_tool_that_raises_is_an_execution_error_not_a_model_fault():
    """HTTP 500 is the world failing, not the model emitting a bad call."""
    recorded = await _run(
        _calls((0, "c1", "web_fetch", '{"url":"x"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [_raising_tool()],
    )
    assert _faults(recorded) == {"web_fetch": "execution"}


@pytest.mark.asyncio
async def test_a_denied_call_is_recorded_but_is_not_the_model_s_fault():
    """The user's decision. Recorded so counts reconcile, excluded from both rates."""

    async def deny(tool_name, summary, job_id):
        return False

    tool = _ok_tool()
    tool.approval_tier = "write"
    recorded = await _run(
        _calls((0, "c1", "read", '{"path":"a"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [tool],
        request_approval=deny,
    )
    assert _faults(recorded) == {"read": "denied"}


@pytest.mark.asyncio
async def test_an_approval_gate_crash_is_our_fault_not_the_model_s():
    async def explode(tool_name, summary, job_id):
        raise RuntimeError("gate is broken")

    tool = _ok_tool()
    tool.approval_tier = "write"
    recorded = await _run(
        _calls((0, "c1", "read", '{"path":"a"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [tool],
        request_approval=explode,
    )
    assert _faults(recorded) == {"read": "gate_failed"}


@pytest.mark.asyncio
async def test_the_nested_eval_bridge_records_with_its_own_origin():
    """``dispatch_tool`` never reaches ``park`` and would otherwise be uncounted.

    Missing it would silently under-count every eval-driven tool call \u2014 exactly
    the composition-heavy runs the accuracy figure is most wanted for. ``origin``
    separates them because a nested call is the model's CODE calling a tool, not
    the model emitting one, and conflating the two would let a scripted retry
    loop dominate the figure.
    """

    async def execute(tool_call_id, args, signal, on_update, context):
        await context.dispatch_tool("read", {"path": "a"})
        await context.dispatch_tool("nope_tool", {"path": "a"})
        await context.dispatch_tool("read", {"path": 123})
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="eval", content=[TextContent(text="done")]
        )

    eval_tool = AgentTool(
        name="eval",
        parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        execute=execute,
    )
    recorded = await _run(
        _calls((0, "c1", "eval", '{"code":"x"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [_ok_tool(), eval_tool],
    )
    nested = [(n, f) for n, o, f, _ms in recorded if o == "nested"]
    assert nested == [("read", ""), ("nope_tool", "unknown_tool"), ("read", "invalid_arguments")]
    # The outer eval call itself is a MODEL call and recorded as one.
    assert ("eval", "model", "") in [(n, o, f) for n, o, f, _ms in recorded]


@pytest.mark.asyncio
async def test_a_raising_callback_cannot_break_a_turn():
    """Analytics must never raise into a turn \u2014 the package's stated contract.

    A host hook that throws is a host bug, and the loop still has to finish the
    turn and pair every tool result. Guarded in the loop rather than trusted to
    the host, because a measurement that can kill the thing it measures is not
    a measurement.
    """
    calls: list[str] = []

    def explode(name, origin, fault, ms):
        calls.append(name)
        raise RuntimeError("analytics is broken")

    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=_Scripted(
            [
                _calls((0, "c1", "read", '{"path":"a"}')) + [StreamEndEvent(stop_reason="toolUse")],
                [StreamEndEvent(stop_reason="stop")],
            ]
        ),
        record_tool_call=explode,
    )
    context = LoopContext(
        system_blocks=["sys"], tools=[_ok_tool()], tool_context=ToolContext(session_id="s1")
    )
    results = [event async for event in AgentLoop().run([], context, config)]
    assert calls == ["read"], "the hook was reached"
    assert results, "the turn completed despite the hook raising"


@pytest.mark.asyncio
async def test_no_callback_configured_is_a_silent_no_op():
    """A host that does not want analytics pays nothing and sees no error."""
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=_Scripted(
            [
                _calls((0, "c1", "read", '{"path":"a"}')) + [StreamEndEvent(stop_reason="toolUse")],
                [StreamEndEvent(stop_reason="stop")],
            ]
        ),
    )
    context = LoopContext(
        system_blocks=["sys"], tools=[_ok_tool()], tool_context=ToolContext(session_id="s1")
    )
    assert [event async for event in AgentLoop().run([], context, config)]


@pytest.mark.asyncio
async def test_a_session_less_context_records_nothing():
    """No session id means no row to attribute: recording it would be a lie."""
    recorded: list[Any] = []
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=_Scripted(
            [
                _calls((0, "c1", "read", '{"path":"a"}')) + [StreamEndEvent(stop_reason="toolUse")],
                [StreamEndEvent(stop_reason="stop")],
            ]
        ),
        record_tool_call=lambda *a: recorded.append(a),
    )
    context = LoopContext(
        system_blocks=["sys"], tools=[_ok_tool()], tool_context=ToolContext(session_id="")
    )
    async for _ in AgentLoop().run([], context, config):
        pass
    assert recorded == []


@pytest.mark.asyncio
async def test_the_origins_the_loop_emits_are_exactly_the_ones_the_reader_partitions_on():
    """Pins the writer to the reader across a deliberate layering gap.

    ``harness`` carries no analytics dependency by design (see
    ``LoopConfig.record_tool_call``), so the loop passes these strings as
    literals while ``analytics.model`` names them as constants. Nothing but this
    test stops the two sides drifting on a spelling — and a drift would not
    fail, it would silently move every model-emitted call into the nested
    bucket, emptying the benchmarking figure's denominator instead of raising.
    """
    from local_operator.analytics.model import ORIGIN_MODEL, ORIGIN_NESTED

    async def execute(tool_call_id, args, signal, on_update, context):
        await context.dispatch_tool("read", {"path": "a"})
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="eval", content=[TextContent(text="done")]
        )

    eval_tool = AgentTool(
        name="eval",
        parameters={"type": "object", "properties": {"code": {"type": "string"}}},
        execute=execute,
    )
    recorded = await _run(
        _calls((0, "c1", "eval", '{"code":"x"}')) + [StreamEndEvent(stop_reason="toolUse")],
        [_ok_tool(), eval_tool],
    )
    emitted_origins = {origin for _n, origin, _f, _ms in recorded}
    assert emitted_origins == {ORIGIN_MODEL, ORIGIN_NESTED}
