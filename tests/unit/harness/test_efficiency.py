"""Behavioral guards for reduced tool steps without relaxed execution gates.

The concurrency assertions observe overlap/order, not laptop timing. Timeouts
only diagnose a deadlock when the old serial implementation cannot satisfy an
explicit rendezvous. Programmatic tests drive the real eval worker through the
assembled loop, so a bridge that merely exposes a Python helper cannot pass.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest

from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    AbortSignal,
    AgentEndEvent,
    AgentTool,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolContext,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from tests.unit.harness.test_loop import ScriptedStream

MODEL = ModelSpec(provider="test", model_id="efficiency")


@pytest.mark.asyncio
async def test_resource_identity_probes_run_off_the_event_loop():
    loop_thread = threading.get_ident()
    probe_threads = []

    def resources(args, cwd):
        probe_threads.append(threading.get_ident())
        return ("path",)

    async def execute(call_id, args, signal, update, context):
        return ToolResult(tool_call_id=call_id, tool_name="write")

    tool = AgentTool(
        name="write",
        description="write",
        parameters={},
        execute=execute,
        concurrency="exclusive",
        resource_keys=resources,
    )
    await _run([tool], [_calls([("write", {})]), _done()])
    assert probe_threads and all(thread != loop_thread for thread in probe_threads)


def _calls(names_args: list[tuple[str, dict[str, Any]]], turn: int = 0):
    return [
        *[
            StreamToolCallDelta(
                index=i, id=f"{turn}:{i}", name=name, argument_delta=json.dumps(args)
            )
            for i, (name, args) in enumerate(names_args)
        ],
        StreamEndEvent(stop_reason="toolUse"),
    ]


def _done():
    return [StreamTextDelta(delta="complete"), StreamEndEvent(stop_reason="stop")]


async def _run(tools, turns, *, context=None, fallback=None, signal=None):
    stream = ScriptedStream(turns)
    state = LoopContext(tools=tools, tool_context=context or ToolContext())
    config = LoopConfig(
        model=MODEL,
        stream_fn=stream,
        resolve_fallback_tool=fallback,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
    )
    events = [
        event async for event in AgentLoop().run([Message.user("work")], state, config, signal)
    ]
    return state, stream, events


@pytest.mark.asyncio
async def test_independent_keyed_writes_overlap_but_same_resource_stays_ordered():
    started = set()
    together = asyncio.Event()

    async def execute(call_id, args, signal, update, context):
        started.add(args["path"])
        if len(started) == 2:
            together.set()
        await asyncio.wait_for(together.wait(), 5)
        return ToolResult(
            tool_call_id=call_id, tool_name="write", content=[TextContent(text=args["path"])]
        )

    tool = AgentTool(
        name="write",
        description="write",
        concurrency="exclusive",
        parameters={"type": "object"},
        execute=execute,
        resource_keys=lambda args, cwd: (args["path"],),
    )
    _, _, events = await _run(
        [tool], [_calls([("write", {"path": "a"}), ("write", {"path": "b"})]), _done()]
    )
    assert not any(
        event.result.is_error for event in events if isinstance(event, ToolExecutionEndEvent)
    )
    assert started == {"a", "b"}

    active = peak = 0
    order = []

    async def serial(call_id, args, signal, update, context):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        order.append(call_id)
        active -= 1
        return ToolResult(tool_call_id=call_id, tool_name="write")

    tool.execute = serial
    await _run([tool], [_calls([("write", {"path": "a"})] * 3), _done()])
    assert peak == 1
    assert order == ["0:0", "0:1", "0:2"]


@pytest.mark.asyncio
async def test_shared_fanout_is_bounded_without_losing_results():
    active = peak = 0

    async def execute(call_id, args, signal, update, context):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return ToolResult(tool_call_id=call_id, tool_name="read")

    tool = AgentTool(name="read", description="read", parameters={}, execute=execute)
    _, _, events = await _run([tool], [_calls([("read", {})] * 25), _done()])
    assert peak == 8
    assert len([e for e in events if isinstance(e, ToolExecutionEndEvent)]) == 25


@pytest.mark.asyncio
async def test_shared_slots_refill_without_waiting_for_the_slowest_first_wave():
    ninth_started = asyncio.Event()
    active = peak = 0

    async def execute(call_id, args, signal, update, context):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            if args["index"] == 0:
                await asyncio.wait_for(ninth_started.wait(), 5)
            elif args["index"] == 8:
                ninth_started.set()
            await asyncio.sleep(0)
            return ToolResult(tool_call_id=call_id, tool_name="read")
        finally:
            active -= 1

    tool = AgentTool(name="read", description="read", parameters={}, execute=execute)
    _, _, events = await _run(
        [tool], [_calls([("read", {"index": index}) for index in range(25)]), _done()]
    )
    assert ninth_started.is_set()
    assert peak <= 8
    ends = [event for event in events if isinstance(event, ToolExecutionEndEvent)]
    assert len(ends) == 25
    assert not any(event.is_error for event in ends)


@pytest.mark.asyncio
async def test_abort_backfills_queued_calls_without_inventing_execution_events():
    signal = AbortSignal()
    block = asyncio.Event()
    executed = []

    async def execute(call_id, args, abort, update, context):
        executed.append(call_id)
        if len(executed) == 8:
            signal.abort()
        await block.wait()
        raise AssertionError("abort must cancel the active runners")

    tool = AgentTool(name="read", description="read", parameters={}, execute=execute)
    state, _, events = await asyncio.wait_for(
        _run([tool], [_calls([("read", {})] * 25)], signal=signal), 5
    )
    starts = [e.tool_call_id for e in events if isinstance(e, ToolExecutionStartEvent)]
    ends = [e.tool_call_id for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert len(executed) == 8
    assert sorted(starts) == sorted(ends) == sorted(executed)
    assert len([m for m in state.messages if isinstance(m, Message) and m.role == "tool"]) == 25


@pytest.mark.asyncio
async def test_legacy_read_write_barriers_and_unknown_resources_remain_serial():
    order = []

    async def execute(call_id, args, signal, update, context):
        order.append((args["label"], "start"))
        await asyncio.sleep(0)
        order.append((args["label"], "end"))
        return ToolResult(tool_call_id=call_id, tool_name="tool")

    read = AgentTool(name="read", description="read", parameters={}, execute=execute)
    write = AgentTool(
        name="write",
        description="write",
        parameters={},
        execute=execute,
        concurrency="exclusive",
        resource_keys=lambda args, cwd: ("file",),
    )
    await _run(
        [read, write],
        [
            _calls(
                [
                    ("read", {"label": "before"}),
                    ("write", {"label": "write"}),
                    ("read", {"label": "after"}),
                ]
            ),
            _done(),
        ],
    )
    assert order == [
        (label, edge) for label in ("before", "write", "after") for edge in ("start", "end")
    ]


@pytest.mark.asyncio
async def test_exact_error_cycle_gets_one_recovery_then_a_bounded_stop():
    async def execute(call_id, args, signal, update, context):
        return ToolResult(
            tool_call_id=call_id,
            tool_name="broken",
            is_error=True,
            content=[TextContent(text="unknown column")],
        )

    tool = AgentTool(name="broken", description="broken", parameters={}, execute=execute)
    state, stream, events = await _run(
        [tool],
        [
            *[_calls([("broken", {"column": "missing"})], turn=i) for i in range(7)],
            _done(),
        ],
    )
    assert len(stream.requests) == 6
    assert (
        sum("Harness recovery notice" in m.text for m in state.messages if isinstance(m, Message))
        == 1
    )
    terminal = next(e for e in reversed(events) if isinstance(e, AgentEndEvent))
    assert terminal.error and terminal.error.startswith("No progress:")
    assert len([m for m in state.messages if isinstance(m, Message) and m.role == "tool"]) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("success", [True, False])
async def test_successful_polling_and_changing_errors_do_not_trigger_guard(success):
    count = 0

    async def execute(call_id, args, signal, update, context):
        nonlocal count
        count += 1
        return ToolResult(
            tool_call_id=call_id,
            tool_name="poll",
            is_error=not success,
            content=[TextContent(text="pending" if success else f"attempt {count}")],
        )

    tool = AgentTool(name="poll", description="poll", parameters={}, execute=execute)
    _, stream, events = await _run(
        [tool],
        [
            *[_calls([("poll", {})], turn=i) for i in range(8)],
            _done(),
        ],
    )
    assert len(stream.requests) == 9
    assert next(e for e in reversed(events) if isinstance(e, AgentEndEvent)).error is None


@pytest.mark.asyncio
async def test_real_eval_composes_discovered_calls_with_validation_and_approval(tmp_path):
    from local_operator.tools.eval import _KERNELS, _close_kernel, build_eval_tool

    executed = []
    approvals = []

    async def execute(call_id, args, signal, update, context):
        executed.append(args["n"])
        return ToolResult(
            tool_call_id=call_id,
            tool_name="mcp__records",
            content=[
                TextContent(text=json.dumps({"value": args["n"] * 2, "irrelevant": "x" * 50000}))
            ],
        )

    async def approve(name, summary):
        approvals.append(name)
        return name != "denied"

    remote = AgentTool(
        name="mcp__records",
        description="records",
        approval_tier="write",
        parameters={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
            "additionalProperties": False,
        },
        execute=execute,
    )
    denied = remote.model_copy(update={"name": "denied"})
    context = ToolContext(cwd=str(tmp_path), session_id=str(tmp_path), request_approval=approve)
    code = (
        "import json\n"
        'large = tool("large", content="z" * 100000)\n'
        'assert large["content"][0]["text"] == "100000"\n'
        'total = sum(json.loads(tool("mcp__records", n=n)["content"][0]["text"])["value"] '
        "for n in range(3))\n"
        'assert tool("mcp__records", n="invalid")["is_error"]\n'
        'assert tool("denied", n=10)["is_error"]\n'
        'assert tool("unavailable", n=10)["is_error"]\n'
        'assert tool("eval", code="pass")["is_error"]\n'
        "total"
    )
    try:

        async def large_call(call_id, args, signal, update, context):
            return ToolResult(
                tool_call_id=call_id,
                tool_name="large",
                content=[TextContent(text=str(len(args["content"])))],
            )

        large_tool = AgentTool(
            name="large",
            description="large frame",
            parameters={},
            approval_tier="read",
            execute=large_call,
        )
        state, stream, events = await _run(
            [build_eval_tool()],
            [_calls([("eval", {"code": code})]), _done()],
            context=context,
            fallback={remote.name: remote, "denied": denied, "large": large_tool}.get,
        )
        result = next(m for m in state.messages if isinstance(m, Message) and m.role == "tool")
        assert not result.is_error, result.text
        assert "result: 6" in result.text
        assert executed == [0, 1, 2]
        assert approvals.count("mcp__records") == 3
        assert "denied" in approvals
        assert "irrelevant" not in stream.requests[-1].messages[-1].text
        assert all([t.name for t in r.tools] == ["eval"] for r in stream.requests)
        starts = [e.tool_call_id for e in events if isinstance(e, ToolExecutionStartEvent)]
        ends = [e.tool_call_id for e in events if isinstance(e, ToolExecutionEndEvent)]
        assert sorted(starts) == sorted(ends)
    finally:
        kernel = _KERNELS.pop(context.session_id, None)
        if kernel is not None:
            await _close_kernel(kernel)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_large_bridge_response_preserves_execution_outcome(tmp_path, failed):
    from local_operator.tools.eval import build_eval_tool, close_session_kernel

    executions = []

    async def execute(call_id, args, signal, update, context):
        executions.append(call_id)
        return ToolResult(
            tool_call_id=call_id,
            tool_name="mutation",
            is_error=failed,
            content=[TextContent(text="x" * 2_000_000)],
        )

    tool = AgentTool(name="mutation", description="write receipt", parameters={}, execute=execute)
    context = ToolContext(cwd=str(tmp_path), session_id=str(tmp_path))
    code = (
        'receipt = tool("mutation")\n'
        f'assert receipt["is_error"] is {failed}\n'
        'assert receipt["details"]["executed"]\n'
        'assert receipt["details"]["bridge_truncated"]\n'
        'assert "Do not repeat a mutation" in receipt["content"][0]["text"]\n'
        '"outcome preserved"'
    )
    try:
        state, _, _ = await _run(
            [build_eval_tool()],
            [_calls([("eval", {"code": code})]), _done()],
            context=context,
            fallback={tool.name: tool}.get,
        )
        assert len(executions) == 1
        result = next(m for m in state.messages if isinstance(m, Message) and m.role == "tool")
        assert not result.is_error, result.text
        assert "outcome preserved" in result.text
    finally:
        await close_session_kernel(context.session_id)


@pytest.mark.asyncio
async def test_daemon_thread_cannot_inherit_next_eval_cells_bridge(tmp_path):
    from local_operator.tools.eval import build_eval_tool, close_session_kernel

    executions = []

    async def execute(call_id, args, signal, update, context):
        executions.append(call_id)
        return ToolResult(tool_call_id=call_id, tool_name="read")

    remote = AgentTool(name="read", description="read", parameters={}, execute=execute)
    context = ToolContext(cwd=str(tmp_path), session_id=str(tmp_path))
    first = '''import threading
release = threading.Event()
finished = threading.Event()
errors = []
old_alias = tool
def stale_thread():
    release.wait()
    try:
        old_alias("read")
    except RuntimeError as exc:
        errors.append(str(exc))
    finally:
        finished.set()
threading.Thread(target=stale_thread, daemon=True).start()
"ready"'''
    second = '''release.set()
assert finished.wait(10), "stale bridge reader wedged"
assert errors and "foreground eval execution thread" in errors[0]
assert not tool("read")["is_error"]
"foreground preserved"'''
    try:
        state, _, _ = await _run(
            [build_eval_tool()],
            [
                call
                for call in (
                    _calls([("eval", {"code": first})], 0),
                    _calls([("eval", {"code": second})], 1),
                    _done(),
                )
            ],
            context=context,
            fallback={remote.name: remote}.get,
        )
        results = [m for m in state.messages if isinstance(m, Message) and m.role == "tool"]
        assert len(executions) == 1
        assert all(not m.is_error for m in results), [m.text for m in results]
        assert "foreground preserved" in results[-1].text
    finally:
        await close_session_kernel(context.session_id)
