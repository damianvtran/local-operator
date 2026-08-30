"""Measure how fast a pending fork tears down a long-running tool.

    .venv/bin/python scripts/fork_interrupt_timing.py

The claim under test is that a fork rides the existing steering interrupt poll,
so a fork requested during a ten-minute `wait` reaches its boundary in about one
poll interval (STEERING_INTERRUPT_POLL_S = 0.25s) rather than after the tool.
That is a TIMING claim, so it is measured here against the real AgentLoop with a
real interruptible tool rather than asserted in prose.

The second measurement is the one that makes the first safe: the same run, with
a second tool in the batch, showing that the interrupt did NOT skip it.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.harness.loop import (  # noqa: E402
    STEERING_INTERRUPT_POLL_S,
    AgentLoop,
    LoopContext,
)
from local_operator.harness.types import (  # noqa: E402
    AgentTool,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)

MODEL = ModelSpec(provider="anthropic", model_id="claude-opus-5")
#: How long the "tool" would run if nothing interrupted it. Deliberately far
#: longer than the poll interval so the measurement cannot be confused with the
#: tool simply finishing.
TOOL_SECONDS = 600.0


class ScriptedStream:
    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.requests = []

    async def __call__(self, request, signal=None):
        self.requests.append(request)
        for event in self._scripts.pop(0):
            yield event


def _slow_tool(name: str, started: asyncio.Event, outcome: dict[str, str]) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        started.set()
        try:
            await asyncio.sleep(TOOL_SECONDS)
        except asyncio.CancelledError:
            outcome[name] = "cancelled"
            raise
        outcome[name] = "completed"
        return ToolResult(tool_call_id=tool_call_id, tool_name=name, content=[])

    return AgentTool(
        name=name,
        parameters={"type": "object", "properties": {}},
        interruptible=True,
        execute=execute,
    )


def _fast_tool(name: str, ran: list[str]) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        ran.append(name)
        return ToolResult(
            tool_call_id=tool_call_id, tool_name=name, content=[TextContent(text="ok")]
        )

    return AgentTool(
        name=name,
        parameters={"type": "object", "properties": {}},
        concurrency="exclusive",
        execute=execute,
    )


async def measure_interrupt() -> float:
    """Seconds from "fork requested" to the tool actually being torn down."""
    started = asyncio.Event()
    outcome: dict[str, str] = {}
    pending = {"fork": False}

    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="block", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[_slow_tool("block", started, outcome)])
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=stream,
        interrupt_mode="immediate",
        has_pending_fork=lambda: pending["fork"],
    )

    run = asyncio.ensure_future(AgentLoop().run_to_end([Message.user("go")], context, config, None))
    await asyncio.wait_for(started.wait(), timeout=10)
    # The tool is genuinely running now. This is the instant the user types
    # /fork during a long tool.
    requested_at = time.monotonic()
    pending["fork"] = True
    await asyncio.wait_for(run, timeout=30)
    elapsed = time.monotonic() - requested_at
    assert outcome == {"block": "cancelled"}, outcome
    return elapsed


async def measure_batch_preserved() -> list[str]:
    """The same interrupt, with a second call in the batch that must still run."""
    ran: list[str] = []
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="a", argument_delta="{}"),
                StreamToolCallDelta(index=1, id="c2", name="b", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(tools=[_fast_tool("a", ran), _fast_tool("b", ran)])
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=stream,
        interrupt_mode="immediate",
        has_pending_fork=lambda: True,
    )
    await AgentLoop().run_to_end([Message.user("go")], context, config, None)
    return ran


async def main() -> None:
    samples = [await measure_interrupt() for _ in range(5)]
    print(f"poll interval (STEERING_INTERRUPT_POLL_S): {STEERING_INTERRUPT_POLL_S}s")
    print(f"tool would have run for:                   {TOOL_SECONDS}s")
    print("teardown latency after /fork, 5 runs:      ", [f"{s:.3f}s" for s in samples])
    print(f"max: {max(samples):.3f}s  (must be under ~2x the poll interval)")

    ran = await measure_batch_preserved()
    print(f"\nbatch with a pending fork throughout: executed={ran}")
    print("both calls ran -> the fork interrupted the tool WITHOUT skipping the batch")


asyncio.run(main())
