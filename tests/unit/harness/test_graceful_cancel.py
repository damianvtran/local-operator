"""Boundary-respecting cancel: stop after the tool finishes, not mid-tool.

The distinction these tests defend is the whole reason the feature exists. An
``abort`` fires the turn's signal and cancels the running tool task, which is
right for a human pressing Esc. A supervised agent's tools have external side
effects a machine operator cannot repair — a half-transferred ``git push``, a
partly-created merge request — so a supervisor's cancel must land at the
boundary where every call has ALREADY produced a paired result, and before the
next model request is spent.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    AgentEndEvent,
    AgentTool,
    Message,
    StreamEndEvent,
    StreamToolCallDelta,
    TextContent,
    ToolExecutionEndEvent,
    ToolResult,
)
from tests.unit.harness.test_loop import make_config


def _tool(name: str = "touch", on_execute: Any = None) -> AgentTool:
    async def execute(  # type: ignore[no-untyped-def]
        tool_call_id, args, signal, on_update, context
    ):
        if on_execute is not None:
            on_execute()
        return ToolResult(
            tool_call_id=tool_call_id, tool_name=name, content=[TextContent(text="ok")]
        )

    return AgentTool(name=name, parameters={"type": "object"}, execute=execute)


class _Scripted:
    """One tool call on the first model call, plain text on any later one.

    ``calls`` is the assertion that matters: it proves whether the loop spent
    another provider request after the cancel.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, request: Any, signal: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            yield StreamToolCallDelta(index=0, id="call_1", name="touch")
            yield StreamToolCallDelta(index=0, argument_delta="{}")
            yield StreamEndEvent(stop_reason="toolUse")
            return
        yield StreamEndEvent(stop_reason="stop")


async def _events(tool: AgentTool, **config_kwargs: Any) -> tuple[list[Any], _Scripted]:
    stream = _Scripted()
    context = LoopContext(tools=[tool])
    config = make_config(stream, **config_kwargs)
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]
    return events, stream


@pytest.mark.asyncio
async def test_cancel_lands_after_the_tool_completes_not_during() -> None:
    """The in-flight tool RUNS TO COMPLETION and no second model call is made.

    This is the property that protects an external side effect: the tool body
    finishes even though the cancel becomes pending while it is running.
    """
    completed: list[str] = []
    pending = {"cancel": False}

    def during_tool() -> None:
        # The supervisor cancels WHILE the tool is in flight, mirroring a
        # cancel pressed during a push.
        pending["cancel"] = True
        completed.append("tool finished")

    events, stream = await _events(
        _tool(on_execute=during_tool),
        graceful_cancel_requested=lambda: pending["cancel"],
    )

    assert completed == ["tool finished"], "the tool must not be cut mid-flight"
    assert stream.calls == 1, "no model request may be spent after the cancel"
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and ends[-1].aborted is True, "the turn must end as aborted"
    # The finished work stays in the transcript as a paired tool result.
    assert any(isinstance(event, ToolExecutionEndEvent) for event in events)


@pytest.mark.asyncio
async def test_no_cancel_runs_to_normal_completion() -> None:
    """Control: with no cancel pending the loop takes its second model call."""
    events, stream = await _events(_tool(), graceful_cancel_requested=lambda: False)

    assert stream.calls == 2
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and not ends[-1].aborted


@pytest.mark.asyncio
async def test_absent_hook_preserves_existing_behaviour() -> None:
    """``None`` — every existing host — must behave exactly as before."""
    events, stream = await _events(_tool())

    assert stream.calls == 2
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and not ends[-1].aborted


@pytest.mark.asyncio
async def test_a_raising_hook_does_not_strand_the_turn() -> None:
    """A broken host hook degrades to "no cancel", never kills the run."""

    def explode() -> bool:
        raise RuntimeError("host is broken")

    events, stream = await _events(_tool(), graceful_cancel_requested=explode)

    assert stream.calls == 2, "a raising hook must not stop the turn"
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and not ends[-1].aborted
