"""A child's bounded trajectory must not spend its window on invisible rows.

The subagent page renders the last ``TRAJECTORY_CAP`` entries of a child's
trajectory and dispatches on ``type`` with no branch for ``reasoning_delta``: the
rows would paint nothing while consuming the window. Reasoning is one event per
reasoning token, so without the skip in ``_make_relay`` an ordinary reasoning
child evicts the tool calls and messages the page exists to show — measured on
the shipped relay and page fold: 3 model calls x 250 fragments left one tool row
of three (review round 1, MAJOR-1). This drives the SHIPPED relay, so the
assertion is about the rows a reader would actually get.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.jobs import TRAJECTORY_SEQ_KEY, AsyncJobManager
from local_operator.harness.subagent import TRAJECTORY_CAP, _make_relay
from local_operator.harness.types import (
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ReasoningDeltaEvent,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)


class _Jobs:
    """The one jobs-manager call the relay makes (roster invalidation).

    Cast rather than faked wholesale: the relay touches exactly this member on
    the paths this test drives, and a real ``AsyncJobManager`` would drag the
    scheduler and a job store into a test about a list of rows.
    """

    def _notify_roster_change(self) -> None:
        return None


async def _absorb(event: Any) -> None:
    """The parent-stream sink: progress events are not what this test measures."""
    return None


def _relay() -> tuple[Any, Any]:
    job = SimpleNamespace(trajectory=[])
    relay = _make_relay(
        "job-1",
        "child",
        job,
        cast(AsyncJobManager, _Jobs()),
        _absorb,
        lambda _progress: None,
        {},
    )
    return job, relay


def _assistant(text: str = "answer") -> Message:
    return Message(id="m1", role="assistant", content=[TextContent(text=text)])


async def _reasoning_call(relay: Any, fragments: int) -> None:
    """One model call that thinks: start, N fragments, one tool, end."""
    await relay(MessageStartEvent(message=_assistant("")))
    for index in range(fragments):
        await relay(ReasoningDeltaEvent(message_id="m1", delta=f"thought{index} "))
    await relay(
        ToolExecutionStartEvent(tool_call_id="c1", tool_name="Read", intent="reading the file")
    )
    await relay(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="Read",
            result=ToolResult(tool_call_id="c1", tool_name="Read", content=[]),
        )
    )
    await relay(MessageEndEvent(message=_assistant()))


@pytest.mark.asyncio
async def test_a_reasoning_child_keeps_every_visible_row() -> None:
    """Three thinking calls, 250 fragments each: no reasoning row, no eviction.

    The counts are the reviewer's repro; ``TRAJECTORY_CAP`` is 500, so 750
    fragments alone would have filled the window twice over and left the page
    showing one tool row of three.
    """
    job, relay = _relay()
    for _ in range(3):
        await _reasoning_call(relay, 250)

    kinds = [entry["type"] for entry in job.trajectory]
    assert "reasoning_delta" not in kinds, "display-only rows must not consume the window"
    assert kinds.count("tool_execution_start") == 3
    assert kinds.count("tool_execution_end") == 3
    assert kinds.count("message_end") == 3
    assert len(job.trajectory) < TRAJECTORY_CAP


@pytest.mark.asyncio
async def test_everything_a_reader_can_see_still_reaches_the_trajectory() -> None:
    """The skip is the reasoning family only — the discriminating half.

    A relay that dropped more than ``reasoning_delta`` would pass the test above
    and break the page, so the visible families are asserted by name, including a
    message that streams TEXT, which is the row nearest the one being skipped.
    """
    job, relay = _relay()
    await relay(MessageStartEvent(message=_assistant("")))
    await relay(MessageUpdateEvent(message=_assistant("half"), delta="half"))
    await relay(ReasoningDeltaEvent(message_id="m1", delta="a thought"))
    await relay(
        ToolExecutionStartEvent(tool_call_id="c1", tool_name="Read", intent="reading the file")
    )
    await relay(MessageEndEvent(message=_assistant()))

    kinds = [entry["type"] for entry in job.trajectory]
    assert kinds == [
        "message_start",
        "message_update",
        "tool_execution_start",
        "message_end",
    ]
    # Every retained entry carries the identity the page keys its rows by, and
    # the sequence is dense because nothing invisible consumed a number.
    assert [entry[TRAJECTORY_SEQ_KEY] for entry in job.trajectory] == [0, 1, 2, 3]
