"""Invariants on the harness event/message contract types.

These are the shapes every front end (TUI, server websockets, exec --json) and
the compaction layer program against, so a field that can contradict itself is
a UI defect waiting to happen rather than a style question.
"""

from __future__ import annotations

from local_operator.harness.types import (
    TextContent,
    ToolExecutionEndEvent,
    ToolResult,
)


def test_tool_end_error_flag_cannot_disagree_with_result():
    """``ToolExecutionEndEvent.is_error`` mirrors ``result.is_error``.

    UIs and the JSON exec stream read the event-level flag, so a producer that
    sets only the result's flag renders a failed tool as a success — the exact
    defect the TUI showed (a ``permission denied`` grep result drawn with the
    success glyph) before this invariant existed.
    """
    failed = ToolResult(
        tool_call_id="t1",
        tool_name="grep",
        content=[TextContent(text="permission denied")],
        is_error=True,
    )
    event = ToolExecutionEndEvent(tool_call_id="t1", tool_name="grep", result=failed)
    assert event.is_error is True
    assert event.model_dump()["is_error"] is True


def test_tool_end_clean_result_stays_clean():
    ok = ToolResult(tool_call_id="t2", tool_name="read", content=[TextContent(text="ok")])
    clean = ToolExecutionEndEvent(tool_call_id="t2", tool_name="read", result=ok)
    assert clean.is_error is False


def test_explicit_event_flag_is_never_downgraded():
    """The loop stamps aborted/synthetic results via the event-level flag, so a
    clean result must not clear it."""
    ok = ToolResult(tool_call_id="t3", tool_name="bash", content=[TextContent(text="ok")])
    forced = ToolExecutionEndEvent(tool_call_id="t3", tool_name="bash", result=ok, is_error=True)
    assert forced.is_error is True
