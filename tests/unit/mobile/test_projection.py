"""The projection fold is the phone's single source of render semantics —
these tests pin the TUI-parity contract: one line per tool call, diff counts,
streaming rows that update in place, subagent roster aggregation."""

from __future__ import annotations

from local_operator.harness.types import (
    AgentEndEvent,
    AgentStartEvent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    NoticeEvent,
    SubagentEndEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    TextContent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.mobile.projection import (
    ProjectionFold,
    _diff_counts,
    _summarize_args,
)
from local_operator.mobile.types import PROJECTION_TRANSCRIPT_LIMIT, SessionProjection


def make_fold() -> ProjectionFold:
    return ProjectionFold(SessionProjection(session_id="s1", pid=1))


def test_streaming_assistant_row_updates_in_place() -> None:
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))
    assert fold.projection.streaming is True

    message = Message.assistant()
    fold.fold_event(MessageStartEvent(message=message))
    fold.fold_event(MessageUpdateEvent(message=message, delta="Hel"))
    fold.fold_event(MessageUpdateEvent(message=message, delta="lo"))
    fold.fold_event(
        MessageEndEvent(message=message.model_copy(update={"content": [TextContent(text="Hello")]}))
    )
    fold.fold_event(AgentEndEvent(generation=1))

    rows = [e for e in fold.projection.transcript if e.kind == "assistant"]
    assert len(rows) == 1
    assert rows[0].text == "Hello"
    assert rows[0].final is True
    assert fold.projection.streaming is False


def test_tool_row_lifecycle_one_line_with_diff_counts() -> None:
    fold = make_fold()
    fold.fold_event(
        ToolExecutionStartEvent(
            tool_call_id="t1", tool_name="write", args={"path": "/a/b/c.py", "content": "x"}
        )
    )
    row = fold.projection.transcript[-1]
    assert row.kind == "tool"
    assert row.tool_state == "running"
    assert row.summary == "/a/b/c.py"

    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="t1",
            tool_name="write",
            result=ToolResult(
                tool_call_id="t1",
                content=[TextContent(text="wrote 10 lines")],
                details={"added": 8, "removed": 2, "diff": "@@"},
            ),
        )
    )
    assert row.tool_state == "done"
    assert (row.diff_added, row.diff_removed) == (8, 2)
    assert row.details["output"] == "wrote 10 lines"
    assert row.details["diff"] == "@@"
    assert row.details["args"]["path"] == "/a/b/c.py"


def test_failed_tool_row_carries_the_error() -> None:
    fold = make_fold()
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="t2", tool_name="bash", args={"command": "false"})
    )
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="t2",
            tool_name="bash",
            result=ToolResult(
                tool_call_id="t2", content=[TextContent(text="exit 1: boom")], is_error=True
            ),
        )
    )
    row = fold.projection.transcript[-1]
    assert row.tool_state == "failed"
    assert "boom" in row.error


def test_subagent_roster_running_first_then_settled() -> None:
    fold = make_fold()
    fold.fold_event(SubagentStartEvent(job_id="j1", label="first"))
    fold.fold_event(SubagentStartEvent(job_id="j2", label="second"))
    fold.fold_event(SubagentProgressEvent(job_id="j1", label="first", progress="reading files"))
    fold.fold_event(
        SubagentEndEvent(job_id="j2", label="second", status="completed", result_text="done")
    )

    rows = fold.projection.subagents
    assert rows[0].job_id == "j1"  # running sorts first
    assert rows[0].progress == "reading files"
    assert rows[1].status == "completed"
    assert rows[1].result_text == "done"


def test_history_fold_pairs_tool_calls_with_results() -> None:
    fold = make_fold()
    call = ToolCall(id="c1", name="read", arguments={"path": "/x.py"})
    history = [
        Message.user("look at x"),
        Message.assistant("reading it", tool_calls=[call]),
        Message.tool_result(ToolResult(tool_call_id="c1", content=[TextContent(text="file body")])),
    ]
    fold.fold_history(history)
    kinds = [e.kind for e in fold.projection.transcript]
    assert kinds == ["user", "assistant", "tool"]
    tool_row = fold.projection.transcript[-1]
    assert tool_row.tool_state == "done"
    assert tool_row.summary == "/x.py"
    assert tool_row.details["output"] == "file body"


def test_transcript_is_capped_from_the_front() -> None:
    fold = make_fold()
    for i in range(PROJECTION_TRANSCRIPT_LIMIT + 25):
        fold.fold_event(NoticeEvent(text=f"note {i}"))
    assert len(fold.projection.transcript) == PROJECTION_TRANSCRIPT_LIMIT
    # The OLDEST rows dropped: the tail is what a phone renders.
    assert fold.projection.transcript[-1].text == f"note {PROJECTION_TRANSCRIPT_LIMIT + 24}"
    assert fold.projection.transcript[0].text == "note 25"


def test_todo_refresh_replaces_wholesale() -> None:
    fold = make_fold()
    fold.set_todos([{"text": "a", "status": "pending"}, {"text": "b", "status": "done"}])
    assert [t.status for t in fold.projection.todos] == ["pending", "done"]
    fold.set_todos([{"text": "a", "status": "done"}])
    assert len(fold.projection.todos) == 1


def test_summarize_args_priority_and_compaction() -> None:
    assert _summarize_args("write", {"path": "/p/q.py", "content": "long"}) == "/p/q.py"
    assert _summarize_args("bash", {"command": "ls -la"}) == "ls -la"
    long_value = "x" * 200
    assert len(_summarize_args("read", {"path": long_value})) <= 80
    assert _summarize_args("noop", {}) == "noop"


def test_diff_counts_only_from_reported_details() -> None:
    assert _diff_counts(None) == (0, 0)
    assert _diff_counts({}) == (0, 0)
    assert _diff_counts({"added": 3, "removed": 1}) == (3, 1)
    assert _diff_counts({"lines_added": "5", "lines_removed": 2}) == (5, 2)
    assert _diff_counts({"added": "junk"}) == (0, 0)


def test_projection_version_bumps_on_every_fold() -> None:
    fold = make_fold()
    v0 = fold.projection.version
    fold.fold_event(NoticeEvent(text="hi"))
    assert fold.projection.version > v0
