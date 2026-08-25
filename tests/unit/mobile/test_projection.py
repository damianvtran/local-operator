"""The projection fold is the phone's single source of render semantics —
these tests pin the TUI-parity contract: one line per tool call, diff counts,
streaming rows that update in place, subagent roster aggregation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from local_operator.harness.comms import SubagentComms
from local_operator.harness.types import (
    AgentEndEvent,
    AgentMessage,
    AgentStartEvent,
    ImageContent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelChangeEvent,
    NoticeEvent,
    SubagentEndEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    TextContent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
    TurnEndEvent,
)
from local_operator.mobile.projection import (
    ProjectionFold,
    _diff_counts,
    _image_refs,
    _summarize_args,
)
from local_operator.mobile.types import (
    PROJECTION_TRANSCRIPT_LIMIT,
    PendingRequest,
    SessionProjection,
)
from local_operator.session.session import Session


def make_fold() -> ProjectionFold:
    return ProjectionFold(SessionProjection(session_id="s1", pid=1))


def test_model_change_repaints_the_composer_chip() -> None:
    """A fallback must rename the chip the phone shows, not just the notice."""
    fold = make_fold()
    fold.projection.model_label = "anthropic/claude-opus-4-8"
    fold.fold_event(
        ModelChangeEvent(
            provider="xai",
            model_id="grok-4.6",
            effort="high",
            is_fallback=True,
            reason="quota exhausted",
        )
    )
    assert fold.projection.model_label == "xai/grok-4.6"
    assert fold.projection.effort == "high"


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
            duration_s=1.25,
            result=ToolResult(
                tool_call_id="t1",
                content=[TextContent(text="wrote 10 lines")],
                details={"added": 8, "removed": 2, "diff": "@@"},
                duration_s=1.25,
            ),
        )
    )
    assert row.tool_state == "done"
    assert (row.diff_added, row.diff_removed) == (8, 2)
    assert row.details["output"] == "wrote 10 lines"
    assert row.details["diff"] == "@@"
    assert row.details["args"]["path"] == "/a/b/c.py"
    assert row.elapsed_s == 1.2


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


def test_subagent_details_seed_nested_descendants_for_recursive_navigation() -> None:
    """Nested jobs never emit lifecycle events through the root fold.

    The shared registry must still project a complete root -> child ->
    grandchild graph so every advertised child id resolves on the phone, and a
    later refresh must keep enriching that same nested record.
    """

    class Jobs:
        def __init__(self) -> None:
            self.rows = {
                "parent": SimpleNamespace(status="running", agent_role="coder", latest_details={}),
                "child": SimpleNamespace(
                    status="completed", agent_role="reviewer", latest_details={}
                ),
                "grandchild": SimpleNamespace(
                    status="running", agent_role="scout", latest_details={"progress": "reading"}
                ),
            }

        def get(self, job_id: str):
            return self.rows.get(job_id)

    session = SimpleNamespace(jobs=Jobs())
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("parent", "parent", prompt="plan")
    comms.record_launch("child", "child", parent_job_id="parent", prompt="build")
    comms.record_launch("grandchild", "grandchild", parent_job_id="child", prompt="inspect")
    fold = make_fold()
    # Only the direct child reaches the root event stream.
    fold.fold_event(SubagentStartEvent(job_id="parent", label="parent"))

    fold.set_subagent_details(comms)
    by_id = {row.job_id: row for row in fold.projection.subagents}
    assert set(by_id) == {"parent", "child", "grandchild"}
    assert by_id["parent"].child_ids == ["child"]
    assert by_id["child"].child_ids == ["grandchild"]
    assert by_id["grandchild"].parent_job_id == "child"
    assert by_id["grandchild"].ancestors == ["parent", "child"]
    assert by_id["grandchild"].activity == "reading"

    session.jobs.rows["grandchild"].latest_details = {"progress": "summarizing"}
    fold.set_subagent_details(comms)
    refreshed = {row.job_id: row for row in fold.projection.subagents}
    assert refreshed["grandchild"].activity == "summarizing"


def test_history_fold_pairs_tool_calls_with_results() -> None:
    fold = make_fold()
    call = ToolCall(id="c1", name="read", arguments={"path": "/x.py"})
    history: list[AgentMessage] = [
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


def test_mid_run_turn_end_does_not_settle_streaming() -> None:
    """Regression: a multi-batch turn must stay "in progress" across the
    per-model-turn boundaries inside it.

    ``TurnEndEvent`` fires after every assistant turn that made tool calls and
    the run will continue (harness.loop ~589), so a turn with several tool
    batches emits several TurnEndEvents before its single AgentEndEvent. The
    session keeps ``is_streaming`` True across them and the TUI working line
    stays up; the phone must match. Folding TurnEndEvent as a streaming
    terminal blanked the working line mid-run and (with the reconcile latch)
    pinned it off. Only AgentEndEvent settles the turn.
    """
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))

    # Batch 1: a tool runs, then a mid-run TurnEndEvent (more work to come).
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="c1", tool_name="bash", args={}, intent="step 1")
    )
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="bash",
            result=ToolResult(tool_call_id="c1", content=[TextContent(text="ok")], is_error=False),
        )
    )
    fold.fold_event(TurnEndEvent(message=Message.assistant()))

    # Still streaming; the working line shows the model-wait, not blank.
    assert fold.projection.streaming is True
    assert fold.projection.activity == "thinking"

    # Batch 2 continues normally, then the run's single terminal settles it.
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="c2", tool_name="bash", args={}, intent="step 2")
    )
    assert fold.projection.streaming is True
    assert fold.projection.activity == "step 2"
    fold.fold_event(AgentEndEvent(generation=1))
    assert fold.projection.streaming is False
    assert fold.projection.activity == ""
    assert fold.projection.stop_reason == "completed"


def test_pinned_opener_never_costs_the_newest_row() -> None:
    """Regression: once the transcript passed the cap WITH the opening user
    message pinned at the head, the newest row stopped reaching the phone.

    ``_cap_tail`` pins the first user message and fills the rest from the tail.
    It used to make room by dropping ``tail[-1]`` — the row JUST appended — and
    because the cap runs on every append, the transcript froze: past the cap no
    new tool call or notice ever appeared (the field report's missing "last
    several tool calls"). The pin must cost the OLDEST tail row, never the
    newest, and the tail must keep advancing.
    """
    fold = make_fold()
    fold.note_user_message("the opening ask — this names the whole conversation")
    for i in range(PROJECTION_TRANSCRIPT_LIMIT + 40):
        fold.fold_event(NoticeEvent(text=f"note {i}"))

    transcript = fold.projection.transcript
    assert len(transcript) == PROJECTION_TRANSCRIPT_LIMIT
    # Opener stays pinned at the head so the phone always knows the topic.
    assert transcript[0].kind == "user"
    assert transcript[0].text.startswith("the opening ask")
    # The single newest row is present — the whole point of the fix.
    assert transcript[-1].text == f"note {PROJECTION_TRANSCRIPT_LIMIT + 39}"

    # The tail keeps advancing on further appends (it used to be frozen).
    fold.fold_event(NoticeEvent(text="the very latest"))
    transcript = fold.projection.transcript
    assert len(transcript) == PROJECTION_TRANSCRIPT_LIMIT
    assert transcript[0].kind == "user"
    assert transcript[-1].text == "the very latest"


def test_todo_refresh_replaces_wholesale() -> None:
    # The store is phased: one implicit "Todos" phase carries a flat list.
    fold = make_fold()
    fold.set_todos(
        [
            {
                "name": "Todos",
                "items": [
                    {"text": "a", "status": "pending"},
                    {"text": "b", "status": "done"},
                ],
            }
        ]
    )
    assert len(fold.projection.todos) == 1
    assert [t.status for t in fold.projection.todos[0].items] == ["pending", "done"]
    fold.set_todos([{"name": "Todos", "items": [{"text": "a", "status": "done"}]}])
    assert len(fold.projection.todos) == 1
    assert len(fold.projection.todos[0].items) == 1


def test_todo_multi_phase_structure_preserved() -> None:
    """A real multi-phase store maps to TodoPhase/TodoItem with every field —
    text/status/reason — intact and phase order preserved."""
    fold = make_fold()
    fold.set_todos(
        [
            {
                "name": "Design",
                "items": [
                    {"text": "sketch", "status": "done"},
                    {"text": "review", "status": "dropped"},
                ],
            },
            {
                "name": "Build",
                "items": [
                    {"text": "impl", "status": "pending"},
                    {"text": "deploy", "status": "blocked", "reason": "waiting on infra"},
                ],
            },
        ]
    )
    phases = fold.projection.todos
    assert [p.name for p in phases] == ["Design", "Build"]
    assert [t.text for t in phases[0].items] == ["sketch", "review"]
    blocked = phases[1].items[1]
    assert blocked.status == "blocked"
    assert blocked.reason == "waiting on infra"
    # to_json is the wire shape the front-end reads: nested phase → items.
    wire = phases[1].to_json()
    assert wire == {
        "name": "Build",
        "items": [
            {"text": "impl", "status": "pending", "reason": ""},
            {"text": "deploy", "status": "blocked", "reason": "waiting on infra"},
        ],
    }


def test_todo_legacy_flat_list_coerced_to_one_phase() -> None:
    """A hand-attached legacy flat list (item dicts at the top level, no phase
    wrapper) is coerced via ``_as_phases`` to a single implicit "Todos" phase
    rather than rendering empty-text rows — the bug the phased callers exposed."""
    fold = make_fold()
    fold.set_todos([{"text": "a", "status": "pending"}, {"text": "b", "status": "done"}])
    assert len(fold.projection.todos) == 1
    assert fold.projection.todos[0].name == "Todos"
    assert [t.text for t in fold.projection.todos[0].items] == ["a", "b"]


def test_todo_open_count_across_phases() -> None:
    """The session-list ``todos_open`` badge counts open (pending OR blocked)
    items across ALL phases; done/dropped never count."""
    fold = make_fold()
    fold.set_todos(
        [
            {
                "name": "A",
                "items": [
                    {"text": "1", "status": "pending"},
                    {"text": "2", "status": "done"},
                ],
            },
            {
                "name": "B",
                "items": [
                    {"text": "3", "status": "blocked", "reason": "x"},
                    {"text": "4", "status": "dropped"},
                    {"text": "5", "status": "pending"},
                ],
            },
        ]
    )
    # Mirror the daemon's list-summary arithmetic: pending + blocked across
    # phases = 3 (items 1, 3, 5), done/dropped excluded.
    p = fold.projection
    open_count = sum(
        1 for phase in p.todos for t in phase.items if t.status in ("pending", "blocked")
    )
    assert open_count == 3


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


def test_pending_queue_shows_front_and_counts() -> None:
    """A parallel tool batch opens several approvals at once. The fold must
    show the FRONT one and report the total, and clearing one by id must
    surface the next — not dismiss every sibling (the mobile hang report)."""
    fold = make_fold()
    a = PendingRequest(request_id="a", kind="approval", title="bash")
    b = PendingRequest(request_id="b", kind="approval", title="write")
    fold.push_pending(a)
    fold.push_pending(b)
    assert fold.projection.pending is a
    assert fold.projection.pending_count == 2

    # Answering the FRONT reveals the next, count decrements.
    fold.pop_pending("a")
    assert fold.projection.pending is b
    assert fold.projection.pending_count == 1

    # Answering an out-of-order id (the second card) is honoured too.
    fold.pop_pending("b")
    assert fold.projection.pending is None
    assert fold.projection.pending_count == 0


def test_pop_pending_out_of_order_keeps_the_other_card() -> None:
    """Concurrent gates settle in whatever order the user answers, not the
    order enqueued: popping the second must leave the first still showing."""
    fold = make_fold()
    fold.push_pending(PendingRequest(request_id="a", kind="approval", title="bash"))
    fold.push_pending(PendingRequest(request_id="b", kind="ask", title="which?"))
    fold.pop_pending("b")
    assert fold.projection.pending is not None
    assert fold.projection.pending.request_id == "a"
    assert fold.projection.pending_count == 1


def test_set_pending_still_replaces_for_the_tui_mirror() -> None:
    """The TUI-mirror handle uses set_pending: the terminal serializes its own
    approvals, so the phone mirrors exactly one card or none."""
    fold = make_fold()
    fold.push_pending(PendingRequest(request_id="a", kind="approval", title="bash"))
    fold.set_pending(PendingRequest(request_id="z", kind="ask", title="q"))
    assert fold.projection.pending_count == 1
    assert fold.projection.pending is not None
    assert fold.projection.pending.request_id == "z"
    fold.set_pending(None)
    assert fold.projection.pending is None
    assert fold.projection.pending_count == 0


def test_image_refs_are_index_and_mime_only() -> None:
    """User-turn attachments project as lightweight references (image-only
    index + mime), never bytes — the pixels are fetched lazily so a per-token
    repaint stays small. A text caption does not shift the index."""
    message = Message.user(
        "look at these",
        [
            ImageContent(data="AAAA", mime_type="image/png"),
            ImageContent(data="BBBB", mime_type="image/jpeg"),
        ],
    )
    refs = _image_refs(message)
    assert refs == [
        {"index": 0, "mime_type": "image/png"},
        {"index": 1, "mime_type": "image/jpeg"},
    ]
    # No base64 leaks into the reference.
    assert all("data" not in r for r in refs)


def test_history_fold_carries_image_refs_on_user_rows() -> None:
    fold = make_fold()
    history: list[AgentMessage] = [
        Message.user("with a shot", [ImageContent(data="AAAA", mime_type="image/png")]),
    ]
    fold.fold_history(history)
    user_rows = [e for e in fold.projection.transcript if e.kind == "user"]
    assert len(user_rows) == 1
    assert user_rows[0].images == [{"index": 0, "mime_type": "image/png"}]


def test_absorb_user_event_upgrades_echoed_row_with_image_refs() -> None:
    """A phone-sent prompt is echoed WITHOUT refs (the handle has no persisted
    id yet); the real MessageStartEvent then carries the attachments, and the
    fold must upgrade the echoed row in place so the sender sees thumbnails."""
    fold = make_fold()
    fold.note_user_message("with a shot")  # optimistic echo, no images
    echoed = fold.projection.transcript[-1]
    assert echoed.images == []

    message = Message.user("with a shot", [ImageContent(data="AAAA", mime_type="image/png")])
    added = fold.absorb_user_event(message)
    assert added is False  # de-duped, not a second row
    assert len([e for e in fold.projection.transcript if e.kind == "user"]) == 1
    upgraded = fold.projection.transcript[-1]
    assert upgraded.id == message.id
    assert upgraded.images == [{"index": 0, "mime_type": "image/png"}]
