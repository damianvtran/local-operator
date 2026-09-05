"""The projection fold is the phone's single source of render semantics —
these tests pin the TUI-parity contract: one line per tool call, diff counts,
streaming rows that update in place, subagent roster aggregation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.comms import SubagentComms
from local_operator.harness.types import (
    AgentEndEvent,
    AgentMessage,
    AgentStartEvent,
    CustomMessage,
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
    _projection_from_json,
)
from local_operator.session.runtime.registry import SessionRecord
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


def test_legacy_subagent_projection_rebuilds_new_detail_collections() -> None:
    """A rolling upgrade can pair an older registrant with the current daemon.

    The older row lacks every recursive-detail collection added by this PR;
    rebuilding at the shared socket seam must produce the complete browser
    contract rather than forwarding undefined values into React.
    """
    record = SessionRecord(
        pid=42,
        kind="tui",
        session_id="s1",
        conversation_name="legacy",
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="secret",
    )
    projection = _projection_from_json(
        {
            "session_id": "s1",
            "pid": 0,
            "subagents": [{"job_id": "legacy", "label": "legacy child"}],
        },
        record,
    )

    row = projection.subagents[0]
    assert projection.pid == 42
    assert row.ancestors == []
    assert row.ancestor_ids == []
    assert row.child_ids == []
    assert row.peer_ids == []
    assert row.transcript == []
    assert row.todos == []
    assert projection.to_json()["subagents"][0]["child_ids"] == []


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
    assert by_id["grandchild"].ancestor_ids == ["parent", "child"]
    assert by_id["grandchild"].activity == "reading"

    session.jobs.rows["grandchild"].latest_details = {"progress": "summarizing"}
    fold.set_subagent_details(comms)
    refreshed = {row.job_id: row for row in fold.projection.subagents}
    assert refreshed["grandchild"].activity == "summarizing"


def test_nested_subagent_completion_refreshes_selected_detail() -> None:
    """A nested row has no root lifecycle event to settle its phone detail."""

    job = SimpleNamespace(
        status="running",
        agent_role="coder",
        model_label="test/running",
        latest_details={"progress": "working"},
        result_text=None,
        error_text=None,
    )
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: job))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("parent", "parent")
    comms.record_launch("nested", "nested", parent_job_id="parent")
    fold = make_fold()

    fold.set_subagent_details(comms)
    selected = {row.job_id: row for row in fold.projection.subagents}["nested"]
    assert (selected.status, selected.progress, selected.activity) == (
        "running",
        "working",
        "working",
    )

    job.status = "completed"
    job.model_label = "test/completed"
    job.latest_details = {}
    job.result_text = "finished result"
    fold.set_subagent_details(comms)

    selected = {row["job_id"]: row for row in fold.projection.to_json()["subagents"]}["nested"]
    assert selected["status"] == "completed"
    assert selected["model_label"] == "test/completed"
    assert selected["result_text"] == "finished result"
    assert selected["error_text"] == ""
    assert selected["progress"] == ""
    assert selected["activity"] == ""


@pytest.mark.parametrize(
    ("status", "result_text", "error_text"),
    [
        ("completed", "durable completed result", None),
        ("failed", None, "provider failed; retry from the parent"),
    ],
)
def test_swept_nested_outcome_survives_fresh_projection_and_reconnect(
    status: str, result_text: str | None, error_text: str | None
) -> None:
    """The comms record is the only lifecycle source after manager retention."""

    class Jobs:
        def __init__(self) -> None:
            self.rows = {
                "nested": SimpleNamespace(
                    status="running",
                    agent_role="coder",
                    model_label="test/model",
                    latest_details={"progress": "stale progress"},
                    result_text=None,
                    error_text=None,
                )
            }

        def get(self, job_id: str):
            return self.rows.get(job_id)

    session = SimpleNamespace(jobs=Jobs())
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("parent", "parent")
    comms.record_launch("nested", "nested", parent_job_id="parent")
    fold = make_fold()
    fold.set_subagent_details(comms)
    assert {row.job_id: row for row in fold.projection.subagents}["nested"].activity == (
        "stale progress"
    )

    comms.record_outcome("nested", status, error_text=error_text, result_text=result_text)
    del session.jobs.rows["nested"]

    # Refreshing the selected projection must clear stale live activity, and a
    # brand-new fold models the first SSE snapshot after a reconnect.
    fold.set_subagent_details(comms)
    reconnect = make_fold()
    reconnect.set_subagent_details(comms)
    for projection in (fold.projection, reconnect.projection):
        selected = {row.job_id: row for row in projection.subagents}["nested"]
        assert selected.status == status
        assert selected.result_text == (result_text or "")
        assert selected.error_text == (error_text or "")
        assert selected.progress == ""
        assert selected.activity == ""


@pytest.mark.parametrize(
    "parents",
    [
        {"a": "a"},
        {"a": "b", "b": "a"},
    ],
)
def test_subagent_metadata_projection_tolerates_legacy_parent_cycles(
    parents,
) -> None:  # noqa: ANN001
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: None))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    for job_id, parent_id in parents.items():
        comms.record_launch(job_id, job_id, parent_job_id=parent_id)
    fold = make_fold()
    fold.set_subagent_details(comms)
    assert {row.job_id for row in fold.projection.subagents} == set(parents)


def test_subagent_metadata_projection_never_constructs_transcript(monkeypatch) -> None:
    """The ordinary event path must remain memory-only regardless of child count."""
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: None))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    for index in range(75):
        comms.record_launch(f"child-{index}", f"child {index}")

    class ForbiddenTranscript:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            raise AssertionError("metadata projection opened a child transcript")

    monkeypatch.setattr("local_operator.session.transcript.Transcript", ForbiddenTranscript)
    fold = make_fold()
    for _ in range(100):
        fold.set_subagent_details(comms)
    assert len(fold.projection.subagents) == 75


def test_hydrated_subagent_details_never_place_transcript_on_the_wire() -> None:
    """A subagent's transcript is fetched lazily, never carried in the fold.

    Embedding even a tail-capped child transcript per subagent pushed the
    full-repaint projection past the daemon's 1 MB control-frame limit, so every
    push was dropped as oversized and the phone fell back to the stale durable
    fold. ``set_subagent_hydrated_details`` must therefore land todos (small,
    needed for the live working line) while leaving ``row.transcript`` empty; the
    transcript is served on demand from the child-history endpoint instead.
    """
    from local_operator.mobile.types import TranscriptEntry

    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: None))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("child", "child")
    fold = make_fold()
    fold.set_subagent_details(comms)

    heavy = [
        TranscriptEntry(id=f"row-{i}", kind="assistant", text="x" * 4096)
        for i in range(PROJECTION_TRANSCRIPT_LIMIT * 2)
    ]
    assert fold.set_subagent_hydrated_details(
        "child", heavy, [{"text": "verify", "status": "pending"}]
    )
    row = fold._subagents["child"]
    assert row.transcript == []
    assert row.todos and row.todos[0].items[0].text == "verify"
    # The serialized wire frame must contain no subagent transcript entries.
    wire = fold.projection.to_json()
    assert all(sub["transcript"] == [] for sub in wire["subagents"])


def test_live_fold_bounds_subagent_prompt_and_outcome_on_the_wire() -> None:
    """Uncapped prompt/result text re-wedges the frame the transcript cap saved.

    The list projection is a full repaint pushed ~30x/s and every subagent row
    rides in it, so uncapped prompt/result/error text scales the frame with
    roster depth: a power-user session at 80+ subagents put hundreds of KB of
    prompt text into one frame, back toward the 1 MB control-frame cap. The row
    only needs a preview; the full text is retained by the daemon and served
    through getSubagentDetail. This pins the wire bounds so the regression cannot
    silently return.
    """
    from local_operator.mobile.projection import (
        SUBAGENT_OUTCOME_CHARS,
        SUBAGENT_PROMPT_PREVIEW_CHARS,
    )

    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: None))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("child", "child", prompt="P" * 50_000)
    comms.record_outcome("child", "completed", result_text="R" * 50_000)
    fold = make_fold()
    fold.set_subagent_details(comms)

    wire_row = fold.projection.to_json()["subagents"][0]
    assert len(wire_row["prompt"]) <= SUBAGENT_PROMPT_PREVIEW_CHARS
    assert len(wire_row["result_text"]) <= SUBAGENT_OUTCOME_CHARS
    # An empty error field stays empty (a cap must not manufacture a placeholder).
    assert wire_row["error_text"] == ""


def test_live_fold_keeps_failed_child_error_text_generous() -> None:
    """A failed child's ``error_text`` must survive on the wire, unlike result.

    ``error_text`` is ``str(exc)`` from the parent runner and is never in the
    child transcript, so the lazy /history fetch cannot recover it — the wire
    value is the only copy the phone's Outcome panel renders. Capping it at the
    200-char ``result_text`` preview would truncate the failure tail everywhere
    with no recovery (F1), so the live lifecycle merge must carry it generously
    (``SUBAGENT_ERROR_CHARS``) while still bounding it. Pins that behaviour and
    that a multi-line trace keeps its line breaks.
    """
    from local_operator.mobile.projection import (
        SUBAGENT_ERROR_CHARS,
        SUBAGENT_OUTCOME_CHARS,
    )

    error = "Traceback (most recent call last):\n" + "\n".join(
        f"  frame {i}: boom in module_{i}" for i in range(200)
    )
    assert len(error) > SUBAGENT_ERROR_CHARS  # long enough to exercise the cap
    job = SimpleNamespace(
        status="running",
        agent_role="reviewer",
        model_label="test/model",
        latest_details={"progress": "checking"},
        result_text=None,
        error_text=None,
    )
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: job))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("child", "child")
    comms.record_outcome("child", "failed", error_text=error)
    fold = make_fold()
    fold.set_subagent_details(comms)

    wire_row = fold.projection.to_json()["subagents"][0]
    # NOT clipped to the 200-char result preview; the failure tail rides.
    assert len(wire_row["error_text"]) > SUBAGENT_OUTCOME_CHARS
    assert len(wire_row["error_text"]) <= SUBAGENT_ERROR_CHARS
    assert "\n" in wire_row["error_text"]  # multi-line structure preserved


def test_recorded_terminal_outcome_never_regresses_to_running_job_row() -> None:
    """The runner records terminal state before the manager stamps its row."""

    job = SimpleNamespace(
        status="running",
        agent_role="coder",
        model_label="test/model",
        latest_details={"progress": "stale progress"},
        result_text=None,
        error_text=None,
    )
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: job))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("nested", "nested")
    comms.record_outcome("nested", "completed", result_text="settled result")

    fold = make_fold()
    fold.set_subagent_details(comms)
    selected = fold.projection.subagents[0]
    assert (selected.status, selected.result_text) == ("completed", "settled result")
    assert (selected.progress, selected.activity) == ("", "")


def test_nested_subagent_failure_refreshes_error_and_clears_progress() -> None:
    job = SimpleNamespace(
        status="running",
        agent_role="reviewer",
        model_label="test/model",
        latest_details={"progress": "checking"},
        result_text=None,
        error_text=None,
    )
    session = SimpleNamespace(jobs=SimpleNamespace(get=lambda job_id: job))
    comms = SubagentComms(cast(Session, cast(Any, session)))
    comms.record_launch("parent", "parent")
    comms.record_launch("nested", "nested", parent_job_id="parent")
    fold = make_fold()

    fold.set_subagent_details(comms)
    job.status = "failed"
    job.latest_details = {"progress": "stale progress"}
    job.error_text = "provider failed"
    fold.set_subagent_details(comms)

    selected = {row.job_id: row for row in fold.projection.subagents}["nested"]
    assert selected.status == "failed"
    assert selected.result_text == ""
    assert selected.error_text == "provider failed"
    assert selected.progress == ""
    assert selected.activity == ""


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


def test_history_fold_maps_peer_message_to_its_own_kind() -> None:
    from local_operator.session.peer import PEER_MESSAGE_MESSAGE_TYPE

    fold = make_fold()
    sender = {"pid": 42, "conversation_name": "peer", "model_label": "test/model"}
    peer = CustomMessage(
        custom_type=PEER_MESSAGE_MESSAGE_TYPE,
        attribution="user",
        details={"text": "<wrapped>hi</wrapped>", "body": "hi there", "sender": sender},
    )
    fold.fold_history([peer])
    rows = fold.projection.transcript
    assert len(rows) == 1
    # The phone renders the raw body, never the model-facing wrapped envelope,
    # and carries the sender for the card label.
    assert rows[0].kind == "peer_message"
    assert rows[0].text == "hi there"
    assert rows[0].details["sender"] == sender


def test_note_peer_message_appends_optimistic_row() -> None:
    fold = make_fold()
    fold.note_peer_message("live echo", sender={"pid": 7, "conversation_name": "peer"})
    row = fold.projection.transcript[-1]
    assert row.kind == "peer_message"
    assert row.text == "live echo"
    assert row.details["sender"]["pid"] == 7


def _steer_envelope(body: str) -> str:
    """The model-facing wrapper ``SubagentComms._format_to_child`` builds for a
    steer, restated here so the fold tests pin the render contract against the
    exact shape a persisted steer row carries."""
    return (
        "<parent-message>\n"
        "This changes your instructions. Apply it from now on, and drop work it "
        "makes pointless.\n\n"
        f"{body}\n"
        "</parent-message>"
    )


def test_history_fold_renders_a_persisted_steer_as_the_parents_words() -> None:
    """A hub steer persists as a plain user Message whose text is the
    model-facing envelope. The phone must show the parent's own words as a
    parent_message row, never the XML — the phone's history fold sees only
    LLM-visible messages (the journaled fact is a custom row that never
    replays), so body extraction is its only path."""
    from local_operator.mobile.projection import fold_messages_to_entries

    steer = Message.user(_steer_envelope("Focus on retries"), id="steer-1")
    entries = fold_messages_to_entries([Message.user("do the thing"), steer])

    assert [(entry.kind, entry.text) for entry in entries] == [
        ("user", "do the thing"),
        ("parent_message", "Focus on retries"),
    ]
    assert "<parent-message>" not in " ".join(entry.text for entry in entries)


def test_fold_history_renders_a_persisted_steer_as_the_parents_words() -> None:
    """The attach rebuild applies the same rule as the lazy-load fold, so an
    attaching phone and a paged history agree about the row."""
    fold = make_fold()
    fold.fold_history([Message.user(_steer_envelope("Focus on retries"), id="steer-1")])

    rows = fold.projection.transcript
    assert len(rows) == 1
    assert rows[0].kind == "parent_message"
    assert rows[0].text == "Focus on retries"
    assert "<parent-message>" not in rows[0].text


def test_a_live_delivered_steer_never_renders_the_envelope() -> None:
    """A hub steer delivered mid-turn announces itself as a user
    MessageStartEvent carrying the envelope text. The fold must paint the
    parent's words, not the XML, exactly like the durable folds."""
    fold = make_fold()
    message = Message.user(_steer_envelope("Focus on retries"), id="steer-1")

    added = fold.absorb_user_event(message)

    assert added is True
    row = fold.projection.transcript[-1]
    assert row.kind == "parent_message"
    assert row.text == "Focus on retries"
    assert "<parent-message>" not in row.text


def test_a_phone_typed_envelope_reconciles_its_echo_instead_of_doubling() -> None:
    """A message the phone sent that happens to carry the envelope shape must
    reconcile against its optimistic echo, not append beside it.

    The envelope branch used to return before the pending-echo pop, so the row
    rendered twice under ONE id — and the stranded echo still displayed the raw
    XML this path exists to suppress. Two rows sharing an id is also bad input
    for the web client's list reconciliation.
    """
    fold = make_fold()
    envelope = _steer_envelope("Focus on retries")
    fold.note_user_message(envelope, steer=True, message_id="cmd-1")

    added = fold.absorb_user_event(Message.user(envelope, id="cmd-1"))

    assert added is False
    rows = fold.projection.transcript
    assert len(rows) == 1
    assert (rows[0].id, rows[0].kind, rows[0].text) == (
        "cmd-1",
        "parent_message",
        "Focus on retries",
    )
    assert "<parent-message>" not in rows[0].text


def test_a_user_quoting_the_envelope_keeps_their_own_words() -> None:
    """Extraction requires the builder's exact instruction preamble, so a human
    quoting the wrapper keeps their words and their ``user`` row."""
    from local_operator.mobile.projection import fold_messages_to_entries

    quoted = "<parent-message>\nwhy does my log show this?\n\nsecret plan\n</parent-message>"
    entries = fold_messages_to_entries([Message.user(quoted, id="human-1")])

    assert [(entry.kind, entry.text) for entry in entries] == [("user", quoted)]


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


def test_working_line_says_thinking_until_text_actually_streams() -> None:
    """``message_start`` is a model call in flight, not prose.

    The loop yields ``MessageStartEvent`` from a placeholder at the top of every
    provider call, before the first token; a tool-only call never streams text
    after it. Folding that as "responding" told the phone the model was
    writing for the whole of every call. The TUI's WorkingBlock keys on the
    first non-empty delta, and the phone's working line follows the same rule.
    """
    fold = make_fold()
    fold.fold_event(AgentStartEvent(generation=1))

    # A tool-only call: placeholder, then the tool, with no text between.
    first = Message.assistant()
    fold.fold_event(MessageStartEvent(message=first))
    assert fold.projection.activity == "thinking"
    fold.fold_event(MessageEndEvent(message=first))
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="c1", tool_name="bash", args={}, intent="probing")
    )
    assert fold.projection.activity == "probing"
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="bash",
            result=ToolResult(tool_call_id="c1", content=[TextContent(text="ok")], is_error=False),
        )
    )
    assert fold.projection.activity == "thinking"

    # A prose call: still the model-wait until the first non-empty delta.
    second = Message.assistant()
    fold.fold_event(MessageStartEvent(message=second))
    assert fold.projection.activity == "thinking"
    fold.fold_event(MessageUpdateEvent(message=second, delta=""))
    assert fold.projection.activity == "thinking"
    fold.fold_event(MessageUpdateEvent(message=second, delta="Here "))
    assert fold.projection.activity == "responding"
    fold.fold_event(MessageUpdateEvent(message=second, delta="it is."))
    assert fold.projection.activity == "responding"
    fold.fold_event(
        MessageEndEvent(
            message=second.model_copy(update={"content": [TextContent(text="Here it is.")]})
        )
    )
    assert fold.projection.activity == "thinking"
    fold.fold_event(AgentEndEvent(generation=1))
    assert fold.projection.activity == ""


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


# --- issue #231: user echoes de-duped by id, not by a 3-entry tail window ----


def test_a_phone_steer_is_not_repainted_once_assistant_rows_push_it_out() -> None:
    """Issue #231, reproduced: the defect the tail window guaranteed.

    A phone steer is echoed optimistically, then the turn keeps running — the
    engine only drains the steering queue at a LATER tool boundary, so by the
    time the message's own ``MessageStartEvent`` arrives the echo is several
    assistant rows back. The three-entry scan could no longer see it and
    painted the steer a second time. The registry keys on the message id, so
    distance from the tail is irrelevant.
    """
    fold = make_fold()
    command_id = "cmd-steer-1"
    fold.note_user_message("use the other endpoint", steer=True, message_id=command_id)
    assert len([e for e in fold.projection.transcript if e.kind == "steer"]) == 1

    # The turn goes on: four assistant rows push the echo well past the window
    # the old scan looked at.
    for index in range(4):
        fold.fold_event(
            MessageStartEvent(message=Message.assistant(f"still working {index}")),
        )
    assert len(fold.projection.transcript) == 5

    # Only NOW is the steer drained and announced, carrying the id the handle
    # supplied as the message id.
    fold.absorb_user_event(Message.user("use the other endpoint", id=command_id))

    rows = [e for e in fold.projection.transcript if e.text == "use the other endpoint"]
    assert len(rows) == 1, "the drain repainted a steer already on the phone"
    assert rows[0].kind == "steer", "the echo's own row survives, not a fresh user row"


def test_absorb_user_event_returns_false_for_a_registered_echo() -> None:
    """The de-dupe contract the callers read: a registered echo is folded into
    the existing row, so the fold reports it did NOT add one."""
    fold = make_fold()
    fold.note_user_message("already on screen", steer=True, message_id="cmd-2")
    added = fold.absorb_user_event(Message.user("already on screen", id="cmd-2"))
    assert added is False
    assert len([e for e in fold.projection.transcript if e.text == "already on screen"]) == 1


def test_a_distinct_message_with_colliding_words_still_paints() -> None:
    """The mirror image of the same defect: a genuinely NEW message whose text
    matched a recent row was swallowed by the window. Different id, different
    message, so it must appear."""
    fold = make_fold()
    fold.note_user_message("continue", steer=True, message_id="cmd-3")
    fold.absorb_user_event(Message.user("continue", id="a-different-message"))
    rows = [e for e in fold.projection.transcript if e.text == "continue"]
    assert len(rows) == 2, "a distinct message must not be eaten by a word collision"


def test_a_registered_echo_row_is_not_consumed_by_a_colliding_neighbour() -> None:
    """The tail fallback (for handles with no id) must not spend a row that has
    an EXACT event coming: that row's own announcement would then find its
    entry gone and paint the duplicate this issue exists to remove."""
    fold = make_fold()
    fold.note_user_message("same words", steer=True, message_id="cmd-4")

    # An id-less announcement carrying the same words: not the registered
    # steer's, so it paints rather than consuming that row.
    fold.absorb_user_event(Message.user("same words", id="foreign"))
    assert len([e for e in fold.projection.transcript if e.text == "same words"]) == 2

    # The registered steer's own event still upgrades its row, adding nothing.
    fold.absorb_user_event(Message.user("same words", id="cmd-4"))
    assert len([e for e in fold.projection.transcript if e.text == "same words"]) == 2


def test_the_echoed_row_adopts_the_persisted_message_id() -> None:
    """The row is keyed by the command id until the real message arrives; from
    then on it must carry the message id, which is what a later history fold
    and the web client's list reconciliation agree on."""
    fold = make_fold()
    fold.note_user_message("key me", steer=True, message_id="cmd-5")
    assert fold.projection.transcript[-1].id == "cmd-5"
    fold.absorb_user_event(Message.user("key me", id="cmd-5"))
    assert fold.projection.transcript[-1].id == "cmd-5"


def test_a_legacy_handle_without_an_id_keeps_the_tail_dedup() -> None:
    """A handle that supplies no id (older/third-party) still de-dupes its own
    echo through the historical tail scan — the compatibility path."""
    fold = make_fold()
    fold.note_user_message("no id supplied", steer=True)
    added = fold.absorb_user_event(Message.user("no id supplied"))
    assert added is False
    assert len([e for e in fold.projection.transcript if e.text == "no id supplied"]) == 1


def test_a_history_fold_clears_pending_echoes() -> None:
    """A wholesale rebuild replaces the rows the entries pointed at, so a
    surviving entry would reference a row no longer in the transcript."""
    fold = make_fold()
    fold.note_user_message("pending across the fold", steer=True, message_id="cmd-6")
    assert fold._pending_user_echoes
    fold.fold_history([Message.user("something else entirely")])
    assert fold._pending_user_echoes == {}
