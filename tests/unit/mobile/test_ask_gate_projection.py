"""The ask gate on the phone's projections: settle-only rows, marker drops.

Design docs/design/ask-gate.md §3 row 8. The live fold under ``queued_engine``
is the phone's settle-only arm — no row while an ask composes or runs, one row
at settle for a raise, nothing for a divert (the marker) — and the history fold
subtracts a diverted ask's call chip and result row so every client receives
clean rows. A never-run ending is EXEMPT from the settle-only refusal (the
design-review contract line): it paints and settles exactly as with the gate
off.
"""

from __future__ import annotations

from typing import Any

from local_operator.harness.types import (
    Message,
    TextContent,
    ToolCall,
    ToolCallComposeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.mobile.projection import ProjectionFold, fold_messages_to_entries
from local_operator.mobile.types import SessionProjection

MARKER = {"ask_gate": {"hidden": True, "verdict": "clear", "reason": "plainly best"}}
ASK_ARGS = {
    "questions": [
        {
            "id": "q0",
            "question": "Which database?",
            "options": [{"label": "staging"}, {"label": "prod"}],
        }
    ]
}


def make_fold(*, queued: bool) -> ProjectionFold:
    return ProjectionFold(SessionProjection(session_id="s1", pid=1), queued_engine=queued)


def _tool_rows(fold: ProjectionFold) -> list[Any]:
    return [row for row in fold.projection.transcript if row.kind == "tool"]


def _end(fold: ProjectionFold, *, marker: bool, text: str = "Ask a-1 queued.") -> None:
    fold.fold_event(
        ToolExecutionEndEvent(
            tool_call_id="call-ask",
            tool_name="ask",
            result=ToolResult(
                tool_call_id="call-ask",
                tool_name="ask",
                content=[TextContent(text=text)],
                details=dict(MARKER) if marker else {},
            ),
        )
    )


# --- the live fold -----------------------------------------------------------


def test_queued_engine_suppresses_live_rows_and_mounts_only_at_settle() -> None:
    fold = make_fold(queued=True)

    fold.fold_event(ToolCallComposeEvent(tool_call_id="call-ask", tool_name="ask"))
    assert _tool_rows(fold) == [], "no dictation row while the gate may divert"

    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="call-ask", tool_name="ask", args=ASK_ARGS)
    )
    assert _tool_rows(fold) == [], "no running row while the gate may divert"

    _end(fold, marker=False)
    rows = _tool_rows(fold)
    assert len(rows) == 1, "a raise's receipt row appears at settle"
    assert rows[0].tool_state == "done"
    # The identity from the START frame rode through: the receipt row carries
    # the question the ask was about (details stringify list values, so match
    # on the content, not the container).
    assert "Which database?" in str(rows[0].details["args"])


def test_queued_engine_drops_the_diverted_result() -> None:
    fold = make_fold(queued=True)
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="call-ask", tool_name="ask", args=ASK_ARGS)
    )
    _end(fold, marker=True, text="[Ask clearance] No question was put to the user.")
    assert _tool_rows(fold) == [], "a diverted ask settles nothing"


def test_the_blocking_arm_keeps_todays_mount_and_drops_on_the_marker() -> None:
    """Mode False: the dictation row mounts as today; a marker result removes it."""
    fold = make_fold(queued=False)
    fold.fold_event(ToolCallComposeEvent(tool_call_id="call-ask", tool_name="ask", intent="asking"))
    assert len(_tool_rows(fold)) == 1, "today's mount on the blocking arm"

    _end(fold, marker=True)
    assert _tool_rows(fold) == [], "the mixed-build fallback drops on the marker"


def test_the_blocking_arm_settles_a_normal_result_as_today() -> None:
    fold = make_fold(queued=False)
    fold.fold_event(
        ToolExecutionStartEvent(tool_call_id="call-ask", tool_name="ask", args=ASK_ARGS)
    )
    _end(fold, marker=False)
    rows = _tool_rows(fold)
    assert len(rows) == 1 and rows[0].tool_state == "done"


def test_never_run_ending_paints_under_the_queued_engine() -> None:
    """The contract line: a died-before-asking call must stay visible."""
    fold = make_fold(queued=True)
    fold.fold_event(
        ToolCallComposeEvent(
            tool_call_id="call-ask",
            tool_name="ask",
            not_run_reason="the turn ended before the call could run",
        )
    )
    rows = _tool_rows(fold)
    assert len(rows) == 1, "a never-run ask paints its verdict row"
    assert rows[0].tool_state in ("failed", "interrupted")


# --- the history fold --------------------------------------------------------


def _history(*, marker: bool) -> list[object]:
    return [
        Message.user("deploy it"),
        Message(
            role="assistant",
            content=[TextContent(text="")],
            tool_calls=[ToolCall(id="call-ask", name="ask", arguments=ASK_ARGS)],
        ),
        Message(
            role="tool",
            content=[TextContent(text="[Ask clearance] No question was put to the user.")],
            tool_call_id="call-ask",
            tool_name="ask",
            provider_payload={"details": dict(MARKER)} if marker else None,
        ),
    ]


def test_history_fold_skips_a_diverted_ask_pair() -> None:
    entries = fold_messages_to_entries(_history(marker=True))
    tools = [entry for entry in entries if entry.kind == "tool"]
    assert tools == [], "no chip, no result row for a divert"
    assert all("Ask clearance" not in (entry.text or "") for entry in entries)


def test_history_fold_keeps_a_visible_ask_pair() -> None:
    entries = fold_messages_to_entries(_history(marker=False))
    tools = [entry for entry in entries if entry.kind == "tool"]
    assert len(tools) == 1, "the control: an ordinary ask keeps its chip and row"
    assert tools[0].tool_call_id == "call-ask"
