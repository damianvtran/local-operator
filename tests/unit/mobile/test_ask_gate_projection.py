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
from local_operator.mobile.projection import (
    ProjectionFold,
    _summarize_args,
    fold_messages_to_entries,
)
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

    # THE SUMMARY IS ONE OF THOSE CARRIES (Q-5, round 2): the settle mint fills
    # it from the stashed start args, so live gate-on agrees with the gate-off
    # live start arm AND with what every replay paints from the call's
    # arguments. Before the fill the mint row was blank (``summary == ""``)
    # while both other paints showed the args summary — a reconnect visibly
    # changed the row. The three-way equality is the contract; the blocking
    # fold below is the live control and ``_history`` the replay one.
    expected = _summarize_args("ask", ASK_ARGS)
    assert rows[0].summary == expected, "the settle mint carries the args summary"

    blocking = make_fold(queued=False)
    blocking.fold_event(ToolCallComposeEvent(tool_call_id="call-ask", tool_name="ask"))
    blocking.fold_event(
        ToolExecutionStartEvent(tool_call_id="call-ask", tool_name="ask", args=ASK_ARGS)
    )
    _end(blocking, marker=False)
    assert _tool_rows(blocking)[0].summary == expected, "the gate-off live paint agrees"

    replayed = [
        entry for entry in fold_messages_to_entries(_history(marker=False)) if entry.kind == "tool"
    ]
    assert replayed and replayed[0].summary == expected, "the replay paint agrees"


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


#: The two fields the §4 byte-identity line's unit substitution excludes, each
#: for a MEASURED reason rather than convenience (the literal claim that the
#: queued and blocking arms serialize identically is false on exactly these):
#: ``version`` counts a row's fold updates and the settle-only arm legitimately
#: makes fewer (its row is born at settle); ``intent`` is the compose frame's
#: live annotation, and the settle-only contract refuses to REGISTER at compose
#: by design, while the END frame — the only one the queued arm folds into a
#: row — carries none. ``summary`` WAS on this list through round 1 for that
#: same END-frame reason; the Q-5 settle-mint fill now derives it from the
#: stashed start args, measured equal to the blocking arm's start-derived and
#: the replay's call-derived summary, so it is unmasked and pinned equal instead
#: (see the three-way assert in the mint cell below). Everything else (ids,
#: order, states, details, output, flags) must be byte-equal, which is what this
#: pins; the wire-level ``LOP_ASK_GATE=0`` parity is QA's cell 2b.
_PARITY_ASIDE_FIELDS = ("version", "intent")


def _masked(rows: list[dict[str, Any]]) -> bytes:
    import json

    masked: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        for field in _PARITY_ASIDE_FIELDS:
            if field in copy:
                copy[field] = None
        masked.append(copy)
    return json.dumps(masked, sort_keys=True, default=str).encode()


def test_a_non_diverted_ask_is_projection_identical_to_the_gate_off_run() -> None:
    """A raise leaves NO trace: queued-engine fold == blocking-arm fold.

    The §4 line's "``SessionProjection`` byte-identical for a probe session
    without diverts" as a unit cell: the SAME event sequence (compose with an
    intent, start with args, unmarked end) folded twice — once under the queued
    engine, once on the blocking arm that never probes — and compared after the
    documented field mask above. Anything the gate could leak (a marker, a
    changed state, a dropped detail) fails here.
    """

    def run(*, queued: bool) -> list[dict[str, Any]]:
        fold = make_fold(queued=queued)
        fold.fold_event(
            ToolCallComposeEvent(tool_call_id="call-ask", tool_name="ask", intent="asking")
        )
        fold.fold_event(
            ToolExecutionStartEvent(tool_call_id="call-ask", tool_name="ask", args=ASK_ARGS)
        )
        _end(fold, marker=False)
        return [row.to_json() for row in fold.projection.transcript]

    assert _masked(run(queued=True)) == _masked(
        run(queued=False)
    ), "a raise differing beyond the masked lifecycle fields is a gate leak"


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


def _history(*, marker: bool) -> list[Any]:
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
