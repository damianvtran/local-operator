"""The quiet turn's shared predicates (``harness.rows``): one decision per shape.

Design docs/design/quiet-turns.md §5, slice S1. Both folds — the TUI's and the
phone's — read the persisted ``no_reply`` pair through these functions and
neither re-derives the name or the marker, so each predicate is pinned against
the shapes it must accept and the near-misses it must refuse: a rename that
stops matching fails here rather than as a sentinel row painted into a
transcript.

THE LOAD-BEARING NEGATIVE lives at the bottom: the quiet tool is deliberately
not a ``HIDDEN_TOOL_NAMES`` member, and the pair stays in the display pages the
dashboard's quiet-close rule reads (the window-level consequence is pinned in
``tests/unit/session/test_history_window.py``; this file pins the set itself).
"""

from __future__ import annotations

from types import SimpleNamespace

from local_operator.harness.rows import (
    HIDDEN_TOOL_NAMES,
    QUIET_TURN_TOOL,
    is_hidden_tool_call,
    is_hidden_tool_name,
    is_quiet_turn_call,
    is_quiet_turn_name,
    is_quiet_turn_result,
)
from local_operator.harness.types import QUIET_TURN_KEY, ToolCall, ToolResult

# --- is_quiet_turn_name / is_quiet_turn_call ---------------------------------


def test_the_name_predicate_matches_exactly() -> None:
    assert is_quiet_turn_name(QUIET_TURN_TOOL) is True
    assert is_quiet_turn_name("no_reply") is True
    # Near-misses: case, padding and a longer name must all refuse, because
    # this predicate gates paint seams rather than normalising for display.
    for other in (None, "", "no_reply ", " no_reply", "No_Reply", "no_reply_2", "read"):
        assert is_quiet_turn_name(other) is False, other


def test_the_call_predicate_reads_a_tool_call_shape() -> None:
    assert is_quiet_turn_call(ToolCall(id="q1", name=QUIET_TURN_TOOL, arguments={})) is True
    assert is_quiet_turn_call(SimpleNamespace(name=QUIET_TURN_TOOL)) is True
    assert is_quiet_turn_call(SimpleNamespace(name="read")) is False
    assert is_quiet_turn_call(SimpleNamespace()) is False
    assert is_quiet_turn_call(None) is False


# --- is_quiet_turn_result ----------------------------------------------------


def test_the_result_predicate_reads_the_marker() -> None:
    marked = ToolResult(
        tool_call_id="q1", tool_name=QUIET_TURN_TOOL, details={QUIET_TURN_KEY: True}
    )
    assert is_quiet_turn_result(marked) is True
    assert is_quiet_turn_result(SimpleNamespace(details={QUIET_TURN_KEY: True})) is True
    # Negatives: absent details, an empty mapping, and the strictness the core
    # readers share (``is True``, not truthiness — the loop and the session's
    # quiet predicate read the same marker the same way).
    assert is_quiet_turn_result(SimpleNamespace(details={})) is False
    assert is_quiet_turn_result(SimpleNamespace(details=None)) is False
    assert is_quiet_turn_result(SimpleNamespace()) is False
    assert is_quiet_turn_result(None) is False
    assert is_quiet_turn_result(SimpleNamespace(details={QUIET_TURN_KEY: 1})) is False
    assert is_quiet_turn_result(SimpleNamespace(details={"other": True})) is False
    # The marker is the pair's OWN fact: a result whose name never says
    # ``no_reply`` still classifies by the marker, and vice versa — the two
    # predicates answer different questions and the folds ask both.
    assert (
        is_quiet_turn_result(
            ToolResult(tool_call_id="q1", tool_name="", details={QUIET_TURN_KEY: True})
        )
        is True
    )


# --- the load-bearing set exclusion ------------------------------------------


def test_the_quiet_turn_is_not_a_hidden_tool_name() -> None:
    """The pair must stay in the served rows, so it must not join that set.

    ``HIDDEN_TOOL_NAMES`` is subtracted from the display pages and the served
    journal rows; the dashboard needs the pair as the structural close of its
    settled turn (design §5). The folds hide it at their own paint seams
    instead, through the predicates above — NOT through this set. A future
    change that adds the name here would blank the pair on the very surfaces
    that need it, and fails this test first.
    """
    assert QUIET_TURN_TOOL not in HIDDEN_TOOL_NAMES
    assert is_hidden_tool_name(QUIET_TURN_TOOL) is False
    assert is_hidden_tool_call(ToolCall(id="q1", name=QUIET_TURN_TOOL, arguments={})) is False
