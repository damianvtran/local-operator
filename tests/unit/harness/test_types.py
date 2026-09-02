"""Invariants on the harness event/message contract types.

These are the shapes every front end (TUI, server websockets, exec --json) and
the compaction layer program against, so a field that can contradict itself is
a UI defect waiting to happen rather than a style question.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_operator.harness.types import (
    AskOption,
    AskQuestion,
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


# --- ask: the recommendation is normalised to the top ------------------------


def _ask(labels: list[str], recommended: int | None) -> AskQuestion:
    return AskQuestion(
        id="rollout",
        question="Which rollout?",
        options=[AskOption(label=label) for label in labels],
        recommended=recommended,
    )


def test_a_mid_list_recommendation_is_hoisted_to_the_top():
    """Normalised in the MODEL, so every surface gets it — including the mobile
    wire, which carries no ``recommended`` field and can express the
    recommendation only as position."""
    question = _ask(["a", "b", "c", "d"], 2)
    assert question.recommended == 0
    assert [option.label for option in question.options] == ["c", "a", "b", "d"]


def test_hoisting_preserves_the_relative_order_of_the_rest():
    """A rotation, not a swap: what the model did not recommend is still ranked,
    and a swap would promote whatever sat at index 0 over all of it."""
    question = _ask(["a", "b", "c", "d"], 3)
    assert [option.label for option in question.options] == ["d", "a", "b", "c"]


def test_a_recommendation_already_at_the_top_is_untouched():
    question = _ask(["a", "b", "c"], 0)
    assert question.recommended == 0
    assert [option.label for option in question.options] == ["a", "b", "c"]


def test_no_recommendation_leaves_the_authored_order_alone():
    """Without a recommendation there is nothing to promote, and reordering
    would silently discard the model's own ranking."""
    question = _ask(["a", "b", "c"], None)
    assert question.recommended is None
    assert [option.label for option in question.options] == ["a", "b", "c"]


def test_normalising_an_already_normalised_question_is_a_no_op():
    """Idempotent, because a question is re-validated on every round trip it
    makes (model_validate of a dumped question, a resumed session): a hoist that
    ran twice would rotate the list a second time and endorse a different row."""
    once = _ask(["a", "b", "c"], 1)
    twice = AskQuestion.model_validate(once.model_dump())
    thrice = AskQuestion.model_validate(twice.model_dump())
    assert [option.label for option in twice.options] == ["b", "a", "c"]
    assert [option.label for option in thrice.options] == ["b", "a", "c"]
    assert twice.recommended == 0 and thrice.recommended == 0


def test_an_out_of_range_recommendation_is_still_refused():
    """The bounds check runs BEFORE the hoist: reordering against an index that
    indexes nothing would turn a correctable error into a scrambled list."""
    with pytest.raises(ValidationError) as excinfo:
        _ask(["a", "b"], 5)
    assert "recommended must index options (0..1)" in str(excinfo.value)
