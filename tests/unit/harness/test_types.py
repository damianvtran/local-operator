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


def test_a_negative_recommendation_is_refused_rather_than_indexing_from_the_end():
    """``-1`` is a valid Python index and would silently promote the LAST
    option — endorsing something the model did not choose, which is the exact
    failure the bounds check exists to prevent. The message states the usable
    range so the model can correct it."""
    with pytest.raises(ValidationError) as excinfo:
        _ask(["a", "b", "c"], -1)
    assert "recommended must index options (0..2)" in str(excinfo.value)


def test_a_refused_recommendation_never_reorders_the_options():
    """The refusal path must leave the list alone: a question that came back
    both rejected AND scrambled would hand the model a correction to make
    against options it no longer recognises."""
    options = [AskOption(label=label) for label in ("a", "b", "c")]
    with pytest.raises(ValidationError):
        AskQuestion(id="rollout", question="Which rollout?", options=options, recommended=7)
    assert [option.label for option in options] == ["a", "b", "c"]


def test_the_hoist_survives_a_json_round_trip():
    """A question is re-parsed from JSON on transcript replay and on the way to
    a subagent, not only from a dumped dict — the order the user was shown has
    to be the order that comes back."""
    once = _ask(["a", "b", "c", "d"], 2)
    replayed = AskQuestion.model_validate_json(once.model_dump_json())
    assert [option.label for option in replayed.options] == ["c", "a", "b", "d"]
    assert replayed.recommended == 0


def test_hoisting_moves_the_whole_option_not_just_its_label():
    """Each option carries the consequence line the user decides on, so a hoist
    that moved labels alone would pair the promoted option with somebody else's
    description."""
    question = AskQuestion(
        id="rollout",
        question="Which rollout?",
        options=[
            AskOption(label="a", description="cheapest"),
            AskOption(label="b", description="keeps history"),
            AskOption(label="c", description="safest"),
        ],
        recommended=2,
    )
    assert (question.options[0].label, question.options[0].description) == ("c", "safest")
    assert [option.description for option in question.options] == [
        "safest",
        "cheapest",
        "keeps history",
    ]


def test_a_secret_question_still_refuses_a_recommendation_after_the_hoist_landed():
    """The secret branch returns before the hoist, so a credential paste can
    never be reordered or preselected into endorsing a value nobody can see."""
    with pytest.raises(ValidationError) as excinfo:
        AskQuestion(id="GITHUB_TOKEN", question="Paste it.", options=[], secret=True, recommended=0)
    assert "no options to recommend" in str(excinfo.value)
