"""The goal-continuation prompt is recognised as harness chrome, on every surface.

The one decision these tests pin lives in ``harness.rows.is_harness_chrome``, and
what this file adds to it is the second leg: the goal judge's continuation prompt
interpolates the standing goal, so it has no fixed string to put in
``harness_chrome_prompts()`` and a consumer that compared against the tuple would
paint it as the operator's own words. The sibling case is
``is_connectivity_continuation_instruction``, which exists for the same reason and
is tested the same way.

The discriminating half is the negative one: a recogniser that matched by
substring over the goal text would silently swallow a real user message that
happened to quote the goal, which is a far worse failure than painting one chrome
row.
"""

from __future__ import annotations

from local_operator.harness.rows import harness_chrome_prompts, is_harness_chrome
from local_operator.session.goal import MAX_GOAL_CHARS
from local_operator.session.goal_judge import (
    GOAL_CONTINUATION_HEAD,
    GOAL_CONTINUATION_PROMPT,
    GOAL_CONTINUATION_TAIL,
    goal_continuation_prompt,
    is_goal_continuation_instruction,
)


def test_the_produced_prompt_is_recognised_for_every_goal_shape():
    """The family is "any goal in the template", not one enumerated string."""
    for goal in (
        "",
        "Do the thing",
        "Line one\n\nLine two\n- and a bullet",
        "x" * MAX_GOAL_CHARS,
    ):
        prompt = goal_continuation_prompt(goal)
        assert is_goal_continuation_instruction(prompt), goal[:40]


def test_the_producer_clips_an_oversized_goal():
    """An oversized goal would ride the attach frame as a dropped line."""
    prompt = goal_continuation_prompt("y" * (MAX_GOAL_CHARS + 500))
    embedded = prompt[len(GOAL_CONTINUATION_HEAD) : -len(GOAL_CONTINUATION_TAIL)]
    assert embedded == "y" * MAX_GOAL_CHARS


def test_surrounding_whitespace_does_not_defeat_the_recogniser():
    """Hosts hand this decision text in with and without a trailing newline."""
    prompt = goal_continuation_prompt("Ship it")
    assert is_goal_continuation_instruction(f"  {prompt}\n\n")
    assert is_harness_chrome(f"{prompt}\n")


def test_a_message_that_merely_opens_with_the_head_is_the_users_own():
    """A prefix match would eat a real message; the tail check is what stops it."""
    assert not is_goal_continuation_instruction(
        GOAL_CONTINUATION_HEAD + "I typed this myself, actually"
    )
    assert not is_goal_continuation_instruction(GOAL_CONTINUATION_HEAD)
    assert not is_harness_chrome(GOAL_CONTINUATION_HEAD + "and then I kept typing")


def test_quoting_the_template_in_another_sentence_is_not_chrome():
    """The recogniser matches the SHAPE, never a substring of the goal."""
    assert not is_goal_continuation_instruction(
        "Reminder: the app sends " + GOAL_CONTINUATION_TAIL.strip() + " after each turn."
    )
    assert not is_goal_continuation_instruction(
        f"why does it say {GOAL_CONTINUATION_HEAD!r} at the top of my transcript?"
    )


def test_the_fixed_edges_are_the_only_thing_matched():
    """Stated as a property so a future edit that loosens one edge fails here."""
    prompt = goal_continuation_prompt("Ship it")
    assert prompt.startswith(GOAL_CONTINUATION_HEAD)
    assert prompt.endswith(GOAL_CONTINUATION_TAIL)
    assert GOAL_CONTINUATION_PROMPT == GOAL_CONTINUATION_HEAD + "{goal}" + GOAL_CONTINUATION_TAIL


def test_the_shared_decision_answers_true_and_still_false_for_ordinary_text():
    """``is_harness_chrome`` is the ONE decision the surfaces call."""
    assert is_harness_chrome(goal_continuation_prompt("Ship the release"))
    for ordinary in (
        "hello",
        "goal: ship the release",
        "continue",
        "",
    ):
        assert not is_harness_chrome(ordinary), ordinary


def test_the_goal_prompt_is_not_in_the_fixed_tuple():
    """It embeds the goal, so listing it there would make the tuple a lie."""
    assert GOAL_CONTINUATION_PROMPT not in harness_chrome_prompts()
    assert not any("{goal}" in entry for entry in harness_chrome_prompts())
