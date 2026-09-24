"""The goal-continuation AND goal-mode-loop prompts are chrome, on every surface.

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
from local_operator.session.goal_loop import (
    LOOP_GOAL_HEAD,
    LOOP_GOAL_PROMPT,
    LOOP_GOAL_TAIL,
    LOOP_PROMPT,
    is_loop_goal_instruction,
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


# ---------------------------------------------------------------------------
# The goal-mode LOOP's working turn: the same family problem, one producer over.
# ---------------------------------------------------------------------------


def test_the_loop_goal_turn_is_recognised_for_every_goal_shape():
    """Agent review round 3: ``LOOP_GOAL_PROMPT`` had NEITHER leg of the decision.

    It is not in ``harness_chrome_prompts()`` (it interpolates the goal) and it
    matched no recogniser, so every surface that folds by the text match painted
    the loop's own words as the operator's. Measured on a real session through the
    desktop route before this: the persisted row read
    ``stamp=no, chrome-recognised=False``. The structural stamp now rides the same
    row (``_prompt_loop_turn`` on the TUI, the loop callback in ``serving.py``),
    and this is the text half, for the builds and folds that predate the stamp.
    """
    for goal in (
        "",
        "Verify the fixture goal",
        "Line one\n\nLine two\n- and a bullet",
        "x" * MAX_GOAL_CHARS,
    ):
        prompt = LOOP_GOAL_PROMPT.format(goal=goal)
        assert is_loop_goal_instruction(prompt), goal[:40]
        assert is_harness_chrome(prompt), goal[:40]


def test_the_loop_goal_edges_are_the_only_thing_matched():
    """Stated as a property so an edit that loosens an edge fails here."""
    prompt = LOOP_GOAL_PROMPT.format(goal="Ship it")
    assert prompt.startswith(LOOP_GOAL_HEAD)
    assert prompt.endswith(LOOP_GOAL_TAIL)
    assert LOOP_GOAL_PROMPT == LOOP_GOAL_HEAD + "{goal}" + LOOP_GOAL_TAIL


def test_a_message_that_merely_opens_with_the_loop_head_is_the_users_own():
    """A prefix match would eat a real message; the tail check is what stops it."""
    assert not is_loop_goal_instruction(LOOP_GOAL_HEAD + "I typed this myself")
    assert not is_loop_goal_instruction(LOOP_GOAL_HEAD)
    assert not is_harness_chrome(LOOP_GOAL_HEAD + "and then I kept typing")


def test_quoting_the_loop_template_in_another_sentence_is_not_chrome():
    """The recogniser matches the SHAPE, never a substring of the goal."""
    assert not is_loop_goal_instruction(
        "Reminder: the loop sends " + LOOP_GOAL_TAIL.strip() + " every iteration."
    )
    assert not is_harness_chrome(f"why does it say {LOOP_GOAL_HEAD!r} at the top?")


def test_the_loop_goal_prompt_is_not_in_the_fixed_tuple():
    """It embeds the goal, so listing it there would make the tuple a lie."""
    assert LOOP_GOAL_PROMPT not in harness_chrome_prompts()


def test_the_count_loops_prompt_keeps_its_exact_match_leg():
    """The sibling that never interpolates anything stays a fixed member."""
    assert LOOP_PROMPT in harness_chrome_prompts()
    assert is_harness_chrome(LOOP_PROMPT)
