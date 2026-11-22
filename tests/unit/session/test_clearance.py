"""Tests for the legacy clearance prompt and verdict grammar."""

from __future__ import annotations

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.session.clearance import (
    CLEARANCE_PROMPT,
    REASON_MAX_CHARS,
    build_clearance_prompt,
    parse_reason,
    parse_verdict,
)


def _question(
    question: str = "Deploy now?",
    options: list[tuple[str, str]] | None = None,
    **overrides: object,
) -> AskQuestion:
    pairs = options or [("Ship it", "after the freeze"), ("Wait", "until Monday")]
    return AskQuestion(
        id=str(overrides.pop("id", "q0")),
        question=question,
        options=[AskOption(label=label, description=desc) for label, desc in pairs],
        **overrides,  # type: ignore[arg-type]
    )


# --- parse_verdict -----------------------------------------------------------


class TestParseVerdict:
    def test_exact_two_line_answer(self) -> None:
        assert parse_verdict("VERDICT: clear\nREASON: obvious") == "clear"
        assert parse_verdict("VERDICT: resolve\nREASON: not mine") == "resolve"
        assert parse_verdict("VERDICT: raise\nREASON: their call") == "raise"

    def test_deliberation_last_match_wins(self) -> None:
        """A reasoning model thinks in the open; the answer is what it ENDS on."""
        text = "It could be clear — but the choice is theirs.\nVERDICT: clear\nVERDICT: raise"
        assert parse_verdict(text) == "raise"

    def test_decoration_is_tolerated(self) -> None:
        """The design's own §4 example: ``**VERDICT:** clear`` must parse."""
        assert parse_verdict("**VERDICT:** clear") == "clear"
        assert parse_verdict("> VERDICT: clear") == "clear"
        assert parse_verdict("# VERDICT: raise") == "raise"
        assert parse_verdict("VERDICT:**resolve**") == "resolve"

    def test_case_insensitive(self) -> None:
        assert parse_verdict("verdict: CLEAR") == "clear"
        assert parse_verdict("Verdict: Resolve") == "resolve"

    def test_no_verdict_is_none(self) -> None:
        assert parse_verdict("I think you should ship it.") is None
        assert parse_verdict("") is None
        assert parse_verdict(None) is None

    def test_truncated_or_placeholder_is_none(self) -> None:
        assert parse_verdict("VERDICT:") is None
        assert parse_verdict("VERDICT: maybe") is None
        # The template's own placeholder is not an answer.
        assert parse_verdict("VERDICT: clear|resolve|raise") is None

    def test_a_verdict_word_mid_sentence_is_not_a_match(self) -> None:
        """Full-line only: prose that mentions the grammar is deliberation."""
        assert parse_verdict("The verdict: clear was my first instinct") is None


# --- parse_reason ------------------------------------------------------------


class TestParseReason:
    def test_reason_line(self) -> None:
        assert parse_reason("VERDICT: clear\nREASON: the log settles it") == "the log settles it"

    def test_missing_reason_is_empty_not_an_error(self) -> None:
        assert parse_reason("VERDICT: clear") == ""
        assert parse_reason("") == ""

    def test_missing_reason_keeps_the_verdict(self) -> None:
        """The verdict is the only routing input (design §2.3)."""
        assert parse_verdict("VERDICT: clear") == "clear"

    def test_reason_is_truncated_to_the_wire_bound(self) -> None:
        long = "x" * (REASON_MAX_CHARS + 40)
        assert len(parse_reason(f"REASON: {long}")) == REASON_MAX_CHARS

    def test_decoration_is_stripped(self) -> None:
        assert parse_reason("**REASON:** a thing**") == "a thing"

    def test_last_reason_wins(self) -> None:
        assert parse_reason("REASON: first\nREASON: second") == "second"


# --- build_clearance_prompt --------------------------------------------------


class TestPrompt:
    def test_renders_questions_options_and_the_recommendation_marker(self) -> None:
        prompt = build_clearance_prompt([_question(recommended=1)])
        # The hoist moved "Wait" (the recommended option) to the top; the
        # renderer marks position 0 as the recommendation.
        assert "Q1: Deploy now?" in prompt
        assert "1. Wait (recommended) — until Monday" in prompt
        assert "2. Ship it — after the freeze" in prompt

    def test_a_multi_select_set_marks_every_recommended_row(self) -> None:
        prompt = build_clearance_prompt(
            [
                _question(
                    options=[("A", "one"), ("B", "two"), ("C", "three")],
                    multi=True,
                    recommended=[2, 0],
                )
            ]
        )
        # Hoisted in the listed order: C, A lead; B is not recommended.
        assert "1. C (recommended) — three" in prompt
        assert "2. A (recommended) — one" in prompt
        assert "3. B — two" in prompt
        assert prompt.count("(recommended)") == 2

    def test_multi_select_marker(self) -> None:
        assert "[multi-select]" in build_clearance_prompt([_question(multi=True)])
        assert "[multi-select]" not in build_clearance_prompt([_question(multi=False)])

    def test_no_recommendation_renders_none(self) -> None:
        prompt = build_clearance_prompt([_question(recommended=None)])
        assert "(recommended)" not in prompt

    def test_the_genuine_user_list_rides_verbatim(self) -> None:
        """Requirement 5: this list is where "never miss a decision" lives.

        Checked against the whitespace-normalised prompt: the template wraps
        mid-phrase ("a preference,\n  name, or roster…"), so the sentence is
        the unit, not the line.
        """
        flat = " ".join(CLEARANCE_PROMPT.split())
        built = " ".join(build_clearance_prompt([_question()]).split())
        for bullet in (
            "critical go/no-go decisions",
            "opinionated architectural choices",
            "missing access or credentials",
            "a preference, name, or roster only they can state",
            "A destructive or irreversible action is NEVER cleared.",
            "If ANY question is the user's, raise.",
        ):
            assert bullet in flat
            assert bullet in built

    def test_template_has_one_format_site(self) -> None:
        """The builder is how every caller renders it; no second ``.format``."""
        assert "{ask_block}" in CLEARANCE_PROMPT
        assert "{" not in CLEARANCE_PROMPT.replace("{ask_block}", "")

    def test_multiple_questions_numbered_in_order(self) -> None:
        prompt = build_clearance_prompt(
            [_question(question="First?"), _question(question="Second?")]
        )
        assert prompt.index("Q1: First?") < prompt.index("Q2: Second?")
