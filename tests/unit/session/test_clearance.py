"""``session.clearance`` — the ask gate's model-facing grammar, pinned pure.

Every routing decision the gate makes that is not a provider call lives in
this module: the prompt it sends, the strict verdict grammar it accepts, the
content digest the honor rule records, and the two decision-point notes the
model reads when an ask is diverted. Each is a pure function, and each is
pinned here against the design's §2.3/§2.5/§2.4 contracts — the grammar
especially, because a parser that accepts too much routes a deliberation as
an answer and a parser that accepts too little enqueues everything.
"""

from __future__ import annotations

from types import SimpleNamespace

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.session.clearance import (
    CLEARANCE_PROMPT,
    REASON_MAX_CHARS,
    build_clearance_prompt,
    clearance_note,
    fingerprint,
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


# --- fingerprint -------------------------------------------------------------


def _digest(*questions: AskQuestion) -> str:
    return fingerprint(list(questions))


class TestFingerprint:
    def test_whitespace_collapses(self) -> None:
        a = _question(question="Deploy   now?\n ")
        b = _question(question="Deploy now?")
        assert _digest(a) == _digest(b)

    def test_option_labels_and_descriptions_are_included(self) -> None:
        base = _question()
        relabelled = _question(options=[("Ship now", "after the freeze"), ("Wait", "until Monday")])
        redescribed = _question(options=[("Ship it", "right away"), ("Wait", "until Monday")])
        assert _digest(base) != _digest(relabelled)
        assert _digest(base) != _digest(redescribed)

    def test_id_recommended_and_multi_are_excluded(self) -> None:
        """The user-facing content is question + options; those three are not it.

        ``recommended=0`` and ``recommended=None`` produce the SAME option
        order (the hoist is the identity for index 0), so the digest must
        agree: a re-raise that merely pins the recommendation it already had
        is the same content and must hit the recorded fingerprint.
        """
        plain = _digest(_question(recommended=None, multi=False))
        assert plain == _digest(_question(id="other", recommended=0, multi=False))
        assert plain == _digest(_question(id="third", recommended=0, multi=True))

    def test_duck_typed_fields_do_not_leak_into_the_digest(self) -> None:
        """Fields outside question+options are not read, whatever the object."""
        stub = SimpleNamespace(
            question="Deploy now?",
            options=[
                SimpleNamespace(label="Ship it", description="after the freeze"),
                SimpleNamespace(label="Wait", description="until Monday"),
            ],
            id="z",
            recommended=1,
            multi=True,
            secret=False,
        )
        assert fingerprint([stub]) == _digest(_question())

    def test_order_is_significant(self) -> None:
        first = _question(options=[("A", "one"), ("B", "two")])
        swapped = _question(options=[("B", "two"), ("A", "one")])
        assert _digest(first) != _digest(swapped)

    def test_stable_across_serialization(self) -> None:
        question = _question()
        reserialized = AskQuestion.model_validate(question.model_dump())
        assert _digest(question) == _digest(reserialized)

    def test_question_order_is_significant(self) -> None:
        a = _question(question="First?")
        b = _question(question="Second?")
        assert fingerprint([a, b]) != fingerprint([b, a])


# --- build_clearance_prompt --------------------------------------------------


class TestPrompt:
    def test_renders_questions_options_and_the_recommendation_marker(self) -> None:
        prompt = build_clearance_prompt([_question(recommended=1)])
        # The hoist moved "Wait" (the recommended option) to the top; the
        # renderer marks position 0 as the recommendation.
        assert "Q1: Deploy now?" in prompt
        assert "1. Wait (recommended) — until Monday" in prompt
        assert "2. Ship it — after the freeze" in prompt

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


# --- clearance_note ----------------------------------------------------------


class TestNotes:
    def test_clear_note_contract(self) -> None:
        note = clearance_note("clear", "obvious")
        assert "No question was put to the user." in note
        assert "NOT asked" in note
        assert "`ask` again" in note  # the re-raise licence
        assert "Check's reason: obvious" in note
        # No user-attribution register anywhere ("the receipt is not consent").
        assert "the user said" not in note
        assert "the user chose" not in note

    def test_resolve_note_contract(self) -> None:
        note = clearance_note("resolve", "unclear")
        assert "No question was put to the user." in note
        assert "ONE `task` subagent" in note
        assert "re-raise" in note or "`ask` again" in note
        assert "Check's reason: unclear" in note

    def test_missing_reason_omits_the_suffix(self) -> None:
        assert "Check's reason" not in clearance_note("clear", "")

    def test_no_note_for_raise(self) -> None:
        """A raise enqueues and the model gets today's receipt — no note."""
        # ``clearance_note`` is only ever called for clear/resolve by the
        # callable; the NOTE-level pin is that nothing in this module builds
        # a user-attribution sentence even as a fallback.
        for verdict in ("clear", "resolve"):
            note = clearance_note(verdict, "r")
            assert "The user was" not in note.replace("The user was NOT asked", "")
