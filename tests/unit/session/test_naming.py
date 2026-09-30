"""Focused ``parse_title`` / generate_title / acceptance-cascade coverage.

The TUI suite owns scheduling (when the call fires, what a failure costs the
latch). This file owns the parser: tagged Claude replies, untagged Grok
replies, leaked thinking, JSON wrappers, and the generate_title / generate_retitle
failure modes that used to live only as comments on the TUI tests. It also owns
the Tier 0-3 acceptance cascade: the structural gate's corpus (the seventeen
wrapped forms from the operator's report), the classifier matrix, the bounded
retry, and the Tier 3 opener fallback.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from local_operator.session import naming


def test_tagged_title_still_works_including_surrounding_prose() -> None:
    assert (
        naming.parse_title("Sure.\n<title>the login redirect loop</title>\nDone.")
        == "The login redirect loop"
    )


def test_self_closing_title_is_the_no_topic_sentinel() -> None:
    assert naming.parse_title("<title/>") is None
    assert naming.parse_title("no topic here\n<title />") is None


def test_untagged_short_title_is_accepted_and_sentence_cased() -> None:
    """Grok 4.6 and other non-Anthropic models emit a bare 3–7 word title.

    Rejecting those replies is how those sessions silently kept the opener
    excerpt on the band forever.
    """
    assert naming.parse_title("grok session title stuck on opener") == (
        "Grok session title stuck on opener"
    )


def test_untagged_long_essay_is_rejected() -> None:
    essay = (
        "This conversation is about debugging why grok sessions keep the "
        "opener excerpt as the title instead of generating a real one from "
        "the model's reply, which is a long explanation rather than a title"
    )
    assert naming.parse_title(essay) is None


def test_untagged_too_many_words_is_rejected() -> None:
    words = " ".join(f"word{i}" for i in range(naming.MAX_TITLE_WORDS + 1))
    assert naming.parse_title(words) is None


def test_thinking_wrapped_title_prefers_the_visible_tag() -> None:
    raw = (
        "<think>I should name this <title>wrong title</title> maybe</think>\n"
        "<title>right title</title>"
    )
    assert naming.parse_title(raw) == "Right title"


def test_fenced_reasoning_then_a_visible_title() -> None:
    raw = "```reasoning\nplanning the name\n```\n<title>visible title</title>"
    assert naming.parse_title(raw) == "Visible title"


def test_json_object_title_is_unwrapped() -> None:
    assert naming.parse_title('{"title": "the login redirect loop"}') == ("The login redirect loop")


def test_fenced_json_title_is_unwrapped() -> None:
    raw = '```json\n{"title": "the login redirect loop"}\n```'
    assert naming.parse_title(raw) == "The login redirect loop"


def test_empty_whitespace_and_quotes_only_are_none() -> None:
    assert naming.parse_title("") is None
    assert naming.parse_title("   \n\t  ") is None
    assert naming.parse_title('""') is None
    assert naming.parse_title("<title>   </title>") is None
    assert naming.parse_title('<title>""</title>') is None


def test_thinking_preamble_without_a_title_is_rejected() -> None:
    assert naming.parse_title("Thinking process:\nI will now name this") is None
    assert naming.parse_title("Here's my reasoning:\nthis is about login") is None


def test_unclosed_title_tag_is_not_kept_as_markup() -> None:
    """Truncated ``<title>…`` must never store the tag characters (review M1).

    The remainder can still name the session if it already looks like a
    title; the literal ``<title>`` must not survive into the stored name.
    """
    assert naming.parse_title("<title>the login redirect loop") == ("The login redirect loop")
    assert "<title>" not in (naming.parse_title("<title>the login redirect loop") or "")
    assert naming.parse_title("<title>") is None


def test_unclosed_think_without_a_visible_title_is_rejected() -> None:
    assert naming.parse_title("<think>planning a name for the login loop") is None
    assert naming.parse_title("<thinking>still reasoning about this") is None
    assert naming.parse_title("<reasoning>drafting the title") is None
    # A closed tag drafted *inside* the unclosed envelope is still thinking.
    assert naming.parse_title("<think>planning\n<title>wrong title</title>") is None


def test_unclosed_thinking_fence_is_rejected() -> None:
    assert naming.parse_title("```thinking\nplanning the name") is None
    assert naming.parse_title("```reasoning\nstill inside the envelope") is None


def test_closed_think_then_untagged_short_title_still_works() -> None:
    raw = "<think>I should name this after the login bug</think>\nthe login redirect loop"
    assert naming.parse_title(raw) == "The login redirect loop"


def test_fenced_json_unwraps_regardless_of_language_tag() -> None:
    """A fence language is not a title (review m2). ``python`` used to win."""
    raw = '```python\n{"title": "The login redirect loop"}\n```'
    assert naming.parse_title(raw) == "The login redirect loop"
    assert naming.parse_title(raw) != "Python"


def test_chatty_preamble_then_a_later_short_title() -> None:
    """First-line-only parse kept the preamble (review m1). Prefer the later line."""
    raw = "Sure, I'll name this.\n\nThe login redirect loop"
    assert naming.parse_title(raw) == "The login redirect loop"
    assert naming.parse_title(raw) != "Sure, I'll name this"


@pytest.mark.asyncio
async def test_generate_title_returns_none_for_the_sentinel() -> None:
    async def answer(system: str, prompt: str) -> str:
        return "<title/>"

    assert await naming.generate_title("fix the login redirect loop", answer) is None


@pytest.mark.asyncio
async def test_generate_title_returns_none_on_provider_failure() -> None:
    async def boom(system: str, prompt: str) -> str:
        raise RuntimeError("429 rate limited")

    assert await naming.generate_title("fix the login redirect loop", boom) is None


@pytest.mark.asyncio
async def test_ask_for_title_logs_the_swallowed_provider_failure(caplog) -> None:
    """The module docstring promises "its failures go to the log file" — before
    the fix the failure was swallowed with NO log line at all, and "sessions
    don't get a name" was diagnosable only by analytics archaeology. The line
    names the exception type and message (the provider's own words carry the
    diagnosis: "authentication failed (HTTP 401)") and never the prompt or any
    user content."""

    async def boom(system: str, prompt: str) -> str:
        raise RuntimeError("authentication failed (HTTP 401): invalid key")

    with caplog.at_level(logging.WARNING, logger="local_operator.session.naming"):
        assert (
            await naming._ask_for_title("system text", "prompt text not in the log", boom, 30)
            is naming.CALL_FAILED
        )
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "conversation naming call failed" in message
    assert "RuntimeError" in message
    assert "authentication failed (HTTP 401)" in message
    assert "prompt text" not in message


@pytest.mark.asyncio
async def test_ask_for_title_logs_a_timeout_without_the_prompt(caplog) -> None:
    """A stall spends the whole budget and used to vanish the same way."""

    async def never(system: str, prompt: str) -> str:
        await asyncio.sleep(60)
        return "<title>x</title>"

    with caplog.at_level(logging.WARNING, logger="local_operator.session.naming"):
        assert (
            await naming._ask_for_title("system", "prompt never logged", never, 0.05)
            is naming.CALL_FAILED
        )
    assert len(caplog.records) == 1
    assert "timed out" in caplog.records[0].getMessage()
    assert "prompt never logged" not in caplog.records[0].getMessage()


@pytest.mark.asyncio
async def test_ask_for_title_is_silent_on_cancellation(caplog) -> None:
    """Cancellation is a routine shutdown of the detached naming task, not a
    failure — a WARNING per session at exit would be noise, so the cancel arm
    stays out of the log entirely (return value still CALL_CANCELLED)."""

    async def cancelled(system: str, prompt: str) -> str:
        raise asyncio.CancelledError()

    with caplog.at_level(logging.WARNING, logger="local_operator.session.naming"):
        assert (
            await naming._ask_for_title("system", "prompt", cancelled, 30) is naming.CALL_CANCELLED
        )
    assert caplog.records == []


@pytest.mark.asyncio
async def test_generate_title_accepts_an_untagged_short_reply() -> None:
    async def answer(system: str, prompt: str) -> str:
        return "the login redirect loop"

    assert (
        await naming.generate_title("fix the login redirect loop", answer)
        == "The login redirect loop"
    )


@pytest.mark.asyncio
async def test_generate_retitle_sentinel_and_restatement_and_failure_are_none() -> None:
    async def sentinel(system: str, prompt: str) -> str:
        return "<title/>"

    async def restatement(system: str, prompt: str) -> str:
        return "<title>fix the LOGIN flow</title>"

    async def boom(system: str, prompt: str) -> str:
        raise RuntimeError("429")

    assert (
        await naming.generate_retitle("Fix the login flow", "and the logout too", sentinel) is None
    )
    assert (
        await naming.generate_retitle("Fix the login flow", "and the logout too", restatement)
        is None
    )
    assert await naming.generate_retitle("Fix the login flow", "rewrite the importer", boom) is None


# ---------------------------------------------------------------------------
# The acceptance cascade (Tier 0-3)
# ---------------------------------------------------------------------------

#: The five wrapped forms the operator's report carried, verbatim (plus the
#: stale-pr twin from the same era). Every one of them reached storage through
#: ``parse_title`` before the gate existed; all of them must be refused now.
WRAPPED_FROM_THE_REPORT = [
    "<Stale PR recovery task>",
    "<stale-pr recovery local-operator-ui>",
    "<\u56d7>Composer ArrowUp history recall UI issues",
    "<\u56e7>Assess releasing PR 192 or close",
    "<|\uff5cDSML\uff5c|ai_title>Minerva merge requirements before MR merge</ai_title>",
]

#: Twelve more observed forms, kept as regression fixtures.
WRAPPED_CORPUS = [
    "<Cabin>",
    "<\u70ed\u70b9\u805a\u7126>Radient suggestion ordering and dedupe",
    "<\u5212\u91cd\u70b9>revamp Agent hub teams UX layout",
    "< Hearing loss resolution errors in tui",
    "<\u56d7>Local operator subagent result pane takeover",
    "< Hearing handoff documentation review request",
    "< Bavarian International School>",
    "< Credential tool could not be reached",
    "<notes protocol scratch markdown files>",
    "<L title>Local Operator UI install instructions",
    "<\u56e7>GOAL-CONTINUATION messages visible in transcripts</\u56e7>",
    "<\uff5c\uff5cDSML\uff5c\uff5c calls>",
]


class _Replies:
    """A ``complete_once`` fake answering from a fixed list, in order.

    Raises when the cascade asks for a reply the list does not hold, so every
    test using it is also an assertion that the attempt stayed inside its
    bound of two naming samples.
    """

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        if not self.replies:
            raise AssertionError("the cascade spent more naming samples than its bound allows")
        return self.replies.pop(0)


class _FitCheck:
    """A fit-check fake answering one verdict per call, in order.

    A verdict list shorter than the calls made repeats its last entry; an
    exhausted list answers ``None`` ("no verdict"). ``states`` records what
    each call was shown.
    """

    def __init__(self, *verdicts: str | None) -> None:
        self.verdicts = list(verdicts)
        self.states: list[str] = []

    async def __call__(self, state: str) -> str | None:
        self.states.append(state)
        if not self.verdicts:
            return None
        if len(self.states) <= len(self.verdicts):
            return self.verdicts[len(self.states) - 1]
        return self.verdicts[-1]


def _theme_rows() -> list[SimpleNamespace]:
    """The minimal ``(role, text)`` history the re-title/refresh samplers read."""
    return [
        SimpleNamespace(role="user", text="fix the login redirect loop"),
        SimpleNamespace(role="assistant", text="done"),
    ]


# -- Tier 0: the structural gate -------------------------------------------


def test_every_wrapped_form_from_the_report_fails_the_structural_gate() -> None:
    for wrapped in WRAPPED_FROM_THE_REPORT:
        assert naming.validate_generated_title(wrapped, "opener text") is None, wrapped


def test_the_wider_wrapped_corpus_fails_the_gate_too() -> None:
    for wrapped in WRAPPED_CORPUS:
        assert naming.validate_generated_title(wrapped, "opener text") is None, wrapped


def test_the_gate_keeps_plain_titles_unchanged() -> None:
    opener = "please fix the redirect loop when logging in"
    for title in (
        "Fix the login redirect loop",
        "Bulk export columns",
        "Revamp Agent hub teams UX layout",
        "The login redirect loop",
    ):
        assert naming.validate_generated_title(title, opener) == title, title


def test_the_gate_refuses_damage_control_and_repetition_shapes() -> None:
    assert naming.validate_generated_title("Fix\x00the loop", "opener") is None
    assert naming.validate_generated_title("\u200b\u200b\u200b", "opener") is None
    assert naming.validate_generated_title("Fix \ufffd the loop", "opener") is None
    assert naming.validate_generated_title("Fix \uff1cthe loop\uff1e", "opener") is None
    assert naming.validate_generated_title("Loop loop loop forever", "opener") is None
    assert naming.validate_generated_title("aaaaaaaaaaaaaa loop", "opener") is None


def test_the_gate_refuses_a_body_equal_to_the_opener_label() -> None:
    opener = "fix the login redirect loop"
    label = naming.fallback_from_opener(opener)
    assert naming.validate_generated_title(label, opener) is None
    # The Tier 3 derivation is exempt: it IS that label, deliberately.
    assert naming.validate_generated_title(label, opener, allow_opener_label=True) == label
    # ...and a body a word different from the label is not an echo.
    assert naming.validate_generated_title("Login redirect loop fix", opener) == (
        "Login redirect loop fix"
    )


# -- the cascade: retry, classifier matrix, bounds -------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", WRAPPED_FROM_THE_REPORT)
async def test_the_corrective_resample_recovers_every_reported_form(wrapped: str) -> None:
    replies = _Replies(wrapped, "Recovered release triage")
    acceptance = await naming.generate_title_acceptance("recover the release notes", replies)
    assert acceptance.title == "Recovered release triage", wrapped
    assert acceptance.tier == naming.TIER_GENERATED
    assert acceptance.heal is False
    assert len(replies.calls) == 2
    assert naming.TITLE_CORRECTIVE_ADDENDUM in replies.calls[1][0]


@pytest.mark.asyncio
async def test_a_reachable_classifier_accepts_a_fitting_sample_without_a_retry() -> None:
    replies = _Replies("The login redirect loop")
    fit = _FitCheck(naming.TITLE_FITS)
    acceptance = await naming.generate_title_acceptance(
        "fix the login redirect loop bug", replies, fit_check=fit
    )
    assert acceptance.title == "The login redirect loop"
    assert acceptance.tier == naming.TIER_GENERATED
    assert len(replies.calls) == 1
    assert len(fit.states) == 1
    assert "The login redirect loop" in fit.states[0]


@pytest.mark.asyncio
async def test_an_unfit_sample_is_corrected_and_the_second_check_decides() -> None:
    replies = _Replies("First candidate", "Second candidate")
    fit = _FitCheck(naming.TITLE_DOESNT_FIT, naming.TITLE_FITS)
    acceptance = await naming.generate_title_acceptance(
        "recover the release notes", replies, fit_check=fit
    )
    assert acceptance.title == "Second candidate"
    assert len(replies.calls) == 2
    assert len(fit.states) == 2
    assert naming.TITLE_CORRECTIVE_ADDENDUM in replies.calls[1][0]


@pytest.mark.asyncio
async def test_an_unfit_retry_ends_in_the_tier_three_fallback() -> None:
    opener = "recover the release notes"
    replies = _Replies("First candidate", "Second candidate")
    fit = _FitCheck(naming.TITLE_DOESNT_FIT, naming.TITLE_DOESNT_FIT)
    acceptance = await naming.generate_title_acceptance(opener, replies, fit_check=fit)
    assert acceptance.title == naming.fallback_from_opener(opener)
    assert acceptance.tier == naming.TIER_FALLBACK
    assert acceptance.heal is True
    assert len(replies.calls) == 2
    assert len(fit.states) == 2, "the classifier bound is two calls per attempt"


@pytest.mark.asyncio
async def test_a_second_fit_check_that_cannot_answer_fails_open() -> None:
    replies = _Replies("First candidate", "Second candidate")
    fit = _FitCheck(naming.TITLE_DOESNT_FIT, None)
    acceptance = await naming.generate_title_acceptance(
        "recover the release notes", replies, fit_check=fit
    )
    assert acceptance.title == "Second candidate"
    assert acceptance.tier == naming.TIER_GENERATED


@pytest.mark.asyncio
async def test_an_absent_classifier_hedges_with_a_plain_second_sample() -> None:
    replies = _Replies("First candidate", "Second candidate")
    acceptance = await naming.generate_title_acceptance("recover the release notes", replies)
    assert acceptance.title == "Second candidate"
    assert acceptance.tier == naming.TIER_GENERATED
    assert len(replies.calls) == 2
    # The hedge is the PLAIN prompt, not the corrective addendum.
    assert replies.calls[1][0] == replies.calls[0][0]
    assert naming.TITLE_CORRECTIVE_ADDENDUM not in replies.calls[1][0]


@pytest.mark.asyncio
async def test_cant_tell_hedges_like_an_absent_classifier() -> None:
    replies = _Replies("First candidate", "Second candidate")
    fit = _FitCheck(naming.TITLE_CANT_TELL)
    acceptance = await naming.generate_title_acceptance(
        "recover the release notes", replies, fit_check=fit
    )
    assert acceptance.title == "Second candidate"
    assert len(fit.states) == 1
    assert len(replies.calls) == 2


@pytest.mark.asyncio
async def test_an_erroring_classifier_hedges_instead_of_rejecting() -> None:
    async def broken(state: str) -> str | None:
        raise RuntimeError("vendor down")

    replies = _Replies("First candidate", "Second candidate")
    acceptance = await naming.generate_title_acceptance(
        "recover the release notes", replies, fit_check=broken
    )
    assert acceptance.title == "Second candidate"


@pytest.mark.asyncio
async def test_a_dirty_hedge_keeps_the_first_accepted_sample() -> None:
    replies = _Replies("First candidate", "<wrapped hedge>")
    acceptance = await naming.generate_title_acceptance("recover the release notes", replies)
    assert acceptance.title == "First candidate"
    assert len(replies.calls) == 2


@pytest.mark.asyncio
async def test_both_samples_dirty_ends_in_a_bracket_free_opener_fallback() -> None:
    opener = "recover the release notes"
    replies = _Replies("<first wrapped>", "<second wrapped>")
    acceptance = await naming.generate_title_acceptance(opener, replies)
    assert acceptance.title == naming.fallback_from_opener(opener)
    assert acceptance.tier == naming.TIER_FALLBACK
    assert acceptance.heal is True
    assert "<" not in acceptance.title and ">" not in acceptance.title
    assert "\uff5c" not in acceptance.title
    assert naming.validate_generated_title(acceptance.title, opener, allow_opener_label=True)


@pytest.mark.asyncio
async def test_an_unusable_opener_ends_with_nothing_and_the_heal_armed() -> None:
    replies = _Replies("<one>", "<two>")
    acceptance = await naming.generate_title_acceptance("<\u56d7>", replies)
    assert acceptance.title == ""
    assert acceptance.tier == naming.TIER_FALLBACK
    assert acceptance.heal is True


@pytest.mark.asyncio
async def test_a_low_signal_opener_spends_nothing_and_arms_nothing() -> None:
    replies = _Replies()  # any call raises AssertionError
    acceptance = await naming.generate_title_acceptance("thanks!", replies)
    assert acceptance.title == ""
    assert acceptance.tier == naming.TIER_NONE
    assert acceptance.heal is False
    assert replies.calls == []


@pytest.mark.asyncio
async def test_a_sentinel_answer_is_not_corrected() -> None:
    replies = _Replies("<title/>")
    acceptance = await naming.generate_title_acceptance("recover the release notes", replies)
    assert acceptance.title == ""
    assert acceptance.heal is False
    assert len(replies.calls) == 1


@pytest.mark.asyncio
async def test_a_failed_call_stays_a_failure_and_spends_nothing_more() -> None:
    async def boom(system: str, prompt: str) -> str:
        raise RuntimeError("429 rate limited")

    acceptance = await naming.generate_title_acceptance("recover the release notes", boom)
    assert acceptance.title == ""
    assert acceptance.heal is False


@pytest.mark.asyncio
async def test_generate_title_returns_the_tier_three_fallback_when_exhausted() -> None:
    opener = "recover the release notes"
    replies = _Replies("<first wrapped>", "<second wrapped>")
    assert await naming.generate_title(opener, replies) == naming.fallback_from_opener(opener)


@pytest.mark.asyncio
async def test_the_re_title_never_falls_back_and_still_gets_the_retry() -> None:
    replies = _Replies("<wrapped one>", "<wrapped two>")
    assert (
        await naming.generate_retitle(
            "Standing title", "and now the billing importer", replies, turns=[]
        )
        is None
    )
    assert len(replies.calls) == 2
    replies2 = _Replies("<wrapped one>", "Billing importer rewrite")
    assert (
        await naming.generate_retitle(
            "Standing title", "and now the billing importer", replies2, turns=[]
        )
        == "Billing importer rewrite"
    )


@pytest.mark.asyncio
async def test_a_wrapped_refresh_reply_is_refused_without_changing_the_title() -> None:
    replies = _Replies("<wrapped refresh>")
    result = await naming.refresh_title(
        "Standing title", replies, turns=_theme_rows(), newest="and the signup flow"
    )
    assert result.outcome == naming.TITLE_UNCHANGED
    assert result.title == ""
    assert len(replies.calls) == 1, "the on-demand path takes Tier 0 only"


@pytest.mark.asyncio
async def test_a_wrapped_refresh_on_an_unnamed_session_reports_nothing_yet() -> None:
    replies = _Replies("<wrapped refresh>")
    result = await naming.refresh_title(
        "", replies, turns=_theme_rows(), newest="and the signup flow"
    )
    assert result.outcome == naming.TITLE_NOTHING_YET


@pytest.mark.asyncio
async def test_a_clean_refresh_still_replaces_the_title() -> None:
    replies = _Replies("Fresh clean title")
    result = await naming.refresh_title(
        "Standing title", replies, turns=_theme_rows(), newest="and the signup flow"
    )
    assert result.outcome == naming.TITLE_REFRESHED
    assert result.title == "Fresh clean title"


# -- the fit-check seam itself ---------------------------------------------


@pytest.mark.asyncio
async def test_the_fit_check_adapter_resolves_per_call_and_fails_open() -> None:
    class _Answer:
        value = naming.TITLE_DOESNT_FIT

    class _Seam:
        async def decide(self, *, state: str, question: object) -> object:
            return _Answer()

    holder: dict[str, object] = {"seam": _Seam()}
    check = naming.title_fit_check(lambda: holder["seam"])
    assert await check("state") == naming.TITLE_DOESNT_FIT

    # Resolved PER CALL: swapping the seam after the adapter was built is the
    # documented way tests and hosts inject one.
    holder["seam"] = None
    assert await check("state") is None

    class _Broken:
        async def decide(self, *, state: str, question: object) -> object:
            raise RuntimeError("vendor down")

    holder["seam"] = _Broken()
    assert await check("state") is None

    class _Unknown:
        async def decide(self, *, state: str, question: object) -> object:
            return SimpleNamespace(value="something_else")

    holder["seam"] = _Unknown()
    assert await check("state") is None

    # A seam with no ``decide`` at all is "no classifier", not an error.
    holder["seam"] = object()
    assert await check("state") is None


@pytest.mark.asyncio
async def test_the_fit_check_question_carries_the_three_choice_ids() -> None:
    question = naming.title_fit_question()
    assert question.id == naming.TITLE_QUESTION_ID
    assert set(question.criteria) == {
        naming.TITLE_FITS,
        naming.TITLE_DOESNT_FIT,
        naming.TITLE_CANT_TELL,
    }


def test_the_fit_state_carries_the_candidate_and_a_bounded_excerpt() -> None:
    state = naming.title_fit_state("Candidate title", "word " * 500)
    assert "Candidate title" in state
    assert len(state) < 1200
    assert "\u2026" in state  # the module's one truncation marker
