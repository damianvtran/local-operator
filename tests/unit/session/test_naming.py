"""Focused ``parse_title`` / generate_title coverage.

The TUI suite owns scheduling (when the call fires, what a failure costs the
latch). This file owns the parser: tagged Claude replies, untagged Grok
replies, leaked thinking, JSON wrappers, and the generate_title / generate_retitle
failure modes that used to live only as comments on the TUI tests.
"""

from __future__ import annotations

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
