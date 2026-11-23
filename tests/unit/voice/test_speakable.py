"""What is read aloud (design V8): agent text, cleaned and capped."""

from __future__ import annotations

import pytest

from local_operator.voice import speakable as sp


def test_code_urls_and_markdown_are_not_read_aloud() -> None:
    text = (
        "# Result\n"
        "**Done.** See https://example.com/a?b=c and [the docs](https://x.y/z).\n"
        "```python\nprint('secret')\n```\n"
        "- first item\n- `second` item\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n"
    )
    spoken = sp.speakable(text).text
    assert "print" not in spoken
    assert "http" not in spoken
    assert "**" not in spoken and "`" not in spoken and "#" not in spoken and "|" not in spoken
    assert sp.CODE_PLACEHOLDER in spoken
    assert "the docs" in spoken and "link" in spoken
    assert spoken.startswith("Result Done.")


def test_the_stand_in_words_are_the_callers_and_taken_literally() -> None:
    """The module speaks English by default; a route hands it the locale's words, and a
    backslash in one is text, not a regex group reference."""
    text = "Open https://example.com first.\n```\nx = 1\n```\nThen stop."
    spoken = sp.speakable(text, code_omitted=r"sin c\1digo", link=r"enlace\g<0>").text
    assert r"enlace\g<0>" in spoken and r"sin c\1digo" in spoken
    assert "link" not in spoken and "code block omitted" not in spoken


def test_an_unterminated_fence_still_hides_the_code() -> None:
    assert "rm -rf" not in sp.speakable("Run this:\n```sh\nrm -rf build").text


def test_embedded_commands_are_never_read_aloud() -> None:
    """B2: a ``[[…]]`` control token is machinery, not words for a listener."""
    assert sp.clean("Done. [[notify: build green]] See you.") == "Done. See you."
    assert sp.clean("[[a]] and [[b: c]]") == "and"
    # The link-suffixed form goes whole, rather than leaving a "[x" link text.
    assert sp.clean("a [[x]](http://y) b") == "a b"
    # Bounded to one line, so a stray opener cannot swallow the rest of a reply.
    assert sp.clean("keep [[ this reply") == "keep [[ this reply"
    assert sp.clean("first [[open\nsecond]] end") == "first [[open second]] end"
    # An ordinary markdown link is still read as its text.
    assert sp.clean("see [the docs](http://x) now") == "see the docs now"


def test_short_text_is_not_truncated() -> None:
    result = sp.speakable("Two files changed.", max_chars=800)
    assert result == sp.Speakable("Two files changed.", truncated=False)


def test_the_cap_cuts_on_a_sentence_and_never_exceeds_the_limit() -> None:
    text = " ".join(f"Sentence number {index} is here." for index in range(100))
    result = sp.speakable(text, max_chars=200)
    assert result.truncated
    assert len(result.text) <= 200
    assert result.text.endswith(sp.TRUNCATION_TAIL)
    body = result.text[: -len(sp.TRUNCATION_TAIL)].rstrip()
    assert body.endswith("is here.")


def test_the_tail_is_spoken_in_the_first_person_and_rides_inside_the_cap() -> None:
    """The listener cannot see the screen, so the cut is said, not implied.

    The tail counts against the cap like any other spoken word: at the
    registry's floor the budget left for the answer is still positive, which
    is the arithmetic that forbids a longer tail.
    """
    assert sp.TRUNCATION_TAIL == "That's where I'll stop — the rest is on screen."
    text = " ".join(f"Sentence number {index} is here." for index in range(100))
    result = sp.speakable(text, max_chars=sp.MIN_MAX_SPOKEN_CHARS)
    assert result.truncated
    assert result.text.endswith(sp.TRUNCATION_TAIL)
    assert len(result.text) <= sp.MIN_MAX_SPOKEN_CHARS
    body = result.text[: -len(sp.TRUNCATION_TAIL)].rstrip()
    assert len(body) >= 20


def test_the_tail_is_never_clipped_even_when_the_cut_has_no_whitespace() -> None:
    """A cap must not eat the tail's final full stop (review N5).

    With no whitespace inside the window there is no word boundary, so the cut
    is the whole window; the full stop added to end that cut then made the
    assembled string one character over, and slicing it to the cap removed the
    TAIL's own full stop. The listener, who cannot see the cut, lost the only
    sign that the reply had stopped. Unbroken tokens and unspaced CJK text are
    the two shapes that reach the branch, so both are swept across the cap
    range rather than spot-checked.
    """
    unbroken = "x" * 500
    unspaced = "这是一段没有任何空格的中文文本，用来测试截断的行为。" * 20
    # Up to the shorter sample's length: above it the text is not truncated at
    # all, and the branch under test is never reached.
    for cap_value in range(sp.MIN_MAX_SPOKEN_CHARS, len(unbroken)):
        for source in (unbroken, unspaced):
            result = sp.speakable(source, max_chars=cap_value)
            assert result.truncated
            assert len(result.text) <= cap_value
            assert result.text.endswith(sp.TRUNCATION_TAIL), (cap_value, result.text[-60:])
            assert result.text.endswith(".")


def test_the_cap_falls_back_to_a_word_boundary() -> None:
    text = "word " * 400
    result = sp.speakable(text, max_chars=150)
    assert result.truncated and len(result.text) <= 150
    assert "wor " not in result.text


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, sp.DEFAULT_MAX_SPOKEN_CHARS),
        ("garbage", sp.DEFAULT_MAX_SPOKEN_CHARS),
        (True, sp.DEFAULT_MAX_SPOKEN_CHARS),
        (5, sp.MIN_MAX_SPOKEN_CHARS),
        (10**9, sp.MAX_MAX_SPOKEN_CHARS),
        ("300", 300),
    ],
)
def test_a_hand_edited_cap_never_breaks_voice(value: object, expected: int) -> None:
    assert sp.normalise_max_chars(value) == expected


def test_a_caller_supplied_tail_replaces_the_english_one_inside_the_cap() -> None:
    """The route passes the active locale's phrase; the budget is that tail's own length."""
    tail = "Aquí me detengo: el resto está en pantalla."
    text = " ".join(f"Sentence number {index} is here." for index in range(100))
    result = sp.speakable(text, max_chars=sp.MIN_MAX_SPOKEN_CHARS, tail=tail)
    assert result.truncated
    assert result.text.endswith(tail) and sp.TRUNCATION_TAIL not in result.text
    assert len(result.text) <= sp.MIN_MAX_SPOKEN_CHARS
    assert sp.speakable(text, max_chars=200).text.endswith(sp.TRUNCATION_TAIL)


def test_display_text_keeps_a_table_as_one_line_per_row() -> None:
    raw = (
        "| Hook | Fires | Fixes today? |\n|---|:---:|---:|\n"
        "| **Pre** | yes | no |\n| Post | no | yes |"
    )
    assert sp.display_text(raw) == "Hook · Fires · Fixes today?\nPre · yes · no\nPost · no · yes"
    # The spoken string is still flat: that is the contract the audio depends on.
    assert "\n" not in sp.clean(raw) and "·" not in sp.clean(raw)


def test_display_text_handles_a_table_without_outer_pipes() -> None:
    raw = "Name | Value\n---|---\nfoo | 1\nbar | 2\nAfter the table."
    assert sp.display_text(raw) == "Name · Value\nfoo · 1\nbar · 2\nAfter the table."


def test_display_text_keeps_bullet_markers_and_nesting() -> None:
    raw = "- one\n  * **two** and `code`\n1. first\n2) second\n+ plus"
    assert sp.display_text(raw) == "- one\n  * two and code\n1. first\n2) second\n+ plus"


def test_display_text_drops_heading_and_quote_markers_and_unwraps_emphasis() -> None:
    raw = "## Title\n> quoted _text_\nplain ~~gone~~ **bold**"
    assert sp.display_text(raw) == "Title\nquoted text\nplain gone bold"


def test_display_text_fence_is_the_placeholder_on_its_own_line() -> None:
    assert (
        sp.display_text("Before.\n```py\nx = 1\n```\nAfter.")
        == "Before.\n(code block omitted)\nAfter."
    )
    assert sp.display_text("Here. ```\nls\n``` Done.") == "Here.\n(code block omitted)\nDone."
    assert sp.display_text("```\nls", code_omitted="sin código") == "sin código"


def test_display_text_links_keep_text_and_bare_urls_stay_whole() -> None:
    raw = "See [the docs](https://a.com/x_y_z) or https://b.com/a_b_c_d/*star* now."
    assert sp.display_text(raw) == "See the docs or https://b.com/a_b_c_d/*star* now."
    assert sp.display_text("[[open file]](x) done") == "done"


def test_display_text_normalises_newlines_and_spacing() -> None:
    raw = "one  \t two\r\n\r\n\r\n\r\nthree\rfour   \n\n"
    assert sp.display_text(raw) == "one two\n\nthree\nfour"
    assert sp.display_text("") == "" and sp.display_text(" \n ") == ""


def test_display_text_is_uncapped() -> None:
    long = " ".join(f"Sentence {n}." for n in range(500))
    assert sp.display_text(long) == long
