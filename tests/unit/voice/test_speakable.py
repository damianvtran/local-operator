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


@pytest.mark.parametrize("sentinel", ["\ue000999\ue001", "\ue0000\ue001", "\ue000", "\ue001"])
def test_display_text_survives_the_private_use_sentinel_in_its_input(sentinel: str) -> None:
    """A reply carrying the code points ``display_text`` parks spans behind must not
    crash the route (IndexError) or restore someone else's span."""
    out = sp.display_text(f"x {sentinel} y")
    assert "\ue000" not in out and "\ue001" not in out and out.startswith("x ")
    # After a URL, ``\ue0000\ue001`` used to restore as the URL a second time.
    assert sp.display_text("https://a.com \ue0000\ue001 z") == "https://a.com 0 z"
    assert sp.display_text("see `a_b` \ue0001\ue001 end") == "see a_b 1 end"
    out = sp.display_text("```\ncode\n```\n\ue0000\ue001 https://a.com/x")
    assert "\ue000" not in out and "\ue001" not in out


def test_display_text_leaves_identifiers_intact() -> None:
    raw = "Edit `my_var_name` and src/foo_bar_baz.py then a*b*c and __init__ and __name__.py."
    assert sp.display_text(raw) == raw.replace("`", "")
    assert sp.display_text("call my_func_name() in some_file.py") == (
        "call my_func_name() in some_file.py"
    )
    # Code spans are untouched inside, backticks removed; emphasis inside is NOT unwrapped.
    assert sp.display_text("`**not bold**` and `_x_`") == "**not bold** and _x_"
    assert sp.display_text("foo*bar and *it*") == "foo*bar and it"
    assert sp.display_text("2*3*4 = 24") == "2*3*4 = 24"


def test_display_text_still_unwraps_real_emphasis() -> None:
    assert sp.display_text("**bold** *it* _it_ ~~gone~~ __two words__") == (
        "bold it it gone two words"
    )
    assert sp.display_text("(**x**), _y_. **Note**: ok") == "(x), y. Note: ok"
    # CJK has no spaces around a bold run; it must still unwrap.
    assert sp.display_text("你好**粗体**文字") == "你好粗体文字"


def test_display_text_unwraps_nested_emphasis_in_both_passes() -> None:
    """One sweep takes the outer pair; the inner pair needs the second pass."""
    assert sp.display_text("**_x_**") == "x"
    assert sp.display_text("_**x**_ and ~~**y**~~") == "x and y"


def test_display_text_unwraps_emphasis_around_a_url_and_keeps_the_url() -> None:
    assert sp.display_text("**https://a.com/x_y**") == "https://a.com/x_y"
    assert sp.display_text("_https://a.com/x_ and (https://b.com/y).") == (
        "https://a.com/x and (https://b.com/y)."
    )
    assert sp.display_text("a `https://a.com/x_y_z` b") == "a https://a.com/x_y_z b"


def test_display_text_table_rule_takes_one_dash_and_escaped_pipes_stay_in_the_cell() -> None:
    assert sp.display_text("|a|b|\n|-|-|\n|1|2|") == "a · b\n1 · 2"
    assert sp.display_text("a | b\n--|--\n1 | 2") == "a · b\n1 · 2"
    assert sp.display_text("|x \\| y|`c \\| d`|\n|--|--|") == "x | y · c | d"
    # A lone dash is text, not a delimiter row (no header above it).
    assert sp.display_text("-\n--\nfoo") == "-\n--\nfoo"
    # ...and an empty header is not one either, so the row is kept rather than lost.
    assert sp.display_text("|\n|-|") == "-"
    # Empty cells are dropped, not placeholders.
    assert sp.display_text("| a | b |\n|---|---|\n| 1 |  | 3 |") == "a · b\n1 · 3"


def test_display_text_hides_a_tilde_fence_like_a_backtick_one() -> None:
    assert sp.display_text("Before\n~~~py\nx_y = 1\n~~~\nAfter") == (
        "Before\n(code block omitted)\nAfter"
    )
    assert sp.display_text("~~~\nunclosed") == "(code block omitted)"
    # Not at the start of a line: ordinary text.
    assert sp.display_text("a ~~~ b") == "a ~~~ b"


def test_display_text_leaves_a_very_long_line_unemphasised_and_stays_fast() -> None:
    import time

    started = time.perf_counter()
    out = sp.display_text("_a " * 20000)
    assert out == ("_a " * 20000).strip()
    # Under the line limit the markers still pair up.
    assert sp.display_text("_it_ " * 100).strip() == " ".join(["it"] * 100)
    many = "\n".join("_a " * 660 for _ in range(30))
    sp.display_text(many)
    assert time.perf_counter() - started < 2.0
