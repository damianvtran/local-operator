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


# -- spoken summary: the agent's ``Spoken:`` paragraph (docs/design/voice-spoken-summary.md) --


REPLY = "Here is the long report with many details.\n\nSecond paragraph of detail."


def test_no_marker_is_byte_identical_to_the_whole_reply_path() -> None:
    plain = sp.speakable(REPLY)
    assert plain == sp.cap(sp.clean(REPLY), sp.DEFAULT_MAX_SPOKEN_CHARS)
    assert plain.source == "reply"
    assert sp.select_spoken(REPLY) is None


@pytest.mark.parametrize(
    "marker",
    ["Spoken:", "**Spoken:**", "**Spoken**:", "_Spoken:_", "spoken:", "SPOKEN:", "  Spoken:"],
)
def test_a_paragraph_opening_with_the_label_is_spoken_without_the_label(marker: str) -> None:
    spoken = sp.speakable(f"{REPLY}\n\n{marker} The point is done.")
    assert spoken.text == "The point is done."
    assert spoken.source == "summary" and spoken.truncated is False


def test_a_label_wrapping_the_whole_paragraph_leaves_no_stray_marker() -> None:
    assert sp.speakable("Long.\n\n**Spoken: all wrapped.**").text == "all wrapped."


def test_the_last_marker_paragraph_wins() -> None:
    text = "Spoken: Early.\n\nMore work.\n\nspoken: Final answer.\n\nTail."
    assert sp.speakable(text).text == "Final answer."


def test_the_paragraph_runs_to_the_next_blank_line_only() -> None:
    text = "Intro.\n\nSpoken: First line.\nSecond line.\n\nAfter."
    assert sp.speakable(text).text == "First line. Second line."


def test_an_empty_marker_falls_back_to_the_whole_reply_not_silence() -> None:
    for text in (f"{REPLY}\n\nSpoken:", f"{REPLY}\n\nSpoken:   \n", f"{REPLY}\n\n**Spoken:**"):
        spoken = sp.speakable(text)
        assert spoken.source == "reply" and spoken.text
        assert "Here is the long report" in spoken.text
        # The label is machinery: the reply is spoken, the word "Spoken" never is.
        assert "spoken" not in spoken.text.lower(), text


def test_a_marker_that_cleans_to_nothing_falls_back_too() -> None:
    spoken = sp.speakable("Real reply here.\n\nSpoken: `` ")
    assert spoken.source == "reply" and "Real reply here." in spoken.text


def test_the_label_only_counts_at_a_paragraph_start() -> None:
    for text in (
        "He said Spoken: nothing at all.",
        "A line\nSpoken: not after one newline.",
        "The spoken: word in a sentence.",
        "- Spoken: inside a list.",
    ):
        assert sp.select_spoken(text) is None, text
        assert sp.speakable(text).source == "reply"


def test_a_code_fence_containing_the_label_is_not_a_marker() -> None:
    fenced = "Example:\n\n```\nSpoken: do not say this\n\nSpoken: nor this\n```\n\nDone."
    assert sp.select_spoken(fenced) is None
    assert sp.select_spoken("Example:\n\n~~~\nSpoken: x\n~~~") is None
    # An unfinished fence (a message cut mid-stream) hides its tail as well.
    assert sp.select_spoken("Working.\n\n```\nSpoken: x") is None
    # A fence inside the paragraph does not end it at the fence's own blank line.
    inside = "Spoken: Before.\n```\nx\n\ny\n```\nAfter."
    assert sp.speakable(inside).text == f"Before. {sp.CODE_PLACEHOLDER} After."


def test_a_trailing_partial_message_has_no_marker_and_falls_back() -> None:
    partial = f"{REPLY}\n\nSpok"
    assert sp.select_spoken(partial) is None
    assert sp.speakable(partial).source == "reply"


def test_the_selected_paragraph_is_cleaned_like_any_other_text() -> None:
    spoken = sp.speakable("Report.\n\nSpoken: **Done**, see https://x.y/z and `run it`.")
    assert spoken.text == "Done, see link and run it."


def test_a_long_summary_is_cut_by_the_cap_and_says_so() -> None:
    summary = " ".join(f"Sentence {n} is here." for n in range(80))
    spoken = sp.speakable(f"Short reply.\n\nSpoken: {summary}", max_chars=200)
    assert spoken.source == "summary" and spoken.truncated is True
    assert len(spoken.text) <= 200 and spoken.text.endswith(sp.TRUNCATION_TAIL)
    assert "Short reply" not in spoken.text


def test_summary_false_never_selects() -> None:
    text = f"{REPLY}\n\nSpoken: Only this."
    gate = sp.speakable(text, summary=False)
    assert gate.source == "reply" and "Here is the long report" in gate.text


def test_selection_is_a_pure_function_of_the_text() -> None:
    text = f"{REPLY}\n\nSpoken: Same every time."
    assert {sp.speakable(text) for _ in range(5)} == {sp.speakable(text)}


@pytest.mark.parametrize(
    "marker", ["    Spoken:", "\tSpoken:", "  \tSpoken:", "     **Spoken:**", "\t_Spoken:_"]
)
def test_an_indented_code_block_opening_with_the_label_is_not_a_marker(marker: str) -> None:
    """Four spaces or a tab make a code block, which may quote the protocol (m1)."""
    text = f"{REPLY}\n\n{marker} do not say this"
    assert sp.select_spoken(text) is None
    spoken = sp.speakable(text)
    assert spoken.source == "reply" and "do not say this" not in spoken.text.split("Second")[0]
    # ...also as the very first paragraph, where the scan starts at offset 0.
    assert sp.select_spoken(f"{marker} first and only") is None


@pytest.mark.parametrize("indent", ["", " ", "  ", "   "])
def test_up_to_three_spaces_of_indent_still_open_a_marker(indent: str) -> None:
    assert sp.select_spoken(f"{REPLY}\n\n{indent}Spoken: Fine.") == "Fine."
    assert sp.select_spoken(f"{indent}Spoken: Fine.") == "Fine."


def test_a_quoted_block_is_not_a_marker() -> None:
    """A ``>`` prefix is a quote of someone else's words, so ``> Spoken:`` is not the
    agent's own summary (decided, m4): the quote marker is not whitespace."""
    assert sp.select_spoken("Quote follows.\n\n> Spoken: not mine.") is None
    assert sp.select_spoken("Quote follows.\n\n> Spoken: not mine.\n\nSpoken: Mine.") == "Mine."


def test_crlf_line_endings_select_the_same_paragraph() -> None:
    text = "Long answer.\r\n\r\nSpoken: First line.\r\nSecond line.\r\n\r\nAfter."
    assert sp.select_spoken(text) == "First line.\r\nSecond line."
    assert sp.speakable(text).text == "First line. Second line."
    assert sp.speakable("Long.\r\n\r\n**Spoken:** Done.\r\n").text == "Done."
    # A label alone, CRLF, summary after the blank line.
    assert sp.speakable("Long.\r\n\r\n**Spoken:**\r\n\r\nDone.").text == "Done."


def test_unicode_summaries_are_spoken_whole() -> None:
    for summary in ("完了しました。二つ変わりました。", "Listo: ñandú, café ☕.", "Готово 👍"):
        spoken = sp.speakable(f"Longer.\n\nSpoken: {summary}")
        assert spoken.source == "summary" and spoken.text == summary
    # The label is ASCII protocol, case-folded; a lookalike is plain prose.
    assert sp.select_spoken("Reply.\n\nSpoken：fullwidth colon") is None
    assert sp.select_spoken("Reply.\n\nSρoken: lookalike") is None


def test_a_label_alone_takes_the_next_paragraph_as_the_summary() -> None:
    """Q3 (a): the label is never the thing spoken; its summary is the paragraph after."""
    for label in ("**Spoken:**", "Spoken:", "_Spoken:_", "**Spoken**:"):
        spoken = sp.speakable(f"{REPLY}\n\n{label}\n\nDone. Two things changed.")
        assert spoken.text == "Done. Two things changed." and spoken.source == "summary"
    # More than one blank line between them, and trailing paragraphs after the summary.
    text = f"{REPLY}\n\n**Spoken:**\n\n\n\nDone.\n\nAfter."
    assert sp.select_spoken(text) == "Done."


def test_a_later_marker_still_wins_over_a_label_alone() -> None:
    text = "Long.\n\n**Spoken:**\n\nTaken from the next paragraph.\n\nSpoken: But this is later."
    assert sp.select_spoken(text) == "But this is later."


def test_a_trailing_label_alone_falls_back_to_the_reply_without_the_label() -> None:
    """Q3 (b): nothing follows the label, so the reply is spoken and the label dropped."""
    for text in (f"{REPLY}\n\n**Spoken:**", f"{REPLY}\n\n**Spoken:**\n\n", "**Spoken:**"):
        spoken = sp.speakable(text)
        assert spoken.source == "reply"
        assert "spoken" not in spoken.text.lower()
    assert sp.speakable(f"{REPLY}\n\nSpoken:").text == sp.speakable(REPLY).text
    # ...and the label dropped is only the empty FINAL one, never an earlier paragraph.
    kept = sp.speakable("Spoken: Early body.\n\nMore.\n\nSpoken:")
    assert kept.source == "reply" and "Early body." in kept.text and "More." in kept.text
