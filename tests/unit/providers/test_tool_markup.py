"""Leaked DeepSeek tool-call markup is stripped from streamed assistant text.

Fixtures are synthesized in the shapes observed live on radient/auto (bars are
fullwidth U+FF5C); none of them is a real transcript line.
"""

from __future__ import annotations

import logging
import re
import time

import pytest

from local_operator.providers import tool_markup
from local_operator.providers.tool_markup import ToolMarkupFilter, strip_tool_markup

DSML = "\uff5cDSML\uff5c"


def _chunked(text: str, size: int, *, has_tool_calls: bool) -> str:
    stream = ToolMarkupFilter()
    out = [stream.feed(text[i : i + size]) for i in range(0, len(text), size)]
    return "".join(out) + stream.finish(has_tool_calls=has_tool_calls)


#: (leaked text, turn has structured calls, text that must survive)
OBSERVED_SHAPES = [
    pytest.param(
        f'{DSML} parameter name="i" string="true">Notes filed quietly',
        True,
        "",
        id="dsml-led-whole-turn",
    ),
    pytest.param(
        'parameter name="channel_id" string="true">C0EXAMPLE01',
        True,
        "",
        id="bare-parameter-no-dsml-token",
    ),
    pytest.param(
        "Reading the slice for today again.\n\n"
        f'{DSML} parameter name="channel_id" string="true">U0EXAMPLE02',
        True,
        "Reading the slice for today again.",
        id="prose-then-dsml",
    ),
    pytest.param(
        "Searching the archive next.\n\n" ' parameter name="keywords" string="false">["alpha"]',
        True,
        "Searching the archive next.",
        id="prose-then-indented-bare-parameter",
    ),
    pytest.param(
        f'{DSML} parameter name="keywords" string="false">["alpha", "beta"]',
        True,
        "",
        id="dsml-led-json-value",
    ),
    pytest.param(
        '="i" string="true">Filed, staying quiet\n\n'
        f'{DSML} parameter name="i" string="true">Filed again',
        True,
        "",
        id="truncated-head-then-dsml",
    ),
    pytest.param(
        'name="keywords" string="false">["x"]\n'
        ' name="keywords" string="false">["y", "z"]\n'
        f'{DSML} parameter name="keywords" string="false">["y", "w"]',
        True,
        "",
        id="several-parameter-lines",
    ),
    pytest.param(
        'i" string="true">Filed, staying quiet',
        True,
        "",
        id="name-tail-head-whole-turn",
    ),
    pytest.param(
        'Checking.\ni" string="true">value',
        True,
        "Checking.",
        id="prose-then-name-tail-head",
    ),
    pytest.param(
        'keywords" string="false">["a"]',
        True,
        "",
        id="whole-name-without-attribute-head",
    ),
    pytest.param(
        'Checking.\n\nC0EXAMPL" string="true">value\n' 'i string="false">v2\n' ' string="true">v3',
        True,
        "Checking.",
        id="name-tail-quote-optional-and-empty",
    ),
    pytest.param(
        "Checking.\r\n\r\n" f'{DSML} parameter name="i" string="true">v\r\n',
        True,
        "Checking.",
        id="crlf-leaves-no-carriage-return",
    ),
]


@pytest.mark.parametrize(("text", "has_calls", "expected"), OBSERVED_SHAPES)
def test_observed_shapes_are_stripped(text: str, has_calls: bool, expected: str) -> None:
    assert strip_tool_markup(text, has_tool_calls=has_calls) == expected


@pytest.mark.parametrize(("text", "has_calls", "expected"), OBSERVED_SHAPES)
@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 13])
def test_markup_split_across_chunks_is_stripped(
    text: str, has_calls: bool, expected: str, size: int
) -> None:
    assert _chunked(text, size, has_tool_calls=has_calls) == expected


def test_dsml_token_split_mid_token_never_reaches_output() -> None:
    """A chunk ending inside the token must hold, not emit, the partial token."""
    stream = ToolMarkupFilter()
    emitted = [
        stream.feed("Done.\n\n\uff5cDS"),
        stream.feed("ML\uff5c parameter name="),
        stream.feed('"i" string="true">x'),
    ]
    assert emitted == ["Done.", "", ""]
    assert stream.finish(has_tool_calls=True) == ""


def test_dsml_markup_is_stripped_even_without_structured_calls() -> None:
    text = f'Answer.\n{DSML} parameter name="i" string="true">x\nmore answer'
    assert strip_tool_markup(text, has_tool_calls=False) == "Answer.\nmore answer"


def test_bare_parameter_line_without_calls_is_kept_verbatim() -> None:
    text = 'Example:\nparameter name="channel_id" string="true">C0EXAMPLE01'
    assert strip_tool_markup(text, has_tool_calls=False) == text
    assert _chunked(text, 3, has_tool_calls=False) == text


@pytest.mark.parametrize(
    "text",
    [
        'The native tag reads parameter name="x" string="true">value mid-sentence.',
        '```\nparameter name="x" string="true">value\n```\n',
        'name="foo" is the attribute you want.\n',
        "<b>bold</b> and |a|b| tables\n|-|-|\n",
        "Plain answer.\n\n\nWith blank lines.\n\n",
    ],
    ids=["mid-sentence", "fenced", "name-attr-prose", "html-and-table", "blank-lines"],
)
def test_prose_mentioning_the_syntax_is_untouched_even_with_calls(text: str) -> None:
    assert strip_tool_markup(text, has_tool_calls=True) == text
    for size in (1, 4, 7):
        assert _chunked(text, size, has_tool_calls=True) == text


def test_ordinary_prose_streams_without_being_held() -> None:
    stream = ToolMarkupFilter()
    # A line's first word could still be a parameter-name tail (``i" string=``),
    # so it waits for the character that rules that out; the rest streams.
    assert stream.feed("Hello ") == ""
    assert stream.feed("world") == "Hello world"
    assert stream.feed(" again") == " again"
    # The newline is held until the next line proves it is not a markup gap.
    assert stream.feed(".\nNext, more") == ".\nNext, more"
    assert stream.finish(has_tool_calls=False) == ""


def test_name_tail_head_split_across_the_quote_is_held_not_emitted() -> None:
    """QA round 1, Q1: the line is undecided while it is still a name tail."""
    stream = ToolMarkupFilter()
    assert stream.feed("Checking.\n") == "Checking."
    assert stream.feed("i") == ""
    assert stream.feed('" ') == ""
    assert stream.feed('string="tr') == ""
    assert stream.feed('ue">value') == ""
    assert stream.finish(has_tool_calls=True) == ""


def test_name_tail_head_without_calls_is_kept_verbatim() -> None:
    text = 'Checking.\ni" string="true">value'
    assert strip_tool_markup(text, has_tool_calls=False) == text


@pytest.mark.parametrize(
    "text",
    [
        "Ok\n",
        "Total: 3 items\n",
        'say "hello" to them\n',
        'keywords are "quoted" here\n',
        "i\n",
        'i" is a quote mark\n',
        "\u65e5\u672c\u8a9e\u306e\u6587\u7ae0\u3067\u3059\u3002\n",
    ],
    ids=[
        "short",
        "word-colon",
        "quote-later",
        "word-then-quote",
        "lone-ident",
        "ident-quote",
        "cjk",
    ],
)
def test_lines_that_only_start_like_a_name_tail_are_prose(text: str) -> None:
    assert strip_tool_markup(text, has_tool_calls=True) == text
    for size in (1, 2, 5):
        assert _chunked(text, size, has_tool_calls=True) == text


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            f'```x``` is the syntax\nprose\n{DSML} parameter name="i" string="true">v',
            id="inline-span-opens-no-fence",
        ),
        pytest.param(
            f'````\n```\n````\n{DSML} parameter name="i" string="true">v',
            id="shorter-run-does-not-close-four-backticks",
        ),
        pytest.param(
            f'~~~\nx\n```\n~~~\n{DSML} parameter name="i" string="true">v',
            id="other-character-does-not-close",
        ),
        pytest.param(
            f'```\nx\n``` trailing\n```\n{DSML} parameter name="i" string="true">v',
            id="closer-takes-no-info-string",
        ),
    ],
)
def test_fences_follow_commonmark(text: str) -> None:
    """Review M2 / QA Q2: the DSML line after each block is outside any fence."""
    expected = text.rsplit("\n", 1)[0]
    assert strip_tool_markup(text, has_tool_calls=True) == expected
    for size in (1, 3, 7):
        assert _chunked(text, size, has_tool_calls=True) == expected


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            f'````\n```\n{DSML} parameter name="i" string="true">v\n```\n````\n',
            id="inside-four-backticks",
        ),
        pytest.param(
            f'```python\n{DSML} parameter name="i" string="true">v\n```\n',
            id="info-string",
        ),
        pytest.param(
            f'~~~ `x`\n{DSML} parameter name="i" string="true">v\n~~~\n',
            id="tilde-info-may-hold-backticks",
        ),
    ],
)
def test_markup_inside_a_fence_is_kept(text: str) -> None:
    assert strip_tool_markup(text, has_tool_calls=True) == text
    for size in (1, 3, 7):
        assert _chunked(text, size, has_tool_calls=True) == text


@pytest.mark.parametrize("indent", ["\u3000", "\u00a0", "\f", " \t"])
def test_dsml_line_indented_by_any_whitespace_is_dropped_without_calls(indent: str) -> None:
    """Review N1: the stream skips indentation as ``\\s`` (U+3000, U+00A0, form
    feed) before classifying a line, so dropping the DSML lines from a no-call
    tail must skip the same class, or a line already judged markup is kept."""
    text = f'Hi\n{indent}{DSML} parameter name="i" string="true">v\nafter\n'
    expected = "Hi\nafter\n"
    assert strip_tool_markup(text, has_tool_calls=False) == expected
    for size in (1, 3, 7):
        assert _chunked(text, size, has_tool_calls=False) == expected


def test_crlf_prose_without_markup_is_byte_identical() -> None:
    text = "Line one.\r\n\r\nLine two.\r\nEnd\r"
    assert strip_tool_markup(text, has_tool_calls=True) == text
    for size in (1, 2, 3):
        assert _chunked(text, size, has_tool_calls=True) == text


def test_crlf_split_between_deltas_leaves_no_carriage_return() -> None:
    stream = ToolMarkupFilter()
    out = stream.feed("Checking.\r") + stream.feed("\n")
    out += stream.feed(f'{DSML} parameter name="i" string="true">v')
    assert out + stream.finish(has_tool_calls=True) == "Checking."


def test_with_calls_the_turn_ends_at_the_first_markup_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Review N2(c), M1: prose after a markup line goes with it, and is logged by
    size only. The observed data has no prose there; this pins the rule."""
    secret = "prose after the call"
    text = f'Config:\nname="x" string="true">oops\n{secret}\n'
    with caplog.at_level(logging.DEBUG, logger="local_operator.providers.tool_markup"):
        assert strip_tool_markup(text, has_tool_calls=True) == "Config:"
    records = [r.getMessage() for r in caplog.records]
    assert records == [
        "tool markup filter: dropped 1 non-markup line(s), 21 chars, after a markup "
        "line in a turn with tool calls"
    ]
    assert secret not in records[0]


def test_dropping_only_markup_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    text = f'Done.\n\n{DSML} parameter name="i" string="true">v\nname="j" string="false">[1]'
    with caplog.at_level(logging.DEBUG, logger="local_operator.providers.tool_markup"):
        assert strip_tool_markup(text, has_tool_calls=True) == "Done."
    assert caplog.records == []


def test_without_calls_the_text_after_a_bare_line_is_released_at_the_end() -> None:
    stream = ToolMarkupFilter()
    assert stream.feed('Example:\nname="x" string="true">v\n') == "Example:"
    assert stream.feed("more prose\n") == ""
    assert stream.finish(has_tool_calls=False) == '\nname="x" string="true">v\nmore prose\n'


@pytest.mark.parametrize("delta", ["\n", " ", "\r\n", "\t\n"])
def test_whitespace_runs_are_linear(delta: str) -> None:
    """QA Q3: 200 000 one-character whitespace deltas took 5.9 s when every delta
    re-scanned the held run. Counting the characters the filter reads keeps the
    test independent of the host's speed: the bound is linear with a small
    constant, which the old quadratic code misses by four orders of magnitude."""
    reads = 0
    real_match = tool_markup._SPACE_RUN.match

    class CountingPattern:
        def match(self, text: str, pos: int = 0) -> re.Match[str] | None:
            nonlocal reads
            found = real_match(text, pos)
            reads += found.end() - pos + 1 if found else 1
            return found

    deltas = 20_000
    stream = ToolMarkupFilter()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tool_markup, "_SPACE_RUN", CountingPattern())
        emitted = [stream.feed(delta) for _ in range(deltas)]
        emitted.append(stream.feed("end"))
    assert "".join(emitted) + stream.finish(has_tool_calls=False) == delta * deltas + "end"
    assert reads <= 3 * len(delta) * deltas
    assert sum(len(part) for part in stream._gap + stream._indent) == 0


def test_long_whitespace_stream_finishes_quickly() -> None:
    """Wall-clock backstop for QA Q3 with a bound ~50x the measured cost."""
    stream = ToolMarkupFilter()
    started = time.perf_counter()
    for _ in range(200_000):
        stream.feed("\n")
    stream.feed("end")
    stream.finish(has_tool_calls=False)
    assert time.perf_counter() - started < 5.0
