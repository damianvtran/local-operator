"""Leaked DeepSeek tool-call markup is stripped from streamed assistant text.

Fixtures are synthesized in the shapes observed live on radient/auto (bars are
fullwidth U+FF5C); none of them is a real transcript line.
"""

from __future__ import annotations

import pytest

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
    assert stream.feed("Hello ") == "Hello "
    assert stream.feed("world") == "world"
    # The newline is held until the next line proves it is not a markup gap.
    assert stream.feed(".\nNext") == ".\nNext"
    assert stream.finish(has_tool_calls=False) == ""
