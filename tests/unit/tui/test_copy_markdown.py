"""The rendered→source alignment behind markdown copy.

These tests pin the engine directly (no app) so a misalignment names the
construct that broke rather than surfacing as a wrong clipboard three calls
away. Each builds the real flattened frame at a width, aligns it, and slices.
"""

from __future__ import annotations

import pytest
from rich.console import Console
from rich.markdown import Markdown

from local_operator.tui.markdown_theme import (
    brand_markdown_theme,
    install_markdown_theme,
)
from local_operator.tui.widgets._copy_markdown import align, slice_markdown
from local_operator.tui.widgets.assistant import flatten

MIXED = """Intro sentence here.

- first bullet that is long enough to wrap onto a second rendered row definitely
- second bullet

> a quoted line that is also long enough to wrap around onto more than one row
> second quoted source line

```python
def f(x):
    return x
```

Closing paragraph.
"""


@pytest.fixture(autouse=True)
def _theme() -> None:
    install_markdown_theme()


def _rows(source: str, width: int) -> list[str]:
    console = Console(width=width, theme=brand_markdown_theme())
    return flatten(Markdown(source), width, console).plain.split("\n")


def _full_slice(source: str, width: int) -> str:
    rows = _rows(source, width)
    return slice_markdown(source, align(source, rows), 0, len(rows) - 1)


@pytest.mark.parametrize("width", [40, 44, 64, 78, 120])
def test_full_message_slices_to_its_source(width: int) -> None:
    """A whole-message copy is the message's markdown, at any width.

    The wrap changes with the width; the source must not. This is the property
    that makes the copy independent of how the frame happened to fold.
    """
    copied = _full_slice(MIXED, width)
    assert copied.split() == MIXED.strip().split()  # same words, same order
    assert "- first bullet" in copied and "- second bullet" in copied
    assert "> a quoted line" in copied and "> second quoted source line" in copied
    assert "```python" in copied and "```" in copied
    assert "▌" not in copied and "•" not in copied


def test_a_quote_slice_is_requoted() -> None:
    """Selecting only the quote pastes a valid blockquote, not bare lines."""
    rows = _rows(MIXED, 44)
    mapping = align(MIXED, rows)
    quote_rows = [i for i, r in enumerate(rows) if r.strip().startswith("▌")]
    copied = slice_markdown(MIXED, mapping, quote_rows[0], quote_rows[-1])
    assert copied == (
        "> a quoted line that is also long enough to wrap around onto more than one row\n"
        "> second quoted source line"
    )


def test_a_mid_fence_slice_is_refenced() -> None:
    """Selecting the code rows re-wraps them in the fence, language kept."""
    rows = _rows(MIXED, 44)
    mapping = align(MIXED, rows)
    code_rows = [i for i, r in enumerate(rows) if r.strip() in {"def f(x):", "return x"}]
    copied = slice_markdown(MIXED, mapping, code_rows[0], code_rows[-1])
    assert copied == "```python\ndef f(x):\n    return x\n```"


def test_bold_inside_a_quote_keeps_its_markers() -> None:
    """Inline markup survives the round trip through the frame."""
    source = (
        "A reply:\n\n"
        "> Thanks for the report. I verified the **flagged** value in out/main/index.jsc:\n"
        "> it's the public project API key (phc_...), publishable by design.\n"
    )
    copied = _full_slice(source, 76)
    assert "**flagged**" in copied
    assert copied.count("> ") >= 2  # both quote lines, re-prefixed


# -- review-round regressions (F1–F5) -----------------------------------------
def test_a_paragraph_sharing_a_quotes_first_word_is_not_swallowed() -> None:
    """F1a: the earliest matching source line wins, so a paragraph anchors to
    itself, not to a later quote that happens to share its first word."""
    source = "See the docs below.\n\n> See the quoted reply line one that wraps a bit"
    copied = _full_slice(source, 40)
    assert "See the docs below." in copied
    assert "> See the quoted reply line" in copied


def test_a_reference_link_definition_survives_a_full_copy() -> None:
    """F1b: a link definition renders nothing, so it never anchors — but a copy
    that reaches the end of the message must still carry it."""
    source = "See [the docs][docs] now.\n\n[docs]: https://example.com"
    copied = _full_slice(source, 40)
    assert "[docs]: https://example.com" in copied


def test_a_heading_after_a_quote_is_not_requoted() -> None:
    """F1c: the quote prefix applies to quote lines only, not to a heading or
    paragraph that follows the quote run."""
    source = "> quote line\n\n## Notes\n\nbody text"
    copied = _full_slice(source, 40)
    assert "## Notes" in copied and "> ## Notes" not in copied
    assert "body text" in copied and "> body text" not in copied


def test_a_fence_with_a_blank_line_stays_balanced() -> None:
    """F2: the closing fence is appended only when the slice does not already
    include the fence's own closing marker — never after a trailing paragraph."""
    source = "```\nfirst\n\nsecond\n```\n\nafter para"
    copied = _full_slice(source, 40)
    assert copied.count("```") == 2
    assert copied.rstrip().endswith("after para")


def test_an_ordered_list_copies_with_its_markers() -> None:
    """F3: Rich paints an ordered marker as a bare number (`` 1 alpha``), which
    must not be read as the item's content word — the items anchor and copy."""
    assert _full_slice("1. alpha\n2. beta\n3. gamma", 40) == "1. alpha\n2. beta\n3. gamma"
    mixed = _full_slice("1. first\n2. second\n   - sub a\n   - sub b\n3. third", 40)
    assert "1. first" in mixed and "3. third" in mixed


def test_a_truncated_table_cell_still_anchors() -> None:
    """F4: Rich truncates a long cell with ``…``; the row anchors on the stem."""
    long_cell = "x" * 80
    source = f"| name | value |\n|------|-------|\n| {long_cell} | 1 |"
    rows = _rows(source, 50)
    mapping = align(source, rows)
    body = [i for i, r in enumerate(rows) if "…" in r or "xxx" in r]
    assert body, "no truncated row found to test"
    copied = slice_markdown(source, mapping, body[0], body[-1])
    assert long_cell in copied
