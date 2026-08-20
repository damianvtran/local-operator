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
