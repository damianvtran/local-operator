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


# -- review-round-2 regressions (G1–G3) ---------------------------------------
def test_a_number_led_paragraph_is_not_read_as_a_list_marker() -> None:
    """G1: ``2026 roadmap`` glues the digits to the content (no marker space),
    so the paragraph anchors and copies — it is not an ordered-list marker."""
    assert _full_slice("2026 roadmap items are here", 40) == "2026 roadmap items are here"


def test_an_item_whose_content_starts_with_a_number_copies() -> None:
    """G2: ``- 3 ways`` / ``> 3 ways`` begin their content with a number; the
    number is content, not a marker, so the items anchor and copy."""
    assert _full_slice("- 3 ways to fix it\n- 4 more things", 40) == (
        "- 3 ways to fix it\n- 4 more things"
    )
    copied = _full_slice("> 3 ways forward\n> 5 steps back", 40)
    assert "3 ways forward" in copied and "5 steps back" in copied


def test_a_nested_quote_keeps_its_levels() -> None:
    """G3: each quote line keeps its own ``>``/``>>`` prefix from the source —
    re-applying a single prefix would flatten the nesting."""
    assert _full_slice("> outer\n>> inner\n> outer again", 40) == (
        "> outer\n>> inner\n> outer again"
    )


# -- review-round-3 regression (H1) -------------------------------------------
@pytest.mark.parametrize(
    "source",
    [
        "100. item hundred\n101. item hundred one",  # three-digit markers
        "1. alpha\n2. beta",  # one-digit markers
        "2026 roadmap",  # number-led paragraph (flush left)
        "42",  # a paragraph that is only a number
        "- 7",  # an item that is only a number
        "- 3 ways to fix",  # an item whose content starts with a number
        "1. first\n\n2026 was the year",  # ordered list, then a number-led paragraph
    ],
)
def test_number_markers_and_number_led_content_both_anchor(source: str) -> None:
    """H1: the marker/content discriminator is the row's INDENT, not the number's
    length — Rich indents a real ordered marker (`` 100 item``) at any width and
    leaves a number-led paragraph flush left (``2026 roadmap``)."""
    copied = _full_slice(source, 40)
    assert copied.split() == source.strip().split()


# -- agent review round 3 regression (R3-1) -----------------------------------
@pytest.mark.parametrize("width", [28, 42, 60, 80, 100, 120])
def test_a_table_row_is_placed_only_by_a_cell_it_opens(width: int) -> None:
    """R3-1: the first-cell match is an OPENER test, never containment.

    Round 1 narrowed the haystack from the whole source line to the first cell
    but kept ``in``. A first cell is short, so a word from one cell is easily a
    substring of a DIFFERENT row's cell — ``'at'`` inside ``api-gateway`` via
    ``g-at-eway`` — and the row that matched was copied whole, which is issue
    #399's payload arriving through the word path instead of the whole-line one.

    Asserted as a MAPPING property rather than through a clipboard string
    because that is what actually broke: a continuation carrying ``at narrow
    widths`` must stay UNPLACED (the copy path reads ``None`` as "assume
    nothing" and answers with the lit glyphs), and it must never be placed on
    the ``api-gateway`` line two rows below it.
    """
    source = (
        "| service | notes |\n| --- | --- |\n"
        "| api | the gateway service whose note is long enough to wrap at narrow widths |\n"
        "| api-gateway | the second service, also with a note that folds across rows |"
    )
    lines = source.split("\n")
    api_line = lines.index(
        "| api | the gateway service whose note is long enough " "to wrap at narrow widths |"
    )
    gateway_line = lines.index(
        "| api-gateway | the second service, also with a note " "that folds across rows |"
    )

    rows = _rows(source, width)
    mapping = align(source, rows)
    for row, placed in zip(rows, mapping):
        if placed != gateway_line:
            continue
        # Only a row that actually paints the ``api-gateway`` cell may claim it.
        assert row.lstrip().startswith("api-gateway"), (
            f"width {width}: row {row!r} was placed on the api-gateway line it "
            f"does not open — a different row's markdown would be copied for it."
        )
    # The ``api`` row itself must still be placed; the fix must not refuse rows.
    assert api_line in mapping, f"width {width}: the api row lost its source line"


@pytest.mark.parametrize("width", [28, 42, 60, 80, 100, 120])
def test_a_numeric_table_header_never_claims_a_body_row(width: int) -> None:
    """R3-1: a numeric header cannot match its own line, and must match nothing.

    ``_row_word`` reads a leading bare number as an ordered-list marker, so the
    header of ``| 1 | 10 |`` takes its row word from the SECOND cell — ``10`` —
    which matches no first cell on its own line. Under containment its only
    possible match was a lookahead into a body row (``'10' in '100'``), so the
    reader who lit the header was handed the first body row's markdown.

    ``_number_opens_row`` cannot rescue this and correctly does not try:
    ``table_column`` is still ``None`` on the first row of the table (R3-2), so
    it refuses. Refusing is the safe answer; the branch beside it filling the
    gap with a wrong row is not.
    """
    source = (
        "| 1 | 10 |\n| --- | --- |\n"
        "| 100 | the first ranked entry whose note is long enough to wrap at widths |\n"
        "| 101 | the second ranked entry, also with a note that folds across rows |"
    )
    lines = source.split("\n")
    body = [i for i, line in enumerate(lines) if line.startswith(("| 100 ", "| 101 "))]

    rows = _rows(source, width)
    mapping = align(source, rows)
    for row, placed in zip(rows, mapping):
        if placed not in body:
            continue
        head = lines[placed].split("|")[1].strip()
        assert row.lstrip().startswith(head), (
            f"width {width}: row {row!r} was placed on {lines[placed]!r}, a body row "
            f"it does not open — the header would copy a body row's markdown."
        )


@pytest.fixture
def _markers_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn `display.heading_markers` ON for one test.

    The flag defaults OFF — the colour ramp gives six distinct levels on its
    own — but the alignment must keep working for the users who enable it,
    which is precisely the case that regressed once already.
    """
    import local_operator.tui.markdown_theme as _mt

    real = _mt.settings_get
    monkeypatch.setattr(
        _mt,
        "settings_get",
        lambda key, default=None: (
            True if key == "display.heading_markers" else real(key, default)
        ),
    )


@pytest.mark.usefixtures("_markers_on")
@pytest.mark.parametrize("width", [40, 60, 80])
def test_headings_copy_with_their_markers(width: int) -> None:
    """Every heading level survives a copy, marker intact.

    ``markdown_theme._flat_heading`` paints the ``#`` markers rich strips at
    parse time, so a heading row arrives here as ``## Section`` while its
    source line is also ``## Section``. Until ``_RENDERED_PREFIX_RE`` and
    ``_MARKER_TOKEN_RE`` knew about the marker, the rendered side anchored on
    ``#`` and the source side on ``section``: the two never matched, the row
    was left unplaced, and EVERY heading vanished from the copied markdown.
    """
    source = (
        "# Title\n\n"
        "Some prose that will wrap across a couple of rendered rows here.\n\n"
        "## Section\n\n"
        "- item one\n\n"
        "### Deep\n\n"
        "#### Deeper\n\n"
        "##### Deepest\n\n"
        "###### Deepest still\n"
    )
    rows = _rows(source, width)
    mapping = align(source, rows)
    copied = slice_markdown(source, mapping, 0, len(rows) - 1)

    for heading in (
        "# Title",
        "## Section",
        "### Deep",
        "#### Deeper",
        "##### Deepest",
        "###### Deepest still",
    ):
        assert heading in copied, (
            f"width {width}: {heading!r} was dropped from the copied markdown; "
            "the heading row failed to anchor to its source line."
        )


@pytest.mark.usefixtures("_markers_on")
def test_each_heading_level_paints_its_own_marker_width() -> None:
    """``h3`` paints ``###``, not ``#``.

    The level is parsed off the ``hN`` tag. Deriving it from the tag's LENGTH
    instead (``len("h2") - 1`` is 1, not 2) silently painted a single ``#`` at
    every level, which is the one failure mode that defeats the whole point of
    the marker: six levels rendered as one. The alignment tests above still
    passed, because a uniform marker anchors just as well as a correct one.
    """
    source = "# One\n\n## Two\n\n### Three\n\n#### Four\n\n##### Five\n\n###### Six\n"
    rows = [row for row in _rows(source, 60) if row.strip()]
    markers = [row.split(" ", 1)[0] for row in rows]
    assert markers == [
        "#",
        "##",
        "###",
        "####",
        "#####",
        "######",
    ], f"heading markers did not scale with level: {markers}"


def test_headings_copy_cleanly_with_markers_off() -> None:
    """The DEFAULT path: no markers rendered, headings still copy as markdown.

    `display.heading_markers` defaults OFF, so the rendered row is bare text
    while its source line still carries `##`. The alignment has to hold in
    BOTH directions — the marker-aware strip added for the on-case must not
    break the off-case, which is the one every user gets.
    """
    source = (
        "# Title\n\n"
        "Some prose that will wrap across a couple of rendered rows here.\n\n"
        "## Section\n\n"
        "- item one\n\n"
        "### Deep\n"
    )
    rows = _rows(source, 60)
    assert not any(
        row.lstrip().startswith("#") for row in rows
    ), f"markers rendered while the flag is off: {rows}"
    copied = slice_markdown(source, align(source, rows), 0, len(rows) - 1)
    for heading in ("# Title", "## Section", "### Deep"):
        assert heading in copied, f"{heading!r} was dropped from the copied markdown"
