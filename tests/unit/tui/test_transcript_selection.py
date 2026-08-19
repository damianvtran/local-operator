"""Drag-selection and COPY, per block type (TUI-021).

Reported from the field: "highlighting of the user message works but I can't
seem to highlight the agent messages which is important to be able to
copy/paste agent content". The prompt was selectable and the model's own output
was not, which is the wrong way round — a user copies the answer, not the
question.

The mechanism is entirely in Textual 8.2.8 and is worth stating here, because
this app has twice reasoned about it from memory and been wrong:

* **The highlight** is applied by ``Content.render_strips``
  (``textual/content.py``), which forwards ``options.selection`` into
  ``_wrap_and_format`` and stylizes each row's span. ``RichVisual.render_strips``
  (``textual/visual.py``) never reads ``options.selection`` at all, so a widget
  whose visual is a Rich renderable paints no band however hard you drag.
* **The clipboard** is ``Widget.get_selection`` (``textual/widget.py``), whose
  default body is::

      visual = self._render()
      if isinstance(visual, (Text, Content)):
          text = str(visual)
      else:
          return None

  So a non-``Content`` visual returns ``None``, and ``Screen.get_selected_text``
  drops it from the join.
* **Which visual a widget gets** is decided by ``visualize()``
  (``textual/visual.py``): ``str`` and ``rich.text.Text`` are promoted to a
  ``Content``; EVERY other Rich renderable is wrapped in a ``RichVisual``.

A ``Markdown`` is therefore unselectable and uncopyable by construction, and
that — not a CSS flag, not ``ALLOW_SELECT`` — was the bug. ``AssistantBlock``
now FLATTENS its markdown to one styled ``Text`` before applying it
(``assistant.flatten``), which is the whole fix.

**What a copy produces**, decided once and asserted here: THE GLYPHS THAT WERE
HIGHLIGHTED, minus each row's own gutter and its trailing pad. Not the source
markup. The reader pointed at a rendered frame, and a paste that put ``**``
back around a word they saw in bold would be a different document. Two
consequences worth naming rather than discovering:

* a fenced code block copies as THE CODE — no fence markers, no decoration,
  which is what someone pasting a snippet wants and is only true because
  ``IslandCodeBlock`` renders a bare ``Syntax`` with no slab;
* a link copies as its LABEL. The URL is not on the row — Rich puts it in an
  OSC-8 hyperlink, which stays clickable on the frame (asserted below) but is
  not text. Appending it to the clipboard would put characters in the paste
  that no highlighted cell contained, and the highlight and the clipboard are
  computed from the same rows by the same ``Selection.get_span`` precisely so
  they cannot disagree.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from rich.cells import cell_len
from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text
from textual import events
from textual.content import Content
from textual.geometry import Offset
from textual.selection import Selection
from textual.visual import RichVisual

from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.glyphs import tool_icon
from local_operator.tui.markdown_theme import install_markdown_theme
from local_operator.tui.widgets.assistant import AssistantBlock, flatten
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.toast import Toast
from local_operator.tui.widgets.tool_card import OUTPUT_INDENT, ToolCard
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    RichBlock,
    TranscriptBlock,
    TranscriptView,
    UserBlock,
)
from local_operator.tui.widgets.welcome import WelcomeView
from tests.unit.tui.conftest import StyledTranscriptApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: The message every markdown assertion below is made against: bold, inline
#: code, a link, an ordered list and a fenced block, which are exactly the five
#: constructs whose rendered form differs from their source.
MARKDOWN = """Here is the **plan** with `inline_code` and a [link](https://example.com/x).

1. first step
2. second step

```python
def f(x):
    return x + 1
```

Done."""

#: The frame that markdown paints at 60 cells — and, because a copy is the
#: frame, also the clipboard. Written out rather than derived so a change to
#: either the rendering or the copy rule has to be stated here.
MARKDOWN_ROWS = [
    "Here is the plan with inline_code and a link.",
    "",
    " 1 first step",
    " 2 second step",
    "",
    "def f(x):",
    "    return x + 1",
    "",
    "Done.",
]


@pytest.fixture(autouse=True)
def _brand_markdown() -> None:
    """The shipped markdown element table, which the copy rule depends on.

    ``IslandCodeBlock`` is what makes a fenced block render as bare code; under
    Rich's default ``CodeBlock`` the same selection would copy a Monokai slab
    padded out to the full width.
    """
    install_markdown_theme()


def _copy_all(app: StyledTranscriptApp, widget: TranscriptBlock) -> str | None:
    """The clipboard after selecting the whole of ``widget`` and copying.

    Goes through ``Screen.get_selected_text`` rather than calling
    ``get_selection`` directly, so the assertions cover the path the ``copy``
    binding actually takes — including the ``None``-means-drop filtering that
    decides whether a block contributes to a multi-block copy at all.
    """
    app.screen.selections = {widget: Selection(None, None)}
    return app.screen.get_selected_text()


async def _mounted(app: StyledTranscriptApp, *blocks: TranscriptBlock) -> TranscriptView:
    view = app.query_one(TranscriptView)
    for block in blocks:
        view.append_block(block)
    return view


# -- the mechanism -----------------------------------------------------------
def test_rich_renderable_cannot_highlight_and_a_text_can() -> None:
    """The Textual 8.2.8 gate itself, so the fix rests on a checked fact.

    Both halves in one test because they are one claim: ``visualize`` sorts by
    TYPE, and everything else follows from which side of that sort a block
    lands on.
    """
    assert RichVisual.render_strips.__doc__ is not None
    # The Rich path ignores the selection argument; the Content path forwards it.
    import inspect

    from textual.content import Content as ContentVisual

    rich_source = inspect.getsource(RichVisual.render_strips)
    content_source = inspect.getsource(ContentVisual.render_strips)
    assert "selection" not in rich_source
    assert "selection=options.selection" in content_source


@pytest.mark.asyncio
async def test_selectable_visual_per_block_type() -> None:
    """Which blocks can be highlighted AND copied, stated per type.

    The four that carry text a user would paste all reach the screen as a
    ``Content``. ``RichBlock`` deliberately does not — see
    :func:`test_rich_block_is_not_selectable`.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        user = UserBlock("summarise the ingest path")
        assistant = AssistantBlock()
        notice = NoticeBlock("resume with: local-operator --resume abc123", "note")
        card = ToolCard("t1", "bash", {"command": "pytest -q"})
        await _mounted(app, user, assistant, notice, card)
        assistant.update_text(MARKDOWN)
        assistant.finalize_text()
        card.mark_failed("assert 1 == 2", "E   assert 1 == 2\nFAILED tests/x.py")
        card.toggle_expanded()
        await pilot.pause()

        for block in (user, assistant, notice, card):
            assert isinstance(block._render(), Content), type(block).__name__
            assert block.allow_select is True, type(block).__name__
            assert _copy_all(app, block), type(block).__name__


@pytest.mark.asyncio
async def test_rich_block_is_not_selectable() -> None:
    """``RichBlock`` stays uncopyable, and that is the intended state.

    It wraps its renderable in a ``rich.padding.Padding`` — a Rich renderable,
    so ``RichVisual``, so no band and no clipboard. Left that way on purpose:
    it carries APP-authored structure (``/help`` columns, listings), not model
    or user text, and the renderables it is handed include tables whose flatten
    would put box-drawing characters on the clipboard — the "decorated
    rendering" this whole rule refuses. Making it selectable also means moving
    ``flatten`` out of ``assistant`` to break the import cycle, which is a
    change to make when there is a reason to copy a ``/help`` table.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        block = RichBlock("plain rich renderable")
        await _mounted(app, block)
        await pilot.pause()
        assert isinstance(block._render(), RichVisual)
        assert block.get_selection(Selection(None, None)) is None


# -- markdown: what actually lands on the clipboard --------------------------
@pytest.mark.asyncio
async def test_markdown_copies_as_the_rendered_frame() -> None:
    """The whole assistant message, copied — the acceptance case.

    Every construct at once, because the interesting thing is that they are
    consistent: bold and inline code lose their markers, the fence loses its
    backticks and keeps its indentation, the ordered list keeps its rendered
    marker, and the link keeps its label.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(MARKDOWN)
        block.finalize_text()
        await pilot.pause()

        copied = _copy_all(app, block)
        assert copied is not None, "the agent message contributed nothing to the copy"
        assert copied == "\n".join(MARKDOWN_ROWS)
        # The five constructs, called out so a failure says WHICH one moved.
        assert "**" not in copied and "plan" in copied  # bold
        assert "`" not in copied and "inline_code" in copied  # inline code
        assert "```" not in copied  # fence markers
        assert "def f(x):\n    return x + 1" in copied  # the code, verbatim
        assert "https://example.com/x" not in copied  # the URL is not text
        assert "link" in copied  # its label is


@pytest.mark.asyncio
async def test_link_url_stays_on_the_frame_as_a_hyperlink() -> None:
    """The URL is not copied because it is not TEXT — but it is not lost.

    The flatten carries Rich's OSC-8 hyperlink through onto the painted
    segment, so the link is still clickable in a terminal that supports it.
    This is the half of the trade that makes "copy the label" defensible rather
    than lossy, so it is pinned.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(MARKDOWN)
        block.finalize_text()
        await pilot.pause()
        block._render_content()
        links = {
            segment.style.link
            for strip in block._render_cache.lines
            for segment in strip._segments
            if segment.style and segment.style.link
        }
        assert links == {"https://example.com/x"}


@pytest.mark.asyncio
async def test_fenced_code_copies_without_the_pad() -> None:
    """A code fence pastes as runnable code, trailing pad trimmed.

    Rich pads every rendered row out to the full width, and ``Syntax`` paints
    its ground across all of it. Without the right-trim in
    ``TranscriptBlock.get_selection`` a two-line snippet arrives with fifty
    spaces welded to each line.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text("```python\nx = 1\n```")
        block.finalize_text()
        await pilot.pause()
        assert _copy_all(app, block) == "x = 1"


# -- the other blocks --------------------------------------------------------
@pytest.mark.asyncio
async def test_user_block_copy_carries_no_gutter_glyph() -> None:
    """The ``▌`` never reaches the clipboard — the regression this guards.

    A gutter column is exactly the kind of thing that silently ends up in a
    paste, and the block's own case for existing is the PASTED SNIPPET, where a
    rule welded to the front of every line is the difference between code and
    noise. Indentation the user authored is kept, which is the same reason.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        block = UserBlock("summarise the ingest path\n\n    def f(x):\n        return x")
        await _mounted(app, block)
        await pilot.pause()
        copied = _copy_all(app, block)
        assert copied is not None
        assert UserBlock.RULE not in copied
        assert copied == "summarise the ingest path\n\n    def f(x):\n        return x"


@pytest.mark.asyncio
async def test_notice_copy_is_the_sentence_not_the_glyph_field() -> None:
    """A notice pastes as its text: no spine indent, no kind glyph.

    Both rows of a wrapped notice, because the glyph field is a FIXED width
    that continuation rows fill with blanks — so the uniform ``GLYPH_COLS``
    count is only correct if the hanging indent really is the same width.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(40, 40)) as pilot:
        block = NoticeBlock("ctrl+c again to exit - resume with: session-abc123", "warning")
        await _mounted(app, block)
        await pilot.pause()
        copied = _copy_all(app, block)
        assert copied is not None
        assert copied.startswith("ctrl+c again to exit")
        assert "·" not in copied and "⚠" not in copied
        for row in copied.split("\n"):
            assert row == row.lstrip(" "), f"row carries the glyph field: {row!r}"


@pytest.mark.asyncio
async def test_tool_card_expanded_output_copies_unindented() -> None:
    """An expanded card's stderr pastes straight into a bug report.

    Both gutters at once: the summary row loses its Nerd Font icon field — a
    private-use codepoint that pastes as a replacement box and restates nothing
    the tool NAME beside it does not — and every row below loses the card's own
    ``OUTPUT_INDENT``. The receipt itself (name, what ran, outcome, duration)
    stays, because that is what a user selecting a settled row is after.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        card = ToolCard("t1", "bash", {"command": "pytest -q"})
        await _mounted(app, card)
        card.mark_failed("assert 1 == 2", "E   assert 1 == 2\nFAILED tests/x.py")
        assert card.toggle_expanded() is True
        await pilot.pause()
        copied = _copy_all(app, card)
        assert copied is not None
        rows = copied.split("\n")
        assert "E   assert 1 == 2" in rows
        assert "FAILED tests/x.py" in rows
        assert rows[0].startswith("bash")  # no icon, no leading pad
        assert tool_icon("bash") not in copied
        assert "✗" in rows[0] and "pytest -q" in rows[0]  # the receipt survives
        assert card.copy_gutter(0) == ToolCard.ICON_COLS
        assert card.copy_gutter(1) == OUTPUT_INDENT


# -- highlight and clipboard cannot disagree ---------------------------------
@pytest.mark.asyncio
async def test_partial_selection_copies_only_the_highlighted_rows() -> None:
    """A drag over rows 5-6 copies rows 5-6, clipped at the drag's columns.

    The same ``Selection.get_span`` the highlight uses, so this is the
    invariant rather than a second implementation of the arithmetic.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(MARKDOWN)
        block.finalize_text()
        await pilot.pause()
        # From the start of the fence's first row to mid-way through its second.
        selection = Selection.from_offsets(Offset(x=0, y=5), Offset(x=8, y=6))
        assert block.get_selection(selection) == ("def f(x):\n    retu", "\n")


@pytest.mark.asyncio
async def test_a_drag_starting_inside_the_gutter_still_copies_from_the_prose() -> None:
    """Clamped, not clipped: the gutter is never content, wherever the drag began.

    The reader can and does start a drag on the rule itself. Returning a COUNT
    from ``copy_gutter`` rather than a prefix string is what lets the span
    start be lifted to the first prose cell instead of copying half a bar.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        block = UserBlock("summarise the ingest path")
        await _mounted(app, block)
        await pilot.pause()
        selection = Selection.from_offsets(Offset(x=0, y=0), Offset(x=11, y=0))
        assert block.get_selection(selection) == ("summarise", "\n")


# -- the flatten's own claims ------------------------------------------------
@pytest.mark.parametrize("width", [40, 60, 80])
@pytest.mark.parametrize(
    "prefix,tail",
    [
        ("First paragraph here.\n\n", "Second paragraph tail."),
        ("Intro line.\n\n", "```python\ndef f():\n    pass\n```"),
        ("```sh\nls -la\n```\n\n", "After the fence."),
        ("- one\n- two\n\n", "Trailing prose."),
        ("# Title\n\n", "Body text under it."),
    ],
)
def test_flatten_splices_exactly(prefix: str, tail: str, width: int) -> None:
    """``flatten(prefix) + "\\n" + flatten(tail) == flatten(Group(...))``.

    The frozen-prefix cache is only sound because of this. Rich ends every
    renderable with a newline, so joining two flattened halves with one newline
    reproduces the grouped render — which is what lets a settled prefix be
    cached as TEXT (rendered once) instead of as a ``Markdown`` the compositor
    re-rendered on every repaint.
    """
    grouped = flatten(Group(Markdown(prefix), Markdown(tail)), width).plain
    joined = flatten(Markdown(prefix), width).plain + "\n" + flatten(Markdown(tail), width).plain
    assert grouped == joined


def test_flatten_drops_richs_trailing_newline() -> None:
    """No phantom last row: Rich closes a block with a newline, this does not.

    Kept, it is a blank row the markdown never had and a row of height the
    block would reserve and never paint.
    """
    text = flatten(Markdown("one line"), 40)
    assert not text.plain.endswith("\n")
    assert text.plain.count("\n") == 0


def test_flatten_is_a_text_so_the_block_is_selectable() -> None:
    """The type is the entire point, so assert the type."""
    assert isinstance(flatten(Markdown("**bold**"), 40), Text)


# -- the costs the flatten takes on --------------------------------------------
@pytest.mark.asyncio
async def test_frame_is_unchanged_by_the_flatten() -> None:
    """The rows Textual paints are the rows the ``Markdown`` painted.

    The flatten buys selection; it must not move a cell. Both paths walk the
    same segment stream from the same console, and this asserts they land on
    the same text — the claim ``AssistantBlock``'s docstring makes.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        flat = AssistantBlock()
        raw = TranscriptBlock()
        await _mounted(app, flat, raw)
        flat.update_text(MARKDOWN)
        flat.finalize_text()
        raw.set_content(Markdown(MARKDOWN))
        # The comparison block is given the SAME height, so both render the same
        # number of strips; `flat` pins its own and `raw` would be measured.
        # Read from the PIN, not from `size`: the pin is written synchronously by
        # `_apply_rows`, whereas `size` is zero until the layout has run.
        pinned = flat.styles.height
        assert pinned is not None, "the flatten did not pin a height"
        raw.styles.height = pinned.value
        await pilot.pause()

        flat._render_content()
        raw._render_content()
        # Compared with each row's trailing pad removed, which is what "must not
        # move a cell" actually means. The flatten pads a BLANK row out to the
        # block width where Rich emits nothing at all: visually identical (both
        # are background), but not byte-identical, and it is load-bearing — a
        # zero-cell row takes no selection style, so a multi-paragraph answer
        # highlighted as disconnected slabs while `get_selection` returned one
        # continuous string (design round 12, D2). Right-stripping keeps the
        # real invariant: every painted glyph in the same place on the same row.
        assert [strip.text.rstrip() for strip in flat._render_cache.lines] == [
            strip.text.rstrip() for strip in raw._render_cache.lines
        ]


@pytest.mark.asyncio
async def test_the_highlight_over_a_blank_row_is_not_a_hole() -> None:
    """A selection spanning paragraphs paints one band, not a stack of slabs.

    Design round 12, D2. Rich pads a row that has CONTENT out to the block
    width but emits a row that has none as nothing at all, and the selection
    style is applied to the cells a row HAS — so the blank rows between
    paragraphs took 0 cells of highlight while ``get_selection`` returned one
    continuous string. The highlight has to describe what gets copied, and a
    reader dragging over a four-paragraph answer saw four disconnected blocks.

    Asserted on the CONTENT's rows rather than the painted strips, because
    that is where the selection style is applied — the compositor pads the
    strip afterwards either way, which is exactly why this was invisible until
    someone measured the highlight itself.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text("First paragraph.\n\nSecond paragraph.\n\nThird.")
        block.finalize_text()
        await pilot.pause()

        rendered = block.renderable
        assert isinstance(rendered, Text), "the flatten did not produce a Text"
        rows = rendered.plain.split("\n")
        blanks = [row for row in rows if not row.strip()]
        assert blanks, "no blank row between the paragraphs, so this proves nothing"
        assert all(
            cell_len(row) == cell_len(rows[0]) for row in rows
        ), f"a row cannot take the selection style: {[cell_len(r) for r in rows]}"


@pytest.mark.asyncio
async def test_height_is_pinned_to_the_row_count() -> None:
    """A block that authors its own rows KNOWS its height.

    Pinned for the reason ``UserBlock._build`` records: Textual caches its
    content-height measurement on the WIDTH alone, and the first measurement is
    taken of the fallback-width build — so ``auto`` reserves the inflated count
    and paints a hole under a message that was rebuilt narrower.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(MARKDOWN)
        block.finalize_text()
        await pilot.pause()
        pinned = block.styles.height
        assert pinned is not None, "height left to auto — the block is being MEASURED"
        assert pinned.value == len(MARKDOWN_ROWS)
        assert block.size.height == len(MARKDOWN_ROWS)


@pytest.mark.asyncio
async def test_resize_rebuilds_at_the_new_width() -> None:
    """The flatten bakes the width in, so a resize is a content change.

    A ``Markdown`` re-folded itself per repaint and needed telling nothing; a
    ``Text`` carries the fold it was built with. Asserted on a paragraph long
    enough that the two widths cannot produce the same row count.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(80, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(" ".join(["ingest"] * 40))
        block.finalize_text()
        await pilot.pause()
        wide = block.size.height

        await pilot.resize_terminal(40, 40)
        await pilot.pause()
        await pilot.pause()
        narrow = block.size.height

        assert narrow > wide
        # The PIN moved too, not just the layout: a stale pin is the exact
        # failure mode — rows rebuilt at the new width, height still reserving
        # the old count.
        repinned = block.styles.height
        assert repinned is not None and repinned.value == narrow
        copied = _copy_all(app, block)
        assert copied is not None
        # Re-wrapped, but still the same words in the same order.
        assert copied.split() == ["ingest"] * 40


@pytest.mark.asyncio
async def test_streaming_copy_is_the_streaming_frame() -> None:
    """A mid-stream copy is the mid-stream FRAME, splice and all.

    The live block is a spliced frozen prefix plus a freshly flattened tail,
    and the settled one is a single flatten of the whole text. Those two do NOT
    produce identical rows, and that difference PREDATES the flatten: the old
    ``Group(Markdown(prefix), Markdown(tail))`` splice dropped the blank row
    markdown puts between two block elements in exactly the same place.
    Measured on this message at 64 columns, the frozen prefix ends
    ``"```\\n\\n"`` and the tail is ``"Done."``, so the live frame runs
    ``return x + 1`` straight into ``Done.`` where the settled frame separates
    them — one row that appears when the turn settles.

    Asserted rather than fixed because the copy rule is "the glyphs that were
    highlighted", so a copy taken mid-stream SHOULD be the mid-stream frame,
    and closing the gap means changing the boundary semantics that TUI-010 and
    TUI-011 pin. What must hold, and does, is that the words and their order
    never depend on when the copy was taken.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        for cut in range(20, len(MARKDOWN), 25):
            block.update_text(MARKDOWN[:cut])
        block.update_text(MARKDOWN)
        await pilot.pause()
        assert block._frozen_text, "no prefix ever froze — the splice is untested"
        live = _copy_all(app, block)
        assert live is not None

        block.finalize_text()
        await pilot.pause()
        settled = _copy_all(app, block)
        assert settled is not None

        assert live.split() == settled.split()  # same words, same order
        assert [row for row in live.split("\n") if row] == [
            row for row in settled.split("\n") if row
        ]
        # The pre-existing difference, named so a change to it is deliberate.
        assert live.count("\n\n") == settled.count("\n\n") - 1


def test_the_splice_reproduces_the_group_it_replaced() -> None:
    """The flatten changed the TYPE Textual sees and nothing else.

    The claim ``AssistantBlock`` rests on: the spliced live render is the frame
    the old ``Group`` painted, byte for byte, so selection was bought without
    moving a cell. Built detached, which is also the path
    ``TranscriptBlock.set_content`` exists to keep working.
    """
    block = AssistantBlock()
    for cut in range(20, len(MARKDOWN), 25):
        block.update_text(MARKDOWN[:cut])
    block.update_text(MARKDOWN)
    prefix = block._frozen_text
    assert prefix, "no prefix ever froze — the splice is untested"
    tail = MARKDOWN[len(prefix) :]
    was_group = flatten(Group(Markdown(prefix), Markdown(tail)), 60)
    assert block._flat_rows(60).plain == was_group.plain


@pytest.mark.asyncio
async def test_settled_rows_counts_the_flattened_prefix() -> None:
    """Row accounting comes from the flatten, not a second render.

    The frozen prefix is already laid out at the block's own width, so the
    count is the count the compositor will paint — which a re-measure at a
    guessed width was not.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text("frozen prefix\n\nlive tail")
        await pilot.pause()
        assert block._frozen_flat is not None
        assert block.settled_rows() == block._frozen_flat.plain.count("\n") + 1


# -- the band's own ink ------------------------------------------------------
@pytest.mark.asyncio
async def test_selection_band_is_the_brand_step_not_textuals_blue() -> None:
    """The highlight paints ``$lo-faint`` under ``$lo-fg``.

    Textual's default is ``textual-dark``'s ``ansi_bright_blue`` on
    ``ansi_black`` — the only saturated blue in the app, landing on prose the
    reader is mid-gesture over. Asserted from the painted STRIPS rather than
    from the stylesheet, because "the rule exists" and "the rule reached the
    cells" are different claims and only the second one is the feature.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text("select me")
        block.finalize_text()
        await pilot.pause()

        app.screen.selections = {block: Selection(None, None)}
        block.refresh()
        await pilot.pause()
        block._render_content()

        band = theme_mod.semantic_color("faint").lower()
        ink = theme_mod.semantic_color("fg").lower()
        painted: set[tuple[str, str]] = set()
        for strip in block._render_cache.lines:
            for segment in strip._segments:
                if not segment.text.strip():
                    continue
                style = segment.style
                # Asserted rather than filtered: a painted cell with no colour
                # is the band FAILING to reach it, which is the whole feature,
                # and a comprehension that skipped it would pass vacuously.
                assert style is not None, f"no style at all on {segment.text!r}"
                colour, ground = style.color, style.bgcolor
                assert colour is not None, f"no ink on {segment.text!r}"
                assert ground is not None, f"the band never painted {segment.text!r}"
                painted.add(
                    (colour.get_truecolor().hex.lower(), ground.get_truecolor().hex.lower())
                )
        assert painted == {(ink, band)}


# -- the copy itself ---------------------------------------------------------
#
# Reported from the field after the highlight shipped: "while I can highlight
# text from agent messages, pressing cmd+C to copy does not copy to clipboard".
# It cannot: Ghostty binds ``super+c=copy_to_clipboard:mixed`` WITHOUT
# ``performable:``, so cmd+C is eaten by the terminal, which then copies its
# own (empty) selection — the app never sees the key. Ctrl+C is the interrupt.
# With no key left to bind, the RELEASE is the copy, and these pin that path
# through the real app: the mouse events a terminal actually sends, the
# clipboard Textual actually writes, and the toast that is the only reason a
# working copy is distinguishable from the broken one that was reported.
def _mouse(app: OperatorApp, kind: type, x: int, y: int) -> Any:
    """One SGR mouse report, as ``Screen._forward_event`` receives it."""
    return kind(
        app.screen,
        x=x,
        y=y,
        delta_x=0,
        delta_y=0,
        button=1,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=x,
        screen_y=y,
    )


async def _drag(app: OperatorApp, pilot: Any, start: tuple[int, int], end: tuple[int, int]) -> None:
    """Press, move, release — the gesture, not a hand-set ``selections`` dict.

    Driven through the events because the copy hangs off ``TextSelected``,
    which only a real release posts; assigning ``screen.selections`` would
    assert the formatting and skip the trigger.
    """
    app.screen._forward_event(_mouse(app, events.MouseDown, *start))
    await pilot.pause()
    app.screen._forward_event(_mouse(app, events.MouseMove, *end))
    await pilot.pause()
    app.screen._forward_event(_mouse(app, events.MouseUp, *end))
    await pilot.pause()
    await pilot.pause()


async def _seeded(app: OperatorApp, pilot: Any) -> AssistantBlock:
    """One prompt and one FINISHED answer in the transcript, settled.

    Three properties this has to guarantee before a drag means anything,
    because each of them changes the FRAME and therefore the clipboard:

    * **The splash is gone.** Appended through the app's own path rather than
      straight into the view, because that is what retires ``WelcomeView`` —
      leave it displayed and it overlaps the transcript's rows, so a press
      lands on no content widget and Textual widens the selection to the whole
      container. Seen intermittently before this: the same drag copied the
      prompt as well as the answer, in three runs out of eight.

    * **Finalized.** Mid-stream the flatten splices a settled prefix to a live
      tail and loses one blank row at the join (pinned by
      ``test_streaming_copy_is_the_streaming_frame``), so an unfinalized block
      would have these tests asserting the paragraph break away.
    * **Built at its real width.** ``AssistantBlock._flat_width`` falls back to
      ``FALLBACK_WIDTH`` while the block has no size, so text applied before
      the first layout is folded at the wrong width and only ``on_resize``
      puts it right. Asserted rather than waited out: a test that drags a
      half-settled frame does not fail, it counts a different number of rows.
    """
    app._append_block(UserBlock("summarise the ingest path"))
    block = AssistantBlock()
    app._append_block(block)
    await pilot.pause()
    block.update_text("Here is the **plan** with `inline_code`.\n\nSecond paragraph here.")
    block.finalize_text()
    await pilot.pause()
    await pilot.pause()
    assert not app.query_one(WelcomeView).display
    assert block.size.width and block._built_width == block.size.width
    return block


def _pilot_app(session: FakeSession | None = None) -> OperatorApp:
    """The real app over a fake session, with the checker's objection in ONE place.

    ``FakeSession`` is what every pilot suite here drives, but pyright will not
    accept it as a ``SessionProtocol`` — the mismatch is member variance rather
    than a missing member — so constructing the app inline puts the same error
    on every test that does it. Confined here instead.
    """
    return OperatorApp(cast("Any", lambda: _factory(session or FakeSession())))


@pytest.mark.asyncio
async def test_releasing_a_drag_puts_the_frame_on_the_clipboard() -> None:
    """The acceptance case: highlight an answer, let go, it is copied.

    ``_clipboard`` is what ``App.copy_to_clipboard`` sets on its way to the
    OSC 52 write, so this is the byte sequence the terminal is handed — checked
    end to end against Ghostty 1.3.2 in cmux by replaying that escape and
    reading ``pbpaste``.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        block = await _seeded(app, pilot)
        app._clipboard = ""

        await _drag(app, pilot, (block.region.x, block.region.y), (79, 23))

        # The RENDERED frame, which is the whole rule: no ``**`` around
        # "plan", no backticks around "inline_code", no trailing pad — and the
        # paragraph break kept, because the reader selected two paragraphs.
        assert app._clipboard.splitlines() == [
            "Here is the plan with inline_code.",
            "",
            "Second paragraph here.",
        ]


@pytest.mark.asyncio
async def test_a_copy_says_so() -> None:
    """A silent copy is indistinguishable from the bug that was reported.

    The count is the assertion, not merely the word: a toast that fired on an
    empty selection would be the same lie in the other direction.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        block = await _seeded(app, pilot)
        toast = app.query_one(Toast)
        assert toast.message == ""

        await _drag(app, pilot, (block.region.x, block.region.y), (79, 23))
        assert toast.message == "copied 3 lines"
        # The card cannot drift from what was taken: the count is recomputed
        # from the clipboard rather than restated.
        assert toast.message == f"copied {len(app._clipboard.splitlines())} lines"

        # A plain click clears the selection before posting `TextSelected`, so
        # it must copy nothing and leave the previous card alone rather than
        # announcing a copy that did not happen.
        toast.dismiss_toast()
        await _drag(app, pilot, (block.region.x, block.region.y), (block.region.x, block.region.y))
        assert toast.message == ""


@pytest.mark.asyncio
async def test_the_app_frame_never_reaches_the_clipboard() -> None:
    """A drag that overshoots the transcript copies the answer, not the app.

    Measured before ``Chrome``: the same gesture pasted ``❯`` and the whole
    status band — ``◆ test/model › ⌂ ~/local-operator`` and its interior
    padding — under the answer. Textual sweeps EVERY widget the drag crosses
    into ``selections``, and the transcript's neighbours are the composer's
    chevron and the band, so overshooting by one row is the common case rather
    than an edge one.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.query_one(Editor).insert("draft prompt in the composer")
        await _seeded(app, pilot)

        await _drag(app, pilot, (0, 0), (79, 23))

        assert "❯" not in app._clipboard
        assert "auto-approve" not in app._clipboard
        assert "draft prompt" not in app._clipboard
        # Trailing pad is the other half of the same rule: a paste that carries
        # the frame's width is as unusable as one that carries its furniture.
        assert app._clipboard.splitlines() == [row.rstrip() for row in app._clipboard.splitlines()]
        assert app._clipboard


@pytest.mark.asyncio
@pytest.mark.parametrize("focus_transcript", [False, True])
async def test_a_live_selection_never_swallows_the_interrupt(focus_transcript: bool) -> None:
    """Ctrl+C is the interrupt with text highlighted, from either focus.

    Textual's ``Screen`` binds ``ctrl+c,super+c`` to ``screen.copy_text`` and
    sits between the focused widget and the App in the binding chain, so before
    ``TranscriptScreen`` this gesture copied and ``aborts`` stayed EMPTY — and
    stayed empty for every later press, because ``action_copy_text`` never
    clears the selection. Both focuses are driven because the app reaches both:
    a drag beginning ON a block focuses the transcript, one beginning in the
    gutter leaves the composer focused, and only the composer's path goes
    through ``Editor._on_key``.
    """
    session = FakeSession()
    app = _pilot_app(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        block = await _seeded(app, pilot)
        await _drag(app, pilot, (block.region.x, block.region.y), (79, 23))
        assert app.screen.selections, "the selection must still be live for this to mean anything"

        if focus_transcript:
            app.query_one(TranscriptView).focus()
        else:
            app.query_one(Editor).focus()
        await pilot.pause()

        session.aborts.clear()
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert session.aborts == ["interrupted"]


@pytest.mark.asyncio
async def test_the_highlight_outlives_the_copy() -> None:
    """Releasing copies; it does not also wipe what the user pointed at.

    The band is the only record of WHAT was taken — the toast says an act
    happened, not its extent — so clearing it on release would read as the
    selection having been lost at the moment it was used.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        block = await _seeded(app, pilot)
        await _drag(app, pilot, (block.region.x, block.region.y), (79, 23))
        assert block in app.screen.selections


# -- the composer ------------------------------------------------------------
#
# Reported from the field after the transcript copy shipped: text highlighted
# in the COMPOSER "doesn't copy properly". Both halves of the copy story above
# miss a `TextArea`, and neither is obvious from the transcript's code:
#
# * `TextArea._watch_selection` calls `app.clear_selection()` on EVERY caret
#   move, and the mouse-down that begins a composer drag is a caret move. So
#   `Screen.selections` is emptied on the first event of the gesture and is
#   still empty at release — measured before the fix: `editor.selected_text`
#   was `'summarise the inges'` while `screen.get_selected_text()` was `None`.
#   `TextArea` also captures the mouse on press, so `Screen._select_state`
#   never leaves `None` and the screen never sees a selection to begin with.
# * `TextArea`'s own `ctrl+c,super+c` -> `action_copy` binding cannot save it:
#   cmd+C is eaten by the terminal (the same Ghostty binding the transcript
#   docstrings name) and Ctrl+C is consumed by `Editor._on_key` as this app's
#   interrupt before any binding runs.
#
# So the composer reports its own release (`Editor._copy_drag` ->
# `EditorCopied`) and the app answers it through the same clipboard write and
# the same toast as the transcript.


async def _composer_drag(
    app: OperatorApp,
    pilot: Any,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    """A drag over the composer, as `Screen._forward_event` receives it.

    Separate from `_drag` only in that it does not assert a screen selection
    afterwards: over a `TextArea` there is never going to be one, which is the
    whole reason this path exists.
    """
    app.screen._forward_event(_mouse(app, events.MouseDown, *start))
    await pilot.pause()
    if start != end:
        app.screen._forward_event(_mouse(app, events.MouseMove, *end))
        await pilot.pause()
    app.screen._forward_event(_mouse(app, events.MouseUp, *end))
    await pilot.pause()
    await pilot.pause()


async def _composer(app: OperatorApp, pilot: Any, text: str) -> Editor:
    """The composer holding `text`, settled and focused.

    Settling matters for the same reason it does in `_seeded`: the editor's
    region is read to aim the drag, and a region read before the first layout
    aims at the wrong row.
    """
    editor = app.query_one(Editor)
    editor.focus()
    editor.load_text(text)
    await pilot.pause()
    await pilot.pause()
    return editor


def _cell(editor: Editor, row: int, column: int) -> tuple[int, int]:
    """Screen coordinates of one document cell.

    Computed from the widget's own gutter rather than written as
    `region.x + column`: the composer inherits `padding: 0 1` (see the
    stylesheet's note on the flush card-on-composer seam), so the naive form
    aims one column left and every assertion below would be off by a character
    — which is a test that passes while describing the wrong gesture.
    """
    return (
        editor.region.x + editor.gutter.left + column,
        editor.region.y + editor.gutter.top + row,
    )


@pytest.mark.asyncio
async def test_releasing_a_composer_drag_copies_what_was_highlighted() -> None:
    """The acceptance case: highlight your own draft, let go, it is copied.

    Measured before the fix, this same gesture left `_clipboard` untouched
    while the highlight sat on screen — the reported bug, in one assertion.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        app._clipboard = ""

        await _composer_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))

        assert editor.selected_text == "summarise the ingest"
        # The clipboard IS the highlight — the property the transcript's
        # `get_selection` gets from sharing one computation, and which this
        # path has to get by taking the widget's own selected text.
        assert app._clipboard == editor.selected_text
        # ...and the screen still has no selection of its own, so this could
        # only have come from the editor. If this ever starts passing through
        # `Screen.selections`, the mechanism changed and these tests are stale.
        assert not app.screen.selections


@pytest.mark.asyncio
async def test_a_composer_copy_says_so_in_the_same_words() -> None:
    """One receipt for both gestures, or the toast becomes a tell.

    A copy out of the input has to be indistinguishable from a copy out of the
    transcript; a different wording (or a silent composer copy) would make the
    user reason about which widget they dragged over.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "first line here\nsecond line here")
        toast = app.query_one(Toast)
        assert toast.message == ""

        # Ends inside the second row, so the copy is a partial line — the
        # count in the toast is rows spanned, not lines completed.
        await _composer_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 1, 11))

        assert app._clipboard == "first line here\nsecond line"
        assert toast.message == "copied 2 lines"
        # Recomputed from the clipboard rather than restated, exactly as the
        # transcript's test does: the card cannot drift from what was taken.
        assert toast.message == f"copied {len(app._clipboard.splitlines())} lines"


@pytest.mark.asyncio
async def test_a_click_in_the_composer_copies_nothing() -> None:
    """Placing the caret is not a copy, and must not announce one.

    The common gesture in this widget by far. A click leaves `start == end`,
    and a toast on every caret placement would be noise on top of a clipboard
    the user did not ask to change.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        app._clipboard = "SOMETHING THE USER PUT THERE"
        toast = app.query_one(Toast)

        await _composer_drag(app, pilot, _cell(editor, 0, 5), _cell(editor, 0, 5))

        assert app._clipboard == "SOMETHING THE USER PUT THERE"
        assert toast.message == ""


@pytest.mark.asyncio
async def test_a_drag_over_an_empty_composer_copies_nothing() -> None:
    """The placeholder is not the user's text.

    An empty composer still paints `Message Local Operator…`, and dragging
    across it selects nothing — `selected_text` is `""`. Copying the invitation
    would put words on the clipboard that the user never wrote.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "")
        app._clipboard = "SOMETHING THE USER PUT THERE"
        toast = app.query_one(Toast)

        await _composer_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 15))

        assert editor.selected_text == ""
        assert app._clipboard == "SOMETHING THE USER PUT THERE"
        assert toast.message == ""


@pytest.mark.asyncio
async def test_clicking_an_attachment_marker_selects_it_without_copying() -> None:
    """The chip gesture keeps its meaning: select to delete, not to copy.

    `Editor._on_mouse_up` selects a whole marker on a click inside one so that
    backspace removes it atomically. That selection is the APP's, not a range
    the user dragged, so it must not reach the clipboard — otherwise reaching
    for the delete gesture silently replaces whatever the user had copied, and
    raises a receipt claiming they asked for it.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "[Image #1, 100x100] describe this")
        app._clipboard = "SOMETHING THE USER PUT THERE"
        toast = app.query_one(Toast)

        await _composer_drag(app, pilot, _cell(editor, 0, 3), _cell(editor, 0, 3))

        assert app._clipboard == "SOMETHING THE USER PUT THERE"
        assert toast.message == ""


@pytest.mark.asyncio
async def test_a_composer_drag_copies_a_marker_as_the_text_that_cites_it() -> None:
    """Dragging ACROSS a marker copies the citation, not a decoration of it.

    A marker is painted as a chip but it is document text — the same characters
    `_submit` sends and `resolve_markers` reads. Copying what is on the row
    means the paste re-cites the image in another draft, which is the only
    behaviour that makes a copied prompt reusable.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "[Image #1, 100x100] describe this")
        app._clipboard = ""

        # `len("[Image #1, 100x100]")` exactly: the drag ends ON the cell
        # after the closing bracket, and the selection end is exclusive, so
        # this is the marker and not one character of the prose after it.
        await _composer_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 19))

        assert app._clipboard == "[Image #1, 100x100]"


@pytest.mark.asyncio
async def test_a_composer_drag_leaves_the_draft_alone() -> None:
    """Copying is not cutting, and the caret gesture still ends where it did.

    Worth pinning because the copy is bolted onto the mouse-up that `TextArea`
    also uses to finalise its selection: a copy path that touched the document,
    or that collapsed the selection it had just taken, would corrupt a draft
    mid-sentence.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")

        await _composer_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))

        assert editor.text == "summarise the ingest path please"
        # The highlight outlives the copy here for the same reason it does in
        # the transcript: it is the only record of what was taken.
        assert editor.selected_text == "summarise the ingest"


@pytest.mark.asyncio
async def test_a_composer_drag_still_leaves_ctrl_c_as_the_interrupt() -> None:
    """The copy must not buy itself a key, least of all this one.

    The transcript's copy was deliberately hung off the release so that Ctrl+C
    could stay the interrupt and the first rung of the exit ladder. The
    composer's copy is hung off the release for the same reason, and this is
    the assertion that keeps a future "just bind ctrl+c in the editor" from
    quietly reintroducing the swallowed-abort bug in the one widget that is
    focused in essentially every frame.
    """
    session = FakeSession()
    app = _pilot_app(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        await _composer_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))
        assert editor.selected_text, "the highlight must be live for this to mean anything"

        session.aborts.clear()
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert session.aborts == ["interrupted"]
