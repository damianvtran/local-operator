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

import pytest
from rich.cells import cell_len
from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text
from textual.content import Content
from textual.geometry import Offset
from textual.selection import Selection
from textual.visual import RichVisual

from local_operator.tui import theme as theme_mod
from local_operator.tui.glyphs import tool_icon
from local_operator.tui.markdown_theme import install_markdown_theme
from local_operator.tui.widgets.assistant import AssistantBlock, flatten
from local_operator.tui.widgets.tool_card import OUTPUT_INDENT, ToolCard
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    RichBlock,
    TranscriptBlock,
    TranscriptView,
    UserBlock,
)
from tests.unit.tui.conftest import StyledTranscriptApp

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
