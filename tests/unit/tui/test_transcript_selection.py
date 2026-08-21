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
from local_operator.tui.widgets.toast import TOAST_FAILURE_MS, Toast
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
async def test_markdown_copies_as_markdown() -> None:
    """The whole assistant message copies as its source markdown.

    The copy is NOT the rendered frame: a messenger or email client does not
    recognise a ``▌`` quote bar or a ``•`` bullet, so the clipboard carries the
    message's markdown, which Slack, GitHub, Linear and Notion all render and
    which reads as tidy plain text anywhere else. Bold keeps its ``**``, the
    fence keeps its backticks and language, the list keeps its markers, and the
    link keeps its label (its URL stays on the frame as a hyperlink, asserted
    below).
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
        # The five constructs, called out so a failure says WHICH one moved.
        assert "**plan**" in copied  # bold keeps its markers
        assert "`inline_code`" in copied  # inline code keeps its backticks
        assert "```python\ndef f(x):\n    return x + 1\n```" in copied  # fence, verbatim
        assert "1. first step" in copied and "2. second step" in copied  # ordered list
        assert "[link](https://example.com/x)" in copied  # the link, label and URL


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
async def test_fenced_code_copies_as_a_fence() -> None:
    """A code fence copies as a fenced block: backticks, language, no pad.

    Copying the source rather than the frame means the trailing pad Rich paints
    across every row never reaches the clipboard at all, and the block arrives
    as a runnable, language-tagged fence a markdown reader renders as code.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text("```python\nx = 1\n```")
        block.finalize_text()
        await pilot.pause()
        assert _copy_all(app, block) == "```python\nx = 1\n```"


# -- markdown copy: the regression this feature answers -----------------------
@pytest.mark.asyncio
async def test_blockquote_copies_as_markdown_not_the_bar() -> None:
    """A blockquote's ``▌`` never reaches the clipboard — the reported bug.

    The reader drags over a quoted reply and pastes it into Slack or an email;
    what arrives must be the ``>`` markdown those clients render, not the
    half-block bar Rich painted. Bold inside the quote keeps its ``**`` for the
    same reason.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(80, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(
            "Here is a reply you can paste:\n\n"
            "> Thanks for the report. I verified the **flagged** value in out/main/index.jsc:\n"
            "> it's the public project API key (phc_...), publishable by design.\n"
        )
        block.finalize_text()
        await pilot.pause()
        copied = _copy_all(app, block)
        assert copied is not None
        assert "▌" not in copied
        assert "> Thanks for the report. I verified the **flagged** value" in copied
        assert "> it's the public project API key" in copied


@pytest.mark.asyncio
async def test_table_copies_as_markdown_pipes() -> None:
    """A table copies as markdown pipes, not box-drawing.

    The rendered frame draws the table with ``─`` rules and drops the pipes;
    pasted into a markdown reader that is not a table. The source pipes survive
    the copy, header row and divider included.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(80, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(
            "Results:\n\n| Name | Score |\n|------|-------|\n| alpha | 0.91 |\n| beta | 0.87 |\n"
        )
        block.finalize_text()
        await pilot.pause()
        copied = _copy_all(app, block)
        assert copied is not None
        assert "| Name | Score |" in copied
        assert "| alpha | 0.91 |" in copied
        assert "─" not in copied


@pytest.mark.asyncio
async def test_a_multi_block_copy_joins_plain_and_markdown() -> None:
    """A drag across a user block and an answer copies both, cleanly.

    The user block copies its text verbatim (no ``▌``), the assistant block its
    markdown; ``Screen.get_selected_text`` joins the two contributions, so this
    pins the seam rather than each block alone.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(80, 40)) as pilot:
        user = UserBlock("summarise the ingest path")
        assistant = AssistantBlock()
        await _mounted(app, user, assistant)
        assistant.update_text("Here is the **plan**.")
        assistant.finalize_text()
        await pilot.pause()
        app.screen.selections = {
            user: Selection(None, None),
            assistant: Selection(None, None),
        }
        copied = app.screen.get_selected_text()
        assert copied is not None
        assert "▌" not in copied
        assert "summarise the ingest path" in copied
        assert "**plan**" in copied


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
async def test_partial_selection_copies_a_valid_markdown_slice() -> None:
    """A drag over the fence's rows copies the fence, closed.

    The highlighted rows come from the same ``Selection.get_span`` the band
    uses, so the copy covers what was lit — but a partial markdown selection is
    re-fenced on the way out, because raw ``def f(x):\\n    retu`` is neither
    the code the reader pointed at nor valid markdown. The slice opens and
    closes the fence so it pastes as a code block.
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
        expected = ("```python\ndef f(x):\n    return x + 1\n```", "\n")
        assert block.get_selection(selection) == expected


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
async def test_streaming_copy_is_stable_across_the_splice() -> None:
    """A mid-stream copy and a settled copy carry the same markdown.

    The copy is the message's source, aligned to the highlighted rows — so it
    does not inherit the live frame's splice gap (the frozen prefix plus a fresh
    tail drops a blank row the settled frame keeps). What must hold is that the
    words and their order never depend on when the copy was taken, and that the
    markdown constructs survive either way.
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
        assert "**plan**" in live and "```python" in live


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

        # The message's MARKDOWN: ``**`` around "plan", backticks around
        # "inline_code", and the paragraph break kept, because the reader
        # selected two paragraphs.
        assert app._clipboard.splitlines() == [
            "Here is the **plan** with `inline_code`.",
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
    *,
    copy: bool = False,
) -> None:
    """A drag over the composer, as `Screen._forward_event` receives it.

    Separate from `_drag` only in that it does not assert a screen selection
    afterwards: over a `TextArea` there is never going to be one, which is the
    whole reason this path exists.

    ``copy=True`` arms `copy_on_release` with an explicit Ctrl+C over the
    drag's own cells mapped to document locations. Only tests that need the
    arm WITHOUT the helper's full ceremony (sentinel, row-0 span — see
    `_composer_copy_drag`) should pass it; the rest belong on the helper.
    """
    if copy:
        editor = app.query_one(Editor)
        editor.selection = Selection(
            editor.get_target_document_location(_mouse(app, events.MouseDown, *start)),
            editor.get_target_document_location(_mouse(app, events.MouseUp, *end)),
        )
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()
    app.screen._forward_event(_mouse(app, events.MouseDown, *start))
    await pilot.pause()
    if start != end:
        app.screen._forward_event(_mouse(app, events.MouseMove, *end))
        await pilot.pause()
    app.screen._forward_event(_mouse(app, events.MouseUp, *end))
    await pilot.pause()
    await pilot.pause()


async def _composer_copy_drag(
    app: OperatorApp,
    pilot: Any,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    """An ARMED composer drag: the release copies, as after an explicit copy.

    The composer no longer copies on a bare release — a highlight there is
    usually the first half of a retype or delete, so the release copies only
    while the user has signalled "I am taking text" by asking for a copy first
    (`Editor.copy_on_release`). Most copy-behaviour tests in this file are
    about what an explicit copy takes and claims, so they drive this helper
    and keep every assertion they already made; tests about the DEFAULT are
    the ones that state the new rule.

    The arming copy's own range is the drag's row-0 span. It cannot be the
    drag's exact range: a range ending on a LATER row would keep the caret at
    that row, and the drag's row-0 mouse-down would land outside it — which
    the selection watcher reads as the copy's highlight being replaced and
    disarms the flag the drag is meant to exercise. A row-0 span contains the
    drag's down, and collapsing to a caret INSIDE the range is not a
    disarm (a select-to-retype lands there too). A single-row drag's arm is
    therefore its own range exactly; a multi-row one's is the first row —
    both are what a user who just copied "this part" would actually do.
    """
    app._clipboard = "SENTINEL"
    editor = app.query_one(Editor)
    # `get_target_document_location` needs a full event; `Offset` is all it
    # reads. The UP event is aimed one column PAST the drag's end column:
    # drag selection ends EXCLUSIVE at the released cell, so this keeps the
    # arming range a strict superset of the drag's row-0 span — a single-row
    # drag's release would otherwise collapse to the copy's own caret, and
    # the watcher's literal `!=` disarms on it.
    editor.selection = Selection(
        editor.get_target_document_location(_mouse(app, events.MouseDown, *start)),
        editor.get_target_document_location(_mouse(app, events.MouseUp, end[0] + 1, start[1])),
    )
    await pilot.pause()
    await pilot.press("ctrl+c")
    await pilot.pause()
    await _composer_drag(app, pilot, start, end)


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
async def test_releasing_a_composer_drag_copies_nothing_by_default() -> None:
    """The reported defect: highlighting to select clobbered the clipboard.

    A composer highlight is usually the first half of an EDIT — drag a phrase
    to retype it, drag a word to delete it — so copying on release replaced
    whatever the user had on their clipboard with text they were about to
    throw away. The transcript keeps release-copies, where a highlight is
    read-only text being taken; the composer's release selects and nothing
    more.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        app._clipboard = "SOMETHING THE USER PUT THERE"
        toast = app.query_one(Toast)

        await _composer_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))

        # The drag still SELECTS — it is the copy that is gone.
        assert editor.selected_text == "summarise the ingest"
        assert app._clipboard == "SOMETHING THE USER PUT THERE"
        assert toast.message == ""


@pytest.mark.asyncio
async def test_ctrl_c_with_a_live_range_copies_it_and_arms_the_next_drag() -> None:
    """The composer's copy gesture is explicit: highlight, then Ctrl+C.

    The field report that put copy-on-release in the composer was right that
    the widget had no working copy key: cmd+C is eaten by the terminal and
    Ctrl+C was always the interrupt. This is the sequence that fixes it
    without the clobber: the press copies the live range (instead of
    interrupting, which a real range protects the user from needing), and the
    drag the user makes next — while still in the taking-things frame of
    mind — copies on release like the transcript's.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        app._clipboard = ""
        editor.selection = Selection((0, 0), (0, 20))
        await pilot.pause()

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app._clipboard == "summarise the ingest"
        # The explicit copy also kept the draft and raised no interrupt: it
        # consumed the press entirely.
        assert editor.text == "summarise the ingest path please"
        assert editor.copy_on_release, "an explicit copy must arm the follow-up drag"

        # And the very next drag copies on release — the armed gesture. What
        # matters is that the release wrote the NEW highlight (a bare drag
        # would leave the first copy's text on the clipboard); the drag's own
        # `selected_text` is the claim, as in every other composer test.
        await _composer_drag(app, pilot, _cell(editor, 0, 29), _cell(editor, 0, 34))
        assert editor.selected_text == app._clipboard
        assert app._clipboard == "ase"


@pytest.mark.asyncio
async def test_the_drag_copy_arm_retires_with_the_highlight_it_copied() -> None:
    """The arm is not a mode the user gets stuck in.

    It belongs to the copy conversation: once the copy's highlight leaves the
    screen the next selection is a new decision, and the select-to-edit
    default must be the state the user returns to. Otherwise one stray copy
    would turn every later select-to-delete back into a clipboard clobber,
    which is the defect this change exists to remove.

    Driven with the REAL armed drag rather than the bare press: the arm is
    the next drag's, so its retirement is the gesture flag's retirement —
    after the follow-up drag copies, the user carrying on (moving the caret
    off the range that drag took) is what ends the conversation.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))
        assert editor.copy_on_release, "the armed drag must leave the conversation open"
        assert editor._copy_gesture, "the copy's gesture must be live for this to mean anything"

        await pilot.press("right")
        await pilot.pause()
        assert not editor._copy_gesture, "the gesture outlived its own highlight"
        assert not editor.copy_on_release, "the arm outlived the highlight it copied"

        app._clipboard = "SOMETHING THE USER PUT THERE"
        await _composer_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))
        assert app._clipboard == "SOMETHING THE USER PUT THERE"


@pytest.mark.asyncio
async def test_releasing_an_armed_composer_drag_copies_what_was_highlighted() -> None:
    """After an explicit copy, the follow-up drag IS the transcript's gesture.

    The user pressed Ctrl+C over a range, so they are taking text out of the
    composer; the next release copies what it highlighted and the receipt
    says so — the transcript's rule, reached through an explicit signal
    instead of assumed of every drag.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))

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
        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 1, 11))

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
        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 19))

        assert app._clipboard == "[Image #1, 100x100]"


@pytest.mark.asyncio
async def test_a_composer_drag_leaves_the_draft_alone() -> None:
    """Selecting is not cutting, and the caret gesture still ends where it did.

    Worth pinning because the release is where `TextArea` finalises its
    selection: any handling bolted onto that mouse-up that touched the
    document, or that collapsed the selection it had just made, would corrupt
    a draft mid-sentence.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))

        assert editor.text == "summarise the ingest path please"
        # The highlight outlives the copy here for the same reason it does in
        # the transcript: it is the only record of what was taken.
        assert editor.selected_text == "summarise the ingest"


@pytest.mark.asyncio
async def test_ctrl_c_with_no_selection_never_reaches_the_interrupt() -> None:
    """A draft takes the key first — and a copy never changes that.

    The composer Ctrl+C rungs are, in order: copy (only while a real range is
    live), then the draft — a half-typed prompt means "scrap that", filed to
    history, not aborted — and only then the interrupt. The
    highlight-then-Ctrl+C copy sits at the TOP of that ladder by range, so it
    cannot swallow the draft's press in the composer's resting state (a
    collapsed caret). Asserted both ways around the new gesture because the
    copy is what this file changed: it must neither steal the key with no
    range live, nor leave state behind that steals it afterwards.
    """
    session = FakeSession()
    app = _pilot_app(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        assert not editor.selected_text, "no range may be live for this to mean anything"

        # First press, before any copy ever happened: the draft rung, not the
        # interrupt, and the draft is filed rather than destroyed.
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert session.aborts == [], "no range was live, so nothing may divert the key to abort"
        assert editor.text == "", "the draft was not cleared"
        assert "summarise the ingest path please" in editor.prompt_history()

        # And after a REAL copy gesture has run — the new path this change
        # adds — the same press on the next draft is still the draft's: the
        # copy must leave no state behind that claims the key.
        await _composer(app, pilot, "a second draft, with a copy made in it")
        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))
        await pilot.press("right")
        await pilot.pause()
        assert not editor.selected_text, "the copy's highlight must be gone"
        session.aborts.clear()
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert session.aborts == [], "a finished copy left the key diverted to abort"
        assert editor.text == "", "the second draft was not cleared"


# -- what a release does NOT copy (review round 1, F1/F2) ---------------------
#
# `Editor._on_mouse_up` fires for every mouse-up routed to the composer, not
# only for drags the composer began — and `TextArea` leaves its selection live
# after a drag, so the widget is holding a stale range most of the time. The
# first version of this fix copied on all of them, which meant a transcript
# copy was overwritten by the composer's old highlight a moment after being
# made. `_copy_drag` gates on `_selecting`, which is true only between
# `TextArea`'s own mouse-down and mouse-up.


@pytest.mark.asyncio
async def test_a_transcript_drag_released_over_the_composer_keeps_the_transcript_copy() -> None:
    """Overshooting a transcript drag into the composer must not clobber it.

    This is the ordinary way to select to the end of an answer — the composer
    is docked directly below the transcript, and the existing transcript tests
    deliberately drag to `(79, 23)` for the same reason. Measured before the
    `_selecting` gate: the user dragged the agent's answer and got their own
    draft on the clipboard, with a toast confirming the wrong copy.

    Both handlers run for this one release (`TextSelected` from the screen and,
    before the gate, `EditorCopied` from the widget), and the composer's
    message is delivered second — so it always won.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        block = await _seeded(app, pilot)
        editor = await _composer(app, pilot, "my private draft")

        # An ARMED composer copy first, so the editor is holding a live
        # selection and a live `copy_on_release` exactly as it would be after
        # an explicit copy in use — the worst case for the clobber, since the
        # stale range is both present AND authorised to copy on release.
        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 16))
        assert app._clipboard == "my private draft"
        assert editor.selected_text == "my private draft"
        assert editor.copy_on_release, "the arm must be live for this to mean anything"

        # Now drag the ANSWER and release over the composer.
        app.screen._forward_event(_mouse(app, events.MouseDown, block.region.x, block.region.y))
        await pilot.pause()
        app.screen._forward_event(_mouse(app, events.MouseMove, 70, block.region.y + 2))
        await pilot.pause()
        app.screen._forward_event(_mouse(app, events.MouseMove, *_cell(editor, 0, 5)))
        await pilot.pause()
        app.screen._forward_event(_mouse(app, events.MouseUp, *_cell(editor, 0, 5)))
        await pilot.pause()
        await pilot.pause()

        assert "my private draft" not in app._clipboard
        assert app._clipboard == app.screen.get_selected_text()


@pytest.mark.asyncio
async def test_ctrl_c_copies_a_keyboard_selection_without_a_mouse() -> None:
    """The explicit copy works for a shift+arrow range too.

    A keyboard selection never enters the release machinery at all — there is
    no drag for `_copy_drag` to gate on — and before the explicit gesture this
    range was simply uncopyable: Ctrl+C interrupted, cmd+C never arrived. The
    press is the whole gesture now, so it has to cover both ways a range can
    be made.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "my private draft")
        app._clipboard = "SOMETHING THE USER PUT THERE"

        await pilot.press("home", *["shift+right"] * 10)
        await pilot.pause()
        assert editor.selected_text == "my private", "the keyboard selection must exist"

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app._clipboard == "my private"


@pytest.mark.asyncio
async def test_the_read_only_composer_still_copies_what_it_shows() -> None:
    """Subagent mode is read-only, and a drag over it is still a copy.

    The text there is the app's rather than the user's, which is exactly why
    someone would want to lift it out. Pinned because `_copy_drag`'s reason for
    copying the document text is phrased in terms of what the user typed, and
    the read-only case must not be read as an oversight.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "someone else wrote this")
        editor.read_only = True
        await pilot.pause()
        app._clipboard = ""

        # Read-only or not, the gesture is the explicit one: a range, then
        # Ctrl+C. The text is the app's, which is exactly why someone would
        # lift it out — and why it never sits on the clipboard unasked.
        editor.selection = Selection((0, 0), (0, 7))
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app._clipboard == "someone"
        assert editor.text == "someone else wrote this"


# -- what the receipt CLAIMS (review round 1 F3, design round 1 D1/D2/D3/D5) --


@pytest.mark.asyncio
async def test_a_sub_line_copy_is_counted_in_characters() -> None:
    """ "1 line" was a claim the frame contradicted.

    Two ways at once: the user dragged three words out of a line, and in the
    composer a long draft SOFT-WRAPS, so one document line is painted as three
    highlighted rows. The receipt said `1 line` while the user looked at three
    rows of highlight — and the count is the only information the card carries.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        toast = app.query_one(Toast)

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))

        assert app._clipboard == "summarise"
        assert toast.message == "copied 9 characters"


@pytest.mark.asyncio
async def test_a_wrapped_selection_is_not_reported_as_one_line() -> None:
    """The composer's common case: a draft too long for one row.

    The regression this pins is specifically the DISAGREEMENT between the
    receipt and the frame, so it asserts the widget really is painting more
    rows than the document has lines.
    """
    long_draft = (
        "please summarise the ingest path and then explain how the dedupe stage "
        "interacts with the watermark, including the retry semantics and what "
        "happens on a partial failure"
    )
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, long_draft)
        toast = app.query_one(Toast)
        assert editor.region.height > 1, "the draft must actually wrap for this to mean anything"

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 2, 60))

        assert "\n" not in app._clipboard
        assert "line" not in toast.message
        assert toast.message == f"copied {len(app._clipboard)} characters"


@pytest.mark.asyncio
async def test_a_multi_line_copy_is_still_counted_in_lines() -> None:
    """Spanning lines keeps the transcript's familiar receipt.

    The unit follows the shape of the selection: lines are the useful magnitude
    once there is more than one, and this is the wording the transcript copy
    has always used for a multi-paragraph take.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "first line here\nsecond line here")
        toast = app.query_one(Toast)

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 1, 11))

        assert app._clipboard == "first line here\nsecond line"
        assert toast.message == "copied 2 lines"


@pytest.mark.asyncio
async def test_one_line_and_its_break_is_not_reported_as_two_lines() -> None:
    """`count("\\n") + 1` read a trailing newline as a whole further line.

    "Select this line" — drag from the start of one row to the start of the
    next — is a natural gesture in a multi-line draft, and it reported
    `copied 2 lines` for one line of text (review round 1, F3).
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "first line here\nsecond line here")
        toast = app.query_one(Toast)

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 1, 0))

        assert app._clipboard == "first line here\n"
        assert toast.message != "copied 2 lines"
        # 16, not 15: the break is a character the clipboard carries. Cell-width
        # counting scored it 0 and reported 15 (design round 2, D10).
        assert toast.message == "copied 16 characters"


@pytest.mark.asyncio
async def test_a_copy_receipt_does_not_evict_an_actionable_notice() -> None:
    """A courtesy card must not take the slot from a failure the user must read.

    The single toast slot was sized for startup-scale events. Once a routine
    editing gesture can write it, an MCP failure — the 10 s variant, naming a
    server and an error — was displaced by a drag in the composer, and startup
    is exactly when someone is typing their first prompt (design round 1, D2).

    The copy itself still happens: it is the CLAIM that stands down, not the
    clipboard write.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        toast = app.query_one(Toast)
        toast.show("mcp: failed: github — command not found: gh", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()
        app._clipboard = ""

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))

        assert toast.message.startswith("mcp: failed")
        assert app._clipboard == "summarise"

        # ...and it is still there a keystroke later. The deference is only
        # half the protection: the copy raised no card, so it must also not
        # come to OWN the card it deferred to and withdraw it on the next edit
        # (review round 3, F10/F11). This test builds the state that bug needs,
        # so it is the one that should catch it.
        await pilot.press("x")
        await pilot.pause()

        assert toast.message.startswith("mcp: failed")
        assert toast.display


@pytest.mark.asyncio
async def test_a_copy_receipt_still_replaces_an_ordinary_one() -> None:
    """Only ACTIONABLE notices hold the slot; the deference is not blanket.

    Without this the previous test would also pass if the copy receipt had
    simply stopped showing at all.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        toast = app.query_one(Toast)
        toast.show("mcp: 2 connected (14 tools)")
        await pilot.pause()

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))

        assert toast.message == "copied 9 characters"


@pytest.mark.asyncio
async def test_typing_over_a_copied_selection_retires_the_receipt() -> None:
    """Select-to-overwrite: the commonest edit in any input.

    Drag a word, type the replacement, and the receipt sat there for the rest
    of its five seconds asserting a copy of characters that no longer existed
    in the field (design round 1, D3).

    The CLIPBOARD is deliberately untouched — the copy really happened, and a
    paste a minute later must still produce it. Only the claim is withdrawn.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        toast = app.query_one(Toast)

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 14), _cell(editor, 0, 20))
        assert app._clipboard == "ingest"
        assert toast.message == "copied 6 characters"

        await pilot.press("d", "e", "d", "u", "p", "e")
        await pilot.pause()

        assert toast.message == ""
        assert app._clipboard == "ingest"
        assert editor.text == "summarise the dedupe path please"


@pytest.mark.asyncio
async def test_typing_does_not_dismiss_a_notice_that_is_not_a_copy_receipt() -> None:
    """The retirement is scoped to the card the copy raised.

    An MCP failure that arrived after the copy is not the editor's business,
    and dismissing it because the user carried on typing would be the eviction
    bug (D2) reintroduced from the other side.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        toast = app.query_one(Toast)

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 14), _cell(editor, 0, 20))
        toast.show("mcp: failed: github — command not found: gh", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()

        await pilot.press("d", "e", "d", "u", "p", "e")
        await pilot.pause()

        assert toast.message.startswith("mcp: failed")


# -- whose receipt is it? (review round 2 F5/F9, design round 2 D8) -----------
#
# The D3 retirement first asked whether the card's message started with
# `copied `, which cannot tell one receipt from another — a transcript copy
# raises a `copied …` card too. So turning to the composer and typing the next
# prompt withdrew a claim that was still perfectly true. The app now remembers
# the `Toast.generation` of the card the COMPOSER's copy raised, and retires
# that card and no other.


@pytest.mark.asyncio
async def test_typing_does_not_retire_the_transcript_s_copy_receipt() -> None:
    """A transcript copy is not falsified by typing in the composer.

    Copy an answer, turn to the composer to write the follow-up, and the
    receipt for the answer must survive the first keystroke: nothing about that
    copy went stale, and the composer never touched the text it describes.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        block = await _seeded(app, pilot)
        editor = await _composer(app, pilot, "summarise the ingest path please")
        toast = app.query_one(Toast)

        # A composer copy FIRST, so the editor is armed exactly as it would be
        # in use — this is the state that made the old prefix check misfire.
        await _composer_copy_drag(app, pilot, _cell(editor, 0, 14), _cell(editor, 0, 20))
        assert toast.message == "copied 6 characters"

        await _drag(app, pilot, (block.region.x, block.region.y), (79, 23))
        transcript_receipt = toast.message
        assert transcript_receipt == "copied 3 lines"

        editor.focus()
        await pilot.press("h")
        await pilot.pause()

        assert toast.message == transcript_receipt
        assert toast.display


@pytest.mark.asyncio
async def test_replacing_the_buffer_disarms_the_receipt() -> None:
    """`load_text` is the other mutation funnel, and it must stand the flag down.

    Textual's ``text`` setter calls ``load_text`` directly, so ``edit()`` never
    runs for a submit, a `/clear`, a history step or `begin_model_query`. The
    flag survived all of them, and the next keystroke — whenever it came, and
    whatever was on screen by then — withdrew that card (review round 2, F5).
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        toast = app.query_one(Toast)

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 14), _cell(editor, 0, 20))
        assert editor._copied, "the copy must arm the flag for this to mean anything"

        editor.clear_content()
        await pilot.pause()
        assert not editor._copied

        # A card raised after the buffer was replaced is not this copy's, so
        # typing must leave it alone.
        toast.show("mcp: 2 connected (14 tools)")
        await pilot.pause()
        await pilot.press("h")
        await pilot.pause()

        assert toast.message == "mcp: 2 connected (14 tools)"


# -- what the number counts (review round 2 F6/F7, design round 2 D10) --------


@pytest.mark.asyncio
async def test_a_line_and_its_break_is_not_reported_as_zero_characters() -> None:
    """`cell_len` scores a newline 0, so a real copy announced nothing.

    "Select this line" puts a trailing break on the clipboard, which
    `splitlines()` still counts as one line — so it takes the character branch,
    and under cell-width counting the receipt read `copied 0 characters` for a
    genuine clipboard write (design round 2, D10). A receipt that says 0 reads
    as a failure.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "first line here\nsecond line here")
        toast = app.query_one(Toast)

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 1, 0))

        assert app._clipboard == "first line here\n"
        assert toast.message == "copied 16 characters"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("draft", "columns", "expected"),
    [
        ("👍 ok", 2, "copied 1 character"),
        ("日本語 text", 3, "copied 1 character"),
    ],
)
async def test_a_wide_glyph_counts_as_the_one_character_it_is(
    draft: str, columns: int, expected: str
) -> None:
    """The word `characters` has to mean characters.

    Cells were tried first, on the theory that the receipt should count what is
    painted. But an emoji is ONE glyph the user highlighted and `cell_len` calls
    it two, and a tab is scored 0 while the editor paints it expanded — so cell
    width matched the frame only where it already matched `len` (review round 2,
    F6). The reason given for the count now has a test that decides it.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, draft)
        toast = app.query_one(Toast)

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, columns))

        assert toast.message == expected


@pytest.mark.asyncio
async def test_a_sub_line_transcript_copy_is_counted_in_characters_too() -> None:
    """The receipt is shared, so the transcript's wording changed as well.

    Deliberate and consistent with the composer — the unit follows the shape of
    the selection wherever it was made — but it was changed behaviour with no
    test asserting it (review round 2, F7).
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        block = await _seeded(app, pilot)
        toast = app.query_one(Toast)

        await _drag(
            app, pilot, (block.region.x, block.region.y), (block.region.x + 13, block.region.y)
        )

        assert "\n" not in app._clipboard
        assert toast.message == f"copied {len(app._clipboard)} characters"


@pytest.mark.asyncio
async def test_a_declined_receipt_does_not_adopt_the_notice_it_deferred_to() -> None:
    """A copy that raised NO card has nothing to withdraw later.

    `Toast.show` declines the slot while an actionable notice is up (D2), so a
    copy made in that window paints nothing. Reading the generation
    unconditionally then pointed the app at the FAILURE's card, and the next
    keystroke dismissed the very notice the deference existed to protect
    (review round 3) — the eviction bug arriving by the back door, one round
    after it was closed at the front.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        toast = app.query_one(Toast)
        toast.show("mcp: failed: github — command not found: gh", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))
        assert app._clipboard == "summarise", "the copy itself must still happen"
        assert toast.message.startswith("mcp: failed")

        await pilot.press("x")
        await pilot.pause()

        assert toast.message.startswith("mcp: failed")
        assert toast.display


@pytest.mark.asyncio
async def test_a_held_receipt_is_withdrawn_if_its_text_is_edited_away() -> None:
    """A deferred claim can go stale before it is ever painted.

    Promoting the declined receipt to a card that outlives the gesture (D9) put
    it outside the reach of the staleness retirement: `Editor._copied` is a
    one-shot flag consumed by the first edit, and that edit lands while the
    receipt is still held — with nothing showing to retire, because the copy
    correctly declined to arm one (D13). The card then appeared, already false,
    when the slot freed (design round 4, D14).

    The window makes it likely rather than exotic: the notice runs ten seconds
    and the user is typing into the composer the whole time.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        toast = app.query_one(Toast)
        toast.show("mcp: failed: github — command not found: gh", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))
        assert app._clipboard == "summarise"

        # The user types over what they copied, while the notice still holds
        # the slot.
        await pilot.press("z")
        await pilot.pause()

        toast.dismiss_toast()
        await pilot.pause()

        assert toast.message == ""
        assert toast.display is False
        # The clipboard is untouched, as everywhere else: only the claim goes.
        assert app._clipboard == "summarise"


@pytest.mark.asyncio
async def test_a_held_receipt_still_arrives_when_its_text_survives() -> None:
    """The withdrawal is scoped to a claim that became false.

    Without this, D14's fix could pass by dropping every deferred receipt,
    which would undo D9 — the copy would go unacknowledged again.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        toast = app.query_one(Toast)
        toast.show("mcp: failed: github — command not found: gh", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))

        toast.dismiss_toast()
        await pilot.pause()

        assert toast.message == "copied 9 characters"
        assert toast.display


@pytest.mark.asyncio
async def test_a_composer_edit_does_not_discard_the_transcript_s_held_receipt() -> None:
    """The held slot is shared, so a withdrawal has to name its own card.

    `EditorCopyStale` is evidence about the COMPOSER's buffer, and the first
    version of the D14 withdrawal dropped whatever was held — so typing in the
    composer threw away a transcript copy's receipt that was still perfectly
    true (review round 4, F14). The same failure as F5/D8, one layer down: the
    composer's staleness signal reaching a card it does not own.

    The composer copy at the start is what makes this bite: it arms `_copied`,
    so the later keystroke actually posts the stale message.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        block = await _seeded(app, pilot)
        editor = await _composer(app, pilot, "draft prompt")
        toast = app.query_one(Toast)

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 5))
        assert editor._copied, "the composer's flag must be armed for this to mean anything"

        toast.show("mcp: failed: github — command not found: gh", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()

        # A TRANSCRIPT copy, deferred behind the notice.
        await _drag(app, pilot, (block.region.x, block.region.y), (79, 23))
        assert toast._deferred is not None

        editor.focus()
        await pilot.press("x")
        await pilot.pause()
        toast.dismiss_toast()
        await pilot.pause()

        assert toast.message == "copied 3 lines"
        assert toast.display


@pytest.mark.asyncio
async def test_a_receipt_promoted_from_the_hold_can_still_be_retired() -> None:
    """A card that waited for the slot is still the composer's card.

    Tracking the receipt by `Toast.generation` named only a card that had been
    PAINTED, so one that was deferred and then promoted when the slot freed
    became unretirable — an edit could no longer falsify it. Ownership rides
    the card through both states, which is what collapses them into one case
    (review round 4).
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        toast = app.query_one(Toast)
        toast.show("mcp: failed: github — command not found: gh", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()

        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))
        toast.dismiss_toast()
        await pilot.pause()
        assert toast.message == "copied 9 characters", "the held card must have been promoted"

        await pilot.press("z")
        await pilot.pause()

        assert toast.message == ""
        assert toast.display is False


@pytest.mark.asyncio
async def test_a_new_selection_after_a_copy_does_not_rearm_the_interrupt() -> None:
    """D22, design review round 7. The deferral belongs to ONE highlight.

    Ctrl+C hands the key to an in-flight copy, and the composer's copy lands on
    mouse release — so the deferral has to outlive the release by exactly as
    long as the copy's own highlight is on screen. Testing "`_copied` and *a*
    selection" was one predicate too weak: after the copied range was collapsed
    by a caret move, an unrelated shift+arrow selection re-armed the deferral,
    so the first press aborted instead of clearing and the second QUIT with the
    draft unfiled. Two pixel-identical frames carried opposite meanings.

    Driven with a real drag, so `_copied` is set by the widget's own
    `_copy_drag` rather than by the test.
    """
    session = FakeSession()
    app = _pilot_app(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))
        assert editor._copy_gesture, "the drag should have started a copy gesture"

        # The caret moves, collapsing the copied range...
        await pilot.press("right")
        await pilot.pause()
        # ...and the user makes a NEW, unrelated selection with the keyboard.
        # The range is hand-set rather than key-driven for the same reason the
        # assertions below changed: under the explicit-copy gesture a live
        # range makes the NEXT Ctrl+C a copy, so no selection-making key can be
        # followed by a press that belongs to the draft — that press is a copy
        # now, which is the new rule this change WANTS. What D22 actually
        # protects is the gesture claim: an unrelated selection must not
        # resurrect the old copy's deferral, and the copy this press takes
        # must be the range on screen, nothing else.
        editor.selection = Selection((0, 10), (0, 20))
        await pilot.pause()
        assert (
            editor.selected_text == "the ingest"
        ), "the new selection must be live to mean anything"
        # The watcher retired the GESTURE claim when the caret moved off the
        # range it took, and an unrelated selection cannot resurrect it. The
        # receipt flag is deliberately untouched: the clipboard still holds what
        # it took, so the toast is still true (R18-1).
        assert not editor._copy_gesture, "an unrelated selection kept the gesture claim alive"
        assert editor._copied, "the receipt was destroyed along with the gesture"

        session.aborts.clear()
        app._clipboard = ""
        await pilot.press("ctrl+c")
        await pilot.pause()

        # The press is a copy — of the UNRELATED range, and only that. No
        # abort, no cleared draft, no exit ladder armed: the D22 failure was
        # all three, and each is still refused here.
        assert session.aborts == [], "an unrelated selection re-armed the copy deferral"
        assert editor.text == "summarise the ingest path please", "the draft was touched"
        assert app._clipboard == "the ingest"

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running, "the second tap quit with the draft unfiled"


@pytest.mark.asyncio
async def test_a_blurred_composer_still_paints_the_copy_it_is_deferring_to() -> None:
    """D25, design review round 8. "On screen" must not quietly become "focused".

    Ctrl+C hands the key to a copy while the highlight that copy took is still
    VISIBLE, and the user learns that rule from the pixels. Blur is the one
    place where "on screen" and "focused" come apart: the composer can lose
    focus with its selection still painted, and the deferral correctly follows
    the paint rather than the focus.

    That makes the promise true today by a property nothing pins — a future
    change to blurred-selection styling would silently turn this into the fifth
    lost draft in this rung's history. This asserts the paint, so such a change
    fails here loudly instead.
    """
    session = FakeSession()
    app = _pilot_app(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))
        assert editor._copy_gesture, "the drag should have started a copy gesture"

        def selection_is_painted() -> bool:
            """Is the composer's selection tint anywhere on its row?

            Text alone would not catch a regression here — a composer that
            stopped HIGHLIGHTING its selection still paints the same
            characters, so the STYLE is the whole subject. The tint is read
            from the live paint while focused rather than hard-coded, so a
            theme change cannot turn this into a false alarm.
            """
            row = editor.region.y
            strip = app.screen._compositor.render_strips()[row]
            return tint in {
                str(segment.style.bgcolor)
                for segment in strip
                if segment.style is not None and segment.style.bgcolor is not None
            }

        row = editor.region.y
        focused_strip = app.screen._compositor.render_strips()[row]
        tints = [
            str(segment.style.bgcolor)
            for segment in focused_strip
            if segment.style is not None and segment.style.bgcolor is not None
        ]
        assert tints, "the focused composer painted no selection tint at all"
        tint = tints[0]

        # Blur the composer WITHOUT touching the selection.
        app.set_focus(None)
        await pilot.pause()
        assert not editor.has_focus, "the composer should be blurred"
        assert editor.selected_text, "blur must not clear the selection itself"

        # The highlight the deferral promises the user is still on screen, so
        # the rule the comment states still describes what they can see.
        assert selection_is_painted(), (
            "a blurred composer stopped painting the copy's highlight, so the "
            "Ctrl+C deferral now rests on state the user cannot see"
        )


@pytest.mark.asyncio
async def test_a_receipt_retires_even_when_the_caret_moved_before_the_edit() -> None:
    """R18-1, agent review round 18. Two claims, two lifetimes, one flag each.

    The copy receipt is edit-scoped: it asserts something about text the user
    can still see, so the first edit to that text withdraws it (design round 1,
    D3). The Ctrl+C deferral is gesture-scoped: it ends when the highlight the
    copy took stops being the highlight on screen.

    Pointing the single `_copied` flag at the gesture lifetime gave the right
    answer to the second question and destroyed the first — a caret move
    retired the receipt's flag, so a later edit had nothing left to withdraw and
    the toast sat there claiming a copy of characters that no longer existed.

    Every other D3 test edits immediately after the drag with the highlight
    still up, which is the one path a selection watcher leaves alone. This is
    the path that was broken: caret move FIRST, edit second.
    """
    session = FakeSession()
    app = _pilot_app(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        await _composer_copy_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))
        assert editor._copied, "the drag should have posted a receipt"

        # The caret moves off the copied range, retiring the GESTURE claim only.
        await pilot.press("right")
        await pilot.pause()
        assert not editor._copy_gesture, "the gesture claim should have retired"
        assert editor._copied, "the receipt is still true and must survive a caret move"

        # NOW the user edits the copied characters away.
        await pilot.press("backspace")
        await pilot.pause()

        assert not editor._copied, (
            "the receipt outlived the text it describes: a caret move before the "
            "edit left nothing for the edit to withdraw"
        )
        rows = [strip.text for strip in app.screen._compositor.render_strips()]
        assert not any(
            "copied" in row.lower() for row in rows
        ), f"a stale copy receipt is still painted: {[r for r in rows if 'copied' in r.lower()]}"
