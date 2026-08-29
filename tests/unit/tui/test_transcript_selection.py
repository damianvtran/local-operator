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

import asyncio
import time
from typing import Any, cast

import pytest
from rich.cells import cell_len
from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text
from textual import events
from textual.content import Content
from textual.document._document import Selection as DocumentSelection
from textual.geometry import Offset
from textual.selection import Selection
from textual.visual import RichVisual

from local_operator.tui import theme as theme_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.glyphs import tool_icon
from local_operator.tui.markdown_theme import install_markdown_theme
from local_operator.tui.widgets import _copy_markdown
from local_operator.tui.widgets.assistant import AssistantBlock, flatten
from local_operator.tui.widgets.editor import BARREN_CLICK_WINDOW_S, Editor
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


# -- a sub-line take copies what was highlighted -----------------------------
#
# Reported from the field: dragging the eight cells of one word inside a
# rendered bullet announced ``copied 115 characters`` and put the WHOLE source
# line on the clipboard. ``AssistantBlock.get_selection`` reduced the selection
# to a first and last row and dropped the column pair ``get_span`` returns, and
# ``slice_markdown`` is row-granular by contract, so every partial row was
# widened to the source line beneath it.
#
# The rule that answers it, and the reason it is drawn where it is: there is no
# third option between "the glyphs" and "the whole source line". Column-trimmed
# markdown would need a rendered column to index a source column, and it does
# not — ``frontend`` sits at rendered column 57 and source column 58 in the
# fixture below, because ``- `` paints as `` • `` (+1) while the ``**`` vanishes
# (-2), an offset that is content-dependent and signed. So a take that does not
# cover the full content of its rows AND touches at most one source line copies
# glyphs; everything wider stays markdown.
#: The reported message: a bold word mid-bullet, plus a second bullet so a
#: multi-line drag has somewhere to go. At width 150 the first bullet renders
#: as one row, which is what makes the sub-row drag below the reported gesture.
BULLETS = (
    "Here is what I found:\n\n"
    "- Transient failures in the ingest path never reach the **frontend** "
    "without a retry, so the user sees a stale row.\n"
    "- The second bullet exists so a multi-line drag has somewhere to go.\n"
)

#: One long source line, rendered at a width that folds it across rows. The
#: wrap is the point: it is ONE source line painted as several, so a phrase
#: dragged over the fold is the same defect as one inside a single row.
WRAPPED = (
    "This is a single but quite important paragraph that will certainly "
    "wrap across several rendered rows at a narrow terminal width.\n"
)

#: A blockquote long enough to WRAP, which the module had no fixture for — and
#: that gap is why a green suite missed the furniture leak of review round 1
#: (R1-1) and design round 1 (D1). It is the one wrapped construct whose
#: continuation rows carry furniture: Rich repeats ``▌`` on every row of a
#: quote, so a sub-line take over its fold is the case where a gutter of 0
#: strips nothing.
WRAPPED_QUOTE = (
    "Here is a reply you can paste:\n\n"
    "> This is a fairly long quoted sentence that will certainly wrap across "
    "more than one rendered row at this width.\n"
)

#: A single source line holding a token wider than a narrow render segment.
#: Rich folds it MID-TOKEN and consumes nothing, unlike a fold at a space —
#: so a take across this fold must rejoin with no separator at all.
LONG_TOKEN = "See supercalifragilisticexpialidociousandthensome_verylongtoken_here now.\n"

#: A line using BOTH the open and closed form of a compound. The closed form is
#: what made the round-2 rejoin weld the open one shut: the discriminator asked
#: whether ``file`` + ``system`` occurs ANYWHERE in the line, and it does, so a
#: real space-fold was judged mid-token (review round 2, R2-1; design round 2,
#: D2-2). Position is the only thing that separates the two occurrences.
COMPOUND_PAIR = (
    "The filesystem and the file system layer are different things in this "
    "codebase, remember that.\n"
)

#: Ordinary English compound pairs, swept rather than sampled: a
#: membership-versus-position bug reproduces on whichever pair happens to fall
#: either side of the fold at a given width, so one fixture proves very little.
COMPOUND_PAIRS = (
    ("filesystem", "file system"),
    ("login", "log in"),
    ("setup", "set up"),
    ("checkout", "check out"),
    ("timeout", "time out"),
    ("backup", "back up"),
    ("workflow", "work flow"),
    ("runtime", "run time"),
    ("database", "data base"),
    ("frontend", "front end"),
    ("username", "user name"),
    ("hostname", "host name"),
    ("keyword", "key word"),
    ("website", "web site"),
    ("dropdown", "drop down"),
)

#: CJK prose. ``align`` cannot place these rows (no anchor word survives), so
#: the copy takes the ``source_line is None`` fallback — the branch that
#: returned ``" "`` unconditionally and put a space into text whose script never
#: writes one (review round 2, R2-2; design round 2, D2-3).
CJK_PARAGRAPH = "这是一个很长的中文句子用来测试换行和复制的行为是否正确无误请仔细检查每一个字符是否都被正确地复制到剪贴板中。\n"

JAPANESE_PARAGRAPH = "これは日本語の段落です。この行は端末の幅で折り返されますので、コピーの境界を確認するのに適しています。\n"

#: Emoji in Latin prose: double-width like CJK, but space-delimited, so a fold
#: here DID consume a space. The control that keeps the CJK fix from being
#: written as "wide characters never rejoin with a space" (review round 2
#: verified emoji already copied correctly and asked that it stay that way).
EMOJI_PROSE = "The status 🎉🎉🎉 report 🚀🚀🚀 with many emoji 🔥 and more text here now.\n"

#: A fenced block indented INSIDE a list item — numbered steps with a command
#: under each, one of the most common shapes this app paints. The fence sits at
#: four spaces, which a three-space-bounded ``_FENCE_RE`` did not recognise, so
#: ``classify`` saw no fence, ``furniture_width`` took its LIST branch over code
#: and read the leading ``1`` as a rendered ordered marker (design round 2,
#: D2-1). The code lines deliberately open with digits.
NESTED_FENCE = (
    "1. Start the service:\n"
    "\n"
    "    ```sh\n"
    "    1 / 0\n"
    "    docker compose up -d\n"
    "    ```\n"
    "\n"
    "2. Then check the row count:\n"
    "\n"
    "    ```sh\n"
    "    3 rows expected\n"
    "    psql -c 'select count(*) from ingest'\n"
    "    ```\n"
)

#: A bulleted list nested inside a blockquote. Rich paints ``▌  • text`` — two
#: constructs' furniture on one row — and a branch chain that returns on the
#: first match strips the bar and leaves the bullet (design round 2, D2-4).
QUOTED_LIST = (
    "> Here is a quoted list:\n"
    ">\n"
    "> - a quoted bullet item that is long enough to wrap across two rows\n"
)


def _rendered_rows(block: AssistantBlock) -> list[str]:
    """The frame's rows, as ``get_selection`` and the highlight both see them."""
    visual = block._render()
    assert isinstance(visual, Content)
    return visual.plain.split("\n")


def _find(rows: list[str], word: str) -> tuple[int, int, int]:
    """``(row, start, end)`` of ``word`` in the RENDERED frame.

    Located rather than hard-coded because the whole point of these tests is
    that rendered columns are not source columns: a literal column here would
    encode the very assumption the fix exists to deny, and would drift with any
    change to how the markdown theme paints a bullet.
    """
    for index, row in enumerate(rows):
        if word in row:
            start = row.index(word)
            return index, start, start + len(word)
    raise AssertionError(f"{word!r} is not on the frame: {rows!r}")


@pytest.mark.asyncio
async def test_a_word_dragged_out_of_a_bullet_copies_only_that_word() -> None:
    """The reported bug: 8 cells highlighted must not copy 115 characters.

    Pins that a sub-row take is no longer widened to its source line. The
    negative assertion is the one that fails on the old code — the word alone
    is a substring of the over-copy, so equality is what makes this a
    regression test rather than a smoke test.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(150, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(BULLETS)
        block.finalize_text()
        await pilot.pause()

        row, start, end = _find(_rendered_rows(block), "frontend")
        selection = Selection.from_offsets(Offset(x=start, y=row), Offset(x=end, y=row))
        copied = block.get_selection(selection)

        assert copied is not None
        assert copied == ("frontend", "\n")
        # The rest of the source line, which the old code copied wholesale.
        assert "Transient failures" not in copied[0]
        # The receipt the user reads is a count of exactly this string.
        assert len(copied[0]) == 8


@pytest.mark.asyncio
async def test_a_phrase_across_a_wrap_boundary_copies_only_the_phrase() -> None:
    """A wrapped paragraph is ONE source line, so a drag over its fold is sub-line.

    Pins that the gate counts SOURCE LINES and not rendered rows. A row-count
    gate (``len(content) == 1``) fixes the reported case and leaves this one
    live: measured at width 60 it still returned all 128 characters of the
    paragraph for a two-row drag. Anyone narrowing the gate has to fail here.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(60, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(WRAPPED)
        block.finalize_text()
        await pilot.pause()

        rows = _rendered_rows(block)
        first_row, start, _ = _find(rows, "important")
        last_row, _, end = _find(rows, "several")
        assert first_row != last_row, "the fixture must actually wrap, or this proves nothing"

        selection = Selection.from_offsets(Offset(x=start, y=first_row), Offset(x=end, y=last_row))
        copied = block.get_selection(selection)

        assert copied is not None
        # Rejoined with the space the terminal consumed at the fold, not with
        # the fold itself: the rows share one source line, so the break is an
        # artifact of this width (design round 1, D2).
        assert copied[0] == "important paragraph that will certainly wrap across several"
        assert "\n" not in copied[0]
        assert copied[0] in WRAPPED
        assert "This is a single" not in copied[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["exact", "into-the-pad", "sentinel"])
async def test_a_whole_row_still_copies_markdown_source(form: str) -> None:
    """Full coverage keeps the markdown path, however the drag reported its end.

    ``end`` arrives three ways — the exact glyph count, a column inside Rich's
    trailing pad when the drag overran the last glyph, and ``-1`` for "to end of
    row" — and all three mean the same thing to a reader. Pins the ``rstrip()``
    predicate: Rich pads each row to its RENDER SEGMENT's width (measured 146
    for this bullet against 112 glyphs), so a predicate against the block width
    or raw ``len(row)`` would call the exact-end case partial and degrade a
    whole-line copy to rendered text.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(150, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(BULLETS)
        block.finalize_text()
        await pilot.pause()

        rows = _rendered_rows(block)
        row, _, _ = _find(rows, "frontend")
        ends = {"exact": len(rows[row].rstrip()), "into-the-pad": len(rows[row]), "sentinel": -1}
        # Hand-built rather than via ``from_offsets``, which normalises a -1 x
        # into the selection START and so cannot express the sentinel form.
        selection = cast("Any", _SpanOnRow(row, (0, ends[form])))

        copied = block.get_selection(selection)
        assert copied is not None
        assert copied[0] == (
            "- Transient failures in the ingest path never reach the **frontend** "
            "without a retry, so the user sees a stale row."
        )


class _SpanOnRow:
    """A selection of one hand-set span on one row.

    ``Selection.from_offsets`` normalises its offsets, which makes the ``-1``
    end sentinel unreachable through the public constructor — it is a value
    Textual's own ``get_span`` PRODUCES for the rows in the middle of a
    multi-row drag, not one an offset pair can state. Only ``get_span`` is
    called by ``get_selection``, so duck-typing it is enough and keeps the test
    honest about which of the three end forms it is exercising.
    """

    def __init__(self, row: int, span: tuple[int, int]) -> None:
        self._row = row
        self._span = span

    def get_span(self, y: int) -> tuple[int, int] | None:
        return self._span if y == self._row else None


@pytest.mark.asyncio
async def test_a_sub_row_take_inside_a_fence_copies_the_code_not_the_fence() -> None:
    """Part of a code line copies that code, unfenced.

    Pins that the glyph path does not re-fence a partial take: ``slice_markdown``
    would return a three-line fenced block for a four-cell drag, which is both
    more than was highlighted and not the snippet the reader pointed at.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(64, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text("```python\ndef f(x):\n    return x + 1\n```")
        block.finalize_text()
        await pilot.pause()

        row, start, end = _find(_rendered_rows(block), "f(x)")
        selection = Selection.from_offsets(Offset(x=start, y=row), Offset(x=end, y=row))

        assert block.get_selection(selection) == ("f(x)", "\n")


@pytest.mark.asyncio
async def test_a_partial_multi_line_selection_still_copies_markdown() -> None:
    """Two bullets with partial ends copy BOTH bullets whole. Deliberately.

    This is the accepted cost of the rule, pinned so it cannot be "fixed" into
    a degrade-anywhere rule without confronting the choice. Trimming the ends
    would need a rendered-to-source column mapping that does not exist, and
    degrading the whole take to glyphs would put the ``•`` and ``▌`` furniture
    back into the paste — which is exactly the report that
    ``test_blockquote_copies_as_markdown_not_the_bar`` guards against.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(150, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(BULLETS)
        block.finalize_text()
        await pilot.pause()

        first_row, _, _ = _find(_rendered_rows(block), "frontend")
        # Mid-way through the first bullet to mid-way through the second.
        selection = Selection.from_offsets(Offset(x=10, y=first_row), Offset(x=20, y=first_row + 1))
        copied = block.get_selection(selection)

        assert copied is not None
        assert copied[0].splitlines() == [
            "- Transient failures in the ingest path never reach the **frontend** "
            "without a retry, so the user sees a stale row.",
            "- The second bullet exists so a multi-line drag has somewhere to go.",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [MARKDOWN, BULLETS], ids=["mixed-constructs", "bullets"])
async def test_a_whole_message_copy_is_byte_identical_markdown(text: str) -> None:
    """Select-all still returns the source verbatim, not a rendered flattening.

    The regression this pins is the one a sub-line rule is most likely to cause
    by accident: the glyph path is strictly better for a partial take and
    strictly WORSE for a whole one, so a gate that is even slightly too eager
    degrades every select-all to rendered text — bullets as ``•``, bold with
    its ``**`` stripped, the fence unfenced. Asserted as byte equality rather
    than by substring (which the older markdown tests use) because a leaked
    glyph path passes every substring check while dropping the blank separator
    rows and the trailing pad, and equality is the only predicate that sees it.

    ``BULLETS`` is included alongside ``MARKDOWN`` because it is the fixture
    the sub-line tests drag inside of: the same message must copy whole when
    the whole of it is taken.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(150, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(text)
        block.finalize_text()
        await pilot.pause()

        # ``update_text`` keeps the message's own trailing newline, which is not
        # part of any rendered row and so is not part of the copy.
        assert _copy_all(app, block) == text.rstrip("\n")


@pytest.mark.asyncio
async def test_a_word_inside_a_blockquote_copies_without_the_bar() -> None:
    """A sub-row quote take is the word, not the ``▌`` and not the ``>`` line.

    The counterpart to ``test_blockquote_copies_as_markdown_not_the_bar``: a
    drag that starts PAST the bar never crosses that cell, so the bar is not
    part of what was highlighted and is not part of the copy.
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

        row, start, end = _find(_rendered_rows(block), "flagged")
        selection = Selection.from_offsets(Offset(x=start, y=row), Offset(x=end, y=row))
        copied = block.get_selection(selection)

        assert copied is not None
        assert copied == ("flagged", "\n")
        assert "▌" not in copied[0]
        assert ">" not in copied[0]


@pytest.mark.asyncio
async def test_a_quote_dragged_from_column_zero_drops_the_bar() -> None:
    """A column-0 quote drag copies the QUOTED TEXT, never the ``▌``.

    This pinned the opposite until review round 1 (R1-1) and design round 1
    (D1) independently rejected it: the bar is painted decoration, it exists
    nowhere in the user's document, and a paste carrying it shows a glyph they
    cannot account for. It also made the paste FORMAT depend on one invisible
    cell of drag start — column 0 gave rendered glyphs with the bar, column 1
    gave clean text — which is the surprise the whole method exists to remove.

    The rejected fix was a regex over the row, and rejecting it was right: it
    CORRUPTS CODE, turning a fenced ``  1 / 0`` into ``/ 0`` because a code
    line starting with a digit is indistinguishable from an ordered marker by
    glyph alone. The mapping-aware gutter it named as the safe alternative is
    what now runs — ``_copy_markdown.furniture_width`` asks which SOURCE LINE
    the row came from, so a fence is refused explicitly rather than by luck
    (pinned by ``test_a_column_zero_drag_inside_a_fence_keeps_the_code``).
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(80, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text("Intro:\n\n> Thanks for the report, it is fixed now.\n")
        block.finalize_text()
        await pilot.pause()

        rows = _rendered_rows(block)
        row = next(i for i, text in enumerate(rows) if text.startswith("▌"))
        selection = Selection.from_offsets(Offset(x=0, y=row), Offset(x=16, y=row))
        copied = block.get_selection(selection)

        assert copied is not None
        assert copied[0] == "Thanks for the"
        assert "▌" not in copied[0]


@pytest.mark.asyncio
async def test_a_sub_line_take_across_a_wrapped_quote_never_copies_the_bar() -> None:
    """The R1-1/D1 leak: a drag over a wrapped quote's fold must not paste ``▌``.

    The regression both reviewers found independently, and the case the module
    had no fixture for. A wrapped blockquote is ONE source line painted as
    several rows, so it passes the source-line gate onto the glyph path — and
    every one of those rows carries a ``▌``, because Rich repeats the bar on
    continuations. With the block's inherited ``copy_gutter`` of 0 the clamp
    stripped nothing and the bar reached the clipboard, which the markdown path
    it replaced had never done.

    Two assertions rather than one: no bar (the leak) and no newline (D2, since
    this take also crosses a soft wrap), because the fold is where both defects
    surface at once.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(60, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(WRAPPED_QUOTE)
        block.finalize_text()
        await pilot.pause()

        rows = _rendered_rows(block)
        bars = [i for i, row in enumerate(rows) if row.startswith("▌")]
        assert len(bars) >= 2, f"the quote must actually wrap, or this proves nothing: {rows!r}"

        selection = Selection.from_offsets(Offset(x=10, y=bars[0]), Offset(x=12, y=bars[1]))
        copied = block.get_selection(selection)

        assert copied is not None
        assert "▌" not in copied[0]
        assert "\n" not in copied[0]
        assert copied[0] == "a fairly long quoted sentence that will certainly"


@pytest.mark.asyncio
async def test_a_column_zero_drag_inside_a_bullet_drops_the_marker() -> None:
    """A column-0 bullet drag copies the item's text, never the ``•``.

    The bullet half of D1, identical in mechanism to the quote bar and called
    out in review round 1 (R1-3) as the case the documentation named nowhere.
    Pinned separately because the two take different branches of
    ``furniture_width``: a quote strips a repeated bar, a list strips a marker
    on the row that opens the item.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(150, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(BULLETS)
        block.finalize_text()
        await pilot.pause()

        rows = _rendered_rows(block)
        row, _, _ = _find(rows, "Transient")
        selection = Selection.from_offsets(Offset(x=0, y=row), Offset(x=36, y=row))
        copied = block.get_selection(selection)

        assert copied is not None
        assert copied[0] == "Transient failures in the ingest"
        assert "•" not in copied[0]


@pytest.mark.asyncio
async def test_a_column_zero_drag_inside_a_fence_keeps_the_code() -> None:
    """Furniture stripping must never touch code — the reason it is mapping-aware.

    The hazard the rejected regex fix would have caused, pinned so the cheap
    implementation cannot come back: a fenced line beginning with a digit looks
    exactly like a rendered ordered-list marker, and a glyph-level strip turns
    ``1 / 0`` into ``/ 0``. ``furniture_width`` refuses on the SOURCE line's
    fence membership, so the code copies byte for byte including its indent.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(60, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text("```python\nif x:\n    1 / 0  # boom\n```\n")
        block.finalize_text()
        await pilot.pause()

        rows = _rendered_rows(block)
        row = next(i for i, text in enumerate(rows) if "1 / 0" in text)
        # Column 0 to just past the ``0``, stopping short of the comment: a
        # column-0 start is what reaches the furniture clamp, and leaving the
        # comment out is what keeps this on the sub-line path rather than
        # widening to the whole fenced block.
        end = rows[row].index("/ 0") + len("/ 0")
        selection = Selection.from_offsets(Offset(x=0, y=row), Offset(x=end, y=row))
        copied = block.get_selection(selection)

        assert copied is not None
        # The digit survives: a glyph-level strip would read `` 1 `` as an
        # ordered-list marker and hand back ``/ 0``.
        assert copied[0].strip() == "1 / 0"
        assert "#" not in copied[0]


@pytest.mark.asyncio
async def test_a_take_across_a_mid_token_fold_rejoins_with_no_space() -> None:
    """A fold inside a long token consumed nothing, so the rejoin adds nothing.

    The counterpart to the wrap-fold rejoin, and the case a blanket
    ``" ".join`` would corrupt: Rich breaks a token wider than the render
    segment mid-token, so putting a space back would invent a character the
    document never had and silently break a pasted URL or identifier. Measured
    reachable in ordinary prose at ordinary widths, which is why
    ``wrap_separators`` walks the source line instead of assuming a space.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(34, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(LONG_TOKEN)
        block.finalize_text()
        await pilot.pause()

        rows = _rendered_rows(block)
        content = [i for i, row in enumerate(rows) if row.strip()]
        first = next(i for i in content if "supercal" in rows[i])
        last = next(i for i in content if "here now." in rows[i])
        assert first != last, "the token must actually fold, or this proves nothing"

        selection = Selection.from_offsets(
            Offset(x=rows[first].index("supercal"), y=first),
            Offset(x=rows[last].index("here") + len("here"), y=last),
        )
        copied = block.get_selection(selection)

        assert copied is not None
        assert copied[0] == "supercalifragilisticexpialidociousandthensome_verylongtoken_here"
        assert copied[0] in LONG_TOKEN
        assert "\n" not in copied[0]


@pytest.mark.asyncio
async def test_a_whole_table_row_take_keeps_the_markdown_pipes() -> None:
    """The escape hatch that makes D5 an accepted cost rather than a defect.

    A sub-line take across a table row copies ``alpha  0.91`` without the
    ``|``, which design round 1 (D5) flagged as the one construct where the
    rendered form is less useful than the source. It stays the rule because the
    WHOLE-row gesture still reaches the markdown path and still yields a
    pasteable row. Pinned so a future widening of the sub-line gate cannot take
    that fallback away silently.
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

        rows = _rendered_rows(block)
        row, _, _ = _find(rows, "alpha")
        selection = Selection.from_offsets(Offset(x=0, y=row), Offset(x=len(rows[row]), y=row))
        copied = block.get_selection(selection)

        assert copied is not None
        assert copied[0] == "| alpha | 0.91 |"


@pytest.mark.asyncio
async def test_a_drag_entirely_inside_the_trailing_pad_copies_nothing() -> None:
    """A drag over Rich's pad selects no glyph, so nothing is written. Decided.

    Review round 1 (R1-2) asked whether the silence is right. It is, and it is
    pinned here rather than left implicit: the pad is not content the reader can
    see, so there is nothing truthful to copy, and ``_put_on_clipboard`` drops
    an empty payload without a receipt — the same answer a zero-width click
    already gets. A toast here would announce a copy that did not happen.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(150, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(BULLETS)
        block.finalize_text()
        await pilot.pause()

        rows = _rendered_rows(block)
        row, _, _ = _find(rows, "Transient")
        visible_end = len(rows[row].rstrip())
        assert visible_end < len(rows[row]), "the row must carry a pad, or this proves nothing"

        selection = Selection.from_offsets(
            Offset(x=visible_end + 5, y=row), Offset(x=visible_end + 20, y=row)
        )
        copied = block.get_selection(selection)

        assert copied is not None
        assert copied[0] == ""


@pytest.mark.asyncio
async def test_a_table_cell_copies_the_cell() -> None:
    """One cell of a rendered table copies that cell, not the pipe row.

    Pins that a sub-row take does not reintroduce markdown furniture the reader
    cannot see: the frame draws no pipes, so a copy carrying ``| alpha | 0.91 |``
    would contain characters no highlighted cell held.
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

        row, start, end = _find(_rendered_rows(block), "alpha")
        selection = Selection.from_offsets(Offset(x=start, y=row), Offset(x=end, y=row))
        copied = block.get_selection(selection)

        assert copied is not None
        assert copied == ("alpha", "\n")
        assert "|" not in copied[0]


@pytest.mark.asyncio
async def test_a_zero_width_selection_copies_nothing() -> None:
    """A click is not a drag: an empty span copies an empty string.

    Pins the click case, which under the old row-only reduction returned the
    whole source line — a plain click on an answer would have put 115
    characters on the clipboard. ``_put_on_clipboard`` drops a falsy payload
    before the OSC 52 write, so this is also what keeps a click from raising a
    ``copied 0 characters`` receipt.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(150, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(BULLETS)
        block.finalize_text()
        await pilot.pause()

        row, start, _ = _find(_rendered_rows(block), "frontend")
        selection = Selection.from_offsets(Offset(x=start, y=row), Offset(x=start, y=row))

        assert block.get_selection(selection) == ("", "\n")


@pytest.mark.asyncio
async def test_the_reported_drag_end_to_end_reports_a_small_count() -> None:
    """The whole gesture through the real app: drag a word, read the receipt.

    The unit assertions above pin ``get_selection``; this pins what the USER
    sees, which is where the bug was reported from. It goes through the real
    mouse events, the real ``TextSelected`` release, the real clipboard write
    and the real toast, because the receipt is the only part of the copy the
    reader can check — ``copied 115 characters`` for an 8-cell drag was the
    report.
    """
    app = _pilot_app()
    async with app.run_test(size=(150, 26)) as pilot:
        await pilot.pause()
        app._append_block(UserBlock("summarise the ingest path"))
        block = AssistantBlock()
        app._append_block(block)
        await pilot.pause()
        block.update_text(BULLETS)
        block.finalize_text()
        await pilot.pause()
        await pilot.pause()
        app._clipboard = ""

        row, start, end = _find(_rendered_rows(block), "frontend")
        y = block.region.y + row
        await _drag(app, pilot, (block.region.x + start, y), (block.region.x + end, y))

        assert app._clipboard == "frontend"
        assert app.query_one(Toast).message == "copied 8 characters"


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
# The first fix copied on release, matching the transcript. That clobbered
# the clipboard on every select-to-edit drag — the reported follow-up. The
# composer now copies only on an explicit Ctrl+C (`Editor.action_copy` ->
# `EditorCopied`); a drag selects and nothing more. The app still answers
# through the same clipboard write and the same toast as the transcript.


async def _composer_drag(
    app: OperatorApp,
    pilot: Any,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    """A drag over the composer, as `Screen._forward_event` receives it.

    Separate from `_drag` only in that it does not assert a screen selection
    afterwards: over a `TextArea` there is never going to be one, which is the
    whole reason this path exists. A composer drag never copies — that is the
    rule this file exists to pin — so there is no `copy=` switch.
    """
    app.screen._forward_event(_mouse(app, events.MouseDown, *start))
    await pilot.pause()
    if start != end:
        app.screen._forward_event(_mouse(app, events.MouseMove, *end))
        await pilot.pause()
    app.screen._forward_event(_mouse(app, events.MouseUp, *end))
    await pilot.pause()
    await pilot.pause()


async def _composer_copy(
    app: OperatorApp,
    pilot: Any,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    """The composer's copy gesture: drag to select, then Ctrl+C.

    The drag is how a user makes the range (and how these tests' coordinates
    were written); it copies nothing. The press is the copy. Tests that used
    to drive an armed drag keep their coordinates and their assertions about
    WHAT was taken.
    """
    await _composer_drag(app, pilot, start, end)
    await pilot.press("ctrl+c")
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
async def test_ctrl_c_with_a_live_range_copies_it() -> None:
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
        editor.selection = DocumentSelection((0, 0), (0, 20))
        await pilot.pause()

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app._clipboard == "summarise the ingest"
        # The explicit copy also kept the draft and raised no interrupt: it
        # consumed the press entirely.
        assert editor.text == "summarise the ingest path please"
        toast = app.query_one(Toast)
        assert toast.message == "copied 20 characters"

        # And a subsequent drag copies NOTHING — there is no armed-next-drag
        # mode. Review round 1 F1: the arm outlived the highlight that
        # authorised it and clobbered the clipboard on a later select-to-edit.
        # Design round 1 D2: it was a hidden mode with nothing on screen to
        # show it was on. Dropped rather than patched.
        await _composer_drag(app, pilot, _cell(editor, 0, 24), _cell(editor, 0, 34))
        assert app._clipboard == "summarise the ingest"


@pytest.mark.asyncio
async def test_a_drag_after_an_explicit_copy_still_copies_nothing() -> None:
    """F1, review round 1. There is no armed-next-drag mode.

    An earlier version of this change armed the NEXT drag to copy on release
    after an explicit Ctrl+C, so a user taking several passages could keep
    dragging. The arm never retired when the copied highlight left the
    screen (the press did not set `_copy_gesture`, and `watch_selection`
    only disarmed inside that branch), so a later select-to-edit clobbered
    the clipboard — the original defect, one keystroke later. Dropped: a
    composer copy is always the explicit press.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))
        assert app._clipboard == "summarise the ingest"

        await pilot.press("right")
        await pilot.pause()
        app._clipboard = "SOMETHING THE USER PUT THERE"
        await _composer_drag(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))
        assert app._clipboard == "SOMETHING THE USER PUT THERE"


@pytest.mark.asyncio
async def test_an_explicit_composer_copy_takes_what_was_highlighted() -> None:
    """Highlight, Ctrl+C, the clipboard IS the highlight.

    The property the transcript's `get_selection` gets from sharing one
    computation, and which this path has to get by taking the widget's own
    selected text. The screen still has no selection of its own, so this
    could only have come from the editor.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))

        assert editor.selected_text == "summarise the ingest"
        assert app._clipboard == editor.selected_text
        assert not app.screen.selections


async def _composer_multi_click(
    app: OperatorApp,
    pilot: Any,
    editor: Editor,
    column: int,
    times: int,
    row: int = 0,
) -> None:
    """A double/triple click in the composer, through the CLICK path.

    Deliberately `pilot.click(times=...)` rather than the hand-built
    MouseDown/Move/Up of `_composer_drag`. The distinction is the whole reason
    the double-click defect survived a green suite: `_composer_drag` posts the
    three mouse events and stops, but a real terminal ALSO produces a `Click`
    carrying a chain count, and `Widget._on_click` is what reacts to that
    chain. A helper that never emits a Click cannot model the gesture the field
    report was about, so these tests use the pilot's own click, which builds
    the chain the driver builds.

    The composer's PLACEMENT settles several frames after its text does. A
    multi-line draft grows the widget, which moves the whole input dock up the
    screen, and `region.y` was still migrating (15 -> 16 here) for up to five
    pauses after `_composer` returned. `pilot.click` resolves the widget-
    relative offset against the region AT CLICK TIME, so a click aimed one
    frame early lands a row high: a row-1 click silently became a row-0 click
    and the assertion then described a gesture the test never made.

    Worse, `pilot.click` resolves the offset to SCREEN coordinates ONCE and
    then pauses between each event of the chain, so a dock that moves partway
    through sends the later clicks of a triple-click to a coordinate that is no
    longer the row they were aimed at. Waiting for the region to hold still
    across several consecutive frames — not merely to differ from the last one
    — is what makes a row-aimed click land where it says.
    """
    stable = 0
    previous = None
    for _ in range(40):
        await pilot.pause()
        current = editor.region
        stable = stable + 1 if current == previous and editor.size.height > row else 0
        previous = current
        if stable >= 8:
            break
    assert editor.size.height > row, f"the composer never grew to row {row}"
    offset = (editor.gutter.left + column, editor.gutter.top + row)

    # VERIFY THE AIM, do not merely wait for it. `stable >= 4` was not always
    # enough for the taller fixtures: the dock was still migrating when
    # `pilot.click` resolved its offset, so a row-1 click landed on row 0 about
    # 14% of the time and the D2 regression test failed intermittently while the
    # product was correct (design review round 2, D2-2). A flaky test pinning a
    # data-loss fix is worse than no test, because the next real regression
    # reads as "that one's just flaky".
    #
    # Recording where the click will ACTUALLY land — through the same resolver
    # the handler uses — turns a mis-aimed gesture into a loud setup failure
    # naming the row it hit, instead of a confusing assertion about a selection
    # the test never really made.
    landed: list[int] = []
    original = type(editor)._on_click

    async def _record(self: Editor, event: Any) -> None:
        landed.append(self.get_target_document_location(event)[0])
        await original(self, event)

    type(editor)._on_click = _record
    try:
        await pilot.click(editor, offset=offset, times=times)
        await pilot.pause()
        await pilot.pause()
    finally:
        type(editor)._on_click = original
    assert landed, "the click never reached the editor's chain handler"
    assert landed[-1] == row, (
        f"the click was aimed at row {row} but landed on row {landed[-1]} — "
        "the composer dock moved between resolving the offset and delivering "
        "the click, so this run tested a different gesture"
    )


@pytest.mark.asyncio
async def test_double_click_selects_the_word_under_the_pointer() -> None:
    """The reported defect: double-click made no selection the composer could copy.

    Measured under a real pty before the fix: `Widget._on_click` answered a
    chain of 2 with `text_select_all()`, which writes a SCREEN selection —
    `{Editor(): Selection(None, None)}`. For a `TextArea` that entry is inert
    twice over: it paints nothing (the widget renders its own lines rather
    than a `Content`), and `Widget.get_selection` yields no text for it. The
    document selection stayed collapsed, so `selected_text` was `''` and the
    highlight the user believed they had made did not exist.

    Pins the document meaning of the gesture: the word, taken from the widget's
    own `_word_pattern` boundaries, in `TextArea.selection` where both the
    painted highlight and `selected_text` read it.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")

        # Column 18 sits inside "ingest" (columns 14-20).
        await _composer_multi_click(app, pilot, editor, 18, times=2)

        assert editor.selection == DocumentSelection((0, 14), (0, 20))
        assert editor.selected_text == "ingest"
        # The inert screen selection is gone rather than merely ignored: an
        # entry there makes the next mouse-down clear it and puts this widget
        # in the map the transcript-copy path walks.
        assert not app.screen.selections


@pytest.mark.asyncio
async def test_triple_click_selects_the_whole_composer_line() -> None:
    """Chain 3 takes the line, the gesture's meaning in every other editor.

    Before the fix this was worse than chain 2: `Widget._on_click` escalated to
    the CONTAINER, so `Screen.selections` picked up the input row and the
    chrome around it and a copy took the frame's furniture — captured on a real
    pty as a clipboard holding `'\\ndraft cleared — ↑ to recover\\n…◆ test/model
    › ⌂ …'`. The composer's own line was the one thing not in it.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")

        await _composer_multi_click(app, pilot, editor, 18, times=3)

        assert editor.selection == DocumentSelection((0, 0), (0, 32))
        assert editor.selected_text == "summarise the ingest path please"
        assert not app.screen.selections


@pytest.mark.asyncio
async def test_double_click_then_ctrl_c_copies_and_keeps_the_draft() -> None:
    """THE reported sequence, end to end: highlight by double-click, then Ctrl+C.

    This is what the user actually did, and before the fix it did not merely
    fail to copy — it DESTROYED THE PROMPT. With no live range, Ctrl+C fell
    through to the interrupt rung, whose first tap clears the draft. The report
    ("I can't properly copy via ctrl/cmd+C on highlighted text in the
    composer") understates it: the gesture cost the user their text.

    Pins all three outcomes, because a fix that restored the copy while still
    clearing the draft would satisfy a narrower assertion.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        app._clipboard = "SOMETHING THE USER PUT THERE"
        toast = app.query_one(Toast)

        await _composer_multi_click(app, pilot, editor, 18, times=2)
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app._clipboard == "ingest"
        assert toast.message == "copied 6 characters"
        assert editor.text == "summarise the ingest path please"


@pytest.mark.asyncio
async def test_a_multi_click_does_not_copy_on_its_own() -> None:
    """Selecting is not copying, and the click must not write the clipboard.

    The same rule `_copy_drag` enforces for a drag and `_on_mouse_up` for a
    marker click: a composer highlight is usually the first half of an edit, so
    a gesture that merely SELECTS must leave the clipboard alone. Without this,
    the double-click fix would reintroduce the original field report — "the
    copy can end up clearing something you have in the clipboard" — through a
    new gesture.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        app._clipboard = "SOMETHING THE USER PUT THERE"
        toast = app.query_one(Toast)

        await _composer_multi_click(app, pilot, editor, 18, times=2)
        await _composer_multi_click(app, pilot, editor, 18, times=3)

        assert app._clipboard == "SOMETHING THE USER PUT THERE"
        assert toast.message == ""


@pytest.mark.asyncio
async def test_a_single_click_still_only_places_the_caret() -> None:
    """Chain 1 is a caret move, not a selection — the commonest gesture here.

    The guard the word/line handler needs: a fix that widened EVERY click
    would make placing the caret select a word, so the next character typed
    would replace it. Pins the collapsed selection a plain click leaves.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")

        await _composer_multi_click(app, pilot, editor, 18, times=1)

        assert editor.selection.start == editor.selection.end
        assert editor.selected_text == ""


@pytest.mark.asyncio
async def test_ctrl_c_after_a_single_click_still_reaches_the_interrupt() -> None:
    """The invariant the fix must not trade away (D17).

    Ctrl+C's interrupt meaning cannot become conditional on a highlight, and
    the exit ladder's first rung is what clears the draft. A single click
    leaves no range, so the press must still reach the interrupt — the same
    behaviour as before this change, asserted here because the new handler runs
    on the same event that decides it.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")

        await _composer_multi_click(app, pilot, editor, 18, times=1)
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert editor.text == ""


@pytest.mark.asyncio
async def test_double_click_on_whitespace_takes_the_gap_not_a_neighbour() -> None:
    """A click on a gap selects the gap, never a word the user did not point at.

    `_word_span` splits on `TextArea._word_pattern`, so a run of spaces is its
    own span. Silently snapping to the word on either side would put text on
    the clipboard that the pointer was not over — the same class of surprise as
    copying without being asked.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")

        # Column 9 is the single space between "summarise" and "the".
        await _composer_multi_click(app, pilot, editor, 9, times=2)

        assert editor.selected_text == " "


@pytest.mark.asyncio
async def test_double_click_on_an_empty_composer_selects_nothing() -> None:
    """The placeholder is not the user's text, so there is nothing to take.

    An empty composer still paints `Message Local Operator…`. The word span is
    computed from the DOCUMENT, which is empty, so the gesture yields an empty
    range and the Ctrl+C that may follow keeps its interrupt meaning.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "")

        await _composer_multi_click(app, pilot, editor, 4, times=2)

        assert editor.selected_text == ""


@pytest.mark.asyncio
async def test_a_fourth_click_keeps_the_line_selected() -> None:
    """Clicking on past three does not flicker the highlight off.

    `CLICK_CHAIN_TIME_THRESHOLD` keeps counting while the user keeps clicking
    in place, so chain 4 is reachable by accident. Folding it onto the line
    keeps the selection stable rather than collapsing it, which would read as
    the highlight dropping out on its own.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")

        await _composer_multi_click(app, pilot, editor, 18, times=4)

        assert editor.selected_text == "summarise the ingest path please"


@pytest.mark.asyncio
async def test_a_multi_click_on_a_blank_line_leaves_a_live_range() -> None:
    """A blank line between paragraphs must not answer the gesture with nothing.

    Regression for design round 1, D2. Paragraphs split by an empty line are
    the commonest shape of a real prompt. Both chains used to fall through to a
    collapsed range there, so the frame after the gesture was byte-identical to
    the frame before it — the "the highlight did not exist" signature of the
    original defect — and the Ctrl+C that followed found no range and CLEARED
    THE DRAFT.

    The range taken is the row's own line break, the only character the line
    has, so the gesture answers with something real and the draft survives the
    press. Pinned for both chains because they reach it by different branches.
    """
    for times in (2, 3):
        app = _pilot_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            editor = await _composer(
                app, pilot, "first paragraph of my prompt\n\nsecond paragraph here"
            )
            app._clipboard = "SOMETHING THE USER PUT THERE"

            await _composer_multi_click(app, pilot, editor, 0, times=times, row=1)

            assert editor.selection == DocumentSelection((1, 0), (2, 0))
            assert editor.selected_text == "\n"

            # The press the user makes next: it copies the range instead of
            # falling through to the rung that scraps the draft.
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app._clipboard == "\n"
            assert editor.text == "first paragraph of my prompt\n\nsecond paragraph here"


@pytest.mark.asyncio
async def test_a_multi_click_on_the_last_empty_row_stays_collapsed() -> None:
    """The trailing blank row has no line break to take, so it takes nothing.

    The bound on the D2 fix. A range on the final row would have to run
    BACKWARDS into the previous line and take a character the user never
    pointed at, so the selection stays collapsed there — which also keeps the
    empty composer correct (design round 1, D7 verified that as good, and it
    must not regress).
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "a draft with a trailing newline\n")

        await _composer_multi_click(app, pilot, editor, 0, times=2, row=1)

        assert editor.selection.start == editor.selection.end
        assert editor.selected_text == ""


@pytest.mark.asyncio
async def test_a_deliberately_slow_double_click_is_still_a_double_click() -> None:
    """A gesture 0.7 s apart selects, rather than eating the draft.

    Regression for design round 1, D3. Textual's default
    `CLICK_CHAIN_TIME_THRESHOLD` is 0.5 s, so two clicks 0.7 s apart were not a
    chain at all: no selection, and the Ctrl+C pressed next found no range and
    fell through to the rung that clears the composer. The user's intent is the
    same in both cases and nothing on the frame tells them which one they made,
    so the outcome must not hinge on 200 ms they cannot perceive.

    Driven through `App.on_event` rather than `pilot.click(times=...)`: the
    pilot builds the `Click` with the chain count already decided, so it cannot
    exercise the arithmetic under test. Real MouseDown/MouseUp pairs are what
    the chain is computed from.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        app._clipboard = "SOMETHING THE USER PUT THERE"

        async def press_and_release() -> None:
            """One physical click, as the driver delivers it."""
            x = editor.region.x + editor.gutter.left + 18
            y = editor.region.y + editor.gutter.top
            arguments: dict[str, Any] = dict(
                widget=None,
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
            await app.on_event(events.MouseDown(**arguments))
            await app.on_event(events.MouseUp(**arguments))
            await pilot.pause()

        await press_and_release()
        # Longer than Textual's 0.5 s default, inside this app's own threshold.
        await asyncio.sleep(0.7)
        await press_and_release()
        await pilot.pause()

        assert editor.selected_text == "ingest", "the slow pair was not read as a chain"

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app._clipboard == "ingest"
        assert editor.text == "summarise the ingest path please", "the draft was cleared"


@pytest.mark.asyncio
async def test_ctrl_c_still_interrupts_after_a_stale_multi_click_range() -> None:
    """The exit ladder stays reachable after a REFLEXIVE double-click.

    Regression for agent review round 1, R1-2. `action_copy`'s rule is that a
    live range makes the press a copy, and that is deliberate for a range the
    user MADE deliberately (shift+arrow, a drag). A double-click in an input
    box is reflexive, so this change put a persistent range one stray gesture
    away — and with it, Ctrl+C could never reach the interrupt while the agent
    was working, at any number of presses.

    The measured ladder after this fix, re-measured in agent review round 2
    (R2-2) because round 1's own figures did not reproduce: press 1 copies,
    press 2 clears the draft, press 3 interrupts, press 4 interrupts and exits.
    An untouched composer interrupts on press 2 and exits on press 3, so the
    accepted residual cost is ONE EXTRA TAP for a user who reflexively
    double-clicked mid-turn.

    The rule is kept, and the ladder is restored by the SECOND press: the first
    copies the range and collapses the caret to its end, which is the same
    "collapsing the caret hands the key back" escape `action_copy` documents,
    now performed by the copy itself rather than requiring the user to know
    about it. One press is still a copy, so nothing about the explicit-copy
    gesture changes.
    """
    session = FakeSession()
    app = _pilot_app(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        session.streaming = True
        await pilot.pause()

        await _composer_multi_click(app, pilot, editor, 18, times=2)
        assert editor.selected_text == "ingest", "the gesture must leave a range"

        # First press: the copy the range earns.
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app._clipboard == "ingest"
        assert session.aborts == [], "the copy must not also interrupt"
        assert not editor.selected_text, "the copy must hand the key back"

        # Second press: the key means what it means with no range live.
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert editor.text == "", "the draft rung never ran"
        assert "summarise the ingest path please" in editor.prompt_history()

        # Third press: the interrupt, with the draft already filed.
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert session.aborts == ["interrupted"], "the interrupt stayed unreachable"


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
        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 1, 11))

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
        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 19))

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

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))

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
        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))
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

        # A composer copy first, so the editor is holding a live selection
        # exactly as it would be after an explicit copy in use — the worst
        # case for the clobber, since the stale range is present and would
        # have been authorised to copy on release under the old rule.
        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 16))
        assert app._clipboard == "my private draft"
        assert editor.selected_text == "my private draft"

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
        editor.selection = DocumentSelection((0, 0), (0, 7))
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

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))

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

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 2, 60))

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

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 1, 11))

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

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 1, 0))

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

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))

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

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))

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

        await _composer_copy(app, pilot, _cell(editor, 0, 14), _cell(editor, 0, 20))
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

        await _composer_copy(app, pilot, _cell(editor, 0, 14), _cell(editor, 0, 20))
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
        await _composer_copy(app, pilot, _cell(editor, 0, 14), _cell(editor, 0, 20))
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

        await _composer_copy(app, pilot, _cell(editor, 0, 14), _cell(editor, 0, 20))
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

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 1, 0))

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

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, columns))

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

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))
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

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))
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

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))

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

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 5))
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

        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 9))
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
        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))
        assert editor._copied, "the copy should have posted a receipt"

        # The caret moves, collapsing the copied range...
        await pilot.press("right")
        await pilot.pause()
        # ...and the user makes a NEW, unrelated selection. Under the
        # explicit-copy gesture a live range makes the NEXT Ctrl+C a copy of
        # THAT range — which is the new rule this change wants. What D22
        # actually protects is that an unrelated selection cannot resurrect
        # a leftover deferral that would abort or quit; the press copies the
        # range on screen, nothing else.
        editor.selection = DocumentSelection((0, 10), (0, 20))
        await pilot.pause()
        assert (
            editor.selected_text == "the ingest"
        ), "the new selection must be live to mean anything"
        assert editor._copied, "the receipt was destroyed along with the highlight"

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
        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))
        assert editor.selected_text, "the copy's highlight must be on screen"

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
        await _composer_copy(app, pilot, _cell(editor, 0, 0), _cell(editor, 0, 20))
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


# -- review round 2 / design round 2 -----------------------------------------
@pytest.mark.asyncio
async def test_a_fold_between_two_words_keeps_the_space_the_user_had() -> None:
    """Pins that the wrap rejoin is POSITIONAL, not a substring membership test.

    The regression this replaces: ``wrap_separator`` asked whether the last
    token of one row concatenated to the first token of the next occurred
    ANYWHERE in the source line. A line using both ``file system`` and
    ``filesystem`` answers yes, so a fold at a real space was judged mid-token
    and the two words were welded shut — ``filesystem layer`` where the user
    highlighted ``file system layer`` (review round 2, R2-1; design round 2,
    D2-2).

    That is silent corruption: unlike the over-copy it replaced and unlike the
    newline before it, the paste looks like well-formed prose and the reader
    cannot tell a character is missing. Swept across widths because which pair
    lands either side of the fold is width-dependent.
    """
    welded: list[tuple[int, str]] = []
    for width in range(28, 40):
        app = StyledTranscriptApp()
        async with app.run_test(size=(width, 40)) as pilot:
            block = AssistantBlock()
            await _mounted(app, block)
            block.update_text(COMPOUND_PAIR)
            block.finalize_text()
            await pilot.pause()

            rows = _rendered_rows(block)
            # Anchor on the OPEN form's first word where it is followed by the
            # fold, i.e. the last row that ends in "file" — the closed compound
            # "filesystem" appears earlier in the same line, which is the whole
            # point of the fixture, so a plain index() would find the wrong one.
            last_row, _, end = _find(rows, "layer")
            first_row = next(
                (i for i in range(last_row - 1, -1, -1) if rows[i].rstrip().endswith("file")),
                None,
            )
            if first_row is None:
                continue  # the fold does not fall between "file" and "system" here
            start = rows[first_row].rstrip().rindex("file")

            selection = Selection.from_offsets(
                Offset(x=start, y=first_row), Offset(x=end, y=last_row)
            )
            copied = block.get_selection(selection)
            assert copied is not None

            # The clipboard must be text the source actually contains. A welded
            # compound is not, which is what makes this a regression test.
            if copied[0] not in COMPOUND_PAIR:
                welded.append((width, copied[0]))

    assert not welded, f"the rejoin destroyed a word boundary: {welded}"


@pytest.mark.asyncio
async def test_no_ordinary_compound_pair_is_welded_at_any_width() -> None:
    """The sweep that a single fixture cannot stand in for.

    A membership-versus-position bug fires only when the two words either side
    of the fold also appear welded elsewhere in the line, so it hides from any
    sampled test whose fixture happens to fold elsewhere. Review round 2 found
    it by probing 15 ordinary compound pairs; design round 2 found 30
    reproductions across five prose lines. This pins the whole class rather
    than the one example that was reported.
    """
    failures: list[tuple[str, int, str]] = []
    for joined, spaced in COMPOUND_PAIRS:
        head = spaced.split()[0]
        source = (
            f"The {joined} and the {spaced} layer are different things in this "
            f"codebase, remember that.\n"
        )
        for width in (29, 31, 33, 35):
            app = StyledTranscriptApp()
            async with app.run_test(size=(width, 40)) as pilot:
                block = AssistantBlock()
                await _mounted(app, block)
                block.update_text(source)
                block.finalize_text()
                await pilot.pause()

                rows = _rendered_rows(block)
                try:
                    last_row, _, end = _find(rows, "layer")
                except AssertionError:
                    continue
                first_row = next(
                    (i for i in range(last_row - 1, -1, -1) if rows[i].rstrip().endswith(head)),
                    None,
                )
                if first_row is None:
                    continue  # this width does not fold between the pair's words
                start = rows[first_row].rstrip().rindex(head)

                selection = Selection.from_offsets(
                    Offset(x=start, y=first_row), Offset(x=end, y=last_row)
                )
                copied = block.get_selection(selection)
                assert copied is not None
                if copied[0] not in source:
                    failures.append((spaced, width, copied[0]))

    assert not failures, f"compound pairs welded shut: {failures}"


@pytest.mark.asyncio
async def test_a_cjk_fold_does_not_invent_a_space() -> None:
    """Pins the ``align``-returned-``None`` fallback as conservative.

    ``wrap_separator`` returned ``" "`` unconditionally when the row could not
    be placed. CJK rows are exactly that case — no anchor word survives, so the
    whole message maps to ``None`` — and CJK never breaks at a space, so every
    fold gained a character the document does not contain anywhere (review
    round 2, R2-2; design round 2, D2-3).

    In Japanese an interpolated space is not cosmetic; it reads as a deliberate
    break. Asserted as "the clipboard is a substring of the source", the same
    invariant the compound test uses, because that is the property a paste has
    to have.
    """
    for label, text in (("chinese", CJK_PARAGRAPH), ("japanese", JAPANESE_PARAGRAPH)):
        for width in (30, 34, 40):
            app = StyledTranscriptApp()
            async with app.run_test(size=(width, 40)) as pilot:
                block = AssistantBlock()
                await _mounted(app, block)
                block.update_text(text)
                block.finalize_text()
                await pilot.pause()

                rows = _rendered_rows(block)
                content = [i for i, row in enumerate(rows) if row.strip()]
                if len(content) < 2:
                    continue
                first_row, last_row = content[0], content[1]

                selection = Selection.from_offsets(
                    Offset(x=2, y=first_row),
                    Offset(x=max(1, len(rows[last_row].rstrip()) - 4), y=last_row),
                )
                copied = block.get_selection(selection)
                assert copied is not None
                assert " " not in copied[0], (
                    f"{label} at width {width} gained a space the source never had: "
                    f"{copied[0]!r}"
                )
                assert copied[0] in text, (
                    f"{label} at width {width} copied text absent from the source: "
                    f"{copied[0]!r}"
                )


@pytest.mark.asyncio
async def test_an_emoji_fold_still_rejoins_with_its_space() -> None:
    """The control that keeps the CJK fix from over-reaching.

    Emoji are double-width like CJK, but they sit in space-delimited Latin
    prose, so a fold beside one DID consume a space and the rejoin must put it
    back. Review round 2 verified emoji already copied correctly and explicitly
    asked that width handling not be re-engineered, so this pins the boundary:
    the rule is about scripts that do not write spaces, not about cell width.
    """
    for width in (30, 34, 40):
        app = StyledTranscriptApp()
        async with app.run_test(size=(width, 40)) as pilot:
            block = AssistantBlock()
            await _mounted(app, block)
            block.update_text(EMOJI_PROSE)
            block.finalize_text()
            await pilot.pause()

            rows = _rendered_rows(block)
            first_row, start, _ = _find(rows, "status")
            last_row, _, end = _find(rows, "emoji")
            if first_row == last_row:
                continue

            selection = Selection.from_offsets(
                Offset(x=start, y=first_row), Offset(x=end, y=last_row)
            )
            copied = block.get_selection(selection)
            assert copied is not None
            assert (
                copied[0] in EMOJI_PROSE
            ), f"emoji prose at width {width} did not rejoin as written: {copied[0]!r}"


@pytest.mark.asyncio
async def test_a_fence_indented_inside_a_list_item_keeps_its_first_character() -> None:
    """The blocker: code nested under a list item lost its leading character.

    ``test_a_column_zero_drag_inside_a_fence_keeps_the_code`` uses a TOP-LEVEL
    fence, where both ``classify`` and ``align`` already worked, so it never
    discriminated this case — it passes on the prior head too.

    Two independent causes, either sufficient (design round 2, D2-1). The fence
    pattern allowed at most three leading spaces while a fence under a list item
    is conventionally indented four, so ``classify`` returned an empty covered
    set; and ``align`` attributed the code rows to the LIST's source line rather
    than to the fence body. With ``fenced`` false and a source line matching the
    list pattern, ``furniture_width`` read the code's own leading ``1`` or ``3``
    as a rendered ordered marker and stripped it, so ``3 rows expected`` pasted
    as ``rows expected``. Silent corruption of a command the user highlighted.
    """
    for width in (40, 60):
        app = StyledTranscriptApp()
        async with app.run_test(size=(width, 40)) as pilot:
            block = AssistantBlock()
            await _mounted(app, block)
            block.update_text(NESTED_FENCE)
            block.finalize_text()
            await pilot.pause()

            rows = _rendered_rows(block)
            code_row, _, _ = _find(rows, "3 rows expected")

            # Column 0 through the row's end: the gesture that takes a line of
            # code as a line of code.
            selection = Selection.from_offsets(
                Offset(x=0, y=code_row), Offset(x=len(rows[code_row]), y=code_row)
            )
            copied = block.get_selection(selection)
            assert copied is not None
            assert (
                "3 rows expected" in copied[0]
            ), f"the code's leading character was deleted at width {width}: {copied[0]!r}"

            # And across the fold into the next code line, the shape design
            # round 2 reported as 'rows expected psql -c'.
            next_row, _, _ = _find(rows, "psql -c")
            across = Selection.from_offsets(
                Offset(x=0, y=code_row), Offset(x=len(rows[next_row].rstrip()), y=next_row)
            )
            copied_across = block.get_selection(across)
            assert copied_across is not None
            assert "3 rows expected" in copied_across[0], (
                f"a take across the fold dropped the leading 3 at width {width}: "
                f"{copied_across[0]!r}"
            )


def test_a_fence_is_recognised_at_a_list_item_indent() -> None:
    """The first of D2-1's two causes, pinned at the unit it lives in.

    A fence under ``- `` is indented four spaces and under ``1. `` five, so a
    three-space bound made ``classify`` blind to the most common nested shape.
    Pinned separately from the widget test because the widget test would still
    pass if only the second cause were fixed, and this is the cheaper signal
    about which one regressed.
    """
    for indent in range(0, 9):
        source = f"{' ' * indent}```python\n{' ' * indent}1 / 0\n{' ' * indent}```\n"
        lines: list[str] = list(source.split("\n"))
        covered, markers = _copy_markdown.classify(lines)
        assert markers == {0, 2}, f"indent {indent} hid the fence markers: {markers}"
        assert 1 in covered, f"indent {indent} left the code line uncovered: {covered}"


@pytest.mark.asyncio
async def test_a_bullet_inside_a_quote_leaks_neither_the_bar_nor_the_dot() -> None:
    """Constructs COMPOSE: quote furniture and list furniture on the same row.

    ``furniture_width`` tested the quote pattern first and returned, so a list
    inside a blockquote stripped the ``▌`` and kept the ``•`` — a glyph nowhere
    in the user's document — and the paste format still flipped on a one-cell
    change of drag start (design round 2, D2-4).

    Asserted over a range of start columns because the format flip is precisely
    a disagreement between adjacent columns: the whole-row gesture must give the
    same answer wherever inside the painted furniture it began.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(44, 40)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(QUOTED_LIST)
        block.finalize_text()
        await pilot.pause()

        rows = _rendered_rows(block)
        bullet_row, _, _ = _find(rows, "quoted bullet")

        answers: dict[int, str] = {}
        for column in range(0, 5):
            selection = Selection.from_offsets(
                Offset(x=column, y=bullet_row), Offset(x=len(rows[bullet_row]), y=bullet_row)
            )
            copied = block.get_selection(selection)
            assert copied is not None
            answers[column] = copied[0]
            assert (
                "•" not in copied[0]
            ), f"start column {column} leaked a painted bullet: {copied[0]!r}"
            assert (
                "▌" not in copied[0]
            ), f"start column {column} leaked a painted quote bar: {copied[0]!r}"

        assert (
            len(set(answers.values())) == 1
        ), f"the paste format flips with the drag's start cell: {answers}"


# -- the oracle sweep --------------------------------------------------------
#: Ordinary assistant output, not a minimised repro. Three rounds of findings
#: were each correct on the case the previous round named and wrong on a
#: neighbouring one, so the corpus is deliberately BROAD rather than pointed:
#: every construct this app actually emits, and the inline shapes (long tokens,
#: trailing markup, intraword punctuation) whose interaction with a fold is what
#: the walk has to get right.
ORACLE_CORPUS: dict[str, str] = {
    "prose_inline_markup": (
        "The **frontend** cache lives at `/var/lib/local-operator/cache.sqlite3` "
        "and is managed by the *supervisor* process, see [the docs](https://d.io/x)."
    ),
    "prose_trailing_code": (
        "The cache lives at /var/lib/local-operator/sessions/cache-index.sqlite3 "
        "managed by `lop`"
    ),
    "prose_trailing_bold": (
        "Download https://github.com/damianvtran/local-operator/releases/latest/"
        "download/bundle.tgz now **today**"
    ),
    "snake_and_kebab": (
        "Set database_connection_pool_max_size_in_the_config and the "
        "some-very-long-kebab-case-flag-name-here before the deploy runs."
    ),
    "compound_open_and_closed": (
        "The run time of the runtime is fine, and the set up of the setup, the "
        "log in of the login, and the back end of the backend all check out."
    ),
    "bullets": (
        "- a plain bullet item that is long enough to wrap across several rows\n"
        "- another bullet mentioning /Users/somebody/Library/Application Support/x\n"
        "  - a nested bullet that also wraps because it is quite long indeed"
    ),
    "ordered": (
        "1. the first step which is long enough that it wraps at narrow widths\n"
        "2. the second step, running `systemctl restart ingest-worker` for real\n"
        "   1. a nested ordered step that is also long enough to fold somewhere"
    ),
    "quote": (
        "> a quoted paragraph that is long enough to wrap across several rows\n"
        "> and continues here with https://example.com/a/long/path?x=1 inside it"
    ),
    "nested_quote": (
        "> outer quote text that wraps because it is long enough to do so\n"
        ">> an inner nested quote that also wraps at the widths swept here"
    ),
    "quote_in_list": (
        "- a bullet holding a quote\n"
        "  > the quoted line inside the bullet, long enough that it folds\n"
    ),
    "list_in_quote": (
        "> - a bullet inside a quote that is long enough to wrap somewhere\n"
        "> - a second such bullet with a_snake_case_identifier_in_it here"
    ),
    "table": (
        "| name | score | notes |\n"
        "| --- | --- | --- |\n"
        "| alpha | 0.91 | the first row with some longer note text here |\n"
        "| beta | 0.72 | the second row, also with a reasonably long note |"
    ),
    "cjk": "这是一个中文段落，用于验证在终端宽度下折行之后复制粘贴不会插入多余的空格字符。",
    "cjk_ja": "これは日本語の段落です。この行は端末の幅で折り返されますので、コピーの境界を検証します。",
    "cjk_mixed": ("この API は runtime を使います。The server は port 8080 で listen します。"),
    "emoji": "The report 🚀 is ready 🎉 and the summary 📊 follows with more text to wrap here.",
}

#: Fences are swept separately: every indent 0-8, inside list items, and with
#: bodies whose lines LOOK structural. D2-1 was a deleted first character here
#: and R3-2 a weld between two body lines, so the shape earns its own corpus.
ORACLE_FENCE_CORPUS: dict[str, str] = {
    f"fence_indent_{indent}": (
        "Run these in order:\n\n"
        f"{' ' * indent}```sh\n"
        f"{' ' * indent}cd /srv/ingest\n"
        f"{' ' * indent}psql -h db.internal -U ingest -c 'select count(*) from staging.rows'\n"
        f"{' ' * indent}systemctl restart ingest-worker\n"
        f"{' ' * indent}```\n\nThen check the dashboard.\n"
    )
    for indent in range(0, 9)
} | {
    "fence_in_list": (
        "1. Run the check:\n\n"
        "    ```sh\n"
        "    ./scripts/check --verbose --with-a-long-flag-name --and-another one\n"
        "    ```\n"
    ),
    "fence_structural_body": (
        "```text\n"
        "- this line looks like a bullet but is code and must stay verbatim\n"
        "> this line looks like a quote but is code and must stay verbatim\n"
        "3. this line looks ordered but is code and must stay verbatim here\n"
        "```\n"
    ),
}


async def _oracle_truth(text: str) -> list[str]:
    """The rendered rows at a width so wide that NOTHING folds.

    The oracle is independent of the code under test: it is the same renderer,
    asked a question with no wrapping in it. A take at a narrow width has to be
    findable in this, because folding is the only thing that changed.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(400, 120)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(text)
        block.finalize_text()
        await pilot.pause()
        return [row.rstrip() for row in _rendered_rows(block) if row.strip()]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(ORACLE_CORPUS | ORACLE_FENCE_CORPUS))
async def test_every_take_is_a_contiguous_substring_of_the_source(name: str) -> None:
    """THE STRUCTURAL TEST: sweep widths, and assert no take invents or deletes.

    Every previous round's tests pinned the SYMPTOM that round reported, and
    every round the next defect landed in the neighbouring case nobody had named
    while the whole suite stayed green. This test pins the PROPERTY instead: a
    copy must be text the reader could find in their own document.

    A single invariant catches all four failure modes seen so far without
    predicting which one to look for -- an invented space (R3-1) breaks it, a
    weld across two source lines (R2-1, R3-2) breaks it, a deleted leading
    character (D2-1) breaks it, and a space put into CJK (R2-2) breaks it.

    Swept broadly rather than sampled, because every defect so far was reachable
    at ordinary widths and missed by sampling.
    """
    text = (ORACLE_CORPUS | ORACLE_FENCE_CORPUS)[name]
    truth = await _oracle_truth(text)

    for width in range(30, 90, 2):
        app = StyledTranscriptApp()
        async with app.run_test(size=(width, 120)) as pilot:
            block = AssistantBlock()
            await _mounted(app, block)
            block.update_text(text)
            block.finalize_text()
            await pilot.pause()

            rows = _rendered_rows(block)
            content = [i for i, row in enumerate(rows) if row.strip()]
            if len(content) < 2:
                continue

            # A drag from just inside the first content row, ending both AT the
            # last row's end and SHORT of it. Stopping short is the natural
            # gesture for highlighting exactly a URL, and it is the case that
            # violates the walk's end-anchored precondition -- sweeping only
            # full-coverage drags reports all clear while the defect is live.
            last_row = content[-1]
            end_columns = {len(rows[last_row].rstrip()), 20, 10}
            for start_column in (1, 2):
                for end_column in sorted(e for e in end_columns if e > 0):
                    selection = Selection.from_offsets(
                        Offset(x=start_column, y=content[0]),
                        Offset(x=end_column, y=last_row),
                    )
                    copied = block.get_selection(selection)
                    if copied is None:
                        continue
                    for line in copied[0].split("\n"):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        # TWO truthful answers, and exactly two. The glyph path
                        # claims to be what was PAINTED, so it must be findable in
                        # the unfolded frame; the markdown path claims to be what
                        # the model WROTE, so it must be findable in the source.
                        # Anything in neither is a character this code invented.
                        # Testing against both is what keeps the oracle honest
                        # without a marker blacklist -- a blacklist would be the
                        # same guess-rather-than-know shape the fix removes.
                        if any(stripped in row for row in truth):
                            continue
                        if stripped in text:
                            continue
                        raise AssertionError(
                            f"{name} at width {width}, start column {start_column}: "
                            f"copied a line that is in neither the rendered frame nor "
                            f"the source.\n  copied: {line!r}\n  truth : {truth!r}"
                        )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(ORACLE_FENCE_CORPUS))
async def test_no_two_fenced_lines_are_ever_welded_into_one(name: str) -> None:
    """Two commands in a fence must never arrive as one runnable line (R3-2).

    The sharpest payload found so far: ``align`` cannot place the continuation
    rows of an over-long fence body line, the sub-line gate DROPPED those
    ``None`` mapping entries rather than treating them as absence of evidence,
    and a take spanning one collapsed to a single-source-line judgement it never
    was. The clipboard then joined two shell commands with a space.

    Asserted against the source's own line boundaries rather than against a
    rendered row, because the harm is specifically that a line boundary the
    document has went missing.

    The predicate is FRAGMENT-level, not whole-line. Requiring two COMPLETE
    source body lines in one output line made the test narrower than the defect
    it was written for: the user-visible harm is a command TAIL meeting the next
    command's HEAD (``cd /srv/ingest psql -h``), and a fence whose lines are
    longer than the pane -- the common case for shell and SQL -- never puts a
    whole body line on one row. The whole-line form passed on all nine
    ``fence_indent_*`` fixtures, the shape D3-2 actually lived in, and caught the
    defect only by luck on ``fence_structural_body``, whose body lines are short
    enough to appear whole (design round 4, D4-2). Anchoring on a chunk of each
    line catches the weld wherever it lands.

    Swept from width 28 and across start columns 0-2 for the same reason: the
    control welds at widths and start columns the old range never visited.
    """
    text = ORACLE_FENCE_CORPUS[name]
    body = [
        line.strip()
        for line in text.split("\n")
        if line.strip() and not line.strip().startswith("```")
    ]
    # A weld joins the END of one body line to the START of another, so each
    # line is probed by its own head and tail rather than by its whole text.
    # Only DISCRIMINATING chunks count: two body lines may legitimately share a
    # head or tail (``fence_structural_body``'s all end "must stay verbatim"),
    # and a shared chunk cannot say which line a fragment came from, so counting
    # it would report a weld on an honest single-line take.
    chunk = 12
    marks: list[tuple[str, list[str]]] = []
    for line in body:
        if len(line) < chunk:
            continue
        parts = sorted(
            part
            for part in {line[:chunk], line[-chunk:]}
            if sum(part in other for other in body) == 1
        )
        if parts:
            marks.append((line, parts))

    for width in range(28, 76, 2):
        app = StyledTranscriptApp()
        async with app.run_test(size=(width, 120)) as pilot:
            block = AssistantBlock()
            await _mounted(app, block)
            block.update_text(text)
            block.finalize_text()
            await pilot.pause()

            rows = _rendered_rows(block)
            content = [i for i, row in enumerate(rows) if row.strip()]
            for first in content:
                for last in content:
                    if last <= first:
                        continue
                    # Ends SHORT of the last row as well as covering it. The
                    # weld is only reachable on a sub-line take, and a take that
                    # covers every row fully goes down the markdown path -- so a
                    # sweep of full-coverage drags alone reports all clear while
                    # the defect is live. That is precisely how three rounds of
                    # sampled tests stayed green.
                    ends = {len(rows[last].rstrip()), 20, 8}
                    for start_column in (0, 1, 2):
                        for end_column in sorted(e for e in ends if e > 0):
                            selection = Selection.from_offsets(
                                Offset(x=start_column, y=first),
                                Offset(x=end_column, y=last),
                            )
                            copied = block.get_selection(selection)
                            if copied is None:
                                continue
                            for line in copied[0].split("\n"):
                                # An output line carrying a fragment of two
                                # DISTINCT source body lines is a weld: the
                                # boundary between them is gone, and a shell
                                # handed that line runs two commands.
                                hits = [
                                    src
                                    for src, parts in marks
                                    if src != line.strip() and any(part in line for part in parts)
                                ]
                                assert len(hits) < 2, (
                                    f"{name} at width {width}, rows {first}..{last}, start "
                                    f"column {start_column}: welded two source lines into "
                                    f"one.\n  copied line: {line!r}\n  welded    : {hits!r}"
                                )


#: Constructs whose over-long lines leave continuation rows that ``align``
#: cannot place. That is the shape where a copy can silently SUBSTITUTE one
#: source line for another, so the substitution oracle sweeps them specifically.
ORACLE_SUBSTITUTION_CORPUS: dict[str, str] = {
    "fence_long_commands": (
        "Run these in order:\n\n```sh\ncd /srv/ingest\n"
        "psql -h db.internal -U ingest -c 'select count(*) from staging.rows where state = 1'\n"
        "systemctl restart ingest-worker --now --no-block\n```\n\nThen check the dashboard.\n"
    ),
    "fence_sql_then_prose": (
        "```sql\n"
        "select a.id, a.name, b.value from alpha a join beta b on b.id = a.beta_id;\n"
        "update alpha set flagged = true where created_at < now() - interval '30 days';\n"
        "```\n\nTRAILING_ONE is a paragraph after the fence that was never highlighted.\n"
    ),
    "table_long_notes": (
        "| name | score | notes |\n| --- | --- | --- |\n"
        "| alpha | 0.91 | the first row with some longer note text here that wraps |\n"
        "| beta | 0.72 | the second row, also with a reasonably long note here |"
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=(
                # PRE-EXISTING and unchanged by this PR -- identical at the merge
                # base, at `6f72a29e` and here. ``align`` maps a wrapped table
                # row's continuation rows to the NEXT table row, so a whole-row
                # take copies the wrong row. That lives in ``align``'s table
                # handling rather than in this copy path, so it is tracked as
                # issue #399 rather than widened into this PR. Marked xfail
                # rather than dropped from the corpus: the oracle keeps reporting
                # it, and the marker fails loudly once #399 is fixed.
                [
                    pytest.mark.xfail(
                        reason="pre-existing table mis-mapping, issue #399", strict=True
                    )
                ]
                if name == "table_long_notes"
                else []
            ),
        )
        for name in sorted(ORACLE_SUBSTITUTION_CORPUS)
    ],
)
async def test_a_fully_covered_row_is_never_replaced_by_a_different_line(name: str) -> None:
    """A take must contain the rows the reader LIT, not merely real source text.

    The companion invariant to the contiguity oracle, and the hole that admitted
    R4-1. Contiguity is a MEMBERSHIP test: it catches text this code invented and
    text it welded, but a copy that hands back a different, genuine source line
    is still perfectly contiguous and sails through. That is exactly what the
    unplaced-edge widening did -- dragging the ``psql`` command copied the
    ``systemctl`` command, so a reader pasting into a terminal ran a command they
    never highlighted (review round 4, R4-1; design round 4, D4-1).

    So this asserts the other half: every rendered row the drag covered in FULL
    must be represented in the copy. A fully covered row is unambiguous about the
    reader's intent in a way a partial one is not, which is what makes the
    assertion safe to state without predicting which construct will break it.
    """
    text = ORACLE_SUBSTITUTION_CORPUS[name]

    for width in range(28, 102, 2):
        app = StyledTranscriptApp()
        async with app.run_test(size=(width, 120)) as pilot:
            block = AssistantBlock()
            await _mounted(app, block)
            block.update_text(text)
            block.finalize_text()
            await pilot.pause()

            rows = _rendered_rows(block)
            for index, row in enumerate(rows):
                lit = row.strip()
                # Short rows carry too little to identify; a fence marker row is
                # furniture whose own text is legitimately re-emitted elsewhere.
                if len(lit) < 12 or lit.startswith("```"):
                    continue
                selection = Selection.from_offsets(
                    Offset(x=0, y=index), Offset(x=len(row.rstrip()), y=index)
                )
                copied = block.get_selection(selection)
                if copied is None:
                    continue
                # Compared with folds flattened: the markdown path may return the
                # line unwrapped, which is a truthful answer to the same gesture.
                haystack = " ".join(copied[0].split())
                needle = " ".join(lit.split())[:12]
                assert needle in haystack, (
                    f"{name} at width {width}, row {index}: the fully highlighted row is "
                    f"not in the copy -- a different line was substituted for it.\n"
                    f"  highlighted: {lit!r}\n  copied     : {copied[0]!r}"
                )


@pytest.mark.asyncio
async def test_widening_over_unplaced_rows_needs_a_placed_row_to_rescue() -> None:
    """``_markdown_for_rows`` widens only where a lit line can be recovered.

    Direct cover for the widening itself, which shipped in round 3 with no test
    of its own -- and was therefore free to substitute one source line for
    another. Both halves are asserted together because the finding is precisely
    that the two cases have DIFFERENT right answers:

    * with a placed row in the band, widening over an unplaced EDGE row rescues a
      line ``slice_markdown`` would otherwise drop;
    * with NO placed row there is nothing to rescue, the un-widened slice is
      empty, and the caller's fallback to the frame copy is the correct minimal
      answer. Widening there dragged the window onto whichever line happened to
      be placed (review round 4, R4-1; design round 4, D4-1).
    """
    text = ORACLE_SUBSTITUTION_CORPUS["fence_sql_then_prose"]
    app = StyledTranscriptApp()
    async with app.run_test(size=(76, 120)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(text)
        block.finalize_text()
        await pilot.pause()

        rows = _rendered_rows(block)
        mapping = _copy_markdown.align(block._full_text, rows)
        target = next(i for i, row in enumerate(rows) if "a.beta_id;" in row)
        assert mapping[target] is None, "fixture no longer reproduces an unplaced row"

        # No placed row in the band: refuse rather than widen onto a neighbour.
        assert block._markdown_for_rows(mapping, target, target) == ""

        # The reader sees the minimal answer, not the construct plus the
        # paragraph below the closing fence that the band never touched.
        selection = Selection.from_offsets(
            Offset(x=0, y=target), Offset(x=len(rows[target].rstrip()), y=target)
        )
        copied = block.get_selection(selection)
        assert copied is not None
        assert copied[0].strip() == "a.beta_id;", copied[0]
        assert "TRAILING_ONE" not in copied[0], copied[0]

    # ...and the rescue itself still works. A band whose FIRST row is unplaced
    # but which reaches a placed row: without widening, ``slice_markdown`` picks
    # only the placed line and the lit `psql` command is dropped from the copy.
    text = ORACLE_SUBSTITUTION_CORPUS["fence_long_commands"]
    app = StyledTranscriptApp()
    async with app.run_test(size=(76, 120)) as pilot:
        block = AssistantBlock()
        await _mounted(app, block)
        block.update_text(text)
        block.finalize_text()
        await pilot.pause()

        rows = _rendered_rows(block)
        mapping = _copy_markdown.align(block._full_text, rows)
        first = next(i for i, row in enumerate(rows) if "psql" in row)
        last = next(i for i in range(first + 1, len(mapping)) if mapping[i] is not None)
        assert mapping[first] is None, "fixture no longer reproduces an unplaced edge row"

        # The un-widened slice is what would ship without the rescue.
        unwidened = _copy_markdown.slice_markdown(block._full_text, mapping, first, last)
        assert "psql" not in unwidened, unwidened

        widened = block._markdown_for_rows(mapping, first, last)
        assert "psql -h db.internal" in widened, widened
        # ...and the rescue does not weld the two commands onto one line.
        assert not any(
            "psql" in line and "systemctl" in line for line in widened.split("\n")
        ), widened


@pytest.mark.asyncio
async def test_a_fold_at_a_double_space_puts_both_spaces_back() -> None:
    """A rejoin restores the whitespace run VERBATIM, not one space (R4-2).

    ``_skip_painted_nothing`` reported WHETHER it had skipped whitespace, so a
    fold landing on a run of two or more spaces rejoined with exactly one and the
    document lost a character it had. The walk placed these rows, so this is not
    a refusal failure -- it is the same "a plausible character looks like an
    answer" shape one level down, and the fix is to carry the count.
    """
    text = "A line ending in two spaces  and then continuing with more text to force a fold."

    seen = 0
    for width in range(24, 92, 2):
        app = StyledTranscriptApp()
        async with app.run_test(size=(width, 120)) as pilot:
            block = AssistantBlock()
            await _mounted(app, block)
            block.update_text(text)
            block.finalize_text()
            await pilot.pause()

            rows = _rendered_rows(block)
            content = [i for i, row in enumerate(rows) if row.strip()]
            if len(content) < 2:
                continue
            for last in content[1:]:
                selection = Selection.from_offsets(
                    Offset(x=1, y=content[0]), Offset(x=len(rows[last].rstrip()), y=last)
                )
                copied = block.get_selection(selection)
                if copied is None or "\n" in copied[0]:
                    continue
                stripped = copied[0].strip()
                if not stripped:
                    continue
                seen += 1
                # The single truthful answer for a rejoined sub-line take: it is
                # a verbatim span of the source, double space included.
                assert stripped in text, (
                    f"width {width}: the rejoin did not reproduce the source's spacing.\n"
                    f"  copied: {copied[0]!r}"
                )

    assert seen, "the sweep never reached a rejoined take"


def test_skip_painted_nothing_reports_the_whitespace_it_consumed() -> None:
    """The unit behind R4-2: the consumed run comes back, not a boolean.

    Pinned at the function level as well as end-to-end, because the bool return
    was locally plausible -- ``" " if saw_space else ""`` reads as correct until
    the run is longer than one character.
    """
    assert _copy_markdown._skip_painted_nothing("a  b", 1) == (3, "  ")
    assert _copy_markdown._skip_painted_nothing("a b", 1) == (2, " ")
    assert _copy_markdown._skip_painted_nothing("a\tb", 1) == (2, "\t")
    assert _copy_markdown._skip_painted_nothing("ab", 1) == (1, "")
    # Markers that paint nothing are skipped without contributing whitespace.
    assert _copy_markdown._skip_painted_nothing("a**  b", 1) == (5, "  ")


def test_a_fence_marker_in_indented_code_reads_as_a_fence_on_purpose() -> None:
    """The knowingly-accepted `_FENCE_RE` trade-off, pinned (review round 3, R3-4).

    Widening the pattern past three leading spaces is what lets ``classify`` see
    the fence under a list item, which is routine in this app's output. The cost
    is that a literal ```` ``` ```` inside a FOUR-SPACE indented code block now
    reads as a fence marker too.

    That cost is accepted because the failure directions are not symmetric:
    over-recognising a fence makes the copy VERBATIM, which is the conservative
    answer for a clipboard, while under-recognising one deletes characters from
    the user's code (design round 2, D2-1). Pinned so the next person to narrow
    the pattern learns the widening was deliberate rather than reading this as a
    bug to fix, and so the conservative DIRECTION is checked rather than assumed.
    """
    lines = ["Text:", "", "    ```", "    still indented code", "    ```", ""]
    covered, markers = _copy_markdown.classify(lines)

    # The accepted consequence, stated as the assertion rather than in prose.
    assert markers == {2, 4}, f"the indented backticks did not read as markers: {markers}"
    # ...and the direction that makes it acceptable: content stays covered, so
    # the copy is verbatim rather than losing a leading character.
    assert 3 in covered, f"the indented code line lost its fence cover: {covered}"


async def _click_pair(
    app: OperatorApp,
    pilot: Any,
    editor: Editor,
    drift: int,
    vertical: int = 0,
    base_row: int = 0,
    base_column: int = 18,
) -> None:
    """Two SINGLE clicks `(drift, vertical)` cells apart, as a terminal sends them.

    `pilot.click(times=2)` builds the Click with its chain count already
    decided, so it cannot model a pair that Textual scores as two singles.
    These are hand-built MouseDown/MouseUp pairs through the app's own event
    path, which is where `_click_chain_last_offset` is compared.

    `vertical` and `base_row` exist because the drift band is a RADIUS and was
    silently one-sided on BOTH axes (R3-2): the caller has to be able to aim the
    second click above the first, which needs a base row that is not row 0 and a
    draft tall enough to hold it.
    """
    base = editor.region.offset + Offset(
        editor.gutter.left + base_column, editor.gutter.top + base_row
    )
    for offset in (base, base + Offset(drift, vertical)):
        for kind in (events.MouseDown, events.MouseUp):
            app.post_message(
                kind(
                    widget=editor,
                    x=offset.x,
                    y=offset.y,
                    delta_x=0,
                    delta_y=0,
                    button=1,
                    shift=False,
                    meta=False,
                    ctrl=False,
                    screen_x=offset.x,
                    screen_y=offset.y,
                    style=None,
                )
            )
        await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
async def test_a_barren_multi_click_on_the_last_row_does_not_clear_the_draft() -> None:
    """The blank LAST row answers Ctrl+C with nothing, not with a lost draft.

    Regression for design review round 2, D2. `_line_break_span` must stay
    collapsed on the final row — there is no following row to reach, and
    widening backwards would take a character the user never pointed at, which
    is also what keeps the empty composer correct (D7). So the range cannot
    reach this row, and for a while that meant the row still ate the draft.

    It is not an edge case: shift+enter is this composer's newline, so a user
    who finishes a paragraph and hits it twice SITS on a blank last row while
    they think of the next sentence. Double-clicking there painted a frame
    byte-identical to the one before it and the Ctrl+C that followed cleared
    the draft — the exact sentence this whole change exists to eliminate.

    Fixed on the GESTURE rather than the range: a multi-click that produced no
    range declines to let the very next press take the interrupt rung.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        # Typed, not loaded: shift+enter is the key the finding is about.
        for character in "first paragraph":
            await pilot.press(character if character != " " else "space")
        await pilot.press("shift+enter")
        await pilot.press("shift+enter")
        draft = editor.text
        assert draft.endswith("\n\n"), "the fixture must sit on a blank last row"

        row = editor.document.line_count - 1
        await _composer_multi_click(app, pilot, editor, 0, times=2, row=row)
        assert not editor.selected_text, "the last row must still take no range"

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert editor.text == draft, "the barren gesture still cleared the draft"


@pytest.mark.asyncio
async def test_a_double_click_that_drifts_one_cell_does_not_clear_the_draft() -> None:
    """A one-column drift is still one gesture, so the draft survives.

    Regression for design review round 2, D2-1. Widening the chain's TIME
    window (D3) covered only one axis: `App._on_event` also requires an EXACT
    cell match against `_click_chain_last_offset`, so a fast pair one column
    apart is two singles, makes no selection, and the Ctrl+C that follows
    clears the draft. The slow clicker and the jittery clicker are the same
    deliberate user the D3 rationale optimises for.

    The pair deliberately still makes NO SELECTION — inventing a range from a
    gesture Textual scored as two singles would paint a highlight the user did
    not ask for. It only declines the interrupt rung for one press.
    """
    for drift in (1, 2):
        app = _pilot_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            editor = await _composer(app, pilot, "summarise the ingest path please")
            for _ in range(8):
                await pilot.pause()
            draft = editor.text

            await _click_pair(app, pilot, editor, drift)
            assert not editor.selected_text, "a near miss must not invent a range"

            await pilot.press("ctrl+c")
            await pilot.pause()
            assert editor.text == draft, f"a {drift}-cell drift cleared the draft"


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", [6, -6, 3, -3])
async def test_a_deliberate_second_click_elsewhere_keeps_the_draft_rung(drift: int) -> None:
    """The near-miss window must not swallow a real second caret placement.

    The bound on D2-1. Two clicks a few cells apart are one jittery gesture;
    two clicks further apart are a user deliberately placing a second caret
    somewhere else, and Ctrl+C there means what it means on any untouched
    composer. A rule that suppressed the draft rung for any pair of clicks
    would make the composer's commonest key unpredictable.

    PARAMETRISED IN BOTH DIRECTIONS, which is the assertion that was missing.
    The band is a radius, so it has no sign — but it was computed with
    `Offset.clamped`, which restricts x and y to values ABOVE ZERO rather than
    taking a magnitude. Every negative component became `0`, so a second click
    ANY distance to the left measured as no drift at all and armed the claim
    (agent review round 3, R3-2). This test passed only because it used `+6`;
    at `-6` it failed, and the product was wrong in exactly the way the test
    was written to forbid.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "summarise the ingest path please")
        for _ in range(8):
            await pilot.pause()

        await _click_pair(app, pilot, editor, drift)

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert (
            editor.text == ""
        ), f"a deliberate second click {drift} cells away suppressed the draft rung"
        assert "summarise the ingest path please" in editor.prompt_history()


@pytest.mark.asyncio
@pytest.mark.parametrize("vertical", [3, -3])
async def test_a_deliberate_second_click_on_another_row_keeps_the_draft_rung(
    vertical: int,
) -> None:
    """The drift band is symmetric on the VERTICAL axis too.

    Round 3 left this axis explicitly unproven: a row-below click leaves a
    one-row composer, so the reviewer could measure the arithmetic but not the
    behaviour. It needs a draft tall enough that a three-row drift in EITHER
    direction still lands inside the widget, and a base click on a middle row so
    an upward drift has somewhere to go.

    Measured that way the axis was affected exactly as the arithmetic predicted:
    `dy=+3` correctly fell outside the band while `dy=-3` was clamped to zero
    drift and armed the claim, so a user placing a second caret three rows UP
    lost a press (R3-2).
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        # Seven rows, typed through the composer's own newline, so a drift of
        # three from row three stays inside the widget in both directions.
        for index in range(7):
            for character in f"row {index} of a tall draft":
                await pilot.press(character if character != " " else "space")
            if index < 6:
                await pilot.press("shift+enter")
        stable, previous = 0, None
        for _ in range(60):
            await pilot.pause()
            current = editor.region
            stable = stable + 1 if current == previous and editor.size.height > 6 else 0
            previous = current
            if stable >= 8:
                break
        assert editor.size.height > 6, "the composer never grew to seven rows"
        draft = editor.text

        await _click_pair(app, pilot, editor, 0, vertical=vertical, base_row=3, base_column=6)

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert (
            editor.text == ""
        ), f"a deliberate second click {vertical} rows away suppressed the draft rung"
        assert draft.splitlines()[0] in "\n".join(editor.prompt_history())


@pytest.mark.asyncio
async def test_a_barren_click_never_swallows_an_interrupt_on_an_empty_composer() -> None:
    """A genuine interrupt during a running turn is never absorbed. THE invariant.

    The barren rung's entire justification is that clearing the draft is the
    sentence this change exists to eliminate. On an EMPTY composer there is no
    draft, the press was always destined for the interrupt, and absorbing it
    protected nothing while costing a real interrupt on a live turn.

    Worse, it CHAINED. Double-clicking again when nothing appeared to happen is
    precisely the user's reflex, and each gesture re-armed the claim, so the
    interrupt was never reached at all — the exit ladder becoming unreachable,
    which is the invariant this whole area exists to protect (agent review round
    3, R3-1). The claim is now read only when there is a draft to protect, so it
    can only ever divert the press away from the DRAFT rung.
    """
    session = FakeSession()
    session.streaming = True
    app = _pilot_app(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        assert editor.text == "", "the fixture must have no draft to protect"

        # Four rounds of the user's actual reflex: click, nothing happens,
        # press, nothing happens, click again.
        for round_number in range(1, 5):
            await _composer_multi_click(app, pilot, editor, 0, times=2, row=0)
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert session.aborts, (
                f"round {round_number}: a barren gesture swallowed a genuine interrupt "
                "on an empty composer, and re-clicking chains it indefinitely"
            )
            session.aborts.clear()
            # Disarm the EXIT ladder between rounds, not the claim under test.
            # Each round leaves `_last_interrupt_at` set, so the next round's
            # press would land inside `DOUBLE_INTERRUPT_WINDOW_S` and correctly
            # interrupt-AND-EXIT — ending the app mid-test and reporting a
            # product failure that is really the ordinary second rung. This
            # models a user who pauses between attempts, which is the shape of
            # the reflex the finding describes.
            app._last_interrupt_at = 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("seam", ["keystroke", "blur"])
async def test_the_barren_claim_retires_when_the_gesture_ends(seam: str) -> None:
    """The claim is gesture-scoped, so it ends when the gesture demonstrably does.

    The window bounds how long the claim can be wrong for; it does not stop it
    being wrong. Nothing retired it when the user moved on, so it survived them
    typing and survived the composer losing focus, and the next Ctrl+C was
    absorbed on the strength of a gesture that was over (agent review round 3,
    R3-3).
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        await _composer_multi_click(app, pilot, editor, 0, times=2, row=0)
        assert editor.barren_multi_click, "the barren gesture was not recorded"

        if seam == "keystroke":
            await pilot.press("n")
        else:
            editor.blur()
        await pilot.pause()

        assert (
            not editor.barren_multi_click
        ), f"the claim outlived the gesture across the {seam} seam"


@pytest.mark.asyncio
async def test_a_barren_click_before_a_submit_does_not_eat_the_turns_interrupt() -> None:
    """Ctrl+C aimed at a turn the user just started always reaches it.

    The seam that costs most, driven end to end rather than by inspecting the
    flag: click in the composer, type, submit, then reach for Ctrl+C to stop a
    prompt you have decided is wrong. Submitting is well inside a reaction-time
    budget, so the 1.5 s window bounded the damage without preventing it and the
    interrupt was swallowed (R3-3).

    Two independent seams retire the claim here — the editor's own submit and
    the app's turn dispatch — because a prompt HELD through a compaction reaches
    `_start_turn` long after the submit that queued it.
    """
    session = FakeSession()
    app = _pilot_app(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        for character in "wrong prompt":
            await pilot.press(character if character != " " else "space")
        await pilot.press("shift+enter")
        await pilot.press("shift+enter")

        row = editor.document.line_count - 1
        await _composer_multi_click(app, pilot, editor, 0, times=2, row=row)
        assert editor.barren_multi_click, "the barren gesture was not recorded"

        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()
        assert not editor.barren_multi_click, "the claim survived the submit"

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert session.aborts == [
            "interrupted"
        ], "a barren click made before the submit ate the interrupt aimed at the turn"


@pytest.mark.asyncio
async def test_an_absorbed_press_withdraws_the_exit_hint_it_cannot_honour() -> None:
    """The screen never promises an exit the absorbed press does not make.

    The draft rung already resets the ladder and removes the hint for this exact
    reason, in a comment on the line above: leaving `ctrl+c again to exit` on
    screen would promise an exit the next press does not make. The barren rung
    returned early and did neither, reintroducing the stale promise its
    neighbour was written to prevent (design review round 3, D3-2).

    Note the reachable shape of this changed with R3-1: the rung is now gated on
    a draft, so the fixture arms the ladder on the empty composer and THEN types
    one.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        # Arm the ladder while the composer is empty: this is what paints the hint.
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app._exit_hint is not None, "the fixture never armed the exit hint"

        for character in "my prompt":
            await pilot.press(character if character != " " else "space")
        await pilot.press("shift+enter")
        await pilot.press("shift+enter")
        draft = editor.text

        row = editor.document.line_count - 1
        await _composer_multi_click(app, pilot, editor, 0, times=2, row=row)
        await pilot.press("ctrl+c")
        await pilot.pause()

        assert editor.text == draft, "the absorbed press did not protect the draft"
        assert app._exit_hint is None, "a stale exit hint outlived the press it invited"
        rows = [strip.text for strip in app.screen._compositor.render_strips()]
        assert not any(
            "again to exit" in row for row in rows
        ), "the screen still promises an exit the absorbed press does not make"
        assert app._last_interrupt_at == 0.0, "the ladder stayed armed under an absorbed press"


@pytest.mark.asyncio
async def test_a_barren_multi_click_suppresses_exactly_one_press() -> None:
    """The exit ladder stays reachable after a barren gesture.

    THE bound on the D2/D2-1 remedy, and the reason it is scoped to a bounded
    recent gesture rather than to live selection state. D17 is the lost-draft
    bug where a composer diverted Ctrl+C indefinitely on stale state: the
    interrupt never fired, the user pressed again, and the app exited with the
    draft never filed to history.

    So the claim is SPENT by the press that reads it. One press is absorbed;
    the next is an ordinary first press, and the full ladder — draft, then
    interrupt, then interrupt-and-exit — runs from there unchanged.
    """
    session = FakeSession()
    app = _pilot_app(session)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        for character in "first paragraph":
            await pilot.press(character if character != " " else "space")
        await pilot.press("shift+enter")
        await pilot.press("shift+enter")
        draft = editor.text
        session.streaming = True
        await pilot.pause()

        row = editor.document.line_count - 1
        await _composer_multi_click(app, pilot, editor, 0, times=2, row=row)

        # Press 1: absorbed by the barren gesture.
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert editor.text == draft
        assert session.aborts == []

        # Press 2: the ordinary draft rung, exactly as on an untouched composer.
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert editor.text == "", "the claim outlived the press that spent it"
        assert draft.strip() in "\n".join(editor.prompt_history())

        # Press 3: the interrupt. The ladder is reachable, one rung further on.
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert session.aborts == ["interrupted"], "the exit ladder became unreachable"


@pytest.mark.asyncio
async def test_a_stale_barren_click_stops_suppressing_the_interrupt() -> None:
    """The barren claim EXPIRES, so it can never strand the exit ladder.

    The other half of the D17 argument. A claim that suppressed a press
    indefinitely would be the same hazard by a new route: a user who clicked
    minutes ago, forgot, and now needs out. The window is a reaction-time
    budget for the hand moving from mouse to keyboard, so a press after it has
    passed gets the ordinary rung with no memory of the gesture at all.
    """
    app = _pilot_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = await _composer(app, pilot, "a draft with a trailing newline\n")

        row = editor.document.line_count - 1
        await _composer_multi_click(app, pilot, editor, 0, times=2, row=row)
        assert editor.barren_multi_click, "the barren gesture was not recorded"

        # Age the claim past its window rather than sleeping through it.
        editor._barren_click_at = time.monotonic() - (BARREN_CLICK_WINDOW_S + 0.1)
        assert not editor.barren_multi_click, "the claim did not expire"

        await pilot.press("ctrl+c")
        await pilot.pause()
        assert editor.text == "", "a stale claim still diverted the key"
