"""The reasoning block: the model's thinking, streamed dim, then retired.

Behaviour first, under the REAL sheet (``StyledTranscriptApp``): the block's own
docstring claims three properties — a labelled block of its own, a bounded live
TAIL, and a transient lifetime — and each of them is a property of the rendered
frame, not of the source.
"""

from __future__ import annotations

import pytest
from rich.color import Color
from rich.style import Style
from rich.text import Text

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.reasoning import (
    REASONING_LABEL,
    REASONING_RETAINED_CHARS,
    REASONING_TAIL_CHARS,
    REASONING_TAIL_MARKER,
    REASONING_VISIBLE_ROWS,
    ReasoningBlock,
)
from local_operator.tui.widgets.transcript import TranscriptView
from tests.unit.tui.conftest import StyledTranscriptApp


async def _settle(pilot) -> None:  # type: ignore[no-untyped-def]
    await pilot.pause()
    await pilot.pause()


def test_the_labels_are_the_ones_the_frame_shows() -> None:
    """A block whose header is the only thing naming it must name itself.

    Asserted rather than assumed because the value is a user-facing word that a
    refactor could change without any other test noticing (the header is the
    difference between dim prose and "there is a truncated answer above me").
    It must NOT be the working line's word: that line sits one row below and
    answers a different question — the state of the call, not what the model is
    producing — and two adjacent rows saying "thinking" is the redundancy the
    UX review flagged (U2, and design D1 for the same pair).
    """
    assert REASONING_LABEL == "reasoning"
    assert REASONING_LABEL != "thinking"


def _plain(block: ReasoningBlock) -> str:
    """The block's authored text, narrowed for the type checker.

    ``TranscriptBlock.renderable`` is typed ``RenderableType | None`` (other
    blocks hand Rich a Group or a Markdown), so reading ``.plain`` off it is a
    type error even though THIS block always authors a ``Text``. Asserting the
    type is stricter than a cast: a future change that hands Rich something else
    fails here instead of silently comparing rendered markup.
    """
    renderable = block.renderable
    assert isinstance(renderable, Text), type(renderable)
    return renderable.plain


@pytest.mark.asyncio
async def test_reasoning_streams_labelled_above_the_answer_and_stops_growing() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 30)) as pilot:
        view = app.query_one(TranscriptView)
        block = ReasoningBlock()
        view.append_block(block)
        block.update_text("weighing the options")
        await _settle(pilot)

        rendered = _plain(block)
        assert REASONING_LABEL in rendered
        assert "weighing the options" in rendered
        # Header row + one content row, PINNED: the pinned height is what keeps
        # a growing block from reserving rows it never paints.
        assert block.size.height == 2

        block.update_text(" ".join(f"word{index}" for index in range(400)))
        await _settle(pilot)
        # Bounded in ROWS: a long think cannot push the answer off the viewport.
        assert block.size.height == REASONING_VISIBLE_ROWS + 1
        # And the live tail is what is painted (the newest words), not the head.
        assert "word399" in _plain(block)


@pytest.mark.asyncio
async def test_retiring_the_phase_collapses_it_to_its_header() -> None:
    """``retire`` collapses the phase: ONE row saying this call reasoned.

    Removing the block would delete rows from under the reader's cursor, and
    keeping its rows was the other extreme — one ~7-row block per model call,
    dozens per session, with no way to dismiss them (UX review round 1, U1).
    The collapsed row is the middle the tool cards already use, and it still
    refuses text: the fragments that arrive after the phase closes must not
    grow a block the reader has watched settle.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 30)) as pilot:
        view = app.query_one(TranscriptView)
        block = ReasoningBlock()
        view.append_block(block)
        block.update_text(" ".join(f"word{index}" for index in range(200)))
        await _settle(pilot)
        assert block.size.height == REASONING_VISIBLE_ROWS + 1

        block.retire()
        block.update_text("a later thought that must never be painted")
        await _settle(pilot)

        assert block.size.height == 1
        assert REASONING_LABEL in _plain(block)
        assert "word199" not in _plain(block), "the rows are shed, not frozen"
        assert "later thought" not in _plain(block)


@pytest.mark.asyncio
async def test_a_wider_lane_rewraps_the_same_tail_at_the_same_row_bound() -> None:
    """A re-wrap is a HEIGHT change, and the bound survives it.

    The block authors its own rows, so a stale width would bake the old fold in
    (the discipline ``refit_width`` exists for). Asserted by capturing the wrap
    at two widths: the row count may move, the ceiling may not.
    """
    text = " ".join(f"token{index}" for index in range(60))
    narrow = ReasoningBlock()
    narrow.update_text(text)
    narrow.refit_width(40)
    wide = ReasoningBlock()
    wide.update_text(text)
    wide.refit_width(120)

    assert len(_plain(narrow).splitlines()) >= len(_plain(wide).splitlines())
    assert len(_plain(narrow).splitlines()) <= REASONING_VISIBLE_ROWS + 1
    assert len(_plain(wide).splitlines()) <= REASONING_VISIBLE_ROWS + 1


def test_the_character_tail_bound_keeps_the_newest_words() -> None:
    """The character bound is what makes a flush O(1) in the thinking length.

    Held at the block level (no app needed): the tail is sliced BEFORE wrapping,
    so a model that has been thinking for two minutes costs the same per flush as
    one that started a second ago — and what a reader gets is the newest words.
    RETENTION is bounded too, not just the paint: the block holds the raw tail
    rather than the whole phase (review round 1, MINOR-2), which is why the
    retained bound is larger than the painted one.
    """
    block = ReasoningBlock()
    block.update_text("H" * (REASONING_TAIL_CHARS * 10) + " the newest words")
    plain = _plain(block)

    assert "the newest words" in plain
    assert plain.count("H") <= REASONING_TAIL_CHARS
    assert len(block.text()) <= REASONING_RETAINED_CHARS


def test_a_dropped_row_run_is_marked() -> None:
    """Rows the block shed are marked, so the reader knows the block is a tail.

    Without the marker a reader who looks away returns mid-sentence with no way
    to tell whether the model started mid-thought or the block dropped rows —
    the "is what I am seeing honest?" question this block exists to answer (UX
    review round 1, U3).
    """
    long_phase = ReasoningBlock()
    long_phase.update_text(" ".join(f"token{index}" for index in range(400)))
    lines = _plain(long_phase).splitlines()
    body = lines[1:]
    assert body, "a long phase paints rows"
    # The rows carry the hanging indent, so the marker sits just inside it.
    assert body[0].lstrip().startswith(REASONING_TAIL_MARKER)
    assert "…" not in "".join(
        row.lstrip() for row in body[1:]
    ), "one marker, on the first painted row"
    # The marker REPLACES the row's first cells rather than extending it, so the
    # row still fits the lane the wrap was computed for.
    assert len(lines) <= REASONING_VISIBLE_ROWS + 1

    short_phase = ReasoningBlock()
    short_phase.update_text("one short thought, nothing dropped")
    assert not _plain(short_phase).splitlines()[1].lstrip().startswith(REASONING_TAIL_MARKER)


def test_the_reasoning_ink_clears_the_contrast_bar() -> None:
    """Both rows are `muted`; the body was `dim`, below AA on both ramps.

    Measured by the designer in the rendered frames: `dim` is 4.55:1 on the dark
    ramp and 3.77:1 on the light one, under the 4.5:1 bar for prose a reader is
    meant to read; `muted` clears both (8.62:1 / 7.18:1). The header keeps its
    distinctness from the glyph and the hanging indent rather than from being
    brighter than the text under it (design round 1, D2).
    """
    block = ReasoningBlock()
    block.update_text("weighing the options")
    renderable = block.renderable
    assert isinstance(renderable, Text)
    inks = {
        span.style.color
        for span in renderable.spans
        if isinstance(span.style, Style) and span.style.color is not None
    }
    assert Color.parse(theme_mod.semantic_color("muted")) in inks
    assert Color.parse(theme_mod.semantic_color("dim")) not in inks
