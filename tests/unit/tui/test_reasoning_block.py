"""The reasoning block: the model's thinking, streamed dim, then retired.

Behaviour first, under the REAL sheet (``StyledTranscriptApp``): the block's own
docstring claims three properties — a labelled block of its own, a bounded live
TAIL, and a transient lifetime — and each of them is a property of the rendered
frame, not of the source.
"""

from __future__ import annotations

import pytest
from rich.text import Text

from local_operator.tui.widgets.reasoning import (
    REASONING_LABEL,
    REASONING_TAIL_CHARS,
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
    """
    assert REASONING_LABEL == "thinking"


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
async def test_retiring_the_phase_freezes_it_where_the_answer_found_it() -> None:
    """``retire`` stops accepting text: the rows a user watched must not move.

    The reason is the same one the finalized-block protocol exists for elsewhere
    in the transcript — a block that keeps changing under a reader — plus one of
    its own: the answer's block mounts beside it in the same frame, so a phase
    that kept absorbing fragments would grow the block the user is reading while
    the answer arrives below it.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 30)) as pilot:
        view = app.query_one(TranscriptView)
        block = ReasoningBlock()
        view.append_block(block)
        block.update_text("weighing the options")
        await _settle(pilot)
        before = _plain(block)

        block.retire()
        block.update_text("a later thought that must never be painted")
        await _settle(pilot)

        assert _plain(block) == before
        assert "later thought" not in _plain(block)


@pytest.mark.asyncio
async def test_a_wider_lane_rewraps_the_same_tail_at_the_same_row_bound() -> None:
    """A re-wrap is a HEIGHT change, and the bound survives it.

    The block authors its own rows, so a stale width would bake the old fold in
    (the discipline ``Refit_width`` exists for). Asserted by capturing the wrap
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
    """
    block = ReasoningBlock()
    block.update_text("H" * (REASONING_TAIL_CHARS + 500) + " the newest words")
    plain = _plain(block)

    assert "the newest words" in plain
    assert plain.count("H") <= REASONING_TAIL_CHARS
    assert "…" not in plain, "the slice must be hard, not an elision the reader cannot see through"
