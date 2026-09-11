"""The ledger's shared name column: ONE spine, moved as one.

``TranscriptView`` sizes a single column from the longest tool name in the
ledger, and every row draws its summary after it. The growth is deliberate —
two MCP tools sharing a seven-character prefix stay distinguishable wherever the
frame can afford the width — so the property that has to hold is not "the
column is a constant" but "a row is never painted on a column the other rows
are not on".

It stopped holding on three paths, all of which left painted rows behind:

* ``insert_blocks`` dropped the derived width and repainted nobody, so the rows
  already on screen kept the width they were given while the first revealed row
  derived a wider one for itself. Paging history in tore the ledger.
* ``remove_block`` and ``clear_blocks`` did the same, in the shrink direction.
* ``append_block``'s growth path early-returned when the derived width was
  unset, which is exactly the state those paths leave behind: the appended row
  painted on a freshly derived column and the rows already painted kept theirs.

In every case hovering a row repaired that row alone (``on_enter`` →
``refresh_row``), so the ledger visibly re-aligned under the pointer one row at
a time. The assertions here are on the frame the compositor painted, with no
pointer anywhere in the sequence.
"""

from __future__ import annotations

import pytest
from rich.text import Text
from textual.app import App

from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import TranscriptBlock, TranscriptView
from tests.unit.tui.conftest import StyledTranscriptApp

#: A name long enough to widen the column past the floor. It renders through
#: ``display_name`` as a SHORTER label, so the tests measure what is painted
#: rather than what was passed in.
WIDE_TOOL = "mcp__linear_create_initiative"
LONGER_TOOL = "list_variables"


def painted_row(app: App[None], block: TranscriptBlock) -> str:
    """The row the frame actually painted for ``block``.

    Read from the compositor rather than from the widget's own renderable: the
    claim under test is about what a reader sees, and a card holds content it
    has not painted yet (`refresh_row` builds it whether or not the row is on
    frame).
    """
    strips = list(app.screen._compositor.render_strips())
    y = block.region.y
    assert 0 <= y < len(strips), "the ledger row is not on the painted frame"
    return strips[y].text


def summary_col(app: App[None], block: TranscriptBlock, summary: str) -> int:
    """Cell the row's SUMMARY starts in — the far edge of the shared name column."""
    return painted_row(app, block).index(summary)


def _bash(call_id: str, summary: str) -> ToolCard:
    return ToolCard(call_id, "bash", {"command": summary}, "")


async def _settle(pilot) -> None:
    """Wait for the SETTLED frame, which is the one a reader is left with.

    A card builds its row in its constructor, where it has no parent to ask for
    the shared column, so the first paint after a mount can still be that build;
    the ledger's own layout pass replaces it on the next. Two pauses is the same
    settle the peer-card suite uses for the same reason.
    """
    await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
async def test_paging_an_older_row_in_repaints_the_rows_already_on_screen() -> None:
    """Scroll-up reveal: the newcomers may not paint on a wider spine alone."""
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 24)) as pilot:
        view = app.query_one(TranscriptView)
        first = _bash("c1", "echo alpha")
        second = _bash("c2", "echo beta")
        view.append_block(first)
        view.append_block(second)
        await _settle(pilot)
        narrow = summary_col(app, first, "echo alpha")
        assert summary_col(app, second, "echo beta") == narrow

        # An OLDER row, revealed above the viewport, whose name is longer than
        # anything the ledger has drawn so far.
        revealed = ToolCard("older", WIDE_TOOL, {"command": "reveal me"}, "")
        view.insert_blocks(0, [revealed])
        await _settle(pilot)

        # The column grew, and all three rows moved with it — no hover anywhere.
        shared = summary_col(app, revealed, "reveal me")
        assert shared > narrow
        assert summary_col(app, first, "echo alpha") == shared
        assert summary_col(app, second, "echo beta") == shared


@pytest.mark.asyncio
async def test_appending_a_longer_name_repaints_the_rows_already_painted() -> None:
    """Append growth has to reach the rows behind it, not only the newcomer.

    The append is reached with the derived column already invalidated and rows
    still painted — the state a removal or a page reveal leaves the ledger in,
    and the one the growth path's guard on the (unset) cache read as "nothing to
    do".
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 24)) as pilot:
        view = app.query_one(TranscriptView)
        first = _bash("c1", "echo alpha")
        second = _bash("c2", "echo beta")
        view.append_block(first)
        view.append_block(second)
        await _settle(pilot)
        narrow = summary_col(app, first, "echo alpha")

        wide = ToolCard("wide", WIDE_TOOL, {"command": "wide row"}, "")
        view.append_block(wide)
        await _settle(pilot)
        assert summary_col(app, wide, "wide row") > narrow

        # Removing a ledger row invalidates the derived column — deliberately
        # without asserting the frame here: this test is about what the append
        # does with the ledger in that state.
        view.remove_block(wide)
        await _settle(pilot)

        longer = ToolCard("longer", LONGER_TOOL, {"command": "list them"}, "")
        view.append_block(longer)
        await _settle(pilot)

        expected = summary_col(app, longer, "list them")
        assert expected > narrow
        assert summary_col(app, first, "echo alpha") == expected
        assert summary_col(app, second, "echo beta") == expected


@pytest.mark.asyncio
async def test_a_page_that_does_not_move_the_column_repaints_nothing() -> None:
    """The repaint is driven by the WIDTH moving, not by the event happening.

    A reveal must be able to hand the ledger a page without walking every row it
    already painted, and the O(1) append fast path must stay O(1) — otherwise
    this fix would have bought alignment with a per-event rescan, which is the
    cost the ledger's growth path exists to avoid. Asserted as call COUNTS
    rather than a duration: a threshold calibrated on this machine is not a
    threshold on CI, and "how many rows were re-rendered" is a fact about work
    rather than about the clock.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 24)) as pilot:
        view = app.query_one(TranscriptView)
        first = _bash("c1", "echo alpha")
        second = _bash("c2", "echo beta")
        view.append_block(first)
        view.append_block(second)
        await _settle(pilot)

        repaints: list[str] = []
        for card in (first, second):
            original = card.refresh_row

            def counting(card=card, original=original) -> None:
                repaints.append(card.tool_call_id)
                original()

            card.refresh_row = counting  # type: ignore[method-assign]

        # A page the column already fits: one derivation, no repaints.
        view.insert_blocks(0, [_bash("old1", "echo old")])
        await _settle(pilot)
        assert repaints == []

        # A page carrying a longer name: every row behind it re-renders, once.
        view.insert_blocks(0, [ToolCard("old2", LONGER_TOOL, {"command": "list them"}, "")])
        await _settle(pilot)
        assert sorted(repaints) == ["c1", "c2"]


def _on_frame(app: App[None], block: TranscriptBlock) -> bool:
    """Whether the frame is currently painting ``block`` at all."""
    return 0 <= block.region.y < len(list(app.screen._compositor.render_strips()))


def composed_row(block: TranscriptBlock) -> str:
    """The row text ``block`` is holding, for a row the frame is not painting.

    The compositor cannot speak for a row outside the viewport, so the widget's
    own composed content is the instrument there — and it is the honest one for
    the claim: this is what the row will paint when it is revealed, unless the
    reveal itself rebuilds it.
    """
    rendered = block.render()
    # The row is a Rich ``Text`` in every renderable state the ledger has; the
    # cast is for the union ``Widget.render`` declares.
    return rendered.plain.splitlines()[0] if isinstance(rendered, Text) else str(rendered)


@pytest.mark.asyncio
async def test_a_row_scrolled_out_of_view_is_repainted_too() -> None:
    """A row the frame is not showing is rebuilt as well.

    Read from the row's own composed content, because that is the only thing a
    cell outside the viewport can be judged by: the reader who scrolls back to
    it is who this is for, and a pointer never reaches it (hovering is what used
    to repair a row).
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 20)) as pilot:
        view = app.query_one(TranscriptView)
        first = _bash("c0", "echo step0")
        view.append_block(first)
        rest = [_bash(f"c{index}", f"echo step{index}") for index in range(1, 14)]
        for card in rest:
            view.append_block(card)
        await _settle(pilot)
        assert not _on_frame(app, first), "the fixture has to push the row off the frame"
        narrow = composed_row(rest[-1]).index("echo step13")

        # The column moves while `first` is above the viewport.
        view.insert_blocks(0, [ToolCard("old", LONGER_TOOL, {"command": "list them"}, "")])
        await _settle(pilot)

        assert view.tool_name_col > narrow
        off_frame = composed_row(first).index("echo step0")
        assert off_frame > narrow
        assert off_frame == composed_row(rest[-1]).index("echo step13")
