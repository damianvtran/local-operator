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

What is asserted is the COLUMN, never frame bytes or whole-row text: hover
legitimately rewrites a row's own text (``_build_row`` lights the ``⟨expand⟩``
offer when the row is hovered), so "the frame is unchanged" is the wrong
instrument even where the alignment is right. The shared column is what must not
move under any of these transitions.
"""

from __future__ import annotations

import pytest
from rich.text import Text
from textual.app import App

from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import (
    TOOL_NAME_COL,
    TranscriptBlock,
    TranscriptView,
)
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


@pytest.mark.asyncio
async def test_a_call_that_starts_earns_the_column_for_its_row() -> None:
    """The promotion out of ``composing`` is a path that moves the column.

    ``contributes_name`` is False while a call is still being dictated and True
    once the call it names has started, so a row's name begins counting towards
    the spine at exactly this transition — the inverse of entering ``composing``,
    which already invalidates the column. Without it the ledger stays on the
    floor and the live row paints its own name truncated (``list_va…``), and
    neither settling nor hovering repairs that: hover re-fits the row at the
    stale column, so only some unrelated later resync would.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 24)) as pilot:
        view = app.query_one(TranscriptView)
        settled = _bash("c1", "echo alpha")
        view.append_block(settled)
        await _settle(pilot)
        narrow = summary_col(app, settled, "echo alpha")

        live = ToolCard("live", "bash", {})
        view.append_block(live)
        live.set_composing(12, LONGER_TOOL)
        await _settle(pilot)
        # Dictated, not started: the name may not move the ledger's column yet.
        assert summary_col(app, settled, "echo alpha") == narrow

        live.begin_running(LONGER_TOOL, {"command": "list them"}, None)
        await _settle(pilot)

        shared = summary_col(app, live, "list them")
        assert shared > narrow
        assert summary_col(app, settled, "echo alpha") == shared
        # The row the column grew FOR is painted in full, not ellipsised into a
        # column that would have fitted it.
        assert LONGER_TOOL in painted_row(app, live)


@pytest.mark.asyncio
async def test_removing_the_widest_row_brings_the_spine_back_down() -> None:
    """The shrink half of the funnel: only a re-scan can say how far.

    Growth can be answered by the newcomer's own name; a removal cannot, so it
    keeps the full re-derivation — and the rows LEFT BEHIND are the ones that
    have to be told, which is what dropping the cache alone failed to do.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 24)) as pilot:
        view = app.query_one(TranscriptView)
        first = _bash("c1", "echo alpha")
        second = _bash("c2", "echo beta")
        view.append_block(first)
        view.append_block(second)
        await _settle(pilot)
        floor = summary_col(app, first, "echo alpha")

        wide = ToolCard("wide", WIDE_TOOL, {"command": "wide row"}, "")
        view.append_block(wide)
        await _settle(pilot)
        grown = summary_col(app, wide, "wide row")
        assert grown > floor
        assert summary_col(app, first, "echo alpha") == grown
        assert summary_col(app, second, "echo beta") == grown

        view.remove_block(wide)
        await _settle(pilot)

        assert summary_col(app, first, "echo alpha") == floor
        assert summary_col(app, second, "echo beta") == floor
        assert view.tool_name_col == TOOL_NAME_COL


@pytest.mark.asyncio
async def test_every_row_agrees_on_the_column_rung_at_narrow_widths() -> None:
    """Across the narrow threshold no row may diverge from its neighbours.

    ``tool_name_col`` is the DERIVED column; what a row paints with is that value
    clamped by the row's OWN frame — ``ToolCard._name_col`` answers the floor
    below ``NAME_GROWTH_MIN_ROW``, and ``name_budget`` shrinks it further — so
    the derived value and the painted one legitimately disagree there. The
    property that has to hold is the one this file is about: at every rung all
    rows agree, in both directions across the threshold. A rung where one row
    disagrees is the tear, not the clamp.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 24)) as pilot:
        view = app.query_one(TranscriptView)
        first = _bash("c1", "echo alpha")
        second = _bash("c2", "echo beta")
        wide = ToolCard("wide", WIDE_TOOL, {"command": "wide row"}, "")
        for card in (first, second, wide):
            view.append_block(card)
        await _settle(pilot)

        def rung(summary: str) -> int:
            return summary_col(app, first, "echo alpha")

        at_full_width = rung("echo alpha")
        assert summary_col(app, wide, "wide row") == at_full_width

        for width in (68, 50):
            await pilot.resize_terminal(width, 24)
            await _settle(pilot)
            narrowed = rung("echo alpha")
            assert narrowed < at_full_width, "the row clamp has to bite below the threshold"
            assert summary_col(app, second, "echo beta") == narrowed
            assert summary_col(app, wide, "wide row") == narrowed

        await pilot.resize_terminal(100, 24)
        await _settle(pilot)
        assert rung("echo alpha") == at_full_width
        assert summary_col(app, wide, "wide row") == at_full_width
