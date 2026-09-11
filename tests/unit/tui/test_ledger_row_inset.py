"""The ledger's SHARED left inset, pinned across every ledger row type.

A tool card, a scheduled-wake receipt and an inbound peer receipt are three
row types on one ledger, and the operator reported the defect that gives this
file its name: "sometimes the peer message doesn't have the proper left
padding like the other tool cards". The wake and peer rows were built without
the one-cell inset `ToolCard` draws (:data:`ROW_INDENT`), so their icon sat
flush against the card's own left wall while the tool rows around them were
inset — icon, name column and summary each one cell left of the rows they sat
between, which breaks the column the ledger exists to be.

These assert the INVARIANT (every ledger row's icon starts on the same cell)
and the ladder (the inset is given up below :data:`ROW_INDENT_MIN_WIDTH` for
all three) rather than a hand-written row prefix: a test spelling the indent
into an expected string passes for whatever the string says and cannot see the
rows disagree with each other, which is the bug that was reported.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from rich.text import Text

from local_operator.tui.app import OperatorApp
from local_operator.tui.glyphs import tool_icon
from local_operator.tui.widgets.tool_card import (
    OUTPUT_INDENT,
    ROW_INDENT,
    ROW_INDENT_MIN_WIDTH,
    ToolCard,
)
from local_operator.tui.widgets.transcript import PeerMessageBlock, WakeBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: A live fire: the envelope the model gets, then the prompt it delivers.
WAKE_TEXT = (
    "(alarm) Scheduled wake w3 (3/8, every 1h) — "
    'cancel with wake({op:"cancel",id:"w3"}) once its goal is met.\n\ncheck the build'
)
SENDER = {"pid": 48213, "conversation_name": "lo-release-window"}


def _built_row(block: object) -> Text:
    """The row AS PAINTED — the applied renderable, not a re-build.

    Read off the widget rather than recomputed so a resize, a hover or an
    expansion cannot let the assertion measure a row the user never saw.
    """
    rendered = block.renderable  # type: ignore[attr-defined]
    assert isinstance(rendered, Text)
    return rendered


def _icon_column(row: Text) -> int:
    """Cells of leading whitespace before the first painted glyph."""
    plain = row.plain
    return cell_len(plain) - cell_len(plain.lstrip(" "))


def _ledger_rows() -> list[ToolCard | WakeBlock | PeerMessageBlock]:
    return [
        ToolCard("t1", "read", {"path": "a.py"}),
        WakeBlock(WAKE_TEXT),
        PeerMessageBlock("gates are green", SENDER),
    ]


@pytest.mark.asyncio
async def test_every_ledger_row_starts_its_icon_on_the_same_cell() -> None:
    """One transcript, all three row types: the glyph column is a column.

    Asserted as an equality ACROSS the rows and against the tool card, which
    is the row that already had the inset — a literal expectation in one row
    would not notice the other two drifting away from it.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        rows = _ledger_rows()
        for block in rows:
            app._append_block(block)
        await pilot.pause()
        await pilot.pause()

        columns = {type(block).__name__: _icon_column(_built_row(block)) for block in rows}
        assert len(set(columns.values())) == 1, columns
        assert set(columns.values()) == {ROW_INDENT}, columns
        # And the first painted cell of each row is that row's OWN glyph — the
        # alignment is the icon column, not two rows agreeing on whitespace.
        for block in rows:
            assert _built_row(block).plain.lstrip(" ").startswith(tool_icon(block.tool_name))


@pytest.mark.parametrize(
    "make",
    [
        lambda: ToolCard("t1", "read", {"path": "a.py"}),
        lambda: WakeBlock(WAKE_TEXT),
        lambda: PeerMessageBlock("gates are green", SENDER),
    ],
    ids=["tool", "wake", "peer"],
)
def test_the_inset_ladder_is_the_same_for_every_row_type(make) -> None:
    """Present at the threshold, gone below it — for all three row types.

    The inset is breathing room, and breathing room is the first thing a
    narrow ledger spends: one more cell there pushes the row past the rung
    where the `⟨∅⟩` answer survives. Asserted at the boundary because that is
    the part that can regress, and per row type because a ladder that only
    `ToolCard` walks is how the two rows drifted apart in the first place.
    """
    block = make()
    at_threshold = block._build_row(ROW_INDENT_MIN_WIDTH).plain
    below = block._build_row(ROW_INDENT_MIN_WIDTH - 1).plain
    assert _icon_column(Text(at_threshold)) == ROW_INDENT
    assert _icon_column(Text(below)) == 0


@pytest.mark.asyncio
async def test_the_copy_gutter_agrees_with_the_row_as_built() -> None:
    """The gutter is read back off the row, never assumed.

    Both the inset and the icon are this widget's own layout, so both leave
    with the copy. A gutter that over-counts eats the first character of the
    row's name; an under-count pastes a leading space into a bug report. The
    check is against the built row at BOTH rungs, so the narrow case cannot
    silently keep a wide-width gutter.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        rows = _ledger_rows()
        for block in rows:
            app._append_block(block)
        await pilot.pause()
        await pilot.pause()

        for block in rows:
            assert block.copy_gutter(0) == _icon_column(_built_row(block)) + ToolCard.ICON_COLS
            assert block.copy_gutter(1) == OUTPUT_INDENT

        # Below the threshold the gutter sheds the inset it is no longer drawn
        # with, so the copied row still starts at the name.
        for width in (ROW_INDENT_MIN_WIDTH, ROW_INDENT_MIN_WIDTH - 1):
            for block in rows:
                block._built_width = width
                built = block._build_row(width).plain
                assert block.copy_gutter(0) == _icon_column(Text(built)) + ToolCard.ICON_COLS
