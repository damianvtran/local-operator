"""Adaptive spacing — the transcript's vertical rhythm.

The rule replaced a flat "everything is flush" layout. Flush is right for a
run of one-line notices (one thing said in parts) and wrong for prose, which
then runs into whatever follows it, and wrong for tool rows, where flush
stacking made a batch of separate actions read as one wrapped block — the
field report was "there should be one line spacing between each". So the gap
is decided per block from what precedes it AND from whether the block is
airy, and the decision is a pure function (:func:`needs_gap_above`) tested
here in isolation, plus an integration pass through a live ``TranscriptView``
that checks the class actually lands.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import (
    GAP_CLASS,
    NoticeBlock,
    TranscriptBlock,
    TranscriptView,
    UserBlock,
    WorkingBlock,
    needs_gap_above,
)
from tests.unit.tui.conftest import StyledTranscriptApp


class _Stub(TranscriptBlock):
    """A block whose spacing inputs are set directly, with no rendering.

    The decision function must be reasonable about blocks it has never seen
    — the rule is about kind and height, not about class identity.
    """

    def __init__(
        self,
        kind: str,
        rows: int = 1,
        *,
        lead: bool = False,
        transient: bool = False,
        airy: bool = False,
    ):
        super().__init__()
        setattr(self, "SPACING_KIND", kind)
        setattr(self, "SPACING_LEAD", lead)
        setattr(self, "SPACING_TRANSIENT", transient)
        setattr(self, "SPACING_AIRY", airy)
        self._rows = rows

    def spans_multiple_rows(self) -> bool:
        return self._rows > 1


# --- the rule --------------------------------------------------------------


def test_the_first_block_never_takes_a_gap() -> None:
    """Content meets the top edge; a leading blank row is wasted screen."""
    assert needs_gap_above(None, _Stub("tool")) is False
    assert needs_gap_above(None, _Stub("user", lead=True)) is False


def test_consecutive_one_line_blocks_stay_flush_unless_they_are_airy() -> None:
    """Two kinds of "a list", spaced two different ways.

    A run of one-line notices IS one thing said in parts, so it stays dense.
    A run of tool rows is a list of SEPARATE actions, and stacked flush the
    user read them as one wrapped block and reported the app as broken
    ("there should also be one line spacing between each"). Airy is how a
    block says which of the two it is; before it existed both stacked.
    """
    assert needs_gap_above(_Stub("notice", rows=1), _Stub("notice", rows=1)) is False
    assert needs_gap_above(_Stub("tool", rows=1, airy=True), _Stub("tool", rows=1, airy=True)) is (
        True
    )


def test_an_airy_block_still_meets_the_top_edge() -> None:
    """Airy separates NEIGHBOURS; it is not ``SPACING_LEAD``. With nothing
    above, a blank first row is wasted screen — which is exactly the row the
    field report also complained about ("too much spacing above")."""
    assert needs_gap_above(None, _Stub("tool", airy=True)) is False


def test_a_multi_row_block_pushes_the_next_one_away() -> None:
    """Prose, or an expanded tool card, needs air or the next block reads
    as its continuation."""
    assert needs_gap_above(_Stub("tool", rows=8), _Stub("tool", rows=1)) is True
    assert needs_gap_above(_Stub("assistant", rows=6), _Stub("assistant", rows=1)) is True


def test_a_change_of_kind_always_opens_a_gap() -> None:
    """The change of subject is the cue, whatever the heights are."""
    assert needs_gap_above(_Stub("tool", rows=1), _Stub("assistant", rows=1)) is True
    assert needs_gap_above(_Stub("assistant", rows=1), _Stub("tool", rows=1)) is True
    assert needs_gap_above(_Stub("notice", rows=1), _Stub("rich", rows=1)) is True


def test_a_turn_leading_block_always_takes_a_gap() -> None:
    """A user prompt starts a turn — it gets air even after its own kind."""
    assert needs_gap_above(_Stub("user"), _Stub("user", lead=True)) is True
    assert needs_gap_above(_Stub("tool"), _Stub("user", lead=True)) is True


def test_transient_blocks_neither_take_a_gap_nor_give_one() -> None:
    """The working line appears and vanishes mid-turn: if it could open a
    gap, that blank row would flash in and out under the settled rows."""
    working = _Stub("working", transient=True)
    assert needs_gap_above(_Stub("tool"), working) is False
    assert needs_gap_above(working, _Stub("assistant")) is False


# --- the real block classes carry the right metadata -----------------------


def test_block_classes_declare_distinct_spacing_kinds() -> None:
    kinds = {
        UserBlock.SPACING_KIND,
        NoticeBlock.SPACING_KIND,
        AssistantBlock.SPACING_KIND,
        ToolCard.SPACING_KIND,
    }
    assert len(kinds) == 4, kinds
    assert UserBlock.SPACING_LEAD is True
    assert WorkingBlock.SPACING_TRANSIENT is True
    assert ToolCard.SPACING_TRANSIENT is False
    # The tool row is the ONLY airy block. Prose and notices are not: a
    # paragraph is already separated by its kind change, and a run of notices
    # is deliberately dense. Airy spreading to a second class is how "one
    # blank row between actions" becomes "a blank row between everything".
    assert ToolCard.SPACING_AIRY is True
    assert UserBlock.SPACING_AIRY is False
    assert NoticeBlock.SPACING_AIRY is False
    assert AssistantBlock.SPACING_AIRY is False


def test_a_collapsed_tool_card_reports_a_single_row() -> None:
    card = ToolCard("t", "bash", {"command": "ls"})
    card.mark_done("one\ntwo\nthree")
    assert card.spans_multiple_rows() is False
    card.toggle_expanded()
    assert card.spans_multiple_rows() is True


def test_assistant_multirow_is_answered_from_the_source_text() -> None:
    """Cheap and honest: no Markdown render just to make a spacing decision."""
    block = AssistantBlock()
    assert block.spans_multiple_rows() is False  # empty
    block._full_text = "one short line"
    assert block.spans_multiple_rows() is False
    block._full_text = "a paragraph\n\nand another"
    assert block.spans_multiple_rows() is True
    block._full_text = "x" * 500  # wraps at any sane width
    assert block.spans_multiple_rows() is True


# --- integration through a live container ----------------------------------


class _Harness(App[None]):
    """Bare app hosting only the transcript, so nothing else can style it."""

    def compose(self) -> ComposeResult:
        yield TranscriptView()


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [40, 80, 100, 200])
async def test_a_ledger_of_tool_rows_is_one_row_each_under_the_real_sheet(
    width: int,
) -> None:
    """Two regressions in one frame.

    The first: cards that settle BEFORE their first layout used to resolve to
    two rows each and stay there, double-spacing the whole ledger at some
    widths and not others. Settling without an intervening pause is the
    realistic path — the engine emits tool start and tool end in one
    synchronous burst — so the test reproduces exactly that ordering rather
    than a convenient one.

    The second: the gap between actions is exactly ONE row, measured off the
    painted geometry rather than off the class. The class only says a margin
    was asked for; the row offsets say what the user sees, and they are what
    catches a `.tool-card` margin sneaking in on top of `.gap-above` and
    doubling it.
    """
    cases: list[tuple[str, dict[str, object]]] = [
        ("bash", {"command": "pytest tests/unit -q"}),
        ("grep", {"pattern": "parse"}),
        ("read", {"path": "src/parser.py"}),
        ("write", {"path": "src/parser.py"}),
    ]
    app = StyledTranscriptApp()
    async with app.run_test(size=(width, 24)) as pilot:
        view = app.query_one(TranscriptView)
        cards = []
        for index, (name, args) in enumerate(cases):
            card = ToolCard(str(index), name, args)
            view.append_block(card)
            cards.append(card)
        # No pause: settle in the same burst the append happened in.
        cards[0].mark_done("66 passed")
        cards[1].mark_failed("permission denied while reading the file")
        cards[2].mark_done("ok")
        cards[3].mark_done("Overwrote src/parser.py.", {"added": 12, "removed": 3})
        await pilot.pause()
        await pilot.pause()

        assert [card.size.height for card in cards] == [1, 1, 1, 1]
        # First meets the top edge; every following action takes its own row
        # of air. Two cells of pitch for a one-row card IS one blank row.
        assert not cards[0].has_class(GAP_CLASS)
        assert all(card.has_class(GAP_CLASS) for card in cards[1:])
        offsets = [card.region.y for card in cards]
        assert [b - a for a, b in zip(offsets, offsets[1:])] == [2, 2, 2], offsets


@pytest.mark.asyncio
async def test_a_real_mouse_click_expands_and_collapses_under_the_real_sheet() -> None:
    """The pinned collapsed height must not defeat the expansion, and the one
    blank row below survives a card growing to four rows and back."""
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 24)) as pilot:
        view = app.query_one(TranscriptView)
        card = ToolCard("a", "bash", {"command": "ls -la"})
        below = ToolCard("b", "bash", {"command": "pwd"})
        view.append_block(card)
        view.append_block(below)
        card.mark_done("total 8\nfile-a\nfile-b")
        below.mark_done("/tmp")
        await pilot.pause()
        assert card.size.height == 1
        assert below.region.y - card.region.y == 2

        await pilot.click(card)
        await pilot.pause()
        assert card.expanded is True
        assert card.size.height == 4
        # Still exactly one blank row: the expansion added rows to the card,
        # not to the space under it.
        assert below.region.y - card.region.y == 5

        await pilot.click(card)
        await pilot.pause()
        assert card.expanded is False
        assert card.size.height == 1
        assert below.region.y - card.region.y == 2


@pytest.mark.asyncio
async def test_container_applies_the_gap_class_only_where_the_rule_says() -> None:
    app = _Harness()
    async with app.run_test():
        view = app.query_one(TranscriptView)

        first = ToolCard("a", "read", {"path": "one.py"})
        second = ToolCard("b", "read", {"path": "two.py"})
        prose = AssistantBlock()
        notice_a = NoticeBlock("first", "info")
        notice_b = NoticeBlock("second", "info")
        for block in (first, second, prose, notice_a, notice_b):
            view.append_block(block)

        assert not first.has_class(GAP_CLASS)  # nothing above it
        assert second.has_class(GAP_CLASS)  # a separate action gets its own row
        assert prose.has_class(GAP_CLASS)  # different kind
        assert notice_a.has_class(GAP_CLASS)  # different kind again
        assert not notice_b.has_class(GAP_CLASS)  # a list of notices stays dense


@pytest.mark.asyncio
async def test_the_working_line_does_not_disturb_the_rows_around_it() -> None:
    """The transient line is invisible to spacing from BOTH sides.

    Checked on a non-airy pair, because that is where forgetting to skip the
    transient anchor actually shows: two notices with the working line
    between them must stay as flush as they would be side by side. On tool
    rows the airy rule would produce a gap either way and hide the bug.
    """
    app = _Harness()
    async with app.run_test():
        view = app.query_one(TranscriptView)

        first = NoticeBlock("one", "info")
        working = WorkingBlock()
        second = NoticeBlock("two", "info")
        card = ToolCard("a", "read", {"path": "one.py"})
        for block in (first, working, second, card):
            view.append_block(block)

        assert not working.has_class(GAP_CLASS)
        # Spaced against the NOTICE above it, not against the transient line
        # between them — which would have made it a change of kind and a gap.
        assert not second.has_class(GAP_CLASS)
        assert card.has_class(GAP_CLASS)


@pytest.mark.asyncio
async def test_expanding_a_card_keeps_the_gap_below_it_at_one_row() -> None:
    """The gap below is re-decided on every height change, and lands the same.

    It used to FLIP: a collapsed tool row was flush against the next one, and
    expanding it opened a gap the collapse took back. Now every action owns a
    blank row unconditionally, so the interesting property is that the
    re-decision is IDEMPOTENT — a card growing to twenty rows and back must
    not accumulate, drop, or double the single row beneath it.
    """
    app = _Harness()
    async with app.run_test():
        view = app.query_one(TranscriptView)

        card = ToolCard("a", "bash", {"command": "ls -la"})
        below = ToolCard("b", "bash", {"command": "pwd"})
        view.append_block(card)
        view.append_block(below)
        card.mark_done("total 8\nfile-a\nfile-b")
        below.mark_done("/tmp")

        assert below.has_class(GAP_CLASS)
        card.toggle_expanded()
        assert below.has_class(GAP_CLASS)
        card.toggle_expanded()
        assert below.has_class(GAP_CLASS)


@pytest.mark.asyncio
async def test_removing_a_block_clears_the_gap_it_justified() -> None:
    """The block promoted to the top must not keep a gap decided against a
    block that is no longer there.

    The live caller is ``app.py`` retiring the streaming working block; the
    D9 boot hint this test was originally written for no longer exists (the
    welcome splash subsumed it), so it is named for the invariant instead.
    """
    app = _Harness()
    async with app.run_test():
        view = app.query_one(TranscriptView)

        hint = NoticeBlock("type a message to begin", "info")
        prompt = UserBlock("hello")
        view.append_block(hint)
        view.append_block(prompt)
        assert prompt.has_class(GAP_CLASS)

        view.remove_block(hint)
        assert not prompt.has_class(GAP_CLASS)


@pytest.mark.asyncio
async def test_a_user_prompt_always_opens_a_turn_with_air() -> None:
    app = _Harness()
    async with app.run_test():
        view = app.query_one(TranscriptView)

        view.append_block(UserBlock("first"))
        second = UserBlock("second")
        view.append_block(second)
        assert second.has_class(GAP_CLASS)
