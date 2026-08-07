"""Adaptive spacing — the transcript's vertical rhythm.

The rule replaced a flat "everything is flush" layout. Flush is right for a
run of one-line tool traces (a ledger should read as a ledger) and wrong for
prose, which then runs into whatever follows it. So the gap is decided per
block from what precedes it, and the decision is a pure function
(:func:`needs_gap_above`) tested here in isolation, plus an integration pass
through a live ``TranscriptView`` that checks the class actually lands.
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

    def __init__(self, kind: str, rows: int = 1, *, lead: bool = False, transient: bool = False):
        super().__init__()
        setattr(self, "SPACING_KIND", kind)
        setattr(self, "SPACING_LEAD", lead)
        setattr(self, "SPACING_TRANSIENT", transient)
        self._rows = rows

    def spans_multiple_rows(self) -> bool:
        return self._rows > 1


# --- the rule --------------------------------------------------------------


def test_the_first_block_never_takes_a_gap() -> None:
    """Content meets the top edge; a leading blank row is wasted screen."""
    assert needs_gap_above(None, _Stub("tool")) is False
    assert needs_gap_above(None, _Stub("user", lead=True)) is False


def test_consecutive_one_line_tool_rows_stay_flush() -> None:
    """The whole point: a batch of tool calls reads as one dense ledger."""
    previous = _Stub("tool", rows=1)
    assert needs_gap_above(previous, _Stub("tool", rows=1)) is False


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


def test_consecutive_notices_stay_flush() -> None:
    """Same kind, one row each: a list of notices is a list, not paragraphs."""
    assert needs_gap_above(_Stub("notice"), _Stub("notice")) is False


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
    """The regression this exists for: cards that settle BEFORE their first
    layout used to resolve to two rows each and stay there, double-spacing
    the whole ledger at some widths and not others.

    Settling without an intervening pause is the realistic path — the engine
    emits tool start and tool end in one synchronous burst — so the test
    reproduces exactly that ordering rather than a convenient one.
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
        assert not any(card.has_class(GAP_CLASS) for card in cards)


@pytest.mark.asyncio
async def test_a_real_mouse_click_expands_and_collapses_under_the_real_sheet() -> None:
    """The pinned collapsed height must not defeat the expansion."""
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

        await pilot.click(card)
        await pilot.pause()
        assert card.expanded is True
        assert card.size.height == 4
        assert below.has_class(GAP_CLASS)

        await pilot.click(card)
        await pilot.pause()
        assert card.expanded is False
        assert card.size.height == 1
        assert not below.has_class(GAP_CLASS)


@pytest.mark.asyncio
async def test_container_applies_the_gap_class_only_where_the_rule_says() -> None:
    app = _Harness()
    async with app.run_test():
        view = app.query_one(TranscriptView)

        first = ToolCard("a", "read", {"path": "one.py"})
        second = ToolCard("b", "read", {"path": "two.py"})
        prose = AssistantBlock()
        view.append_block(first)
        view.append_block(second)
        view.append_block(prose)

        assert not first.has_class(GAP_CLASS)  # nothing above it
        assert not second.has_class(GAP_CLASS)  # flush against a one-line row
        assert prose.has_class(GAP_CLASS)  # different kind


@pytest.mark.asyncio
async def test_the_working_line_does_not_disturb_the_rows_around_it() -> None:
    app = _Harness()
    async with app.run_test():
        view = app.query_one(TranscriptView)

        first = ToolCard("a", "read", {"path": "one.py"})
        working = WorkingBlock()
        second = ToolCard("b", "read", {"path": "two.py"})
        view.append_block(first)
        view.append_block(working)
        view.append_block(second)

        assert not working.has_class(GAP_CLASS)
        # The card after the working line is spaced against the CARD above
        # it, not against the transient line between them.
        assert not second.has_class(GAP_CLASS)


@pytest.mark.asyncio
async def test_expanding_a_card_gives_the_block_below_it_room() -> None:
    """The gap is re-decided when a card grows, and given back when it shrinks."""
    app = _Harness()
    async with app.run_test():
        view = app.query_one(TranscriptView)

        card = ToolCard("a", "bash", {"command": "ls -la"})
        below = ToolCard("b", "bash", {"command": "pwd"})
        view.append_block(card)
        view.append_block(below)
        card.mark_done("total 8\nfile-a\nfile-b")
        below.mark_done("/tmp")

        assert not below.has_class(GAP_CLASS)
        card.toggle_expanded()
        assert below.has_class(GAP_CLASS)
        card.toggle_expanded()
        assert not below.has_class(GAP_CLASS)


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
