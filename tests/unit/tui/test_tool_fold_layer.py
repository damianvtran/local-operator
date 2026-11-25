"""Slice 4 of the multi-level fold: the bars, and the layer that owns them.

``tests/unit/tui/test_tool_outline.py`` pins the pure tree and
``test_transcript_display_mode.py`` pins the stored token. This file covers the
seam between them: the bars a mounted ``TranscriptView`` grows, which rows they
hide, what they say, how a mode switch and a live append leave them alone, and
the two rules that outrank every default (a waiting call is never hidden; the
keyboard never lands on an invisible row).

Real app, real stylesheet, real widgets — a bar's row is a width and an ink
question, and a construction-only assertion could agree with the code while the
painted row disagrees.
"""

from __future__ import annotations

import re
import time

import pytest

from local_operator.tui.tool_fold import TURN_BAR_MIN_ROWS, ToolFoldLayer
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.tool_card import DURATION_COL, ToolCard, row_indent
from local_operator.tui.widgets.transcript import (
    FoldBarBlock,
    NoticeBlock,
    PeerMessageBlock,
    QuietGroupBlock,
    TranscriptView,
    UserBlock,
    WorkingBlock,
)
from tests.unit.tui.conftest import StyledTranscriptApp

SENDER = {"pid": 7, "conversation_name": "peer-a"}

#: Five DIFFERENT tools, so a test that only means "three calls" never grows a
#: same-tool sub-group by accident (four consecutive calls of one tool are a
#: level of their own, pinned separately below).
TOOLS = ("read", "browser", "glob", "grep", "edit")


def _bars(view: TranscriptView) -> list[FoldBarBlock]:
    """The fold bars currently mounted, in mount order."""
    return [child for child in view.children if isinstance(child, FoldBarBlock)]


def _user(view: TranscriptView, anchor: str, text: str = "do it") -> UserBlock:
    block = UserBlock(text)
    # A stable anchor: the bar's key is ``tf:turn:<user row id>``, so a test that
    # compares keys across two views needs the ids to agree without sharing
    # widget identity.
    block.navigation_anchor_id = anchor
    view.append_block(block)
    return block


def _tool(
    view: TranscriptView,
    call_id: str,
    name: str = "bash",
    *,
    seconds: float = 1.0,
    failed: bool = False,
) -> ToolCard:
    card = ToolCard(call_id, name, {"command": f"ls {call_id}"})
    view.append_block(card)
    if failed:
        card.restore(state="error", error="boom", duration_s=seconds)
    else:
        card.restore(state="success", result_text="ok", duration_s=seconds)
    return card


def _running(
    view: TranscriptView, call_id: str, name: str = "bash", *, age: float = 7.0
) -> ToolCard:
    card = ToolCard(call_id, name, {"command": f"ls {call_id}"})
    view.append_block(card)
    # The card's own clock is what the bar reads; seed it the way a call that
    # started seven seconds ago would read.
    card._started = time.monotonic() - age
    return card


def _answer(view: TranscriptView, text: str = "done") -> AssistantBlock:
    block = AssistantBlock()
    view.append_block(block)
    block.update_text(text)
    block.finalize_text()
    return block


def _narration(view: TranscriptView, text: str = "checking") -> AssistantBlock:
    block = AssistantBlock()
    block.mark_narration()
    view.append_block(block)
    block.update_text(text)
    block.finalize_text()
    return block


async def _settle(pilot, turns: int = 4) -> None:
    """Pump enough frames for a mount and its deferred re-derive to land."""
    for _ in range(turns):
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_settled_turn_folds_under_one_turn_bar() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        user = _user(view, "anchor:u1")
        cards = [_tool(view, f"t{index}") for index in range(TURN_BAR_MIN_ROWS)]
        answer = _answer(view)
        await _settle(pilot)

        (bar,) = _bars(view)
        assert bar.kind == "turn"
        assert bar.key == "tf:turn:anchor:u1:0"
        assert not bar.expanded, "a settled turn starts folded"
        assert user.display and answer.display, "the prompt and the answer stay on screen"
        assert [card.display for card in cards] == [False, False, False]
        assert "3 actions" in bar._build_row(120).plain

        # One row in both states: opening is the members' reveal, not this
        # row's growth.
        closed_rows = bar._row_count
        bar.activate()
        await _settle(pilot)
        assert all(card.display for card in cards)
        assert bar._row_count == closed_rows


@pytest.mark.asyncio
async def test_by_response_draws_runs_and_no_turn_bar() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        cards = [_tool(view, f"t{index}") for index in range(3)]
        _answer(view)
        await _settle(pilot)

        (bar,) = _bars(view)
        assert bar.kind == "run"
        assert bar.key == "tf:run:tool:t0:0"
        assert [card.display for card in cards] == [False, False, False]


@pytest.mark.asyncio
async def test_two_runs_under_one_turn_bar_keep_a_bar_each() -> None:
    """A turn is not flat when narration splits it, so both levels exist."""
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        for index in range(3):
            _tool(view, f"a{index}")
        _narration(view)
        for index in range(3):
            _tool(view, f"b{index}")
        _answer(view)
        await _settle(pilot)

        bars = _bars(view)
        assert [(bar.kind, bar.key) for bar in bars] == [
            ("turn", "tf:turn:anchor:u1:0"),
            ("run", "tf:run:tool:a0:0"),
            ("run", "tf:run:tool:b0:0"),
        ]
        assert not any(bar.display for bar in bars[1:]), "closed turn hides its run bars too"

        bars[0].activate()
        await _settle(pilot)
        assert all(bar.display for bar in bars), "opening the turn reveals the run bars"


@pytest.mark.asyncio
async def test_a_mode_switch_keeps_cards_mounted_and_the_open_choice() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        cards = [_tool(view, f"t{index}") for index in range(3)]
        _answer(view)
        await _settle(pilot)
        (run_bar,) = _bars(view)
        run_bar.activate()
        await _settle(pilot)
        assert run_bar.expanded and all(card.display for card in cards)
        widget_ids = [card.id for card in cards]

        run_bar.focus()
        await _settle(pilot)
        assert app.focused is run_bar

        view.set_display_mode("by-turn")
        await _settle(pilot)
        (turn_bar,) = _bars(view)
        assert turn_bar.kind == "turn"
        assert [card.id for card in cards] == widget_ids, "no ToolCard is remounted"
        assert app.focused is turn_bar, "the switch lands focus on the nearest bar"

        view.set_display_mode("by-response")
        await _settle(pilot)
        (restored,) = _bars(view)
        assert restored.key == "tf:run:tool:t0:0"
        assert restored.expanded, "the reader's choice survived the switch"
        assert all(card.display for card in cards)


@pytest.mark.asyncio
async def test_the_live_tail_run_grows_open_without_taking_focus() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        cards = [_tool(view, f"t{index}", TOOLS[index]) for index in range(3)]
        tail = WorkingBlock("thinking", "thinking", clock=False)
        view.pin_tail(tail)
        await _settle(pilot)

        (bar,) = _bars(view)
        key = bar.key
        assert bar.expanded, "the run a live turn is still growing stays open"
        assert all(card.display for card in cards)
        cards[0].focus()
        await _settle(pilot)
        assert app.focused is cards[0]

        late = [_tool(view, f"t{index}", TOOLS[index]) for index in (3, 4)]
        await _settle(pilot)
        (grown,) = _bars(view)
        assert grown is bar and grown.key == key, "growth never re-keys the bar"
        assert grown.expanded and all(card.display for card in cards + late)
        assert app.focused is cards[0], "an append never takes focus"


@pytest.mark.asyncio
async def test_collapsing_over_the_focused_row_moves_focus_to_the_bar() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        cards = [_tool(view, f"t{index}") for index in range(3)]
        _answer(view)
        await _settle(pilot)
        (bar,) = _bars(view)
        bar.activate()
        await _settle(pilot)
        cards[1].focus()
        await _settle(pilot)
        assert app.focused is cards[1]

        bar.activate()
        await _settle(pilot)
        assert app.focused is bar, "focus moves to the bar BEFORE the row goes"
        assert not cards[1].display


@pytest.mark.asyncio
async def test_a_waiting_call_splits_the_run_and_is_never_hidden() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        for index in range(3):
            _tool(view, f"a{index}", TOOLS[index])
        waiting = ToolCard("gate", "bash", {"command": "rm -rf build"})
        view.append_block(waiting)
        waiting.mark_waiting()
        for index in range(3):
            _tool(view, f"b{index}", TOOLS[index])
        await _settle(pilot)

        bars = _bars(view)
        assert [bar.kind for bar in bars] == ["run", "run"], "a waiting call splits the run"
        assert waiting.display, "a call awaiting the reader's approval is never hidden"


@pytest.mark.asyncio
async def test_a_dead_turn_pins_only_its_last_failure() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        cards = [
            _tool(view, "d0"),
            _tool(view, "d1", failed=True),
            _tool(view, "d2", failed=True),
        ]
        await _settle(pilot)

        (bar,) = _bars(view)
        assert bar.kind == "run"
        assert [card.display for card in cards] == [False, False, True]
        row = bar._build_row(120).plain
        assert "2 failed" in row
        # The pin is about REACHING the failure; the danger tally is about the
        # harness having said the turn failed (L6). A stop with no failure
        # notice keeps the tally muted, so the test states both halves.
        assert not bar._noanswer, "a stop with no failure notice is not an alarm"

        view.append_block(NoticeBlock("turn failed: boom", "error"))
        await _settle(pilot)
        alarm = next(bar for bar in _bars(view) if bar.kind == "run")
        assert alarm._noanswer, "a real failure notice is the alarm"


@pytest.mark.asyncio
async def test_a_receipt_group_keeps_its_tools_and_gets_no_second_bar() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        view.append_block(PeerMessageBlock("one", SENDER))
        cards = [_tool(view, f"t{index}") for index in range(3)]
        view.append_block(PeerMessageBlock("two", SENDER))
        await _settle(pilot)

        bars = [block for block in view.blocks() if isinstance(block, QuietGroupBlock)]
        assert len(bars) == 1, "the receipt fold still owns its span"
        assert _bars(view) == [], "the tool fold does not bar a receipt's tools"
        assert cards[0] in bars[0].members()

        bars[0].activate()
        await _settle(pilot)
        assert all(card.display for card in cards), "open, they are plain rows as before"


@pytest.mark.asyncio
async def test_bar_keys_hold_still_while_the_tail_grows() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        for index in range(3):
            _tool(view, f"t{index}", TOOLS[index])
        await _settle(pilot)
        (bar,) = _bars(view)
        key = bar.key
        for index in (3, 4):
            _tool(view, f"t{index}", TOOLS[index])
            await _settle(pilot)
            (again,) = _bars(view)
            assert again.key == key
            assert f"{index + 1} actions" in again._build_row(120).plain


@pytest.mark.asyncio
async def test_the_summary_names_actions_kinds_failures_and_a_right_aligned_clock() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        _tool(view, "s0", seconds=1.0)
        _tool(view, "s1", seconds=2.0, failed=True)
        _tool(view, "s2", seconds=3.0)
        await _settle(pilot)

        (bar,) = _bars(view)
        row = bar._build_row(120).plain
        assert "3 actions" in row
        assert "shell" in row
        assert "1 failed" in row
        # The clock takes the LAST cells of the row, right-aligned to the same
        # column a tool card's duration occupies — and in the tool row's own
        # grammar (`format_clock`): six seconds reads `6.0s`, not `6s`.
        assert row.endswith("6.0s".rjust(DURATION_COL)), row


@pytest.mark.asyncio
async def test_a_same_tool_sub_group_gets_its_own_bar() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        cards = [_tool(view, f"b{index}", "browser", seconds=2.0) for index in range(5)]
        await _settle(pilot)

        bars = _bars(view)
        assert [(bar.kind, bar.key) for bar in bars] == [
            ("run", "tf:run:tool:b0:0"),
            ("sub", "tf:sub:tool:b0:0"),
        ]
        run_bar, sub_bar = bars
        assert "5 actions · browser" in run_bar._build_row(120).plain
        sub_row = sub_bar._build_row(120).plain
        assert "browser" in sub_row and "×5" in sub_row
        assert not any(card.display for card in cards), "run closed hides the sub bar too"

        run_bar.activate()
        await _settle(pilot)
        assert sub_bar.display, "opening the run reveals its sub-group bar"
        sub_bar.activate()
        await _settle(pilot)
        assert all(card.display for card in cards)


@pytest.mark.asyncio
async def test_the_live_run_bar_names_the_call_in_flight() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        _tool(view, "t0")
        _tool(view, "t1")
        _running(view, "t2", age=7.0)
        view.pin_tail(WorkingBlock("thinking", "thinking", clock=False))
        await _settle(pilot)

        (bar,) = _bars(view)
        row = bar._build_row(120).plain
        assert "◐ bash" in row
        # The running call's own clock, in the tool row's grammar: sub-10 s
        # keeps one decimal (`7.3s`), so the assertion holds against the
        # fraction of a second the settle pumped past 7.0.
        assert re.search(r"◐ bash ls t2 7\.\ds$", row), row


@pytest.mark.asyncio
async def test_right_opens_and_steps_in_and_left_closes_or_steps_out() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        cards = [_tool(view, f"b{index}", "browser") for index in range(5)]
        await _settle(pilot)
        run_bar, sub_bar = _bars(view)

        run_bar.focus()
        await _settle(pilot)
        await pilot.press("right")
        await _settle(pilot)
        assert run_bar.expanded
        await pilot.press("right")
        await _settle(pilot)
        assert app.focused is sub_bar, "right on an open bar steps into its first child"

        await pilot.press("left")
        await _settle(pilot)
        assert app.focused is run_bar, "left on a closed bar steps out to its parent"

        await pilot.press("left")
        await _settle(pilot)
        assert not run_bar.expanded
        assert not any(card.display for card in cards)


@pytest.mark.asyncio
async def test_a_batch_append_derives_the_same_bars_as_the_live_stream() -> None:
    """Replay parity: the tree comes from the ROWS, not the event stream."""
    live_keys: list[str] = []
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        for index in range(3):
            _tool(view, f"t{index}")
        _answer(view)
        await _settle(pilot)
        live_keys = [bar.key for bar in _bars(view)]

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        with view.batch_append():
            _user(view, "anchor:u1")
            for index in range(3):
                _tool(view, f"t{index}")
            _answer(view)
        await _settle(pilot, turns=8)
        assert [bar.key for bar in _bars(view)] == live_keys


@pytest.mark.asyncio
async def test_the_layer_is_the_views_own_visibility_authority() -> None:
    """Every hidden row says so through ``fold_vetoes`` and ``sync_display``."""
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        assert isinstance(view._tool_fold, ToolFoldLayer)
        _user(view, "anchor:u1")
        cards = [_tool(view, f"t{index}") for index in range(3)]
        await _settle(pilot)
        (bar,) = _bars(view)
        veto = f"tool-fold:{bar.key}"
        assert all(veto in card.fold_vetoes for card in cards)
        bar.activate()
        await _settle(pilot)
        assert all(veto not in card.fold_vetoes for card in cards)


def test_the_disclosure_glyphs_are_single_cell() -> None:
    """The mark spends the ledger's ICON cell, so it has to measure one.

    ``glyphs._single_cell`` is the gate (a two-cell mark would shift the name
    column and the right-aligned clock by a cell); this pins the two constants
    the fold actually paints.
    """
    from rich.cells import cell_len

    from local_operator.tui.glyphs import FOLD_CLOSED_GLYPH, FOLD_OPEN_GLYPH

    assert cell_len(FOLD_CLOSED_GLYPH) == 1
    assert cell_len(FOLD_OPEN_GLYPH) == 1
    assert FOLD_CLOSED_GLYPH != FOLD_OPEN_GLYPH, "open and closed must differ in a colourless frame"


@pytest.mark.asyncio
async def test_the_bar_offers_expand_and_collapse_only_while_it_has_focus() -> None:
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        for index in range(3):
            _tool(view, f"t{index}", TOOLS[index])
        await _settle(pilot)
        (bar,) = _bars(view)

        assert "expand" not in bar._build_row(120).plain
        bar.focus()
        await _settle(pilot)
        assert "expand" in bar._build_row(120).plain
        bar.activate()
        await _settle(pilot)
        assert "collapse" in bar._build_row(120).plain


@pytest.mark.asyncio
async def test_a_wide_script_bar_drops_kinds_rather_than_the_failure_tally() -> None:
    """The bar's budget is measured in CELLS, so the kinds go before the tally.

    The wide text here is a tool's own name: a name is not guaranteed ASCII
    (MCP servers name their own tools), and the summary clause is built from it
    verbatim. The arithmetic is the one the zh-CN count clause will meet the day
    it is longer than English's — ``_cells`` summed ``len``, so a bar believed
    the wide kinds cost half of what they paint and left the row to be cut by
    the ellipsis, with the failure tally at the tail where the cut lands. A
    bar's degradation order is kinds first, the tally never.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(60, 30)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        for index, name in enumerate(("读文件", "搜索", "网页")):
            card = _tool(view, f"c{index}", name)
            if index == 2:
                card.restore(state="error", error="boom", duration_s=1.0)
        await _settle(pilot)

        (bar,) = _bars(view)
        wide = bar._build_row(60).plain
        assert "…" not in wide, "the bar ellipsized its summary instead of shedding kinds"
        assert "读文件, 搜索, 网页" in wide, wide
        assert "✗ 1 failed" in wide, "the failure tally is never the clause to drop"

        # Narrower than the kinds fit: they go whole (never the tally), and the
        # row is not left to the ellipsis. 44 is the measured rung where the
        # three wide names no longer fit and the tally still does.
        narrow = bar._build_row(44).plain
        assert "…" not in narrow, narrow
        assert "读文件" not in narrow, narrow
        assert "✗ 1 failed" in narrow, "the failure tally is never the clause to drop"


@pytest.mark.asyncio
async def test_a_plain_tool_row_sits_in_the_bar_column_in_by_turn() -> None:
    """One tool-row column per frame: a turn too small for a bar is a step too.

    The rendered frame that found this painted an unfolded call at col 3 of a
    100-column frame while a row revealed under a bar sat at col 6 and the bar
    itself at col 4 — three insets for one kind of row, in one mode. A tool row
    belongs in the bar's column whether or not its turn grew a bar; a covered
    row is one level below it; a turn with no call keeps the prose column.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        small = [_tool(view, f"s{index}") for index in range(2)]  # under the bar minimum
        _answer(view)
        _user(view, "anchor:u2")
        big = [_tool(view, f"b{index}") for index in range(TURN_BAR_MIN_ROWS)]
        _answer(view)
        await _settle(pilot)

        (bar,) = _bars(view)
        assert bar.kind == "turn"
        lead = lambda row: len(row) - len(row.lstrip(" "))  # noqa: E731
        assert [card.fold_depth for card in small] == [1, 1]
        assert [lead(card._build_row(120).plain) for card in small] == [row_indent(120, 1)] * 2
        assert lead(bar._build_row(120).plain) == row_indent(120, 1), "bar and row share a column"

        # A covered row is one level below the bar it sits under.
        assert [card.fold_depth for card in big] == [2, 2, 2]
        bar.activate()
        await _settle(pilot)
        assert [lead(card._build_row(120).plain) for card in big] == [row_indent(120, 2)] * 3


# ---------------------------------------------------------------------------
# Remediation round 1 (L2-L16): the rules the review found the layer breaking.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rows_after_the_answer_are_outside_the_turn_bar() -> None:
    """L2: a receipt or notice that arrives after the answer stays visible.

    The turn bar condenses what the agent did BEFORE the answer. Rows that land
    after it are the idling app's own receipts; nothing about the answer hides
    them, and opening the bar must not reveal a second bar over the same calls.
    """
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        cards = [_tool(view, f"t{index}", TOOLS[index]) for index in range(3)]
        _answer(view)
        await _settle(pilot)
        assert [bar.kind for bar in _bars(view)] == ["turn"]

        notice = NoticeBlock("after the answer", "info")
        view.append_block(notice)
        view.append_block(PeerMessageBlock("hello", SENDER))
        await _settle(pilot)

        assert notice.display, "a post-answer notice was hidden under the turn bar"
        assert view.blocks()[-1].display, "a post-answer peer message was hidden"
        assert [bar.kind for bar in _bars(view)] == [
            "turn"
        ], "opening the turn would show a duplicate run bar"
        assert not any(card.display for card in cards)


@pytest.mark.asyncio
async def test_an_error_notice_inside_the_turn_is_never_covered() -> None:
    """L2: an error notice mid-turn is a row the reader must see."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        cards = [_tool(view, f"t{index}", TOOLS[index]) for index in range(3)]
        notice = NoticeBlock("provider error: rate limited", "error")
        view.append_block(notice)
        _answer(view)
        await _settle(pilot)

        bars = _bars(view)
        assert bars and bars[0].kind == "turn"
        assert notice.display, "the turn bar hid an error notice"
        assert not any(card.display for card in cards)
        assert not bars[0]._noanswer, "a recovered error is not a failed turn"


@pytest.mark.asyncio
async def test_an_opened_fold_ends_every_row_on_the_clock_column() -> None:
    """L5: the status column and the bar's clock are one column at one level.

    A summary the row truncates used to ignore the depth the row was drawn at,
    so an open fold's rows ended at two different cells; the bar's own clock sat
    two cells right of the rows it covered (L5/D3+D4).
    """
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        cards = [
            ToolCard("long", "bash", {"command": "cat " + "a-long-path/" * 12}),
            ToolCard("short", "bash", {"command": "ls"}),
            _tool(view, "failed", "bash", failed=True),
        ]
        for card in cards[:2]:
            view.append_block(card)
            card.restore(state="success", result_text="ok", duration_s=1.0)
        _answer(view)
        await _settle(pilot)
        (bar,) = _bars(view)
        bar.activate()
        await _settle(pilot)

        ends = {len(card._build_row(120).plain) for card in cards}
        assert len(ends) == 1, f"the status column is jagged: {ends}"
        assert ends == {
            len(bar._build_row(120).plain)
        }, "the bar's clock is not in the column of the rows it covers"


@pytest.mark.asyncio
async def test_a_focused_bar_carries_the_focus_tint() -> None:
    """L4: a keyboard user must be able to see which bar has focus."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        first = [_tool(view, f"a{index}", TOOLS[index]) for index in range(3)]
        _answer(view)
        _user(view, "anchor:u2")
        [_tool(view, f"b{index}", TOOLS[index]) for index in range(3)]
        _answer(view)
        await _settle(pilot)
        turn_bars = [bar for bar in _bars(view) if bar.kind == "turn"]
        assert len(turn_bars) == 2
        focused, other = turn_bars
        focused.focus()
        await _settle(pilot)

        assert focused.has_focus
        assert str(focused.styles.background) != str(
            other.styles.background
        ), "a focused fold bar paints on the same ground as an unfocused one"
        assert first  # seeded rows above are the fold's own


def test_the_clock_rounds_before_choosing_its_branch() -> None:
    """L13: 9.96s is `10s`, and 59.6s is `1m`, never a wider `10.0s`/`60s`."""
    from local_operator.tui.widgets.tool_card import format_clock

    assert format_clock(0.04) == "<0.1s"
    assert format_clock(0.05) == "0.1s"
    assert format_clock(9.94) == "9.9s"
    assert format_clock(9.96) == "10s"
    assert format_clock(12.95) == "13s"
    assert format_clock(59.6) == "1m"
    assert format_clock(60.0) == "1m"


@pytest.mark.asyncio
async def test_a_run_bar_drops_the_work_name_in_by_response() -> None:
    """L15: layout B has no label; By turn's ledger still names the run."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        for index in range(3):
            _tool(view, f"t{index}", TOOLS[index])
        _narration(view)
        for index in range(3):
            _tool(view, f"u{index}", TOOLS[index])
        _answer(view)
        await _settle(pilot)
        (turn_bar,) = [bar for bar in _bars(view) if bar.kind == "turn"]
        assert "turn" in turn_bar._build_row(120).plain
        first_run = next(bar for bar in _bars(view) if bar.kind == "run")
        assert "work" in first_run._build_row(120).plain

        view.set_display_mode("by-response")
        await _settle(pilot)
        (run_bar,) = [bar for bar in _bars(view) if bar.kind == "run"][:1]
        row = run_bar._build_row(120).plain
        assert "work" not in row, row
        assert row.startswith("  ▸"), row


@pytest.mark.asyncio
async def test_bar_kinds_fold_mcp_names_and_speak_the_catalogue_nouns() -> None:
    """L16: `mcp__linear_create_issue` reads as the call, `bash` as `shell`."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        for index, name in enumerate(("mcp__linear_create_issue", "bash", "task")):
            _tool(view, f"t{index}", name)
        await _settle(pilot)

        (bar,) = _bars(view)
        row = bar._build_row(120).plain
        assert "mcp__" not in row, row
        assert "create_issue" in row, row
        assert "shell" in row, row
        assert "subagents" in row, row


@pytest.mark.asyncio
async def test_a_live_turn_mounts_its_turn_bar_before_its_first_run() -> None:
    """L8: paint order is the derived order, live and replayed alike.

    A live tool-first turn grows its run bars before the answer exists; the turn
    bar is created when the answer lands and used to mount AFTER them, so the
    child sat above its parent and Up from the turn bar landed on it.
    """
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        first = [_tool(view, f"t{index}", TOOLS[index]) for index in range(3)]
        await _settle(pilot)
        _narration(view)
        second = [_tool(view, f"u{index}", TOOLS[index]) for index in range(3)]
        _answer(view)
        await _settle(pilot)

        kinds = [bar.kind for bar in _bars(view)]
        assert kinds == ["turn", "run", "run"], kinds
        turn_bar = _bars(view)[0]
        run_bar = _bars(view)[1]
        assert view.children.index(turn_bar) < view.children.index(run_bar)

        turn_bar.focus()
        await _settle(pilot)
        await pilot.press("up")
        await _settle(pilot)
        assert app.focused is not run_bar, "Up from the turn bar landed on its child"
        assert app.focused not in first + second


@pytest.mark.asyncio
async def test_a_live_call_awaiting_approval_is_not_hidden_by_a_collapse() -> None:
    """L9: the live card is `running`; the session's pending ids are the truth."""
    from types import SimpleNamespace

    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        # The layer reads the APP's session for the pending ids; this fixture
        # mounts only a transcript, so the fake is attached by name.
        setattr(app, "_session", SimpleNamespace(pending_display_tool_ids=lambda: {"gated"}))
        _user(view, "anchor:u1")
        before = [_running(view, f"r{index}", "read") for index in range(3)]
        gated = _running(view, "gated", "write")
        after = [_running(view, f"s{index}", "read") for index in range(3)]
        await _settle(pilot)
        run_bar = next(bar for bar in _bars(view) if bar.kind == "run")
        run_bar.activate()
        await _settle(pilot)
        assert run_bar.expanded, "premise: the run is open"

        run_bar.focus()
        await _settle(pilot)
        await pilot.press("left")
        await _settle(pilot)

        assert gated.display, "collapsing the run hid the call awaiting approval"
        assert not any(card.display for card in before + after)


@pytest.mark.asyncio
async def test_a_derive_failure_keeps_the_previous_frame(caplog) -> None:
    """L12: a layer bug is logged and degrades to the last frame, never the pump."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        for index in range(3):
            _tool(view, f"t{index}", TOOLS[index])
        _answer(view)
        await _settle(pilot)
        layer = view._tool_fold
        before = [(bar.kind, bar.key) for bar in _bars(view)]
        original = layer._derive_now

        def explode() -> None:
            raise RuntimeError("derive bug")

        layer._derive_now = explode  # type: ignore[method-assign]
        try:
            layer.invalidate()
            await _settle(pilot)
        finally:
            layer._derive_now = original  # type: ignore[method-assign]

        assert [(bar.kind, bar.key) for bar in _bars(view)] == before
        assert app.is_running, "the derive exception reached the app"
        assert any("tool fold derive failed" in record.message for record in caplog.records)
