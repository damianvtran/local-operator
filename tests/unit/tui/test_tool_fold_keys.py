"""Slice 5 of the transcript layout switch: keys, indent, ink and the foot.

Slice 4 pinned the bars and what they hide. This file pins the things a reader
then does with them, and the two pieces of geometry the prototype fixed:

* ``f``/``F`` walk the failed calls and open what hides them; ``+``/``-`` open a
  whole turn or put it back to its defaults;
* the PRINTABLE-KEY boundary — a fold bar claims ``f F + -``, a ToolCard still
  hands every printable to the composer (the defect a base-class binding would
  have introduced);
* the two-column inset grid (steps two cells deeper than prose, revealed rows
  two deeper again) and the copy gutter that must agree with it at depth > 0;
* By response's compact grammar: no blank rows inside a step, remarks in the
  secondary ink, and the one-line foot that closes a settled turn.

Real app, real sheet, real widgets: every one of these is a question about a
painted frame, not about a constructor.
"""

from __future__ import annotations

import re

import pytest

from local_operator.tui import theme as theme_mod
from local_operator.tui.tool_fold import ToolFoldLayer
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.tool_card import ToolCard, row_indent
from local_operator.tui.widgets.transcript import (
    GAP_CLASS,
    NoticeBlock,
    TranscriptBlock,
    TranscriptView,
    TurnFootBlock,
    WorkingBlock,
)
from tests.unit.tui.test_tool_fold_layer import (
    _answer,
    _bars,
    _narration,
    _running,
    _settle,
    _tool,
    _user,
)

TOOLS = ("read", "bash", "glob", "grep", "edit")


def _plain(block: TranscriptBlock) -> str:
    """The rows a block paints, as text.

    ``renderable`` is Textual's ``RenderableType`` union; every block this file
    reads builds a ``Text``, so the union is narrowed once here instead of at
    each assertion that only wants to know what the row says.
    """
    from rich.text import Text

    rendered = block.renderable
    assert isinstance(rendered, Text), rendered
    return rendered.plain


def _foots(view: TranscriptView) -> list[TurnFootBlock]:
    return [child for child in view.children if isinstance(child, TurnFootBlock)]


def _visible_children(view: TranscriptView) -> list[TranscriptBlock]:
    return [
        child for child in view.children if isinstance(child, TranscriptBlock) and child.display
    ]


def _gap_cells(previous: TranscriptBlock, block: TranscriptBlock) -> int:
    """Painted rows between two visible blocks (0 = flush)."""
    return block.region.y - (previous.region.y + previous.region.height)


def _layer(view: TranscriptView) -> ToolFoldLayer:
    return view._tool_fold

    def compose(self):  # type: ignore[no-untyped-def]
        from textual.containers import Container

        from local_operator.tui.widgets.toast import Toast

        yield TranscriptView()
        with Container(id="toast-host"):
            yield Toast()


@pytest.mark.asyncio
async def test_f_and_F_walk_the_failed_calls_and_open_what_hides_them() -> None:
    """`f` opens the turn and lands on the first failure; `F` walks back."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        cards = [_tool(view, f"t{index}", TOOLS[index]) for index in range(5)]
        cards[1].restore(state="error", error="boom", duration_s=1.0)
        cards[3].restore(state="error", error="boom", duration_s=1.0)
        _answer(view)
        await _settle(pilot)

        (turn_bar,) = _bars(view)
        assert not turn_bar.expanded
        assert not any(card.display for card in cards), "premise: the turn is folded"

        turn_bar.focus()
        await _settle(pilot)
        await pilot.press("f")
        await _settle(pilot)

        assert app.focused is cards[1], "f did not land on the first failed call"
        assert turn_bar.expanded and cards[1].display, "the bars over it did not open"

        # The jump hands the keyboard to the CALL (the design's contract), and
        # a ToolCard deliberately does not bind f/F — it forwards printables to
        # the composer. So the ladder's next step is taken from a bar again,
        # which is how a reader walks a transcript: the card is the destination,
        # Up is the way back to the bar.
        turn_bar.focus()
        await _settle(pilot)
        await pilot.press("f")
        await _settle(pilot)
        assert app.focused is cards[3], "f did not step to the next failure"

        turn_bar.focus()
        await _settle(pilot)
        await pilot.press("F")
        await _settle(pilot)
        assert app.focused is cards[1], "F did not step back"


@pytest.mark.asyncio
async def test_f_says_so_once_when_the_turn_has_no_failure() -> None:
    """A miss is a TOAST, not a transcript row (fix for the row-per-retry)."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        seen: list[str] = []

        def capture_notice(text: str) -> None:
            seen.append(text)

        view.transient_notice = capture_notice  # type: ignore[method-assign]
        _user(view, "anchor:u1")
        [_tool(view, f"t{index}", TOOLS[index]) for index in range(4)]
        _answer(view)
        await _settle(pilot)
        (turn_bar,) = _bars(view)
        turn_bar.focus()
        await _settle(pilot)

        before = len(view.blocks())
        await pilot.press("f")
        await _settle(pilot)

        assert seen == ["No failed calls in this turn"]
        assert len(view.blocks()) == before, "a miss appended a transcript row"


@pytest.mark.asyncio
async def test_the_miss_is_painted_in_the_apps_own_toast_slot() -> None:
    """Q12: the sentence reaches a PAINTED frame, with nothing stubbed.

    The old path called Textual's `App.notify`, whose rack this app does not
    run (`run_test` defaults it off, and the app paints its own `Toast`); the
    message landed in `app._notifications` and no frame ever showed it, which
    is why the stubbed test above could not see the miss. This one goes through
    the real keypress and the real handler, and reads the compositor.
    """
    from local_operator.tui.widgets.toast import Toast
    from tests.unit.tui.conftest import StyledTranscriptApp

    class _ToastTranscriptApp(StyledTranscriptApp):
        """The transcript beside the app's ONE toast slot (`#toast-host`).

        The shipped stylesheet positions the host, so this is the app's real
        arrangement minus everything the fold does not read. The slot mounts
        the way ``OperatorApp.compose`` mounts it: a container on the toast
        layer holding a single, always-present ``Toast``.
        """

        def compose(self):  # type: ignore[no-untyped-def]
            from textual.containers import Container

            yield TranscriptView()
            with Container(id="toast-host"):
                yield Toast()

    app = _ToastTranscriptApp()
    async with app.run_test(size=(100, 30)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        [_tool(view, f"t{index}", TOOLS[index]) for index in range(3)]
        _answer(view)
        await _settle(pilot)
        (turn_bar,) = _bars(view)
        turn_bar.focus()
        await _settle(pilot)

        toast = app.query_one(Toast)
        assert not toast.display, "premise: the slot starts empty"

        await pilot.press("f")
        await _settle(pilot)

        assert toast.display, "the miss never reached the app's toast slot"
        assert toast.message == "No failed calls in this turn"
        painted = [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]
        assert any(
            "No failed calls in this turn" in line for line in painted
        ), f"the toast is not on the frame: {painted}"


@pytest.mark.asyncio
async def test_plus_opens_every_level_of_the_turn_and_minus_restores_defaults() -> None:
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        first = [_tool(view, f"a{index}", TOOLS[index]) for index in range(3)]
        _narration(view)
        second = [_tool(view, f"b{index}", TOOLS[index]) for index in range(3)]
        _answer(view)
        await _settle(pilot)

        turn_bar, run_a, run_b = _bars(view)
        assert [run_a.expanded, run_b.expanded] == [False, False]
        turn_bar.focus()
        await _settle(pilot)

        await pilot.press("plus")
        await _settle(pilot)
        assert turn_bar.expanded and run_a.expanded and run_b.expanded
        assert all(card.display for card in first + second)

        await pilot.press("minus")
        await _settle(pilot)
        assert not turn_bar.expanded, "- did not return the turn to its default"
        assert not any(card.display for card in first + second)
        assert _layer(view)._open == {}, "- left an explicit choice behind"


@pytest.mark.asyncio
async def test_a_tool_card_hands_its_printables_to_the_composer() -> None:
    """THE PRINTABLE-KEY BOUNDARY: only a fold bar claims f/F/+/-.

    The base class forwards unbound printables to the composer so a focused
    receipt never eats a sentence. Binding these four on
    :class:`ExpandableActionBlock` would swallow them on every ToolCard; the
    layer's bindings live on :class:`FoldBarBlock` alone, and this is the cell
    that would catch a move down the hierarchy.
    """
    from textual.app import App, ComposeResult

    from tests.unit.tui.conftest import TCSS_PATH

    class TypingApp(App[None]):
        CSS_PATH = TCSS_PATH

        def get_css_variables(self) -> dict[str, str]:
            variables = super().get_css_variables()
            variables.update(theme_mod.tcss_variable_map())
            return variables

        def compose(self) -> ComposeResult:
            yield TranscriptView()
            yield Editor()

    app = TypingApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        editor = app.query_one(Editor)
        card = ToolCard("t1", "bash", {"command": "ls"})
        view.append_block(card)
        await pilot.pause()
        await pilot.pause()

        card.focus()
        await pilot.pause()
        assert app.focused is card
        await pilot.press("f")
        await pilot.press("F")
        await pilot.press("plus")
        await pilot.press("minus")
        await pilot.pause()

        assert editor.text == "fF+-", f"the composer lost a key: {editor.text!r}"


@pytest.mark.asyncio
async def test_a_fold_bar_claims_those_same_printables() -> None:
    """The other half of the boundary: on a BAR, `f` navigates and does not type."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        seen: list[str] = []

        def capture_notice(text: str) -> None:
            seen.append(text)

        view.transient_notice = capture_notice  # type: ignore[method-assign]
        _user(view, "anchor:u1")
        [_tool(view, f"t{index}", TOOLS[index]) for index in range(4)]
        _answer(view)
        await _settle(pilot)
        (turn_bar,) = _bars(view)
        turn_bar.focus()
        await _settle(pilot)

        await pilot.press("f")
        await _settle(pilot)
        assert seen == ["No failed calls in this turn"], "the bar did not claim f"
        assert turn_bar.has_focus

        await pilot.press("F")
        await _settle(pilot)
        assert len(seen) == 2, "F did not reach the previous-failure action"

        await pilot.press("plus")
        await _settle(pilot)
        assert turn_bar.expanded, "+ did not reach the open-this-turn action"

        await pilot.press("minus")
        await _settle(pilot)
        assert not turn_bar.expanded, "- did not reach the defaults action"


@pytest.mark.asyncio
async def test_escape_never_closes_a_fold() -> None:
    """Esc keeps its app-wide meaning; it is not a fold key."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        [_tool(view, f"t{index}", TOOLS[index]) for index in range(3)]
        _answer(view)
        await _settle(pilot)
        (turn_bar,) = _bars(view)
        turn_bar.focus()
        turn_bar.activate()
        await _settle(pilot)
        assert turn_bar.expanded

        await pilot.press("escape")
        await _settle(pilot)
        assert turn_bar.expanded, "escape closed the fold"


@pytest.mark.asyncio
async def test_tool_rows_and_bars_sit_two_cells_deeper_than_prose() -> None:
    """The grid: bars at 2 cells, revealed rows at 4, a sub's members at 6."""
    from local_operator.tui.glyphs import display_name
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        cards = [_tool(view, f"t{index}", "bash") for index in range(4)]
        _answer(view)
        await _settle(pilot)

        run_bar = next(bar for bar in _bars(view) if bar.kind == "run")
        assert _plain(run_bar).startswith("  "), _plain(run_bar)[:8]
        assert not cards[0].display

        run_bar.activate()  # open the run: the same-tool sub-bar appears
        await _settle(pilot)
        sub_bar = next(bar for bar in _bars(view) if bar.kind == "sub")
        assert _plain(sub_bar).startswith("    "), "a sub-bar nests one step under its run"

        sub_bar.activate()  # open the sub: the calls are revealed one level in
        await _settle(pilot)
        card = cards[0]
        assert card.display
        plain = _plain(card)
        assert plain.startswith("      "), plain[:10]

        gutter = card.copy_gutter(0)
        assert gutter == row_indent(120, 3) + ToolCard.ICON_COLS
        copied = plain[gutter:]
        assert copied.startswith(display_name("bash")), (
            "the copy gutter ate the first character of the tool name: " f"{copied[:12]!r}"
        )


@pytest.mark.asyncio
async def test_a_pinned_failure_takes_its_groups_depth() -> None:
    """D11: the visible failure of a dead turn sits with the rows it belongs to.

    A dead turn pins its LAST failed call below the collapsed fold so the reason
    is reachable without unfolding sixteen calls. When that call is a member of
    a same-tool sub-group it belongs to the group's column; painting it at the
    run's column left it one step out from the three rows directly above it,
    which reads as a different kind of row.
    """
    from tests.unit.tui.conftest import StyledTranscriptApp, open_all_folds

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        cards = [_tool(view, f"t{index}", "bash", failed=index == 4) for index in range(5)]
        await _settle(pilot)
        # No answer: the last settled turn is the one a pin belongs to.
        await open_all_folds(view, pilot)

        pinned, *members = cards[4], *cards[:4]
        assert pinned.display and all(card.display for card in members)
        painted = {len(_plain(card)) - len(_plain(card).lstrip(" ")) for card in cards}
        assert len(painted) == 1, f"the pinned row is out of its group's column: {painted}"


@pytest.mark.parametrize("width", [80, 100])
@pytest.mark.asyncio
async def test_the_duration_column_is_one_column_at_every_depth(width: int) -> None:
    """D12: time is right-aligned in the LEDGER's column, not each row's own.

    A row revealed two levels deep has a two-cell-wider inset than the sibling
    call beside it in the same run; measuring the right edge off its own body
    width walked its clock two cells left of that sibling's, so one column of
    numbers came out in two columns.
    """
    from tests.unit.tui.conftest import StyledTranscriptApp, open_all_folds

    app = StyledTranscriptApp()
    async with app.run_test(size=(width, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        members = [_tool(view, f"b{index}", "bash", seconds=1.0) for index in range(4)]
        sibling = _tool(view, "r0", "read", seconds=1.0)
        await _settle(pilot)
        await open_all_folds(view, pilot)

        member, sibling_row = members[0], sibling
        assert member.display and sibling_row.display, "premise: both rows are painted"
        member_cells = _plain(member)
        sibling_cells = _plain(sibling_row)
        assert "1.0s" in member_cells and "1.0s" in sibling_cells
        assert member_cells.index("1.0s") == sibling_cells.index("1.0s"), (
            "the clock column stepped between depths: "
            f"{member_cells.index('1.0s')} vs {sibling_cells.index('1.0s')} at {width} cols"
        )


@pytest.mark.parametrize("width", [40, 44, 48, 60, 80])
@pytest.mark.asyncio
async def test_the_turn_bar_never_drops_its_failure_tally(width: int) -> None:
    """U10/D9: at every width the tally survives, inside a ONE-ROW bar.

    At 44 columns — the team's own practical minimum — the bar showed
    `19 actions · …`: the kinds clause had been fitted into the budget and cut
    at the ellipsis, which put the one clause the design never drops behind the
    cut. The ladder is: the bar's name goes, then the kinds, then the tally
    compacts to `✗3`; the ellipsis is never what states a failure.
    """
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(width, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        [_tool(view, f"t{index}", TOOLS[index], failed=index < 3) for index in range(len(TOOLS))]
        _answer(view)
        await _settle(pilot)

        (turn_bar,) = [bar for bar in _bars(view) if bar.kind == "turn"]
        row = _plain(turn_bar)
        assert "✗" in row, f"the failure tally is gone at {width} cols: {row!r}"
        assert "…" not in row, f"an ellipsis stands where a clause should be: {row!r}"
        assert "\n" not in row, row
        assert turn_bar.region.height == 1, f"the bar wrapped at {width} cols"


@pytest.mark.parametrize("width", [40, 44, 60, 80])
@pytest.mark.asyncio
async def test_a_live_bar_keeps_its_tally_and_stays_one_row(width: int) -> None:
    """The live reading is clamped, not paid for with the failure tally.

    At 40 columns the live bar wrapped to two rows and dropped the count
    entirely (U10's secondary arm). The elapsed reading is the cell that may
    shorten; the tally is the one that may not.
    """
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(width, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        for index in range(2):
            _tool(view, f"t{index}", TOOLS[index], failed=index == 1)
        running = _running(view, "t2", "bash", age=7.0)
        await _settle(pilot)

        (bar,) = _bars(view)
        assert bar._live_card is running, "premise: the bar reports the running call"
        row = _plain(bar)
        assert "✗" in row, f"the failure tally is gone at {width} cols: {row!r}"
        assert "\n" not in row, row
        assert bar.region.height == 1, f"the live bar wrapped at {width} cols"


@pytest.mark.asyncio
async def test_the_live_bar_repaints_its_elapsed_without_a_derive(monkeypatch) -> None:
    """R2-4: the bar's 1 Hz clock is armed at mount, not left to the next derive.

    The layer restates a new bar and THEN mounts it, so the restate's own sync
    saw an unmounted widget and started nothing — and Textual marks a widget
    mounted after its Mount event, so `on_mount`'s sync saw the same thing. The
    bar then showed the elapsed it mounted with (7.0s) while the tool row under
    it advanced; nothing else repaints the reading, because every other path
    needs an append. This moves the card's start with NO append and reads the
    painted frame: only the bar's own timer can have repainted it.
    """
    from local_operator.tui.widgets import transcript as transcript_mod
    from tests.unit.tui.conftest import StyledTranscriptApp

    monkeypatch.setattr(transcript_mod, "LIVE_CLOCK_S", 0.05)
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 30)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        _tool(view, "t0", "read")
        _tool(view, "t1", "bash")
        card = _running(view, "t2", "bash", age=7.0)
        await _settle(pilot)

        (bar,) = _bars(view)
        assert bar._live_card is card, "premise: the bar reports the running call"
        assert bar._clock is not None, "the 1 Hz repaint was never armed"

        def painted_bar() -> str:
            return next(
                label.text.rstrip()
                for label in app.screen._compositor.render_strips()
                if "▸" in label.text
            )

        def clock_cells(row: str) -> float:
            """The elapsed the row paints, in seconds (the only `Ns` on it)."""
            match = re.search(r"(\d+(?:\.\d+)?)s\b", row)
            assert match is not None, row
            return float(match.group(1))

        before = painted_bar()
        assert clock_cells(before) >= 6.5, before

        assert card._started is not None  # the live bar is running, so it has a start
        card._started -= 2.0  # two seconds pass with no append and no re-derive
        await pilot.pause(0.2)

        row = painted_bar()
        assert row != before, f"the bar's elapsed froze at its mount value: {row!r}"
        # Two seconds were added; one is enough to prove the repaint happened,
        # and the margin absorbs a tick landing on a rounding boundary.
        assert clock_cells(row) >= clock_cells(before) + 1.0, row


@pytest.mark.asyncio
async def test_a_by_response_step_is_flush_inside_and_bracketed_outside() -> None:
    """No blank rows between steps; one after the prompt and before the answer."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        user = _user(view, "anchor:u1")
        narration = _narration(view, "checking")
        cards = [_tool(view, f"t{index}", TOOLS[index]) for index in range(3)]
        answer = _answer(view)
        await _settle(pilot)

        (run_bar,) = _bars(view)
        assert not run_bar.expanded
        run_bar.activate()
        await _settle(pilot)
        (foot,) = _foots(view)

        rows = _visible_children(view)
        assert rows[0] is user
        assert _gap_cells(user, narration) == 1, "the prompt does not bracket its step"
        chain = [narration, run_bar, *cards]
        for previous, block in zip(chain, chain[1:]):
            assert not block.has_class(GAP_CLASS), f"{type(block).__name__} asked for a gap"
            assert _gap_cells(previous, block) == 0, (
                f"a blank row inside the step between {type(previous).__name__} and "
                f"{type(block).__name__}"
            )
        assert _gap_cells(cards[-1], answer) == 1, "the answer is not set apart"
        assert _gap_cells(answer, foot) == 0, "the foot has air above it"


@pytest.mark.asyncio
async def test_narration_is_muted_in_by_response_and_text_ink_in_by_turn() -> None:
    from tests.unit.tui.conftest import StyledTranscriptApp

    def ink_at(block: AssistantBlock, needle: str) -> str:
        from rich.style import Style
        from rich.text import Text

        text = block.renderable
        assert isinstance(text, Text)
        index = text.plain.index(needle)
        for span in text.spans:
            if span.start <= index < span.end:
                # A span's style is a Style OR a style STRING; only the former
                # carries a resolved colour.
                style = span.style
                color = style.color if isinstance(style, Style) else None
                return color.name if color is not None else ""
        return ""

    muted = theme_mod.semantic_color("muted")

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        narration = _narration(view, "checking the config")
        [_tool(view, f"t{index}", TOOLS[index]) for index in range(3)]
        answer = _answer(view, "all done")
        await _settle(pilot)

        assert ink_at(narration, "checking") == muted
        # The answer carries no span of its own: text ink is the block's CSS
        # base (`AssistantBlock { color: $lo-fg }`), so the assertion that
        # discriminates is that the STEP ink did not follow it there.
        assert ink_at(answer, "all done") != muted

        view.set_display_mode("by-turn")
        await _settle(pilot)
        assert ink_at(narration, "checking") != muted, "the By-turn render kept the By-response ink"


@pytest.mark.asyncio
async def test_the_foot_totals_a_settled_turn_and_is_absent_while_live() -> None:
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        for index in range(3):
            _tool(view, f"t{index}", TOOLS[index], seconds=1.0)
        tail = WorkingBlock("thinking", "thinking", clock=False)
        view.pin_tail(tail)
        await _settle(pilot)
        assert _foots(view) == [], "a live turn showed a settled total"

        view.remove_block(tail)
        _answer(view)
        await _settle(pilot)

        (foot,) = _foots(view)
        # The tool row's own grammar (`format_clock`): three seconds reads
        # `3.0s`, the same trailing clock a member row shows.
        assert _plain(foot) == "✓ 3 actions · 3.0s", _plain(foot)


@pytest.mark.asyncio
async def test_a_dead_turn_gets_a_wrapped_foot_and_its_notice_is_replaced() -> None:
    """The foot carries the reason once, cleaned; the notice row is withheld."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    long_reason = (
        "[session incident (openai/gpt-5)] rate_limited: the provider refused "
        "this request because the account has spent its quota for the hour and "
        "every retry landed on the same wall.  This is why the previous turn "
        "ended. Take it into account before repeating the same request."
    )
    app = StyledTranscriptApp()
    async with app.run_test(size=(80, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        _tool(view, "t0", "bash")
        _tool(view, "t1", "bash", failed=True)
        notice = NoticeBlock(long_reason, "error")
        view.append_block(notice)
        await _settle(pilot)

        (foot,) = _foots(view)
        plain = _plain(foot)
        lines = plain.splitlines()
        assert plain.startswith("✗ turn failed · ")
        assert len(lines) > 1, f"the reason did not wrap at 80 cols: {plain!r}"
        assert all(len(line) <= 80 for line in lines), lines
        assert "the provider refused" in plain
        assert "[session incident" not in plain
        assert "Take it into account" not in plain
        assert not notice.display, "the reason is painted twice"


@pytest.mark.asyncio
async def test_a_replayed_reason_drops_the_providers_suggested_action() -> None:
    """Q14: the incident's advice line is addressed to the MODEL, not the reader.

    A stored provider failure replays as the incident record, which carries a
    `suggested action:` sentence with its own framing ("The provider is failing
    server-side: a retry may work"). The live foot never shows one: the harness's
    error row states the provider's sentence and nothing else. The foot strips
    it for the same reason it strips the incident head and the trailer — it is
    talking to the model, and the step's closing row states WHY once, briefly.
    """
    from tests.unit.tui.conftest import StyledTranscriptApp

    incident = (
        "[session incident (openai/gpt-5)] provider: provider exploded: upstream 500 "
        "suggested action: The provider is failing server-side: a retry may work; "
        "wait and retry. This is why the previous turn ended. Take it into account "
        "before repeating the same request."
    )
    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        _tool(view, "t0", "bash", failed=True)
        view.append_block(NoticeBlock(incident, "error"))
        await _settle(pilot)

        (foot,) = _foots(view)
        plain = _plain(foot)
        assert "upstream 500" in plain, plain
        assert "suggested action" not in plain, plain
        assert "retry may work" not in plain, plain


@pytest.mark.asyncio
async def test_a_turn_paused_on_approval_gets_no_foot() -> None:
    """Waiting is not dead: no failure total under a question nobody answered."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        _tool(view, "t0", "bash")
        waiting = ToolCard("t1", "bash", {"command": "rm -rf build"})
        view.append_block(waiting)
        waiting.mark_waiting()
        await _settle(pilot)

        assert _foots(view) == [], "a paused turn claimed a settled total"


@pytest.mark.asyncio
async def test_by_turn_has_no_foot() -> None:
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        for index in range(3):
            _tool(view, f"t{index}", TOOLS[index])
        _answer(view)
        await _settle(pilot)

        assert _bars(view), "premise: the By-turn turn bar exists"
        assert _foots(view) == []


# ---------------------------------------------------------------------------
# Remediation round 1: the gestures and the foot against the reported repros.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [60, 80, 100])
async def test_a_long_failure_reason_wraps_and_paints_its_last_word(width: int) -> None:
    """L3: the By-response failure foot never truncates the reason.

    The old build wrapped every continuation at the full lane and then added a
    four-cell hanging indent, so each continuation was wider than the block and
    the pinned height clipped the tail. The last word is the assertion: it is
    exactly what the repro lost.
    """
    from tests.unit.tui.conftest import StyledTranscriptApp

    words = " ".join(f"word{index}" for index in range(40))
    app = StyledTranscriptApp()
    async with app.run_test(size=(width, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        for index in range(3):
            _tool(view, f"t{index}", TOOLS[index])
        view.append_block(NoticeBlock(f"turn failed: {words}", "error"))
        await _settle(pilot)

        (foot,) = _foots(view)
        lines = _plain(foot).splitlines()
        lane = foot._built_width
        assert lane > 0
        assert all(len(line) <= lane for line in lines), (lane, [len(line) for line in lines])
        assert "word39" in _plain(foot), _plain(foot)
        assert len(lines) >= 3, "premise: this reason needs wrapping"


@pytest.mark.asyncio
async def test_a_cut_off_turn_keeps_its_notice_and_gets_no_failed_foot() -> None:
    """L6: a cut-off turn is neutral — no `✗ turn failed` foot, notice visible."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        for index in range(3):
            _tool(view, f"t{index}", TOOLS[index])
        notice = NoticeBlock(
            "turn cut off — owner-lost. The transcript holds what it wrote before that "
            "and nothing after.",
            "error",
        )
        view.append_block(notice)
        await _settle(pilot)

        assert _foots(view) == [], "a cut-off turn was painted as a failed one"
        assert notice.display, "the cut-off notice must stay visible"


@pytest.mark.asyncio
async def test_a_replayed_dead_turn_states_its_reason_once() -> None:
    """L10: both stored failure rows are replaced by the one foot."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    incident = (
        "[session incident (openai/gpt-5)] provider error (HTTP 400): the upstream "
        "model rejected the request. Take it into account before repeating."
    )
    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        for index in range(3):
            _tool(view, f"t{index}", "bash", failed=index == 2)
        failed_notice = NoticeBlock("turn failed", "error")
        incident_notice = NoticeBlock(incident, "warning")
        view.append_block(failed_notice)
        view.append_block(incident_notice)
        await _settle(pilot)

        (foot,) = _foots(view)
        assert "provider error (HTTP 400)" in _plain(foot)
        assert not failed_notice.display
        assert not incident_notice.display, "the raw incident row is a second copy"
        assert _plain(foot).count("provider error") == 1


@pytest.mark.asyncio
async def test_a_dead_turns_foot_survives_a_later_notice() -> None:
    """Q13: a command run afterwards must not resurrect the raw failure row.

    The foot is the turn's one statement of why it died. Deciding the turn from
    its last row meant any notice appended later (a slash receipt, a system
    notice) withdrew the foot AND let the stored failure row repaint — the
    reason shown twice, once raw.
    """
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 40)) as pilot:
        view = app.query_one(TranscriptView)
        view.set_display_mode("by-response")
        _user(view, "anchor:u1")
        for index in range(3):
            _tool(view, f"t{index}", "bash", failed=index == 2)
        failure = NoticeBlock("provider exploded: upstream 500", "error")
        view.append_block(failure)
        await _settle(pilot)

        (foot,) = _foots(view)
        assert not failure.display, "premise: the foot withholds the raw row"

        receipt = NoticeBlock("Transcript: By response", "info")
        view.append_block(receipt)
        await _settle(pilot)

        (after,) = _foots(view)
        assert _plain(after) == _plain(foot), "the foot vanished with a later notice"
        assert not failure.display, "the raw failure row repainted"
        assert receipt.display


@pytest.mark.asyncio
async def test_f_reveals_the_failed_call_and_the_ladder_continues_from_the_card() -> None:
    """L7: the jump shows its target, and a second `f` keeps walking."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 14)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        cards = [_tool(view, f"t{index}", TOOLS[index % len(TOOLS)]) for index in range(6)]
        cards[0].restore(state="error", error="first", duration_s=1.0)
        cards[4].restore(state="error", error="second", duration_s=1.0)
        _answer(view)
        await _settle(pilot)

        (turn_bar,) = [bar for bar in _bars(view) if bar.kind == "turn"]
        turn_bar.focus()
        await _settle(pilot)
        await pilot.press("f")
        await _settle(pilot)

        assert app.focused is cards[0], "f skipped the first call's failure"
        assert view.block_row_on_screen(cards[0], 0) is not None, "the target is not on screen"

        await pilot.press("f")
        await _settle(pilot)
        assert app.focused is cards[4], "the second f did not continue the ladder"


@pytest.mark.asyncio
async def test_f_is_scoped_to_the_focused_turn() -> None:
    """L7: a clean turn does not teleport the reader into an old failure."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        misses: list[str] = []
        view.transient_notice = lambda text: misses.append(text)  # type: ignore[method-assign]
        _user(view, "anchor:u1")
        first = [_tool(view, f"a{index}", TOOLS[index]) for index in range(3)]
        first[1].restore(state="error", error="old", duration_s=1.0)
        _answer(view)
        _user(view, "anchor:u2")
        [_tool(view, f"b{index}", TOOLS[index]) for index in range(3)]
        _answer(view)
        await _settle(pilot)

        second_bar = next(bar for bar in _bars(view) if bar.kind == "turn" and "u2" in bar.key)
        second_bar.focus()
        await _settle(pilot)
        await pilot.press("f")
        await _settle(pilot)

        assert app.focused is second_bar, "f left the clean turn for an old failure"
        assert misses == ["No failed calls in this turn"], misses


@pytest.mark.asyncio
async def test_plus_keeps_the_bar_it_acted_on_on_screen() -> None:
    """L7: `+` scrolls the transcript under the reader without losing the bar."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 10)) as pilot:
        view = app.query_one(TranscriptView)
        for turn in range(4):
            _user(view, f"anchor:u{turn}")
            for index in range(3):
                _tool(view, f"t{turn}{index}", TOOLS[index])
            _answer(view)
        await _settle(pilot)
        bars = [bar for bar in _bars(view) if bar.kind == "turn"]
        bar = bars[-1]
        bar.focus()
        await _settle(pilot)
        await pilot.press("plus")
        await _settle(pilot)

        assert view.block_row_on_screen(bar, 0) is not None, "the bar scrolled off the top"


@pytest.mark.asyncio
async def test_minus_on_a_child_bar_restores_its_turn_and_right_skips_narration() -> None:
    """L11: `-` is turn-scoped; `→` lands on the first focusable child."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        _narration(view)
        cards = [_tool(view, f"t{index}", TOOLS[index]) for index in range(3)]
        _narration(view, "checking again")
        cards += [_tool(view, f"u{index}", TOOLS[index]) for index in range(3)]
        _answer(view)
        await _settle(pilot)

        (turn_bar,) = [bar for bar in _bars(view) if bar.kind == "turn"]
        turn_bar.focus()
        await _settle(pilot)
        await pilot.press("right")  # open
        await _settle(pilot)
        await pilot.press("right")  # step into the first child
        await _settle(pilot)
        run_bar = next(bar for bar in _bars(view) if bar.kind == "run")
        assert app.focused is run_bar, "right landed on narration, not the child bar"

        await pilot.press("left")  # close the run, focus steps out
        await _settle(pilot)
        await pilot.press("minus")
        await _settle(pilot)
        assert not run_bar.expanded
        assert not any(card.display for card in cards), "- did not restore the turn's defaults"


@pytest.mark.asyncio
async def test_f_reveals_a_failure_below_the_viewport_too() -> None:
    """L7: the jump scrolls its target into view in either direction."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 10)) as pilot:
        view = app.query_one(TranscriptView)
        _user(view, "anchor:u1")
        cards = [_tool(view, f"t{index}", TOOLS[index % len(TOOLS)]) for index in range(9)]
        cards[8].restore(state="error", error="boom", duration_s=1.0)
        _answer(view)
        await _settle(pilot)

        (turn_bar,) = _bars(view)
        turn_bar.activate()
        await _settle(pilot)
        assert all(card.display for card in cards), "premise: the turn is open"
        turn_bar.focus()
        await _settle(pilot)
        view.scroll_to(y=0, animate=False, immediate=True)
        await _settle(pilot)
        assert view.block_row_on_screen(cards[8], 0) is None, "premise: the failure is below"

        await pilot.press("f")
        await _settle(pilot)
        assert app.focused is cards[8]
        assert view.block_row_on_screen(cards[8], 0) is not None
