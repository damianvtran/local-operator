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

import pytest

from local_operator.tui import theme as theme_mod
from local_operator.tui.tool_fold import ToolFoldLayer
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.tool_card import ToolCard
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
    """A miss reports through the app's notice row, not a new transcript row."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(120, 40)) as pilot:
        view = app.query_one(TranscriptView)
        seen: list[tuple[str, str]] = []

        def capture_notice(text: str, kind: str = "info") -> None:
            seen.append((text, kind))

        view.notice = capture_notice  # type: ignore[method-assign]
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

        assert seen == [("No failed calls in this turn", "info")]
        assert len(view.blocks()) == before, "a miss appended a transcript row"


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

        def capture_notice(text: str, kind: str = "info") -> None:
            seen.append(text)

        view.notice = capture_notice  # type: ignore[method-assign]
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
    """The grid: bars at 2 cells, revealed rows at 4 — and the copy agrees."""
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
        assert _plain(sub_bar).startswith("  "), "a sub-bar left the bar column"

        sub_bar.activate()  # open the sub: the calls are revealed one level in
        await _settle(pilot)
        card = cards[0]
        assert card.display
        plain = _plain(card)
        assert plain.startswith("    "), plain[:10]

        gutter = card.copy_gutter(0)
        assert gutter == 4 + ToolCard.ICON_COLS
        copied = plain[gutter:]
        assert copied.startswith(display_name("bash")), (
            "the copy gutter ate the first character of the tool name: " f"{copied[:12]!r}"
        )


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
