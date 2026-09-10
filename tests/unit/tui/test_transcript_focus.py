"""The transcript hands the keyboard back.

Two defects, one subject. Tab on a focused transcript row walked the ledger
instead of leaving it, so the presses needed to reach the input scaled with the
conversation; and ``TranscriptView`` was itself a focus stop that answered
every keystroke with silence.

Every click here derives its offset from a ``.region`` read AT TEST TIME.
``Pilot.click`` bounds-checks against ``screen.size.region``, which is two cells
smaller per axis than ``screen.region``, so a hardcoded coordinate raises
``OutOfBounds`` on one layout and passes on another.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.tui.app import COMPOSER_FOCUSED_CLASS, OperatorApp
from local_operator.tui.session_interaction import SessionDraft
from local_operator.tui.session_presentation import (
    DraftRecoveryNotice,
    HistoryPageNotice,
    OlderHistoryNotice,
)
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import (
    PeerMessageBlock,
    TranscriptBlock,
    TranscriptView,
    WakeBlock,
)

from .test_app_pilot import FakeSession, _factory


def _app() -> OperatorApp:
    return OperatorApp(lambda: _factory(FakeSession()))


def _seed_cards(app: OperatorApp, count: int) -> list[ToolCard]:
    """Append ``count`` settled tool rows and hand them back in order."""
    cards = []
    for index in range(count):
        card = ToolCard(f"t{index}", "bash", {"command": "ls"})
        app._append_block(card)
        card.mark_done(f"output {index}")
        cards.append(card)
    return cards


def _row_kinds() -> list[tuple[str, Any]]:
    """One factory per FOCUSABLE row kind that inherits the tab binding.

    The three notices are the point of the list: they descend from
    ``NoticeBlock``, not from ``ExpandableActionBlock``, so a binding placed one
    level too low passes the first rows here and fails the last three.
    """
    return [
        ("ToolCard", lambda: ToolCard("t", "bash", {"command": "ls"})),
        ("WakeBlock", lambda: WakeBlock("(alarm) Scheduled wake w1 (1).\n\ncheck the build")),
        ("PeerMessageBlock", lambda: PeerMessageBlock("hello", {"pid": 7})),
        ("HistoryPageNotice", HistoryPageNotice),
        ("OlderHistoryNotice", lambda: OlderHistoryNotice("Older messages")),
        (
            "DraftRecoveryNotice",
            lambda: DraftRecoveryNotice("token", SessionDraft(text="unsent")),
        ),
    ]


@pytest.mark.asyncio
async def test_tab_from_a_tool_card_returns_to_the_composer() -> None:
    """ONE press, from any row, regardless of how much is above it.

    The reported defect: Tab fell through to the screen's ``focus_next``, which
    walked the ledger card by card. Measured before the fix with three cards —
    ``card -> card -> Editor``, three presses to escape a surface that in a real
    session is hundreds of rows long.

    ``editor.text`` is asserted empty because the composer indents on Tab
    (TUI-013): a route that focused it by re-posting the key would arrive with
    a stray tab character in the buffer.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        cards = _seed_cards(app, 3)
        await pilot.pause()

        cards[0].focus()
        await pilot.pause()
        assert app.focused is cards[0]

        await pilot.press("tab")
        await pilot.pause()

        assert app.focused is editor
        assert app.query_one("#input-dock").has_class(COMPOSER_FOCUSED_CLASS)
        assert editor.text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("name,factory", _row_kinds(), ids=[n for n, _ in _row_kinds()])
async def test_tab_returns_from_every_focusable_row_kind(name: str, factory: Any) -> None:
    """The binding lives on ``TranscriptBlock``, and this is what proves it.

    Put it on ``ExpandableActionBlock`` instead and the three action rows here
    still pass while all three notices keep trapping Tab.

    TWO cards are seeded BELOW the row under test, and that is what makes the
    assertion mean anything. With the row last in the focus chain, the screen's
    ``focus_next`` fallback reaches the Editor by itself and the test passes
    with the binding removed — verified, it did. Rows underneath make the
    unbound behaviour land on the next ROW instead, so only the binding can
    produce the Editor.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        block = factory()
        app._append_block(block)
        _seed_cards(app, 2)
        await pilot.pause()
        block.focus()
        await pilot.pause()
        assert app.focused is block, f"{name} did not take focus"

        await pilot.press("tab")
        await pilot.pause()

        assert app.focused is editor, f"tab on {name} landed on {type(app.focused).__name__}"
        assert app.query_one("#input-dock").has_class(COMPOSER_FOCUSED_CLASS)


@pytest.mark.asyncio
async def test_up_and_down_still_walk_the_ledger() -> None:
    """Tab leaving is not Tab replacing: the arrows are still the ledger walk.

    ``focus_neighbour`` is untouched by this slice, and the scoped walk between
    rows is the thing Tab was a badly scoped duplicate OF.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        cards = _seed_cards(app, 3)
        await pilot.pause()

        cards[0].focus()
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        assert app.focused is cards[1]

        await pilot.press("down")
        await pilot.pause()
        assert app.focused is cards[2]

        await pilot.press("up")
        await pilot.pause()
        assert app.focused is cards[1]


@pytest.mark.asyncio
async def test_walking_off_the_ledger_keeps_its_original_landing_places() -> None:
    """``focus_neighbour`` is UNCHANGED by this slice, and this pins that.

    Off the bottom lands in the composer; off the top lands on the transcript
    itself, so the scroll keys take over. Both are the behaviour
    ``focus_neighbour``'s docstring has always described.

    Worth a test rather than assumed: an earlier revision of this slice cleared
    ``TranscriptView.can_focus``, which moved the top landing to the Editor and
    silently broke transcript scrolling across six suites. This asserts the
    landing places directly, so that change cannot be made again without a red
    test naming exactly what moved.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        view = app.query_one(TranscriptView)
        cards = _seed_cards(app, 3)
        await pilot.pause()

        cards[0].focus()
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()
        assert app.focused is view

        cards[-1].focus()
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        assert app.focused is editor


@pytest.mark.asyncio
async def test_enter_and_space_still_expand_a_focused_row() -> None:
    """``_bound_keys`` still names the row's own keys after the derivation moved.

    Space is the case that matters: it is printable AND the row's toggle, so a
    ``_bound_keys`` that lost it would type a space into the composer instead of
    expanding the row the user is standing on.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        cards = _seed_cards(app, 2)
        await pilot.pause()

        cards[0].focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert cards[0].expanded is True
        assert app.focused is cards[0]

        await pilot.press("space")
        await pilot.pause()
        assert cards[0].expanded is False
        assert app.focused is cards[0]
        assert editor.text == ""


@pytest.mark.asyncio
async def test_typing_on_a_focused_row_still_reaches_the_composer_intact() -> None:
    """Re-asserted against the REAL app because the derivation changed.

    ``_BOUND_KEYS`` (a class-body frozenset) became ``_bound_keys()`` (read from
    the merged binding map), and the passthrough tests it on every printable
    key. Every character is checked, not just the focus move: dropping the first
    one to "wake" the composer is the failure this forecloses.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        cards = _seed_cards(app, 1)
        await pilot.pause()

        cards[0].focus()
        await pilot.pause()
        assert app.focused is cards[0]

        await pilot.press(*"hi there")
        await pilot.pause()
        assert editor.text == "hi there"
        assert app.focused is editor


@pytest.mark.asyncio
async def test_tab_on_a_row_is_refused_while_the_composer_is_read_only() -> None:
    """Refuse rather than steal: a read-only composer answers no key.

    The subagent page and the login prompt both make the composer read-only,
    and focusing it there would hand the keyboard to a field that refuses every
    edit — the same swallowing this MR removes elsewhere. Tab correctly does
    nothing and focus stays where the reader put it.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        cards = _seed_cards(app, 2)
        await pilot.pause()

        cards[0].focus()
        await pilot.pause()
        app._set_composer_read_only(True)
        await pilot.pause()
        assert editor.can_focus is False

        cards[0].focus()
        await pilot.pause()
        assert app.focused is cards[0]
        await pilot.press("tab")
        await pilot.pause()

        assert app.focused is cards[0]
        assert editor.can_focus is False


@pytest.mark.asyncio
async def test_a_notice_losing_interactivity_does_not_land_focus_off_screen() -> None:
    """R1 — where a blurred head notice hands focus, MEASURED here.

    ``OlderHistoryNotice.set_interactive(False)`` blurs before clearing
    ``can_focus``, and that order is load-bearing: reversing it sent focus ~770
    rows up to the topmost card, off screen, with the reader's next Enter
    expanding something they could not see (review round 3, R9). The remedy
    lands on ``TranscriptView``, which is the one focus target guaranteed to be
    inside the viewport.

    Re-measured here rather than taken on trust, because this slice made
    ``TranscriptView`` forward keystrokes and an earlier revision of it removed
    the container from the focus chain entirely — which took this landing place
    away and moved focus to the Editor.

    Asserted two ways on purpose. The ON-SCREEN assertion is the one that
    actually guards R1: it fails for ANY landing widget outside the viewport,
    whatever its type, which is the property the ~770-row jump violated and a
    type check alone would not catch.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        notice = OlderHistoryNotice("Older messages")
        app._append_block(notice)
        _seed_cards(app, 6)
        await pilot.pause()

        notice.focus()
        await pilot.pause()
        assert app.focused is notice

        notice.set_interactive(False)
        await pilot.pause()

        landed = app.focused
        assert landed is not None, "focus was dropped entirely"
        assert landed is app.query_one(TranscriptView), f"focus landed on {type(landed).__name__}"
        assert app.screen.region.contains_region(
            landed.region
        ), f"{type(landed).__name__} at {landed.region} is outside {app.screen.region}"

        # And the landing place is not a keystroke sink: the reader's next
        # character reaches the composer instead of vanishing into the
        # container, which is what the rest of this slice is for.
        await pilot.press("z")
        await pilot.pause()
        assert editor.text == "z"
        assert app.focused is editor


@pytest.mark.asyncio
async def test_typing_at_the_focused_transcript_reaches_the_composer() -> None:
    """The transcript stays focusable, so it must not eat what is typed at it.

    Defect C, measured on ``origin/main``: focus ``TranscriptView``, type
    ``hello``, and ``editor.text`` was still ``''`` with focus still on the
    container — every keystroke gone, and nothing on the frame to say so.

    Removing the focus stop is NOT the fix and was tried: ``can_focus`` gates
    ``allow_vertical_scroll`` and Textual's anchor-release paths, so clearing
    it broke thirteen tests across six suites. Forwarding is the fix, and this
    is the test that says the defect is actually gone rather than relocated.

    Every character is asserted, not just the focus move: dropping the first
    one to "wake" the composer is the failure this forecloses.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        _seed_cards(app, 2)
        await pilot.pause()

        view = app.query_one(TranscriptView)
        view.focus()
        await pilot.pause()
        assert app.focused is view, "the transcript must still be focusable"

        await pilot.press(*"hello")
        await pilot.pause()

        assert editor.text == "hello"
        assert app.focused is editor


@pytest.mark.asyncio
async def test_a_focused_rows_own_keys_are_not_eaten_by_the_container() -> None:
    """The container's passthrough must not swallow a CHILD's affordance.

    Space is both a printable character and a row's toggle. A focused row
    leaves it to its own bindings, so the key bubbles — and the container's
    ``on_key`` sits on that path. Without the ``has_focus`` guard it caught
    Space on the way past and typed it into the composer while the row stayed
    collapsed: measured ``editor.text == ' '``.

    That is the identical defect ``ExpandableActionBlock.on_key`` already
    documents, reappearing one level up, which is why it is pinned at this
    level too rather than trusted to the row's own test.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        cards = _seed_cards(app, 2)
        await pilot.pause()

        cards[0].focus()
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()

        assert cards[0].expanded is True, "space did not reach the row's toggle"
        assert app.focused is cards[0]
        assert editor.text == ""


@pytest.mark.asyncio
async def test_clicking_blank_transcript_leaves_focus_on_the_composer() -> None:
    """Clearing ``can_focus`` is necessary but not sufficient.

    Textual leaves focus where it was when a click lands on a widget that
    cannot take it, so without ``TranscriptView.on_click`` a card kept focus
    while the user clicked the empty column beside it.

    The click site is derived from the regions at test time and asserted to hit
    NO block first, so the test cannot silently degrade into clicking a row.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        cards = _seed_cards(app, 2)
        await pilot.pause()

        view = app.query_one(TranscriptView)
        blocks = view.region
        # A row below the last card, still inside the transcript: blank column.
        blank_y = min(
            cards[-1].region.bottom + 1,
            blocks.bottom - 1,
            app.screen.size.region.height - 1,
        )
        offset = (blocks.x + blocks.width // 2, blank_y)
        hit, _ = app.screen.get_widget_at(*offset)
        assert not isinstance(hit, TranscriptBlock), f"click site hit a {type(hit).__name__}"

        cards[0].focus()
        await pilot.pause()
        assert app.focused is cards[0]

        await pilot.click(offset=offset)
        await pilot.pause()

        assert app.focused is editor
        assert app.query_one("#input-dock").has_class(COMPOSER_FOCUSED_CLASS)


@pytest.mark.asyncio
async def test_clicking_a_row_still_focuses_and_expands_it() -> None:
    """The blank-click handler must not eat the row's own click.

    ``TranscriptView.on_click`` deliberately does not call ``event.stop()`` and
    returns early when the click hit a block, so click-to-expand is unchanged.
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        cards = _seed_cards(app, 2)
        await pilot.pause()

        await pilot.click(cards[0])
        await pilot.pause()

        assert cards[0].expanded is True
        assert app.focused is cards[0]
