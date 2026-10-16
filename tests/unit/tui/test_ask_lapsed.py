"""A LAPSED ask on the TUI: collapsed behind a status word, read-only, never opened.

Decisions D10-D12 (ask-lapsed retention record, 2026-10-15). An ask whose
deadline fired (the fold's ``timed_out``) is still OUTSTANDING to the store — a
late answer through chat is still accepted and the wire's ``asks_open`` still
counts it — but no surface OFFERS an answer any more: Ben's words were "it's
expired… we don't need to allow people to see and focus it".

So on the TUI a lapsed row reads the way a DECLINED row already reads: one
settled line with a status chip (``timed out``), inert on click and Enter, in
the Settled half. ``x`` dismiss stays, because dismissing is the one op the
queue accepts on a timed-out ask and it answers nobody. The open policy never
auto-opens over lapsed rows, and the door (``f4``/the bar) on a lapsed-only
queue opens the LIST, never a card. A card that is already open when its ask
lapses stays on screen, re-titled, with submit and decline disabled.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.asks import policy, store
from local_operator.harness.rows import ask_timeout_notice
from local_operator.tui import ask_open_policy
from local_operator.tui.app import OperatorApp
from local_operator.tui.ask_open_policy import QueueReading, read_queue
from local_operator.tui.widgets.ask_picker import AskPickerScreen
from local_operator.tui.widgets.ask_queue import (
    ASK_TOGGLE_KEY,
    FILTER_OUTSTANDING,
    FILTER_SETTLED,
    AskBar,
    AskQueueList,
    ask_rows,
    drawer_headline,
    filter_counts,
    settled_chip,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_ask_queue_surface import _AskSession, _row

pytestmark = pytest.mark.asyncio

LAPSED_TITLE_TAIL = "the agent goes ahead without your answer"


@pytest.fixture
def enabled(monkeypatch):
    """The queued-ask feature ON, the open-by-default policy OFF (a door test)."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    monkeypatch.setattr(ask_open_policy, "AUTO_OPEN", False)


def _app(session: FakeSession) -> OperatorApp:
    return OperatorApp(lambda: _factory(session))


async def _settle(pilot, turns: int = 3) -> None:
    for _ in range(turns):
        await pilot.pause()


# -- the store's one predicate ------------------------------------------------


async def test_the_lapsed_set_is_timed_out_alone_and_a_subset_of_outstanding():
    """D10: ``is_lapsed`` sits beside ``is_outstanding`` and never widens the tally."""
    assert store.LAPSED_STATUSES <= store.OUTSTANDING_STATUSES
    assert store.is_lapsed(store.STATUS_TIMED_OUT)
    for status in (
        store.STATUS_OPEN,
        store.STATUS_ANSWERED,
        store.STATUS_LATE,
        store.STATUS_DECLINED,
        store.STATUS_DISMISSED,
        store.STATUS_WITHDRAWN,
        store.STATUS_EXPIRED,
        None,
        "",
    ):
        assert not store.is_lapsed(status), status


# -- the row: a settled one-liner with a `timed out` chip ---------------------


async def test_a_lapsed_row_is_settled_and_wears_the_timed_out_chip():
    rows = ask_rows([_row("a1", "Deploy now?"), _row("a2", "Which region?", status="timed_out")])
    lapsed = rows[1]
    assert lapsed.lapsed and lapsed.settled
    assert not lapsed.answerable and not lapsed.pending
    assert settled_chip(lapsed) == ("timed out", "muted")
    # It FILTERS under Settled, the way a declined row does.
    assert filter_counts(rows) == {"all": 2, FILTER_OUTSTANDING: 1, FILTER_SETTLED: 1}
    listing = AskQueueList(rows)
    listing.set_filter(FILTER_SETTLED)
    assert [row.ask_id for row in listing.visible_rows] == ["a2"]
    painted = listing.render().plain.splitlines()[-1]
    assert painted.rstrip().endswith("· timed out"), painted
    # No countdown and no "still answerable" promise on the row.
    assert "answerable" not in listing.render().plain


async def test_a_lapsed_row_is_inert_to_pick_and_decline_and_keeps_dismiss():
    listing = AskQueueList(ask_rows([_row("a1", "Which region?", status="timed_out")]))
    posted: list[Any] = []
    listing.post_message = lambda message: posted.append(message)  # type: ignore[method-assign]
    listing.action_pick()
    listing.action_decline()
    assert posted == []
    listing.action_dismiss()
    assert [type(message).__name__ for message in posted] == ["Dismiss"]
    assert posted[0].ask_id == "a1"
    # The header advertises only what the row honours.
    assert listing.header_hints() == ("x dismiss", "esc collapse")


async def test_the_drawer_and_the_bar_stop_counting_lapsed_asks_as_moved_on():
    """The drawer register no longer has a "moved on" half: a lapsed row is settled."""
    rows = ask_rows([_row("a1", "Deploy now?"), _row("a2", "Which region?", status="timed_out")])
    assert drawer_headline(rows) == "1 question waiting"
    assert drawer_headline(ask_rows([_row("a2", "x", status="timed_out")])) == "All asks settled"


async def test_click_and_enter_on_a_lapsed_row_open_nothing_and_x_dismisses(enabled):
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a2", "Which region?", status="timed_out"),
                    _row("a3", "Which tier?", status="timed_out"),
                ]
            )
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        listing.focus()
        await _settle(pilot)
        await pilot.press("enter")
        await _settle(pilot)
        assert not app.query(AskPickerScreen), "Enter mounted a card for a lapsed ask"
        top = listing.content_region.y - listing.region.y
        await pilot.click(AskQueueList, offset=(4, top + listing.HEADER_ROWS))
        await _settle(pilot)
        assert not app.query(AskPickerScreen), "a click mounted a card for a lapsed ask"
        assert session.answered == [] and session.declined == []
        listing.focus()
        await pilot.press("x")
        await _settle(pilot)
        assert session.dismissed[0][0] == "a2"


# -- the door: a lapsed-only queue opens the list, never a card ---------------


async def test_the_door_on_a_lapsed_only_queue_opens_the_list(enabled):
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a2", "Which region?", status="timed_out")]))
        await _settle(pilot)
        bar = app.query_one(AskBar)
        # The bar stays reachable (Dismiss lives behind it) but owes nothing.
        assert bar.display
        assert "waiting" not in bar.render().plain
        await pilot.press(ASK_TOGGLE_KEY)
        await _settle(pilot)
        assert app.query(AskQueueList), "the door did not open the list"
        assert not app.query(AskPickerScreen), "the door opened a card for a lapsed ask"


async def test_one_open_ask_beside_lapsed_ones_still_goes_straight_to_its_card(enabled):
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows([_row("a1", "Deploy now?"), _row("a2", "Which region?", status="timed_out")])
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        assert app._ask_mounted_id == "a1"


# -- a card that is open when its ask lapses ----------------------------------


async def test_a_mounted_card_whose_ask_lapses_stays_but_no_longer_sends(enabled):
    """The card is not yanked from under the reader, but it stops taking answers."""
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        card = app.query_one(AskPickerScreen)
        assert not card.read_only
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?", status="timed_out")]))
        await _settle(pilot)
        card = app.query_one(AskPickerScreen)
        assert card.is_attached, "the lapse yanked the card"
        assert card.read_only
        assert "timed out" in card._title and LAPSED_TITLE_TAIL in card._title, card._title
        assert "still reaches" not in card._title
        card.focus()
        await _settle(pilot)
        await pilot.press("enter")
        await _settle(pilot)
        # The phone's route in (``tui_handle`` -> ``answer_current``) is refused
        # too, with a sentence the relay hands back to the phone.
        with pytest.raises(ValueError, match="timed out"):
            card.answer_current(["Yes"])
        await _settle(pilot)
        assert session.answered == [], "a lapsed card still sent an answer"
        assert session.declined == []
        # Esc still closes it, back to the list where the row is collapsed.
        await pilot.press("escape")
        await _settle(pilot)
        assert not app.query(AskPickerScreen)


async def test_a_card_mounted_from_the_list_returns_to_the_list_after_lapse(enabled):
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?"), _row("a2", "Which tier?")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        app.query_one(AskQueueList).focus()
        await pilot.press("enter")
        await _settle(pilot)
        assert app._ask_mounted_id == "a1"
        app._sync_ask_surface(
            ask_rows([_row("a1", "Deploy now?", status="timed_out"), _row("a2", "Which tier?")])
        )
        await _settle(pilot)
        app.query_one(AskPickerScreen).focus()
        await pilot.press("escape")
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        lapsed = next(row for row in listing.rows if row.ask_id == "a1")
        assert lapsed.settled
        assert session.answered == []


# -- the open policy ---------------------------------------------------------


async def test_read_queue_calls_a_lapsed_only_frame_settled():
    """D12: the tally counts lapsed rows, so it must not read as a hidden pending ask."""
    t0 = 1_000_000
    # Two rows published, both lapsed: no outstanding ``created_at`` to name, and the
    # wire tally is 2 because ``asks_open`` still counts timed-out asks.
    assert (
        read_queue(published=2, outstanding_created_ms=[], tally=2, opened_at_ms=t0, lapsed=2)
        is QueueReading.SETTLED
    )
    # A tally ABOVE what the rows name (open + lapsed) is still a dropped row.
    assert (
        read_queue(published=2, outstanding_created_ms=[], tally=3, opened_at_ms=t0, lapsed=2)
        is QueueReading.UNRESOLVED
    )


async def test_a_conversation_with_only_lapsed_asks_does_not_open_by_itself(monkeypatch):
    from tests.unit.tui.test_ask_open_default import _app as _policy_app
    from tests.unit.tui.test_ask_open_default import (
        _ask,
        _boot,
        _Conversation,
        _decided,
        _surface,
        _until,
    )

    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    monkeypatch.setattr(ask_open_policy, "AUTO_OPEN", True)
    session = _Conversation(
        "conv-a",
        [_ask("t1", status="timed_out"), _ask("t2", "Which tier?", status="timed_out")],
    )
    app = _policy_app(session)
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        assert await _until(pilot, lambda: _decided(app))
        assert _surface(app) == "closed", "a lapsed-only queue opened by itself"


# -- the receipt ---------------------------------------------------------------


async def test_the_timeout_receipt_says_the_agent_goes_ahead_without_the_answer():
    text, kind = ask_timeout_notice({"ask_id": "a1", "waited_s": 2520})
    assert text == "Timed out after 42m — the agent goes ahead without your answer (ask a1)"
    assert "still answer" not in text
    assert kind == "warning"
