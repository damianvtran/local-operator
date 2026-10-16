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


async def test_a_lapsed_row_takes_view_and_dismiss_and_never_decline():
    listing = AskQueueList(ask_rows([_row("a1", "Which region?", status="timed_out")]))
    posted: list[Any] = []
    listing.post_message = lambda message: posted.append(message)  # type: ignore[method-assign]
    listing.action_decline()
    assert posted == []
    listing.action_dismiss()
    assert [type(message).__name__ for message in posted] == ["Dismiss"]
    assert posted[0].ask_id == "a1"
    # Enter is the explicit LOOK (UX U1): it opens the read-only card.
    posted.clear()
    listing.action_pick()
    assert [type(message).__name__ for message in posted] == ["Picked"]
    # The header advertises only what the row honours.
    assert listing.header_hints() == ("x dismiss", "enter view", "esc collapse")


async def test_the_drawer_and_the_bar_stop_counting_lapsed_asks_as_moved_on():
    """The drawer register no longer has a "moved on" half: a lapsed row is settled."""
    rows = ask_rows([_row("a1", "Deploy now?"), _row("a2", "Which region?", status="timed_out")])
    assert drawer_headline(rows) == "1 question waiting"
    assert drawer_headline(ask_rows([_row("a2", "x", status="timed_out")])) == "All asks settled"


async def test_x_on_a_lapsed_row_dismisses_and_sends_nothing_else(enabled):
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
        app.query_one(AskQueueList).focus()
        await pilot.press("x")
        await _settle(pilot)
        assert session.dismissed[0][0] == "a2"
        assert session.answered == [] and session.declined == []


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
        # No row is lit as "what Enter takes": nothing on a lapsed card is taken.
        assert not any(line.startswith("❯") for line in card.render_lines_for_test())
        card.focus()
        await _settle(pilot)
        await pilot.press("enter")
        await _settle(pilot)
        # ``answer_current`` is the blocking-card route (``tui_handle._ask_pending``);
        # no queued card is registered there today, so this pins a DEFENSIVE guard,
        # not the phone's path. A phone answers a queued ask through ``ask_respond``,
        # which the queue still accepts as ``late`` (D10 leaves the store alone).
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


# -- round 1 remediation ------------------------------------------------------


async def test_the_middle_half_is_named_pending():
    """Design D1 / UX U4 / review m2: nothing in the middle half has "moved on"."""
    from local_operator.tui.widgets.ask_queue import (
        EMPTY_OUTSTANDING,
        EMPTY_SETTLED,
        FILTER_LABELS,
    )

    assert FILTER_LABELS[FILTER_OUTSTANDING] == "Pending"
    assert EMPTY_OUTSTANDING == "No asks are pending. They have all settled — see Settled."
    assert EMPTY_SETTLED == "No asks have settled yet. They are all still pending."
    binding = next(b for b in AskQueueList.BINDINGS if getattr(b, "key", "") == "2")
    assert getattr(binding, "description", None) == "Pending"


async def test_the_bar_counts_timed_out_asks_in_questions():
    """Design D5: one noun across surfaces."""
    from local_operator.tui.widgets.ask_queue import queue_headline

    assert queue_headline(0, 1) == "1 question timed out"
    assert queue_headline(2, 3) == "2 questions waiting · 3 questions timed out"


def _glyph_ink(painted: Any) -> str:
    """The bar glyph's colour name, lower-cased — the first painted span."""
    span = next(span for span in painted.spans if span.start == 0)
    color = getattr(span.style, "color", None)
    return str(getattr(color, "name", "")).lower()


def _current_id(listing: AskQueueList) -> str:
    row = listing.current()
    assert row is not None
    return row.ask_id


async def test_the_bar_glyph_is_muted_when_nothing_is_answerable(enabled):
    """Design D4 / UX U2: a lapsed-only bar is not the live-question accent."""
    from local_operator.tui import theme as theme_mod

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a2", "Which region?", status="timed_out")]))
        await _settle(pilot)
        muted = str(theme_mod.semantic_color("muted")).lower()
        accent = str(theme_mod.semantic_color("accent")).lower()
        ink = _glyph_ink(app.query_one(AskBar).render())
        assert ink == muted and ink != accent, ink
        app._sync_ask_surface(
            ask_rows([_row("a1", "Deploy now?"), _row("a2", "Which region?", status="timed_out")])
        )
        await _settle(pilot)
        assert _glyph_ink(app.query_one(AskBar).render()) == accent


async def test_sidebar_marks_and_total_count_answerable_asks_only(enabled, tmp_path, monkeypatch):
    """Design D3 / UX U2, matching desktop #1116: a lapsed-only session owes nothing."""
    from tests.unit.tui.test_ask_fleet import _row as fleet_row
    from tests.unit.tui.test_ask_fleet import _seed

    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)
    _seed(tmp_path, "s-mixed", [fleet_row("b1"), fleet_row("b2", status="timed_out")])
    _seed(tmp_path, "s-lapsed", [fleet_row("c1", status="timed_out")])
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        counts, total = app._read_fleet_asks()
        assert counts == {"s-mixed": 1}
        assert total == 1
        app._ask_marks = dict(counts)
        app._sync_ask_surface(ask_rows([_row("a9", "Which tier?", status="timed_out")]))
        await _settle(pilot)
        marks = app._session_sidebar._asking
        assert str(session.session_id) not in marks, marks
        assert app._session_sidebar._asks_total == 1


async def test_enter_and_click_on_a_lapsed_row_open_the_read_only_card(enabled):
    """UX U1: the explicit look Ben asked for — the full question and its options."""
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
        await pilot.press("enter")
        await _settle(pilot)
        card = app.query_one(AskPickerScreen)
        assert app._ask_mounted_id == "a2"
        assert card.read_only
        assert "timed out" in card._title
        lines = "\n".join(card.render_lines_for_test())
        assert "Which region?" in lines and "Yes" in lines and "No" in lines
        await pilot.press("enter")
        await pilot.press("1")
        await _settle(pilot)
        assert session.answered == [] and session.declined == []
        # Esc returns to the list, on the row the user looked at.
        await pilot.press("escape")
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        assert _current_id(listing) == "a2"
        # A click opens it the same way.
        top = listing.content_region.y - listing.region.y
        await pilot.click(AskQueueList, offset=(4, top + listing.HEADER_ROWS + 1))
        await _settle(pilot)
        assert app._ask_mounted_id == "a3"
        assert app.query_one(AskPickerScreen).read_only


async def test_the_cursor_starts_on_the_first_answerable_row():
    """UX U3: a list mounted over lapsed-then-open rows lands on the open one."""
    rows = ask_rows(
        [
            _row("t1", "Old one?", status="timed_out"),
            _row("a1", "Deploy now?"),
            _row("a2", "Which tier?"),
        ]
    )
    listing = AskQueueList(rows)
    assert _current_id(listing) == "a1"


async def test_the_header_hints_follow_the_row_under_the_cursor():
    """UX U3: no `enter answer · d decline` over a lapsed row."""
    rows = ask_rows([_row("a1", "Deploy now?"), _row("t1", "Old one?", status="timed_out")])
    listing = AskQueueList(rows)
    assert _current_id(listing) == "a1"
    assert listing.header_hints() == ("enter answer", "d decline", "esc collapse")
    listing.action_move(1)
    assert _current_id(listing) == "t1"
    assert listing.header_hints() == ("x dismiss", "enter view", "esc collapse")


async def test_x_dismiss_is_acknowledged_at_once(enabled):
    """`x` gives immediate feedback: the row is marked in flight until the reply lands."""
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
        await pilot.press("x")
        await _settle(pilot)
        assert session.dismissed[0][0] == "a2"
        assert listing.in_flight("a2"), "the row gave no sign the key landed"
        # A second press on the same row does not send twice.
        await pilot.press("x")
        await _settle(pilot)
        assert [d[0] for d in session.dismissed] == ["a2"]
        # The frame that drops the row ends the in-flight mark.
        app._sync_ask_surface(ask_rows([_row("a3", "Which tier?", status="timed_out")]))
        await _settle(pilot)
        assert not app._ask_in_flight


async def test_a_refused_dismiss_clears_the_in_flight_mark(enabled):
    session = _AskSession()
    session.dismiss_refusals["a2"] = "already dismissed on the phone."
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
        app.query_one(AskQueueList).focus()
        await pilot.press("x")
        await _settle(pilot)
        assert "a2" not in app._ask_in_flight


async def test_a_read_only_card_refuses_paste_and_space_and_stops_inviting_text(enabled):
    """Review m1 / UX U6: nothing typed lands, and the Other row stops saying "type it here"."""
    from local_operator.tui.widgets.ask_picker import OTHER_HINT

    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        multi = _row("a1", "Which regions?")
        multi["questions"][0]["multi"] = True
        app._sync_ask_surface(ask_rows([multi]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        card = app.query_one(AskPickerScreen)
        card.focus()
        lapsed = dict(multi, status="timed_out")
        app._sync_ask_surface(ask_rows([lapsed]))
        await _settle(pilot)
        assert card.read_only
        await pilot.press("space")
        await _settle(pilot)
        assert card.checked_indexes == []
        card.action_jump_other()
        from textual import events

        card.on_paste(events.Paste("hello"))
        await _settle(pilot)
        assert card.typed_text == ""
        lines = "\n".join(card.render_lines_for_test())
        assert OTHER_HINT not in lines
        assert "type it here" not in lines


async def test_a_lapse_drops_text_typed_before_it(enabled):
    """UX U6: `Other: back` must not stay painted as if it would be sent."""
    session = _AskSession()
    app = _app(session)
    async with app.run_test(size=(120, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        card = app.query_one(AskPickerScreen)
        card.focus()
        card.action_jump_other()
        await pilot.press("b", "a", "c", "k")
        await _settle(pilot)
        assert card.typed_text == "back"
        app._sync_ask_surface(ask_rows([_row("a1", "Deploy now?", status="timed_out")]))
        await _settle(pilot)
        assert "back" not in "\n".join(card.render_lines_for_test())


async def test_x_dismiss_outlives_the_drawer_clause_at_sixty_columns():
    """UX U8: the only key a lapsed row takes is never the first hint shed."""
    rows = ask_rows([_row("t1", "Old one?", status="timed_out")])
    listing = AskQueueList(rows)
    painted = listing.header_text(60)
    assert "x dismiss" in painted, painted
    assert "Settled · 1" in listing.header_text(50), listing.header_text(50)


async def test_a_fleet_row_lapsing_on_screen_becomes_read_only(enabled):
    """Review m3: an `open` fleet row past its deadline is lapsed on the tick."""
    from tests.unit.tui.test_ask_fleet import _row as fleet_row

    app = _app(_AskSession())
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        now = app._ask_now_ms()
        past = fleet_row("b1", session_id="s-other", expires_at=now - 1_000)
        live = fleet_row("b2", session_id="s-other")
        app._ask_scope = "fleet"
        app._ask_fleet_rows = ask_rows([past, live])
        app._mount_ask_list(scope="fleet")
        await _settle(pilot)
        app._on_ask_tick()
        await _settle(pilot)
        by = {row.ask_id: row for row in app._ask_fleet_rows}
        assert by["b1"].lapsed and not by["b1"].answerable
        assert by["b2"].answerable
