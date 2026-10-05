"""The TUI's FLEET half: settled rows, the three-way filter, marks, the engage chain.

Design ``docs/design/ask-nonblocking.md`` §4/§5 plus the amendment section §11
(TUI fleet scope, filter, marks). Three things are being proved here, and each
one is a claim a still cannot make:

* **The list shows the WHOLE queue and the filter picks a half.** ``ask_rows``
  stops dropping settled rows, so the halves have to partition every status and
  a half that is empty has to say so rather than reading as an empty queue.
* **Every session with a pending ask is MARKED, and the fleet total is the sum**
  — sourced from the cross-session index, unioned with the current session's
  live wire count so a just-queued ask marks before any index write.
* **A fleet row is answered through the ENGAGE seam addressed by its OWN
  session** (A7) — the phone daemon's cold-answer arm — while a row of the
  adopted session keeps the owner contract.

The kill switch is asserted too (A8): with the flag off there is no fleet
total, no extra marks and no fleet scope, exactly as ``_sync_ask_surface``'s
early return has always made the rest absent.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from rich.cells import cell_len

from local_operator.asks import policy, store
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.ask_queue import (
    DELIVERING_MARK,
    EMPTY_ALL,
    EMPTY_OUTSTANDING,
    EMPTY_SETTLED,
    FILTER_ALL,
    FILTER_OUTSTANDING,
    FILTER_SETTLED,
    SCOPE_FLEET,
    SCOPE_SESSION,
    IN_FLIGHT_MARK,
    AskQueueList,
    ask_rows,
    delivering_chip,
    drawer_headline,
    empty_sentence,
    filter_counts,
    settled_chip,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory

pytestmark = pytest.mark.asyncio

_NOW = int(time.time() * 1000)


def _row(
    ask_id: str, *, status: str = "open", delivered: bool = False, **extra: Any
) -> dict[str, Any]:
    """One fold-shaped ask record, the shape ``ask_rows`` flattens."""
    return {
        "ask_id": ask_id,
        "created_at": _NOW - 60_000,
        "expires_at": _NOW + 3_600_000,
        "timeout_s": 3600,
        "urgent": False,
        "status": status,
        "delivered": delivered,
        "questions": [
            {
                "id": "q1",
                "question": f"Question for {ask_id}?",
                "options": [{"label": "yes"}, {"label": "no"}],
                "multi": False,
                "recommended": None,
                "secret": False,
                "persist": False,
            }
        ],
        **extra,
    }


class _RecordingSession(FakeSession):
    """The owner contract, recorded — the path a current-session row keeps."""

    def __init__(self) -> None:
        super().__init__()
        self.answered: list[tuple[str, dict[str, Any], str]] = []
        self.declined: list[tuple[str, str]] = []

    def respond_ask(self, ask_id, answers, *, by="unknown"):
        self.answered.append((ask_id, dict(answers), by))
        return {"ok": True}

    def decline_ask(self, ask_id, *, by="unknown"):
        self.declined.append((ask_id, by))
        return {"ok": True}

    def dismiss_ask(self, ask_id, *, by="unknown"):
        return {"ok": True}


def _app(session: FakeSession | None = None) -> OperatorApp:
    return OperatorApp(lambda: _factory(session or _RecordingSession()))


async def _settle(pilot, turns: int = 3) -> None:
    for _ in range(turns):
        await pilot.pause()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)


@pytest.fixture
def isolated_index(monkeypatch, tmp_path):
    """Point ``config_dir()`` at a scratch root and return it.

    ``read_index`` sweeps an entry whose session directory is GONE, so a seeded
    session needs its directory too — that is the index's own orphan rule, not
    a test ritual.
    """
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)
    return tmp_path


def _seed(config_root, session_id: str, asks: list[dict[str, Any]]) -> None:
    (config_root / "sessions" / session_id).mkdir(parents=True, exist_ok=True)
    store.write_entry(config_root, session_id, cwd=f"/tmp/{session_id}", asks=asks)


# -- the view-model: every status has exactly one half ------------------------


async def test_ask_rows_keeps_every_status_and_carries_delivered():
    rows = ask_rows(
        [
            _row("a1"),
            _row("a2", status="timed_out"),
            _row("a3", status="answered", delivered=True),
            _row("a4", status="answered", delivered=False),
            _row("a5", status="declined"),
            _row("a6", status="expired"),
        ]
    )
    assert [row.ask_id for row in rows] == ["a1", "a2", "a3", "a4", "a5", "a6"]
    by = {row.ask_id: row for row in rows}
    assert by["a1"].waiting and by["a1"].answerable and by["a1"].pending
    assert by["a2"].moved_on and by["a2"].pending and by["a2"].answerable
    assert by["a3"].settled and not by["a3"].delivering
    # ANSWERED BUT NOT DELIVERED is §10's in-flight half: pending, not answerable.
    assert by["a4"].delivering and by["a4"].pending and not by["a4"].answerable
    assert by["a5"].settled and by["a6"].settled
    assert filter_counts(rows) == {
        FILTER_ALL: 6,
        FILTER_OUTSTANDING: 3,
        FILTER_SETTLED: 3,
    }


async def test_the_halves_partition_every_status():
    statuses = ["open", "timed_out", "answered", "late", "declined", "dismissed", "expired"]
    for status in statuses:
        for delivered in (True, False):
            row = ask_rows([_row("a1", status=status, delivered=delivered)])[0]
            assert row.pending != row.settled, (status, delivered)
            assert row.answerable or row.delivering or row.settled


# -- the drawer register -----------------------------------------------------


async def test_the_drawer_register_spells_both_halves_when_mixed():
    mixed = ask_rows([_row("a1"), _row("a2", status="timed_out")])
    assert drawer_headline(mixed) == "1 waiting, 1 moved on"
    assert drawer_headline(ask_rows([_row("a1"), _row("a2")])) == "2 questions waiting"
    # All settled reads the DESKTOP DRAWER's own constant (round 2: D4/F3). The
    # chip register counts what is owed and has no words for this state, and
    # `N settled` was a second spelling of a sentence the desktop already ships.
    assert drawer_headline(ask_rows([_row("a1", status="declined")])) == "All asks settled"
    assert drawer_headline(mixed, open_count=9, truncated=True) == "9 outstanding"


async def test_the_drawer_register_speaks_moved_on_for_a_moved_on_only_queue():
    """The drawer's own vocabulary, not the chip's (round 2: F3/D4).

    The segments above the sentence and the rows below it both say "moved on";
    the chip clause said "timed out" for the same half. One half, one word —
    and the bar keeps the chip register it owns.
    """
    assert drawer_headline(ask_rows([_row("a1", status="timed_out")])) == "1 question moved on"
    two = ask_rows([_row("a1", status="timed_out"), _row("a2", status="timed_out")])
    assert drawer_headline(two) == "2 questions moved on"


async def test_the_drawer_register_never_calls_a_delivering_row_settled():
    """§10's second half, and the contradiction round 2 found (F3/Q1/U4).

    A delivering row is PENDING — it sits in the `Waiting or moved on` segment
    and keeps the session's mark — so `N settled` over it was the header
    contradicting its own segment. It gets §5's own words instead, and never the
    desktop's "you can still change it": this surface has no revise wire.
    """
    delivering = ask_rows([_row("a1", status="answered", delivered=False)])
    assert delivering[0].pending and not delivering[0].settled
    assert drawer_headline(delivering) == "1 answer delivering — the agent will be told"
    two = ask_rows(
        [
            _row("a1", status="answered", delivered=False),
            _row("a2", status="late", delivered=False),
        ]
    )
    assert drawer_headline(two) == "2 answers delivering — the agent will be told"
    # Beside a waiting ask the delivering clause JOINS the sentence.
    joined = ask_rows([_row("a1"), _row("a2", status="answered", delivered=False)])
    assert drawer_headline(joined) == (
        "1 question waiting · 1 answer delivering — the agent will be told"
    )
    # Once it is delivered it is settled, and no clause can contradict that.
    delivered = ask_rows([_row("a1", status="answered", delivered=True)])
    assert drawer_headline(delivered) == "All asks settled"


async def test_a_settled_status_word_reads_as_its_own_field():
    """D6: `declined` shared the question's own ink, so the row read as one sentence.

    Two spaces were the only boundary between `Should the retry budget double?`
    and the word the reader has to classify. The `·` is this row's own tail
    separator (`handle · deadline`), so the status lands in a register the row
    already has.
    """
    listing = AskQueueList(ask_rows([_row("a1", status="declined")]))
    painted = listing.render().plain.splitlines()[-1]
    assert painted.rstrip().endswith("· declined")
    # The boundary is a SEPARATOR, not the whitespace it used to be.
    assert "  declined" not in painted
    # The bar's own register is untouched by this: it is the list that needed
    # the field boundary.
    assert settled_chip(listing.rows[0]) == ("declined", "muted")


async def test_a_delivering_row_is_pending_but_not_answerable():
    """It counts in the middle half and is inert to every gesture (F3/U4/A1)."""
    row = ask_rows([_row("a1", status="answered", delivered=False)])[0]
    listing = AskQueueList([row])
    assert listing.counts == {"all": 1, "outstanding": 1, "settled": 0}
    assert listing.visible_rows == [row]
    assert drawer_headline([row]) != "1 settled"
    # ...and the row's own tail says what it is, where a settled row wears its
    # chip: the reader can tell the two apart by words, not by colour.
    assert delivering_chip(row) == ("answered, delivering", "success")
    assert settled_chip(row) == (row.status, "success")
    rendered = listing.render().plain
    assert "answered, delivering" in rendered
    # Both gestures are refused at the widget, with no message posted.
    posted: list[Any] = []
    listing.post_message = lambda message: posted.append(message)  # type: ignore[method-assign]
    listing.action_pick()
    listing.action_decline()
    listing.action_dismiss()
    assert posted == []


async def test_the_drawer_register_keeps_the_urgency_word_when_the_halves_mix():
    rows = ask_rows([_row("a1", urgent=True), _row("a2", status="timed_out", urgent=True)])
    # OPEN urgent asks only — the same set the bar counts, so a timed-out one
    # cannot inflate the word beside a bar that shows the hue alone.
    assert drawer_headline(rows) == "1 waiting, 1 moved on · 1 urgent"


async def test_a_truncated_frame_withholds_the_split_and_states_the_tally():
    rows = ask_rows([_row("a1"), _row("a2", status="timed_out")])
    assert drawer_headline(rows) == "1 waiting, 1 moved on"
    assert drawer_headline(rows, open_count=20, truncated=True) == "20 outstanding"


async def test_an_empty_half_names_the_half_holding_the_rows():
    open_only = ask_rows([_row("a1")])
    settled_only = ask_rows([_row("a1", status="declined")])
    mixed = ask_rows([_row("a1"), _row("a2", status="answered", delivered=True)])
    assert empty_sentence([], FILTER_ALL) == EMPTY_ALL
    assert empty_sentence(mixed, FILTER_ALL) is None
    assert empty_sentence(open_only, FILTER_SETTLED) == EMPTY_SETTLED
    assert empty_sentence(settled_only, FILTER_OUTSTANDING) == EMPTY_OUTSTANDING
    assert empty_sentence(mixed, FILTER_OUTSTANDING) is None
    assert empty_sentence(mixed, FILTER_SETTLED) is None


# -- the header: scope subject, segments, live counts, the sacrifice order ----


async def test_the_header_carries_the_scope_subject_and_the_three_segments():
    rows = ask_rows([_row("a1"), _row("a2", status="timed_out")])
    session_head = AskQueueList(rows, scope=SCOPE_SESSION).header_text(200)
    assert session_head.startswith("? ")
    assert "This conversation" not in session_head
    for segment in ("All · 2", "Waiting or moved on · 2", "Settled · 0"):
        assert segment in session_head, session_head
    fleet_head = AskQueueList(rows, scope=SCOPE_FLEET).header_text(200)
    assert fleet_head.startswith("? All conversations · 1 waiting, 1 moved on")


async def test_the_header_keeps_the_control_longest_and_drops_the_clause_whole():
    rows = ask_rows([_row("a1"), _row("a2", status="timed_out")])
    listing = AskQueueList(rows)
    wide = listing.header_text(200)
    assert "d decline" in wide and "esc collapse" in wide
    assert "1 waiting, 1 moved on" in wide
    narrow = listing.header_text(60)
    assert cell_len(narrow) <= 60
    # The control outlives everything: a filtered view that cannot be
    # un-filtered is worse than an unadvertised key.
    assert "Waiting or moved on" in narrow
    assert "1 waiting, 1 moved on" not in narrow


async def test_the_header_hints_follow_the_view():
    """Hints describe what the VISIBLE rows can do (round 2: D2/F7/U9).

    The set used to be spent by width alone, so an all-settled queue advertised
    `enter answer · d decline` over rows that are inert by construction — a
    dead-end affordance, inverted against the rows it sat over.
    """
    settled = AskQueueList(ask_rows([_row("a1", status="declined")]))
    assert settled.header_hints() == ("esc collapse",)
    assert "enter answer" not in settled.header_text(200)
    assert "d decline" not in settled.header_text(200)
    assert "esc collapse" in settled.header_text(200)
    # An EMPTY HALF advertises nothing either — the middle half can hold only
    # delivering rows, which are pending but not answerable.
    delivering = AskQueueList(ask_rows([_row("a1", status="answered", delivered=False)]))
    assert delivering.header_hints() == ("esc collapse",)
    # A moved-on row IS answerable, and `x dismiss` names a key it can honour.
    moved = AskQueueList(ask_rows([_row("a1", status="timed_out")]))
    assert moved.header_hints() == ("d decline", "enter answer", "x dismiss", "esc collapse")
    # An OPEN row cannot be dismissed (dismiss is offered on a timed-out ask
    # alone), so that key is not advertised over it.
    opened = AskQueueList(ask_rows([_row("a1")]))
    assert "x dismiss" not in opened.header_hints()


async def test_the_irreversible_hint_outlives_the_reversible_one():
    """`d decline` is kept longest, and the base named it at 100x30 (D3).

    The selected row already carries the `❯` cue for Enter; `d` has no other
    teacher anywhere on the frame, and it cannot be undone.
    """
    rows = ask_rows(
        [
            _row("a1"),
            _row("a2", status="timed_out"),
            _row("a3", status="answered", delivered=True),
            _row("a4", status="declined"),
        ]
    )
    listing = AskQueueList(rows)
    for width in (60, 80, 100, 130):
        painted = listing.header_text(width)
        if "decline" in painted:
            assert "d decline" in painted, (width, painted)
        if "enter answer" in painted:
            assert "d decline" in painted, (width, painted)
    # D3's own case: the SESSION scope at 100x30 — the size the base named it at.
    assert "d decline" in listing.header_text(100)
    # The FLEET scope spends 6 more cells on its subject, so at 100 the hints are
    # gone whole rather than clipped — the control survives, which is what A9 and
    # D3's own `keep one-line + control-survives-last` rule require.
    fleet = AskQueueList(rows, scope=SCOPE_FLEET)
    assert "All conversations" in fleet.header_text(100)
    assert "decline" not in fleet.header_text(100)
    assert "d decline" in fleet.header_text(130)


async def test_the_segments_are_one_painted_line_and_never_split_a_hint():
    rows = ask_rows([_row("a1"), _row("a2", status="timed_out")])
    listing = AskQueueList(rows)
    for width in (47, 60, 80, 100, 130, 190):
        painted = listing.header_text(width)
        assert "\n" not in painted
        assert cell_len(painted) <= width, (width, painted)
        for hint in listing.HEADER_HINTS:
            if hint.split()[-1] in painted:
                assert hint in painted, (width, hint)


# -- the filter, through the real app ----------------------------------------


async def test_the_filter_keys_and_the_segments_scope_the_visible_rows(enabled):
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows(
                [
                    _row("a1"),
                    _row("a2", status="timed_out"),
                    _row("a3", status="declined"),
                ]
            )
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        assert [row.ask_id for row in listing.visible_rows] == ["a1", "a2", "a3"]
        await pilot.press("3")
        await _settle(pilot)
        assert listing.filter_value == FILTER_SETTLED
        assert [row.ask_id for row in listing.visible_rows] == ["a3"]
        await pilot.press("2")
        await _settle(pilot)
        assert listing.filter_value == FILTER_OUTSTANDING
        assert [row.ask_id for row in listing.visible_rows] == ["a1", "a2"]
        # `]` cycles all → outstanding → settled → all.
        await pilot.press("]")
        await _settle(pilot)
        assert listing.filter_value == FILTER_SETTLED
        await pilot.press("]")
        await _settle(pilot)
        assert listing.filter_value == FILTER_ALL
        await pilot.press("[")
        await _settle(pilot)
        assert listing.filter_value == FILTER_SETTLED


async def test_a_truncated_frame_MOUNTS_with_the_backend_tally(enabled):
    """A6 reaches the list the moment it is built, not only on the next snapshot.

    The wire's ``asks_open``/``asks_truncated`` have to ride the MOUNT as well as
    the reconcile: a list built without them states the visible split from rows
    that are a prefix of the queue, which is the claim the truncated form exists
    to avoid — and it would keep claiming it until some later snapshot happened
    to land.
    """
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows([_row("a1"), _row("a2", status="timed_out")]),
            open_count=7,
            truncated=True,
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        header = app.query_one(AskQueueList).render().plain.splitlines()[0]
        assert "7 outstanding" in header, header
        assert "waiting, " not in header, header


async def test_a_filtered_empty_half_still_names_the_half_holding_the_rows(enabled):
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        # TWO outstanding rows, so the expansion lands on the LIST rather than a
        # single ask's card — the filter only exists on the list.
        app._sync_ask_surface(ask_rows([_row("a1"), _row("a2", status="timed_out")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        await pilot.press("3")
        await _settle(pilot)
        painted = listing.render().plain
        assert listing.visible_rows == []
        assert EMPTY_SETTLED in painted, painted
        assert painted.splitlines()[-1] == EMPTY_SETTLED
        await pilot.press("2")
        await _settle(pilot)
        assert EMPTY_OUTSTANDING not in listing.render().plain


async def test_an_answered_but_undelivered_row_is_pending_and_wears_the_glyph(enabled):
    """§10's delivering half: recorded, not yet in the transcript, still owed."""
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a4", status="answered", delivered=False)]))
        await _settle(pilot)
        # Nothing is ANSWERABLE, so the expansion lands on the list rather than
        # a card — and the row it shows is the delivering one.
        app._expand_asks()
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        assert [row.ask_id for row in listing.visible_rows] == ["a4"]
        assert listing.counts[FILTER_OUTSTANDING] == 1
        assert DELIVERING_MARK in listing.render().plain


async def test_a_settled_row_is_read_only_and_its_gestures_are_inert(enabled):
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a3", status="declined")]))
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        assert [row.ask_id for row in listing.rows] == ["a3"]
        for key in ("enter", "d", "x"):
            await pilot.press(key)
            await _settle(pilot)
        # No card, no refusal toast, nothing bought: an expiry or a decline is
        # not a failure, so there is nothing to refuse (A1).
        assert len(app.query("AskPickerScreen")) == 0
        assert session.answered == [] and session.declined == []
        assert app._ask_mode is True


async def test_a_header_segment_is_a_press_target(enabled):
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app._sync_ask_surface(
            ask_rows([_row("a1"), _row("a2", status="timed_out"), _row("a3", status="declined")])
        )
        await _settle(pilot)
        app._expand_asks()
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        spans = dict(
            (value, (start, end))
            for value, start, end in listing._segment_spans(max(1, int(listing.content_size.width)))
        )
        assert FILTER_SETTLED in spans
        start, _end = spans[FILTER_SETTLED]
        top = listing.region.y + listing._top_inset()
        await pilot.click(AskQueueList, offset=(start, top - listing.region.y))
        await _settle(pilot)
        assert listing.filter_value == FILTER_SETTLED


# -- the fleet tally and the marks -------------------------------------------


async def test_the_fleet_tally_sums_the_index_and_the_marks_cover_every_session(
    enabled, isolated_index
):
    _seed(isolated_index, "s-other", [_row("b1"), _row("b2", status="timed_out")])
    _seed(isolated_index, "s-settled", [_row("c1", status="declined")])
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        counts, total = app._read_fleet_asks()
        assert counts == {"s-other": 2}
        assert total == 2
        app._ask_marks = dict(counts)
        app._sync_ask_surface(ask_rows([_row("a1")]))
        await _settle(pilot)
        marks = app._session_sidebar._asking
        assert marks["s-other"] == 2
        # The current session's LIVE wire count is unioned in — the index lags it.
        session = app._session
        assert session is not None
        assert marks[str(session.session_id)] == 1
        # A4: the total is the SUM of the unioned marks, so a live ask the index
        # has not learned about yet is still counted — the footer and the marks
        # on screen state one number for one fleet.
        assert app._session_sidebar._asks_total == 3


async def test_an_index_with_nothing_outstanding_marks_nobody_and_totals_zero(
    enabled, isolated_index
):
    _seed(isolated_index, "s-settled", [_row("c1", status="answered", delivered=True)])
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        counts, total = app._read_fleet_asks()
        assert counts == {} and total == 0
        app._ask_marks = dict(counts)
        app._sync_ask_surface(ask_rows([]))
        await _settle(pilot)
        assert app._session_sidebar._asking == {}
        assert app._session_sidebar._asks_total == 0


async def test_the_sidebar_footer_never_states_a_zero_fleet_total(enabled):
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        sidebar = app._session_sidebar
        assert sidebar._asks_note() == ""
        assert sidebar._asks_hit(4, sidebar.size.height - 1) is False
        sidebar.set_asks_total(3)
        await _settle(pilot)
        assert sidebar._asks_note() == "asks: 3"
        painted = sidebar._footer_line(120)[0]
        assert "asks: 3" in painted
        # The note is a door only where it is PAINTED, and only while the
        # total is stated at all.
        assert sidebar._footer_line(1)[0] == "…"
        assert sidebar._footer_line(120)[0].endswith("asks: 3")
        sidebar.set_asks_total(0)
        await _settle(pilot)
        assert sidebar._asks_note() == ""


async def test_the_kill_switch_hides_every_fleet_surface(monkeypatch, enabled, isolated_index):
    _seed(isolated_index, "s-other", [_row("b1")])
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app._ask_marks = {"s-other": 1}
        app._ask_fleet_rows = ask_rows([_row("b1", session_id="s-other")])
        app._ask_scope = SCOPE_FLEET
        monkeypatch.setattr(policy, "NONBLOCKING_ASK", False)
        app._sync_ask_surface(ask_rows([_row("a1")]))
        await _settle(pilot)
        assert app._ask_marks == {}
        assert app._ask_fleet_rows == []
        assert app._ask_scope == SCOPE_SESSION
        assert app._session_sidebar._asking == {}
        assert app._session_sidebar._asks_total == 0
        # And the door is inert rather than opening a scope the build lacks.
        app.action_open_fleet_asks()
        await _settle(pilot)
        assert app._ask_scope == SCOPE_SESSION


# -- the engage chain (A7) ---------------------------------------------------


class _FakeAttachClient:
    def __init__(self, record: dict[str, Any], raise_on: set[str] | None = None) -> None:
        self._record = record
        self._raise_on = raise_on or set()

    async def ask_respond(self, ask_id, answers, *, by=""):
        if "respond" in self._raise_on:
            raise RuntimeError("that ask is not in this session's queue — it may belong elsewhere.")
        self._record["respond"] = (ask_id, {k: list(v) for k, v in answers.items()}, by)
        return "ok"

    async def ask_decline(self, ask_id, *, by=""):
        self._record["decline"] = (ask_id, by)
        return "ok"

    async def ask_dismiss(self, ask_id, *, by=""):
        self._record["dismiss"] = (ask_id, by)
        return "ok"

    def close(self) -> None:
        self._record["closed"] = True


def _patch_engage(monkeypatch, record: dict[str, Any], raise_on: set[str] | None = None) -> None:
    async def _fake(config_dir, session_id, work, **kwargs):
        record["engage"] = (str(session_id), type(work).__name__, getattr(work, "ask_id", None))
        return _FakeAttachClient(record, raise_on), "detail"

    monkeypatch.setattr("local_operator.mobile.attach_client.engage_session_client", _fake)


async def test_a_fleet_row_of_another_session_answers_through_the_engage_chain(
    enabled, isolated_index, monkeypatch
):
    record: dict[str, Any] = {}
    _patch_engage(monkeypatch, record)
    _seed(isolated_index, "s-other", [_row("b1")])
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        row = ask_rows([_row("b1", session_id="s-other")])[0]
        await app._fleet_ask_worker(row, "respond", {"q1": ["Yes"]})
        assert record["engage"] == ("s-other", "AskErrand", "b1")
        assert record["respond"] == ("b1", {"q1": ["Yes"]}, "terminal")
        assert record["closed"] is True


async def test_a_fleet_decline_and_dismiss_ride_the_same_chain(
    enabled, isolated_index, monkeypatch
):
    record: dict[str, Any] = {}
    _patch_engage(monkeypatch, record)
    _seed(isolated_index, "s-other", [_row("b1"), _row("b2", status="timed_out")])
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        rows = ask_rows(
            [
                _row("b1", session_id="s-other"),
                _row("b2", status="timed_out", session_id="s-other"),
            ]
        )
        await app._fleet_ask_worker(rows[0], "decline", None)
        assert record["decline"] == ("b1", "terminal")
        await app._fleet_ask_worker(rows[1], "dismiss", None)
        assert record["dismiss"] == ("b2", "terminal")


async def test_a_refused_fleet_answer_says_so_in_the_asks_own_words(
    enabled, isolated_index, monkeypatch
):
    record: dict[str, Any] = {}
    _patch_engage(monkeypatch, record, raise_on={"respond"})
    _seed(isolated_index, "s-other", [_row("b1")])
    seen: list[str] = []
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        monkeypatch.setattr(app, "_ask_gesture_refusal", lambda text: seen.append(text))
        row = ask_rows([_row("b1", session_id="s-other")])[0]
        await app._fleet_ask_worker(row, "respond", {"q1": ["Yes"]})
        assert seen and "may belong elsewhere" in seen[0]
        assert record["closed"] is True


async def test_a_fleet_row_of_this_session_keeps_the_owner_contract(enabled, isolated_index):
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app._ask_scope = SCOPE_FLEET
        own = ask_rows([_row("a1", session_id=str(session.session_id))])[0]
        other = ask_rows([_row("b1", session_id="s-other")])[0]
        app._ask_fleet_rows = [own, other]
        assert app._fleet_answer_starts_here(own) is False
        assert app._fleet_answer_starts_here(other) is True
        app._decline_ask("a1")
        await _settle(pilot)
        assert session.declined == [("a1", "terminal")]


async def test_the_settle_handler_routes_by_the_rows_own_session(enabled, isolated_index):
    """The seam the mounted card actually calls: ``_ask_settle_callback(row)``.

    The two halves of DECISION 3 meet here. A row of the adopted session must
    reach the OWNER contract (``_on_queue_ask_settle``), and a row belonging to
    another conversation must reach the engage seam (``_on_fleet_ask_settle``) —
    whichever surface happens to be on screen. Asserted by swapping both targets
    for recorders, so the assertion is about the ROUTE and not about the op.
    """
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        routed: list[tuple[str, object]] = []
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            app, "_on_fleet_ask_settle", lambda row, answers: routed.append(("fleet", row))
        )
        monkeypatch.setattr(
            app, "_on_queue_ask_settle", lambda ask_id, answers: routed.append(("owner", ask_id))
        )
        app._ask_scope = SCOPE_FLEET
        own = ask_rows([_row("a1", session_id=str(session.session_id))])[0]
        other = ask_rows([_row("b1", session_id="s-other")])[0]
        app._ask_fleet_rows = [own, other]

        app._ask_settle_callback(own)({"q1": ["Yes"]})
        app._ask_settle_callback(other)({"q1": ["Yes"]})
        assert [kind for kind, _ in routed] == ["owner", "fleet"]
        assert routed[0][1] == "a1"
        assert routed[1][1] is other
        monkeypatch.undo()


async def test_an_in_flight_row_refuses_a_second_gesture(enabled, isolated_index):
    """A7's double-fire guard, at the surface: the mark and the refusal are one state."""
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app._ask_scope = SCOPE_FLEET
        app._ask_fleet_rows = ask_rows([_row("b1", session_id="s-other")])
        app._mount_ask_list(scope=SCOPE_FLEET)
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        current = listing.current()
        assert current is not None
        assert current.ask_id == "b1"
        listing.set_in_flight("b1", True)
        assert listing.in_flight("b1") is True
        assert listing.render().plain  # paints, does not raise
        posted: list[object] = []
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(listing, "post_message", lambda message: posted.append(message))
        listing.action_pick()
        listing.action_decline()
        listing.action_dismiss()
        assert posted == [], "an in-flight row must not fire twice"
        listing.set_in_flight("b1", False)
        listing.action_pick()
        assert len(posted) == 1
        monkeypatch.undo()


async def test_the_fleet_list_opens_from_the_sidebar_note(enabled, isolated_index):
    _seed(isolated_index, "s-other", [_row("b1")])
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app.action_open_fleet_asks()
        await _settle(pilot, 6)
        assert app._ask_scope == SCOPE_FLEET
        listing = app.query_one(AskQueueList)
        assert listing._scope == SCOPE_FLEET
        assert "All conversations" in listing.header_text(200)
        assert [row.ask_id for row in listing.rows] == ["b1"]
        # Leaving the surface leaves the scope with it.
        app._collapse_asks()
        await _settle(pilot)
        assert app._ask_scope == SCOPE_SESSION
        assert app._ask_fleet_rows == []


# -- round 2: the blockers and the majors ------------------------------------
#
# Each of these reproduces a finding from the round-1 reviews. They are grouped
# here rather than beside their subjects because what they have in common is the
# SHAPE of the failure: a surface that a snapshot, a second press or a second
# gesture used to take away.


async def test_a_snapshot_never_tears_down_a_fleet_list_or_card(enabled, isolated_index):
    """F1, reproduced: both teardown checks read the CURRENT session's rows.

    A fleet row belongs to another conversation, so it is never in this
    session's fold — and a snapshot lands on any state change, so the door
    opened and the surface vanished under the reader.
    """
    _seed(isolated_index, "s-other", [_row("b1")])
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        # PRIME THE SWAP DETECTOR FIRST, so only the two teardown checks under
        # test can fire: a session SWAP legitimately collapses the surface (the
        # dock belongs to the conversation), and a fleet surface must be judged
        # against the branches the reviewer named.
        app._ask_session = app._session
        app.action_open_fleet_asks()
        await _settle(pilot, 6)
        listing = app.query_one(AskQueueList)
        assert app._ask_scope == SCOPE_FLEET
        # The ordinary fleet-door case: nothing outstanding HERE, so the
        # snapshot carries no current-session rows at all.
        app._sync_ask_surface(ask_rows([]))
        await _settle(pilot)
        assert app._ask_scope == SCOPE_FLEET
        assert list(app.query(AskQueueList)) == [listing]

        # ...and the CARD, which is how a fleet row is actually answered.
        app.on_ask_queue_list_picked(AskQueueList.Picked("b1"))
        await _settle(pilot, 4)
        assert app._ask_card is not None
        app._sync_ask_surface(ask_rows([]))
        await _settle(pilot)
        assert app._ask_card is not None and app._ask_card.is_attached

        # CONTROL: the session scope still collapses with its last row, which is
        # the behaviour these checks exist for.
        app._collapse_asks()
        await _settle(pilot)
        app._sync_ask_surface(ask_rows([_row("a1")]))
        await _settle(pilot)
        app._mount_ask_list(scope=SCOPE_SESSION)
        await _settle(pilot)
        assert app.query(AskQueueList)
        app._sync_ask_surface(ask_rows([]))
        await _settle(pilot)
        assert not app.query(AskQueueList)


async def test_the_fleet_door_never_mounts_a_second_list(enabled, isolated_index):
    """U1, reproduced: a second press raised `DuplicateIds` and ended the app.

    Both ways in — the note pressed again while the fleet list is up, and `f4`'s
    session list followed by the note — must land on the ONE live widget.
    """
    _seed(isolated_index, "s-other", [_row("b1")])
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app.action_open_fleet_asks()
        await _settle(pilot, 6)
        first = app.query_one(AskQueueList)
        # A SECOND press: the read is repeated, the widget is not.
        app.action_open_fleet_asks()
        await _settle(pilot, 6)
        assert list(app.query(AskQueueList)) == [first]
        assert app._ask_scope == SCOPE_FLEET

        # `f4` puts the SESSION list up (two rows, so it is the LIST and not a
        # card); the note then swaps the SAME widget over to the fleet scope.
        app._sync_ask_surface(ask_rows([_row("a1"), _row("a2")]))
        await _settle(pilot)
        app.action_toggle_asks()
        await _settle(pilot, 4)
        session_listing = app.query_one(AskQueueList)
        assert session_listing._scope == SCOPE_SESSION
        app.action_open_fleet_asks()
        await _settle(pilot, 6)
        assert list(app.query(AskQueueList)) == [session_listing]
        assert session_listing._scope == SCOPE_FLEET
        assert app._ask_scope == SCOPE_FLEET
        assert app._ask_fleet_rows and app._ask_fleet_rows[0].ask_id == "b1"


async def test_a_second_fleet_gesture_is_dropped_and_the_first_engage_survives(
    enabled, isolated_index, monkeypatch
):
    """F2/U5: the guard belongs to the ask id, not to a widget the flow replaces.

    The engage window is the phone's own (30 s + 15 s). A second gesture must be
    dropped WITHOUT re-running the worker, because the group is `exclusive` and
    a re-run would cancel the engage that is already mid-dial.
    """
    hold = asyncio.Event()
    engages: list[Any] = []
    record: dict[str, Any] = {}

    async def _fake(config_dir, session_id, work, **kwargs):
        engages.append((str(session_id), getattr(work, "ask_id", None)))
        await hold.wait()
        return _FakeAttachClient(record), "detail"

    monkeypatch.setattr("local_operator.mobile.attach_client.engage_session_client", _fake)
    _seed(isolated_index, "s-other", [_row("b1")])
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app._ask_fleet_rows = ask_rows([_row("b1", session_id="s-other")])
        app._ask_scope = SCOPE_FLEET
        app._mount_ask_list(scope=SCOPE_FLEET)
        await _settle(pilot)
        listing = app.query_one(AskQueueList)
        row = listing.rows[0]

        app._run_fleet_ask_op(row, "respond", answers={"q1": ["Yes"]})
        await _settle(pilot, 3)
        assert row.ask_id in app._ask_in_flight
        assert listing.in_flight(row.ask_id) is True
        # THE FEEDBACK (U5): the row says it is being sent, in the widget the
        # user is looking at, for the whole window the card could not show it.
        assert IN_FLIGHT_MARK in listing.render().plain

        app._run_fleet_ask_op(row, "respond", answers={"q1": ["Yes"]})
        await _settle(pilot, 2)
        assert len(engages) == 1

        hold.set()
        await _settle(pilot, 8)
        assert len(engages) == 1
        assert record["respond"] == ("b1", {"q1": ["Yes"]}, "terminal")
        assert record["closed"] is True
        assert app._ask_in_flight == set()


async def test_answering_from_the_card_hands_the_user_the_list_in_flight(
    enabled, isolated_index, monkeypatch
):
    """U5: the card's own keys latch, so it cannot be the feedback surface."""
    hold = asyncio.Event()
    record: dict[str, Any] = {}

    async def _fake(config_dir, session_id, work, **kwargs):
        await hold.wait()
        return _FakeAttachClient(record), "detail"

    monkeypatch.setattr("local_operator.mobile.attach_client.engage_session_client", _fake)
    _seed(isolated_index, "s-other", [_row("b1")])
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app.action_open_fleet_asks()
        await _settle(pilot, 6)
        app.on_ask_queue_list_picked(AskQueueList.Picked("b1"))
        await _settle(pilot, 4)
        assert app._ask_card is not None, "the row's card must mount"
        row = app._ask_fleet_rows[0]
        app._ask_settle_callback(row)({"q1": ["Yes"]})
        await _settle(pilot, 3)
        assert app._ask_card is None
        listing = app.query_one(AskQueueList)
        assert listing.in_flight("b1") is True
        assert IN_FLIGHT_MARK in listing.render().plain
        hold.set()
        await _settle(pilot, 8)
        assert record["respond"] == ("b1", {"q1": ["Yes"]}, "terminal")


async def test_the_kill_switch_clears_marks_that_were_already_painted(
    monkeypatch, enabled, isolated_index
):
    """F4: the flag-off branch returned BEFORE the push, so paint survived it."""
    _seed(isolated_index, "s-other", [_row("b1")])
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        sidebar = app._session_sidebar
        # PAINT first, through the app's own painter — the frame a user could
        # have been looking at when the flag went off.
        app._ask_marks = {"s-other": 1}
        app._paint_sidebar_asks()
        await _settle(pilot)
        assert sidebar._asking == {"s-other": 1}
        assert sidebar._asks_total == 1
        monkeypatch.setattr(policy, "NONBLOCKING_ASK", False)
        app._paint_sidebar_asks()
        await _settle(pilot)
        assert sidebar._asking == {}
        assert sidebar._asks_total == 0
        assert sidebar._asks_note() == ""


async def test_a_session_at_the_projection_cap_withholds_the_split(enabled, isolated_index):
    """F6: the one place a backend tally exists is the index this list reads."""
    _seed(isolated_index, "s-big", [_row(f"c{i}") for i in range(policy.PROJECTION_CAP)])
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app.action_open_fleet_asks()
        await _settle(pilot, 6)
        listing = app.query_one(AskQueueList)
        assert app._ask_fleet_truncated is True
        assert app._ask_fleet_open_count == policy.PROJECTION_CAP
        painted = listing.header_text(200)
        assert f"{policy.PROJECTION_CAP} outstanding" in painted
        # The CLAUSE withholds the split; the three segments keep their live
        # counts, which is what makes the three-way control a control (ask 3).
        clause = painted.split("   All ·")[0]
        assert "moved on" not in clause and "settled" not in clause

    # ...and a queue BELOW the cap still states its split. Checked on the
    # helper with a synthetic read: the seeded root above is at the cap, so a
    # second app boot in the same root would inherit it.
    app._refresh_fleet_count_facts(
        [
            {"session_id": "s1", "status": "open"},
            {"session_id": "s1", "status": "open"},
        ]
    )
    assert app._ask_fleet_truncated is False
    assert app._ask_fleet_open_count == 2


async def test_a_fleet_row_with_no_session_id_fails_loudly_not_sideways(
    enabled, isolated_index, monkeypatch
):
    """F10: an empty id is not a licence to answer whoever is on screen."""
    record: dict[str, Any] = {}
    _patch_engage(monkeypatch, record, raise_on={"respond"})
    session = _RecordingSession()
    app = _app(session)
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app._ask_scope = SCOPE_FLEET
        orphan = ask_rows([_row("b1")])[0]
        assert orphan.session_id == ""
        assert app._fleet_answer_starts_here(orphan) is True
        seen: list[str] = []
        monkeypatch.setattr(app, "_ask_gesture_refusal", lambda text: seen.append(text))
        await app._fleet_ask_worker(orphan, "respond", {"q1": ["Yes"]})
        # The refusal is stated, and the ON-SCREEN session answers nothing.
        assert seen
        assert session.answered == []


async def test_a_fleet_row_names_the_conversation_the_sidebar_names(
    enabled, isolated_index, monkeypatch
):
    """U8/U6: the catalogue's title, not the cwd's last segment."""
    _seed(isolated_index, "s-other", [_row("b1")])

    class _Entry:
        id = "s-other"
        subagent = False

        class row:  # noqa: N801 — mirrors the catalogue's own attribute
            name = "Enrichment backfill review"

    monkeypatch.setattr(
        "local_operator.tui.session_catalog.load_catalog", lambda *a, **k: [_Entry()]
    )
    app = _app()
    async with app.run_test(size=(130, 30)) as pilot:
        await _settle(pilot)
        app.action_open_fleet_asks()
        await _settle(pilot, 6)
        listing = app.query_one(AskQueueList)
        painted = listing.render().plain
        # Clipped to the handle budget by design: a title is a sentence and
        # this is a tail.
        assert "Enrichment backfill rev" in painted
        assert "s-other" not in painted
