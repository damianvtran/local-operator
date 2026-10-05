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
    AskQueueList,
    ask_rows,
    drawer_headline,
    empty_sentence,
    filter_counts,
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
                "options": [],
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
    # All settled: the chip register (which counts what is owed) has no words,
    # and the drawer still owes a count of what it is showing.
    assert drawer_headline(ask_rows([_row("a1", status="declined")])) == "1 settled"
    assert drawer_headline(mixed, open_count=9, truncated=True) == "9 outstanding"


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
