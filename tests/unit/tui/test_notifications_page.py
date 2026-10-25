"""The TUI ``/notifications`` page (lane L2): model rules, rung 3, in-view reads, keys.

Every store here is a real ``NotificationStore`` under ``tmp_path`` with synthetic
session ids; the page tests drive the real ``OperatorApp`` (stylesheet loaded)
through ``run_test``. What each test pins is the RULE the design states, so a
refactor that keeps the pixels but loses the rule fails here.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.notifications import items as inbox
from local_operator.notifications.store import NotificationStore
from local_operator.tui import notifications_attention as attention
from local_operator.tui import notifications_model as model
from local_operator.tui import notifications_render as render

S1, S2, S3, S4, S5 = (f"{n:012x}" for n in range(0xA1, 0xA6))


def _seed(root: Path, count: int) -> NotificationStore:
    store = NotificationStore(root)
    # The store caps 30 NEW rows per session per hour; spread across five.
    for index in range(count):
        store.notify((S1, S2, S3, S4, S5)[index % 5], title=f"row {index}", key=f"k{index}")
    return store


def _row(store: NotificationStore, item_id: str) -> dict[str, Any]:
    """The stored row, asserted present: a missing row is a failure, not a None."""
    row = store.get(item_id)
    assert row is not None, f"{item_id} is not in the store"
    return row


def _page(app: Any) -> Any:
    """The mounted page, asserted present before a test reads its state."""
    view = app._notifications_view
    assert view is not None, "/notifications did not mount the page"
    return view


def _open_ask(root: Path, session_id: str, ask_id: str = "ask_1") -> None:
    from local_operator.asks import store as ask_store

    (root / "sessions" / session_id).mkdir(parents=True, exist_ok=True)
    ask_store.write_entry(
        root,
        session_id,
        cwd="/tmp/work",
        asks=[
            {
                "ask_id": ask_id,
                "created_at": int(time.time() * 1000),
                "expires_at": 0,
                "timeout_s": 0,
                "urgent": False,
                "status": "open",
                "delivered": True,
                "questions": [{"question": "Ship it?", "header": "Ship"}],
            }
        ],
    )


# -- model: bulk gestures never sweep ------------------------------------------


def test_mark_all_enumerates_every_page_and_never_reads_a_later_write(tmp_path: Path) -> None:
    store = _seed(tmp_path, 130)
    painted = model.load(tmp_path, model.Filters())
    assert len(painted.items) == inbox.LIST_LIMIT_MAX and not painted.complete
    through, observed, _ = model.enumerate_scope(tmp_path, model.Filters())
    assert len(observed) == 130, "all means the whole scope, not the painted page"
    late = store.notify(S1, title="arrived after the confirm was armed")
    result = model.write(tmp_path, "read", scope=(through, observed))
    assert len(result.changed) == 130
    assert _row(store, late.item_id)["read_rev"] == 0, "a later write must stay unread"
    assert result.unread_count == 1


def test_batches_never_exceed_the_route_cap(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path, 120)
    sizes: list[int] = []
    real = inbox.apply_patch

    def spy(root, op, **kwargs):
        if kwargs.get("observed") is not None:
            sizes.append(len(kwargs["observed"]))
        return real(root, op, **kwargs)

    monkeypatch.setattr(model.inbox, "apply_patch", spy)
    monkeypatch.setattr(model, "BULK_BATCH", 50)
    through, observed, _ = model.enumerate_scope(tmp_path, model.Filters())
    model.write(tmp_path, "read", scope=(through, observed))
    assert sizes == [50, 50, 20]


def test_undo_restores_exactly_what_the_write_changed(tmp_path: Path) -> None:
    store = _seed(tmp_path, 4)
    through, observed, _ = model.enumerate_scope(tmp_path, model.Filters(), session_id=S1)
    dismissed = model.write(tmp_path, "dismiss", scope=(through, observed), session_id=S1)
    assert dismissed.changed and dismissed.inverse == "undismiss"
    restored = model.undo(tmp_path, dismissed)
    assert sorted(restored.changed) == sorted(dismissed.changed)
    for item_id in dismissed.changed:
        assert _row(store, item_id)["dismissed_at"] is None


def test_dismiss_from_a_conversation_reports_open_asks_that_stay(tmp_path: Path) -> None:
    _open_ask(tmp_path, S1)
    _seed(tmp_path, 3)
    log = tmp_path / "asks" / f"{S1}.json"
    before = log.read_bytes()
    through, observed, _ = model.enumerate_scope(tmp_path, model.Filters(), session_id=S1)
    assert any(token["id"].startswith("ask:") for token in observed)
    result = model.write(tmp_path, "dismiss", scope=(through, observed), session_id=S1)
    assert result.skipped_live_asks == 1
    assert log.read_bytes() == before, "the page never writes the ask log"


def test_a_filtered_gesture_never_widens(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    high = store.notify(S1, title="high", priority="high")
    store.notify(S2, title="normal")
    _, observed, _ = model.enumerate_scope(tmp_path, model.Filters(priority="high"))
    assert [token["id"] for token in observed] == [high.item_id]


# -- render: columns hold at every width --------------------------------------


@pytest.mark.parametrize("width", [40, 76, 120])
def test_a_row_is_exactly_its_width(width: int) -> None:
    row = {
        "id": "n_x",
        "source": "notify",
        "priority": "high",
        "title": "Free-credit abuse: 14 sign-ups from one /24 in ten minutes",
        "repeat_count": 6,
        "read": False,
        "session": {"id": S1, "title": "radient fraud watch"},
        "updated_at": int(time.time() * 1000),
    }
    line = render.row_line(row, width, cursor=True, selected=False)
    assert render.cell_len(line.plain) == width


def test_hints_drop_whole_and_never_drop_esc() -> None:
    labels = [("↵", "open"), ("e", "dismiss"), ("space", "select"), ("esc", "back")]
    kept = render.hint_row(labels, 20)
    assert ("esc", "back") in kept
    assert all(entry in labels for entry in kept)


def test_the_sidebar_note_is_hidden_at_zero_and_has_no_emoji() -> None:
    from local_operator.tui.widgets.session_sidebar import SessionSidebar

    sidebar = SessionSidebar()
    assert sidebar._notifications_note() == ""
    sidebar._notifications_total = 3
    note = sidebar._notifications_note()
    assert note == "notifications 3"
    assert note.isascii()


# -- rung 3: claim first, high only, release on a failed spawn ------------------


def test_rung_three_raises_high_once_and_never_normal_or_low(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    high = store.notify(S1, title="high", priority="high")
    store.notify(S1, title="normal", priority="normal")
    store.notify(S1, title="low", priority="low")
    raised: list[str] = []
    for _ in range(2):  # a second TUI process on the same revision
        attention.announce_items(
            tmp_path,
            raise_one=lambda row: raised.append(str(row["id"])) or True,
            raise_digest=lambda rows: True,
            desktop_delivers_update=lambda: False,
        )
    assert raised == [high.item_id]


def test_rung_three_digests_the_overflow_and_releases_a_failed_digest(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    for index in range(5):
        store.notify((S1, S2, S3, S4, S5)[index], title=f"h{index}", priority="high")
    tick = attention.announce_items(
        tmp_path,
        raise_one=lambda row: True,
        raise_digest=lambda rows: False,
        desktop_delivers_update=lambda: False,
    )
    assert len(tick.raised) == attention.BURST_LIMIT
    assert len(tick.released) == 1, "the digest stood for one row and failed: hand it back"
    owed = {row["id"] for row in store.undelivered(priority="high")}
    # The released row is owed again; so is the fifth, which this tick never
    # claimed (one past the cap is read, the rest drain on later ticks).
    assert set(tick.released) <= owed and len(owed) == 2


def test_rung_three_stands_down_for_a_desktop_that_delivers_update(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    store.notify(S1, title="high", priority="high")
    tick = attention.announce_items(
        tmp_path,
        raise_one=lambda row: pytest.fail("the feed owns this raise"),
        raise_digest=lambda rows: True,
        desktop_delivers_update=lambda: True,
    )
    assert tick.stood_down
    assert store.undelivered(priority="high"), "no claim was taken"


def test_a_failed_spawn_hands_the_claim_back(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    item = store.notify(S1, title="high", priority="high")
    tick = attention.announce_items(
        tmp_path,
        raise_one=lambda row: False,
        raise_digest=lambda rows: True,
        desktop_delivers_update=lambda: False,
    )
    assert tick.released == (item.item_id,)
    assert [row["id"] for row in store.undelivered(priority="high")] == [item.item_id]


# -- in-view receipts: only real evidence -------------------------------------


def _visible_then_press(store: NotificationStore, anchor: str) -> tuple[Any, int, int]:
    item = store.notify(S1, title="flagged", anchor_id=anchor)
    time.sleep(0.01)
    visible = int(time.time() * 1000)
    time.sleep(0.01)
    press = int(time.time() * 1000)
    time.sleep(0.005)
    return item, visible, press


def test_a_card_on_screen_then_a_press_reads_it_in_view(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    item, visible, press = _visible_then_press(store, "tc1")
    rows = attention.unread_for_session(tmp_path, S1)
    items, since = attention.seen_candidates(
        rows, [attention.VisibleCard("tc1", visible)], session_id=S1, input_at_ms=press
    )
    result = attention.submit_seen(
        tmp_path, items, session_id=S1, input_at_ms=press, visible_since_ms=since
    )
    assert result is not None and result["changed"] == [item.item_id]
    assert _row(store, item.item_id)["read_via"] == "in_view"


def test_negatives_stay_unread(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    item, visible, press = _visible_then_press(store, "tc1")
    rows = attention.unread_for_session(tmp_path, S1)
    # Scrolled off: no visible card.
    assert attention.seen_candidates(rows, [], session_id=S1, input_at_ms=press)[0] == []
    # Another session focused.
    other = attention.seen_candidates(
        rows, [attention.VisibleCard("tc1", visible)], session_id=S2, input_at_ms=press
    )
    assert other[0] == []
    # A press that landed before the card rendered.
    early = attention.seen_candidates(
        rows, [attention.VisibleCard("tc1", press + 10)], session_id=S1, input_at_ms=press
    )
    assert early[0] == []
    # A new revision after the evidence: the old token is stale at the store.
    store.notify(S1, title="flagged v2", anchor_id="tc1", key="")
    result = attention.submit_seen(
        tmp_path,
        [{"id": item.item_id, "revision": 0, "anchor_id": "tc1"}],
        session_id=S1,
        input_at_ms=press,
        visible_since_ms=visible,
    )
    assert result is not None and result["changed"] == []
    assert _row(store, item.item_id)["read_rev"] == 0


# -- the real page ------------------------------------------------------------


def _app():
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    return OperatorApp(lambda: _factory(FakeSession()))


async def _settle(pilot, ticks: int = 6) -> None:
    for _ in range(ticks):
        await pilot.pause()
        await asyncio.sleep(0.03)


@pytest.mark.asyncio
async def test_opening_the_page_reads_nothing_and_mark_all_asks_first(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    store = _seed(tmp_path, 3)
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        app._cmd_notifications_page("", app._system_notice)
        await _settle(pilot)
        view = _page(app)
        assert len(view.rows) == 3
        assert store.counts()[0] == 3, "opening the page is not reading"
        await pilot.press("A")
        await _settle(pilot)
        assert view.confirming
        await pilot.press("n")
        await _settle(pilot)
        assert store.counts()[0] == 3
        await pilot.press("A", "y")
        await _settle(pilot, 10)
        assert store.counts()[0] == 0
        await pilot.press("z")
        await _settle(pilot, 10)
        assert store.counts()[0] == 3, "z undoes the mark inside the window"


@pytest.mark.asyncio
async def test_notifications_read_opens_the_page_with_the_confirm_armed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    store = _seed(tmp_path, 2)
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        app._cmd_notifications_page("read", app._system_notice)
        await _settle(pilot, 10)
        view = _page(app)
        assert view.confirming
        assert store.counts()[0] == 2, "the typed form asks before it writes"


@pytest.mark.asyncio
async def test_e_on_an_open_ask_never_dismisses_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _open_ask(tmp_path, S1)
    log = tmp_path / "asks" / f"{S1}.json"
    before = log.read_bytes()
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        app._cmd_notifications_page("", app._system_notice)
        await _settle(pilot)
        view = _page(app)
        assert view.current and view.current["source"] == "ask"
        await pilot.press("e")
        await _settle(pilot)
        assert "answer" in view.notice.lower()
        assert log.read_bytes() == before


@pytest.mark.asyncio
async def test_esc_leaves_and_restores_the_transcript(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        app._cmd_notifications_page("", app._system_notice)
        await _settle(pilot)
        assert not app._transcript_view().display
        await pilot.press("escape")
        await _settle(pilot)
        assert app._notifications_view is None
        assert app._transcript_view().display


def test_a_stray_argument_points_at_unread() -> None:
    notices: list[str] = []

    class _Stub:
        _notifications_view = None

    from local_operator.tui.app import OperatorApp

    OperatorApp._cmd_notifications_page(_Stub(), "list", notices.append)  # type: ignore[arg-type]
    assert notices and "/unread" in notices[0]


@pytest.mark.asyncio
async def test_the_narrow_detail_keeps_its_row_and_offers_unread_after_opening(
    tmp_path: Path, monkeypatch
) -> None:
    """``↵`` marks the row read and, on Unread, it leaves the list; the detail
    must still show THAT row (not its successor) and offer ``u unread``."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    store = _seed(tmp_path, 3)
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        app._cmd_notifications_page("", app._system_notice)
        await _settle(pilot)
        view = _page(app)
        opened = view.current
        await pilot.press("enter")
        await _settle(pilot, 12)
        assert view.view_mode == "detail"
        assert view.detail_row is not None and view.detail_row["id"] == opened["id"]
        assert _row(store, opened["id"])["read_rev"] == 1
        assert opened["id"] not in [str(r["id"]) for r in view.rows], "left the Unread list"
        assert ("u", "unread") in [(k, v) for k, v in view._hint_labels()]
