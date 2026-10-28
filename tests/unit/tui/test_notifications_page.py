"""The TUI ``/notifications`` page (lane L2): model rules, rung 3, in-view reads, keys.

Every store here is a real ``NotificationStore`` under ``tmp_path`` with synthetic
session ids; the page tests drive the real ``OperatorApp`` (stylesheet loaded)
through ``run_test``. What each test pins is the RULE the design states, so a
refactor that keeps the pixels but loses the rule fails here.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest
from rich.cells import cell_len
from textual.worker import WorkerState

from local_operator.notifications import items as inbox
from local_operator.notifications.store import NotificationStore
from local_operator.tui import notifications_attention as attention
from local_operator.tui import notifications_model as model
from local_operator.tui import notifications_render as render
from tests.unit.tui.conftest import BOOT_DEADLOCK_GUARD_S

S1, S2, S3, S4, S5 = (f"{n:012x}" for n in range(0xA1, 0xA6))


def _age(root: Path, *, ms: int | None = None) -> None:
    """Move every row's content stamp past ``BANNER_GRACE_MS`` (T2).

    Rung 3 HOLDS a fresh revision so an in-view receipt can cancel it, exactly
    as the desktop feed does; a test that writes and then asserts a raise would
    be measuring the hold. Ageing is how a real elapsed grace is simulated
    without sleeping 2.5 s per case.
    """
    import sqlite3

    from local_operator.notifications.delivery import BANNER_GRACE_MS

    stamp = int(time.time() * 1000) - (BANNER_GRACE_MS if ms is None else ms) - 1
    conn = sqlite3.connect(root / "notifications.db")
    try:
        with conn:
            conn.execute("UPDATE items SET updated_at=?", (stamp,))
    finally:
        conn.close()


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


async def _tick_items(app: Any, pilot: Any, *, timeout_s: float = 5.0) -> None:
    """One real `_notify_inbox_items` pass, WAITED OUT.

    The pass runs in a worker (`asyncio.to_thread`), so a fixed number of
    `pause()` calls is a race: the audit's own reproduction showed passes that
    never ran, which is indistinguishable from the latch bug under test. This
    waits for the latch to clear, which is the observable end of the pass.
    """
    app._notify_inbox_items()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await pilot.pause()
        await asyncio.sleep(0.02)
        if not getattr(app, "_inbox_notify_pending", False):
            # One more pause so the worker's own finally-block state settles.
            await pilot.pause()
            return


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
    _age(tmp_path)
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
    _age(tmp_path)
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
    _age(tmp_path)
    tick = attention.announce_items(
        tmp_path,
        raise_one=lambda row: False,
        raise_digest=lambda rows: True,
        desktop_delivers_update=lambda: False,
    )
    assert tick.released == (item.item_id,)
    assert [row["id"] for row in store.undelivered(priority="high")] == [item.item_id]


def test_rung_three_banners_respect_the_privacy_flag(tmp_path: Path, monkeypatch) -> None:
    """T1: with session names off, no model-written text reaches a banner.

    The rung-3 leg handed ``row["title"]``/``body_md`` straight to the detached
    notifier while rungs 2 and 4 sent the brand and the house sentence, so a
    lock screen showed the agent's subject line on exactly the surface the flag
    exists to protect. Driven through the REAL chain: the settings read, the
    compose-side flag, and the app's own method.
    """
    from local_operator.tui import notify as tui_notify
    from local_operator.tui.app import OperatorApp

    monkeypatch.setattr(tui_notify, "settings_get", lambda key, default=None: False)
    captured: list[tuple[str, str]] = []

    def fake_detached(title: str, body: str, *, session_id: str = "") -> int:
        captured.append((title, body))
        return 1

    monkeypatch.setattr(tui_notify, "detached_notify", fake_detached)
    sent = OperatorApp._raise_inbox_banner(
        cast(Any, object()),  # the method reads nothing off self
        {"title": "Secret campaign numbers", "body_md": "revenue 42", "session_id": S1},
    )
    assert sent is True
    assert captured == [(tui_notify.APP_NAME, tui_notify.BODY_UPDATE)]
    assert "Secret" not in captured[0][0] and "42" not in captured[0][1]
    # And the flag ON still shows the item's own words, so the test is not
    # merely asserting that the leg always hides.
    monkeypatch.setattr(tui_notify, "settings_get", lambda key, default=None: True)
    captured.clear()
    OperatorApp._raise_inbox_banner(
        cast(Any, object()),
        {"title": "PR #9 needs review", "body_md": "two files", "session_id": S1},
    )
    assert captured and captured[0][0] == "PR #9 needs review", captured


@pytest.mark.asyncio
async def test_a_focus_report_alone_never_reads_an_item(tmp_path: Path, monkeypatch) -> None:
    """D12 on the TUI: focus is not an input edge.

    ``_attention_input_at`` is stamped only by real Key/MouseDown events
    (``OperatorApp.on_event``), never by a focus report, so a window that simply
    gains focus must not acknowledge anything. This pins the guard: with focus
    observed and a card on screen but NO input edge, the receipt path returns
    before it can write.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    store = NotificationStore(tmp_path)
    item = store.notify(S1, title="on screen", anchor_id="call_focus")
    app = _app()

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a focus report alone must never submit a receipt")

    monkeypatch.setattr(attention, "submit_seen", refuse)
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        # The app reads the id off its live session; the property is
        # overridden on the class for this test (monkeypatch undoes it), so the
        # real object keeps every other member the app's teardown needs.
        monkeypatch.setattr(
            type(app._session), "session_id", property(lambda self: S1), raising=False
        )
        app._attention_focus_observed = True
        app._attention_input_at = 0.0
        monkeypatch.setattr(
            app,
            "_visible_notify_cards",
            lambda: [attention.VisibleCard(anchor_id="call_focus", visible_since_ms=1)],
        )
        app._receipt_inbox_items_in_view()
        await _settle(pilot)
    row = store.get(item.item_id)
    assert row is not None and row["read_rev"] == 0 and row["read_via"] is None


@pytest.mark.asyncio
async def test_the_sidebar_note_counts_the_badge_and_hides_when_unknown(
    tmp_path: Path, monkeypatch
) -> None:
    """D16: ``unread + waiting_unread``, and unknown paints nothing."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        painted: list[int] = []

        # Record on the REAL sidebar rather than swapping in a stand-in object.
        # The app keeps handling messages while this test body runs, and on the
        # way out of `run_test`: a queued `WelcomeView` resize runs
        # `_sync_boot_layout` -> `_boot_lane_width`, which reads
        # `self._session_sidebar.styles.width`. A bare stand-in had no `styles`,
        # so a resize landing after the swap raised `AttributeError` (CI run
        # 40555547851). Production never holds anything but a mounted
        # `SessionSidebar` there, so the stand-in was the defect, not the read.
        # Record, then CALL THROUGH (review R3), so the sidebar's own clamp and
        # store are exercised rather than stubbed out.
        sidebar = app._session_sidebar
        original = sidebar.set_notifications_total

        def record(total: int) -> None:
            painted.append(total)
            original(total)

        monkeypatch.setattr(sidebar, "set_notifications_total", record)
        app._paint_notifications_total({"unread_count": 2, "waiting_unread_count": 1})
        assert painted == [3], "the badge's own figure"
        assert sidebar._notifications_total == 3
        app._paint_notifications_total(None)
        assert painted == [3, 0], "an unreadable store paints nothing, not a stale count"
        assert sidebar._notifications_total == 0


@pytest.mark.asyncio
async def test_the_wide_pane_takes_the_keys_and_esc_returns_them(
    tmp_path: Path, monkeypatch
) -> None:
    """DT3: at >=120 columns the pane was never focusable, so it never scrolled.

    `↵` on an already-opened row hands the KEYS to the pane; `esc` steps back out
    to the list with its cursor untouched. The footer narrows to the pane's own
    gesture meanwhile, and the pane carries the ▸ marker.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    store = NotificationStore(tmp_path)
    item = store.notify(S1, title="a tall one", body_md="x", priority="high")
    # A SECOND row: opening the first marks it read, and on the default Unread
    # filter a lone row leaves the list — `current` then answers None, which is a
    # property of this fixture rather than of the pane.
    store.notify(S2, title="still unread", priority="normal")
    app = _app()
    async with app.run_test(size=(140, 32)) as pilot:
        await _settle(pilot)
        app._open_notifications_view()
        await _page_idle(app, pilot)
        view = _page(app)
        assert view.wide, "the wide layout is the one under test"
        await pilot.press("enter")
        await _page_idle(app, pilot)
        await pilot.press("enter")  # second ↵: hand the keys to the pane
        await _page_idle(app, pilot)
        assert view._detail_focus is True
        # D2 superseded the round-1 rule that this row narrowed to `esc` alone:
        # the pane keeps acting on its item, so the row must NAME those keys.
        keys = [key for key, _ in view._hint_labels()]
        assert keys[-1] == "esc", keys
        for key in ("o", "a", "e", "u"):
            assert key in keys, (key, keys)
        await pilot.press("escape")
        await _settle(pilot, 6)
        assert view._detail_focus is False, "esc returns the keys to the list"
        assert str(view.current["id"]) == item.item_id, "and the cursor did not move"


@pytest.mark.asyncio
async def test_a_tall_detail_pane_scrolls_under_the_keyboard(tmp_path: Path, monkeypatch) -> None:
    """DT3: the pane's own geometry, measured.

    A detail taller than its box must report a virtual height above its outer
    size, take the keys on the second `↵`, and actually move: `pagedown` changes
    the offset and `end` reaches `max_scroll_y`. Measured on this fixture at
    140x32: outer 60x19, virtual 59x41, offset 0 -> 18 -> 22 (max 22).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    store = NotificationStore(tmp_path)
    body = "\n".join(f"Paragraph {index}: " + ("detail " * 14) for index in range(12))
    item = store.notify(S1, title="a tall one", body_md=body, priority="high")
    store.notify(S2, title="second", priority="normal")
    app = _app()
    async with app.run_test(size=(140, 32)) as pilot:
        await _settle(pilot)
        app._open_notifications_view()
        await _page_idle(app, pilot)
        view = _page(app)
        assert view.wide
        rows = [str(row["id"]) for row in view.rows]
        view._move_to(rows.index(item.item_id))
        await _page_idle(app, pilot)
        await pilot.press("enter")
        # The payload arrives on its own worker: wait for the REAL body, or the
        # pane is measured against the short placeholder and the numbers lie.
        # (A 30-turn poll on the geometry used to stand here; it was the same
        # fixed-turn bet as `_settle`, merely a longer one.)
        await _page_idle(app, pilot)
        outer = view._detail.size.height
        virtual = view._detail.virtual_size.height
        assert virtual > outer, (outer, virtual)
        await pilot.press("enter")  # second ↵: the keys move to the pane
        await _page_idle(app, pilot)
        assert view._detail_focus is True
        cursor_after_open = str(view.current["id"]) if view.current else ""
        start = view._detail.scroll_offset.y
        await pilot.press("pagedown")
        await _settle(pilot, 6)
        after_page = view._detail.scroll_offset.y
        assert after_page > start, (start, after_page)
        await pilot.press("end")
        await _settle(pilot, 6)
        assert view._detail.scroll_offset.y == view._detail.max_scroll_y, (
            view._detail.scroll_offset.y,
            view._detail.max_scroll_y,
        )
        # Paging moved the PANE: the list cursor is where the open left it.
        assert (str(view.current["id"]) if view.current else "") == cursor_after_open


def test_the_confirm_names_rows_and_revisions_separately() -> None:
    """DT5a: `×6` on the row and "1 update" in the confirm meant different things."""
    from local_operator.i18n import keys as i18n_keys
    from local_operator.tui.i18n_status import display_tr

    translate = display_tr("tui.notifications")
    line = translate(
        i18n_keys.tui_notifications.confirm_dismiss_session(
            count=1, updates=6, source="radient fraud watch"
        )
    )
    assert "1 notification" in line, line
    assert "6 updates" in line, line


def test_the_armed_footer_names_its_keys_and_mark_all_has_a_real_key() -> None:
    """DT5b + DT2: y/n in the armed state, and `A` labelled by the namespace."""
    from local_operator.i18n import keys as i18n_keys

    ns = i18n_keys.tui_notifications
    assert ns.hint_confirm() and ns.hint_cancel() and ns.hint_mark_all()


@pytest.mark.asyncio
async def test_the_retry_countdown_never_arms_the_banner_latch(tmp_path: Path, monkeypatch) -> None:
    """R2-1: the countdown must not leave `_inbox_notify_pending` set.

    Round 1 armed the latch BEFORE the countdown returned, so ONE barren pass
    silenced this surface for the life of the process (the auditor measured 140
    consecutive ticks with nothing raised, and independently 70 ticks with no
    worker at all). This drives the real `_notify_inbox_items` through the
    countdown and asserts the latch is clear afterwards and the countdown moved.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # The kill switch is not what this test measures, and the harness's own
    # settings can turn it off; pin it ON so the countdown path is reachable.
    from local_operator.tui import notify as tui_notify

    monkeypatch.setattr(tui_notify, "notifications_enabled", lambda: True)
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        # The app's own 1 s poll also drives this leg; wait for any pass already
        # in flight, or the call under test returns at the latch and proves
        # nothing.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and app._inbox_notify_pending:
            await pilot.pause()
            await asyncio.sleep(0.02)
        assert app._inbox_notify_pending is False
        app._inbox_notify_retry_ticks = 3
        app._notify_inbox_items()
        assert app._inbox_notify_retry_ticks == 2, "the countdown ran"
        assert app._inbox_notify_pending is False, "and the latch was never armed"
        # Drain it, then a pass is allowed again — the leg is not stuck.
        app._notify_inbox_items()
        app._notify_inbox_items()
        assert app._inbox_notify_retry_ticks == 0
        assert app._inbox_notify_pending is False


def test_the_loading_header_invents_no_counts() -> None:
    """DT1: an unloaded snapshot is not an empty one.

    ``snapshot is None`` reaches the header as a placeholder whose counts are all
    zero, so the frame claimed "0 unread · 0 waiting on you" at the one moment the
    operator might decide whether to wait for a contended store.
    """
    placeholder = model.Snapshot(loaded=False)
    line = render.header_line(placeholder, model.Filters(), 100).plain
    assert "0 unread" not in line and "waiting on you" not in line, line
    assert "Notifications" in line
    # The control: a LOADED empty snapshot still states its real zero.
    loaded = render.header_line(model.Snapshot(), model.Filters(), 100).plain
    assert "0 unread" in loaded, loaded


def test_the_volume_keys_survive_the_eighty_column_footer() -> None:
    """DT2: `A` was in no hint row, and `E` was dropped at the default size.

    The ladder dropped widest-first, so the gestures requirement 6 exists for
    went before the ones the detail and the header restate.
    """
    labels = [
        ("↵", "open"),
        ("e", "dismiss"),
        ("E", "dismiss all"),
        ("space", "select"),
        ("A", "read all"),
        ("u", "read"),
        ("o", "conversation"),
        ("f", "filter"),
        ("esc", "back"),
    ]
    kept = [key for key, _ in render.hint_row(labels, 79)]
    for key in ("esc", "e", "E", "space", "A"):
        assert key in kept, (key, kept)
    for key in ("o", "f"):
        assert key not in kept, kept


def test_an_ask_pane_names_its_expiry_projects_and_gesture(tmp_path: Path) -> None:
    """DT4 + D3: the ask detail names its expiry and does not repeat its question.

    D3: the fixture is the REAL serialiser's output, not a hand-built dict. The
    hand-built shape carried a top-level ``expires_at`` that ``ask_item`` never
    emits — the deadline is nested at ``ask.deadline_at`` — so the test passed
    while every real ask pane omitted its expiry.
    """
    from local_operator.notifications.items import ProjectFacts, SessionFacts, ask_item

    # The facts objects are the PRODUCER's own, over an isolated config dir —
    # the repo's convention for a shape test (test_inbox.py's live_state cell) —
    # never hand-built dicts, which is how the old fixture drifted from the
    # producer in the first place.
    row = ask_item(
        {
            "ask_id": "ask_1",
            "session_id": "s1",
            "status": "open",
            "created_at": 1_000,
            "expires_at": 1_000 + 37 * 60 * 1000,
            "questions": [{"question": "Block the /24 for 24 hours, or keep watching?"}],
        },
        {},
        SessionFacts(tmp_path),
        ProjectFacts(tmp_path),
    )
    assert row["ask"]["deadline_at"] == 1_000 + 37 * 60 * 1000, "the producer nests the deadline"
    assert "expires_at" not in row, "the producer does not put the deadline at the top level"
    text = render.detail_text(
        None, row, 75, position=(1, 1), now_ms=1_000 + 60 * 60 * 1000, show_back=False
    ).plain
    assert "expires" in text
    assert "a ↵" in text
    # The question appears ONCE (as the title), not again as a body.
    assert text.count("keep watching?") == 1, text


def test_older_rows_name_their_day_and_the_history_names_what_changed() -> None:
    """DT7 + DT8: a bare clock time under "Older", and five identical lines."""
    now = 1_700_000_000_000
    day = 24 * 3600 * 1000
    assert "/" in render.row_time(now - 5 * day, now), "a date, not a bare time"
    assert ":" in render.row_time(now - 30 * 3600 * 1000, now), "yesterday keeps HH:MM"

    row = {
        "id": "n_1",
        "source": "notify",
        "priority": "high",
        "title": "14 sign-ups, 12 blocked",
        "body_md": "",
        "created_at": now,
        "updated_at": now,
        "revision": 6,
        "repeat_count": 6,
        "session": {"id": "s1"},
    }
    history = [
        {
            "revision": r,
            "at": now - r * 1000,
            "title": f"{r} sign-ups, {r - 2} blocked",
            "excerpt": "Fourteen accounts signed up from 203.0.113.0/24 between 10:12 and 10:40",
        }
        for r in (5, 4, 3, 2, 1)
    ]
    text = render.detail_text(
        {"item": row, "history": history, "asks": [], "projects": []},
        row,
        90,
        position=(1, 6),
        now_ms=now,
        show_back=False,
    ).plain
    assert "5 sign-ups, 3 blocked" in text and "4 sign-ups, 2 blocked" in text, text


def test_the_seen_chip_and_the_low_bars_stay_readable() -> None:
    """DT6 + DT12: `dim` fails AA in 30 of 54 themes, and a lone bar read as `_`."""
    assert render.BARS["low"] == "▁▁▁", render.BARS
    assert all(render.cell_len(bar) == 3 for bar in render.BARS.values()), render.BARS
    row = {
        "id": "n_2",
        "source": "notify",
        "priority": "normal",
        "title": "x",
        "read": True,
        "read_via": "watching",
        "created_at": 1,
        "updated_at": 1,
        "session": {"id": "s1"},
        "seq": 1,
        "revision": 1,
    }
    line = render.row_line(row, 80, cursor=False, selected=False, now_ms=2)
    inks = {str(span.style) for span in line.spans}
    assert not any("dim" in ink for ink in inks if ink), inks


def test_the_header_states_no_counts_for_an_errored_snapshot() -> None:
    """T6/D16: unknown renders nothing, never a zero.

    The page's own rule (stated in its empty/error states) is that absence is
    not emptiness. A header reading "0 on you" above "database disk image is
    malformed" is that rule contradicted one row up.
    """
    failed = model.Snapshot(error="database disk image is malformed")
    line = render.header_line(failed, model.Filters(), 100).plain
    assert "0" not in line, line
    assert "Notification" in line or line.strip(), line
    healthy = model.Snapshot(unread_count=2, waiting_count=1, waiting_unread_count=1)
    wide = render.header_line(healthy, model.Filters(), 100).plain
    # D16's formula: the badge's own `unread + waiting_unread`, and the waiting
    # GROUP beside it — 2 unread + 1 unread ask reads "3 unread · 1 waiting".
    assert "3 unread" in wide and "1 waiting" in wide, wide


def test_the_doorbell_rings_for_a_new_open_ask(tmp_path: Path) -> None:
    """T7: the ask index is part of the page's change detector.

    Open asks live in ``asks/<session>.json``, not in the inbox store, so before
    this an ask raised while the page was open stayed invisible until ``r`` —
    and an ask is the one category that BLOCKS an agent.
    """
    from local_operator.tui.widgets.notifications_view import _store_stamp

    NotificationStore(tmp_path).notify(S1, title="one", priority="high")
    before = _store_stamp(tmp_path)
    index = tmp_path / "asks"
    index.mkdir(parents=True, exist_ok=True)
    (index / f"{S1}.json").write_text(
        '{"schema":1,"session_id":"' + S1 + '","cwd":"/w","updated_at":1,"asks":[]}'
    )
    assert _store_stamp(tmp_path) != before, "a new ask file must ring the doorbell"


def test_source_name_prefers_the_directory_over_the_bare_id() -> None:
    """T10:``0000000000a1`` names nothing a reader can act on."""
    row = {
        "session": {"id": "0000000000a1", "title": "", "agent_id": ""},
        "cwd": "/Users/damian/workspace/repos/lo-notifications-l1",
    }
    assert model.source_name(row) == "lo-notifications-l1"
    assert model.source_name({"session": {"id": "0000000000a1"}, "cwd": ""}) == "0000000000a1"
    assert model.source_name({"session": {"id": "x", "title": "Named"}}) == "Named"


def test_the_watched_sessions_rows_count_as_held(tmp_path: Path) -> None:
    """R3-1: skipping a watched row is a HOLD, not a pass that covered it.

    They were filtered out entirely, so a pass that saw only the watched
    session's items reported nothing held, the app kept its store stamp, and the
    item was never re-examined after the operator switched away — the same
    silence the grace fix removed, one filter earlier.
    """
    store = NotificationStore(tmp_path)
    watched = store.notify(S1, title="on screen", priority="high")
    _age(tmp_path)
    raised: list[str] = []
    tick = attention.announce_items(
        tmp_path,
        raise_one=lambda row: raised.append(str(row["id"])) or True,
        raise_digest=lambda rows: True,
        desktop_delivers_update=lambda: False,
        held_session_id=S1,
    )
    assert raised == [], "a watched conversation is not bannered"
    assert tick.held == 1, "but it IS held, so the caller re-examines it"
    # And once the operator switches away (no hold), it raises.
    later = attention.announce_items(
        tmp_path,
        raise_one=lambda row: raised.append(str(row["id"])) or True,
        raise_digest=lambda rows: True,
        desktop_delivers_update=lambda: False,
    )
    assert raised == [watched.item_id] and later.held == 0


def test_a_titleless_notify_row_shows_its_workspace_directory(tmp_path: Path) -> None:
    """R3-1: the fallback fires on a real NOTIFY row, through the session block.

    `source_name` has read `row["cwd"]` since T10, but a notify row never carried
    it: the session block is what every surface receives, and it held no
    directory — so a conversation with no stored title painted a bare session id.
    The directory is resolved where the session's own directory is already in
    hand (`notifications.items.SessionFacts`), so the fix reaches the desktop and
    the relay too, not only this page.
    """
    session_id = "0000000000a1"
    journal = tmp_path / "sessions" / session_id
    journal.mkdir(parents=True)
    (journal / "desktop.json").write_text(
        '{"cwd": "/Users/damian/workspace/repos/lo-notifications-l1"}', encoding="utf-8"
    )
    store = NotificationStore(tmp_path)
    item = store.notify(session_id, title="a thing happened", priority="normal")
    page = inbox.list_page(tmp_path, state="all")
    row = next(row for row in page["items"] if row["id"] == item.item_id)
    assert row["session"]["cwd"] == "/Users/damian/workspace/repos/lo-notifications-l1"
    assert model.source_name(row) == "lo-notifications-l1"


def test_a_real_ask_row_shows_its_workspace_directory(tmp_path: Path) -> None:
    """R3-2: the fallback fires on a REAL row, not only on a hand-built one.

    The row comes out of the core normalizer (`notifications.items.ask_item`),
    which used to drop the ask's ``cwd`` — so the directory-name fallback could
    never trigger in production and every title-less ask painted a bare id.
    """
    from local_operator.asks import store as ask_store
    from tests.unit.notifications.test_inbox import SID as ASK_SID

    (tmp_path / "sessions" / ASK_SID).mkdir(parents=True, exist_ok=True)
    ask_store.write_entry(
        tmp_path,
        ASK_SID,
        cwd="/Users/damian/workspace/repos/lo-notifications-l1",
        asks=[
            {
                "ask_id": "ask_dir",
                "session_id": ASK_SID,
                "status": "open",
                "created_at": int(time.time() * 1000),
                "questions": [{"question": "Ship it?", "header": "Ship"}],
            }
        ],
    )
    page = inbox.list_page(tmp_path, state="all")
    asks = [row for row in (page.get("waiting") or []) if row.get("source") == "ask"]
    assert asks, "the open ask must be listed"
    assert model.source_name(asks[0]) == "lo-notifications-l1"


def test_rung_three_holds_a_fresh_revision_for_the_grace(tmp_path: Path) -> None:
    """T2: the desktop feed's grace, on the TUI's rung.

    Rung 3 ticks every second, so without the hold it claims before any in-view
    receipt can land — and it also pre-empts rung 4's own hold. The row is left
    UNCLAIMED (never claimed-and-released), so the surface it belongs to still
    owns it once the grace expires.
    """
    store = NotificationStore(tmp_path)
    item = store.notify(S1, title="just landed", priority="high")
    held = attention.announce_items(
        tmp_path,
        raise_one=lambda row: pytest.fail("a fresh revision is held"),
        raise_digest=lambda rows: pytest.fail("a fresh revision is held"),
        desktop_delivers_update=lambda: False,
    )
    assert held.raised == () and held.released == ()
    row = store.get(item.item_id)
    assert row is not None and row["delivered_rev"] == 0, "not claimed, not released"

    _age(tmp_path)
    raised: list[str] = []
    tick = attention.announce_items(
        tmp_path,
        raise_one=lambda row: raised.append(str(row["id"])) or True,
        raise_digest=lambda rows: True,
        desktop_delivers_update=lambda: False,
    )
    assert raised == [item.item_id] and tick.raised == (item.item_id,)


def test_rung_three_leaves_the_watched_session_unclaimed(tmp_path: Path) -> None:
    """T2: an item from the conversation on screen is not news to banner.

    The operator is looking at its card, and the in-view receipt is the
    acknowledgement that fits. The row is SKIPPED, not claimed and released, so
    another surface (or this one after the operator moves away) still owns it.
    """
    store = NotificationStore(tmp_path)
    watched = store.notify(S1, title="on screen", priority="high")
    other = store.notify(S2, title="elsewhere", priority="high")
    _age(tmp_path)
    raised: list[str] = []
    attention.announce_items(
        tmp_path,
        raise_one=lambda row: raised.append(str(row["id"])) or True,
        raise_digest=lambda rows: True,
        desktop_delivers_update=lambda: False,
        held_session_id=S1,
    )
    assert raised == [other.item_id]
    row = store.get(watched.item_id)
    assert row is not None and row["delivered_rev"] == 0 and row["read_rev"] == 0
    # An EMPTY hold notifies: the rule fails toward the banner when the app
    # cannot prove what it is displaying.
    raised.clear()
    attention.announce_items(
        tmp_path,
        raise_one=lambda row: raised.append(str(row["id"])) or True,
        raise_digest=lambda rows: True,
        desktop_delivers_update=lambda: False,
    )
    assert watched.item_id in raised


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


async def _page_idle(app, pilot) -> None:
    """Block until the notifications page has no read, write or arm in flight.

    WHY. Every gesture that touches the store runs in a page worker that hops to
    a THREAD: opening the page (``_load``), opening a row (``_read_detail``, then
    the mark-read write), ``e``/``y``/``z`` (``_write``/``_run_undo``) and the
    confirms ``A``/``E``/``/notifications read`` (``_arm_mark_all``,
    ``_arm_dismiss_session``). Cursor moves in the wide layout schedule the
    detail read behind a ``DETAIL_DEBOUNCE_S`` timer first. ``_settle``'s fixed
    count of turns is a bet that the thread wins in time; on a loaded runner it
    loses, and the test reads a page that has not caught up (``assert
    view.confirming``, ``undo_live`` false, an unscrollable pane). Each worker
    can start another (a write's ``reload``; the ``read`` form arming its confirm
    after the first load), so this waits until the page is quiet, not just until
    the workers that existed on entry are done: the same rule as
    ``await_session_boot`` for the boot worker. The closing ``_settle`` lets the
    painted result lay out. The deadline is a backstop for a wedged worker, not
    the assertion.
    """
    # POLLED, NEVER AWAITED WITH A TIMEOUT (review R1). ``asyncio.wait_for``
    # over ``workers.wait_for_complete`` cancels the inner gather on timeout, the
    # cancel reaches each worker's own task, and Textual's ``wait_for_complete``
    # swallows the ``CancelledError``: the wedged worker came back CANCELLED, the
    # next pass saw nothing pending, and the helper returned success, with the
    # write under test silently killed. Reading worker states touches nothing,
    # so a hang stays a hang until the deadline names it.
    started = time.monotonic()
    deadline_s = BOOT_DEADLOCK_GUARD_S
    seen: dict[int, Any] = {}
    while True:
        await pilot.pause()
        view = app._notifications_view
        assert view is not None, "/notifications did not mount the page"
        page_workers = [w for w in app.workers if w.node is view]
        seen.update((id(w), w) for w in page_workers)
        # Textual drops a worker from ``app.workers`` once it ends, so its end
        # state is read off the workers this helper has already seen.
        for worker in seen.values():
            if worker.state == WorkerState.ERROR:
                pytest.fail(
                    f"page worker {worker.name!r} ({worker.group}) raised: {worker.error!r}"
                )
            # ``reload`` runs ``exclusive=True``, so a newer read cancelling an
            # older ``notifications-load`` is the page working (review R2: that
            # is the ONLY cancel tolerated). The page never cancels a write, a
            # detail read or an arm, so one ending CANCELLED means something
            # killed the action under test and an assertion after it is vacuous.
            if worker.state == WorkerState.CANCELLED and worker.group != "notifications-load":
                pytest.fail(
                    f"page worker {worker.name!r} ({worker.group}) was cancelled; "
                    "an assertion after it would pass vacuously"
                )
        pending = [w for w in page_workers if not w.is_finished]
        # A Timer exposes no public "has it fired" signal; its task is the only
        # one, and a one-shot timer's task finishes once its callback has run.
        # Coupled to textual's private ``Timer._task`` (8.2.8, review R4).
        timer = view._detail_timer
        task = getattr(timer, "_task", None)
        debouncing = task is not None and not task.done()
        if view.is_mounted and view.snapshot is not None and not pending and not debouncing:
            break
        elapsed = time.monotonic() - started
        if elapsed >= deadline_s:
            stuck = ", ".join(f"{w.name!r} group={w.group} state={w.state.name}" for w in pending)
            pytest.fail(
                f"the notifications page never went quiet after {elapsed:.1f}s: "
                f"workers [{stuck or 'none'}], debouncing={debouncing}, "
                f"mounted={view.is_mounted}, loaded={view.snapshot is not None}"
            )
        # Mount, spawns and the debounce are message-pump and timer events with
        # no awaitable handle, and a worker's own handle cannot be awaited with a
        # timeout without cancelling it (above): yield, then re-check.
        await asyncio.sleep(0.02)
    await _settle(pilot)


@pytest.mark.asyncio
async def test_page_idle_fails_loudly_naming_a_wedged_worker(tmp_path: Path, monkeypatch) -> None:
    """Review R1: the helper's backstop must FAIL on a hang, never pass it.

    The first ``_page_idle`` awaited the workers under ``asyncio.wait_for``; on
    timeout that cancelled the wedged worker, Textual swallowed the cancel, and
    the next pass found nothing pending and returned success. Every assertion
    after a hang then read a page whose action had been killed. Here a write is
    held on an event past a shortened guard: the helper must fail within it and
    name the worker, and the worker must still be alive (not cancelled) after.
    """
    import threading

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed(tmp_path, 3)
    release = threading.Event()
    real_write = model.write

    def wedged_write(*args: Any, **kwargs: Any) -> Any:
        # Bounded so a failing assertion below cannot leak a thread forever.
        release.wait(30)
        return real_write(*args, **kwargs)

    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        app._cmd_notifications_page("", app._system_notice)
        await _page_idle(app, pilot)
        view = _page(app)
        monkeypatch.setattr(model, "write", wedged_write)
        try:
            await pilot.press("e")
            started = time.monotonic()
            # Shortened only around the wedge, so a slow first open is not the hang.
            with monkeypatch.context() as guard:
                guard.setattr(sys.modules[__name__], "BOOT_DEADLOCK_GUARD_S", 1.0)
                with pytest.raises(pytest.fail.Exception) as failure:
                    await _page_idle(app, pilot)
            elapsed = time.monotonic() - started
            message = str(failure.value)
            assert "never went quiet" in message, message
            assert "'_write'" in message and "state=RUNNING" in message, message
            assert elapsed < 10.0, f"the 1 s guard took {elapsed:.1f}s to fire"
            writes = [w for w in app.workers if w.node is view and w.name == "_write"]
            assert writes and all(not w.is_finished for w in writes), (
                "the helper cancelled the wedged worker instead of leaving it to the test",
                writes,
            )
        finally:
            release.set()
        await _page_idle(app, pilot)
        assert view.undo_live, "released, the write completes normally"


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
        await _page_idle(app, pilot)
        view = _page(app)
        assert len(view.rows) == 3
        assert store.counts()[0] == 3, "opening the page is not reading"
        await pilot.press("A")
        await _page_idle(app, pilot)
        assert view.confirming
        await pilot.press("n")
        await _settle(pilot)
        assert store.counts()[0] == 3
        # Two presses, not `press("A", "y")`: `A` arms its confirm from a worker
        # (``_arm_mark_all`` enumerates the scope on a thread), so a `y` sent in
        # the same burst lands before there is a confirm to answer.
        await pilot.press("A")
        await _page_idle(app, pilot)
        assert view.confirming
        await pilot.press("y")
        await _page_idle(app, pilot)
        assert store.counts()[0] == 0
        await pilot.press("z")
        await _page_idle(app, pilot)
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
        await _page_idle(app, pilot)
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
        await _page_idle(app, pilot)
        view = _page(app)
        assert view.current and view.current["source"] == "ask"
        await pilot.press("e")
        await _page_idle(app, pilot)
        assert "answer" in view.notice.lower()
        assert log.read_bytes() == before


@pytest.mark.asyncio
async def test_open_pins_its_row_in_wide_mode_too(tmp_path: Path, monkeypatch) -> None:
    """T3: ↵ on the second item keeps the SECOND item in the wide pane.

    The Unread filter is the reproduction: opening marks that row read, it
    leaves the list, the cursor lands on its successor — and the pane, which
    followed the cursor, showed a different item than the one just opened. The
    narrow path was pinned against exactly this; the wide path was not.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed(tmp_path, 3)
    app = _app()
    async with app.run_test(size=(140, 40)) as pilot:
        await _settle(pilot)
        app._cmd_notifications_page("", app._system_notice)
        await _page_idle(app, pilot)
        view = _page(app)
        assert view.wide, "140 columns is the split layout"
        await pilot.press("down")
        await _page_idle(app, pilot)
        second = view.current
        assert second is not None
        await pilot.press("enter")
        await _page_idle(app, pilot)
        assert view.detail_row is not None
        assert view.detail_row["id"] == second["id"], "the opened row stays in the pane"
        # Browsing releases the pin: the pane follows the cursor again.
        await pilot.press("down")
        await _page_idle(app, pilot)
        assert view.detail_row is not None
        assert view.detail_row["id"] == view.current["id"]


@pytest.mark.asyncio
async def test_esc_leaves_and_restores_the_transcript(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        app._cmd_notifications_page("", app._system_notice)
        await _page_idle(app, pilot)
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
        await _page_idle(app, pilot)
        view = _page(app)
        opened = view.current
        await pilot.press("enter")
        await _page_idle(app, pilot)
        assert view.view_mode == "detail"
        assert view.detail_row is not None and view.detail_row["id"] == opened["id"]
        assert _row(store, opened["id"])["read_rev"] == 1
        assert opened["id"] not in [str(r["id"]) for r in view.rows], "left the Unread list"
        assert ("u", "unread") in [(k, v) for k, v in view._hint_labels()]


# -- language and width (the six locale audits' code half) --------------------

#: The bidi isolate controls the paint uses: RLI wraps the sentence, LRI/PDI
#: pin a technical run inside it.
RLI, LRI, PDI = "\u2067", "\u2066", "\u2069"


@pytest.fixture()
def urdu(restore_active_locale: Any) -> Any:
    """Paint in Urdu: the only shipped locale whose layout direction differs."""
    from local_operator.i18n.text import set_active_locale

    set_active_locale("ur")
    return "ur"


@pytest.fixture()
def chinese(restore_active_locale: Any) -> Any:
    """Paint in zh-CN: the shipped locale whose words occupy two cells each."""
    from local_operator.i18n.text import set_active_locale

    set_active_locale("zh-CN")
    return "zh-CN"


def test_urdu_pins_every_literal_command_as_one_technical_island(urdu: str) -> None:
    """ur audit: isolating the SENTENCE is not enough for an in-sentence command.

    Inside an RTL run a bare LTR token is reordered by the terminal — the
    auditor measured ``/notifications`` painting as ``notifications/``. Each of
    the five keys that name a literal command therefore declares it to
    ``display_tr`` as an island, and this pins the island: present, contiguous,
    and in the order it was authored.
    """
    from local_operator.i18n.keys import tui_notifications as keys

    cases = (
        (keys.banner_digest_body(), "/notifications"),
        (keys.refused_argument(), "/notifications"),
        (keys.pointer_unread(), "/unread"),
        (keys.unread_mark_all(count="3"), "/unread read"),
        (keys.unread_mark_these(), "/unread read"),
    )
    for message, command in cases:
        painted = render._t(message)
        assert f"{LRI}{command}{PDI}" in painted, (command, painted)
        # ...and the token itself is never reversed: no second space, no
        # `/unreadread`, no trailing command on the wrong side of its words.
        assert command.split()[-1] in painted and "/" in painted


def test_urdu_paints_the_back_control_with_the_forward_arrow_it_reads_by(
    urdu: str,
) -> None:
    """ur audit: with the sentence isolated as RLI…PDI the interior is RTL.

    The arrow sits at the LOGICAL start, so the terminal paints it at the right
    edge — the edge a right-to-left reader begins from. U+2190 is not mirrored
    by bidi, so the pre-audit ``←`` pointed forward there; the audited glyph is
    ``→``. This pins the logical order and the isolation; the visual position is
    the terminal's bidi resolution, which the auditor measured separately.
    """
    from local_operator.i18n.keys import tui_notifications as keys

    painted = render._t(keys.back())
    assert painted.startswith(RLI) and painted.endswith(PDI), painted
    assert painted[1:-1].startswith("\u2192 "), painted
    assert "\u2190" not in painted


def test_a_row_is_never_wider_than_the_list_whatever_the_script(
    chinese: str, tmp_path: Path
) -> None:
    """zh-CN audit: the columns are CELL budgets, so the padding must be too.

    Measured before the fix at a 78-cell list (80-column terminal): a zh-CN
    just-now row with an ASCII title painted 80 cells, with a Chinese title 86,
    and — in ENGLISH — a Chinese conversation title 82, because the source name
    was right-aligned with ``str.rjust``, which counts characters.
    """
    from rich.cells import cell_len

    store = NotificationStore(tmp_path)
    store.notify(S1, title="CI failed on the release branch", key="ascii", session_title="deploy")
    store.notify(S2, title="发布分支的 CI 失败了", key="cjk", session_title="部署流水线")
    rows, _ = store.list_items(state="all")
    for width in (78, 60, 40):
        for row in rows:
            painted = render.row_line(row, width, cursor=False, selected=False)
            assert cell_len(painted.plain) <= width, (width, cell_len(painted.plain))


def test_a_wide_source_name_does_not_widen_the_row(urdu: str, tmp_path: Path) -> None:
    """The source column is a cell budget too (same audit, second half)."""
    from rich.cells import cell_len

    store = NotificationStore(tmp_path)
    store.notify(S1, title="title", key="wide", session_title="データ同期パイプライン")
    rows, _ = store.list_items(state="all")
    painted = render.row_line(rows[0], 78, cursor=False, selected=False)
    assert cell_len(painted.plain) == 78


def test_the_detail_header_never_paints_a_raw_priority_token(
    tmp_path: Path, restore_active_locale: Any
) -> None:
    """vi audit: the header printed the English data word in every locale.

    ``… + " " + priority`` put the raw token ``normal`` into a Vietnamese
    header. The bars carry the level; ``high`` follows them with the CATALOGUED
    chip, exactly as the list column does — so an untranslated data word never
    appears, and the frozen namespace needs no new key for the default level.
    """
    from local_operator.i18n.keys import tui_notifications as keys
    from local_operator.i18n.text import set_active_locale

    store = NotificationStore(tmp_path)
    store.notify(S1, title="routine", priority="normal", key="n")
    store.notify(S2, title="urgent", priority="high", key="h")
    rows, _ = store.list_items(state="all")
    normal_row = next(row for row in rows if row["priority"] == "normal")
    high_row = next(row for row in rows if row["priority"] == "high")
    for locale in ("en", "vi", "zh-CN"):
        set_active_locale(locale)
        default_header = render.detail_text(
            None, normal_row, 80, position=(1, 2)
        ).plain.splitlines()[0]
        assert "normal" not in default_header, (locale, default_header)
        high_header = render.detail_text(None, high_row, 80, position=(1, 2)).plain.splitlines()[0]
        assert "high" not in high_header.replace(render._t(keys.chip_high()), ""), (
            locale,
            high_header,
        )
        assert render.BARS["high"] in high_header, (locale, high_header)
    set_active_locale("en")


@pytest.mark.asyncio
async def test_the_combined_receipt_wraps_instead_of_clipping(
    tmp_path: Path, monkeypatch, restore_active_locale: Any
) -> None:
    """vi audit: three clauses compose past one row, in English too.

    ``dismissed`` + ``asks_stay`` + ``stale_skipped`` measured 95 cells in
    English and 121 in Vietnamese against the page's 76-cell notice, and the
    rendered frame stopped mid-sentence at "… 2 mục đã". Nothing is shortened:
    the notice takes the second line it needs.
    """
    from local_operator.i18n.text import set_active_locale
    from local_operator.tui.widgets.notifications_view import NotificationsView

    result = model.WriteResult(
        op="dismiss",
        changed=("n_aaaaaaaaaaaaaaaa",),
        tokens=(("n_aaaaaaaaaaaaaaaa", 1),),
        skipped_live_asks=2,
        skipped_stale=3,
    )
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed(tmp_path, 2)
    for locale in ("en", "vi"):
        set_active_locale(locale)
        receipt = NotificationsView._receipt(result)
        # A FRESH app per locale: an App instance runs once.
        app = _app()
        async with app.run_test(size=(80, 24)) as pilot:
            await _settle(pilot)
            app._cmd_notifications_page("", app._system_notice)
            await _page_idle(app, pilot)
            view = _page(app)
            view.notice = receipt
            view._paint()
            await _settle(pilot, 10)
            notice = view._notice_text
            # GEOMETRY AND PAINT TOGETHER: the receipt is longer than one
            # 76-cell row, so the box must be taller than the single content
            # line `.settings-view-detail` used to give it, and the LAST clause
            # — the part that was cut — must be on screen.
            assert cell_len(receipt) > 76, (locale, cell_len(receipt))
            assert notice.size.height >= 2, (locale, notice.size.height)
            # The PAINTED lines, one strip per row: what the operator actually
            # gets, not the string that was handed to the widget.
            painted = " ".join(
                notice.render_line(y).text.strip() for y in range(notice.size.height)
            )
            assert receipt.rstrip()[-8:] in painted, (locale, painted[-70:])
    set_active_locale("en")


def test_every_shipped_locale_fits_its_time_column() -> None:
    """zh-CN audit's own guard: the time column's copy must fit TIME_CELLS.

    ``row_line`` pads the stamp to ``TIME_CELLS`` and never truncates it, so a
    locale whose "now" word is wider than the budget widens every row that uses
    it — the padding cannot save it. This is the only check that keeps the
    column exact as copy changes, so it names the locale and the measured width
    when it fails.
    """
    from local_operator.i18n import catalogues
    from local_operator.i18n.keys import tui_notifications as keys
    from local_operator.i18n.text import set_active_locale

    root = catalogues.catalogue_root()
    locales = ["en"] + sorted(
        path.parent.name
        for path in root.glob("*/tui.notifications.json")
        if path.parent.name != "en"
    )
    assert len(locales) > 1, "no translated catalogue is on disk to check"
    for locale in locales:
        set_active_locale(locale)
        painted = render._t(keys.row_now())
        assert cell_len(painted) <= render.TIME_CELLS, (
            locale,
            cell_len(painted),
            painted,
            render.TIME_CELLS,
        )
    set_active_locale("en")


@pytest.mark.asyncio
async def test_the_undo_key_outranks_every_hint_the_eighty_column_row_drops(
    tmp_path: Path, monkeypatch
) -> None:
    """D1: the undo key must survive the narrow footer while it is live.

    Ranked eighth, ``z`` was the FIRST label an 80-column row dropped, so the
    receipt ("Dismissed 1 update.") sat above a footer that named every gesture
    except the one that reverses it — immediately after `e`, and immediately
    after `E` dismissed a whole conversation. The window is transient, so the
    label costs the row nothing once it closes; while it is open it is the
    safety net that justifies `e` firing without a confirm.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed(tmp_path, 4)
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        app._cmd_notifications_page("", app._system_notice)
        await _page_idle(app, pilot)
        view = _page(app)

        await pilot.press("e")
        await _page_idle(app, pilot)
        assert view.undo_live, "the dismissal did not open an undo window"
        assert "z" in [key for key, _label in view._hint_labels()], "the label was not offered"
        assert "z undo" in view._hints.render().plain, view._hints.render().plain

        # The whole-conversation gesture, which is the one the finding called
        # most destructive: `E` then `y`, then the same footer rule.
        await pilot.press("E")
        await _page_idle(app, pilot)
        assert view.confirming
        await pilot.press("y")
        await _page_idle(app, pilot)
        assert view.undo_live, "the bulk dismissal did not open an undo window"
        painted = view._hints.render().plain
        assert "z undo" in painted, painted


@pytest.mark.asyncio
async def test_switching_language_repaints_the_open_page(tmp_path: Path, monkeypatch) -> None:
    """Q2-1: an open page followed the store, not the locale.

    `_apply_language` repainted the transcript, the welcome band and the status
    bar; the page holds PAINTED copy of its own and its 2 s poll only repaints
    on a store change, so the header kept the previous language indefinitely —
    measured on the real app as still Chinese through a switch to ru and back.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed(tmp_path, 2)
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        app._cmd_notifications_page("", app._system_notice)
        await _page_idle(app, pilot)
        view = _page(app)
        before = view._title.render().plain
        assert before, "the page paints a title"
        app._apply_language("ru")
        await _settle(pilot)
        after = view._title.render().plain
        assert after != before, (before, after)
        assert "Уведомления" in after, after
        app._apply_language("en")
        await _settle(pilot)
        assert "Notifications" in view._title.render().plain, view._title.render().plain


@pytest.mark.asyncio
async def test_the_focused_pane_names_the_gestures_that_still_fire(
    tmp_path: Path, monkeypatch
) -> None:
    """D2: the pane footer said `esc back` while five keys still acted.

    With the wide pane focused, `e`, `E`, `u`, `o` and `a` all acted on the
    PANE's item, and the footer named none of them. The focus marker was also
    painted inside the scrolling text, so the first scroll took it off screen
    and `esc back` was the only cue left.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    store = _seed(tmp_path, 4)
    # A tall item, so the pane can actually scroll: the marker's whole point is
    # that it stays visible when the pane's content does not.
    store.notify(S1, title="a long update", key="long", body_md="line\n" * 60)
    app = _app()
    async with app.run_test(size=(140, 32)) as pilot:
        await _settle(pilot)
        app._cmd_notifications_page("", app._system_notice)
        await _page_idle(app, pilot)
        view = _page(app)
        assert view.wide

        # `enter` opens (and pins) the row; the SECOND `enter` on the row the
        # pane already shows hands it the keys (DT3).
        await pilot.press("enter")
        await _page_idle(app, pilot)
        await pilot.press("enter")
        await _page_idle(app, pilot)
        assert view._detail_focus
        painted = view._hints.render().plain
        for key in ("o", "a", "e", "E", "u", "esc"):
            assert key in [k for k, _label in view._hint_labels()], (key, view._hint_labels())
        assert "esc back" in painted, painted
        assert "e dismiss" in painted, painted

        # The marker survives scrolling: the edge column does not scroll.
        # pgdn is the pane's own key (DT3), so the scroll is driven the way the
        # operator drives it rather than by writing the offset.
        #
        # The pane is only taller than its box once the detail READ has landed:
        # until then it paints the row's one-line fallback, `scroll_relative`
        # clamps to 0, and the presses are silently spent. That read is a
        # thread hop inside a worker, so a fixed count of `_settle` turns lost
        # it on a loaded runner (CI jobs 122475722774, 122526528742). The
        # `_page_idle` above waited on it; assert the premise rather than assume it.
        assert view._detail.max_scroll_y > 0, (
            "the detail read landed but the pane is not scrollable",
            view._detail.virtual_size,
            view._detail.size,
        )
        for _ in range(4):
            await pilot.press("pagedown")
        await _settle(pilot)
        assert view._detail.scroll_offset.y > 0, "the pane did not scroll"
        assert (
            render.CURSOR in view._edge.render().plain.splitlines()[0]
        ), view._edge.render().plain.splitlines()[:2]


def test_a_long_link_url_hangs_at_its_labels_inset() -> None:
    """D6: a folded URL started one column LEFT of its label's inset.

    The URL was appended raw and the widget folded it at column 0, so a wrapped
    link read as a stray left-aligned run under ` Admin: frozen credits`.
    """
    from local_operator.tui.notifications_render import detail_text

    row = {
        "id": "n_link",
        "source": "notify",
        "priority": "normal",
        "title": "14 sign-ups, 12 blocked",
        "body_md": "Seventeen attempts from one /24 between 10:12 and 10:40.",
        "created_at": 1_000,
        "updated_at": 1_000,
        "revision": 1,
        "repeat_count": 1,
        "session": {"id": "s1"},
        "links": [
            {
                "label": "Admin: frozen credits",
                "url": "https://example.invalid/admin/credits/" + "x" * 60,
            }
        ],
    }
    text = detail_text(None, row, 60, position=(1, 1), now_ms=2_000, show_back=False).plain
    lines = text.splitlines()
    label_at = next(i for i, line in enumerate(lines) if "Admin: frozen credits" in line)
    # The URL's own lines follow the label, each hung at the block's inset (2);
    # joined back up they are exactly the URL, so nothing was dropped.
    url_lines: list[str] = []
    for line in lines[label_at + 1 :]:
        if not line.startswith("  ") or not line.strip():
            break
        url_lines.append(line)
    assert len(url_lines) > 1, ("the URL must wrap here, or the case proves nothing", url_lines)
    assert "".join(line[2:] for line in url_lines) == (
        "https://example.invalid/admin/credits/" + "x" * 60
    )
    for line in url_lines:
        assert len(line) <= 60, (len(line), line)


def test_an_ask_pane_never_reads_like_an_error(tmp_path: Path) -> None:
    """D7: "No details were attached." under an ask read like a failure.

    For an ask the question IS the detail, so the line said the item was broken
    when it was complete.
    """
    from local_operator.notifications.items import ProjectFacts, SessionFacts, ask_item
    from local_operator.tui.notifications_render import detail_text

    row = ask_item(
        {
            "ask_id": "ask_1",
            "session_id": "s1",
            "status": "open",
            "created_at": 1_000,
            "expires_at": 1_000 + 37 * 60 * 1000,
            "questions": [{"question": "Block the /24 for 24 hours, or keep watching?"}],
        },
        {},
        SessionFacts(tmp_path),
        ProjectFacts(tmp_path),
    )
    text = detail_text(None, row, 75, position=(1, 1), now_ms=2_000, show_back=False).plain
    assert "No details were attached." not in text, text
    assert "expires" in text, "the meta line still names the expiry"


@pytest.mark.asyncio
async def test_the_focused_pane_names_the_undo_it_can_still_take(
    tmp_path: Path, monkeypatch
) -> None:
    """R4-1: the pane's footer returned before the undo check, so it never named `z`.

    Every other branch passes through `undo_live`; the focused pane's branch
    returns first. It is also the branch that NAMES `e dismiss` / `E dismiss all`,
    so it invited the destructive gesture and hid the key that reverses it — D1's
    exact defect, in the state D2 made reachable.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed(tmp_path, 4)
    app = _app()
    async with app.run_test(size=(160, 40)) as pilot:
        await _settle(pilot)
        app._cmd_notifications_page("", app._system_notice)
        await _page_idle(app, pilot)
        view = _page(app)
        await pilot.press("enter")
        await _page_idle(app, pilot)
        await pilot.press("enter")
        await _page_idle(app, pilot)
        assert view._detail_focus
        await pilot.press("e")
        await _page_idle(app, pilot)
        assert view.undo_live, "the dismissal did not open an undo window"
        painted = view._hints.render().plain
        assert "z undo" in painted, painted


def test_the_eighty_column_row_never_drops_the_primary_action() -> None:
    """R4-2: the D1 re-rank pushed `↵` behind `u` and `space`, so the row dropped OPEN.

    The update row's ladder at the painted 80-column width (75 cells) must keep
    the way out, the undo, and the primary action before it spends anything on
    the secondary gestures.
    """
    labels = [
        ("z", "undo"),
        ("↵", "open"),
        ("e", "dismiss"),
        ("E", "dismiss all"),
        ("space", "select"),
        ("A", "read all"),
        ("u", "read"),
        ("o", "conversation"),
        ("f", "filter"),
        ("esc", "back"),
    ]
    kept = [key for key, _label in render.hint_row(labels, 75)]
    # The ladder drops by RANK but paints in emission order (`esc` is appended
    # last by `_hint_labels`), so membership is the property, not position.
    assert "esc" in kept, kept
    assert "z" in kept, kept
    assert "↵" in kept, ("the primary action was dropped", kept)
    # The tail is what pays: the browse-only keys go first, the volume keys last.
    assert "f" not in kept and "o" not in kept, kept
    assert "e" in kept and "E" in kept, kept
