"""L1b delivery: the banner ladder, its claim, and the completion suppression.

Real SQLite stores, the production session door, a real feed tick and the real
relay routes through FastAPI's ``TestClient``. Synthetic ids and a scratch config
root throughout; nothing here calls a provider or reaches the operator's desktop.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.notifications import delivery
from local_operator.notifications.delivery import (
    RUNG_DETACHED,
    RUNG_FEED,
    RUNG_PRIORITY,
    RUNG_READ,
    deliver_item,
)
from local_operator.notifications.items import apply_patch
from local_operator.notifications.store import NotificationStore
from tests.notification_opt_in import notification_path_opt_in
from tests.unit.notifications.test_inbox import SID, _lease, _run_awaitable


@pytest.fixture(autouse=True)
def banners_allowed() -> Iterator[None]:
    """The visible, deliberate opt-in to the notify path (see ``conftest``).

    The suite arms ``LOCAL_OPERATOR_NO_NOTIFICATIONS`` and runs under a
    redirected HOME, so both of the ladder's rung-independent gates are shut by
    default — which is exactly what round 1 asked them to do, and why the tests
    that assert the gates open them by hand. The body is the shared opt-in
    (``tests/notification_opt_in``), which also waives the identity gate; every
    test here intercepts ``notify`` at the ladder's own seam, so no process is
    ever launched.
    """
    with notification_path_opt_in():
        yield


def _high_item(store: NotificationStore, *, priority: str = "high", key: str = "k") -> Any:
    return store.notify(
        SID, title="Outside PR needs review", body_md="b", priority=priority, key=key
    )


def _row(store: NotificationStore, item_id: str) -> dict[str, Any]:
    """The stored row, asserted present: a missing row is a failure, not a None."""
    row = store.get(item_id)
    assert row is not None, f"{item_id} is not in the store"
    return row


def _started(*_args: Any, **_kwargs: Any) -> bool:
    return True


def _silent(*_args: Any, **_kwargs: Any) -> bool:
    return False


def test_normal_and_low_never_banner(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    for priority in ("normal", "low"):
        item = _high_item(store, priority=priority, key=priority)
        result = deliver_item(
            tmp_path,
            item_id=item.item_id,
            revision=item.revision,
            priority=priority,
            notify=_started,
            presence=None,
            hold=False,
        )
        assert result.rung == RUNG_PRIORITY and not result.raised
    # The silence is a DECISION, not a missing banner: nothing was claimed, so a
    # later surface is still free to raise a revision that becomes high.
    assert all(_row(store, row["id"])["delivered_rev"] == 0 for row in store.list_items()[0])


def test_high_raises_exactly_one_banner_and_keeps_one_durable_row(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    raises: list[dict[str, Any]] = []

    def notify(title: str, body: str, **kwargs: Any) -> bool:
        raises.append({"title": title, "body": body, **kwargs})
        return True

    first = deliver_item(
        tmp_path,
        item_id=item.item_id,
        revision=1,
        priority="high",
        session_id=SID,
        title="Outside PR needs review",
        body="b",
        store=store,
        notify=notify,
        hold=False,
    )
    assert first.raised and first.rung == RUNG_DETACHED and first.claimed
    second = deliver_item(
        tmp_path,
        item_id=item.item_id,
        revision=1,
        priority="high",
        session_id=SID,
        store=store,
        notify=notify,
        hold=False,
    )
    assert not second.raised and second.rung in (RUNG_READ, delivery.RUNG_NONE)
    assert len(raises) == 1
    rows, _ = store.list_items(state="all")
    assert len(rows) == 1 and rows[0]["delivered_rev"] == 1
    assert store.counts()[0] == 1, "claiming a banner is not reading it"


def _race(root: str, item_id: str, index: int, out: Any) -> None:
    store = NotificationStore(root)
    out.put(store.claim_delivery(item_id, f"proc{index}"))


def test_two_processes_cannot_double_banner_one_revision(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    item = store.notify(SID, title="t", priority="high")
    context = multiprocessing.get_context("spawn")
    out = context.Queue()
    workers = [
        context.Process(target=_race, args=(str(tmp_path), item.item_id, i, out)) for i in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
    claims = sorted(out.get(timeout=10) for _ in workers)
    assert claims == [False, True], "exactly one process may claim one revision"
    assert _row(store, item.item_id)["delivered_rev"] == 1


def test_a_receipt_before_the_claim_suppresses_and_a_revision_revives_the_banner(
    tmp_path: Path,
) -> None:
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    apply_patch(tmp_path, "read", ids=[{"id": item.item_id, "revision": 1}])
    suppressed = deliver_item(
        tmp_path,
        item_id=item.item_id,
        revision=1,
        priority="high",
        store=store,
        notify=_started,
        hold=False,
    )
    assert suppressed.rung == RUNG_READ and not suppressed.raised
    # GRACE EXPIRY IS NOT A PERMANENT SUPPRESSION: new content makes the row
    # unread again, and the next raise banners that NEW revision.
    updated = store.notify(SID, title="Outside PR needs review", body_md="now 2 reviews", key="k")
    assert updated.revision == 2
    revived = deliver_item(
        tmp_path,
        item_id=item.item_id,
        revision=2,
        priority="high",
        store=store,
        notify=_started,
        hold=False,
    )
    assert revived.raised and _row(store, item.item_id)["delivered_rev"] == 2


def test_a_tui_showing_the_conversation_stands_the_runtime_down(tmp_path: Path) -> None:
    """R2-2: the rung-3 hold must also silence rung 4.

    The TUI's own leg skips a conversation it is showing (so the operator is not
    bannered about a card they are reading), and the runtime then raised its
    detached banner anyway — the cross-rung duplicate the auditor reproduced.
    Only an explicit True suppresses; unknown notifies (contract §6.3).
    """
    # Watched and active: nothing is raised and nothing is handed off.
    held = delivery.route_item(tmp_path, action="created", priority="high", tui_watching=True)
    assert held.rung == delivery.RUNG_TUI_VIEWER and not held.spoken_for
    # The grace+claim path never runs for it either: route_item is the only door
    # into deliver_item for a written item.
    assert (
        delivery.deliver_item(
            tmp_path,
            item_id="n_aaaaaaaaaaaaaaaa",
            revision=1,
            priority="high",
            tui_watching=True,
            notify=lambda *a, **k: pytest.fail("a watched conversation is not bannered"),
            hold=False,
        ).rung
        == delivery.RUNG_TUI_VIEWER
    )

    # UNKNOWN still notifies: the same call with no evidence raises.
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    raised: list[str] = []
    result = delivery.deliver_item(
        tmp_path,
        item_id=item.item_id,
        revision=1,
        priority="high",
        store=store,
        tui_watching=None,
        notify=lambda *a, **k: raised.append("yes") or True,
        hold=False,
    )
    assert result.raised and raised == ["yes"]
    # And an explicit FALSE behaves the same as unknown.
    _high_item(NotificationStore(tmp_path))
    assert delivery.route_item(
        tmp_path, action="created", priority="high", tui_watching=False
    ).rung in (delivery.RUNG_DETACHED, delivery.RUNG_FEED, delivery.RUNG_GATED)


def test_a_spawn_that_reports_nothing_releases_the_claim(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    result = deliver_item(
        tmp_path,
        item_id=item.item_id,
        revision=1,
        priority="high",
        store=store,
        notify=_silent,
        hold=False,
    )
    assert not result.raised and result.rung == delivery.RUNG_NONE
    # The watermark is back, so the revision is still owed — and the item is
    # still unread, so nothing was acknowledged by the attempt.
    assert _row(store, item.item_id)["delivered_rev"] == 0
    assert store.counts()[0] == 1
    # A RELEASE ROLLS THE COOLDOWN BACK (R5-1): the item was never SHOWN, so a
    # transient spawn failure must not silence it for thirty minutes. The next
    # claim therefore succeeds immediately; what bounds the retry is the caller's
    # own backoff (the TUI's doubling interval, the feed's ITEM_RETRY_BACKOFF_S),
    # not the cooldown.
    assert store.claim_delivery(item.item_id, "desktop"), "nothing was shown, so nothing holds"
    row = _row(store, item.item_id)
    assert row["delivered_at"] is not None, "a SUCCESSFUL claim keeps its stamp"
    assert not store.claim_delivery(item.item_id, "desktop"), "and now the cooldown holds"


def test_a_transient_failure_does_not_suppress_the_item_for_the_cooldown(
    tmp_path: Path,
) -> None:
    """R5-1: rollback on release, retention on success — the pair, both ways."""
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    assert (
        deliver_item(
            tmp_path,
            item_id=item.item_id,
            revision=1,
            priority="high",
            store=store,
            notify=_silent,
            hold=False,
        ).rung
        == delivery.RUNG_NONE
    )
    row = _row(store, item.item_id)
    assert row["delivered_rev"] == 0 and row["delivered_at"] is None, row

    # The same revision now raises, because nothing is standing in its way.
    assert deliver_item(
        tmp_path,
        item_id=item.item_id,
        revision=1,
        priority="high",
        store=store,
        notify=lambda *a, **k: True,
        hold=False,
    ).raised
    assert _row(store, item.item_id)["delivered_at"] is not None


def test_an_advertising_desktop_takes_the_feed_rung_without_a_second_banner(tmp_path: Path) -> None:
    _lease(
        tmp_path,
        session_id=SID,
        input_at_ms=int(time.time() * 1000) - 5_000,
        screen_locked=False,
        can_notify_kinds=["update"],
    )
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    raises: list[str] = []
    result = deliver_item(
        tmp_path,
        item_id=item.item_id,
        revision=1,
        priority="high",
        store=store,
        notify=lambda *a, **k: raises.append("os") or True,
        hold=False,
    )
    assert result.rung == RUNG_FEED and result.spoken_for
    assert raises == [], "the runtime must not spawn a banner the desktop will raise"
    # No claim here: the feed's own tick claims, so the two rungs cannot both win.
    assert _row(store, item.item_id)["delivered_rev"] == 0


def test_presence_activity_never_marks_anything_read(tmp_path: Path) -> None:
    now = int(time.time() * 1000)
    _lease(tmp_path, session_id=SID, input_at_ms=now - 1_000, screen_locked=False)
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    deliver_item(
        tmp_path,
        item_id=item.item_id,
        revision=1,
        priority="high",
        store=store,
        notify=_silent,
        hold=False,
    )
    row = _row(store, item.item_id)
    assert row["read_rev"] == 0 and row["read_via"] is None
    assert store.counts()[0] == 1


def _advertise(feed: Any, subscription: Any, kinds: list[str] | None = None) -> None:
    """Publish presence FOR THIS FEED'S OWN SUBSCRIPTION, the way the route does.

    Production's ``POST /v1/desktop/presence`` looks the subscription id up in
    the serving engine's own table, so the claim it writes names a subscription
    that engine will fan out to. A hand-written lease on another id is another
    process's, and round 1 (R3) is about this feed refusing to claim for it.
    """
    feed.presence.update(
        subscription.id,
        can_notify=True,
        can_notify_kinds=kinds if kinds is not None else ["update"],
        session_id=SID,
        window={"exists": True, "focused": True, "visible": True, "minimized": False},
    )


def _cool(root: Path, *, ms: int | None = None) -> None:
    """Move every row's LAST BANNER ATTEMPT past the cooldown (D15).

    Separate from :func:`_age` on purpose: ``_age`` moves content time so the
    grace has elapsed, and a test that wants the banner cooldown to have expired
    is asserting a different clock.
    """
    import sqlite3

    from local_operator.notifications.store import BANNER_COOLDOWN_MS, now_ms

    stamp = now_ms() - (BANNER_COOLDOWN_MS if ms is None else ms) - 1
    conn = sqlite3.connect(root / "notifications.db")
    try:
        with conn:
            conn.execute("UPDATE items SET delivered_at=?", (stamp,))
    finally:
        conn.close()


def _age(root: Path, *, ms: int = 10_000) -> None:
    """Move every row's content stamp into the past, past its banner grace.

    The ladder HOLDS a fresh revision so an in-view receipt can cancel it
    (N2/Q4), and the feed reads that hold off ``updated_at``. A test that writes
    and ticks immediately would be asserting the hold, not the raise, so it ages
    the rows the way a real elapsed grace does.
    """
    import sqlite3

    store = NotificationStore(root)
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE items SET updated_at = updated_at - ?", (int(ms),))


def _frames(subscription: Any) -> list[dict[str, Any]]:
    """The ``notification`` frames queued for ONE subscription, draining them."""
    from tests.unit.server.test_desktop_feed import _queued

    return [frame for frame in _queued(subscription) if frame["type"] == "notification"]


def _fresh_row(root: Path, session_id: str = "aaaaaaaaaaaa", title: str = "t") -> str:
    """Create one undelivered row and age it past the grace; returns its id."""
    item = NotificationStore(root).notify(session_id, title=title, priority="high")
    _age(root)
    return item.item_id


def test_the_feed_raises_one_update_frame_per_revision_and_one_digest(
    tmp_path: Path,
) -> None:
    from tests.unit.server.test_desktop_feed import _feed, _queued, _tick

    feed = _feed(tmp_path)
    feed._take_baseline()
    subscription = feed.subscribe()
    _tick(feed)
    _queued(subscription)
    store = NotificationStore(tmp_path)
    item = store.notify(SID, title="needs review", priority="high")
    _age(tmp_path)
    # No advertising desktop yet: rung 2 is closed, so nothing is published.
    _tick(feed)
    assert [f for f in _queued(subscription) if f["type"] == "notification"] == []
    _advertise(feed, subscription)
    _tick(feed)
    frames = _frames(subscription)
    assert len(frames) == 1
    payload = frames[0]["payload"]
    assert payload["kind"] == "update" and payload["notification_id"] == item.item_id
    assert frames[0]["payload"]["item_revision"] == 1
    # N1: a member of the UI's wire union, and the item may raise while focused.
    assert payload["focus_policy"] == "always"
    # The same revision never banners twice, however many ticks run. Each read
    # drains the subscription queue, so this collects rather than re-reads.
    for _ in range(3):
        _tick(feed)
        frames.extend(_frames(subscription))
    assert len(frames) == 1
    asyncio.run(feed.close())


def test_the_burst_cap_is_a_window_not_a_tick(tmp_path: Path) -> None:
    """R4/Q2: 10 and 12 highs ⇒ 3 singles + ONE digest, and nothing re-bannered.

    The old leg capped per 100 ms TICK, so the digest stood for one item and the
    rest drained one tick later — ten interruptions in a third of a second. The
    whole set must be visible after the first window, and the rows the digest
    covered must never be raised individually afterwards.
    """
    from tests.unit.server.test_desktop_feed import _feed, _queued, _tick

    for count in (10, 12):
        root = tmp_path / f"burst-{count}"
        root.mkdir()
        feed = _feed(root)
        feed._take_baseline()
        subscription = feed.subscribe()
        _tick(feed)
        _queued(subscription)
        _advertise(feed, subscription)
        store = NotificationStore(root)
        for index in range(count):
            store.notify(f"{index + 1:012x}", title=f"item {index}", priority="high")
        _age(root)
        # The whole burst is written at once, then the feed runs ticks until it
        # is quiet: the window, not the tick, decides the shape.
        frames: list[dict[str, Any]] = []
        for _ in range(12):
            _tick(feed)
            frames.extend(_frames(subscription))
        singles = [f for f in frames if not f["payload"].get("overflow")]
        digests = [f for f in frames if f["payload"].get("overflow")]
        assert len(singles) == 3, (count, len(singles))
        assert len(digests) == 1, (count, len(digests))
        covered = digests[0]["payload"]["count"] + len(singles)
        assert covered == count, (count, covered)
        rows, _ = store.list_items(state="all")
        delivered = {row["id"] for row in rows if row["delivered_rev"] == row["revision"]}
        assert delivered == {row["id"] for row in rows}, "the digest covers the remainder"
        # Nothing the digest stood in for is raised on its own later.
        later: list[dict[str, Any]] = []
        for _ in range(6):
            _tick(feed)
            later.extend(_frames(subscription))
        assert later == [], (count, [f["payload"].get("count") for f in later])
        asyncio.run(feed.close())


def test_only_advertising_subscriptions_receive_an_update_frame(tmp_path: Path) -> None:
    """R3: the frame goes to the subscription that advertised the kind.

    Two subscriptions on one feed, one advertising ``update``: the fan-out used
    to hand the frame to both, so an older app received a kind it cannot focus.
    """
    from tests.unit.server.test_desktop_feed import _feed, _queued, _tick

    feed = _feed(tmp_path)
    feed._take_baseline()
    advertising = feed.subscribe()
    plain = feed.subscribe()
    _tick(feed)
    _queued(advertising)
    _queued(plain)
    _advertise(feed, advertising)
    feed.presence.update(
        plain.id,
        can_notify=True,
        can_notify_kinds=["complete", "error"],
        session_id=SID,
        window={"exists": True, "focused": True, "visible": True, "minimized": False},
    )
    NotificationStore(tmp_path).notify(SID, title="needs review", priority="high")
    _age(tmp_path)
    _tick(feed)
    assert len(_frames(advertising)) == 1
    assert _frames(plain) == [], "an app that did not advertise update gets no item banner"
    asyncio.run(feed.close())


def test_another_runtimes_lease_does_not_let_this_feed_claim(tmp_path: Path) -> None:
    """R1/R5: a lease belonging to ANOTHER presence owner never claims for this one.

    This is the round-1 failure that starved the real desktop: a second runtime
    advertised `update`, so this feed claimed the item and published it to a
    subscription that had not advertised the kind.
    """
    from tests.unit.server.test_desktop_feed import _feed, _queued, _tick

    feed = _feed(tmp_path)
    feed._take_baseline()
    subscription = feed.subscribe()
    _tick(feed)
    _queued(subscription)
    # The OTHER backend's lease: a different publisher instance, its own id.
    _lease(
        tmp_path,
        session_id=SID,
        input_at_ms=int(time.time() * 1000) - 1_000,
        screen_locked=False,
        can_notify_kinds=["update"],
    )
    item = NotificationStore(tmp_path).notify(SID, title="needs review", priority="high")
    _age(tmp_path)
    for _ in range(3):
        _tick(feed)
    assert _frames(subscription) == []
    assert (
        _row(store_of(tmp_path), item.item_id)["delivered_rev"] == 0
    ), "a feed that cannot hand the frame to its own subscriber must not claim it"
    asyncio.run(feed.close())


def store_of(root: Path) -> NotificationStore:
    return NotificationStore(root)


def test_the_post_grace_raise_re_reads_the_input_probe(tmp_path: Path, monkeypatch) -> None:
    """R6-2: the grace window must not freeze the write-time answer.

    The hold is 2.5 s precisely so an in-view receipt can cancel the banner —
    which means the operator may switch TO this conversation while it runs. The
    raise used to inherit the write-time route and would then banner the screen
    they had moved to; it re-reads the probe instead. Unknown still notifies: the
    first answer here is `None` (no report yet) and only the raise sees `True`.
    """
    from tests.unit.session.test_session import ScriptedStream, make_session

    session = make_session(tmp_path, ScriptedStream([]))
    session._config_dir = tmp_path / "root"
    answers = [None, True, True, True]  # write time, then the raise
    session._notify_tui_probe = lambda: answers.pop(0) if answers else True
    seen: list[Any] = []

    def fake_route(*args: Any, **kwargs: Any) -> delivery.DeliveryResult:
        # The write-time route sees None and therefore decides on its own.
        return delivery.DeliveryResult(rung=RUNG_DETACHED, detail="runtime will raise")

    def fake_deliver(*args: Any, **kwargs: Any) -> delivery.DeliveryResult:
        seen.append(kwargs.get("tui_watching"))
        if kwargs.get("tui_watching") is True:
            return delivery.DeliveryResult(
                raised=False, rung=delivery.RUNG_TUI_VIEWER, detail="watched"
            )
        return delivery.DeliveryResult(raised=True, rung=RUNG_DETACHED, claimed=True)

    monkeypatch.setattr(delivery, "route_item", fake_route)
    monkeypatch.setattr(delivery, "deliver_item", fake_deliver)
    door = session._notify_inbox_callable()
    assert door is not None

    async def turn() -> None:
        await door(
            title="Outside PR", body="b", priority="high", key="pr", links=[], anchor_id="c1"
        )
        await session._settle_item_banners()

    _run_awaitable(turn())
    assert seen == [True], "the raise carried the FRESH probe, not the write's None"
    assert session._run_notified_banner is False, "no banner went out, so nothing is suppressed"


def test_the_session_door_banners_a_high_item_and_suppresses_its_completion_banner(
    tmp_path: Path, monkeypatch
) -> None:
    """R1 + N2/Q4: the door decides from the ROUTE, then settles the held raise.

    The tool call may not wait on the banner's own grace, so the route is
    decided synchronously and a detached raise runs in the background; the
    completion banner is suppressed only once that raise actually happened.
    """
    from tests.unit.session.test_session import ScriptedStream, make_session

    session = make_session(tmp_path, ScriptedStream([]))
    session._config_dir = tmp_path / "root"
    raised: list[str] = []

    def fake_route(*args: Any, **kwargs: Any) -> delivery.DeliveryResult:
        return delivery.DeliveryResult(rung=RUNG_DETACHED, detail="runtime will raise")

    def fake_deliver(*args: Any, **kwargs: Any) -> delivery.DeliveryResult:
        raised.append(kwargs["item_id"])
        return delivery.DeliveryResult(raised=True, rung=RUNG_DETACHED, claimed=True)

    monkeypatch.setattr(delivery, "route_item", fake_route)
    monkeypatch.setattr(delivery, "deliver_item", fake_deliver)
    door = session._notify_inbox_callable()
    assert door is not None

    async def turn() -> str:
        sentence = await door(
            title="Outside PR", body="b", priority="high", key="pr", links=[], anchor_id="c1"
        )
        # Nothing has been raised yet: the hold runs after the tool returns.
        assert session._run_notified_banner is False
        await session._settle_item_banners()
        return sentence

    sentence = _run_awaitable(turn())
    assert sentence.startswith("Notified (id n_")
    assert raised, "the held raise ran"
    assert session._run_notified_banner is True
    # A wake run that asked to be told notifies (the shipped rule); this one
    # already put an item banner on screen, so its completion banner is
    # suppressed — but the ERROR arm always notifies (contract §3.4).
    session._run_triggers = {"wake_prompt"}
    session._run_input_types = set()
    session._run_notify_requested = True
    assert session._finalize_attention_notify(_end("")) is False
    assert session._finalize_attention_notify(_end("error")) is True
    session._run_notified_banner = False
    assert session._finalize_attention_notify(_end("")) is True


def test_a_gated_item_leaves_the_completion_banner_alone(tmp_path: Path, monkeypatch) -> None:
    """R1/R2: a gated high item raises nothing, so nothing is suppressed either."""
    from tests.unit.session.test_session import ScriptedStream, make_session

    session = make_session(tmp_path, ScriptedStream([]))
    session._config_dir = tmp_path / "root"
    spawned: list[Any] = []

    def fake_route(*args: Any, **kwargs: Any) -> delivery.DeliveryResult:
        return delivery.DeliveryResult(rung=delivery.RUNG_GATED, detail="notifications disabled")

    monkeypatch.setattr(delivery, "route_item", fake_route)
    monkeypatch.setattr(session, "_spawn_background", lambda coro: spawned.append(coro) or None)
    door = session._notify_inbox_callable()
    assert door is not None
    _run_awaitable(
        door(title="Outside PR", body="b", priority="high", key="pr", links=[], anchor_id="c1")
    )
    assert spawned == [], "a gated item spawns no raise"
    assert session._run_notified_banner is False
    session._run_triggers = {"wake_prompt"}
    session._run_input_types = set()
    session._run_notify_requested = True
    assert session._finalize_attention_notify(_end("")) is True


def test_a_normal_priority_item_leaves_the_completion_banner_alone(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.unit.session.test_session import ScriptedStream, make_session

    session = make_session(tmp_path, ScriptedStream([]))
    session._config_dir = tmp_path / "root"

    def fake_deliver(*args: Any, **kwargs: Any) -> delivery.DeliveryResult:
        return delivery.DeliveryResult(rung=RUNG_PRIORITY)

    monkeypatch.setattr(delivery, "deliver_item", fake_deliver)
    door = session._notify_inbox_callable()
    assert door is not None
    _run_awaitable(
        door(title="routine", body="", priority="normal", key="", links=[], anchor_id="c2")
    )
    assert session._run_notified_banner is False
    session._run_triggers = {"wake_prompt"}
    session._run_input_types = set()
    session._run_notify_requested = True
    assert (
        session._finalize_attention_notify(_end("")) is True
    ), "an item that never bannered must not suppress its run's completion banner"


def _end(kind: str) -> Any:
    from local_operator.harness.types import AgentEndEvent

    return AgentEndEvent(messages=[], error="provider exploded" if kind == "error" else None)


# -- round 1 remediation (R1, R2/Q1, R6, R8/Q6, N2/Q4) ----------------------


def test_no_new_content_never_raises_and_never_suppresses_a_completion(
    tmp_path: Path,
) -> None:
    """R1: only a DELIVERABLE outcome raises; unchanged/absorbed are silent.

    A dismissed keyed update is the case that flooded: contract §1.3 records it
    "silently, no banner", so nothing may claim it — and because nothing was
    raised, the run's own completion banner must still go out.
    """
    store = NotificationStore(tmp_path)
    item = store.notify(SID, title="ci", priority="high", key="ci")
    assert item.action == "created"

    same = store.notify(SID, title="ci", priority="high", key="ci")
    assert same.action == "unchanged"
    unchanged = deliver_item(
        tmp_path,
        item_id=same.item_id,
        revision=same.revision,
        priority="high",
        action=same.action,
        store=store,
        notify=_started,
        hold=False,
    )
    assert unchanged.rung == delivery.RUNG_SILENT
    assert not unchanged.spoken_for, "a no-op must not silence the completion banner"

    apply_patch(tmp_path, "dismiss", ids=[{"id": item.item_id, "revision": 1}])
    changed = store.notify(SID, title="ci again", priority="high", key="ci")
    assert changed.action == "absorbed"
    absorbed = deliver_item(
        tmp_path,
        item_id=changed.item_id,
        revision=changed.revision,
        priority="high",
        action=changed.action,
        store=store,
        notify=_started,
        hold=False,
    )
    assert absorbed.rung == delivery.RUNG_SILENT and not absorbed.spoken_for
    assert _row(store, item.item_id)["delivered_rev"] == 0, "nothing may claim a dismissed row"


def test_a_dismissed_row_is_not_claimable_and_an_escalation_revives_it(
    tmp_path: Path,
) -> None:
    """R1: ``claim_delivery`` refuses a dismissed row; a raised priority re-arms it.

    Dismissal means "not relevant", so no surface may banner that row — the
    flood contract §1.3 exists to prevent. An ESCALATION is the one update that
    legitimately un-dismisses it (``revived``), and that row is claimable again
    because it is a new revision the operator has not seen.
    """
    store = NotificationStore(tmp_path)
    item = store.notify(SID, title="ci", priority="normal", key="ci")
    apply_patch(tmp_path, "dismiss", ids=[{"id": item.item_id, "revision": 1}])
    assert store.claim_delivery(item.item_id, "feed") is False, "dismissed is not claimable"
    escalated = store.notify(SID, title="ci", priority="high", key="ci")
    assert escalated.action == "revived" and escalated.revision == 2
    assert store.claim_delivery(item.item_id, "feed") is True


def test_the_gates_refuse_before_any_claim(tmp_path: Path, monkeypatch) -> None:
    """R2/Q1: the operator's switch and the rig guard apply before the claim.

    A gated revision is NOT claimed — a claim with no raise strands the
    watermark — so a later process with the gates open still banners it.
    """
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    with pytest.MonkeyPatch.context() as shut:
        shut.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
        gated = deliver_item(
            tmp_path,
            item_id=item.item_id,
            revision=1,
            priority="high",
            store=store,
            notify=_started,
            hold=False,
        )
    assert gated.rung == delivery.RUNG_GATED and not gated.spoken_for
    assert _row(store, item.item_id)["delivered_rev"] == 0
    with pytest.MonkeyPatch.context() as rigged_guard:
        rigged_guard.setattr(
            "local_operator.tui.notify.desktop_belongs_to_this_process",
            lambda **kwargs: False,
        )
        rigged = deliver_item(
            tmp_path,
            item_id=item.item_id,
            revision=1,
            priority="high",
            store=store,
            notify=_started,
            hold=False,
        )
    assert rigged.rung == delivery.RUNG_GATED
    assert _row(store, item.item_id)["delivered_rev"] == 0, "a gated row stays owed"
    raised = deliver_item(
        tmp_path,
        item_id=item.item_id,
        revision=1,
        priority="high",
        store=store,
        notify=_started,
        hold=False,
    )
    assert raised.raised and raised.spoken_for


def test_a_str_root_works_end_to_end(tmp_path: Path) -> None:
    """R8/Q6: the door hands a str; the ladder must not raise TypeError."""
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    result = deliver_item(
        str(tmp_path),
        item_id=item.item_id,
        revision=1,
        priority="high",
        store=store,
        notify=_started,
        hold=False,
    )
    assert result.raised and result.rung == RUNG_DETACHED


def test_banner_text_follows_the_privacy_flag() -> None:
    """R2: flag off ⇒ APP_NAME + the house sentence, never the model's text.

    ``_names_allowed`` is patched on the MODULE object, not by dotted string:
    ``local_operator.notifications.compose`` resolves through the package to
    the ``compose()`` function of the same name.
    """
    import sys

    from local_operator.tui.notify import APP_NAME, BODY_UPDATE

    compose_module = sys.modules["local_operator.notifications.compose"]
    with pytest.MonkeyPatch.context() as off:
        off.setattr(compose_module, "_names_allowed", lambda: False)
        title, body = delivery.banner_text("Payroll export: 3 SSNs failed", "acct 4421 …")
    assert (title, body) == (APP_NAME, BODY_UPDATE)
    assert delivery.banner_text("Payroll export", "acct 4421 …") == (
        "Payroll export",
        "acct 4421 …",
    )


def test_the_grace_holds_a_fresh_revision_for_a_receipt_that_cancels_it(
    tmp_path: Path,
) -> None:
    """N2/Q4: the receipt inside the grace cancels the banner it earned.

    The desktop's earliest accepted receipt is its 1 s dwell after a fresh
    input, so a raise at the write could never be cancelled. The hold is what
    makes the operator's "don't re-tell me what I am looking at" real.
    """
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    raises: list[str] = []
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        # The operator's in-view receipt lands inside the hold.
        apply_patch(tmp_path, "read", ids=[{"id": item.item_id, "revision": 1}])

    result = deliver_item(
        tmp_path,
        item_id=item.item_id,
        revision=1,
        priority="high",
        store=store,
        notify=lambda *a, **k: raises.append("os") or True,
        sleep=sleep,
    )
    assert sleeps and sleeps[0] > 0, "the hold is real time, not a comment"
    assert result.rung in (RUNG_READ, delivery.RUNG_SILENT) and not result.raised
    assert raises == []


def test_the_grace_expires_into_a_banner(tmp_path: Path) -> None:
    """N2/Q4: a receipt after the grace does not un-raise an unread revision."""
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    raises: list[str] = []
    result = deliver_item(
        tmp_path,
        item_id=item.item_id,
        revision=1,
        priority="high",
        store=store,
        notify=lambda *a, **k: raises.append("os") or True,
        sleep=lambda seconds: None,
    )
    assert result.raised and result.spoken_for and len(raises) == 1
    assert _row(store, item.item_id)["delivered_rev"] == 1


def test_the_route_is_decided_and_the_grace_only_holds_a_detached_raise(
    tmp_path: Path,
) -> None:
    """The two halves (round-1 N2/Q4 + R3): routing is sync, holding is not.

    The door decides suppression from the ROUTE, and an advertising desktop
    hands the hold to its own feed, so the tool call never waits on a grace
    that belongs to another surface.
    """
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    _lease(
        tmp_path,
        session_id=SID,
        input_at_ms=int(time.time() * 1000) - 1_000,
        screen_locked=False,
        can_notify_kinds=["update"],
    )
    route = delivery.route_item(tmp_path, action="created", priority="high")
    assert route.rung == RUNG_FEED and route.handed_off and route.spoken_for
    # Routing is not claiming: the feed's own tick owns the revision.
    assert _row(store, item.item_id)["delivered_rev"] == 0
    silent = delivery.route_item(tmp_path, action="absorbed", priority="high")
    assert silent.rung == delivery.RUNG_SILENT and not silent.spoken_for


def test_a_feed_that_boots_over_an_owed_revision_raises_it(tmp_path: Path) -> None:
    """Q3/F-C: the doorbell is not the only reason to look.

    The runtime handed this revision to the feed rung and the serve exited
    before its tick. A restarted feed baselines the store's fingerprint, so a
    doorbell-only leg never examines it again and the item raises no banner on
    any surface — the completion banner was suppressed for it too.
    """
    from tests.unit.server.test_desktop_feed import _feed, _tick

    item = NotificationStore(tmp_path).notify(SID, title="needs review", priority="high")
    _age(tmp_path)
    feed = _feed(tmp_path)
    feed._take_baseline()
    subscription = feed.subscribe()
    _advertise(feed, subscription)
    # No write after the baseline: nothing else will move the doorbell.
    for _ in range(3):
        _tick(feed)
    frames = _frames(subscription)
    assert len(frames) == 1 and frames[0]["payload"]["notification_id"] == item.item_id
    asyncio.run(feed.close())


def test_a_claim_failure_mid_pass_strands_nothing(tmp_path: Path) -> None:
    """R5/X5: a claim that raises releases what the pass already claimed.

    The old loop ``claimed = []; break`` discarded rows whose claims HAD
    succeeded: neither published nor released, so they were never raised again.
    """
    from tests.unit.server.test_desktop_feed import _feed, _tick

    feed = _feed(tmp_path)
    feed._take_baseline()
    subscription = feed.subscribe()
    _advertise(feed, subscription)
    store = NotificationStore(tmp_path)
    ids = [
        store.notify(f"{index + 1:09d}", title=f"item {index}", priority="high").item_id
        for index in range(4)
    ]
    _age(tmp_path)
    calls = {"n": 0}
    real = NotificationStore.claim_delivery

    def flaky(self: Any, item_id: str, by: str) -> bool:
        calls["n"] += 1
        if calls["n"] == 3:
            raise sqlite3.OperationalError("database is locked")
        return bool(real(self, item_id, by))

    with pytest.MonkeyPatch.context() as broken:
        broken.setattr(NotificationStore, "claim_delivery", flaky)
        _tick(feed)
    assert _frames(subscription) == [], "nothing may be published from a failed pass"
    # Every row is owed again: the two already claimed were RELEASED, so no
    # revision was stranded unraised.
    rows = [store.get(item_id) for item_id in ids]
    assert all(row is not None and row["delivered_rev"] == 0 for row in rows)
    # NOTHING WAS SHOWN, SO NOTHING HOLDS (R5-1): the released rows lose their
    # attempt stamp and are claimable again at once. The bound that stops a
    # per-tick claim/release storm is the FEED's own retry deadline
    # (ITEM_RETRY_BACKOFF_S), not the thirty-minute cooldown — a transient
    # failure must not cost the banner for half an hour.
    assert store.claim_delivery(ids[0], "desktop"), "a release rolls the stamp back"
    assert store.counts()[0] == 4, "no attempt acknowledged an item"
    asyncio.run(feed.close())


def test_the_feed_item_leg_is_gated_and_claims_nothing(tmp_path: Path) -> None:
    """R2/Q1: with banners off, the feed publishes nothing and strands nothing."""
    from tests.unit.server.test_desktop_feed import _feed, _tick

    feed = _feed(tmp_path)
    feed._take_baseline()
    subscription = feed.subscribe()
    _advertise(feed, subscription)
    item = NotificationStore(tmp_path).notify(SID, title="needs review", priority="high")
    _age(tmp_path)
    with pytest.MonkeyPatch.context() as shut:
        shut.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
        for _ in range(3):
            _tick(feed)
    assert _frames(subscription) == []
    assert _row(store_of(tmp_path), item.item_id)["delivered_rev"] == 0, "gated rows stay owed"
    # The operator turns banners back on: the same feed raises it, once.
    _tick(feed)
    assert len(_frames(subscription)) == 1
    asyncio.run(feed.close())


def test_the_update_frame_carries_housetext_when_the_privacy_flag_is_off(
    tmp_path: Path,
) -> None:
    """R2: the frame is composed through the same privacy rule as the banner."""
    import sys

    from local_operator.tui.notify import APP_NAME, BODY_UPDATE
    from tests.unit.server.test_desktop_feed import _feed, _tick

    feed = _feed(tmp_path)
    feed._take_baseline()
    subscription = feed.subscribe()
    _advertise(feed, subscription)
    store = NotificationStore(tmp_path)
    store.notify(SID, title="Payroll export: 3 SSNs failed", body_md="acct 4421 …", priority="high")
    _age(tmp_path)
    compose_module = sys.modules["local_operator.notifications.compose"]
    with pytest.MonkeyPatch.context() as off:
        off.setattr(compose_module, "_names_allowed", lambda: False)
        _tick(feed)
    payload = _frames(subscription)[0]["payload"]
    assert payload["title"] == APP_NAME and payload["body"] == BODY_UPDATE
    assert "Payroll" not in json.dumps(payload) and "4421" not in json.dumps(payload)
    asyncio.run(feed.close())
