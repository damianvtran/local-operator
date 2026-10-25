"""L1b delivery: the banner ladder, its claim, and the completion suppression.

Real SQLite stores, the production session door, a real feed tick and the real
relay routes through FastAPI's ``TestClient``. Synthetic ids and a scratch config
root throughout; nothing here calls a provider or reaches the operator's desktop.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import time
from pathlib import Path
from typing import Any


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
from tests.unit.notifications.test_inbox import SID, _lease, _run_awaitable


def _high_item(store: NotificationStore, *, priority: str = "high", key: str = "k") -> Any:
    return store.notify(
        SID, title="Outside PR needs review", body_md="b", priority=priority, key=key
    )


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
        )
        assert result.rung == RUNG_PRIORITY and not result.raised
    # The silence is a DECISION, not a missing banner: nothing was claimed, so a
    # later surface is still free to raise a revision that becomes high.
    assert all(store.get(row["id"])["delivered_rev"] == 0 for row in store.list_items()[0])


def test_high_raises_exactly_one_banner_and_keeps_one_durable_row(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    raises: list[dict[str, Any]] = []

    def notify(title: str, body: str, **kwargs: Any) -> bool:
        raises.append({"title": title, "body": body, **kwargs})
        return True

    first = deliver_item(
        tmp_path, item_id=item.item_id, revision=1, priority="high",
        session_id=SID, title="Outside PR needs review", body="b", store=store, notify=notify,
    )
    assert first.raised and first.rung == RUNG_DETACHED and first.claimed
    second = deliver_item(
        tmp_path, item_id=item.item_id, revision=1, priority="high",
        session_id=SID, store=store, notify=notify,
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
        context.Process(target=_race, args=(str(tmp_path), item.item_id, i, out))
        for i in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
    claims = sorted(out.get(timeout=10) for _ in workers)
    assert claims == [False, True], "exactly one process may claim one revision"
    assert store.get(item.item_id)["delivered_rev"] == 1


def test_a_receipt_before_the_claim_suppresses_and_a_revision_revives_the_banner(
    tmp_path: Path,
) -> None:
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    apply_patch(tmp_path, "read", ids=[{"id": item.item_id, "revision": 1}])
    suppressed = deliver_item(
        tmp_path, item_id=item.item_id, revision=1, priority="high", store=store, notify=_started,
    )
    assert suppressed.rung == RUNG_READ and not suppressed.raised
    # GRACE EXPIRY IS NOT A PERMANENT SUPPRESSION: new content makes the row
    # unread again, and the next raise banners that NEW revision.
    updated = store.notify(SID, title="Outside PR needs review", body_md="now 2 reviews", key="k")
    assert updated.revision == 2
    revived = deliver_item(
        tmp_path, item_id=item.item_id, revision=2, priority="high", store=store, notify=_started,
    )
    assert revived.raised and store.get(item.item_id)["delivered_rev"] == 2


def test_a_spawn_that_reports_nothing_releases_the_claim(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    result = deliver_item(
        tmp_path, item_id=item.item_id, revision=1, priority="high", store=store, notify=_silent,
    )
    assert not result.raised and result.rung == delivery.RUNG_NONE
    # The watermark is back, so the next surface can raise this revision — and
    # the item is still unread, so nothing was acknowledged by the attempt.
    assert store.get(item.item_id)["delivered_rev"] == 0
    assert store.counts()[0] == 1
    assert store.claim_delivery(item.item_id, "desktop")


def test_an_advertising_desktop_takes_the_feed_rung_without_a_second_banner(tmp_path: Path) -> None:
    _lease(tmp_path, session_id=SID, input_at_ms=int(time.time() * 1000) - 5_000,
           screen_locked=False, can_notify_kinds=["update"])
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    raises: list[str] = []
    result = deliver_item(
        tmp_path, item_id=item.item_id, revision=1, priority="high", store=store, notify=lambda *a,
        **k: raises.append("os") or True,
    )
    assert result.rung == RUNG_FEED and result.spoken_for
    assert raises == [], "the runtime must not spawn a banner the desktop will raise"
    # No claim here: the feed's own tick claims, so the two rungs cannot both win.
    assert store.get(item.item_id)["delivered_rev"] == 0


def test_presence_activity_never_marks_anything_read(tmp_path: Path) -> None:
    now = int(time.time() * 1000)
    _lease(tmp_path, session_id=SID, input_at_ms=now - 1_000, screen_locked=False)
    store = NotificationStore(tmp_path)
    item = _high_item(store)
    deliver_item(tmp_path, item_id=item.item_id, revision=1, priority="high", store=store,
                 notify=_silent)
    row = store.get(item.item_id)
    assert row["read_rev"] == 0 and row["read_via"] is None
    assert store.counts()[0] == 1


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
    # No advertising desktop yet: rung 2 is closed, so nothing is published.
    _tick(feed)
    assert [f for f in _queued(subscription) if f["type"] == "notification"] == []
    _lease(tmp_path, session_id=SID, input_at_ms=int(time.time() * 1000) - 1_000,
           screen_locked=False, can_notify_kinds=["update"])
    _tick(feed)
    frames = [f for f in _queued(subscription) if f["type"] == "notification"]
    assert len(frames) == 1
    payload = frames[0]["payload"]
    assert payload["kind"] == "update" and payload["notification_id"] == item.item_id
    assert frames[0]["payload"]["item_revision"] == 1
    # The same revision never banners twice, however many ticks run. Each read
    # drains the subscription queue, so this collects rather than re-reads.
    for _ in range(3):
        _tick(feed)
        frames.extend(f for f in _queued(subscription) if f["type"] == "notification")
    assert len(frames) == 1
    asyncio.run(feed.close())


def test_feed_overflow_becomes_one_digest_frame(tmp_path: Path) -> None:
    from tests.unit.server.test_desktop_feed import _feed, _queued, _tick

    feed = _feed(tmp_path)
    feed._take_baseline()
    subscription = feed.subscribe()
    _tick(feed)
    _queued(subscription)
    _lease(tmp_path, session_id=SID, input_at_ms=int(time.time() * 1000) - 1_000,
           screen_locked=False, can_notify_kinds=["update"])
    store = NotificationStore(tmp_path)
    for index in range(5):
        store.notify(f"{index + 1:012x}", title=f"item {index}", priority="high")
    _tick(feed)
    frames = [f for f in _queued(subscription) if f["type"] == "notification"]
    assert len(frames) == 4, "the ceiling's three, plus one digest for what it held back"
    digest = [f for f in frames if f["payload"].get("overflow")]
    assert len(digest) == 1 and digest[0]["payload"]["count"] == 1
    # The queue DRAINS: the held-back item banners on the next tick, exactly once.
    _tick(feed)
    later = [f for f in _queued(subscription) if f["type"] == "notification"]
    assert len(later) == 1, "the item the digest stood in for drains on the next tick"
    assert later[0]["payload"]["notification_id"] not in digest[0]["payload"][
        "notification_ids"
    ], "the digest named the held-back rows; this is the one after them"
    rows, _ = store.list_items(state="all")
    assert all(row["delivered_rev"] == row["revision"] for row in rows)
    asyncio.run(feed.close())


def test_the_session_door_banners_a_high_item_and_suppresses_its_completion_banner(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.unit.session.test_session import ScriptedStream, make_session

    session = make_session(tmp_path, ScriptedStream([]))
    session._config_dir = tmp_path / "root"
    raised: list[str] = []

    def fake_deliver(*args: Any, **kwargs: Any) -> delivery.DeliveryResult:
        raised.append(kwargs["item_id"])
        return delivery.DeliveryResult(raised=True, rung=RUNG_DETACHED, claimed=True)

    monkeypatch.setattr(delivery, "deliver_item", fake_deliver)
    door = session._notify_inbox_callable()
    assert door is not None
    sentence = _run_awaitable(
        door(title="Outside PR", body="b", priority="high", key="pr", links=[], anchor_id="c1")
    )
    assert sentence.startswith("Notified (id n_")
    assert raised and session._run_notified_banner is True
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
    assert session._finalize_attention_notify(_end("")) is True, (
        "an item that never bannered must not suppress its run's completion banner"
    )


def _end(kind: str) -> Any:
    from local_operator.harness.types import AgentEndEvent

    return AgentEndEvent(messages=[], error="provider exploded" if kind == "error" else None)
