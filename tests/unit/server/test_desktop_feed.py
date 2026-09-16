"""The machine-wide desktop feed: what it ships, and what it must never do.

Two defects live here.

The first is a GAP: a completion in a session the desktop app is not displaying
produced nothing at all, because every notification channel was per session and
a bridge exists only while a route holds one. The feed is the missing channel,
and the property that makes it safe is that it OBSERVES the same durable
authority the bridge does (``AttentionStore``) rather than re-deciding whether a
turn finished.

The second is the reason this file is aggressive about the payload: the feed and
a session bridge can both ship a banner for the same completion, and the only
thing that stops that being two banners is ``dedupe_key`` being byte-identical.
So the parity assertion below compares the two payloads FIELD BY FIELD, and the
one field they are allowed to differ on is named in the test.

Cold-path style, mirroring ``test_desktop_notifications.py``: no runtime, no
bridge, no session, no HTTP. The e2e module drives the same objects over real
loopback HTTP.

The third subject is the per-session STATUS channel (the ``session_status``
frame), which closed a latency gap rather than a notification gap: the row's
status is derived by the backend, and the only way a client could learn a new one
was to re-read the whole list, so an answered gate or a finished turn took up to
30 s to appear on a row nobody was looking at. Its tests are grouped under "The
per-session status channel" at the end of this file and pin four separate things:
the frame carries the LIST'S OWN pair (parity, so a second derivation cannot
appear), a heartbeat publishes nothing (the anti-aggressive-poll property), the
feed stays a READER (no reap, no bridge, no runtime), and the tick's cost is a
SYSCALL COUNT rather than a duration.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import local_operator.server.utils.desktop_feed as feed_module
import local_operator.session.runtime.presence as presence_module
from local_operator import procstate
from local_operator.notifications import notification_payload
from local_operator.resume import mark_session_origin
from local_operator.server.utils.desktop_feed import (
    BURST_LIMIT,
    DesktopFeed,
    FeedSubscription,
    _fingerprint,
)
from local_operator.server.utils.desktop_presence import DesktopDeliveryPublisher
from local_operator.server.utils.desktop_sessions import (
    DesktopSessionBridge,
    DesktopSessions,
)
from local_operator.session.attention import AttentionStore
from local_operator.session.catalog import (
    WEDGED_STATUS,
    load_catalog,
    status_dedupe_key,
    status_of,
)
from local_operator.session.runtime import registry
from local_operator.session.runtime.presence import (
    desktop_attending_session,
    desktop_delivery_present,
    desktop_viewing_session,
    reset_cache,
)
from local_operator.session.runtime.types import HEARTBEAT_TIMEOUT_S, SessionRecord
from local_operator.tui.notify import BODY_BACKGROUND_DIGEST, background_digest_title
from local_operator.wakes.store import write_entry


@pytest.fixture(autouse=True)
def _clear_presence_cache():
    """The presence cache is per process and keyed by root; clear it per test.

    Without this, a test that writes a lease reads a snapshot another test's
    root took — and, worse, a test asserting ABSENCE passes on a stale entry
    rather than on nothing having published.
    """
    reset_cache()
    yield
    reset_cache()


def _session(root: Path, session_id: str) -> Path:
    """A user-session directory. No origin marker means "a person made this"."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _publish(root: Path, session_id: str, kind: str = "complete", anchor: str = "a1") -> str:
    token = str(uuid.uuid4())
    AttentionStore(root / "attention.db").publish(f"session/{session_id}", token, anchor, kind)
    return token


def _feed(root: Path, *, bridged: set[str] | None = None) -> DesktopFeed:
    return DesktopFeed(root, bridged=lambda: set(bridged or ()))


def _attend(root: Path, *, session_id: str = "", **window: Any) -> DesktopDeliveryPublisher:
    """Publish a live, notify-capable lease whose window is ATTENDED.

    Built through the real publisher rather than by hand-writing JSON, so the
    aggregate's shape is the one production writes — a hand-written fixture
    would let the reader and the writer drift apart with both suites green.
    """
    window = {"exists": True, "focused": True, "visible": True, "minimized": False, **window}
    publisher = DesktopDeliveryPublisher(root)
    publisher.update(
        "sub-1",
        can_notify=True,
        can_notify_kinds=["complete", "error"],
        session_id=session_id,
        window=window,
    )
    reset_cache()
    return publisher


async def _drain(feed: DesktopFeed, subscription: FeedSubscription, count: int) -> list[Any]:
    collected: list[dict[str, Any]] = []
    async for frame in feed.events(subscription):
        collected.append(frame)
        if len(collected) >= count:
            return collected
    return collected


def _tick(feed: DesktopFeed) -> None:
    """Run one poller tick.

    ``subscribe()`` starts the real timer only when a loop is already running,
    which a synchronous test does not have — so the deterministic tests drive
    the tick directly and ``test_the_doorbell_delivers_a_completion_with_no_
    bridge_anywhere`` proves the timer is wired to it.
    """
    asyncio.run(feed._tick())


def _collect(
    feed: DesktopFeed,
    subscription: FeedSubscription,
    *,
    settle: float = 0.3,
    limit: int = 64,
) -> list[Any]:
    """Every frame the subscription can yield before the stream goes quiet.

    Drain-until-idle rather than "exactly N": the frame count depends on how
    many sessions changed, and a test pinned to a hard count fails for the
    wrong reason the day a change adds one. Frames produced by the ``_tick``
    the caller already ran are queued, so the settle window only has to cover
    the queue, not a timer.
    """

    async def scenario() -> list[Any]:
        collected: list[dict[str, Any]] = []

        async def reader() -> None:
            async for frame in feed.events(subscription):
                collected.append(frame)
                if len(collected) >= limit:
                    return

        task = asyncio.create_task(reader())
        try:
            for _ in range(60):
                before = len(collected)
                await asyncio.sleep(settle / 3)
                if collected and len(collected) == before:
                    break
            return collected
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    return asyncio.run(scenario())


def _queued(subscription: FeedSubscription) -> list[dict[str, Any]]:
    """Exactly the frames queued for this subscription, right now.

    Queue-precise rather than ``_collect``'s drain-until-quiet: ``_collect``
    iterates ``feed.events`` from the beginning every call, so it cannot say
    which TICK produced a frame.

    Takes the queue's ACCOUNTING with it, for the same reason: the feed keeps a
    running byte/entry total in step with the queue and declares a client
    overflowed when it passes ``REPLAY_BYTES``/``REPLAY_COUNT``. Popping frames
    and leaving the total standing would make a test that reads its own frames
    look like a client that never drains — and a later tick would then be dropped
    as an overflow, which reads as a missing frame rather than as a harness bug.
    """
    frames: list[dict[str, Any]] = []
    while not subscription.queue.empty():
        frame = subscription.queue.get_nowait()
        # The queue carries `None` as the OVERFLOW marker (the feed puts one
        # there when a slow client passes `REPLAY_BYTES`). A test helper that
        # reported it as a frame would make every caller's `frame["type"]` a
        # wrong answer rather than a loud one.
        if frame is not None:
            frames.append(frame)
    subscription.queued_sizes.clear()
    subscription.queued_bytes = 0
    return frames


def _notified(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [frame for frame in frames if frame["type"] == "notification"]


def test_the_open_frame_carries_the_snapshot_and_the_connection_terms(tmp_path):
    """The contract a client compiles against, asserted as a whole."""
    root = tmp_path
    sid = "a" * 12
    _session(root, sid)
    token = _publish(root, sid)
    feed = _feed(root)
    subscription = feed.subscribe()

    frames = _collect(feed, subscription)
    asyncio.run(feed.close())

    assert len(frames) == 1
    opened = frames[0]
    assert opened["type"] == "open"
    assert "session_id" not in opened, "the feed is not a session; no id may be invented"
    assert opened["epoch"] == feed.epoch
    assert opened["seq"] == 1
    payload = opened["payload"]
    assert payload["subscription_id"] == subscription.id
    assert payload["heartbeat_seconds"] == 15.0
    assert payload["lease_seconds"] == 45.0
    assert payload["watch_ttl_seconds"] == 45.0
    assert isinstance(payload["catalogue_revision"], int)
    # The snapshot is the point of the open frame: a client that has just
    # connected learns which conversations are unread without a `sessions.list`.
    state = payload["attention"][f"session/{sid}"]
    assert state["completion_token"] == token
    assert state["unseen"] is True
    # ...and it must NOT carry the catalogue's ROWS: those cost a preview read
    # per row, which is the cost this design removes from the sidebar's poll.
    assert "rows" not in payload
    # `supported` is absent on purpose — only a live runtime can answer it, and
    # the renderer's merge preserves the value it already holds rather than
    # letting a catalogue-shaped frame clear it.
    assert "supported" not in state


def test_a_completion_published_before_the_connection_is_never_announced(tmp_path):
    """THE BASELINE RULE. A reconnect must not flood, and history is not news.

    The same rule the bridge's ``if previous:`` guard and the store's no-flood
    bootstrap apply: what recovers a completion missed while the app was away is
    the durable ``unseen`` mark in the snapshot, not a banner about last night.
    """
    root = tmp_path
    old, new = "b" * 12, "c" * 12
    _session(root, old)
    _session(root, new)
    _publish(root, old)

    feed = _feed(root)
    # The baseline is taken with `old` already published, which is exactly the
    # state a client connecting to a store that finished something overnight
    # finds. It must learn that from the open frame's attention snapshot, and
    # from nothing else.
    feed._take_baseline()
    subscription = feed.subscribe()
    _publish(root, new)
    _tick(feed)
    frames = _collect(feed, subscription)
    asyncio.run(feed.close())

    announced = _notified(frames)
    assert [frame["session_id"] for frame in announced] == [new]
    # ...and the pre-connection completion IS in the snapshot, so it is not lost.
    snapshot = frames[0]["payload"]["attention"]
    assert snapshot[f"session/{old}"]["unseen"] is True


def test_the_doorbell_delivers_a_completion_with_no_bridge_anywhere(tmp_path):
    """The whole feature, driven through the REAL timer.

    The deterministic tests below call ``_tick`` directly; this one exists so at
    least one test proves the poller is actually wired to it, at the real 100 ms
    cadence, with nothing else running on the machine.
    """
    root = tmp_path
    sid = "d" * 12
    _session(root, sid)

    async def scenario() -> list[Any]:
        # Built INSIDE the loop: `subscribe()` starts the real poller only when
        # a loop is already running, and this test exists precisely to prove
        # that timer is wired to `_tick`.
        feed = _feed(root)
        subscription = feed.subscribe()

        async def until_notified() -> list[Any]:
            # Drained UNTIL THE FRAME IT IS ABOUT, not for a fixed count: the
            # envelope carries several frame types now (the ``catalogue``
            # invalidation, and the ``session_status`` edge the completion also
            # produces), so a count is a statement about how many OTHER channels
            # happened to fire in this tick rather than about the one under test.
            collected: list[dict[str, Any]] = []
            async for frame in feed.events(subscription):
                collected.append(frame)
                if frame["type"] == "notification":
                    return collected
            return collected

        task = asyncio.create_task(until_notified())
        try:
            await asyncio.sleep(0.3)
            await asyncio.to_thread(_publish, root, sid)
            return await asyncio.wait_for(task, timeout=10.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            await feed.close()

    frames = asyncio.run(scenario())

    announced = _notified(frames)
    assert len(announced) == 1, frames
    assert announced[0]["session_id"] == sid
    assert announced[0]["payload"]["kind"] == "complete"
    assert announced[0]["payload"]["completion_token"]
    # No desktop is connected and nothing else holds the session, so rung 1 did
    # not apply and the feed composes the banner itself.
    assert announced[0]["payload"]["focus_policy"] == "always"


def test_the_feed_ships_the_bridge_payload_byte_for_byte(tmp_path):
    """ONE COMPLETION, ONE BANNER, on two transports.

    What stops the pair being two banners is ``dedupe_key``, so this asserts
    field EQUALITY except for the one field the two may differ on — and asserts
    that field is the ROUTING one rather than content.
    """
    root = tmp_path
    sid = "e" * 12
    session_dir = _session(root, sid)
    token = _publish(root, sid)

    bridge_side = notification_payload(
        "complete", session_dir=session_dir, token=token, session_id=sid
    )
    feed_side = notification_payload(
        "complete",
        session_dir=session_dir,
        token=token,
        session_id=sid,
        focus_policy="always",
    )

    assert bridge_side["dedupe_key"] == feed_side["dedupe_key"]
    assert set(bridge_side) == set(feed_side)
    differing = {key for key in bridge_side if bridge_side[key] != feed_side[key]}
    assert differing == {"focus_policy"}, differing
    assert bridge_side["focus_policy"] == "when_unfocused"
    assert feed_side["focus_policy"] == "always"


def test_a_focused_app_still_banners_a_session_it_is_not_displaying(tmp_path):
    """DESIGN REVIEW BLOCKER B1, as a test.

    The commonest state of all: the app is focused on session A while B
    finishes. ``when_unfocused`` on the feed's frame would be suppressed by the
    desktop's own focus gate, rung 2 would already have silenced the runtime and
    the TUI, and the completion would reach NOBODY — the operator's reported
    symptom, made permanent by the presence mechanism that was meant to fix it.
    """
    root = tmp_path
    displayed, finishing = "1" * 12, "2" * 12
    _session(root, displayed)
    _session(root, finishing)
    publisher = _attend(root, session_id=displayed)

    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _publish(root, finishing)
    _tick(feed)
    frames = _collect(feed, subscription)
    asyncio.run(feed.close())
    publisher.close()

    announced = _notified(frames)
    assert len(announced) == 1, frames
    assert announced[0]["session_id"] == finishing
    assert announced[0]["payload"]["focus_policy"] == "always"


def test_the_session_the_app_is_displaying_raises_no_feed_banner(tmp_path):
    """The other half of B1: rung 1 already applied, so the feed composes nothing.

    A frame for the displayed session would be a banner about a card the user
    can already see, which is the interruption ``when_unfocused`` exists to
    prevent.
    """
    root = tmp_path
    displayed = "3" * 12
    _session(root, displayed)
    publisher = _attend(root, session_id=displayed)

    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _publish(root, displayed)
    _tick(feed)
    frames = _collect(feed, subscription)
    asyncio.run(feed.close())
    publisher.close()

    assert _notified(frames) == []


def test_a_windowless_app_is_not_treated_as_displaying_anything(tmp_path):
    """DESIGN REVIEW m2, backend half.

    On macOS the app survives the last window's closure in the dock and its
    record may still name the session it was showing. A routing decision that
    read that id would treat a closed window as "the card is on screen" and
    raise no banner for the one conversation the user cannot see.
    """
    root = tmp_path
    stale = "4" * 12
    _session(root, stale)
    publisher = _attend(root, session_id=stale, exists=False)

    assert desktop_viewing_session(root) == ""
    assert desktop_attending_session(stale, root) is False
    # It can still ATTEMPT a banner — Electron's Notification is
    # window-independent — which is what keeps rung 2 eligible.
    assert desktop_delivery_present(root, "complete") is True

    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _publish(root, stale)
    _tick(feed)
    frames = _collect(feed, subscription)
    asyncio.run(feed.close())
    publisher.close()

    announced = _notified(frames)
    assert len(announced) == 1, frames
    assert announced[0]["payload"]["focus_policy"] == "always"


def test_a_session_with_a_live_bridge_is_not_composed_again(tmp_path):
    """The steady-state guard against two banners for one completion."""
    root = tmp_path
    sid = "5" * 12
    _session(root, sid)
    feed = _feed(root, bridged={f"session/{sid}"})
    feed._take_baseline()
    subscription = feed.subscribe()
    _publish(root, sid)
    _tick(feed)
    frames = _collect(feed, subscription)
    asyncio.run(feed.close())

    assert _notified(frames) == []


def test_an_acknowledgement_publishes_an_attention_frame_that_clears_the_mark(tmp_path):
    """A read is a durable store change and must reach the feed without a poll."""
    root = tmp_path
    sid = "6" * 12
    _session(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    token = _publish(root, sid)
    _tick(feed)
    AttentionStore(root / "attention.db").acknowledge(f"session/{sid}", token)
    _tick(feed)

    frames = _collect(feed, subscription)
    asyncio.run(feed.close())

    attention = [frame for frame in frames if frame["type"] == "attention"]
    assert attention, frames
    assert attention[-1]["session_id"] == sid
    assert attention[-1]["payload"]["unseen"] is False


def test_a_subagent_child_is_never_announced(tmp_path):
    """A machine's delegated run is not a conversation to banner about."""
    root = tmp_path
    parent, child = "7" * 12, "8" * 12
    _session(root, parent)
    mark_session_origin(_session(root, child), "subagent")

    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _publish(root, parent)
    _publish(root, child)
    _tick(feed)
    frames = _collect(feed, subscription)
    asyncio.run(feed.close())

    scoped = [frame for frame in frames if frame["type"] == "notification"]
    assert scoped, frames
    assert {frame["session_id"] for frame in scoped} == {parent}


def test_a_new_session_directory_publishes_a_catalogue_invalidation(tmp_path):
    """The event that replaces the 5 s ``sessions.list`` timer."""
    root = tmp_path
    _session(root, "9" * 12)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    feed._catalogue_probed_at = 0.0
    _session(root, "a1" * 6)
    _tick(feed)
    frames = _collect(feed, subscription)
    asyncio.run(feed.close())

    catalogue = [frame for frame in frames if frame["type"] == "catalogue"]
    assert catalogue, frames
    assert isinstance(catalogue[0]["payload"]["revision"], int)


def test_a_burst_beyond_the_ceiling_collapses_to_one_digest(tmp_path):
    """A fleet finishing at once must not become a stack of banners."""
    root = tmp_path
    ids = [f"{index:012x}" for index in range(BURST_LIMIT + 3)]
    for session_id in ids:
        _session(root, session_id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    for session_id in ids:
        _publish(root, session_id)
    _tick(feed)
    frames = _collect(feed, subscription)
    asyncio.run(feed.close())

    announced = _notified(frames)
    assert len(announced) == BURST_LIMIT + 1, announced
    digest = announced[-1]["payload"]
    assert digest["burst_count"] == 3
    assert digest["title"] == background_digest_title(3)
    assert digest["body"] == BODY_BACKGROUND_DIGEST
    # No single completion owns a digest, so its key must not collide with a
    # member's own: a digest is not a duplicate of a per-session banner.
    assert digest["dedupe_key"].startswith("burst:")
    assert digest["session_ids"] == ids[-3:]
    # The per-session frames are the FIRST of the burst by completion sequence,
    # so the newest three are the ones collapsed and nothing is announced twice.
    assert {frame["session_id"] for frame in announced[:-1]} == set(ids[:BURST_LIMIT])


def test_the_backlog_bound_says_gap_and_closes_rather_than_dropping_frames(tmp_path):
    """A client too slow to keep up is told, not silently given a hole."""
    root = tmp_path
    feed = _feed(root)
    subscription = feed.subscribe()
    for _ in range(600):
        feed._publish("heartbeat", {"ts": 0.0})

    assert subscription.overflow is True
    frames = _collect(feed, subscription, limit=400)
    asyncio.run(feed.close())

    assert frames[-1]["type"] == "gap"
    assert frames[-1]["payload"]["reason"] == "overflow"
    assert frames[-1]["payload"]["subscription_id"] == subscription.id


def test_a_completion_is_not_replayed_to_a_late_subscriber(tmp_path):
    """``replay=False`` semantics, implemented as the per-subscriber baseline."""
    root = tmp_path
    sid = "b2" * 6
    _session(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    early = feed.subscribe()
    _publish(root, sid)
    _tick(feed)
    late = feed.subscribe()
    _tick(feed)
    late_frames = _collect(feed, late)
    asyncio.run(feed.close())

    assert _notified(late_frames) == []
    # The floor is a durable completion sequence now, not the SSE envelope
    # counter (review round 1, R3): ``early`` connected before the publication,
    # ``late`` after it, so the store's own counter separates them by exactly
    # one — a relationship the envelope counter could only approximate.
    assert early.baseline_completion_sequence == 0
    assert late.baseline_completion_sequence == 1


def test_the_feed_acquires_no_bridge_and_spawns_no_runtime(tmp_path):
    """THE TEST THAT KEEPS THE FEED FROM BECOMING A SPAWNER.

    The tempting optimisation is to reuse the bridge's composer, which acquires
    a session — that would make watching a 200-row catalogue build 200 cold
    facades and 200 SQLite poll loops, and would take ``BRIDGE_COUNT`` with it.
    """
    root = tmp_path
    sid = "c2" * 6
    _session(root, sid)
    pool = DesktopSessions(root)
    assert pool.bridges == {}
    before = _live_threads()

    feed = DesktopFeed(root, bridged=lambda: set(pool.bridges))
    feed._take_baseline()
    subscription = feed.subscribe()
    _publish(root, sid)
    _tick(feed)
    _collect(feed, subscription)
    asyncio.run(feed.close())

    assert pool.bridges == {}, "the feed acquired a session bridge"
    # The wider detector: a runtime started behind the bridge table's back would
    # leave a thread or a child process alive that `bridges` cannot show. The
    # feed's own poller is a task on the caller's loop, so it adds no thread.
    assert _live_threads() <= before, "a feed cycle left a thread behind"


def _live_threads() -> set[int]:
    """Live thread ids. ``Thread.ident`` is ``int | None`` until it has run."""
    return {
        thread.ident
        for thread in threading.enumerate()
        if thread.is_alive() and thread.ident is not None
    }


def test_the_presence_lease_is_never_read_as_a_watch_lease(tmp_path):
    """A delivery claim must not create residency for any session.

    The presence is a machine-wide REACHABILITY answer. Reading it as "somebody
    is watching session X" would suppress that session's banner for a person
    who is not looking at it — and, in the other direction, must never attach
    anything.
    """
    root = tmp_path
    sid = "d2" * 6
    _session(root, sid)
    pool = DesktopSessions(root)
    publisher = _attend(root, session_id=sid)

    assert desktop_delivery_present(root, "complete") is True
    assert desktop_viewing_session(root) == sid
    assert desktop_attending_session(sid, root) is True
    # The gate kind is deliberately NOT covered: the machine-wide feed carries
    # completions only, so a parked `ask` must keep its per-session lease.
    assert desktop_delivery_present(root, "ask") is False
    # Reading the lease attaches nothing anywhere.
    assert pool.bridges == {}
    publisher.close()


def test_the_presence_module_is_importable_without_the_server_stack():
    """Stdlib-only by contract, like ``viewers`` and ``registry``.

    The runtime reads this on the announce path, so it must not drag an HTTP
    stack or a terminal UI into a session process to answer one stat.
    """
    source = Path(presence_module.__file__).read_text()
    for forbidden in ("fastapi", "uvicorn", "textual", "local_operator.server"):
        assert forbidden not in source, f"presence.py must not reach {forbidden}"


def test_the_feed_never_imports_the_ui_or_a_runtime():
    """The feed composes and publishes; it must not acquire or spawn.

    A written check rather than an import-graph one: the failure it guards is
    somebody reaching for the bridge's composer "to avoid duplicating the
    payload", which is the refactor that turns this channel into a spawner.
    """
    from local_operator.server.utils import desktop_feed

    source = Path(desktop_feed.__file__).read_text()
    for forbidden in ("local_operator.mobile", "AttachClient", "attach_existing", "acquire("):
        assert forbidden not in source, f"desktop_feed must not reach {forbidden}"


def test_the_absence_of_a_presence_file_is_the_absent_answer(tmp_path):
    """An old app, or none at all, must read as "nothing here" — never as a lease."""
    assert desktop_delivery_present(tmp_path, "complete") is False
    assert desktop_viewing_session(tmp_path) == ""
    assert desktop_attending_session("e2" * 6, tmp_path) is False


def test_a_stale_or_dead_lease_is_reaped(tmp_path):
    """The two reaping rules, which the presence shares with ``scan_viewers``."""
    root = tmp_path
    path = presence_module.delivery_path(root)
    path.write_text(
        json.dumps(
            {
                "pid": 999_999_999,
                "can_notify": True,
                "can_notify_kinds": ["complete", "error"],
                "subscribers": 1,
                "window": {"exists": True, "focused": True, "visible": True, "minimized": False},
                "session_id": "",
                "heartbeat_at": 0.0,
            }
        )
    )
    reset_cache()
    assert desktop_delivery_present(root, "complete") is False
    assert presence_module.read_delivery(root).present is False


def test_the_bridge_exclusion_matches_the_real_pool(tmp_path: Path) -> None:
    """R10: the hook must speak the pool's key domain AND the pool's liveness.

    Everything here runs against a real ``DesktopSessions`` and a real
    ``DesktopSessionBridge`` registered in its real ``bridges`` map — never a
    hand-built set. That is the point of the test: the defect was precisely that
    the unit test's stand-in (prefixed keys) disagreed with what production
    supplied (bare session ids), so a test that supplies the correct shape
    itself would have stayed green through the whole bug.

    Driven synchronously, like the rest of this file: ``_tick`` owns its own
    event loop, so the bridge is entered through its constructor rather than
    through ``pool.session``'s async context manager.
    """
    root = tmp_path
    sid = "ee" * 6
    _session(root, sid)
    pool = DesktopSessions(root)
    bridge = DesktopSessionBridge(root, sid, cwd=str(root))
    pool.bridges[sid] = bridge
    feed = DesktopFeed(root, bridged=pool.bridged_notify_sessions)
    feed._take_baseline()
    subscription = feed.subscribe()
    try:
        live = bridge.subscribe()
        live.expires = time.monotonic() + 45.0
        live.can_notify = True
        assert pool.bridged_notify_sessions() == {
            f"session/{sid}"
        }, "the hook is not in the feed's key domain: the exclusion is dead code"

        # ...and the LIVENESS half, which is what stops the prefix fix from
        # trading a duplicate banner for a silent hole.
        live.can_notify = False
        assert pool.bridged_notify_sessions() == set(), "a bridge that cannot notify owns nothing"
        live.can_notify = True
        live.expires = time.monotonic() - 1.0
        assert pool.bridged_notify_sessions() == set(), "an expired lease is gone"
        live.expires = time.monotonic() + 45.0
        live.overflow = True
        assert pool.bridged_notify_sessions() == set(), "an overflowing subscriber owns nothing"
        live.overflow = False
        assert pool.bridged_notify_sessions() == {f"session/{sid}"}

        # A LIVE, NOTIFYING BRIDGE OWNS THE BANNER, so the feed yields to it.
        # Drained per tick from the queue rather than through ``_collect``: the
        # question here is which TICK produced a banner, and ``_collect``
        # replays the buffer from the start on every call, which would make the
        # two ticks indistinguishable.
        _publish(root, sid)
        _tick(feed)
        assert _notified(_queued(subscription)) == []

        # THE RETAINED BUT IDLE BRIDGE. A pooled bridge whose subscriber has
        # left announces nothing, so the feed has to speak or the completion
        # reaches nobody at all.
        bridge.subscribers.pop(live.id, None)
        assert pool.bridged_notify_sessions() == set()
        _publish(root, sid)
        _tick(feed)
        assert (
            len(_notified(_queued(subscription))) == 1
        ), "an idle pooled bridge suppressed a banner nobody else raises"
    finally:
        asyncio.run(feed.close())


def test_the_bounded_recovery_publishes_when_the_doorbell_cannot_see(tmp_path):
    """R1's other half: a doorbell that misses a change must not be the end of it.

    Watching the journal sidecars is what closes the WAL case the finding names,
    but a stat tuple can only ever be evidence, not proof — so the revision is
    ALSO read on its own slow clock. This test drives that recovery with the
    doorbell deliberately blinded: the fingerprint is cached against contents
    that then change, which is the finding's own state ("the new main-file
    fingerprint with the OLD database revision"), and the only thing left that
    can notice is the authoritative read.

    Asserted in both directions, because "it published" alone would also be true
    of a tick that ignored the doorbell entirely.
    """
    root = tmp_path
    _session(root, "a1" * 6)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _publish(root, "a1" * 6)
    # THE BLIND DOORBELL: the tick below sees the fingerprint it already has, so
    # nothing in the ordinary path can reach the delta.
    feed._fingerprint = feed_module._db_fingerprint(feed.store.path)
    feed._revision_probed_at = time.monotonic()
    _tick(feed)
    assert _notified(_queued(subscription)) == [], "the blinded doorbell published anyway"

    # ...and the bound is what recovers it: past the interval, the same quiet
    # tick reads the revision and publishes exactly what the doorbell missed.
    feed._revision_probed_at = 0.0
    _tick(feed)
    announced = _notified(_queued(subscription))
    assert len(announced) == 1, announced
    assert announced[0]["session_id"] == "a1" * 6
    assert announced[0]["payload"]["completion_token"]
    asyncio.run(feed.close())


def test_presence_is_read_uncached_where_the_decision_is_terminal(tmp_path):
    """QA round 1's presence matrix, and the two rows that failed.

    The QA matrix built this state for real — a live subscription, a real
    publisher, an isolated root — and read `notification=False` for
    `presence-unfocused-same` and `presence-hidden-same`, where the expected
    answer is a banner. The cause is the presence CACHE (2 s) rather than the
    rule: the probe flipped the window state and published inside that window, so
    the decision was made on focus the user had already given up.

    That matters more here than on the announce path, because this decision is
    TERMINAL: nothing re-decides a suppressed completion, and the runtime's own
    rung 4 defers whenever a desktop is reachable, so a suppression here means no
    surface raised it at all. Hence the uncached read.

    All eight cells, so the fix cannot trade a missing banner for a duplicate
    one: the ONE suppressing state is a focused window showing THIS
    conversation, and every other row banners.
    """
    root = tmp_path
    attended = "a1" * 6
    other = "b2" * 6
    _session(root, attended)
    _session(root, other)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    publisher = _attend(root, session_id=attended)
    window_off = {"exists": True, "focused": False, "visible": False, "minimized": False}
    cells = [
        # name, window, same conversation, does a banner get raised?
        ("focused-same", {"focused": True, "visible": True}, True, False),
        ("focused-other", {"focused": True, "visible": True}, False, True),
        ("unfocused-same", {"focused": False, "visible": True}, True, True),
        ("unfocused-other", {"focused": False, "visible": True}, False, True),
        ("hidden-same", {"focused": False, "visible": False}, True, True),
        ("hidden-other", {"focused": False, "visible": False}, False, True),
        ("no-window-same", {"exists": False}, True, True),
        ("no-window-other", {"exists": False}, False, True),
    ]
    try:
        for name, window, same, expected in cells:
            # Deliberately NOT reset_cache(): the stale answer is the state under
            # test, and the first cell is what warms the cache.
            publisher.update(
                "sub-1",
                can_notify=True,
                can_notify_kinds=["complete"],
                session_id=attended,
                window={**window_off, **window},
            )
            _publish(root, attended if same else other)
            _tick(feed)
            announced = _notified(_queued(subscription))
            assert bool(announced) is expected, f"{name}: {announced}"
            if expected:
                assert announced[0]["session_id"] == (attended if same else other)
    finally:
        publisher.close()
        reset_cache()
        asyncio.run(feed.close())


def test_presence_is_read_once_per_tick_not_once_per_candidate(tmp_path, monkeypatch):
    """REVIEW ROUND 2, R14: one uncached read per candidate SET, not per candidate.

    The banner gate has to read presence UNCACHED (the assertion above), and the
    read is a `mkdir` + `chmod` + `readdir` plus one small read per record — so
    taking it once per CANDIDATE made the scenario this channel exists for, a
    fleet's completions landing in the same tick, pay N of them where one answer
    serves every row. Measured by the reviewer on the pinned head: one completion
    in a tick cost one read, twenty cost twenty.

    Counted at ``read_delivery``, which is the filesystem-level read: the call
    count alone would hide the multiplier, because on the pre-remediation code
    the extra calls were 2 s cache HITS and cost nothing.
    """
    root = tmp_path
    ids = [f"{index:012x}" for index in range(BURST_LIMIT * 5)]
    for session_id in ids:
        _session(root, session_id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    reads: list[Path | None] = []
    real = presence_module.read_delivery

    def counting(read_root=None):
        reads.append(read_root)
        return real(read_root)

    monkeypatch.setattr(presence_module, "read_delivery", counting)
    for session_id in ids:
        _publish(root, session_id)
    _tick(feed)
    for_one_tick = len(reads)
    frames = _queued(subscription)
    asyncio.run(feed.close())

    # The tick really did decide a banner per ceiling slot plus one digest, so
    # the read count below cannot be low because the tick declined to work.
    assert len(_notified(frames)) == BURST_LIMIT + 1
    assert len(ids) > BURST_LIMIT, "the burst must exceed the ceiling to be a burst"
    assert (
        for_one_tick == 1
    ), f"one tick decided {len(_notified(frames))} banners with {for_one_tick} presence reads"


@pytest.mark.parametrize("overflow_size", [1, 3], ids=["one-member", "three-member"])
def test_a_digest_names_its_members_tokens_and_does_not_preclaim_them(tmp_path, overflow_size):
    """REVIEW ROUND 1, R8: a burst digest has to be arbitrable member by member.

    The digest deliberately carries no `completion_token` — no single completion
    owns it — so a client's claim step skips it, and it USED to carry only member
    ids. Nothing then marked those members delivered: the reviewer's
    reproduction showed all three overflow members still claimable after the
    digest was emitted, so any later individual frame for one of them (another
    feed instance, the TUI, a re-delivery) was free to raise a SECOND banner for
    a completion the digest had already announced, and the per-burst cap bounded
    nothing across transports.

    Two facts, and the second is what makes the first worth having: the pairs are
    named, and they are NOT preclaimed. A frame merely being queued must not burn
    a completion, because the client may suppress the banner by its own focus
    rule — the claim belongs immediately before delivery, on the client.

    THE ONE-MEMBER CASE IS THE BOUNDARY THE FINDING NAMED (review round 2, R11):
    `BURST_LIMIT + 1` completions in one tick is the SMALLEST overflow, so the
    digest frame carries exactly one member and `burst_count == 1`. A reader
    that decides "is this a digest" from a count greater than one reads this
    frame as a private banner, finds no `completion_token` to claim, and takes no
    claim at all — so the boundary is pinned here rather than left to be inferred
    from the three-member shape.
    """
    root = tmp_path
    ids = [f"{index:012x}" for index in range(BURST_LIMIT + overflow_size)]
    for session_id in ids:
        _session(root, session_id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    tokens = {session_id: _publish(root, session_id) for session_id in ids}
    _tick(feed)
    frames = _queued(subscription)
    asyncio.run(feed.close())

    digest = _notified(frames)[-1]["payload"]
    assert digest["completion_token"] is None
    # The count is the whole remainder, and the two member fields agree with it
    # at every size — which is the invariant a count-based client test needs and
    # the reason the backend's own contract is "a digest carries member_tokens".
    assert digest["burst_count"] == overflow_size
    assert digest["session_ids"] == ids[BURST_LIMIT:]
    assert digest["member_tokens"] == [
        {"session_id": session_id, "completion_token": tokens[session_id]}
        for session_id in ids[BURST_LIMIT:]
    ]
    store = AttentionStore(root / "attention.db")
    for member in digest["member_tokens"]:
        identity = f"session/{member['session_id']}"
        # Still the digest's to win, through the SAME atomic claim a single
        # frame uses...
        assert store.claim_delivery(identity, member["completion_token"], "desktop") is True
        # ...and exactly once. This is the arbitration the finding asked for: a
        # member claimed by the digest's surface is no longer available to a
        # later individual frame or to another feed instance.
        assert store.claim_delivery(identity, member["completion_token"], "tui") is False


# ---------------------------------------------------------------------------
# The per-session status channel
#
# THE DEFECT THIS SECTION PINS. The row's status is a BACKEND-DERIVED value with
# one home (``catalog.CatalogEntry.status_code``/``status``), and until now the
# only way any client could learn a new one was to re-read the whole list: the
# attention channel ships ``row.attention`` and never ``row.status``, and no feed
# watched ``run/mobile`` or the wake index at all. So an answered gate on a row
# the user was not looking at could take 30 s to appear, and a completion needed
# a list even when the ``attention`` frame had already arrived.
#
# WHAT IS ASSERTED HERE, in one line each: the pair is the LIST'S OWN (parity, so
# a second derivation cannot appear); a heartbeat publishes nothing (the
# anti-aggressive-poll property); revisions are monotone and the list carries the
# same stamp; the probe is a READER (it reaps nothing and acquires no bridge); a
# status that predates the connection is never replayed; the tick's syscall
# budget is a COUNT, not a duration; and ``wedged`` — the transition no file
# write announces — arrives on the probe's clock.
#
# NO WALL-CLOCK ASSERTION APPEARS BELOW. The property each test asserts is
# structural (which frame, from which filesystem state, at which syscall cost),
# and this repo's timing section is explicit that a bound calibrated on this box
# is a bet on machine load rather than a fact about the code.
# ---------------------------------------------------------------------------


#: A pid that cannot be alive on either platform we run on: macOS's ``max_pid``
#: is 99998 and Linux's ``pid_max`` cannot exceed 2**22. That is what lets a test
#: plant a PROVEN-DEAD record without forking anything to produce a corpse.
_DEAD_PID = 9_000_000


def _record(session_id: str, *, pid: int | None = None, **fields: Any) -> SessionRecord:
    """A discovery record for one session, owned by THIS test process by default.

    The pid defaults to the test process's own because the status channel's
    verdict is a fact about a pid and a beat: a fabricated live pid would make
    every assertion below depend on which pids the machine happens to have.
    """
    return SessionRecord(
        pid=os.getpid() if pid is None else pid,
        kind="tui",
        session_id=session_id,
        conversation_name="a conversation",
        cwd=str(Path.cwd()),
        model_label="mock",
        control_port=1,
        control_key="0" * 64,
        **fields,
    )


def _record_publish(root: Path, session_id: str, **fields: Any) -> Path:
    """Write a record through the REAL writer, so the doorbell sees a real write."""
    return registry.publish(_record(session_id, **fields), root)


#: A pid that is ALIVE and is not this process. Discovery records are keyed by
#: pid (``record_path``), so two records for two different sessions need two pids
#: — and ``os.kill(1, 0)`` answers EPERM on every platform we run on, which
#: ``registry.pid_alive`` reads as "alive, not ours", the same as it reads for any
#: other user's process.
_FOREIGN_LIVE_PID = 1


@contextlib.contextmanager
def _extra_live_pid() -> Iterator[int]:
    """One more real live pid, for a test that needs three records at once.

    Spawned rather than invented: the pid is what ``classify`` reads, so a
    fabricated one would only be a live-looking number until the day the machine
    answered for it. Killed in the ``finally`` whatever the test does.
    """
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        yield child.pid
    finally:
        child.kill()
        child.wait()


def _listable_session(root: Path, session_id: str) -> Path:
    """A session the catalogue LISTS: a directory with a transcript in it.

    The parity assertions below compare against ``load_catalog``, so the session
    has to be one the catalogue actually returns rather than one only the feed
    knows about — an empty directory is listed by neither scanner and would make
    the comparison vacuous.
    """
    directory = _session(root, session_id)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    return directory


def _wake(root: Path, session_id: str, *, dormant: bool = False) -> Path | None:
    """Arm (or arm-and-dormant) one session's wake entry, through the real writer.

    ``stopped_at`` is the index's own marker of a STRICTLY stopped session (its
    wakes will never fire), which is the ``dormant`` spelling in the catalogue.
    """
    preserve = {"stopped_at": int(time.time() * 1000)} if dormant else None
    return write_entry(
        root,
        session_id,
        cwd=str(Path.cwd()),
        schedules=[{"next_due_at": int(time.time() * 1000) + 60_000, "cwd": str(Path.cwd())}],
        preserve=preserve,
    )


def _statuses(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [frame for frame in frames if frame["type"] == "session_status"]


@contextlib.contextmanager
def _io_watch(registry_dir: Path) -> Iterator[dict[str, Any]]:
    """Count the filesystem calls one block makes, and which files it read.

    The counting pattern ``tests/unit/session/test_catalog_scan_cost.py`` uses for
    the same kind of claim: the property is "one stat and no record reads on a
    quiet tick", which a COUNT states exactly and a duration only approximates
    once machine load is in the picture.
    """
    counts: dict[str, Any] = {
        "stat": 0,
        "lstat": 0,
        "scandir": 0,
        "record_reads": 0,
        "scandir_paths": [],
    }
    originals: dict[str, Any] = {}
    for name in ("stat", "lstat", "scandir"):
        real = getattr(os, name)
        originals[name] = real

        def wrap(real: Callable[..., Any] = real, name: str = name) -> Any:
            def counting(*args: Any, **kwargs: Any) -> Any:
                counts[name] += 1
                if name == "scandir" and args:
                    counts["scandir_paths"].append(str(args[0]))
                return real(*args, **kwargs)

            return counting

        setattr(os, name, wrap())
    real_read_text = Path.read_text

    def read_text(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.parent == registry_dir:
            counts["record_reads"] += 1
        return real_read_text(self, *args, **kwargs)

    setattr(Path, "read_text", read_text)
    try:
        yield counts
    finally:
        setattr(Path, "read_text", real_read_text)
        for name, real in originals.items():
            setattr(os, name, real)


def test_the_gate_edge_publishes_once_with_the_backends_own_spelling(tmp_path):
    """THE REPORTED SYMPTOM: an answered gate took up to 30 s to appear.

    The only trace an answered gate leaves is the discovery record
    ``set_record_pending`` writes, and the feed watched nothing that could see
    it. The frame carries the pair the LIST derives — asserted here on the
    backend's own spelling and label, because the feed composes neither.
    """
    root = tmp_path
    sid = "e1" * 6
    _listable_session(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    # The 1 s probe is gated SHUT for this test, so every frame below has to come
    # over the DOORBELL — the 10 Hz path the reported symptom is about. Without
    # this the test would also pass on the slow clock, which is the bound the
    # change exists to remove.
    feed._status_probed_at = time.monotonic()

    _record_publish(root, sid, pending="approval")
    _tick(feed)
    frames = _statuses(_queued(subscription))
    assert [frame["session_id"] for frame in frames] == [sid], frames
    assert frames[0]["payload"] == {"code": "approval", "label": "Approval needed", "revision": 1}

    # The answer clears the gate. Exactly one more frame, and its code is the
    # post-gate one: the pair changed, which is the whole emission rule.
    _record_publish(root, sid)
    _tick(feed)
    cleared = _statuses(_queued(subscription))
    assert len(cleared) == 1, cleared
    assert cleared[0]["payload"] == {"code": "attached", "label": "Open", "revision": 2}
    # ...and the doorbell ended with its cache current, which is what makes the
    # NEXT quiet tick cheap rather than a re-read.
    assert feed._registry_fingerprint == _fingerprint(feed._registry_dir)
    asyncio.run(feed.close())


def test_a_first_record_is_delivered_by_the_doorbell_and_not_the_probe(tmp_path):
    """A session the feed has NEVER cached still rides the 10 Hz clock.

    The doorbell's rule is "re-read the records whose own file moved", and a
    session that had no record has no cached file to compare against — so the
    directory moves and nothing the fast path knows about does. Waiting for the
    probe there would make a runtime starting up (or an exec run appearing) an
    order of magnitude slower than the edges this channel exists for, and would
    make the first edge of every cold session the slow one. The fallback that
    answers it looks properly once, and only on a tick whose directory already
    moved — the quiet tick still costs one stat and no reads.
    """
    root = tmp_path
    sid = "eb" * 6
    _listable_session(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    feed._status_probed_at = time.monotonic()  # the probe cannot fire

    _record_publish(root, sid, pending="approval")
    _tick(feed)
    frames = _statuses(_queued(subscription))
    assert [frame["session_id"] for frame in frames] == [sid], frames
    assert frames[0]["payload"]["code"] == "approval"
    asyncio.run(feed.close())


def test_a_heartbeat_rewrite_publishes_nothing(tmp_path):
    """THE ANTI-AGGRESSIVE-POLL PROPERTY, and why the dedupe is on the PAIR.

    Every live session rewrites its whole record every 15 s
    (``HEARTBEAT_INTERVAL_S``) through the same staged write every real edge
    uses. A feed that published on "the record file moved" would emit a frame per
    live session per heartbeat, forever — strictly worse than the 5 s poll this
    replaces. The pair is read from
    ``pending``/``busy``/``detached``/``leaving``, none of which a heartbeat
    touches, so the answer must be: nothing, even though the doorbell RINGS.
    """
    root = tmp_path
    sid = "e2" * 6
    _listable_session(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    _record_publish(root, sid, pending="approval")
    _tick(feed)
    assert len(_statuses(_queued(subscription))) == 1

    for _ in range(3):
        _record_publish(root, sid, pending="approval")
        # THE CLAIM UNDER TEST IS NOT "NOTHING HAPPENED". The directory really
        # moved, so the doorbell really rang and the record really was re-read;
        # asserting that here is what stops this test passing on a feed that
        # simply failed to notice the write.
        assert _fingerprint(feed._registry_dir) != feed._registry_fingerprint
        _tick(feed)
        assert _statuses(_queued(subscription)) == []
    asyncio.run(feed.close())


def test_the_frame_and_the_list_derive_the_status_from_one_home(tmp_path):
    """PARITY: the frame is a second CALLER of the precedence, never a home.

    The design's whole safety argument is that the feed cannot disagree with the
    list about what a row's status is, because both come through
    ``catalog.status_of``. This is the test that fails if a second derivation is
    ever written: it compares the two SURFACES' answers for the same on-disk
    state, so a re-ordered branch or a code named by hand in the feed shows up as
    a disagreement instead of as a green suite.
    """
    root = tmp_path
    gate, working, resident, armed, dormant, finished, failed, cold, wedged, dead = (
        f"{index:012x}" for index in range(10)
    )
    for session_id in (
        gate,
        working,
        resident,
        armed,
        dormant,
        finished,
        failed,
        cold,
        wedged,
        dead,
    ):
        _listable_session(root, session_id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    # Five live pids, because a record is keyed by one: this process, pid 1, and
    # two spawned sleepers. Each session below gets its own, so no write can
    # clobber another's record.
    with _extra_live_pid() as spare, _extra_live_pid() as aged:
        _record_publish(root, gate, pending="approval")
        _record_publish(root, working, pid=_FOREIGN_LIVE_PID, busy=True)
        _record_publish(root, resident, pid=spare, detached=True)
        # THE ARM THAT CARRIES A CLOCK. A quiet beat is what makes the verdict
        # ``wedged``, and the label that comes with it embeds the age — so this
        # is the one arm where the two surfaces could disagree about a STRING
        # rather than about a code, and the age is parked deep inside the minute
        # (125 s, said as ``2m``) so that the two reads either side of the frame
        # cannot straddle a unit boundary and make the comparison flaky.
        quiet_path = _record_publish(root, wedged, pid=aged, busy=True)
        quiet = _record(wedged, pid=aged, busy=True)
        quiet.heartbeat_at = time.time() - HEARTBEAT_TIMEOUT_S - 80
        quiet_path.write_text(json.dumps(quiet.to_json()), encoding="utf-8")
        # THE ARM THAT CARRIES A REAP. A pid that cannot be alive on either
        # platform, so it is proven dead without forking a corpse into being —
        # and the point of the arm is that neither surface may describe it.
        _record_publish(root, dead, pid=_DEAD_PID)
        _wake(root, armed)
        _wake(root, dormant, dormant=True)
        _publish(root, finished, kind="complete")
        _publish(root, failed, kind="error")
        _tick(feed)
        frames = {
            frame["session_id"]: frame["payload"] for frame in _statuses(_queued(subscription))
        }

        listed = {entry.id: (entry.status_code, entry.status) for entry in load_catalog(root)}
    # Every state with an EVENT behind it is announced...
    assert set(frames) == {gate, working, resident, armed, dormant, finished, failed, wedged}
    # ...and each frame carries exactly what the list derives for the same
    # on-disk state. Nothing else in this file needs to know what the vocabulary
    # is, which is the point: there is one precedence and both surfaces call it.
    for session_id, payload in frames.items():
        assert (payload["code"], payload["label"]) == listed[session_id], session_id
    # The wedged arm is in the set above rather than only in the age test below,
    # so a second derivation of the label — the one string that is not a
    # constant — is compared against the list's own spelling.
    assert frames[wedged]["code"] == "wedged"
    assert frames[wedged]["label"].startswith(WEDGED_STATUS), frames[wedged]["label"]
    # A session nothing happened to is not a candidate at all: no frame, because
    # its pair (``recent``) is what the client's own list already says.
    assert cold not in frames
    assert listed[cold] == ("recent", "Recent")
    # A DEAD PID IS NO RECORD ON BOTH SURFACES (review round 1, MINOR 1). This is
    # QA's divergence 5 as an assertion: the feed publishes nothing for the
    # corpse, and the list must not describe it either — a list that painted the
    # dead record's ``busy``/``attached`` did so with the SAME status_revision the
    # frame carried, so the client's strictly-greater guard kept the wrong value
    # for a whole 30 s poll. Both readers now take the one rule.
    assert dead not in frames
    assert listed[dead] == ("recent", "Recent")
    dead_row = feed._row_for(dead)
    assert dead_row is not None  # listable: the catalogue lists it too
    assert status_of(dead_row, feed._attention.get(dead)) == listed[dead]
    asyncio.run(feed.close())


def test_the_wake_index_is_watched_both_ways(tmp_path):
    """Arming AND disarming a wake, which the record doorbell cannot see at all.

    The index lives outside ``run/mobile``, so nothing about a wake moves the
    directory the doorbell watches. The disarm direction is the one a candidate
    set built from the CURRENT index would miss: the entry is gone, so the
    session is not in the index any more — and the row would keep a wake glyph
    until the next 30 s poll.
    """
    root = tmp_path
    sid = "e5" * 6
    _listable_session(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    _wake(root, sid)
    feed._status_probed_at = 0.0
    _tick(feed)
    armed = _statuses(_queued(subscription))
    assert [frame["payload"]["code"] for frame in armed] == ["scheduled"]

    # Disarmed: the entry is REMOVED from the index, which is the case that is
    # invisible to a diff of nothing.
    from local_operator.wakes.store import remove_entry

    remove_entry(root, sid)
    feed._status_probed_at = 0.0
    _tick(feed)
    disarmed = _statuses(_queued(subscription))
    assert [frame["payload"]["code"] for frame in disarmed] == ["recent"]
    assert disarmed[0]["payload"]["revision"] > armed[0]["payload"]["revision"]
    asyncio.run(feed.close())


def test_revisions_are_monotone_and_the_lists_stamp_agrees(tmp_path):
    """The two writers' ordering token, over HTTP's own path.

    Both the frame and the list ship the same fact now, and the client's rule is
    "keep the frame's value if the epochs match and the frame's revision is
    higher". That rule is only sound if the counter is per session, strictly
    increasing, and carried by the list at exactly the value the last frame had —
    which is what this asserts end to end, through ``DesktopSessions.list``.
    """
    root = tmp_path
    sid = "e3" * 6
    _listable_session(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    revisions = []
    for fields in ({"pending": "approval"}, {"busy": True, "pending": None}, {}):
        _record_publish(root, sid, **fields)
        _tick(feed)
        published = _statuses(_queued(subscription))
        assert published, fields
        revisions.append(published[-1]["payload"]["revision"])
    assert revisions == [1, 2, 3]

    pool = DesktopSessions(root)
    rows = asyncio.run(pool.list(50, status_stamps=feed.status_stamps()))
    row = next(entry for entry in rows if entry["id"] == sid)
    assert row["status_epoch"] == feed.epoch
    assert row["status_revision"] == 3
    assert row["status"]["code"] == "attached"

    # ADDITIVE, proven rather than asserted: a caller that passes no stamps gets
    # the response it always got, byte for byte in the fields that matter.
    plain = asyncio.run(pool.list(50))
    for entry in plain:
        assert "status_revision" not in entry
        assert "status_epoch" not in entry


def test_a_preexisting_status_is_not_replayed_to_a_new_connection(tmp_path):
    """``_take_baseline`` primes the pair, so history is not news.

    The same no-flood rule the attention baseline applies: a client that connects
    to a machine with a gate already parked learns it from the ``open`` snapshot's
    list, not from a frame about a transition that happened before it arrived.
    """
    root = tmp_path
    sid = "e6" * 6
    _listable_session(root, sid)
    _record_publish(root, sid, pending="approval")
    _wake(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    for _ in range(3):
        feed._status_probed_at = 0.0
        _tick(feed)
    assert _statuses(_queued(subscription)) == []

    # ...and the very next CHANGE still arrives, so this is a baseline and not a
    # feed that has gone quiet for good.
    _record_publish(root, sid)
    _tick(feed)
    assert [frame["payload"]["code"] for frame in _statuses(_queued(subscription))] == ["attached"]
    asyncio.run(feed.close())


def test_a_stale_heartbeat_publishes_wedged_with_no_write(tmp_path):
    """The transition NO doorbell can see: an age crossing.

    ``live -> wedged`` is ``HEARTBEAT_TIMEOUT_S`` elapsing with the file
    untouched, so the record's fingerprint does not move and the doorbell stays
    silent by construction. The write below is the test's way of AGEING the beat;
    the caches are then refreshed to the state on disk, which is the statement
    "the feed has already seen this file exactly as it is" — i.e. exactly the
    situation production is in when time alone crosses the timeout.
    """
    root = tmp_path
    sid = "e7" * 6
    _listable_session(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    path = _record_publish(root, sid)
    _tick(feed)
    assert [frame["payload"]["code"] for frame in _statuses(_queued(subscription))] == ["attached"]

    aged = _record(sid)
    aged.heartbeat_at = time.time() - HEARTBEAT_TIMEOUT_S - 30.0
    path.write_text(json.dumps(aged.to_json()), encoding="utf-8")
    feed._registry_fingerprint = _fingerprint(feed._registry_dir)
    feed._record_fingerprints[sid] = _fingerprint(path)
    feed._status_probed_at = 0.0
    _tick(feed)

    wedged = _statuses(_queued(subscription))
    assert [frame["payload"]["code"] for frame in wedged] == ["wedged"]
    # The label is the one home's sentence, with the measured age appended — the
    # same string the sidebar's tooltip shows.
    assert wedged[0]["payload"]["label"].startswith(WEDGED_STATUS)
    asyncio.run(feed.close())


def test_the_probe_is_a_reader(tmp_path):
    """READ-ONLY: no reap, no bridge, no runtime — the property this route lives by.

    ``registry.scan`` sweeps a proven-dead record aside by default, and the feed
    calls it once a second: without ``reap=False`` the desktop app's feed would be
    the process that destroys the evidence an incident reader is about to look at.
    A dead record is not a status either — it is NO record, which is the view the
    list lands on too.
    """
    root = tmp_path
    dead, live = "e8" * 6, "e9" * 6
    _listable_session(root, dead)
    _listable_session(root, live)
    pool = DesktopSessions(root)
    feed = DesktopFeed(root, bridged=lambda: set(pool.bridges))
    feed._take_baseline()
    subscription = feed.subscribe()

    dead_path = registry.publish(_record(dead, pid=_DEAD_PID), root)
    _record_publish(root, live)
    feed._registry_fingerprint = None  # the doorbell sees both writes
    feed._status_probed_at = 0.0
    _tick(feed)
    # A SECOND pass of the probe, which is where a reaping scan would have moved
    # the file: one tick could pass on the doorbell alone.
    feed._status_probed_at = 0.0
    _tick(feed)

    assert dead_path.exists(), "the feed reaped a record it does not own"
    assert sorted(path.name for path in (root / "run" / "mobile").glob("*.json")) == sorted(
        [dead_path.name, f"{os.getpid()}.json"]
    )
    # The live session's frame is the positive control: the tick above really did
    # read the directory, so the dead record's silence is a decision rather than a
    # feed that never looked.
    frames = _statuses(_queued(subscription))
    assert {frame["session_id"] for frame in frames} == {live}
    assert frames[-1]["payload"]["code"] == "attached"
    assert pool.bridges == {}, "the feed acquired a session bridge"
    asyncio.run(feed.close())


def test_a_quiet_tick_costs_four_stats_and_no_record_reads(tmp_path):
    """THE I/O BUDGET, as a COUNT. One extra stat is the whole cost of watching.

    The record doorbell is one ``os.stat`` on ``run/mobile`` added to a tick that
    already made three (the attention store and its two journal sidecars), and on
    a store where nothing moved it must read NOTHING: not one record, not the
    wake index, no SQL. Every later byte on this path is paid per event rather
    than per tick, which is the difference between this and the whole-list refetch
    it replaces.
    """
    root = tmp_path
    sid = "ea" * 6
    _listable_session(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _record_publish(root, sid)
    _tick(feed)
    _queued(subscription)

    # Both probes are gated shut, so what is measured is the DOORBELL's cost: the
    # catalogue probe legitimately walks the sessions directory at 1 Hz, and
    # including it would measure a different claim.
    feed._catalogue_probed_at = time.monotonic()
    feed._status_probed_at = time.monotonic()
    with _io_watch(feed._registry_dir) as counts:
        _tick(feed)
    assert _statuses(_queued(subscription)) == []
    assert counts["stat"] + counts["lstat"] <= 4, counts
    assert counts["record_reads"] == 0, counts
    assert counts["scandir"] == 0, counts
    asyncio.run(feed.close())


def test_the_status_probe_reads_only_the_records_and_never_walks_the_store(tmp_path):
    """ONE READ PER LIVE RECORD PER SECOND, and O(live records) rather than O(store).

    The 1 s probe is the authoritative clock for the transitions no write
    announces, and the bound the design puts on it is that it scales with the
    RECORD population — a handful — and never with the session store, which is
    the 120 ms ``sessions.list`` it exists to make unnecessary.
    """
    root = tmp_path
    live = [f"{index:012x}" for index in range(4)]
    for session_id in live:
        _listable_session(root, session_id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    # A record per session needs a pid per session (the file is named for the
    # pid), and the scan reads every record file whatever its verdict, so these
    # are distinct and deliberately not live: this test measures the READS.
    for index, session_id in enumerate(live):
        _record_publish(root, session_id, pid=_DEAD_PID + index)
    _tick(feed)
    _queued(subscription)

    feed._catalogue_probed_at = time.monotonic()  # see the note above
    feed._status_probed_at = 0.0
    with _io_watch(feed._registry_dir) as counts:
        _tick(feed)
    assert counts["record_reads"] == len(live), counts
    assert str(root / "sessions") not in counts["scandir_paths"], counts
    asyncio.run(feed.close())


# ----------------------------------------------------------------------------
# Remediation round 1 (review + QA): the reader's I/O, the clock in one label,
# the invalidation a section move needs, and the order a row is painted in.
# ----------------------------------------------------------------------------


def test_a_machine_that_has_never_run_a_session_gets_no_run_directory(tmp_path):
    """THE READER CREATES NOTHING — including the directory it reads.

    Both of this channel's reads go through ``registry.scan``, which opens with
    ``run_dir()``: a mkdir plus a chmod. So the PR's own claim ("no ``mkdir`` of
    the run directory") was false as written — the first subscriber on a machine
    that had never run a session caused ``run/mobile`` 0700 to appear, and a
    directory an operator removed came back within a second, on every probe
    (review round 1, MAJOR 1 / QA Q2). The probe and the connection baseline now
    decline to scan while the directory is absent, which is what this asserts.

    The positive control is the second half: a REAL writer still creates the
    directory and the feed still sees the record on the doorbell, so the
    assertions above are a reader that looked and declined rather than one that
    never looked at all.
    """
    root = tmp_path
    session_id = "eb" * 6
    _listable_session(root, session_id)
    feed = _feed(root)
    assert not feed._registry_dir.exists(), "the fixture started with a run directory"

    feed._take_baseline()  # the connection baseline
    subscription = feed.subscribe()
    for _ in range(3):  # the 1 s probe's own clock, three times
        feed._status_probed_at = 0.0
        _tick(feed)
    assert not feed._registry_dir.exists(), "the reader created run/mobile"
    assert _statuses(_queued(subscription)) == []

    # A writer creates it, and the doorbell delivers that first record.
    _record_publish(root, session_id)
    assert feed._registry_dir.is_dir(), "the writer did not create the run directory"
    feed._registry_fingerprint = None
    _tick(feed)
    assert [frame["payload"]["code"] for frame in _statuses(_queued(subscription))] == ["attached"]
    asyncio.run(feed.close())


def test_the_status_probe_forks_once_whatever_the_record_population(tmp_path, monkeypatch):
    """Q1's bound, at the level the pathology was measured: the 1 s probe.

    ``classify``'s derived policy spends a ``ps`` fork on every record whose
    heartbeat has gone quiet, and the probe re-runs the whole scan every second —
    so a population of quiet-but-ALIVE records cost one fork per record per
    second. Measured on head: 88 forks on every probe with 200 records, a probe
    of 1.7 s, and the 10 Hz doorbell down to 0.5 Hz. ``ps`` answers for a pid
    LIST, so the probe now asks once for the whole quiet set.

    Counted as FORKS rather than as a duration: the property is which process was
    created, and a fork count is the same number on an idle host and one at load
    100. ``at most one`` rather than ``exactly one`` because Linux answers the
    same set out of ``/proc`` with no fork at all — on macOS this is the
    assertion that a per-record probe fails with three invocations instead of
    one.
    """
    asks: list[list[str]] = []

    class _NoFork:
        """``subprocess`` minus the fork, answering the way ``ps`` would.

        It must ANSWER rather than return nothing: a batch that answers nothing is
        the FAILURE path, which falls back to probing per record by design (QA
        round 2, Q6), so a silent shim would measure the fallback instead of the
        batching this test is about. ``S`` is an ordinary sleeping process — the
        answer a live pid gets.
        """

        def run(self, argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            asks.append([str(item) for item in argv])
            answered = "".join(f"{pid} S\n" for pid in str(argv[-1]).split(","))
            return subprocess.CompletedProcess(argv, 0, stdout=answered)

    monkeypatch.setattr(procstate, "subprocess", _NoFork())
    root = tmp_path
    session_ids = [f"{index:012x}" for index in range(3)]
    for session_id in session_ids:
        _listable_session(root, session_id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _queued(subscription)
    with _extra_live_pid() as spare:
        # One pid per session (a record is keyed by its pid), all three alive and
        # all three past the derived probe gate: the only population that spends
        # the probe at all.
        pids = [os.getpid(), _FOREIGN_LIVE_PID, spare]
        for session_id, pid in zip(session_ids, pids, strict=True):
            path = _record_publish(root, session_id, pid=pid, busy=True)
            quiet = _record(session_id, pid=pid, busy=True)
            quiet.heartbeat_at = time.time() - HEARTBEAT_TIMEOUT_S - 30
            path.write_text(json.dumps(quiet.to_json()), encoding="utf-8")
        asks.clear()
        feed._status_probed_at = 0.0
        _tick(feed)
        assert len(asks) <= 1, f"one probe per tick, whatever the population: {asks}"
        if asks:
            assert sorted(asks[0][-1].split(",")) == sorted(str(pid) for pid in pids)
        assert [frame["payload"]["code"] for frame in _statuses(_queued(subscription))] == [
            "wedged",
            "wedged",
            "wedged",
        ]
        # A HEALTHY population spends nothing: the policy is still spent only
        # where the answer changes what a user is told.
        for session_id, pid in zip(session_ids, pids, strict=True):
            _record_publish(root, session_id, pid=pid, busy=True)
        asks.clear()
        feed._status_probed_at = 0.0
        _tick(feed)
        assert asks == [], "a healthy population probed for zombies"
    asyncio.run(feed.close())


def test_a_wedged_rows_age_ticking_is_not_an_edge(tmp_path):
    """MINOR 2: one label carries a live clock, and the clock is not an event.

    ``CatalogEntry.status``'s ``wedged`` arm embeds ``format_duration(age)``, so
    the pair changes with the clock alone — roughly once a second for the 45-59 s
    window after the beat crosses the timeout, then once a minute, per wedged
    session, with no write and no event behind it. The channel dedupes on
    ``status_dedupe_key`` (the same derivation with its clock term removed) and
    still PUBLISHES the pair, age and all.
    """
    root = tmp_path
    session_id = "ec" * 6
    _listable_session(root, session_id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    path = _record_publish(root, session_id, busy=True)
    _tick(feed)
    assert [frame["payload"]["code"] for frame in _statuses(_queued(subscription))] == ["busy"]

    def age_it(seconds: float) -> None:
        """Move the beat back, through the real record the classifier reads."""
        quiet = _record(session_id, busy=True)
        quiet.heartbeat_at = time.time() - seconds
        path.write_text(json.dumps(quiet.to_json()), encoding="utf-8")
        feed._registry_fingerprint = _fingerprint(feed._registry_dir)
        feed._record_fingerprints[session_id] = _fingerprint(path)
        feed._status_probed_at = 0.0

    age_it(HEARTBEAT_TIMEOUT_S + 1.4)  # 46 s: wedged, and the age is in seconds
    _tick(feed)
    first = _statuses(_queued(subscription))
    assert [frame["payload"]["code"] for frame in first] == ["wedged"]
    published = first[0]["payload"]["label"]
    row = feed._row_for(session_id)
    assert row is not None  # this session is listed: the list builds a row for it
    key_before = status_dedupe_key(row, feed._attention.get(session_id))

    # One second later the label the LIST would render says 47s — the pair really
    # has moved — and the channel still publishes nothing. The row is read AFTER
    # the tick because the feed's own cache is what ``_row_for`` transcribes, and
    # a stale cache would compare the label with itself.
    age_it(HEARTBEAT_TIMEOUT_S + 2.4)
    _tick(feed)
    assert _statuses(_queued(subscription)) == [], "the clock tick alone published a frame"
    row = feed._row_for(session_id)
    assert row is not None
    attention = feed._attention.get(session_id)
    assert status_of(row, attention)[1] != published, "the fixture did not move the label"
    assert status_dedupe_key(row, attention) == key_before
    asyncio.run(feed.close())


def test_a_row_that_changes_section_invalidates_the_catalogue(tmp_path):
    """FINDING 8: a section move needs a LIST read, and the client needs telling.

    ``active`` is ``pending or unseen or live_state``, so a background session
    that finishes leaves "Previous chats" for "Active chats". The glyph reaches
    the row in ~100 ms, but placement only rides the whole-list refetch — measured
    at 7.5-8.9 s on the paired UI PR, and with "Previous chats" collapsed by
    default the user saw nothing at all for that time. The feed cannot ship rows,
    so it ships what it does ship: an invalidation the client's own (tested)
    refetch effect re-runs on.

    Two properties, and the client is what makes them load-bearing: the revision
    must be a number the client has NEVER seen (React re-runs an effect on a
    changed dependency value, so a repeated revision is no invalidation at all),
    and it must be monotone across BOTH causes — a membership move and an
    activity transition.
    """
    root = tmp_path
    session_id = "ed" * 6
    other = "ee" * 6
    for name in (session_id, other):
        _listable_session(root, name)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    # The snapshot a client is handed, straight from the method the stream
    # yields it with: ``_collect`` would drain ``events()`` and tear the
    # subscription down with it, and the frames after this are the point.
    opened = asyncio.run(feed._open_frame(subscription))
    start = opened["payload"]["catalogue_revision"]

    # No runtime, nothing owed: ``recent``, i.e. "Previous chats".
    _tick(feed)
    assert _queued(subscription) == []
    assert feed._activity_seen[session_id] is False

    # It finishes: the pair and the SECTION both move.
    _publish(root, session_id, kind="complete")
    _tick(feed)
    frames = _queued(subscription)
    kinds = [frame["type"] for frame in frames]
    assert kinds.count("catalogue") == 1, kinds
    assert kinds.index("catalogue") > kinds.index("session_status"), kinds
    assert feed._activity_seen[session_id] is True
    revision = frames[kinds.index("catalogue")]["payload"]["revision"]
    assert revision > start, "the client has seen this revision and would not refetch"

    # An edge that moves the PAIR but not the section is not an invalidation: the
    # same completion re-spelled as an error changes the code and leaves the row
    # where it is, so there is nothing for a refetch to re-file.
    _publish(root, session_id, kind="error")
    _tick(feed)
    assert "catalogue" not in [frame["type"] for frame in _queued(subscription)]

    # THE SNAPSHOT AGREES WITH THE COUNTER: a client connecting now is told the
    # same number, and the next invalidation — a SECOND session moving section —
    # is a value it has never seen. (The second completion is on a different
    # session on purpose: an edge that changes a pair without changing its row's
    # SECTION is not an invalidation, which is the other half of the rule.)
    late = feed.subscribe()
    assert asyncio.run(feed._open_frame(late))["payload"]["catalogue_revision"] == revision
    _publish(root, other, kind="complete")
    _tick(feed)
    assert [
        frame["payload"]["revision"] for frame in _queued(late) if frame["type"] == "catalogue"
    ] == [revision + 1]
    asyncio.run(feed.close())


def test_a_burst_of_transitions_costs_one_invalidation(tmp_path):
    """Finding 8, second half: at most ONE invalidation per tick.

    A burst of simultaneous transitions — a fleet starting, a batch finishing —
    must cost one refetch, not N. The causes collapse into a single counter bump
    per tick, and a later tick carries whatever else arrived after the frame.
    """
    root = tmp_path
    session_ids = [f"{index:012x}" for index in range(0x10, 0x14)]
    for session_id in session_ids:
        _listable_session(root, session_id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _queued(subscription)
    for session_id in session_ids:
        _publish(root, session_id, kind="complete")
    _tick(feed)
    frames = _queued(subscription)
    assert len(_statuses(frames)) == len(session_ids)
    assert [frame["type"] for frame in frames].count("catalogue") == 1, frames
    # Consumed, not deferred: nothing is waiting to be published on the next tick
    # (the flag is cleared by the frame that carried it).
    assert feed._catalogue_invalidated is False
    assert [
        frame for frame in _queued(subscription) if frame["type"] in ("catalogue", "session_status")
    ] == []
    asyncio.run(feed.close())


def test_the_unattributed_move_fallback_is_rate_limited(tmp_path, monkeypatch):
    """NIT 1: one fallback per probe interval, not one per doorbell tick.

    The fallback exists for a session's FIRST record — no cached file to compare
    against, so without it that first edge waits for the 1 s clock. But the
    condition it fires on ("the directory moved and no cached record did") is also
    what a write caught mid-stage looks like, and a stream of those used to cost
    a full scan on every tick, up to 10 Hz. The limit is the fallback's OWN clock:
    gating it on the probe's would suppress the very case it exists for, because
    the connection baseline stamps the probe's clock moments before a session
    writes its first record.
    """
    root = tmp_path
    session_id = "ee" * 6
    _listable_session(root, session_id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    scans: list[str] = []
    real_scan = registry.scan

    def counted_scan(*args: Any, **kwargs: Any) -> Any:
        scans.append("scan")
        return real_scan(*args, **kwargs)

    monkeypatch.setattr(registry, "scan", counted_scan)
    _record_publish(root, session_id)
    feed._registry_fingerprint = None
    _tick(feed)
    assert scans == ["scan"], "the first record did not ride the doorbell"
    assert [frame["payload"]["code"] for frame in _statuses(_queued(subscription))] == ["attached"]

    # The directory moves again with nothing cached behind the move: the second
    # fallback is inside the interval, so the probe is not re-run — and the
    # regular probe is gated by the stamp the first one left, so nothing is lost.
    os.utime(feed._registry_dir, None)
    _tick(feed)
    assert scans == ["scan"], f"the fallback ran twice inside its interval: {scans}"
    feed._unattributed_probed_at = 0.0
    os.utime(feed._registry_dir, None)
    _tick(feed)
    assert scans == ["scan", "scan"], "the fallback never came back"
    asyncio.run(feed.close())


def test_the_completion_mark_reaches_the_wire_before_the_status_that_names_it(tmp_path):
    """D1: the frame order a row is painted from — the MARK first, then the code.

    ``docs/DESKTOP_API.md`` documents the order this asserts, because the
    completion mark depends on it: a status frame carrying ``complete``/``Unseen
    completion`` that arrived BEFORE the ``attention`` frame would let a row paint
    the label while its resting ring has not been marked yet. The guarantee is
    structural — ``_emit_delta`` publishes the attention frames for the tick and
    only then hands the same read to ``_publish_status_changes`` — and this pins
    it on the wire rather than in a comment.
    """
    root = tmp_path
    session_id = "ef" * 6
    _listable_session(root, session_id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _record_publish(root, session_id, detached=True)  # a TUI holds it: idle
    _tick(feed)
    _queued(subscription)

    # The turn ends: the durable completion mark is published by the writer, and
    # the runtime's own record write follows it.
    _publish(root, session_id, kind="complete")
    _tick(feed)
    ordered = [
        frame for frame in _queued(subscription) if frame["type"] in ("attention", "session_status")
    ]
    assert [frame["type"] for frame in ordered] == ["attention", "session_status"]
    assert ordered[0]["payload"]["kind"] == "complete"
    assert ordered[1]["payload"]["code"] == "complete"
    asyncio.run(feed.close())


def test_a_failing_tick_is_visible_rather_than_a_quiet_machine(tmp_path, monkeypatch, caplog):
    """Q3: a status channel that has gone DEAD must not read as a quiet one.

    ``_poll_loop`` swallows a bad tick so a transient failure cannot stop the
    channel — but it reported it at DEBUG only, so the one failure mode this
    channel has was indistinguishable from "no status changed" unless somebody
    had DEBUG on for this module. The report is now a WARNING, rate-limited
    rather than per tick.
    """
    root = tmp_path
    session_id = "f0" * 6
    _listable_session(root, session_id)
    feed = _feed(root)
    feed.subscribe()

    async def explode() -> None:
        raise RuntimeError("the status read exploded")

    monkeypatch.setattr(feed, "_tick", explode)

    async def run() -> None:
        task = asyncio.create_task(feed._poll_loop())
        await asyncio.sleep(0.35)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    with caplog.at_level(logging.DEBUG):
        asyncio.run(run())
    warnings = [
        record
        for record in caplog.records
        if record.levelno >= logging.WARNING and record.name == feed_module.__name__
    ]
    assert warnings, "a persistently failing tick logged nothing above DEBUG"
    assert "desktop feed tick failed" in warnings[0].getMessage()
    # RATE-LIMITED, not per-tick noise: several ticks failed inside 0.35 s and
    # this is the only line they produced.
    assert len(warnings) == 1, [record.getMessage() for record in warnings]
    asyncio.run(feed.close())


def test_a_failed_publish_does_not_lose_the_section_move(tmp_path, monkeypatch):
    """Review round 2 MINOR 2: the section-move flag commits LAST too.

    ``_publish_status_changes`` states the COMMIT-LAST rule and follows it for
    ``_status_seen`` and the revisions. The activity map was advancing inside the
    build loop instead, three lines above the rule — so a raising ``_publish``
    (a payload that will not serialize, a ``_frame`` bug, MemoryError) lost the
    invalidation for good: the next tick re-derived the same pair, found the
    activity already recorded, left the flag unset, and the row stayed in the
    wrong section until the 30 s safety poll — the exact symptom this mechanism
    exists to remove. Committing it beside ``_status_seen`` gives it the retry the
    status side already had.

    The retry is the assertion: the first tick's fan-out raises, and the second
    (with a working ``_publish``) must publish the status frame AND the
    invalidation it owes.
    """
    root = tmp_path
    session_id = "ab" * 6
    _listable_session(root, session_id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _queued(subscription)

    # A background session with nothing owed: `recent`, i.e. "Previous chats".
    _publish(root, session_id, kind="complete")  # now it belongs in "Active chats"
    real_publish = feed._publish

    def explode(frame_type: str, payload: dict[str, Any], **kwargs: Any) -> None:
        if frame_type == "session_status":
            raise RuntimeError("the fan-out failed")
        real_publish(frame_type, payload, **kwargs)

    monkeypatch.setattr(feed, "_publish", explode)
    with pytest.raises(RuntimeError):
        _tick(feed)

    monkeypatch.setattr(feed, "_publish", real_publish)
    _tick(feed)
    kinds = [frame["type"] for frame in _queued(subscription)]
    assert kinds.count("session_status") == 1, kinds
    assert (
        kinds.count("catalogue") == 1
    ), "the section move was lost by the failed fan-out instead of being retried"
    asyncio.run(feed.close())
