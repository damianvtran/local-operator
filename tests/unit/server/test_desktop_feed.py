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
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

import local_operator.server.utils.desktop_feed as feed_module
import local_operator.session.runtime.presence as presence_module
from local_operator.notifications import notification_payload
from local_operator.resume import mark_session_origin
from local_operator.server.utils.desktop_feed import (
    BURST_LIMIT,
    DesktopFeed,
    FeedSubscription,
)
from local_operator.server.utils.desktop_presence import DesktopDeliveryPublisher
from local_operator.server.utils.desktop_sessions import (
    DesktopSessionBridge,
    DesktopSessions,
)
from local_operator.session.attention import AttentionStore
from local_operator.session.runtime.presence import (
    desktop_attending_session,
    desktop_delivery_present,
    desktop_viewing_session,
    reset_cache,
)
from local_operator.tui.notify import BODY_BACKGROUND_DIGEST, background_digest_title


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
        task = asyncio.create_task(_drain(feed, subscription, 3))
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
