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
from local_operator.agents import AgentEditFields, AgentRegistry
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
from local_operator.session.creation import CREATED_AT_NAME
from local_operator.session.model_selection import ENV_ALLOW_TEST_HOSTING_NOTIFY
from local_operator.session.runtime import registry
from local_operator.session.runtime.presence import (
    desktop_attending_session,
    desktop_delivery_present,
    desktop_viewing_session,
    reset_cache,
)
from local_operator.session.runtime.types import HEARTBEAT_TIMEOUT_S, SessionRecord
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
from local_operator.tui.notify import BODY_BACKGROUND_DIGEST, background_digest_title
from local_operator.wakes.store import write_entry
from tests.notification_opt_in import notification_path_opt_in


@pytest.fixture(autouse=True)
def _notification_gate_off() -> Iterator[None]:
    """Opt this module IN to the notification path, deliberately and visibly.

    ``tests/conftest.py`` arms ``LOCAL_OPERATOR_NO_NOTIFICATIONS`` for every
    test, and at import time too, because a test that reaches the operator's
    real Notification Centre is a side effect no assertion looks at. The
    banners composed here ARE the subject, so the gate is cleared once for the
    module rather than by each of the twenty tests that need it.

    The body is ``tests/notification_opt_in.notification_path_opt_in`` — the
    shared opt-in — which clears that switch AND waives the test-hosting rule:
    this module's fabricated sessions carry no journal, so only the first applies
    to most cells here, and sharing the helper is what keeps a module that later
    seeds a real selection from having to remember a second variable. The cells
    that assert the RULE close the waiver for their own body.

    Set and restored by hand rather than through ``monkeypatch``, and that is
    deliberate: that fixture is FUNCTION-scoped and SHARED with the tests, so a
    test calling ``monkeypatch.undo()`` (``test_a_compose_failure_costs_the_
    banner_not_the_attention_frame`` does, to put a composer back) would
    silently re-arm this gate mid-test and lose its own banner. The test that
    pins the gate itself (``test_a_silenced_process_composes_no_banner``) sets
    it back explicitly.
    """
    with notification_path_opt_in():
        yield


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


def test_a_bulk_acknowledgement_publishes_one_attention_frame_per_changed_session(tmp_path):
    """The machine-wide half of "clear the pile": every other window learns.

    The response to the batch is the primary path for the client that SENT it;
    every other window and surface converges through this feed, so a bulk write
    that moved two receipts has to publish two `attention` frames and no more.
    The count matters as much as the content: one frame per changed session is
    what keeps a sidebar's per-row merge honest, and a frame for a session that
    did NOT change would repaint a row whose mark is still there.

    Deliberately no `notification` frame anywhere in the log. Clearing a pile is
    not an announcement, and a bulk clear that toasted N banners would make the
    gesture that removes attention the loudest thing in the app.
    """
    root = tmp_path
    first, second, untouched = "7" * 12, "8" * 12, "9" * 12
    for session_id in (first, second, untouched):
        _session(root, session_id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    store = AttentionStore(root / "attention.db")
    tokens = {session_id: _publish(root, session_id) for session_id in (first, second, untouched)}
    _tick(feed)
    # Clear the PUBLICATION frames first. By this point the queue already holds
    # one `attention` frame per published completion, and the log under test is
    # the one the BULK WRITE produces -- reading both together would pass on two
    # frames whether the acknowledgement emitted any or not.
    #
    # Drained through `_queued`, NOT through `_collect`: `_collect` iterates
    # `feed.events()`, and CANCELING that reader unsubscribes the subscription,
    # so an acknowledgement committed afterwards is queued to nobody (measured
    # with this exact test before the change).
    published = _queued(subscription)
    assert {frame["session_id"] for frame in published if frame["type"] == "attention"} == {
        first,
        second,
        untouched,
    }

    results = store.acknowledge_many(
        [(f"session/{first}", tokens[first]), (f"session/{second}", tokens[second])]
    )
    assert [result["status"] for result in results] == ["read", "read"]
    _tick(feed)

    frames = _collect(feed, subscription)
    asyncio.run(feed.close())

    attention = [frame for frame in frames if frame["type"] == "attention"]
    assert sorted(frame["session_id"] for frame in attention) == sorted([first, second]), frames
    for frame in attention:
        assert frame["payload"]["unseen"] is False
    assert untouched not in {frame["session_id"] for frame in attention}
    assert _notified(frames) == [], "clearing a receipt announced something"


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
def _io_watch(registry_dir: Path, *, reads_under: Path | None = None) -> Iterator[dict[str, Any]]:
    """Count the filesystem calls one block makes, and which files it read.

    The counting pattern ``tests/unit/session/test_catalog_scan_cost.py`` uses for
    the same kind of claim: the property is "one stat and no record reads on a
    quiet tick", which a COUNT states exactly and a duration only approximates
    once machine load is in the picture.

    ``reads_under`` widens the read counter from "files directly in
    ``registry_dir``" (the discovery records' own shape) to "any file under this
    subtree", for the one caller whose rows are a directory deeper — an
    ``agents/<id>/agent.yml`` has the row's directory as its parent, so the
    default counter would report a zero it could not have made non-zero. Both
    callers pass it explicitly; there is no default scope that silently counts
    nothing.
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
        scope = reads_under if reads_under is not None else registry_dir
        if self.parent == scope or (reads_under is not None and scope in self.parents):
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

    BOTH CHANNELS, and the second one is not decoration. A heartbeat leaves the
    pair unchanged AND the order key unchanged, so a republish must publish on
    neither ``session_status`` nor ``catalogue``. Asserting only the status
    channel was sufficient by construction until the position comparison was
    hoisted out from behind the pair gate (M1): after that the catalogue channel
    is reachable on a tick whose pair did not move, so this test could pass while
    a no-op republish published an invalidation (measured: with the comparison
    mutated to fire for every candidate the status-only form passed while seven
    sibling guards failed — ``test_the_key_and_the_sort_cannot_drift``'s
    heartbeat case among them).
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
        frames = _queued(subscription)
        assert _statuses(frames) == [], frames
        assert [frame for frame in frames if frame["type"] == "catalogue"] == [], frames
    asyncio.run(feed.close())


def test_a_count_change_publishes_exactly_one_status_edge(tmp_path):
    """U3: the count rides the LABEL, so a count change is an EDGE.

    The requirement this pins is a latency one. The glyph reaches a client in
    under a second, while the list it could instead read its count from is on a
    30 s poll — so a count that travelled as a LIST field would be up to 30 s
    stale beside a mark that was already correct. The channel's dedupe key is
    ``(code, label)`` with the clock term removed, which is what makes the count
    an edge only because it is spelled INTO the label.

    Three claims, in order: ``0 -> 2`` publishes once and ``2 -> 1`` publishes
    once (not twice, and not zero times); a heartbeat rewrite at an unchanged
    count publishes nothing; and a record that reports NO count is neither a
    zero nor a state — it republishes the rung it leaves behind.

    The frame is asserted to be a ``{code, label, revision}`` TRIPLE. That is
    not decoration either: a count carried as a fourth payload field would be a
    shape change every client has to know about, and the design deliberately
    keeps the count inside the label instead.
    """
    root = tmp_path
    sid = "c1" * 6
    _listable_session(root, sid)
    feed = _feed(root)

    def publish(**fields: Any) -> Path:
        return _record_publish(root, sid, detached=True, **fields)

    publish(subagents_running=0, subagents_queued=0)
    feed._take_baseline()
    subscription = feed.subscribe()

    # A reported ZERO is the idle row the client already has: no edge. This is
    # the assertion that would fail if ``None``/0 were folded into the state.
    _tick(feed)
    assert _statuses(_queued(subscription)) == []

    # 0 -> 2
    publish(subagents_running=2)
    _tick(feed)
    frames = _statuses(_queued(subscription))
    assert len(frames) == 1, frames
    assert frames[0]["payload"]["code"] == "delegating"
    assert frames[0]["payload"]["label"] == "2 subagents running"
    assert set(frames[0]["payload"]) == {"code", "label", "revision"}, frames[0]["payload"]

    # 2 -> 1: one edge, and it carries the SINGULAR.
    publish(subagents_running=1)
    _tick(feed)
    frames = _statuses(_queued(subscription))
    assert len(frames) == 1, frames
    assert frames[0]["payload"]["label"] == "1 subagent running"

    # A heartbeat rewrite of the same record changes nothing the pair is read
    # from, so it must publish on NEITHER channel — the anti-aggressive-poll
    # property, here at the one rung whose label is not a constant.
    for _ in range(3):
        publish(subagents_running=1)
        assert _fingerprint(feed._registry_dir) != feed._registry_fingerprint
        _tick(feed)
        assert _queued(subscription) == []

    # A record from a build that reports no count at all: an absent KEY, written
    # straight to the file the way an older runtime's record looks. It must
    # return the row to ``idle`` rather than publishing a zero — and it does
    # publish, because the pair genuinely moved off the delegating rung.
    path = publish(subagents_running=1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["subagents_running"], payload["subagents_queued"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    _tick(feed)
    frames = _statuses(_queued(subscription))
    assert [frame["payload"]["code"] for frame in frames] == ["idle"], frames
    assert frames[0]["payload"]["label"] == "Ready"
    assert "0" not in frames[0]["payload"]["label"]
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
    (
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
        delegating,
    ) = (f"{index:012x}" for index in range(11))
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
        delegating,
    ):
        _listable_session(root, session_id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    # SIX live pids, because a record is keyed by one: this process, pid 1, and
    # three spawned sleepers. Each session below gets its own, so no write can
    # clobber another's record.
    with (
        _extra_live_pid() as spare,
        _extra_live_pid() as aged,
        _extra_live_pid() as counting,
    ):
        _record_publish(root, gate, pending="approval")
        _record_publish(root, working, pid=_FOREIGN_LIVE_PID, busy=True)
        _record_publish(root, resident, pid=spare, detached=True)
        # THE ARM THAT CARRIES A COUNT. A parent whose own turn is idle while it
        # owns running children — the state the whole channel change is for, and
        # the one arm whose label is built from numbers rather than constants.
        # `detached=True` so the row is genuinely `idle` and the rung under test
        # is the one that owns it; a second live pid, because records are keyed
        # by pid and reusing one would clobber another arm's record.
        _record_publish(
            root,
            delegating,
            pid=counting,
            detached=True,
            subagents_running=2,
            subagents_queued=1,
        )
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
    assert set(frames) == {
        gate,
        working,
        resident,
        armed,
        dormant,
        finished,
        failed,
        wedged,
        delegating,
    }
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
    rows = asyncio.run(pool.list(50, status_stamps=feed.status_stamps())).rows
    row = next(entry for entry in rows if entry["id"] == sid)
    assert row["status_epoch"] == feed.epoch
    assert row["status_revision"] == 3
    assert row["status"]["code"] == "attached"

    # ADDITIVE, proven rather than asserted: a caller that passes no stamps gets
    # the response it always got, byte for byte in the fields that matter.
    plain = asyncio.run(pool.list(50)).rows
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

    # All THREE probes are gated shut, so what is measured is the DOORBELL's cost:
    # the catalogue probe legitimately walks the sessions directory at 1 Hz, the
    # authoring probe stats the profile and team rows at 1 Hz, and including either
    # would measure a different claim.
    feed._catalogue_probed_at = time.monotonic()
    feed._status_probed_at = time.monotonic()
    feed._authoring_probed_at = time.monotonic()
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
    feed._authoring_probed_at = time.monotonic()
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
        answer a live pid gets — and the trailing start time is the second field
        the probe reads beside the state: a shim that answers the OLD two-field
        line is a batch that answers nothing, i.e. it measures the fallback.
        """

        def run(self, argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            asks.append([str(item) for item in argv])
            answered = "".join(
                f"{pid} S Mon Sep 21 09:53:01 2026\n" for pid in str(argv[-1]).split(",")
            )
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

    The section move is the case the feed's comparison SUBSUMES: a section move
    always changes the ORDER KEY's first term, and the tests after this one cover
    the moves that do not cross a section at all (see "FINDING 8'S CLASS" below).

    Two properties, and the client is what makes them load-bearing: the revision
    must be a number the client has NEVER seen (React re-runs an effect on a
    changed dependency value, so a repeated revision is no invalidation at all),
    and it must be monotone across BOTH causes — a membership move and a position
    transition.
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

    # No runtime, nothing owed: ``recent``, i.e. "Previous chats" — the section is
    # False and the key's first term is the cold tier.
    _tick(feed)
    assert _queued(subscription) == []
    assert _rank(root, session_id) == (6, 2)

    # It finishes: the pair and the SECTION both move.
    _publish(root, session_id, kind="complete")
    _tick(feed)
    frames = _queued(subscription)
    kinds = [frame["type"] for frame in frames]
    assert kinds.count("catalogue") == 1, kinds
    assert kinds.index("catalogue") > kinds.index("session_status"), kinds
    assert _rank(root, session_id) == (1, 2)
    revision = frames[kinds.index("catalogue")]["payload"]["revision"]
    assert revision > start, "the client has seen this revision and would not refetch"

    # AN EDGE THAT MOVES THE PAIR AND THE POSITION INSIDE ONE SECTION. The same
    # completion re-spelled as an error changes the code AND the ordering category
    # — complete/error/interrupted are categories 1/2/3 — so the row moves within
    # the completed block. This assertion previously said the opposite ("leaves the
    # row where it is"): that was true of a SECTION comparison and false of the row,
    # which was moving while the client was never told.
    _publish(root, session_id, kind="error")
    _tick(feed)
    respelled = [frame for frame in _queued(subscription) if frame["type"] == "catalogue"]
    assert len(respelled) == 1, respelled
    assert _rank(root, session_id) == (2, 2)
    revision = respelled[0]["payload"]["revision"]

    # THE SNAPSHOT AGREES WITH THE COUNTER: a client connecting now is told the
    # same number, and the next invalidation — a SECOND session moving from
    # "Previous chats" into the completed block — is a value it has never seen.
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


# ---------------------------------------------------------------------------
# FINDING 8'S CLASS, not its one case: A ROW THAT REORDERS INSIDE ITS SECTION
#
# ``CatalogEntry.active`` is section MEMBERSHIP (a boolean: ``pending or unseen
# or live_state``); ``CatalogEntry.rank`` is the order the sidebar RENDERS — the
# key ``rank_entries`` sorts by, whose first term is the ordering CATEGORY. A row
# can change category without leaving its section, and placement travels on a
# LIST read, so the feed owes its client an invalidation for that too. Every test
# below pins one clause of the key comparison the feed now performs.
# ---------------------------------------------------------------------------

#: The completion token the drift worlds publish and acknowledge, so the two
#: halves of the acknowledgement case can be built without threading a value
#: through the state builders. A real UUID, because the store parses tokens.
_DRIFT_TOKEN = "d1f7a000-0000-4000-8000-000000000001"


def _birth(root: Path, session_id: str, at: float) -> None:
    """Pin a session's canonical birth, which is the ORDER KEY's third term.

    Written through the same file ``session_created_at`` reads rather than left to
    the filesystem's ``st_birthtime``: two directories created microseconds apart
    would otherwise make an ordering assertion a statement about the fixture's
    timing rather than about the rule under test.
    """
    (root / "sessions" / session_id / CREATED_AT_NAME).write_text(json.dumps(at), encoding="utf-8")


def _rank(root: Path, session_id: str) -> tuple[int, int]:
    """The row's position in the CATALOGUE's own answer: the two mutable key terms.

    Read the way the LIST reads them rather than out of the feed's internals, so an
    assertion cannot pass on a map that has stopped matching the sort. The third
    term is birth and the fourth is the id: both immutable, and the feed cannot
    derive birth at all, so neither can ever produce an edge.
    """
    entry = {row.id: row for row in load_catalog(root)}[session_id]
    return entry.rank[0], entry.rank[1]


def _pair(root: Path, session_id: str) -> tuple[str, str]:
    """The row's DEDUPE PAIR, derived the way the feed derives it — from the same home.

    The assertion that a case is a COLLISION (identical pair, different position)
    has to be made on the pair the comparison actually sees, not on a transcription
    of it: the pair is built from the row's attention state as the store answers it,
    which is what ``entry_for`` gets on both sides.
    """
    entry = {row.id: row for row in load_catalog(root)}[session_id]
    store = AttentionStore(root / "attention.db")
    attention = store.state_many([f"session/{session_id}"]).get(f"session/{session_id}")
    return status_dedupe_key(entry.row, attention)


def test_a_row_that_completes_inside_active_invalidates_the_catalogue(tmp_path):
    """THE REPORTED CASE: the position moved, the SECTION did not.

    A session showing a spinner inside "Active chats" (tier 4, ``active`` True)
    finishes: its category becomes 1 (an unread completion) while ``active`` stays
    True, so the row is in the SAME section in a different slot. Comparing
    ``active`` publishes nothing at all — measured on the real backend as zero
    catalogue frames in 15 s, and still zero across ~100 ticks with the probes
    accelerated 20x, while the client's next list read already led with the
    completed row. The checkmark was prompt (the ``session_status`` frame lands in
    ~100 ms); the POSITION is what stayed stale, and only a list read can carry it.
    """
    root = tmp_path
    completer, elder = "a1" * 6, "a2" * 6
    _listable_session(root, elder)
    _listable_session(root, completer)
    _birth(root, elder, 1_700_000_000.0)
    _birth(root, completer, 1_700_000_600.0)
    _publish(root, elder)  # an older unread completion: tier 1, Active
    _record_publish(root, completer, busy=True)  # working: tier 4, Active
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    opened = asyncio.run(feed._open_frame(subscription))
    start = opened["payload"]["catalogue_revision"]
    _tick(feed)
    assert _queued(subscription) == []
    assert _rank(root, completer) == (4, 2)
    assert [entry.id for entry in load_catalog(root)] == [elder, completer]

    # It finishes: the PAIR moves busy -> complete AND the POSITION moves 4 -> 1.
    _record_publish(root, completer)
    _publish(root, completer)
    _tick(feed)
    frames = _queued(subscription)
    kinds = [frame["type"] for frame in frames]
    assert kinds.count("catalogue") == 1, (
        f"the row finished inside Active (4 -> 1, active True -> True) and the feed "
        f"published {kinds.count('catalogue')} invalidation(s): {kinds} — a comparison "
        f"that asks only whether the row's SECTION moved is blind to this move"
    )
    # AFTER the status frame, so the client paints the checkmark and then re-reads
    # a list that already agrees with it.
    assert kinds.index("catalogue") > kinds.index("session_status"), kinds
    assert _rank(root, completer) == (1, 2)
    revision = frames[kinds.index("catalogue")]["payload"]["revision"]
    assert revision > start, "the client has seen this revision and would not refetch"
    # ...and the list that refetch reads is the one that leads with the completed
    # row: the invalidation is only worth publishing if the backend agrees.
    assert [entry.id for entry in load_catalog(root)] == [completer, elder]
    # The feed's own record of the position IS the position the list just published,
    # so the comparison cannot be satisfied by a map that stopped tracking the sort.
    # Deliberately the LAST assertion: on a tree whose comparison is gated behind the
    # pair, the drift above is what this test reports, not an ``AttributeError`` here.
    assert feed._position_seen[completer][:2] == _rank(root, completer)
    asyncio.run(feed.close())


def test_an_acknowledgement_with_a_record_invalidates_the_catalogue(tmp_path):
    """1 -> 5 with ``active`` unchanged: reading a completion re-files the row.

    A RECORD keeps the row resident (``live_state`` attached), so acknowledging its
    completion moves it out of the completed block (tier 1) into the live band
    (tier 5) while staying in "Active chats" — the same class as the report, and
    the second row of the design's transition table. Without the record the row
    would go Active -> Previous, a SECTION move the old comparison already caught,
    which is exactly why the fixture has one.
    """
    root = tmp_path
    sid, neighbour = "a3" * 6, "a4" * 6
    _listable_session(root, neighbour)
    _listable_session(root, sid)
    _birth(root, neighbour, 1_700_000_000.0)
    _birth(root, sid, 1_700_000_600.0)
    _record_publish(root, sid)
    token = _publish(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _tick(feed)
    assert _queued(subscription) == []
    assert _rank(root, sid) == (1, 2)

    AttentionStore(root / "attention.db").acknowledge(f"session/{sid}", token)
    _tick(feed)
    frames = _queued(subscription)
    kinds = [frame["type"] for frame in frames]
    assert kinds.count("catalogue") == 1, (
        f"acknowledging a resident completion moved the row 1 -> 5 inside Active and the "
        f"feed published {kinds.count('catalogue')} invalidation(s): {kinds} — a "
        f"comparison that asks only whether the row's SECTION moved is blind to this move"
    )
    assert _rank(root, sid) == (5, 2)
    entry = {row.id: row for row in load_catalog(root)}[sid]
    # Only the two MUTABLE terms are compared: the feed builds its row from the
    # registry rather than from the session directory, so it never reads the birth
    # file and its third term is 0.0 where the catalogue's is the real birth. Birth
    # and id cannot move, so every EDGE — which is all the comparison is for — is
    # unaffected by the difference.
    assert entry.active is True
    # Last, and against the list's own key: on a tree whose comparison is gated
    # behind the pair, the drift above is what this test reports, not an
    # ``AttributeError`` here.
    after = feed._position_seen[sid]
    assert after[:2] == (entry.rank[0], entry.rank[1]), "the feed did not record the move"
    asyncio.run(feed.close())


def test_a_resumed_completion_invalidates_the_catalogue(tmp_path):
    """1 -> 4: a turn starting on a row that was sitting in the completed block.

    The operator resumed it, so the runtime publishes a busy record. The row is
    Active before and after (``unseen`` before, ``live_state`` after), the tier
    moves 1 -> 4, and the client needs the list read to move the row back down.
    """
    root = tmp_path
    sid, neighbour = "a5" * 6, "a6" * 6
    _listable_session(root, neighbour)
    _listable_session(root, sid)
    _birth(root, neighbour, 1_700_000_000.0)
    _birth(root, sid, 1_700_000_600.0)
    _publish(root, sid)  # an unread completion with no runtime: tier 1, Active
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _tick(feed)
    assert _queued(subscription) == []
    assert _rank(root, sid) == (1, 2)

    _record_publish(root, sid, busy=True)
    _tick(feed)
    frames = _queued(subscription)
    kinds = [frame["type"] for frame in frames]
    assert kinds.count("catalogue") == 1, (
        f"the row started working again (1 -> 4, active True -> True) and the feed "
        f"published {kinds.count('catalogue')} invalidation(s): {kinds} — a comparison "
        f"that asks only whether the row's SECTION moved is blind to this move"
    )
    assert _rank(root, sid) == (4, 2)
    assert {row.id: row for row in load_catalog(root)}[sid].active is True
    # Last, so a gated comparison reports the drift above rather than an
    # ``AttributeError`` here.
    after = feed._position_seen[sid]
    assert after[:2] == _rank(root, sid), "the feed did not record the move"
    asyncio.run(feed.close())


def test_a_pair_change_that_keeps_the_position_invalidates_nothing(tmp_path):
    """The rule stays NARROW: a gate answered inside the SAME ordering category.

    ``pending: approval -> answer`` changes the derived PAIR (so the frame is
    published — asserted, so this test cannot pass on a dead feed) and leaves the
    row in tier 0, in the same slot, with the same birth and id. Comparing the
    ORDER KEY keeps "the pair changed" from being sufficient on its own: the client
    only re-reads the whole catalogue when the row would land somewhere else.
    """
    root = tmp_path
    sid = "a7" * 6
    _listable_session(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    # The gate arrives over the DOORBELL (the 10 Hz path), so the 1 s probe is
    # gated shut and the pair frames below cannot come from the slow clock.
    feed._status_probed_at = time.monotonic()
    _record_publish(root, sid, pending="approval")
    _tick(feed)
    assert [frame["payload"]["code"] for frame in _statuses(_queued(subscription))] == ["approval"]

    _record_publish(root, sid, pending="answer")
    _tick(feed)
    frames = _queued(subscription)
    assert [frame["payload"]["code"] for frame in _statuses(frames)] == ["answer"], frames
    kinds = [frame["type"] for frame in frames]
    assert "catalogue" not in kinds, (
        f"the pair changed (approval -> answer) inside tier 0, so the row did not move "
        f"and nothing owed a refetch — the feed published {kinds}"
    )
    # The row really did not move, so the only thing that changed is the pair the
    # client is told about: the negative is meaningful against the LIST's own key,
    # which is what the comparison is supposed to track.
    assert _rank(root, sid) == (0, 2)
    asyncio.run(feed.close())


def test_a_section_move_behind_a_byte_identical_pair_still_invalidates(tmp_path):
    """M1: the pair is NOT a superset of the key, so the comparison cannot sit behind it.

    The exact collision the round-1 review reproduced on the real writers, pinned
    here. Two reachable states share ``status_dedupe_key`` byte for byte and differ
    in the ORDER KEY — and the difference is a SECTION move, the subsumed case:

    ================================  =============  ========
    state                             rank (t, band)  active
    ================================  =============  ========
    cold row with an armed wake       (6, 0, …)       False
    that row live and DETACHED, same
    wake still armed                  (5, 2, …)       True
    ================================  =============  ========

    Both derive ``("scheduled", "Scheduled (1 wake)")``: the wake label outranks
    the idle one, and ``wake_rank`` is scoped to COLD rows. A comparison made only
    for pair-CHANGED candidates therefore never runs for this move at all — the row
    climbs out of "Previous chats" into "Active chats" with the gate shut behind
    it and NOTHING published, which is this mechanism's class through a second
    door. The pair is asserted byte-identical in the test itself, so the collision
    cannot quietly stop being one and leave this passing for the wrong reason.

    Reachability is narrow and worth stating rather than hiding: it needs the row
    observed cold-and-armed and then live-idle-DETACHED with no intervening pair
    change (an attach would move the pair). Narrow is not unreachable — a runtime
    that attaches and detaches inside one ``STATUS_PROBE_INTERVAL_S`` does it.
    """
    root = tmp_path
    session_id, neighbour = "ad" * 6, "ae" * 6
    for sid, birth in ((neighbour, 1_700_000_000.0), (session_id, 1_700_000_600.0)):
        _listable_session(root, sid)
        _birth(root, sid, birth)
    _wake(root, session_id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _tick(feed)
    assert _queued(subscription) == []
    before_pair = _pair(root, session_id)
    assert before_pair == ("scheduled", "Scheduled (1 wake)"), before_pair
    assert _rank(root, session_id) == (6, 0)
    assert {row.id: row for row in load_catalog(root)}[session_id].active is False

    # The runtime's own record arrives, detached, and the wake is untouched: the
    # row is live and idle, so the SECTION changes and nothing else the pair sees.
    _record_publish(root, session_id, pid=_FOREIGN_LIVE_PID, detached=True)
    feed._status_probed_at = 0.0
    _tick(feed)
    frames = _queued(subscription)
    kinds = [frame["type"] for frame in frames]
    after_pair = _pair(root, session_id)
    assert after_pair == before_pair, (
        f"this case is only a collision while the pair is byte-identical: "
        f"{before_pair} -> {after_pair}"
    )
    assert _rank(root, session_id) == (5, 2), "the row did not move, so this case proves nothing"
    assert {row.id: row for row in load_catalog(root)}[session_id].active is True
    assert "catalogue" in kinds, (
        f"the row moved from 'Previous chats' into 'Active chats' behind a "
        f"byte-identical pair {before_pair} and the feed published {kinds} — a "
        f"comparison that is skipped whenever the pair is unchanged is blind to "
        f"this move"
    )
    assert (
        "session_status" not in kinds
    ), f"the pair was asserted unchanged above, so no status frame was owed: {kinds}"
    asyncio.run(feed.close())


def test_a_client_connecting_mid_flight_gets_no_invalidation_for_its_first_edge(tmp_path):
    """The PRIME: a fresh connection must not owe one refetch per row.

    ``_prime_status`` seeds the position map from the same derivation the list uses
    for every candidate the connection can see. An ABSENT entry reads as "changed"
    on that row's first edge, so without the prime a client connecting to a machine
    that already had sessions working would be told to re-read the whole catalogue
    once per row — an invalidation for a move that never happened.

    The edge below is the discriminating one: the PAIR moves (busy -> wedged, a
    stale beat) while the POSITION does not (both are tier 4 with the constant
    ``wake_rank``). The status frame is asserted too, so a silent feed cannot make
    this pass.
    """
    root = tmp_path
    working, finished, armed = "a8" * 6, "a9" * 6, "aa" * 6
    for session_id, at in (
        (working, 1_700_000_000.0),
        (finished, 1_700_000_600.0),
        (armed, 1_700_001_200.0),
    ):
        _listable_session(root, session_id)
        _birth(root, session_id, at)
    path = _record_publish(root, working, busy=True)
    _publish(root, finished)
    _wake(root, armed)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _tick(feed)
    assert _queued(subscription) == [], "connecting was treated as an invalidation"

    # The beat goes quiet. Ageing the beat is the test's way of moving the clock;
    # the caches are then refreshed to the file as it is on disk, which is the
    # state the feed is already in when time alone crosses the timeout.
    aged = _record(working, busy=True)
    aged.heartbeat_at = time.time() - HEARTBEAT_TIMEOUT_S - 30.0
    path.write_text(json.dumps(aged.to_json()), encoding="utf-8")
    feed._registry_fingerprint = _fingerprint(feed._registry_dir)
    feed._record_fingerprints[working] = _fingerprint(path)
    feed._status_probed_at = 0.0
    _tick(feed)
    frames = _queued(subscription)
    assert [frame["payload"]["code"] for frame in _statuses(frames)] == ["wedged"], frames
    assert "catalogue" not in [frame["type"] for frame in frames], frames
    assert _rank(root, working) == (4, 2)
    asyncio.run(feed.close())


def test_a_burst_of_completions_inside_active_costs_one_invalidation(tmp_path):
    """Intra-section burst: N reorders in one tick cost ONE refetch, none deferred.

    The extension of the existing burst test to the new cause, because the
    coalescing is the property that keeps this from resurrecting the retired 5 s
    poll: same flag, same once-per-tick guard, so a fleet finishing together still
    buys one whole-catalogue read (~100 ms later) rather than one per row.

    The two rows use two pids on purpose: discovery records are keyed by pid
    (``record_path``), so one pid for both would leave the second row with no
    record at all — and then the burst edge would be its FIRST edge, which is a
    different cause that invalidates for a different reason.
    """
    root = tmp_path
    completer, re_spelled = "ab" * 6, "ac" * 6
    for session_id in (completer, re_spelled):
        _listable_session(root, session_id)
    with _extra_live_pid() as spare:
        _record_publish(root, completer, pid=_FOREIGN_LIVE_PID, busy=True)
        _record_publish(root, re_spelled, pid=spare)
        _publish(root, re_spelled)  # an unread completion on a RESIDENT row: tier 1
        feed = _feed(root)
        feed._take_baseline()
        subscription = feed.subscribe()
        _tick(feed)
        assert _queued(subscription) == []

        # Both move inside Active in the same tick: 4 -> 1 (finishing) and
        # 1 -> 2 (the same completion re-spelled as an error).
        _record_publish(root, completer, pid=_FOREIGN_LIVE_PID)
        _publish(root, completer)
        _publish(root, re_spelled, kind="error")
        _tick(feed)
        frames = _queued(subscription)
        assert len(_statuses(frames)) == 2, frames
        kinds = [frame["type"] for frame in frames]
        assert kinds.count("catalogue") == 1, (
            f"two rows reordered inside Active in one tick and the feed published "
            f"{kinds.count('catalogue')} invalidation(s): {kinds} — a comparison that "
            f"asks only whether a row's SECTION moved is blind to both moves"
        )
        assert _rank(root, completer) == (1, 2)
        assert _rank(root, re_spelled) == (2, 2)
        # Consumed, not deferred: nothing waits for the next tick.
        assert feed._catalogue_invalidated is False
        assert [
            frame
            for frame in _queued(subscription)
            if frame["type"] in ("catalogue", "session_status")
        ] == []
    asyncio.run(feed.close())


def _state_cold(root: Path, sid: str, pid: int) -> None:
    """A directory with nothing behind it: tier 6, "Previous chats"."""


def _state_busy(root: Path, sid: str, pid: int) -> None:
    """Working: a live record with ``busy`` set — tier 4, Active."""
    _record_publish(root, sid, pid=pid, busy=True)


def _state_finished(root: Path, sid: str, pid: int) -> None:
    """The reported transition: the turn ended, the record is quiet, a completion landed."""
    _record_publish(root, sid, pid=pid)
    _publish(root, sid, kind="complete")


def _state_wedged(root: Path, sid: str, pid: int) -> None:
    """A stale beat: tier 4 like ``busy``, a different PAIR (the age sentence)."""
    path = _record_publish(root, sid, pid=pid, busy=True)
    aged = _record(sid, pid=pid, busy=True)
    aged.heartbeat_at = time.time() - HEARTBEAT_TIMEOUT_S - 40.0
    path.write_text(json.dumps(aged.to_json()), encoding="utf-8")


def _state_unread(root: Path, sid: str, pid: int) -> None:
    """An unread completion with no runtime: tier 1, Active."""
    _publish(root, sid, kind="complete")


def _state_unread_error(root: Path, sid: str, pid: int) -> None:
    """The same completion re-spelled as an error: tier 2 — a different slot."""
    _publish(root, sid, kind="error")


def _state_resident_unread(root: Path, sid: str, pid: int) -> None:
    """An unread completion on a RESIDENT row (a record): tier 1, Active."""
    _record_publish(root, sid, pid=pid)
    AttentionStore(root / "attention.db").publish(f"session/{sid}", _DRIFT_TOKEN, "a1", "complete")


def _state_resident_acked(root: Path, sid: str, pid: int) -> None:
    """The same row, read: tier 5 — Active either way, a different slot."""
    _state_resident_unread(root, sid, pid)
    AttentionStore(root / "attention.db").acknowledge(f"session/{sid}", _DRIFT_TOKEN)


def _state_attached(root: Path, sid: str, pid: int) -> None:
    """A live, unoccupied row: tier 5, Active."""
    _record_publish(root, sid, pid=pid)


def _state_idle(root: Path, sid: str, pid: int) -> None:
    """The same live row, detached: tier 5, Active — the SAME position."""
    _record_publish(root, sid, pid=pid, detached=True)


def _state_armed(root: Path, sid: str, pid: int) -> None:
    """A cold row with a wake that will fire: tier 6, wake band 0."""
    _wake(root, sid)


def _state_dormant(root: Path, sid: str, pid: int) -> None:
    """A cold row whose wake is strictly stopped: tier 6, wake band 1."""
    _wake(root, sid, dormant=True)


def _state_detached_armed(root: Path, sid: str, pid: int) -> None:
    """The same armed wake on a LIVE, DETACHED row: tier 5, Active, wake band 2.

    The after-half of the M1 collision. Its pair is the armed wake's —
    ``("scheduled", "Scheduled (1 wake)")``, byte-identical to the cold armed row
    it follows, because the wake label outranks the idle one and ``wake_rank`` is
    scoped to cold rows — while the POSITION moves section, 6 -> 5, band 0 -> 2.
    """
    _record_publish(root, sid, pid=pid, detached=True)
    _wake(root, sid)


#: The drift matrix: ``(name, state before, state after)``. Both halves of every
#: transition are materialised through the REAL writers, on a fresh store, so the
#: comparison is the feed's rule against ``load_catalog``'s own sort.
_DriftState = Callable[[Path, str, int], None]
_DRIFT_CASES: tuple[tuple[str, _DriftState, _DriftState], ...] = (
    ("a working row's heartbeat rewrite", _state_busy, _state_busy),
    ("a working row finishing inside Active", _state_busy, _state_finished),
    ("a working row going quiet inside its tier", _state_busy, _state_wedged),
    ("a completion re-spelled as an error", _state_unread, _state_unread_error),
    ("an acknowledgement of a resident completion", _state_resident_unread, _state_resident_acked),
    ("a resume of an unread completion", _state_unread, _state_busy),
    ("a live row going idle inside its tier", _state_attached, _state_idle),
    ("a cold row starting a turn", _state_cold, _state_busy),
    ("a cold row being armed", _state_cold, _state_armed),
    ("an armed wake going dormant", _state_armed, _state_dormant),
    # The M1 collision: a SECTION move whose pair is byte-identical, so the case
    # only discriminates because the matrix drives every transition through the
    # real writers and asks the comparison for a frame.
    ("a cold armed row attaching as a detached live row", _state_armed, _state_detached_armed),
)


def _mutable_key(root: Path, session_id: str) -> tuple[int, int]:
    """The two terms of the ORDER KEY a transition can move: category + wake band.

    ``-created_at`` and ``id`` are immutable, so they can never produce an edge;
    they are also the two terms the feed cannot derive from the registry alone (a
    row with no session directory reads birth as 0.0). Comparing the mutable terms
    is therefore the complete statement of "did this row move" for both homes.
    """
    entry = {row.id: row for row in load_catalog(root)}[session_id]
    return entry.rank[0], entry.rank[1]


def test_the_key_and_the_sort_cannot_drift(tmp_path):
    """THE GUARD: an invalidation IFF the row's SORT KEY moved.

    ``rank`` IS the key ``rank_entries`` sorts by, so this is the one test that
    fails if a future author widens or narrows the comparison without touching the
    sort — comparing the whole entry (silent cases start publishing), only the
    category (the armed-wake band case stops publishing), or the section again
    (every intra-section case stops publishing).

    WHY THE ROW'S KEY AND NOT THE ID ORDER, which is how the design words it: the
    id order is that key applied to the WHOLE current row set, so a key that moves
    without crossing a neighbour leaves the order it computes unchanged (an armed
    wake going dormant behind a row that is already dormant). The feed holds no
    list — it compares rows one at a time — so the key is the only criterion that
    is a property of the row. Both readings are asserted from the same run: the key
    decides, and the id order before/after is required to have MOVED in at least
    one case and stayed in at least one, so the matrix demonstrably spans both
    sides of what the client can see.
    """
    root = tmp_path
    neighbour, target = "d1" * 6, "d2" * 6
    moved: list[str] = []
    still: list[str] = []
    order_moved: list[str] = []
    with _extra_live_pid() as spare:
        for index, (name, before_state, after_state) in enumerate(_DRIFT_CASES):
            case = root / f"case{index}"
            _listable_session(case, neighbour)
            _listable_session(case, target)
            # The neighbour is a COLD row born LATER than the target, so it starts
            # ABOVE it in "Previous chats" and a target that climbs out of the cold
            # band visibly crosses it. That is what makes the id order below a
            # reading of the client-visible effect rather than a second copy of the
            # key comparison.
            _birth(case, neighbour, 1_700_000_600.0)
            _birth(case, target, 1_700_000_000.0)
            before_state(case, target, spare)
            feed = _feed(case)
            feed._take_baseline()
            subscription = feed.subscribe()
            _tick(feed)
            _queued(subscription)
            key_before = _mutable_key(case, target)
            order_before = [entry.id for entry in load_catalog(case)]

            after_state(case, target, spare)
            # The wake index rides its own probe clock; forcing it is how the wake
            # cases produce an edge at all on a manually driven tick.
            feed._status_probed_at = 0.0
            _tick(feed)
            frames = _queued(subscription)
            published = [frame for frame in frames if frame["type"] == "catalogue"]
            key_after = _mutable_key(case, target)
            order_after = [entry.id for entry in load_catalog(case)]

            assert bool(published) == (key_before != key_after), (
                f"{name}: the sort key moved {key_before} -> {key_after} and the feed "
                f"published {len(published)} invalidation(s) — the comparison and "
                f"rank_entries have drifted apart"
            )
            # The tick really derived this row, so a silent case is not a dead feed.
            # SHAPE-AGNOSTIC, which is the claim above implemented rather than
            # restated: a tree whose comparison stored the boolean SECTION under its
            # own name raised ``AttributeError`` here on the matrix's FIRST (silent)
            # case, i.e. it reported a rename where the iff above exists to report
            # the drift — and a reader checking "does this go red before the fix?"
            # read that as coverage. Reading whichever map the tree has keeps the
            # iff the thing that fails.
            derived = getattr(feed, "_position_seen", None)
            if derived is None:
                derived = getattr(feed, "_activity_seen")
            assert derived.get(target) is not None, f"{name}: never derived"
            (moved if key_before != key_after else still).append(name)
            if order_before != order_after:
                order_moved.append(name)
            asyncio.run(feed.close())
    assert moved, "no case moved the row: this matrix proves nothing"
    assert still, "no case must stay silent: the negative half is missing"
    assert order_moved, "no case changed the list order: the client-visible half is missing"
    assert len(order_moved) < len(
        _DRIFT_CASES
    ), "every case changed the list order: the tie half is missing"


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
    """Review round 2 MINOR 2: the position-change flag commits LAST too.

    ``_publish_status_changes`` states the COMMIT-LAST rule and follows it for
    ``_status_seen`` and the revisions. ``_position_seen`` — the map this rule now
    guards, since the section comparison it was first written for is the subsumed
    case of the ORDER KEY (round 1, M1) — was advancing inside the build loop
    instead, three lines above the rule, so a raising ``_publish`` (a payload that
    will not serialize, a ``_frame`` bug, MemoryError) lost the invalidation for
    good: the next tick re-derived the same pair, found the position already
    recorded, left the flag unset, and the row stayed in the wrong slot until the
    30 s safety poll — the exact symptom this mechanism exists to remove. This case
    is a SECTION move (``recent`` in "Previous chats" -> an unread completion in
    "Active chats"), which is why the assertion below still says section; the rule
    it guards is the same one for an intra-section reorder. Committing the map
    beside ``_status_seen`` gives it the retry the status side already had.

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


# ---------------------------------------------------------------------------
# The test-hosting gates: what keeps a mock session off the operator's screen
# ---------------------------------------------------------------------------

#: A mock session's stored model selection — the row
#: ``Session._persist_selected_model`` journals, and the only durable record of
#: which hosting a conversation actually ran on. Read by
#: ``session_uses_test_hosting``.
_SELECTION_ROW = {
    "id": "selection-1",
    "ts": 1.0,
    "type": "custom",
    "payload": {
        "custom_type": "selected_model",
        "details": {"version": 2, "selector": "test/test-model", "effort": None, "boot": None},
    },
}


def _selection(directory: Path, selector: str) -> None:
    row = json.loads(json.dumps(_SELECTION_ROW))
    row["payload"]["details"]["selector"] = selector
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _bannered(feed: DesktopFeed, subscription: FeedSubscription) -> list[dict[str, Any]]:
    _tick(feed)
    return _notified(_queued(subscription))


def test_a_silenced_process_composes_no_banner(tmp_path, monkeypatch) -> None:
    """The process kill switch reaches the machine-wide channel.

    It always governed ``tui.notify.detached_notify``; the feed pushes a
    ``notification`` frame the desktop app turns into a native banner, so a
    process told not to notify must not offer one either — the switch names the
    process's behaviour, not one of its wires. The control arm in the same cell
    (the same store, published, without the switch) proves the frame was
    otherwise coming.
    """
    root = tmp_path
    sid = "e" * 12
    _session(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    _publish(root, sid)
    assert [frame["session_id"] for frame in _bannered(feed, subscription)] == [sid]

    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    _publish(root, sid)
    assert _bannered(feed, subscription) == []


def test_a_run_that_is_not_the_users_own_offers_no_banner(tmp_path, monkeypatch) -> None:
    """The IDENTITY gate reaches the machine-wide channel as well.

    The kill-switch cell above covers a backend a rig silenced; this one covers
    the backend nobody silenced at all — a rig or a sandbox running under a
    redirected ``HOME``, which is the shape that put a banner on the operator's
    screen every nine seconds on 2026-09-24 through the TUI's own legs. The frame
    here is raised by the ATTACHED APP under its own bundle identity, so the
    offer is the only thing this repository can decline.

    The predicate is answered for this body rather than patched open as the
    module's own opt-in does, which is what makes the cell detect a lost clause:
    the control arm proves the frame was otherwise coming, so this cannot pass by
    the filter, the store or the subscription being broken.
    """
    root = tmp_path
    sid = "f" * 12
    _session(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    _publish(root, sid)
    assert [frame["session_id"] for frame in _bannered(feed, subscription)] == [sid]

    # Recorded as well as answered: the clause has to be ON this path, and a
    # cell that only asserts the absence of a frame would pass if the filter
    # stopped running at all.
    asked: list[int] = []
    monkeypatch.setattr(
        "local_operator.tui.notify.desktop_belongs_to_this_process",
        lambda: (asked.append(1), False)[1],
    )
    _publish(root, sid)
    frames = _bannered(feed, subscription)
    assert asked, f"the offer was never withheld by this gate; frames={frames}"
    assert frames == []


def test_the_stored_hosting_read_stays_off_the_event_loop(tmp_path, monkeypatch) -> None:
    """R1-2: the journal walk is 0.6 ms warm and up to ~745 ms cold, and this
    backend polls on its own loop.

    Asserted on the THREAD the reader ran in rather than on a duration: a
    wall-clock bound on a loaded box is a flake, while "not the loop's thread"
    is the property the fix is about (the neighbouring store reads in this same
    method already hop threads for the same reason). The reader is doubled so
    the assertion is about WHERE it ran, not about what it answered.
    """
    import threading

    root = tmp_path
    sid = "d" * 12
    _session(root, sid)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    threads: list[threading.Thread] = []

    def probe(directory: Path) -> bool:
        threads.append(threading.current_thread())
        return False

    monkeypatch.setattr("local_operator.server.utils.desktop_feed.session_uses_test_hosting", probe)
    _publish(root, sid)
    announced = _bannered(feed, subscription)

    assert threads, "the candidate filter never asked the reader; the cell proved nothing"
    assert all(thread is not threading.main_thread() for thread in threads), threads
    assert [frame["session_id"] for frame in announced] == [sid], announced


def test_a_stored_mock_session_is_never_bannered_by_another_process(tmp_path) -> None:
    """A scratch store outlives the rig that filled it.

    The process switch above cannot reach this case: the backend here never ran
    the mock (a rig did, and exited), it merely POLLS the store the rig left
    behind. Without the per-session read, that store's mock conversations banner
    the operator with the mock's own reply. The control in the same cell is a
    real session in the same store, which must still be announced — the check is
    a filter, not a mute.

    THE MODULE'S OPT-IN IS TURNED BACK OFF HERE, and that is the whole cell: the
    shared fixture waives the test-hosting rule for the suites whose subject is
    the frame, so a test asserting the RULE has to close the waiver it would
    otherwise inherit. The kill switch stays cleared, so the frame is available
    and the rule is the only thing suppressing it.
    """
    # A plain pop, NOT `monkeypatch.delenv`: monkeypatch is set up before this
    # module's autouse opt-in (it is what ``isolate_environment`` requests), so
    # its undo runs AFTER that fixture's teardown and would re-set the escape
    # for every later test in the worker — which is how this cell first made an
    # unrelated suite's rule cell go red. The reader reads fresh, so popping it
    # for the body of this cell is exactly the waiver this test needs.
    os.environ.pop(ENV_ALLOW_TEST_HOSTING_NOTIFY, None)

    root = tmp_path
    mock_sid = "f" * 12
    real_sid = "a" * 12
    _selection(_session(root, mock_sid), "test/test-model")
    _selection(_session(root, real_sid), "openai/gpt-x")
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _publish(root, mock_sid)
    _publish(root, real_sid)

    announced = _bannered(feed, subscription)

    assert [frame["session_id"] for frame in announced] == [real_sid], announced


def test_a_session_with_no_recorded_hosting_is_still_announced(tmp_path) -> None:
    """Failing toward notifying: an unreadable or selection-free journal is not
    evidence of a test session, and muting a real completion would be its own
    bug report.

    THE MODULE'S OPT-IN IS CLOSED FOR THIS BODY, and without that this cell
    proves nothing: the shared fixture waives the test-hosting rule for the whole
    module, so the tolerant branch below the waiver is never reached and the
    assertion holds whatever it does. Its sibling above closes the escape the
    same way, for the same reason. A plain ``os.environ.pop``, NOT
    ``monkeypatch.delenv``: monkeypatch is set up before a module-level autouse
    fixture and so restores after it, re-setting the escape for every later test
    in the worker.
    """
    os.environ.pop(ENV_ALLOW_TEST_HOSTING_NOTIFY, None)

    root = tmp_path
    sid = "b" * 12
    _session(root, sid)  # no transcript at all
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    _publish(root, sid)

    assert [frame["session_id"] for frame in _bannered(feed, subscription)] == [sid]


# ----------------------------------------------------------------------------
# The AUTHORING channel: the frame that keeps the sidebar's Teams/Agents lists
# from waiting for a refresh or a tab switch.
# ----------------------------------------------------------------------------


def _edit_fields(**overrides: Any) -> AgentEditFields:
    """``AgentEditFields`` with every field spelled out, ``None`` but the overrides.

    Both halves are needed. ``AgentEditFields`` is validated in strict mode, so a
    partial construction would silently CLEAR the fields it omits — the defect its
    own docstring documents on the tool path — and pyright reads its model fields as
    required parameters, so a partial one is a type error on top of that. This is
    the same helper the rest of the suite carries (``tests/unit/test_agent_profiles.py``).
    """
    base: dict[str, Any] = dict(
        name=None,
        description=None,
        tags=None,
        categories=None,
        security_prompt=None,
        hosting=None,
        model=None,
        last_message=None,
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        current_working_directory=None,
    )
    base.update(overrides)
    return AgentEditFields(**base)


def _team_fields(**overrides: Any) -> TeamEditFields:
    """``TeamEditFields`` with every field spelled out, for the reason above."""
    base: dict[str, Any] = dict(
        name=None, description=None, manager=None, members=None, instructions=None, project=None
    )
    base.update(overrides)
    return TeamEditFields(**base)


def _profile(root: Path, name: str = "probe-role", description: str = "first") -> str:
    """Create a real profile the way the ``agent`` tool does, and return its id.

    Through ``AgentRegistry`` rather than by writing ``agent.yml`` by hand: the
    thing under test is whether the REGISTRY's own writers move the feed's token,
    and a hand-written fixture would let the writer and the probe drift apart with
    both suites green.
    """
    agent = AgentRegistry(root).create_agent(_edit_fields(name=name, description=description))
    return agent.id


def _agent_yml(root: Path, agent_id: str) -> Path:
    return root / "agents" / agent_id / "agent.yml"


def _authoring(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [frame for frame in frames if frame["type"] == "authoring"]


def test_a_new_profile_publishes_exactly_one_authoring_frame(tmp_path):
    """THE REPORTED DEFECT, at the first of its two surfaces.

    A role authored from inside a session (the ``agent`` tool's ``write_profile``
    reaches ``AgentRegistry.create_agent``, which mkdirs ``agents/<id>/`` and writes
    ``agent.yml``) used to be invisible to the app: nothing in this feed mentioned
    the authoring registries at all, so an open Teams/Agents list kept the rows it
    was mounted with until a refresh or a tab switch. One frame, one refetch.
    """
    root = tmp_path
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    _profile(root)
    feed._authoring_probed_at = 0.0
    _tick(feed)
    frames = _queued(subscription)
    asyncio.run(feed.close())

    published = _authoring(frames)
    assert len(published) == 1, frames
    assert published[0]["payload"] == {"revision": 1}, published[0]
    # The feed is not a session, and a fabricated id would make the client's
    # `observe(sessionId, frame)` look like it had one to attribute this to.
    assert "session_id" not in published[0], published[0]


def test_an_edit_to_what_a_profile_says_publishes_one(tmp_path):
    """The half the name set cannot express: the row was already there."""
    root = tmp_path
    registry = AgentRegistry(root)
    agent = registry.create_agent(_edit_fields(name="probe-role", description="first"))
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    registry.update_agent(agent.id, _edit_fields(description="second"))
    feed._authoring_probed_at = 0.0
    _tick(feed)
    frames = _queued(subscription)
    asyncio.run(feed.close())

    assert len(_authoring(frames)) == 1, frames


def test_an_agent_turn_that_rewrites_agent_yml_publishes_nothing(tmp_path):
    """NEGATIVE PIN — the one a whole-file digest would fail.

    ``update_agent_state`` is the ORDINARY per-turn persistence path: it funnels
    into ``update_agent``, whose ``open("w")`` rewrites ``agent.yml`` in place with
    a new ``mtime_ns`` on every turn. The frame's trigger is a projection over the
    AUTHORED lines, so what a turn moves (``last_message``,
    ``last_message_datetime``, and the ``current_working_directory``
    ``update_agent_state`` threads through) is dropped before the digest. Digesting
    the bytes instead would refetch the sidebar's profiles and teams once per turn
    of every chat — the defect this channel exists to remove, reintroduced by its
    own fix.

    The message is MULTI-LINE and goes through the writer's own ``yaml.dump``: a
    block scalar's continuation lines are the case a naive per-line filter gets
    wrong, and it is the shape real turns carry.
    """
    root = tmp_path
    registry = AgentRegistry(root)
    agent = registry.create_agent(_edit_fields(name="probe-role", description="first"))
    row = _agent_yml(root, agent.id)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    before_stat = row.stat()
    registry.update_agent(
        agent.id,
        _edit_fields(last_message="line one\nline two\n\nline three"),
    )
    after_stat = row.stat()
    # CONTROL: the write really happened, so a quiet feed is the projection's
    # restraint rather than a turn that never touched the file.
    assert (after_stat.st_size, after_stat.st_mtime_ns) != (
        before_stat.st_size,
        before_stat.st_mtime_ns,
    ), "the fixture did not rewrite agent.yml"
    assert "last_message" in row.read_text()

    feed._authoring_probed_at = 0.0
    _tick(feed)
    frames = _queued(subscription)
    assert _authoring(frames) == [], frames

    # ...and the same subscription DOES carry a frame for an authored edit, which
    # is what makes the assertion above evidence rather than decoration.
    registry.update_agent(agent.id, _edit_fields(description="second"))
    feed._authoring_probed_at = 0.0
    _tick(feed)
    after = _queued(subscription)
    asyncio.run(feed.close())
    assert len(_authoring(after)) == 1, after


def test_a_save_agent_state_with_identical_bytes_publishes_nothing(tmp_path):
    """NEGATIVE PIN — the second writer that moves without an authored change.

    ``save_agent_state`` rewrites ``system_prompt.md`` (and four jsonl files) with
    identical bytes on every job and autosave save. Neither file is a term in the
    token — the token is ``agent.yml`` alone — and this cell is what says so out
    loud, because a future "completeness" instinct could add the directory.
    """
    root = tmp_path
    registry = AgentRegistry(root)
    agent = registry.create_agent(_edit_fields(name="probe-role", description="first"))
    registry.set_agent_system_prompt(agent.id, "you are a probe")
    prompt = root / "agents" / agent.id / "system_prompt.md"
    before = prompt.stat()
    state = registry.load_agent_state(agent.id)

    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    registry.save_agent_state(agent.id, state)
    after = prompt.stat()
    assert after.st_mtime_ns != before.st_mtime_ns, "the fixture did not rewrite system_prompt.md"
    assert prompt.read_text() == "you are a probe"

    feed._authoring_probed_at = 0.0
    _tick(feed)
    frames = _queued(subscription)
    asyncio.run(feed.close())

    assert _authoring(frames) == [], frames


def test_a_quiet_window_publishes_no_authoring_frame(tmp_path):
    """The probe RUNS on every tick of this window and still publishes nothing.

    The two other probe clocks are gated shut, and the authoring clock is reset
    before each tick rather than left to its own interval, so what this measures is
    the probe's verdict on an unchanged tree — not the clock's restraint. A feed
    that refetched the sidebar's two lists once a second would be worse than the
    defect it fixed.
    """
    root = tmp_path
    _profile(root)
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    for _ in range(3):
        feed._catalogue_probed_at = time.monotonic()
        feed._status_probed_at = time.monotonic()
        feed._authoring_probed_at = 0.0
        _tick(feed)
    frames = _queued(subscription)
    asyncio.run(feed.close())

    assert frames == [], frames


def test_a_new_team_row_publishes_exactly_one_authoring_frame(tmp_path):
    """The second surface: the team the app (or an agent) just created."""
    root = tmp_path
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    teams = TeamRegistry(root)
    teams.create_team(_team_fields(name="probe-team", members=[TeamMember(role="manager")]))
    feed._authoring_probed_at = 0.0
    _tick(feed)
    frames = _queued(subscription)
    asyncio.run(feed.close())

    assert len(_authoring(frames)) == 1, frames


def test_a_team_save_that_changes_nothing_publishes_nothing(tmp_path):
    """NEGATIVE PIN, and the one the design note is about.

    ``save_team`` publishes EVERY save as a directory swap — ``tempfile.mkdtemp``
    plus two ``os.replace`` calls INSIDE ``teams/`` — so a stat of that directory
    moves on a save that changed nothing at all, and a token carrying it would
    publish once per save. The row's NAME and its PROJECTED CONTENT do not move,
    and those are the terms. This is also why the token's name set skips
    dot-prefixed entries: the swap's staging and backup directories are exactly
    that shape, and counting them would report a change mid-save.
    """
    root = tmp_path
    teams = TeamRegistry(root)
    team = teams.create_team(_team_fields(name="probe-team", members=[TeamMember(role="manager")]))
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    stored = teams.get_team(team.id)
    teams.save_team(stored)
    feed._authoring_probed_at = 0.0
    _tick(feed)
    frames = _queued(subscription)
    assert _authoring(frames) == [], frames

    # CONTROL: an authored team edit DOES publish, so the quiet above is the
    # projection's verdict and not a probe that cannot see teams at all.
    teams.update_team(team.id, _team_fields(description="second"))
    feed._authoring_probed_at = 0.0
    _tick(feed)
    after = _queued(subscription)
    asyncio.run(feed.close())
    assert len(_authoring(after)) == 1, after


def test_a_team_delete_publishes_one(tmp_path):
    root = tmp_path
    teams = TeamRegistry(root)
    team = teams.create_team(_team_fields(name="probe-team", members=[TeamMember(role="manager")]))
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    teams.delete_team(team.id)
    feed._authoring_probed_at = 0.0
    _tick(feed)
    frames = _queued(subscription)
    asyncio.run(feed.close())

    assert len(_authoring(frames)) == 1, frames


@pytest.mark.asyncio
async def test_first_open_waits_for_authoring_baseline_and_keeps_no_replay(tmp_path, monkeypatch):
    """A write during startup belongs to open, not a silent baseline gap.

    Hold the real first baseline at a barrier, author a profile, and let the
    connection try to build ``open``. The old ordering completed ``open`` before
    the baseline adopted the new token, leaving the client with stale lists and
    no later invalidation. The barrier makes that interleaving deterministic.
    """
    root = tmp_path
    feed = _feed(root)
    registry = AgentRegistry(root)
    baseline_entered = threading.Event()
    release_baseline = threading.Event()
    baseline_finished = threading.Event()
    snapshot_before_baseline: list[bool] = []
    original_baseline = feed._take_baseline
    original_snapshot = feed._snapshot

    def held_baseline() -> None:
        baseline_entered.set()
        if not release_baseline.wait(timeout=5):
            raise TimeoutError("test did not release the baseline barrier")
        original_baseline()
        baseline_finished.set()

    def observed_snapshot():
        snapshot_before_baseline.append(not baseline_finished.is_set())
        return original_snapshot()

    monkeypatch.setattr(feed, "_take_baseline", held_baseline)
    monkeypatch.setattr(feed, "_snapshot", observed_snapshot)
    subscription = feed.subscribe()
    frames: list[dict[str, Any]] = []
    opened = asyncio.Event()

    async def pump() -> None:
        async for frame in feed.events(subscription):
            frames.append(frame)
            if frame["type"] == "open":
                opened.set()

    reader = asyncio.create_task(pump())
    try:
        assert await asyncio.to_thread(baseline_entered.wait, 5), "poller never entered baseline"
        # Let the open task run while the real baseline is deliberately held.
        # It must remain blocked rather than build the stale pre-baseline frame.
        await asyncio.sleep(0.05)
        assert not opened.is_set(), "open escaped before the first baseline completed"
        assert snapshot_before_baseline == [], snapshot_before_baseline
        registry.create_agent(
            _edit_fields(name="startup-role", description="authored during startup")
        )
        release_baseline.set()
        await asyncio.wait_for(opened.wait(), timeout=5)
        await asyncio.sleep(feed_module.DOORBELL_INTERVAL_S * 2)
    finally:
        release_baseline.set()
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader
        await feed.close()

    assert snapshot_before_baseline == [False], snapshot_before_baseline
    assert [frame["type"] for frame in frames] == ["open"], frames
    assert feed._authoring_token == feed._authoring_probe()
    assert feed._authoring_invalidated is False


@pytest.mark.asyncio
async def test_cancelling_one_open_waiter_does_not_cancel_shared_baseline(tmp_path, monkeypatch):
    """One disconnect must not cancel the baseline other subscribers await."""
    feed = _feed(tmp_path)
    baseline_entered = threading.Event()
    release_baseline = threading.Event()
    original_baseline = feed._take_baseline

    def held_baseline() -> None:
        baseline_entered.set()
        if not release_baseline.wait(timeout=5):
            raise TimeoutError("test did not release the baseline barrier")
        original_baseline()

    monkeypatch.setattr(feed, "_take_baseline", held_baseline)
    first = feed.subscribe()
    first_events = feed.events(first)

    async def next_open(events):
        return await anext(events)

    first_open = asyncio.create_task(next_open(first_events))
    second_events = None
    second_open = None
    try:
        assert await asyncio.to_thread(baseline_entered.wait, 5), "poller never entered baseline"
        # Let the first connection park on the shared gate before cancelling it.
        await asyncio.sleep(0.05)
        first_open.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first_open
        assert feed._baseline_ready is not None
        assert not feed._baseline_ready.cancelled(), "one subscriber cancelled the shared baseline"

        second = feed.subscribe()
        second_events = feed.events(second)
        second_open = asyncio.create_task(next_open(second_events))
        release_baseline.set()
        opened = await asyncio.wait_for(second_open, timeout=5)
        assert opened["type"] == "open", opened
    finally:
        release_baseline.set()
        if second_open is not None and not second_open.done():
            second_open.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await second_open
        await feed.close()


def test_an_authoring_invalidation_is_not_replayed_to_a_late_subscriber(tmp_path):
    """A reconnecting client is told the COUNTER, never the old frame.

    The frame is an invalidation and not an event: replaying one would make every
    reconnect refetch two lists that have not moved since, and the ``open``
    snapshot already carries the currency the client compares against.
    """
    root = tmp_path
    feed = _feed(root)
    feed._take_baseline()
    early = feed.subscribe()

    _profile(root)
    feed._authoring_probed_at = 0.0
    _tick(feed)
    assert len(_authoring(_queued(early))) == 1

    late = feed.subscribe()
    frames = _collect(feed, late)
    asyncio.run(feed.close())

    assert [frame["type"] for frame in frames] == ["open"], frames
    assert frames[0]["payload"]["authoring_revision"] == 1, frames[0]
    assert "session_id" not in frames[0]["payload"], frames[0]


def test_the_authoring_probe_reads_nothing_when_nothing_moved(tmp_path):
    """THE COST BOUND: O(profiles) in STATS, and no reads at all when nothing moved.

    A row's metadata file is read only when its own stat moved, so an idle probe
    over a populated registry makes stats and no reads. That is the property a
    future "simplification" (dropping the stat memory) would quietly take away, at
    which point every profile is re-read once a second forever.

    Both halves are stated here, because a zero from an instrument that cannot see
    is not a reading: the block after the assertion pushes a DELIBERATE read of the
    same tree through the same counter and requires it to land. The authoring clock
    is left OPEN and the other two are gated shut, so what runs in the measured
    block is this probe.
    """
    root = tmp_path
    registry = AgentRegistry(root)
    for index in range(4):
        registry.create_agent(_edit_fields(name=f"probe-{index}", description="first"))
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    feed._catalogue_probed_at = time.monotonic()
    feed._status_probed_at = time.monotonic()
    # Warm the stat memory, which is what the measured block must NOT have to
    # rewrite: a cold probe reads every row once, by design.
    feed._authoring_probe()

    with _io_watch(feed.agents_dir, reads_under=feed.agents_dir) as counts:
        feed._authoring_probed_at = 0.0
        _tick(feed)
    assert _queued(subscription) == [], "nothing moved; nothing may be published"
    assert counts["record_reads"] == 0, counts
    assert counts["stat"] + counts["lstat"] + counts["scandir"] > 0, counts

    # THE CANARY: the same instrument, the same tree, one deliberate read.
    with _io_watch(feed.agents_dir, reads_under=feed.agents_dir) as canary:
        next(feed.agents_dir.glob("*/agent.yml")).read_text()
    asyncio.run(feed.close())
    assert canary["record_reads"] == 1, canary


def test_the_authoring_probe_creates_nothing(tmp_path):
    """A READER in the strong sense: an absent registry directory stays absent.

    ``AgentRegistry`` and ``TeamRegistry`` both mkdir on construction, so a probe
    that reached for either — the obvious way to ask "are there profiles" — would
    make the feed the process that creates ``agents/`` and ``teams/`` on a machine
    that has never authored anything. This probe takes PATHS only.
    """
    root = tmp_path
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    feed._authoring_probed_at = 0.0
    _tick(feed)
    feed._authoring_probed_at = 0.0
    _tick(feed)
    assert _queued(subscription) == []
    assert not (root / "agents").exists(), "the probe created the agents registry"
    assert not (root / "teams").exists(), "the probe created the teams registry"

    # CANARY: this is the same call on a tree that DOES hold a row, so the absence
    # above is the tree's state and not a probe that returns a constant.
    before = feed._authoring_probe()
    _profile(root)
    asyncio.run(feed.close())
    assert feed._authoring_probe() != before, "the probe did not look at the real tree"


def test_the_authoring_projection_drops_the_turn_keys(tmp_path):
    """The projection, alone: volatile keys ignored, authored keys move the digest."""
    volatile = feed_module._AGENT_VOLATILE_KEYS
    row = tmp_path / "agent.yml"
    authored = (
        "name: probe\n"
        "description: first\n"
        "last_message: ''\n"
        "last_message_datetime: 2026-01-01 00:00:00+00:00\n"
        "current_working_directory: /tmp\n"
        "model: mock\n"
    )
    row.write_text(authored)
    before = feed_module._authoring_digest(row, volatile)
    assert feed_module._authoring_projection(authored, volatile).splitlines() == [
        "name: probe",
        "description: first",
        "model: mock",
    ]

    # A turn: every volatile key moves, including a multi-line message, whose
    # dumped form is a block scalar with indented continuation lines.
    row.write_text(
        authored.replace("last_message: ''", "last_message: 'one\n\n  two\n  three'")
        .replace("2026-01-01 00:00:00+00:00", "2026-01-02 00:00:00+00:00")
        .replace("/tmp", "/elsewhere")
    )
    assert feed_module._authoring_digest(row, volatile) == before

    # An authored edit moves it.
    row.write_text(authored.replace("first", "second"))
    assert feed_module._authoring_digest(row, volatile) != before


def test_the_authoring_frame_gains_no_capability_key():
    """NO KEY, by the rule ``server/routes/capabilities.py`` states.

    A key exists so an EXISTING surface keeps working against a backend that lacks
    the new one. Nothing here is gated: the frame is a latency optimisation on a
    path every client already has (both lists are fetched on mount), an older
    renderer ignores an unknown ``type``, and a newer renderer against an older
    backend keeps today's behaviour — a refresh or a tab switch. What WOULD force
    a key is a client behaviour that DEPENDS on the backend publishing it
    (relaxing a poll, dropping a refetch), and the design does none of those. This
    pin is the cheap half of that argument: it refuses a key named for the frame,
    which is what the comment in that file is there to re-check.
    """
    from local_operator.server.routes.capabilities import capabilities

    result = asyncio.run(capabilities()).result
    assert isinstance(result, dict), result
    features = result["features"]
    assert isinstance(features, dict), features
    assert not [name for name in features if "authoring" in name], features


def test_a_burst_of_authored_rows_costs_one_frame(tmp_path):
    """One refetch per tick, not one per row: a plan that authors four profiles."""
    root = tmp_path
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    registry = AgentRegistry(root)
    for index in range(4):
        registry.create_agent(_edit_fields(name=f"probe-{index}", description="first"))
    feed._authoring_probed_at = 0.0
    _tick(feed)
    frames = _queued(subscription)
    asyncio.run(feed.close())

    assert len(_authoring(frames)) == 1, frames


def test_the_authoring_token_ignores_the_team_writers_staging_directories(tmp_path):
    """The dot-prefix rule, pinned: the swap's own directories are not rows.

    ``save_team`` stages each save as ``.<id>.<rand>`` and moves the live row
    aside as ``.<id>.backup.<rand>`` — both INSIDE ``teams/`` — so a name set that
    counted them would report a change for a save that changed nothing, and could
    report one for a save already published. ``TeamRegistry._load`` skips them for
    the same reason (R5-1), and this pins that the probe agrees with it.
    """
    root = tmp_path
    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()

    teams_dir = root / "teams"
    teams_dir.mkdir()
    feed._authoring_probed_at = 0.0
    _tick(feed)
    before = feed._authoring_token
    assert before is not None

    (teams_dir / ".3d288b16-0c0c-4521-98f9-639364ed5c02.abc123").mkdir()
    (teams_dir / ".3d288b16-0c0c-4521-98f9-639364ed5c02.backup.zzz").mkdir()
    feed._authoring_probed_at = 0.0
    _tick(feed)
    frames = _queued(subscription)
    asyncio.run(feed.close())

    assert feed._authoring_token == before, "the writer's staging directories moved the token"
    assert _authoring(frames) == [], frames


def test_a_workstream_is_announced_and_an_agent_shell_run_is_not(tmp_path):
    """The machine-wide feed's half of the same rule, in both directions.

    The feed composes the machine-wide completion frames and asks
    ``resume.is_user_session`` per record, so the two rows here differ only by
    their marker: an ``agent-shell`` run (a throwaway `lop exec`) is skipped
    exactly as the subagent child above is, while an ``agent-workstream`` — the
    run the operator asked for — is announced like any conversation of theirs.
    Asserted together because a filter that dropped both would pass a test that
    only checked the hidden arm.
    """
    root = tmp_path
    parent, delegated, workstream = "7" * 12, "8" * 12, "9" * 12
    _session(root, parent)
    mark_session_origin(_session(root, delegated), "agent-shell")
    mark_session_origin(
        _session(root, workstream), "agent-workstream", opened_by={"session": parent}
    )

    feed = _feed(root)
    feed._take_baseline()
    subscription = feed.subscribe()
    for session_id in (parent, delegated, workstream):
        _publish(root, session_id)
    _tick(feed)
    frames = _collect(feed, subscription)
    asyncio.run(feed.close())

    scoped = [frame for frame in frames if frame["type"] == "notification"]
    assert scoped, frames
    assert {frame["session_id"] for frame in scoped} == {parent, workstream}
