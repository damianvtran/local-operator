"""The machine-wide desktop event feed: one authenticated stream per BACKEND.

WHY THIS EXISTS. Every notification channel before this one was PER SESSION. A
``notification`` frame rides the SSE stream of a session's bridge, and a bridge
exists only while a route holds one — so a completion in session B, while the
desktop app is displaying session A, produced no composed frame, no banner and
nothing at all. The only remaining announcer was a running TUI's 1 s tick, and
with no TUI running a finished turn was announced by nobody.

This module is the missing channel: ONE stream for the whole process, owned by
``app.state.desktop_feed``, that composes and publishes ``notification`` frames
for sessions that have no bridge.

WHAT MAKES IT SAFE, and the property most likely to be broken by a later edit:

* **It acquires no bridge and spawns no runtime.** Its two dependencies —
  ``AttentionStore.state_many``/``revision`` and ``compose()`` over
  ``sessions/<id>/`` — are both bridge-independent. The tempting "reuse the
  bridge's composer" refactor would make watching a 200-row catalogue build 200
  cold facades and 200 SQLite poll loops, and would take the ``BRIDGE_COUNT``
  ceiling with it. ``tests/unit/server/test_desktop_feed.py`` asserts the
  absence directly (``DesktopSessions.bridges`` stays empty across a feed
  cycle).
* **It mints no second semantics.** The payload is built by the SAME function
  the bridge uses (``notifications.compose.notification_payload``), so
  ``dedupe_key`` is byte-identical and the desktop's local claim map collapses
  the pair into one banner. The one field the feed derives differently is
  ``focus_policy`` — a ROUTING field, not content; see ``_focus_policy_for``.
* **It never replays.** ``notification`` and ``attention`` frames are live-only
  and the connection's first read is a BASELINE: it records the store's current
  revision and announces nothing that predates the connection. A reconnect
  therefore does not flood, which is the same rule the bridge's ``if previous:``
  guard and the store's no-flood bootstrap already apply.

COST. One poller per process, started with the first subscriber and stopped with
the last. Each tick is TWO ``os.stat`` calls on the store and its journal — the
doorbell, borrowed from ``config_watch.py``'s treatment of ``config.yml`` — with
SQL only when the store actually moved. That is what makes detection p50 ~60 ms
where the per-session poll's floor was 1 s.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, cast

from local_operator.notifications import notification_payload
from local_operator.notifications.compose import NotificationKind
from local_operator.server.utils.desktop_presence import DesktopDeliveryPublisher
from local_operator.server.utils.desktop_sessions import (
    BRIDGE_NOTIFIABLE_KINDS,
    REPLAY_BYTES,
    REPLAY_COUNT,
    SUBSCRIBER_COUNT,
    WATCH_TTL,
)
from local_operator.session.attention import AttentionStore
from local_operator.session.runtime.presence import desktop_presence

logger = logging.getLogger(__name__)

#: THE DOORBELL. Two ``os.stat`` calls per tick and no SQL, so the cost of
#: looking is separated from the cost of finding: the store is only opened when
#: its own ``(st_ino, st_size, st_mtime_ns)`` or its journal's moved. 100 ms is
#: the composure of a 1 s tick with the detection floor of a 10 Hz one; the
#: measured alternative — a 250 ms doorbell — costs 60 ms of p50 latency for a
#: quarter of the stat load, which the profile in ``TUI_BACKGROUND_RESPONSIVENESS``
#: gives no reason to want.
DOORBELL_INTERVAL_S = 0.10

#: Silence after which the stream emits a ``heartbeat`` frame. The client's
#: watchdog is ``heartbeat_seconds`` x 3, so a half-open socket after a
#: sleep/wake is detected by the client rather than reading as a live, quiet
#: stream — which is what the session stream's 15 s heartbeat is for too.
HEARTBEAT_INTERVAL_S = 15.0

#: How often the CATALOGUE revision is recomputed, deliberately slower than the
#: doorbell.
#:
#: The revision is a cheap invalidation token for the sidebar's row set, and it
#: is not free: it is a ``readdir`` of the sessions directory plus one stat, so
#: running it at 10 Hz would spend the doorbell's whole budget on a signal whose
#: consumer shows a page of rows. One second is 5x better than the 5 s
#: ``sessions.list`` poll this replaces and keeps the feed's own I/O profile
#: where the design bounds it (two stats per tick).
CATALOGUE_PROBE_INTERVAL_S = 1.0

#: THE BURST CEILING — at most this many individual banners per doorbell tick.
#:
#: Several user sessions can finish within a second (a fleet of subagent-owning
#: sessions, a machine that was asleep and woke), and an uncapped feed turns
#: that into a stack of banners: the worst possible notification-centre
#: experience and the one thing a user will disable the feature over. The
#: remainder is still claimed-complete — the frames below speak for it — and is
#: reported as ONE digest frame naming the count, so nothing is silently
#: dropped.
#:
#: Kept EQUAL to the TUI's ``_BACKGROUND_NOTIFY_MAX_PER_TICK`` by a test rather
#: than by convention: the two are the same promise on two transports, and two
#: hand-maintained 3s are one edit away from disagreeing about it.
BURST_LIMIT = 3


def _fingerprint(path: Path) -> tuple[int, int, int] | None:
    """``(st_ino, st_size, st_mtime_ns)``, or ``None`` when it is not there.

    The comparison this feeds is borrowed wholesale from ``config_watch``, which
    does exactly this for ``config.yml``: three cheap fields that together move
    on any write worth reacting to, and no SQLite connection to open when they
    have not.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


@dataclass(eq=False)
class FeedSubscription:
    """One live SSE client of the feed.

    ``baseline_sequence`` is this connection's own floor: a frame that describes
    a completion published BEFORE this client connected is not news to it, and
    the desktop's rule (Q5) is that a completion missed while the app was away
    is recovered by the durable unseen mark rather than by a late banner. The
    filter is per subscription rather than per poller because a second window
    that connects later must not inherit the first one's longer history.
    """

    id: str
    baseline_sequence: int
    queue: asyncio.Queue[dict[str, Any] | None] = field(default_factory=asyncio.Queue)
    #: Serialized size of each queued entry, in step with ``queue`` because the
    #: two are only ever appended to and popped from together. Kept here rather
    #: than recomputed on read so the backlog bound costs one ``json.dumps`` per
    #: frame per subscriber instead of two.
    queued_sizes: deque[int] = field(default_factory=deque)
    queued_bytes: int = 0
    overflow: bool = False


class DesktopFeed:
    """The process singleton behind ``GET /v1/desktop/events``.

    One instance per HTTP server, created lazily by the route beside the
    ``DesktopSessions`` pool. Not a bridge user: nothing here acquires a
    session, and the only ``DesktopSessions`` contact is the read-only
    ``bridged`` callback, which asks which sessions already have a stream and
    therefore must not be duplicated.
    """

    def __init__(
        self,
        root: Path,
        *,
        bridged: Callable[[], Any] | None = None,
        presence: DesktopDeliveryPublisher | None = None,
    ) -> None:
        self.root = root
        self.sessions_dir = root / "sessions"
        #: A fresh epoch per PROCESS start. Frames carry it so a client can tell
        #: "the backend restarted" from "the stream stuttered", exactly as the
        #: session stream's epoch does.
        self.epoch = secrets.token_hex(8)
        self.sequence = 0
        self.subscribers: dict[str, FeedSubscription] = {}
        self.store = AttentionStore(root / "attention.db")
        #: The delivery lease this process publishes for its own subscribers.
        #: Owned here rather than by the route so its lifetime is the feed's: a
        #: claim is only ever believed while the socket that made it is alive.
        self.presence = presence or DesktopDeliveryPublisher(root)
        self._bridged = bridged or (lambda: frozenset())
        self._task: asyncio.Task[None] | None = None
        self._revision: tuple[int, int, int] | None = None
        self._published_sequence = 0
        self._acknowledgements: dict[str, int] = {}
        self._fingerprint: tuple[int, int, int] | None = None
        self._catalogue_revision = 0
        self._catalogue_names: tuple[str, ...] = ()
        self._catalogue_probed_at = 0.0

    # -- subscribers -------------------------------------------------------

    def subscribe(self) -> FeedSubscription:
        """Register a subscriber and start the poller if it is not running.

        Raising when the table is full is deliberate and matches the session
        stream: a client that cannot be served must be told, not quietly given a
        stream that will never carry anything.
        """
        if len(self.subscribers) >= SUBSCRIBER_COUNT:
            raise RuntimeError("too many desktop feed subscribers")
        subscription = FeedSubscription(id=secrets.token_hex(8), baseline_sequence=self.sequence)
        self.subscribers[subscription.id] = subscription
        self._ensure_poller()
        return subscription

    def unsubscribe(self, subscription: FeedSubscription) -> None:
        """Drop a subscriber, its presence claim, and the poller if it was last."""
        self.subscribers.pop(subscription.id, None)
        self.presence.drop(subscription.id)

    async def events(self, subscription: FeedSubscription) -> AsyncIterator[dict[str, Any]]:
        """Yield ``open`` then every subsequent frame, with heartbeats.

        ``open`` is produced HERE rather than fanned out, because its payload is
        this connection's snapshot (attention state and the catalogue revision)
        and there is nothing to share: two clients that connect a second apart
        must take two snapshots, or the second one's baseline is a lie.
        """
        try:
            yield await self._open_frame(subscription)
            while True:
                try:
                    frame = await asyncio.wait_for(
                        subscription.queue.get(), timeout=HEARTBEAT_INTERVAL_S
                    )
                except TimeoutError:
                    yield self._frame("heartbeat", {"ts": time.time()})
                    continue
                if frame is None:
                    # The backlog bound tripped. Say so and close: the client
                    # reconnects and takes a fresh snapshot, which is the only
                    # honest recovery from a gap it will never see the middle of.
                    yield self._frame(
                        "gap", {"reason": "overflow", "subscription_id": subscription.id}
                    )
                    return
                if subscription.queued_sizes:
                    subscription.queued_bytes -= subscription.queued_sizes.popleft()
                yield frame
        finally:
            self.unsubscribe(subscription)

    async def close(self) -> None:
        """Stop the poller and withdraw the lease. Idempotent."""
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 — teardown
                pass
        self.subscribers.clear()
        self.presence.close()

    # -- frame construction ------------------------------------------------

    def _frame(
        self, frame_type: str, payload: dict[str, Any], *, session_id: str | None = None
    ) -> dict[str, Any]:
        """Advance the receipt cursor and stamp one frame.

        The envelope's SHAPE is the session stream's (``session_id``, ``epoch``,
        ``seq``, ``type``, ``payload``) so the desktop's existing relay needs no
        new parser. ``session_id`` rides only the types that concern one
        session: the feed is not a session, and a fabricated id on ``open``
        would make the client's ``observe(sessionId, frame)`` API look like it
        had a session to attribute a catalogue event to.
        """
        self.sequence += 1
        frame: dict[str, Any] = {"epoch": self.epoch, "seq": self.sequence, "type": frame_type}
        if session_id is not None:
            frame["session_id"] = session_id
        frame["payload"] = payload
        return frame

    def _publish(
        self,
        frame_type: str,
        payload: dict[str, Any],
        *,
        session_id: str | None = None,
        since_sequence: int | None = None,
    ) -> None:
        """Fan a frame out to every subscriber that should see it.

        ``since_sequence`` is the per-subscriber baseline filter, applied only
        to frames that describe a completion: ``attention`` is a level (a stale
        one is corrected by the next, and the merge is revision-guarded) while
        ``notification`` is an edge whose whole value is timeliness.
        """
        frame = self._frame(frame_type, payload, session_id=session_id)
        size = len(json.dumps(frame))
        for subscription in list(self.subscribers.values()):
            if since_sequence is not None and subscription.baseline_sequence >= since_sequence:
                continue
            if subscription.overflow:
                continue
            if (
                subscription.queued_bytes + size > REPLAY_BYTES
                or subscription.queue.qsize() >= REPLAY_COUNT
            ):
                subscription.overflow = True
                subscription.queue.put_nowait(None)
                subscription.queued_sizes.append(0)
                continue
            subscription.queued_bytes += size
            subscription.queued_sizes.append(size)
            subscription.queue.put_nowait(frame)

    # -- the poller --------------------------------------------------------

    def _ensure_poller(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._task = loop.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        """Tick while anybody is listening; stop when nobody is.

        Registered with the first subscriber and torn down with the last
        (``_expire_watches``' shape): an idle backend must not hold a 10 Hz
        timer for a stream nobody is reading.
        """
        try:
            await asyncio.to_thread(self._take_baseline)
            while True:
                await asyncio.sleep(DOORBELL_INTERVAL_S)
                if not self.subscribers:
                    return
                try:
                    await self._tick()
                except Exception:  # noqa: BLE001 — one bad tick is not the feature
                    logger.debug("desktop feed tick failed", exc_info=True)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.debug("desktop feed poller stopped", exc_info=True)

    def _take_baseline(self) -> None:
        """Record the store's state as of the connection, and announce none of it.

        A completion published before the first subscriber arrived is HISTORY.
        Announcing it on connect is the flood the desktop's own Q5 decision
        refuses: what recovers a completion missed while the app was away is the
        durable ``unseen`` mark in the snapshot, not a banner about last night.
        """
        revision = self.store.revision()
        self._revision = revision
        self._published_sequence = revision[0]
        self._acknowledgements = self.store.acknowledgement_map()

    async def _tick(self) -> None:
        # 1. THE DOORBELL. Two stats, no SQL, no connection.
        fingerprint = await asyncio.to_thread(_fingerprint, self.store.path)
        if fingerprint is not None:
            if fingerprint == self._fingerprint:
                return
            self._fingerprint = fingerprint
        # 2. THE REVISION GATE IS THE AUTHORITY, and the delta below is only an
        # optimisation. A heal moves neither `MAX(sequence)` nor
        # `SUM(acknowledged)` — it moves the `supersedes` counter — so a tick
        # that trusted the delta alone would miss it entirely.
        revision = await asyncio.to_thread(self.store.revision)
        if revision != self._revision:
            self._revision = revision
            await self._emit_delta()
        await self._maybe_emit_catalogue()

    async def _emit_delta(self) -> None:
        """Publish one ``attention`` frame per changed session, then banners."""
        published, acknowledgements = await asyncio.gather(
            asyncio.to_thread(self.store.published_since, self._published_sequence),
            asyncio.to_thread(self.store.acknowledgement_map),
        )
        changed: list[str] = []
        fresh: list[dict[str, Any]] = []
        for row in published:
            self._published_sequence = max(self._published_sequence, int(row["sequence"]))
            changed.append(str(row["conversation"]))
            if row["kind"] in BRIDGE_NOTIFIABLE_KINDS:
                fresh.append(row)
        for conversation, acknowledged in acknowledgements.items():
            if self._acknowledgements.get(conversation) != acknowledged:
                changed.append(conversation)
        self._acknowledgements = acknowledgements

        identities = [item for item in dict.fromkeys(changed) if self._is_user_session(item)]
        states: dict[str, dict[str, Any]] = {}
        if identities:
            states = await asyncio.to_thread(self.store.state_many, identities)
            for identity in identities:
                state = states.get(identity)
                if state is not None:
                    # No `supported` key here, deliberately: only a live runtime
                    # can answer it and the feed has none. The renderer's merge
                    # preserves the value it already holds rather than clearing
                    # it, which is what keeps the read receipt working for the
                    # session on screen.
                    self._publish("attention", state, session_id=self._session_id(identity))
        if fresh:
            await self._emit_notifications(fresh, states)

    async def _emit_notifications(
        self, fresh: list[dict[str, Any]], states: dict[str, dict[str, Any]]
    ) -> None:
        """Compose a banner for each newly published, unseen, unbridged session."""
        bridged = set(self._bridged())
        candidates: list[tuple[str, str, str, str, int]] = []
        for row in fresh:
            identity = str(row["conversation"])
            if identity in bridged:
                # THE STEADY-STATE GUARD against two banners for one completion.
                # A session with a live bridge has its own composer and its own
                # `focus_policy`; the feed yields to it rather than racing it.
                continue
            state = states.get(identity)
            if state is None or not state.get("unseen"):
                continue
            policy = self._focus_policy_for(self._session_id(identity))
            if policy is None:
                continue
            candidates.append(
                (identity, str(row["token"]), str(row["kind"]), policy, int(row["sequence"]))
            )
        if not candidates:
            return
        candidates.sort(key=lambda item: item[4])
        for identity, token, kind, policy, sequence in candidates[:BURST_LIMIT]:
            payload = await asyncio.to_thread(
                notification_payload,
                cast(NotificationKind, kind),
                session_dir=self.sessions_dir / self._session_id(identity),
                token=token,
                session_id=self._session_id(identity),
                focus_policy=policy,
            )
            self._publish(
                "notification",
                payload,
                session_id=self._session_id(identity),
                since_sequence=sequence,
            )
        overflow = candidates[BURST_LIMIT:]
        if overflow:
            self._publish(
                "notification",
                self._digest_payload(overflow),
                session_id=self._session_id(overflow[-1][0]),
                since_sequence=overflow[-1][4],
            )

    def _digest_payload(self, overflow: list[tuple[str, str, str, str, int]]) -> dict[str, Any]:
        """One banner standing in for the completions the ceiling held back.

        Composed from the TUI's own digest vocabulary
        (``background_digest_title`` / ``BODY_BACKGROUND_DIGEST`` /
        ``digest_subtitle``) rather than minted here: the two transports make
        the same promise, and a second wording for "several sessions finished"
        is a second thing to keep in step. The count is the whole remainder —
        an absolute number, not one relative to the ceiling.
        """
        from local_operator.tui.notify import (
            BODY_BACKGROUND_DIGEST,
            background_digest_title,
            digest_subtitle,
        )

        kinds = [kind for _identity, _token, kind, _policy, _sequence in overflow]
        # The ids ride along so the client can land the click on the catalogue
        # rather than on an arbitrary member of the set — which is the one
        # decision a digest banner cannot make for the user.
        return {
            "contract": 1,
            "kind": "complete",
            "title": background_digest_title(len(overflow)),
            "status": digest_subtitle(kinds),
            "body": BODY_BACKGROUND_DIGEST,
            "body_is_snippet": False,
            "body_is_failure": False,
            "title_is_session_name": False,
            # No single completion owns this frame, so its dedupe key is keyed
            # on the SET. It must not collide with any member's own key: a
            # digest is not a duplicate of a per-session banner, it is the
            # announcement that several happened.
            "dedupe_key": "burst:" + ",".join(sorted(token for _i, token, _k, _p, _s in overflow)),
            "completion_token": None,
            "session_name": None,
            "focus_policy": "always",
            "burst_count": len(overflow),
            "session_ids": [self._session_id(identity) for identity, *_rest in overflow],
        }

    def _focus_policy_for(self, session_id: str) -> str | None:
        """``focus_policy`` for this completion, or ``None`` for "raise nothing".

        WHY THIS IS DERIVED AND NOT COPIED. ``focus_policy`` is a ROUTING field,
        not content. The per-session frame hard-codes ``when_unfocused``, and
        the desktop suppresses exactly that value while any window is focused —
        so shipping the bridge's payload verbatim meant the commonest state of
        all (user in the app on session A while B finishes) announced nothing,
        on any surface, at all: rung 2 had already silenced the runtime and the
        TUI. That is the operator's original symptom, made permanent by the
        presence mechanism that was supposed to fix it.

        So: a completion for a session the app is NOT displaying is ``always`` —
        the window's focus says nothing about whether the user wants to hear
        that a DIFFERENT conversation finished. A completion for the session the
        app IS attendedly displaying is rung 1: the card is in band on its own
        stream, no banner is raised, and ``None`` says so.
        """
        presence = desktop_presence(self.root)
        if presence.attended and presence.session_id == session_id:
            return None
        return "always"

    # -- catalogue ---------------------------------------------------------

    async def _maybe_emit_catalogue(self) -> None:
        now = time.monotonic()
        if now - self._catalogue_probed_at < CATALOGUE_PROBE_INTERVAL_S:
            return
        self._catalogue_probed_at = now
        revision, names = await asyncio.to_thread(self._catalogue_probe)
        if revision == self._catalogue_revision:
            return
        self._catalogue_revision = revision
        self._catalogue_names = names
        self._publish("catalogue", {"revision": revision})

    def _catalogue_probe(self) -> tuple[int, tuple[str, ...]]:
        """A cheap invalidation token for the sidebar's ROW SET.

        The sessions directory's own ``(inode, mtime_ns)`` moves when a session
        is created or removed, and the NAME SET catches a create/delete that a
        same-nanosecond mtime would hide. Both come from one ``readdir``, with
        no per-directory stat: walking the store to notice that a transcript
        grew would be the per-row scan the 5 s ``sessions.list`` poll was
        retired for, and it would grow with the store exactly as that one did.

        What this therefore does NOT do, stated because it is a real limit: an
        in-place append to an existing transcript (a preview or a title that
        changed under a stable row set) does not move either term. The sidebar's
        30 s safety poll and its refetch on window focus are what cover that,
        and the row content a waiting user actually needs — the unseen mark —
        rides its own ``attention`` frame, which is not gated on this at all.
        """
        names: list[str] = []
        try:
            with os.scandir(self.sessions_dir) as entries:
                for entry in entries:
                    names.append(entry.name)
        except OSError:
            names = []
        names.sort()
        # A STABLE digest, not `hash()`: the token is opaque to the client but it
        # is compared across reconnects, and Python's string hashing is salted
        # per process — so a `hash()` here would report a change to every client
        # that reconnects to a restarted backend, for no reason at all.
        key = ",".join(names) + "|" + repr(_fingerprint(self.sessions_dir))
        return zlib.crc32(key.encode()) & 0x7FFFFFFF, tuple(names)

    # -- identity helpers --------------------------------------------------

    def _is_user_session(self, identity: str) -> bool:
        """Whether this store key is a conversation a person started.

        A subagent child is a machine's delegated run, not a conversation to
        banner about, and children live as siblings under ``sessions/`` — the
        one filter keeps them off the feed entirely. The ``session/`` prefix is
        checked too because an agent transcript's key is ``agent/<name>``, which
        is NOT unique across parents and must never be mistaken for a session
        id.

        Accepts either a store key (``session/<id>``) or a bare directory name,
        because both callers have a different one in hand.
        """
        from local_operator.resume import is_user_session

        session_id = self._session_id(identity)
        if identity != session_id and not identity.startswith("session/"):
            return False
        try:
            return is_user_session(self.sessions_dir / session_id)
        except Exception:  # noqa: BLE001 — an unreadable marker is not a session
            return False

    @staticmethod
    def _session_id(identity: str) -> str:
        """The 12-hex session id a store key names."""
        return identity.split("/", 1)[1] if "/" in identity else identity

    # -- the open frame ----------------------------------------------------

    async def _open_frame(self, subscription: FeedSubscription) -> dict[str, Any]:
        """The connection's snapshot: attention state and a catalogue revision."""
        attention, catalogue_revision = await asyncio.to_thread(self._snapshot)
        return self._frame(
            "open",
            {
                "subscription_id": subscription.id,
                "heartbeat_seconds": HEARTBEAT_INTERVAL_S,
                "lease_seconds": WATCH_TTL,
                "watch_ttl_seconds": WATCH_TTL,
                "catalogue_revision": catalogue_revision,
                "attention": attention,
            },
        )

    def _snapshot(self) -> tuple[dict[str, dict[str, Any]], int]:
        """Every user session's attention state, plus the catalogue revision.

        Deliberately NOT the catalogue's rows: those carry a preview read per
        row, and putting that on the feed would move the sidebar's cost rather
        than remove it. The client already has its rows; what it cannot know
        without this is which of them are unread.

        Runs once per connection, so the per-directory marker read behind
        ``is_user_session`` is affordable here in a way it is not on the
        doorbell — this is one ``sessions.list``-shaped scan minus the previews,
        paid by a client that is opening a stream rather than by a 10 Hz timer.
        """
        revision, names = self._catalogue_probe()
        self._catalogue_revision = revision
        self._catalogue_names = names
        self._catalogue_probed_at = time.monotonic()
        identities = [f"session/{name}" for name in names if self._is_user_session(name)]
        states = self.store.state_many(identities) if identities else {}
        return states, revision
