"""In-process pub/sub broker with sequenced replay, feeding the SSE surface.

WHY THIS EXISTS
---------------
The agent turn runs in a child ``multiprocessing.Process``; the HTTP handlers
run in the parent. Engine events reach the parent only through the per-job
``multiprocessing.Queue`` drained by ``monitor_status_queue``
(:mod:`local_operator.server.utils.job_processor_queue`). An SSE handler
therefore *cannot* call ``Session.subscribe`` — by the time a request is being
served, the session object lives in another process. This broker is the
parent-side seam the pump publishes into and that any number of HTTP
responses can subscribe to.

WHY IT IS SEQUENCED
-------------------
SSE gives us one capability WebSocket never had here: the browser resends
``Last-Event-ID`` automatically on reconnect. That is only useful if the server
assigns a monotonic id per channel and can replay from it, so every published
event gets a sequence number and lands in a bounded per-channel ring buffer.
A client that drops mid-turn resumes exactly where it left off instead of
losing frames or restarting the render.

WHY EVERYTHING IS BOUNDED
-------------------------
This process is long-lived and serves an unbounded number of turns. Every
structure here has a ceiling: the replay buffer per channel, the queue per
subscriber, and the number of retained channels. Overflow is *reported* (via
``dropped``) rather than silently absorbed, because a consumer that knows it
missed frames can reconcile over REST, while one that does not will render a
hole forever. Retention is also time-bounded so a channel nobody ever reads
cannot pin memory.

The broker is deliberately transport-neutral: it knows nothing about SSE,
WebSockets, or FastAPI. The legacy WebSocket fan-out remains a separate,
untouched publish path so the fallback transport stays byte-identical.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Set

logger = logging.getLogger("local_operator.server.utils.event_broker")

# A channel keeps this many events for `Last-Event-ID` replay. Frames for a
# message are cumulative snapshots, so the buffer exists to bridge a
# reconnect, not to store a transcript - the durable record is the agent's
# execution history, reachable over REST.
DEFAULT_REPLAY_BUFFER = 256

# Per-subscriber backlog. A subscriber that cannot keep up with this much
# outstanding work is not going to catch up by growing the queue; it gets a
# gap notice instead.
DEFAULT_SUBSCRIBER_QUEUE = 512


#: Channel key prefixes. The two namespaces are kept distinct so a job id can
#: never collide with a record id (both are opaque strings). Defined here, not
#: in the routes module, so non-HTTP producers (the scheduler) can build channel
#: keys without importing the FastAPI layer (which would cycle).
MESSAGE_CHANNEL = "message"
JOB_CHANNEL = "job"


def message_channel(message_id: str) -> str:
    """The record-keyed channel, parity with the legacy WebSocket key."""
    return f"{MESSAGE_CHANNEL}:{message_id}"


def job_channel(job_id: str) -> str:
    """The job-keyed channel, openable before any record id exists."""
    return f"{JOB_CHANNEL}:{job_id}"


# A channel with no subscribers and no traffic is evicted after this long.
# Terminal channels are kept for the same window so a client that reconnects
# just after completion still observes the terminal frame.
DEFAULT_CHANNEL_TTL_S = 300.0

# Hard ceiling on retained channels, newest kept. Protects against a burst of
# short-lived channels (e.g. one per tool call) outrunning TTL eviction.
DEFAULT_MAX_CHANNELS = 512


@dataclass(frozen=True)
class BrokerEvent:
    """One published event, stamped with its per-channel sequence.

    ``seq`` is what becomes the SSE ``id:`` field and what a client echoes back
    in ``Last-Event-ID``. ``name`` becomes the SSE ``event:`` name. ``data`` is
    the JSON-serialisable body and is treated as immutable by every consumer.
    """

    channel: str
    seq: int
    name: str
    data: Dict[str, Any]
    created_at: float

    def with_channel(self, channel: str, seq: int) -> "BrokerEvent":
        """Re-stamp this event for a second channel.

        The same engine event is published to both a record-keyed and a
        job-keyed channel, and each channel owns its own sequence space, so the
        copy has to carry the other channel's id.
        """
        return BrokerEvent(
            channel=channel, seq=seq, name=self.name, data=self.data, created_at=self.created_at
        )


# ``eq=False`` keeps the default identity hash: subscribers live in a set and
# are distinguished by *who* they are, not by their contents. A value-equal
# dataclass would both be unhashable and let two distinct listeners collapse
# into one entry the moment their counters happened to match.
@dataclass(eq=False)
class _Subscriber:
    """One live consumer of a channel.

    ``dropped`` counts events discarded because the queue was full. The reader
    surfaces it as a gap notice so the UI can reconcile over REST rather than
    render a hole.
    """

    queue: "asyncio.Queue[Optional[BrokerEvent]]"
    dropped: int = 0
    closed: bool = False


@dataclass
class _Channel:
    """Replay buffer plus subscriber set for one stream key."""

    name: str
    buffer: Deque[BrokerEvent] = field(default_factory=deque)
    subscribers: Set[_Subscriber] = field(default_factory=set)
    next_seq: int = 1
    last_activity: float = field(default_factory=time.monotonic)
    #: Set once a terminal event is published. Retained (not evicted at once)
    #: so a late reconnect can still see how the turn ended.
    terminal: bool = False


class EventBroker:
    """Fan-out of sequenced events to any number of in-process listeners.

    Publishing never blocks on a slow reader and never raises into the pump:
    the pump drains a cross-process queue and must not stall or die because an
    HTTP client went away mid-turn.
    """

    def __init__(
        self,
        *,
        replay_buffer: int = DEFAULT_REPLAY_BUFFER,
        subscriber_queue: int = DEFAULT_SUBSCRIBER_QUEUE,
        channel_ttl_s: float = DEFAULT_CHANNEL_TTL_S,
        max_channels: int = DEFAULT_MAX_CHANNELS,
    ) -> None:
        self._channels: Dict[str, _Channel] = {}
        self._replay_buffer = max(1, replay_buffer)
        self._subscriber_queue = max(1, subscriber_queue)
        self._channel_ttl_s = channel_ttl_s
        self._max_channels = max(1, max_channels)

    # ------------------------------------------------------------------
    # publishing
    # ------------------------------------------------------------------

    def publish(
        self,
        channel: str,
        name: str,
        data: Dict[str, Any],
        *,
        terminal: bool = False,
    ) -> BrokerEvent:
        """Append an event to ``channel`` and hand it to every subscriber.

        Synchronous on purpose. The pump calls this from its drain loop; making
        it a coroutine would add a scheduling hop per frame and invite the
        caller to await a slow consumer. Delivery is a bounded ``put_nowait``,
        so the cost is the fan-out itself.
        """
        return self.publish_with(channel, name, lambda _seq: data, terminal=terminal)

    def publish_with(
        self,
        channel: str,
        name: str,
        body_factory: Callable[[int], Dict[str, Any]],
        *,
        terminal: bool = False,
    ) -> BrokerEvent:
        """Publish where the body must contain its own sequence number.

        The SSE envelope echoes ``seq`` inside ``data`` so a client that
        persists event bodies can recover its resume cursor without also
        storing the frame's ``id:``. That means the body cannot be built until
        the sequence is assigned, hence a factory rather than a dict.
        """
        chan = self._channels.get(channel)
        if chan is None:
            chan = self._open_channel(channel)

        seq = chan.next_seq
        event = BrokerEvent(
            channel=channel,
            seq=seq,
            name=name,
            data=body_factory(seq),
            created_at=time.time(),
        )
        chan.next_seq += 1
        chan.last_activity = time.monotonic()
        if terminal:
            chan.terminal = True

        self._retain(chan, event)
        self._deliver(chan, event)
        return event

    def publish_to(
        self,
        channels: Iterable[str],
        name: str,
        data: Dict[str, Any],
        *,
        terminal: bool = False,
    ) -> List[BrokerEvent]:
        """Publish one event to several channels.

        The record-keyed channel (parity with the legacy WebSocket key) and the
        job-keyed channel (which a client can subscribe to *before* the first
        record id exists) carry the same events with independent sequences.
        """
        return [self.publish(c, name, data, terminal=terminal) for c in channels if c]

    def _retain(self, chan: _Channel, event: BrokerEvent) -> None:
        chan.buffer.append(event)
        while len(chan.buffer) > self._replay_buffer:
            chan.buffer.popleft()

    def _deliver(self, chan: _Channel, event: BrokerEvent) -> None:
        for sub in list(chan.subscribers):
            if sub.closed:
                chan.subscribers.discard(sub)
                continue
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop and count. The reader turns this into an explicit gap so
                # the consumer reconciles instead of trusting a partial view.
                sub.dropped += 1

    # ------------------------------------------------------------------
    # subscribing
    # ------------------------------------------------------------------

    def subscribe(self, channel: str, *, after_seq: Optional[int] = None) -> "Subscription":
        """Attach to ``channel``, optionally replaying everything after ``after_seq``.

        Replay is served from the ring buffer before any live event, and both
        share the subscriber's queue, so a resuming client sees one strictly
        increasing sequence with no reordering at the seam. If the requested
        cursor has already fallen out of the buffer the subscription reports a
        gap rather than pretending the replay was complete.
        """
        chan = self._channels.get(channel)
        if chan is None:
            chan = self._open_channel(channel)
        # Attaching to an already-terminal channel must not refresh its TTL: a
        # client that reconnects after `stream.terminal` (auto-retry at the
        # `retry:` hint) would otherwise pin the channel's memory forever
        # (review B-5).
        if not chan.terminal:
            chan.last_activity = time.monotonic()

        sub = _Subscriber(queue=asyncio.Queue(maxsize=self._subscriber_queue))
        chan.subscribers.add(sub)

        gap = False
        if after_seq is not None:
            backlog = [e for e in chan.buffer if e.seq > after_seq]
            # A cursor older than the whole buffer means frames were evicted
            # between the disconnect and the resume.
            if chan.buffer and after_seq + 1 < chan.buffer[0].seq:
                gap = True
            elif not chan.buffer and after_seq + 1 < chan.next_seq:
                gap = True
            elif not chan.buffer and chan.next_seq == 1 and after_seq >= 1:
                # The channel was recreated (TTL eviction or server restart) so
                # all history the cursor referred to is gone; report the gap
                # instead of claiming a clean resume (review B-3).
                gap = True
            for event in backlog:
                try:
                    sub.queue.put_nowait(event)
                except asyncio.QueueFull:
                    sub.dropped += 1
        return Subscription(self, channel, sub, resumed_with_gap=gap)

    def unsubscribe(self, channel: str, sub: _Subscriber) -> None:
        """Detach a subscriber and wake its reader."""
        sub.closed = True
        chan = self._channels.get(channel)
        if chan is not None:
            chan.subscribers.discard(sub)
            # Mirror the subscribe() guard: detaching the last listener of a
            # terminal channel must not refresh its TTL, or a client that
            # reconnects after `stream.terminal` pins the channel forever
            # (review N-2).
            if not chan.terminal:
                chan.last_activity = time.monotonic()
        # Sentinel so a reader parked on `get()` returns promptly.
        try:
            sub.queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    # ------------------------------------------------------------------
    # channel bookkeeping
    # ------------------------------------------------------------------

    def _open_channel(self, channel: str) -> _Channel:
        self._evict()
        chan = _Channel(name=channel)
        self._channels[channel] = chan
        return chan

    def _evict(self) -> None:
        """Drop idle channels, then enforce the hard channel ceiling.

        A channel is only ever removed when nothing is subscribed to it, so
        eviction cannot cut a live stream. The ceiling pass sheds the least
        recently active first.
        """
        now = time.monotonic()
        for name, chan in list(self._channels.items()):
            if chan.subscribers:
                continue
            if now - chan.last_activity >= self._channel_ttl_s:
                del self._channels[name]

        if len(self._channels) <= self._max_channels:
            return
        idle = [(c.last_activity, n) for n, c in self._channels.items() if not c.subscribers]
        idle.sort()
        overflow = len(self._channels) - self._max_channels
        for _, name in idle[:overflow]:
            self._channels.pop(name, None)

    # ------------------------------------------------------------------
    # introspection - used by the capability endpoint and tests
    # ------------------------------------------------------------------

    def channel_names(self) -> List[str]:
        return list(self._channels)

    def last_sequence(self, channel: str) -> int:
        """The highest sequence published to ``channel``; 0 when unknown."""
        chan = self._channels.get(channel)
        return chan.next_seq - 1 if chan else 0

    def is_terminal(self, channel: str) -> bool:
        chan = self._channels.get(channel)
        return bool(chan and chan.terminal)

    def subscriber_count(self, channel: str) -> int:
        chan = self._channels.get(channel)
        return len(chan.subscribers) if chan else 0

    def stats(self) -> Dict[str, Any]:
        """Coarse counters for diagnostics and the capability probe."""
        return {
            "channels": len(self._channels),
            "subscribers": sum(len(c.subscribers) for c in self._channels.values()),
            "buffered_events": sum(len(c.buffer) for c in self._channels.values()),
            "replay_buffer": self._replay_buffer,
            "subscriber_queue": self._subscriber_queue,
            "max_channels": self._max_channels,
            "channel_ttl_s": self._channel_ttl_s,
        }

    def retained(self, channel: str) -> List[BrokerEvent]:
        """Events still held for ``channel``, oldest first.

        The SSE route folds these into the snapshot it sends on attach, which
        is what lets a client that arrives mid-turn - or whose resume cursor
        has already been evicted - paint current state instead of waiting for
        the next delta. Returned as a list copy so a consumer iterating it
        cannot be tripped by a concurrent publish.
        """
        chan = self._channels.get(channel)
        return list(chan.buffer) if chan else []

    def close(self) -> None:
        """Release every subscriber. Called on application shutdown."""
        for name, chan in list(self._channels.items()):
            for sub in list(chan.subscribers):
                self.unsubscribe(name, sub)
        self._channels.clear()


class Subscription:
    """A borrowed view of one channel, safe to use as an async context manager.

    ``resumed_with_gap`` reports that the requested ``Last-Event-ID`` predated
    the retained buffer, so the caller must tell the client to reconcile rather
    than assume continuity.
    """

    def __init__(
        self,
        broker: EventBroker,
        channel: str,
        subscriber: _Subscriber,
        *,
        resumed_with_gap: bool = False,
    ) -> None:
        self._broker = broker
        self._channel = channel
        self._sub = subscriber
        self.resumed_with_gap = resumed_with_gap

    @property
    def pending(self) -> int:
        """Events queued for this subscriber but not yet read.

        The SSE route uses this to tell "attached to a finished channel with
        nothing to say" (close now) from "resuming a finished channel with a
        replay backlog" (drain it first, then close on the terminal frame).
        """
        return self._sub.queue.qsize()

    @property
    def channel(self) -> str:
        return self._channel

    @property
    def dropped(self) -> int:
        """Events discarded because this subscriber fell behind."""
        return self._sub.dropped

    def take_dropped(self) -> int:
        """Read and clear the drop counter, so a gap is reported once."""
        dropped, self._sub.dropped = self._sub.dropped, 0
        return dropped

    @property
    def is_closed(self) -> bool:
        """True once the broker released this subscriber (shutdown/detach).

        The reader needs this to tell a heartbeat tick from a shutdown: both
        surface as ``get() -> None``, but one should emit a keepalive and the
        other must end the response.
        """
        return self._sub.closed

    async def get(self, timeout: float) -> Optional[BrokerEvent]:
        """Next event, or ``None`` on timeout or close.

        A timeout is not an error: it is the heartbeat tick. Check
        :attr:`is_closed` to distinguish the two.
        """
        try:
            event = await asyncio.wait_for(self._sub.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        # ``None`` on the queue is the detach sentinel pushed by unsubscribe().
        if event is None:
            self._sub.closed = True
        return event

    def close(self) -> None:
        self._broker.unsubscribe(self._channel, self._sub)

    async def __aenter__(self) -> "Subscription":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.close()
