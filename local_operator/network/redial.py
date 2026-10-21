"""Per-peer redial state: when a lost link is retried, and what surfaces say meanwhile.

WHY THIS EXISTS (2026-10-21, cloud-node-1). The node's link dropped at 14:45:47Z
(``link_idle`` then ``link_closed cause timeout``) and the desktop sidebar went
straight to an amber ``Link down · unknown``. Three things were missing, and
this module is the one place that holds all three:

1. **A schedule.** Nothing re-established a lost link on its own: every dial
   came from a surface poll (the peer listing, a stream open), so the retry rate
   was the UI's poll rate (the audit shows one ``handshake_refused`` every 2-6 s)
   and the ``RECONNECT_*`` bounds in :mod:`wire` were declared and never read.
   :meth:`RedialTracker.due` is the schedule the relay's redial loop consults:
   exponential from ``RECONNECT_MIN_S`` to ``RECONNECT_MAX_S`` with
   ``RECONNECT_JITTER`` either side, reset by a successful link.
2. **A grace window.** A link that drops and comes back inside a minute is a
   blip, not an outage. :meth:`RedialTracker.snapshot` reports ``reconnecting``
   until :data:`GRACE_S` has passed AND :data:`GRACE_ATTEMPTS` redials have
   failed, and only then ``down`` — so the sidebar can say ``Reconnecting…``
   first and escalate on evidence rather than on the first missed beat.
3. **The facts behind the words.** Attempt count, next retry time and the last
   failure cause (``timeout``, ``peer_closed_silently``, ``connect_failed:…``)
   ride the peer table, so a person reading the tooltip sees why, not a guess.

WHAT IT DOES NOT DO: dial. The relay owns sockets; this owns arithmetic and is
pure apart from the injectable clock and random source, so every rule here is a
unit test with no network. Thread-safe: the relay's reader threads report losses
while its redial loop and listing threads read.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from local_operator.network import wire

#: How long after a link is lost surfaces keep saying ``reconnecting``. 90 s
#: covers three keepalive intervals and a relay restart (measured restart-to-
#: serving on the reference node is well under that), which are the blips that
#: should never read as an outage.
GRACE_S = 90.0

#: Failed redials before a lost link may escalate to ``down`` — BOTH this and
#: :data:`GRACE_S` must pass. Time alone would escalate a peer this device never
#: got round to retrying; attempts alone would escalate a fast-failing blip
#: (a relay restart refuses connections for a second or two) in under ten seconds.
GRACE_ATTEMPTS = 3

#: The states :meth:`RedialTracker.snapshot` reports.
CONNECTED = "connected"
RECONNECTING = "reconnecting"
DOWN = "down"


@dataclass
class PeerRedial:
    """One peer's redial bookkeeping. All times are ``time.time()`` seconds."""

    device_id: str
    lost_at: float
    last_frame_at: float | None = None
    attempts: int = 0
    last_cause: str = ""
    next_retry_at: float = 0.0
    last_attempt_at: float | None = None
    #: Seeded at relay start rather than lost: never connected in this process.
    #: Its start-up dial is due at once; if that fails it becomes an ordinary
    #: loss (``failed`` clears the flag) and is retried on the capped backoff.
    #: The relay never seeds a member it has no routable address for (a dial-only
    #: device declares loopback only), so nothing here redials a laptop forever.
    seeded: bool = False
    history: list[str] = field(default_factory=list)


def backoff_s(attempt: int, *, rand: Callable[[], float] = random.random) -> float:
    """The wait before redial number ``attempt + 1``, jittered and capped.

    ``attempt`` is the number of redials that have already failed. The cap is
    applied BEFORE the jitter so the spread survives at the ceiling (60 s ± 25 %
    rather than everyone retrying in lock-step at exactly 60 s), and the result is
    never below ``RECONNECT_MIN_S * (1 - jitter)``.
    """
    base = min(wire.RECONNECT_MAX_S, wire.RECONNECT_MIN_S * (2 ** max(0, attempt)))
    spread = wire.RECONNECT_JITTER * (2.0 * rand() - 1.0)
    return max(0.0, base * (1.0 + spread))


class RedialTracker:
    """Which peers have lost links, when each is next due, and what to call it."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        rand: Callable[[], float] = random.random,
    ) -> None:
        self._clock = clock
        self._rand = rand
        self._lock = threading.Lock()
        self._peers: dict[str, PeerRedial] = {}

    # -- transitions -----------------------------------------------------------

    def seed(self, device_id: str) -> None:
        """Mark a member for ONE start-up dial (no-op when already tracked)."""
        with self._lock:
            if device_id in self._peers:
                return
            now = self._clock()
            self._peers[device_id] = PeerRedial(
                device_id=device_id, lost_at=now, next_retry_at=now, seeded=True
            )

    def lost(self, device_id: str, *, cause: str, last_frame_at: float | None = None) -> None:
        """A link to ``device_id`` closed. The first redial is due after one backoff.

        A deliberate close (``disconnect``, shutdown) is still recorded: whether to
        redial is the loop's decision (it stops with the relay), and the surfaces
        need the last-frame time either way.
        """
        with self._lock:
            now = self._clock()
            current = self._peers.get(device_id)
            if current is not None and not current.seeded:
                # A second link to the same device closing while we already
                # track a loss (the dial and listener halves of one pair) must
                # not reset the grace clock.
                current.last_frame_at = max(current.last_frame_at or 0.0, last_frame_at or 0.0)
                current.last_cause = cause or current.last_cause
                return
            self._peers[device_id] = PeerRedial(
                device_id=device_id,
                lost_at=now,
                last_frame_at=last_frame_at,
                last_cause=cause,
                next_retry_at=now + backoff_s(0, rand=self._rand),
            )

    def connected(self, device_id: str) -> None:
        """A link is up again: forget the loss and reset the backoff."""
        with self._lock:
            self._peers.pop(device_id, None)

    def failed(self, device_id: str, *, cause: str) -> None:
        """One redial (or any dial) of a tracked peer failed with ``cause``."""
        with self._lock:
            peer = self._peers.get(device_id)
            if peer is None:
                return
            now = self._clock()
            peer.attempts += 1
            peer.seeded = False
            peer.last_attempt_at = now
            peer.last_cause = cause or peer.last_cause
            peer.next_retry_at = now + backoff_s(peer.attempts, rand=self._rand)

    def forget(self, device_id: str) -> None:
        with self._lock:
            self._peers.pop(device_id, None)

    # -- reads -----------------------------------------------------------------

    def due(self) -> list[str]:
        """Device ids whose next redial time has come, oldest-due first."""
        with self._lock:
            now = self._clock()
            rows = [
                peer
                for peer in self._peers.values()
                if peer.next_retry_at <= now
            ]
            rows.sort(key=lambda peer: peer.next_retry_at)
            return [peer.device_id for peer in rows]

    def snapshot(self, device_id: str) -> dict[str, object]:
        """The ``redial`` block a peer-table row carries.

        ``state`` is ``connected`` for an untracked peer. A seeded peer whose
        start-up dial has not run yet has no loss to describe, so it reads the
        same; the listing's own probe is what reports it until a dial fails.
        """
        with self._lock:
            peer = self._peers.get(device_id)
            if peer is None or (peer.seeded and peer.attempts == 0):
                return {"state": CONNECTED, "attempts": 0}
            now = self._clock()
            escalated = (now - peer.lost_at) >= GRACE_S and peer.attempts >= GRACE_ATTEMPTS
            return {
                "state": DOWN if escalated else RECONNECTING,
                "attempts": peer.attempts,
                "lost_at": round(peer.lost_at, 3),
                "last_frame_at": (
                    round(peer.last_frame_at, 3) if peer.last_frame_at is not None else None
                ),
                "next_retry_at": round(peer.next_retry_at, 3),
                "last_error": peer.last_cause,
                "grace_until": round(peer.lost_at + GRACE_S, 3),
            }

    def tracked(self) -> list[str]:
        with self._lock:
            return list(self._peers)


@dataclass
class _Coalesced:
    first_at: float
    suppressed: int = 0


class RefusalCoalescer:
    """Collapse repeats of one identical refusal into one audit row per window.

    WHY: the operator's ``audit.jsonl`` reached 5.3 MB, and the refusal classes
    repeat by construction — a stale identity re-dialling (135 rows for one
    expired member), a peer that refuses silently (47 rows in two days), a LAN
    neighbour on another network (460 ``unknown_network`` rows). Each repeat is
    the same fact. The first is written in full; repeats inside :attr:`window_s`
    are counted, and the next row after the window carries ``repeats: N`` so the
    count is never lost — only the duplicate lines are.
    """

    def __init__(self, window_s: float = 300.0, *, clock: Callable[[], float] = time.time):
        self.window_s = window_s
        self._clock = clock
        self._lock = threading.Lock()
        self._seen: dict[tuple[str, ...], _Coalesced] = {}

    def admit(self, key: tuple[str, ...]) -> tuple[bool, int]:
        """``(write_this_row, repeats_since_the_last_written_row)``."""
        with self._lock:
            now = self._clock()
            entry = self._seen.get(key)
            if entry is not None and now - entry.first_at < self.window_s:
                entry.suppressed += 1
                return False, entry.suppressed
            repeats = entry.suppressed if entry is not None else 0
            self._seen[key] = _Coalesced(first_at=now)
            if len(self._seen) > 512:
                # Bounded: the oldest half goes. A key that returns later is
                # simply written in full again — never a lost row.
                for stale in sorted(self._seen, key=lambda k: self._seen[k].first_at)[:256]:
                    self._seen.pop(stale, None)
            return True, repeats
