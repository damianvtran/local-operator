"""A per-account token bucket for the resolve route — a bound, not a quota.

WHY THE INDEX CAP IS NOT ENOUGH. ``index < PER_MESSAGE_CAP`` is the RENDERER's
numbering: it says "this is the third link in the message I am drawing", and a
client that renumbers its links ``0, 1, 2, 0, 1, 2 …`` gets an unbounded stream of
resolutions from one message. The cap exists so a paste of forty links cannot
become forty outbound fetches, and a property of the caller's own claim cannot be
the server-side bound. Code review round 1 named exactly that gap.

WHAT IT IS NOT: not a quota, not an entitlement, and not an anti-abuse system for
the internet. One process serves one local install, and the caller is either the
operator's own renderer or something already holding their bearer. The bucket is
here so a bug or a loop in a client cannot turn one message into a fetch flood,
which costs the operator bandwidth and spends this install's allowance at origins
that rate-limit by IP. It is deliberately generous for a human and only bites a
burst.

BOUNDED MEMORY, ALSO DELIBERATELY. One bucket per account is two floats; the map
is capped, dropping the least recently touched bucket first. Nothing here is
persisted and nothing survives the process — a restart is not something a client
can choose.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from time import monotonic
from typing import Callable, Final

#: Burst depth: resolutions an account may issue back to back before the bucket
#: has to refill. A message list of 20 links with three previews each is 60
#: resolutions, but they arrive as the renderer draws them and a WARM card costs no
#: request, so the depth only has to cover a screenful of cold links.
BURST: Final = 20

#: Refill rate in tokens per second: 240 resolutions a minute, which no human
#: reads and every legitimate scroll stays under.
REFILL_PER_SECOND: float = 4.0

#: Distinct accounts tracked at once. One install has a handful; the cap means the
#: map cannot be grown at all, rather than "cannot be grown because the id comes
#: from the local session".
MAX_ACCOUNTS: Final = 64

#: The ``Retry-After`` reported by a bucket configured with no refill at all.
_NO_REFILL_RETRY_S: Final = 3600.0


class PreviewThrottled(Exception):
    """A resolution that was about to reach the origin, refused by the bucket.

    Raised from CORE rather than the route because the charge has to happen where a
    fetch does — charging at the HTTP boundary made a re-render of already-cached
    links spend tokens it never used, which is a user-visible bug (seven messages ×
    three links exhausts a burst of twenty on a purely warm screen). The route maps
    this to a ``429`` with ``Retry-After``; core stays HTTP-free.
    """

    def __init__(self, retry_after_s: float) -> None:
        super().__init__("preview resolves are throttled")
        self.retry_after_s = retry_after_s


class ResolveThrottle:
    """Per-account token bucket. ``allow`` is a yes/no; ``retry_after_s`` the wait."""

    def __init__(
        self,
        *,
        burst: float = BURST,
        refill_per_second: float = REFILL_PER_SECOND,
        max_accounts: int = MAX_ACCOUNTS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.burst = burst
        self.refill_per_second = refill_per_second
        self.max_accounts = max_accounts
        # Monotonic so a wall-clock jump can neither hand out a windfall of tokens
        # nor lock an account out until the clock catches up.
        self._clock = clock
        self._level: OrderedDict[str, float] = OrderedDict()
        self._stamped: dict[str, float] = {}
        # Tiny synchronous bookkeeping only, under a lock: the route runs on the
        # loop, and tests may drive this from a thread.
        self._lock = Lock()

    def allow(self, account_id: str) -> bool:
        """Spend one token for ``account_id``; ``False`` when the bucket is empty.

        A NEW account only gets a bucket if there is a slot for it, and a slot is
        only freed by forgetting a bucket that still holds its WHOLE burst — see
        :meth:`_make_room`. When every tracked bucket is depleted the new account is
        refused instead: over-counting is the safe direction for a bound, and one
        install has a handful of accounts.
        """
        with self._lock:
            now = self._clock()
            if account_id not in self._level and len(self._level) >= self.max_accounts:
                self._make_room(now)
                if len(self._level) >= self.max_accounts:
                    return False
            if self._refill(account_id, now) < 1.0:
                return False
            self._level[account_id] -= 1.0
            return True

    def retry_after_s(self, account_id: str) -> float:
        """Seconds until this account holds a token again, for ``Retry-After``.

        Read AFTER a refusal, so a refused client waits the real interval instead of
        retrying into the same refusal. Deliberately does not CREATE a bucket: a
        read must not be able to grow the map.
        """
        if self.refill_per_second <= 0:
            # A bucket with no refill never recovers. Reported as an hour rather
            # than infinity because this number becomes a ``Retry-After`` header
            # and a client needs a finite one.
            return _NO_REFILL_RETRY_S
        with self._lock:
            now = self._clock()
            level = self._level.get(account_id, self.burst)
            last = self._stamped.get(account_id, now)
            level = min(self.burst, level + max(0.0, now - last) * self.refill_per_second)
        missing = max(0.0, 1.0 - level)
        return round(max(0.05, missing / self.refill_per_second), 2)

    def _refill(self, account_id: str, now: float) -> float:
        """``account_id``'s level at ``now``, credited and re-ordered. Holds the lock."""
        level = self._level.get(account_id, self.burst)
        last = self._stamped.get(account_id, now)
        level = min(self.burst, level + max(0.0, now - last) * self.refill_per_second)
        self._level[account_id] = level
        self._level.move_to_end(account_id)
        self._stamped[account_id] = now
        return level

    def _make_room(self, now: float) -> None:
        """Forget one IDLE-FULL bucket, so a new account can have its slot. Holds the lock.

        Evicting a DEPLETED bucket would be a refund: the account would come back to
        a full burst it had already spent, which is a way to buy tokens by being
        forgotten (code review round 2). A bucket still holding its whole burst is
        the only safe one to drop — recreating it later at full burst is
        indistinguishable from having kept it.
        """
        for account_id in list(self._level):
            # Refill on the fly rather than trusting the stored level: the stored
            # one is only current as of that bucket's last touch, so an idle bucket
            # reads as depleted and would never be evictable.
            level = self._level[account_id]
            last = self._stamped.get(account_id, now)
            level = min(self.burst, level + max(0.0, now - last) * self.refill_per_second)
            if level >= self.burst:
                del self._level[account_id]
                self._stamped.pop(account_id, None)
                return


#: The process-wide bucket, built on first use — the same shape, and for the same
#: reason, as :func:`~local_operator.team_bridge.link_preview.cache.get_preview_cache`:
#: one ``lop serve`` process serves one config root, and a function (not a
#: module-level instance) is what lets a test inject its own clock through the seam
#: the route reads.
_PROCESS_THROTTLE: ResolveThrottle | None = None


def get_resolve_throttle() -> ResolveThrottle:
    """The one throttle this process has."""

    global _PROCESS_THROTTLE
    if _PROCESS_THROTTLE is None:
        _PROCESS_THROTTLE = ResolveThrottle()
    return _PROCESS_THROTTLE
