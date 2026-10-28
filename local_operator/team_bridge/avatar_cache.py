"""Process-local HTTP entity storage, never an authorization or freshness cache.

Every hit must be conditionally revalidated by the hub after current-login
admission. A global LRU survives the desktop route's per-request clients but
bounds retention across all config roots and accounts. No bearer is stored.

RETENTION EXPIRY. Entries also age out after ``ttl_s`` (15 minutes) of wall
time since they were stored. That bounds how long a photo lingers in memory
after its sender stops appearing; it is NOT a freshness lifetime -- a live
entry still needs the hub's conditional 304 under the current login before
reuse, and an expired one simply costs a cold fetch. The same class backs the
notification composite cache (``notification_avatar.py``), whose keys share
the ``(scope, authority, ...)`` prefix so ``admit`` purges both on an auth
transition.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Any, Callable

Scope = tuple[str, str]
Authority = tuple[int, str, str]
#: ``(scope, authority, *identity)``. The raw cache uses ``(target, v, size)``
#: as identity; the composite cache adds its render variant. Only the two-item
#: prefix is interpreted here.
Key = tuple[Any, ...]

#: Retention, not freshness: see the module docstring.
RETENTION_S = 15 * 60


@dataclass(frozen=True)
class Entity:
    body: bytes
    mime: str
    etag: str


class AvatarCache:
    def __init__(
        self,
        *,
        max_entries: int = 64,
        max_bytes: int = 8 * 1024 * 1024,
        ttl_s: float = RETENTION_S,
        clock: Callable[[], float] = monotonic,
    ):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.ttl_s = ttl_s
        # Monotonic so a wall-clock jump can neither resurrect nor mass-expire.
        self._clock = clock
        self._entries: OrderedDict[Key, Entity] = OrderedDict()
        self._stored: dict[Key, float] = {}
        self._bytes = 0
        # Loopback test servers and embedded runtimes can use different event
        # loops/threads; only tiny synchronous bookkeeping is held under lock.
        self._lock = Lock()

    def _drop(self, key: Key) -> None:
        entity = self._entries.pop(key, None)
        self._stored.pop(key, None)
        if entity is not None:
            self._bytes -= len(entity.body)

    def admit(self, scope: Scope, authority: Authority | None) -> None:
        """Purge observed obsolete logins; an unknown owner admits nothing."""
        with self._lock:
            for key in list(self._entries):
                if key[0] == scope and key[1] != authority:
                    self._drop(key)

    def get(self, key: Key) -> Entity | None:
        with self._lock:
            entity = self._entries.get(key)
            if entity is not None and self._clock() - self._stored[key] >= self.ttl_s:
                self._drop(key)
                return None
            if entity is not None:
                self._entries.move_to_end(key)
            return entity

    def discard(self, key: Key) -> None:
        with self._lock:
            self._drop(key)

    def put(self, key: Key, entity: Entity) -> None:
        with self._lock:
            self._drop(key)
            # Sweep expired entries so dead bytes never hold the byte cap
            # against a live one (``get`` only expires what it is asked for).
            now = self._clock()
            for stale in [k for k, at in self._stored.items() if now - at >= self.ttl_s]:
                self._drop(stale)
            if len(entity.body) > self.max_bytes or self.max_entries < 1:
                return
            self._entries[key] = entity
            self._stored[key] = self._clock()
            self._bytes += len(entity.body)
            while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                self._drop(next(iter(self._entries)))


AVATAR_CACHE = AvatarCache()
#: Rendered notification composites (``notification_avatar.py``): same bounds
#: as the raw entities, keyed by the same ``(scope, authority)`` prefix.
COMPOSITE_CACHE = AvatarCache()


def admit(scope: Scope, authority: Authority | None) -> None:
    """Apply one observed login authority to EVERY avatar-derived cache.

    The single entry point so an auth transition (sign-out, A->B->A) cannot
    purge raw photos yet leave their rendered composites behind. Reads the
    module globals at call time so a test's monkeypatched cache is honoured.
    """

    AVATAR_CACHE.admit(scope, authority)
    COMPOSITE_CACHE.admit(scope, authority)
