"""Process-local HTTP entity storage, never an authorization or freshness cache.

Every hit must be conditionally revalidated by the hub after current-login
admission. A global LRU survives the desktop route's per-request clients but
bounds retention across all config roots and accounts. No bearer is stored.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

Scope = tuple[str, str]
Authority = tuple[int, str, str]
Key = tuple[Scope, Authority, str, str, int]


@dataclass(frozen=True)
class Entity:
    body: bytes
    mime: str
    etag: str


class AvatarCache:
    def __init__(self, *, max_entries: int = 64, max_bytes: int = 8 * 1024 * 1024):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[Key, Entity] = OrderedDict()
        self._bytes = 0
        # Loopback test servers and embedded runtimes can use different event
        # loops/threads; only tiny synchronous bookkeeping is held under lock.
        self._lock = Lock()

    def _drop(self, key: Key) -> None:
        entity = self._entries.pop(key, None)
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
            if entity is not None:
                self._entries.move_to_end(key)
            return entity

    def discard(self, key: Key) -> None:
        with self._lock:
            self._drop(key)

    def put(self, key: Key, entity: Entity) -> None:
        with self._lock:
            self._drop(key)
            if len(entity.body) > self.max_bytes or self.max_entries < 1:
                return
            self._entries[key] = entity
            self._bytes += len(entity.body)
            while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                self._drop(next(iter(self._entries)))


AVATAR_CACHE = AvatarCache()
