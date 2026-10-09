"""A process-wide cache of COLD REPLAY results, keyed by journal identity.

WHY THIS MODULE EXISTS. A conversation with no frontend checkpoint makes the
replay read the whole journal (``read_replay_suffix`` scans to BOF when there is
no compaction boundary to stop at), and that cost is paid PER REQUEST rather than
per session: the desktop's cold facade is built, used and detached, so the
snapshot, ``/history`` and the SSE frame each rebuild it. Measured on a 35.5 MB
journal with no checkpoint: snapshot 212 ms, snapshot again 261 ms, ``/history``
207 ms, first SSE frame 201 ms — about 200 ms per request, repeatedly, on a
session that carries 15 of the 40 largest real journals. With a checkpoint the
same file costs 3 ms.

The fix has two halves and they are independent. One is the checkpoint the
runtime writes at every turn end (``FrontendStateStore.checkpoint``), which
bounds journals written from now on — but only when a frontend was attached, so
it does nothing for the sessions already on disk. This module is the other half,
and it is what those existing sessions need: the same file, opened three times,
is parsed once.

WHAT THE KEY IS, and why ``size`` is in it. ``(directory, st_ino, st_size,
st_mtime_ns)`` PLUS the request's own cut and wanted custom types, because the
result is a function of all of them. Size is not redundant with mtime here:
:meth:`Transcript._write_entries` RESTORES the file's mtime after a bookkeeping
batch (``preserve_mtime``, the activity clock), so a spend record or a
system/todo snapshot can be appended inside the same mtime instant — and even at
the same nanosecond after a restore — while the bytes on disk moved. Without
size in the key a request served right after such an append would answer from
the older parse. The reverse (a file whose bytes change while size AND inode stay
put) is the residual ``page_cache`` documents for its own key and it is the same
one here: it is unreachable from this codebase's writers, which either append or
replace atomically.

WHY IT IS NOT A ``Transcript``-LEVEL CACHE, and why it is not keyed by session
id: three processes can touch one journal (the desktop server, the TUI, the
mobile daemon), so the identity has to come from the filesystem rather than from
anything this process remembers — exactly the argument ``page_cache`` makes for
its own key. The dependency direction is one-way: this module imports
``transcript`` and ``page_cache``, never the reverse.

MEMORY. Bounded by BOTH entries and accounted bytes, with the same
:func:`~local_operator.session.page_cache.retained_bytes` instrument the page
cache and the display window use, so a cache that admits nothing on the big
sessions is not mistaken for one that is bounded. The accounted size of a
replayed history is ~1.7-4.7x its wire size (measured: 6.1 MB for the 1291
messages of a 35 MB journal), so the budget is stated in those terms. An
oversized entry is refused outright rather than admitted and then evicted: a
cache that thrashes on one session is worse than a cache that never holds it,
because the thrash still pays the parse.

ONE NAMED LIMITATION, because it is a real divergence from a fresh parse rather
than a theoretical one. The replay hydrates externalized attachments, and the key
above covers the JOURNAL, not the attachment store. A reference that did not
resolve at build time — the "transcript references missing attachment" warning
``_internalize_attachments`` logs — stays unresolved in the cached answer until
the journal changes, where a fresh parse would try again. That is tolerable
because the store is content-addressed and write-once (the file name IS the
digest, and the store re-hashes to prove it), so the only way the same digest
resolves later is a store restored behind the reader's back, and the same
restore moves the directory's own mtime. Keying on that mtime instead would churn
the cache on every new screenshot in ANY session — a miss and a full parse after
each, which is the cost this module exists to remove. Named here so the next
reader can price the trade rather than rediscover it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from collections.abc import Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from local_operator.session.page_cache import retained_bytes
from local_operator.session.transcript import TRANSCRIPT_FILENAME

logger = logging.getLogger(__name__)

#: LRU bounds. Both, for the reason ``page_cache`` states for its own pair: an
#: entry count alone is unbounded in bytes on a long conversation (6.1 MB
#: accounted for one 35 MB journal's 1291 replayed messages, and a 121 MB
#: journal's window is larger), and a byte budget alone would evict to a single
#: entry on exactly those sessions.
#:
#: 8 entries is "the sessions this process has open right now" for the desktop
#: server and the phone's daemon — a user moves between a handful — and 48 MiB
#: accounted holds several ordinary conversations while REFUSING the worst real
#: one, which is the right answer for it: re-parsing that session is cheaper
#: than pinning its history in a server that also holds a renderer's pages.
REPLAY_CACHE_ENTRIES = 8
REPLAY_CACHE_BYTES = 48 * 1024 * 1024
#: The same fixed allowance the page cache uses: LRU nodes, key objects and the
#: small side values, so the effective ceiling stays UNDER the nominal one.
_REPLAY_CACHE_ALLOWANCE = 4096


@dataclass(frozen=True)
class ReplayKey:
    """One cold replay request: which file, at which version, for which cut.

    ``directory`` is a string (hashable, and comparable across the processes
    that spell the same directory differently). The four stat fields are the
    invalidation; ``through_id`` and the two type tuples are the request, because
    the answer differs per cut and per wanted custom row — a reconnect's
    ``through_id`` and a cold open's ``None`` are different reads even though
    they read the same bytes.
    """

    directory: str
    inode: int
    size: int
    mtime_ns: int
    through_id: str | None
    #: Whether an absent ``through_id`` is an error (the negotiated cut) or a
    #: legacy fallback. Part of the REQUEST rather than the file: the same bytes
    #: answer differently under the two, so a cache entry made for one must not
    #: serve the other.
    strict_cut: bool
    checkpoint_types: tuple[str, ...]
    opportunistic_types: tuple[str, ...]


@dataclass(frozen=True)
class ColdReplay:
    """What one cold replay produced, in the shape ``_read_transcript`` needs.

    ``messages`` is the replayed history in append order. It is handed to
    callers as a fresh LIST over the same message objects, because the facade
    PREPENDS to it in place when older pages arrive (``self._history[:0] = ...``)
    and a shared list would leak one facade's scrollback into the next facade's
    state. The message objects themselves are shared, which is safe for the same
    reason the transcription does not mutate them after construction: every
    replay path builds new messages and prunes by assignment during the build.
    """

    messages: tuple[Any, ...]
    checkpoint: dict[str, Any] | None
    checkpoints: dict[str, dict[str, Any]]
    order: dict[str, int]
    #: The cold accounting seed (``seed_reported_usage`` over the rows this same
    #: parse read), or ``None`` when the request does not seed one. Computed
    #: INSIDE the build so the entries it reads — the largest thing a cold read
    #: holds, 25,513 rows and 6.1 MB on the reference 35 MB journal — are never
    #: retained by the cache.
    seed_usage: Any = None
    #: Bytes of journal this result read (``ReplaySuffix.bytes_read``), for the
    #: caller's own evidence; not a contract.
    bytes_read: int = 0
    retained: int = 0


@dataclass
class _ReplayCache:
    max_entries: int = REPLAY_CACHE_ENTRIES
    max_bytes: int = REPLAY_CACHE_BYTES
    _entries: "OrderedDict[ReplayKey, ColdReplay]" = field(default_factory=OrderedDict)
    _bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, key: ReplayKey) -> ColdReplay | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
            return entry

    def put(self, key: ReplayKey, value: ColdReplay) -> None:
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous.retained
            self._entries[key] = value
            self._bytes += value.retained
            while self._entries and (
                len(self._entries) > self.max_entries or self._bytes > self.max_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.retained

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def accounted_bytes(self) -> int:
        with self._lock:
            return self._bytes


_REPLAY_CACHE = _ReplayCache()
#: ``key -> (loop, task)`` for an in-flight cold read. The task is the read
#: itself, so a second caller for the same key shares it instead of starting a
#: second parse — the shape ``page_cache`` uses for its pages, and here it is the
#: NORMAL case rather than a race: an open issues its snapshot and its SSE
#: stream back to back, and both build a cold facade.
_FLIGHTS: dict[ReplayKey, tuple[asyncio.AbstractEventLoop, asyncio.Task[ColdReplay]]] = {}


def replay_cache() -> _ReplayCache:
    """The process-wide cache, for tests and for a caller that must clear it."""
    return _REPLAY_CACHE


def replay_key(
    directory: str | Path,
    *,
    through_id: str | None,
    strict_cut: bool,
    checkpoint_types: tuple[str, ...] | str | None = None,
    opportunistic_types: tuple[str, ...] | str | None = None,
) -> ReplayKey | None:
    """``directory``'s current replay key, or ``None`` when the journal is absent.

    A missing journal is not an error here: the caller's own read produces the
    authoritatively-worded failure, and this module never invents a second one.
    The two type arguments are normalised exactly as ``read_replay_suffix``
    normalises them, so a bare string and a one-element tuple name the same
    read.
    """

    def _normalised(value: tuple[str, ...] | str | None) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        return tuple(value or ())

    path = Path(directory) / TRANSCRIPT_FILENAME
    try:
        stat = path.stat()
    except OSError:
        return None
    return ReplayKey(
        directory=str(Path(directory)),
        inode=stat.st_ino,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        through_id=through_id,
        strict_cut=strict_cut,
        checkpoint_types=_normalised(checkpoint_types),
        opportunistic_types=_normalised(opportunistic_types),
    )


def cached_replay(key: ReplayKey) -> ColdReplay | None:
    """A previously produced replay for ``key``, or ``None``."""
    return _REPLAY_CACHE.get(key)


def publish_replay(key: ReplayKey, **values: Any) -> ColdReplay:
    """Build and store a :class:`ColdReplay`, refusing one too large to hold.

    The refusal is silent by design — a large conversation is served from a
    fresh parse exactly as it is today, and logging every miss would be noise on
    the one session that always misses. The measurement is available through
    :func:`replay_cache`.
    """
    messages = tuple(values.pop("messages"))
    retained = 0
    sized = retained_bytes(messages, REPLAY_CACHE_BYTES)
    if sized is None:
        # Over budget on its own: hand the caller the value WITHOUT caching it,
        # so the answer is never affected by the bound.
        logger.debug("cold replay for %s is too large to cache", key.directory)
    else:
        retained = sized + _REPLAY_CACHE_ALLOWANCE
        value = ColdReplay(messages=messages, retained=retained, **values)
        _REPLAY_CACHE.put(key, value)
        return value
    return ColdReplay(messages=messages, retained=retained, **values)


def invalidate(directory: str | Path) -> None:
    """Drop every cached replay for ``directory`` (any version, any cut)."""
    target = str(Path(directory))
    with _REPLAY_CACHE._lock:
        for key in [key for key in _REPLAY_CACHE._entries if key.directory == target]:
            entry = _REPLAY_CACHE._entries.pop(key)
            _REPLAY_CACHE._bytes -= entry.retained


async def load_cold_replay(
    key: ReplayKey | None,
    build: Callable[[], Coroutine[Any, Any, ColdReplay]],
) -> ColdReplay:
    """``build()``, once per key, shared by every concurrent caller and cached.

    Single-flight on the event loop, with the same three properties the page
    cache's reader documents: a follower's cancellation cannot cancel the read
    the others depend on (``shield``), the flight entry is removed by a
    done-callback so a failed leader cannot orphan it, and the loop is carried
    with the entry so a flight left behind by a torn-down test loop reads as
    ABSENT rather than as a future bound to a dead loop.
    """
    if key is None:
        return await build()
    cached = _REPLAY_CACHE.get(key)
    if cached is not None:
        return cached
    loop = asyncio.get_running_loop()
    flight = _FLIGHTS.get(key)
    if flight is not None and flight[0] is loop:
        return await asyncio.shield(flight[1])
    task = asyncio.create_task(build())
    _FLIGHTS[key] = (loop, task)
    task.add_done_callback(lambda _settled: _FLIGHTS.pop(key, None))
    return await asyncio.shield(task)


#: Re-exported so a caller cannot mistake this module's identity rule for a
#: second one: the page cache's ``retained_bytes`` is the instrument both use.
__all__ = [
    "ColdReplay",
    "REPLAY_CACHE_BYTES",
    "REPLAY_CACHE_ENTRIES",
    "ReplayKey",
    "cached_replay",
    "invalidate",
    "load_cold_replay",
    "publish_replay",
    "replay_cache",
    "replay_key",
]
