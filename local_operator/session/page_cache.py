"""A process-wide cache of decoded transcript pages, with single-flight.

WHY THIS MODULE EXISTS, AND WHY IT IS NOT IN ``transcript.py``. The desktop open
path reads the same journal page more than once per open — the authoritative
``snapshot()`` page and then the SSE open frame's, which is the same read again
whenever the frontend cursor has not moved between the two requests (the normal
case for an open) — and two more surfaces re-read the same page on a timer or a
re-converge: the child panel at 1 Hz per open child, and the renderer's
``reconcileTail`` walk, which re-requests pages it already has and merges them by
id. After the backward page reader landed (#1109) each of those reads is cheap on
its own — 1.7 ms on the operator's 261 MB conversation, 8.1 ms for a suffix read
— so this is a SECOND-ORDER optimisation, not the fix for a slow open: it removes
the REPEAT, and it makes switching back to an already-visited conversation touch
no disk at all. It sits at the page read rather than on ``Transcript`` because the
TUI's page reads hold no ``Transcript`` at all (``tui/widgets/subagent_view.py``)
and the desktop bridge's history read is deliberately independent of the runtime.

The module is separate because ``transcript.py`` is a leaf the stdlib-only paths
and the TUI both import, and it holds no process-wide state today. A cache with a
single-flight map is state, and it belongs in a module whose name says so. The
dependency direction is one-way — ``page_cache`` imports ``transcript``, never the
reverse — and ``history_window`` imports the byte instrument below, so nothing can
cycle.

WHAT THE KEY IS. File IDENTITY, not a generation counter. Three writers can touch
one journal from three processes (the desktop server, the TUI, the mobile daemon),
so a counter that only this process observes would serve a stale page across a
compaction — and a WRONG page of the right conversation is exactly the failure
mode this design has to avoid. ``(st_ino, st_size)`` is read from the filesystem,
which makes the invalidation rules follow from how the journal is written rather
than from notifications anyone must remember to send:

- an append moves ``st_size``, so the tail page's old entry is never asked for
  again and ages out of the LRU;
- a rollback (``os.truncate`` on an append that failed) moves it back, and if it
  lands on a size that is still cached that is the CORRECT answer — the rollback
  restores the exact pre-append bytes, so the cached page describes the file that
  is now there;
- ``compact_file`` writes a temp file and ``os.replace``s it, which is a NEW
  inode;
- ``mtime`` is deliberately NOT in the key even though the sibling
  ``DurableFoldCache`` carries one: an append under ``preserve_mtime`` rewinds the
  transcript's mtime on purpose, so mtime would serve a stale tail page after a
  bookkeeping batch while size moves on every append regardless.

NO WRITER CALLBACK, NO GENERATION SUBSCRIPTION, and no lock besides the one the
pool already has (see ``DesktopSessions.session``): the fingerprint IS the
invalidation. Anything that notified this cache would be a mechanism beside the
one that already decides (``desktop_sessions.py``'s ``warm`` docblock is the
in-tree statement of that rule).

THE RESIDUAL, named here because ``TranscriptPageCache``'s docstring points at
it. Identity is ``(st_ino, st_size)``, so a file whose bytes change while BOTH
stay the same is served as a hit. Exactly two shapes reach that:

- an IN-PLACE rewrite at an unchanged size. ``_write_entries``'s rebuild branch
  (``transcript.py``) opens the journal ``"w"`` on the same inode, so it keeps
  the inode AND can land on an equal byte length;
- an INODE RECYCLED onto an identical byte length after an ``os.replace`` or an
  unlink, which is the unbounded one: it is the kernel's business, not this
  cache's.

Neither is reachable from this codebase's writers today — the rebuild branch
runs only when the journal is ABSENT (so there are no older bytes to serve), and
``compact_file`` replaces through a temp file, which is the new-inode path the
key is built to catch. It stays named rather than assumed because the alternative
is a reader inferring a guarantee this class does not make: the delta against the
sibling ``DurableFoldCache`` is that ``mtime`` is excluded here deliberately, so
``size`` is the only part of the key that moves on an append.

CROSS-THREAD DISCIPLINE, and this is the one thing a caller must not get wrong.
The cache and the flight map are mutated ONLY from the event loop: ``get`` in the
façade's own frame, ``put`` in the leader's publish step after its
``asyncio.to_thread`` returns. A closure handed to ``asyncio.to_thread`` must
never call ``get``/``put`` — that would silently introduce cross-thread mutation,
which is the shape of the freeze #401 was.
"""

from __future__ import annotations

import asyncio
import sys
from collections import OrderedDict
from dataclasses import dataclass, is_dataclass
from pathlib import Path

from pydantic import BaseModel

from local_operator.session.transcript import (
    TRANSCRIPT_FILENAME,
    TranscriptPage,
    read_transcript_page,
    validate_page_request,
)

#: LRU bounds. BOTH exist, and that is measured rather than tidy: a 100-row page
#: was 167 KiB on one real conversation and 2.1 MiB on another, a 500-row page
#: 1.4-7.1 MiB, and one single row 937 KiB — so "16 entries" alone is an
#: unbounded cache in bytes on a long-lived server, and a byte budget alone would
#: evict to a single entry on the big conversations.
#:
#: THE BUDGET IS IN ACCOUNTED BYTES, WHICH ARE NOT WIRE BYTES, and that is why it
#: is larger than the sibling display cache's 2 MiB: ``retained_bytes`` charges
#: ``sys.getsizeof`` per object, which measured **1.7-2.7x** a page's serialized
#: size across six real conversations (a 100-row page on the operator's store:
#: 202 KiB/504 KiB, 306 KiB/703 KiB, 691 KiB/1502 KiB, 2.0 MiB/5.4 MiB, and
#: 48 MiB/84 MiB on the session whose rows are enormous). At 4 MiB accounted the
#: cache admitted NOTHING for any of the ordinary ones — measured, by the
#: benchmark's own oversize tally, on a 9 MB conversation whose 100-row page
#: accounts 4.6 MiB — so the bound would be a bound on an empty cache. 24 MiB
#: holds roughly five ordinary pages and still REFUSES the 84 MiB one, which is
#: the right answer for it: holding that page costs more than re-reading it.
PAGE_CACHE_ENTRIES = 16
PAGE_CACHE_BYTES = 24 * 1024 * 1024
#: The same fixed allowance the display cache uses, for the LRU nodes and the
#: key's own objects. Accounted bytes are an OVER-estimate (``sys.getsizeof``
#: charges CPython per-object overhead), so the effective ceiling is under the
#: nominal one — the safe direction, and the reason a re-tune must be sized
#: against ACCOUNTED bytes rather than wire-sized ones.
_PAGE_CACHE_ALLOWANCE = 4096


def retained_bytes(value: object, limit: int) -> int | None:
    """Bound ``value``'s reachable data, including keys, tool metadata and media.

    Moved here from ``history_window`` (its only other call site was
    ``_DisplayWindowCache.put``) because this cache needs the same instrument and
    a second copy of a walk this fiddly would drift. ``None`` means "over
    ``limit``", which the callers read as "do not admit this" — the same
    conservative answer a page too large to cache gets.

    THE ``dataclass`` BRANCH IS LOAD-BEARING, not a nicety. The walk descends
    into ``BaseModel`` (via ``__dict__``), ``dict`` and the container types, but
    ``TranscriptPage`` and ``TranscriptEntry`` are plain ``@dataclass``es, so
    without the branch below the walk stops AT the page and never sees its rows:
    a 2 MiB page would be accounted as a few dozen bytes and the byte bound
    would be inert. The test that pins this asserts the figure EXCEEDS the
    page's serialized size, because a dead instrument returns a reading rather
    than an error (AGENTS.md §Timing) and this one is one missing branch away
    from reporting megabytes as 48 bytes.

    Deliberately an OVER-estimate, exactly as ``sys.getsizeof`` would have it:
    measured 1.7-2.7x a page's serialized size over six real conversations, so a
    byte budget must be sized against ACCOUNTED bytes rather than wire ones — the
    page cache's own constant says what happens when it is not. Shared immutable
    values count once within a page and conservatively again across entries.
    Framework class/schema objects are process-global, not retained here.
    """
    pending = [value]
    # Identity-keyed, and safe only because nothing here is freed mid-walk: the
    # page and its key are held by `pending`/the caller for the whole traversal,
    # so no id() can be recycled into a false "already counted".
    seen: set[int] = set()
    total = 0
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        total += sys.getsizeof(item)
        if total > limit:
            return None
        if isinstance(item, BaseModel):
            pending.extend(
                (
                    item.__dict__,
                    item.__pydantic_fields_set__,
                    item.__pydantic_extra__,
                    item.__pydantic_private__,
                )
            )
        elif is_dataclass(item) and not isinstance(item, type):
            pending.extend(vars(item).values())
        elif isinstance(item, dict):
            pending.extend(item)
            pending.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending.extend(item)
    return total


@dataclass(frozen=True)
class PageKey:
    """The identity of one page read: which file, and which window of it.

    ``directory`` is the session directory as a string (not a ``Path``, so the
    key is hashable and comparable across the two processes' spellings of the
    same directory). ``inode`` and ``size`` are the invalidation (see the module
    docstring); the three cursor fields make the key a page's identity rather
    than merely its file's.
    """

    directory: str
    inode: int
    size: int
    before_id: str | None
    through_id: str | None
    limit: int


def page_key(
    directory: str | Path,
    *,
    before_id: str | None = None,
    through_id: str | None = None,
    limit: int = 100,
) -> PageKey | None:
    """``directory``'s current page key, or ``None`` when its journal is absent.

    A missing journal is not an error here: the façade turns it into an ordinary
    uncached read, so the caller still gets the reader's own ``FileNotFoundError``
    rather than a second, differently-worded one from this module.
    """
    path = Path(directory) / TRANSCRIPT_FILENAME
    try:
        stat = path.stat()
    except OSError:
        return None
    return PageKey(
        directory=str(Path(directory)),
        inode=stat.st_ino,
        size=stat.st_size,
        before_id=before_id,
        through_id=through_id,
        limit=limit,
    )


class TranscriptPageCache:
    """A bounded LRU of decoded pages, keyed by :class:`PageKey`.

    Every failure mode here is ordinary: a page too large to admit is a MISS, an
    evicted key is a MISS, a cold cache is a MISS. Nothing on the read path
    raises because of this class, and nothing about a stale entry can be
    observed without a file whose identity is unchanged (THE RESIDUAL, in the
    module docstring, states which shapes reach that and why neither is
    reachable from this codebase's writers).
    """

    def __init__(
        self, *, entries: int = PAGE_CACHE_ENTRIES, byte_budget: int = PAGE_CACHE_BYTES
    ) -> None:
        self._entries: OrderedDict[PageKey, tuple[TranscriptPage, int]] = OrderedDict()
        self._accounted = 0
        self._entries_cap = entries
        self._byte_cap = byte_budget
        # Evidence counters, for the benchmark's tally and for a test that has to
        # prove a hit did no second read. They are read by the bench only and
        # never asserted on a clock.
        self.hits = 0
        self.misses = 0
        self.oversize = 0

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def accounted_bytes(self) -> int:
        return self._accounted

    def get(self, key: PageKey) -> TranscriptPage | None:
        cached = self._entries.get(key)
        if cached is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        # SHARED, not copied, and what makes that safe is the CONSUMERS rather
        # than any immutability: every consumer only reads — the desktop
        # envelope re-serializes each row with ``to_json()``, the TUI folds rows
        # into its own list of ``SubagentEntry``. Do not read more into it than
        # that. ``TranscriptPage`` is ``frozen=True`` so its ``entries`` tuple
        # cannot be reassigned, but ``TranscriptEntry`` is NOT frozen and its
        # ``payload`` is a plain dict, so this is a contract on callers, not a
        # type guarantee — and freezing the entry type would not buy it either,
        # because freezing forbids attribute assignment, not dict mutation. A
        # future caller that mutates a row must copy first;``_DisplayWindowCache``
        # copies for exactly that reason, over pydantic models rendering does
        # annotate. A page is therefore hundreds of kilobytes of JSON parsed once
        # per switch rather than once per request, which is the whole point.
        return cached[0]

    def put(self, key: PageKey, page: TranscriptPage) -> None:
        size = retained_bytes((key, page), self._byte_cap - _PAGE_CACHE_ALLOWANCE)
        if size is None:
            self.oversize += 1
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._accounted -= previous[1]
        while self._entries and (
            len(self._entries) >= self._entries_cap
            or self._accounted + size > self._byte_cap - _PAGE_CACHE_ALLOWANCE
        ):
            _, (_, removed) = self._entries.popitem(last=False)
            self._accounted -= removed
        self._entries[key] = (page, size)
        self._accounted += size

    def invalidate(self, directory: str | Path) -> None:
        """Drop every page of one session directory, whatever its identity.

        Not needed by the read path — a compaction or an append already moves the
        key — but a caller that knows a session is gone (a delete, a fork that
        emptied one) should not leave its pages resident until the LRU turns over.
        """
        prefix = str(Path(directory))
        for key in [k for k in self._entries if k.directory == prefix]:
            _, removed = self._entries.pop(key)
            self._accounted -= removed

    def reset(self) -> None:
        """Drop everything, counters included. For tests and for a benchmark cell."""
        self._entries.clear()
        self._accounted = 0
        self.hits = 0
        self.misses = 0
        self.oversize = 0


#: The process-wide cache. One per process is the coherent unit: the desktop server
#: is a single uvicorn process holding one ``DesktopSessions`` (cached on
#: ``app.state``), and the TUI is a DIFFERENT process with its own copy — which is
#: correct, because they hold different journals' worth of state and neither can
#: see the other's. A per-``DesktopSessions`` cache would miss the TUI entirely and
#: would be no more correct.
_PAGE_CACHE = TranscriptPageCache()

#: The single-flight map: ``PageKey -> (loop, task)``. Two ``snapshot()`` calls in
#: one switch do the identical read, and they do it on one event loop, in
#: ``asyncio.to_thread`` workers whose parses serialise against the loop through
#: the GIL — so the duplicate is not merely wasted work, it is loop contention on
#: the conversations that are already slow. The loop is carried per entry because
#: an entry left behind by a torn-down test loop must read as ABSENT: awaiting a
#: future bound to a dead loop raises ``RuntimeError``, and ``tests/unit/tui/*``
#: create and tear down loops inside one process.
_FLIGHTS: dict[PageKey, tuple[asyncio.AbstractEventLoop, asyncio.Task[TranscriptPage]]] = {}


def page_cache() -> TranscriptPageCache:
    """The process-wide cache — for a caller that must observe or clear it."""
    return _PAGE_CACHE


def reset_page_cache() -> None:
    """Empty the process-wide cache and forget every in-flight read."""
    _PAGE_CACHE.reset()
    _FLIGHTS.clear()


def _forget(key: PageKey, settled: asyncio.Task[TranscriptPage]) -> None:
    # Identity-checked, so a later read that replaced this entry on a fresh loop
    # does not lose its own flight to this one's completion.
    current = _FLIGHTS.get(key)
    if current is not None and current[1] is settled:
        del _FLIGHTS[key]


async def _read_and_publish(
    key: PageKey | None,
    directory: str,
    before_id: str | None,
    through_id: str | None,
    limit: int,
) -> TranscriptPage:
    """The one read behind a key, published only if the file did not move.

    THE RE-STAT IS THE PUBLICATION GUARD. A read that spans a compaction or an
    append must not become the cached answer: the page it decoded is a real
    answer to the request (today's behaviour, and not this change's to fix), but
    it is not necessarily the answer for the identity it was keyed under. So the
    façade takes the stat before the read and this takes it again after, and only
    an unchanged ``(st_ino, st_size)`` is cached. This runs in the task's own
    frame, on the loop, after ``to_thread`` returns — never inside the closure.
    """
    page = await asyncio.to_thread(
        read_transcript_page,
        directory,
        before_id=before_id,
        through_id=through_id,
        limit=limit,
    )
    if (
        key is not None
        and page_key(directory, before_id=before_id, through_id=through_id, limit=limit) == key
    ):
        _PAGE_CACHE.put(key, page)
    return page


async def load_transcript_page(
    directory: str | Path,
    *,
    before_id: str | None = None,
    through_id: str | None = None,
    limit: int = 100,
) -> TranscriptPage:
    """A cached, single-flighted :func:`read_transcript_page`.

    Same contract, same three special returns, same three precondition errors —
    the preconditions are validated by the reader's own validator so a caller
    cannot tell which entry point answered it. What changes is only who pays for
    the read: the first caller for a key pays it, every other caller for the same
    key shares that one read, and a caller for a key this process has already
    read pays nothing at all.

    The returned page is SHARED with the cache and with any other caller for the
    same key: read the rows, do not mutate them (see :meth:`TranscriptPageCache.get`).
    """
    validate_page_request(before_id, through_id, limit)
    directory = str(directory)
    key = page_key(directory, before_id=before_id, through_id=through_id, limit=limit)
    if key is not None:
        cached = _PAGE_CACHE.get(key)
        if cached is not None:
            return cached
    loop = asyncio.get_running_loop()
    flight = _FLIGHTS.get(key) if key is not None else None
    if flight is not None and flight[0] is loop:
        # A FOLLOWER. ``shield`` because a follower's cancellation must not cancel
        # the read the other waiter — or the leader's own request — depends on;
        # the task is not the follower's to end.
        return await asyncio.shield(flight[1])
    task = asyncio.create_task(_read_and_publish(key, directory, before_id, through_id, limit))
    if key is not None:
        _FLIGHTS[key] = (loop, task)
        # A done-callback rather than a ``finally`` around every awaiter: it fires
        # for cancellation and for failure alike, so a cancelled leader cannot
        # orphan the entry, and a leader whose own await is cancelled still
        # publishes for whoever is left.
        task.add_done_callback(lambda settled: _forget(key, settled))
    return await asyncio.shield(task)
