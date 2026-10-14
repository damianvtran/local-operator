"""Warm the transcript index cache for the journals a user is most likely to open.

WHY THIS EXISTS. The open frame answers its first read from the transcript index,
and an index that is not current yet answers ``building`` — the facts (the run
bars) then arrive one frame later, which is a paint the operator sees arrive.
Measured on this machine: **0 of 13,812 journals carry a current index cache**, so
in practice every conversation's first open pays the scan: 28-32 ms on a 5.9 MB
journal, 169 ms at 35 MB, 559 ms at 118 MB (the numbers #2102's body carries), and
the open frame's deadline is 120 ms — so anything above roughly 20 MB answers
``building`` on its first open and settles later.

Nothing is broken by that; it is simply work that can be done while nobody is
waiting, because the journals a user opens next are almost always the ones they
touched last. So the daemon warms the K most recently modified journals that have
no current cache, once, at startup, off the hot path.

WHAT IT IS NOT. Not a whole-population backfill: the cache for all 13,812
journals would be roughly 1.5 GB on a disk that is 96% full, bought for the 0.5%
of sessions anyone will open, and it would take minutes of scanning to produce.
K≈32 is the size that covers a working set: ~277 MB of journals scanned, ~25 MB
of cache.

THREE GUARDS, each of which answers "not now" rather than "partially":

* **Disk.** Below :data:`PREWARM_MIN_FREE_BYTES` the warm is skipped entirely.
  This is optional work — a cache the machine cannot afford to keep — and the
  operator's own store is the thing that must keep working. Measured with
  ``shutil.disk_usage`` (a read; never a probe write, the rule
  ``session.store_failures`` states for its own check).
* **Load.** The warm is DROPPED when the host is already busy enough that nobody
  is waiting on our latency, and re-checked before every journal, so a machine
  that gets busy mid-warm loses the REST of the queue rather than finishing it.
  The queue is deliberately the first thing to go: it is the only work here with
  no one waiting on it.
* **The event loop.** Selection and scanning both happen in threads or in a task
  that yields between journals, and :func:`start_index_prewarm` returns as soon as
  the task exists. The daemon's startup is the thing an attach waits on, so
  nothing in this module may run inside it — the same rule the tokenizer and
  bytecode warms follow in ``server.app.lifespan``.

EACH JOURNAL'S SCAN IS THE EXISTING ONE. ``transcript_index.start_refresh`` is
single-flight per session, so a user who opens a journal while its warm is in
flight JOINS that scan (``started=False``) instead of paying a second one — the
warm can only ever remove work, never add it to the open path.

IT ALSO PREHEATS THE COLD REPLAY, for the symptom those two caches do not touch:
the FIRST read of a large, not-yet-cached conversation pays a whole-suffix parse
(323-480 ms measured on a 6.7 MB journal — the operator's "cold switch into a big
conversation"). That read needs NO runtime (the cold facade is a viewer over the
journal), and its result lands in the process-local replay cache
(:mod:`local_operator.session.replay_cache`), which is EMPTY at every core start —
so the journals a user opens first after a restart are exactly the ones paying the
parse. The replay pass therefore runs beside the index pass, over the same newest
journals, through the SAME ``build_cold_replay``/``load_cold_replay`` the reader
uses, so a later open is a cache hit and an open that races the preheat JOINS it
rather than double-parsing. It is bounded by the replay cache's own entry bound
(:data:`PREWARM_REPLAYS`) and guarded by the same disk/load checks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
from local_operator.session.replay_cache import (
    REPLAY_CACHE_ENTRIES,
    build_cold_replay,
    cached_replay,
    load_cold_replay,
    replay_key,
)
from local_operator.session.spend import SESSION_SPEND_CUSTOM_TYPE
from local_operator.session.tail_anchor import (
    build_anchor,
    read_anchor,
    validate_anchor,
    write_anchor,
)
from local_operator.session.transcript import TRANSCRIPT_FILENAME
from local_operator.session.transcript_index import probe_index, start_refresh

if TYPE_CHECKING:
    # The return annotation of ``_cold_replay_key`` names it, so it must be BOUND for
    # pyflakes/pyright — a quoted annotation alone still trips F821 when the name is
    # never imported. ``TYPE_CHECKING``-only keeps the import off the runtime path:
    # this module is built at core startup and the name is used in annotations alone.
    from local_operator.session.replay_cache import ReplayKey

logger = logging.getLogger(__name__)

#: How many journals one warm covers. 32 is #2102's figure and the reason is its
#: own measurements: ~277 MB of journals and ~25 MB of cache, which is a working
#: set (the sessions a user moves between) rather than a population backfill.
PREWARM_JOURNALS = 32

#: How many journals one REPLAY preheat materializes. The SMALLER of the index
#: queue's working set and the replay cache's own entry bound, because a replay
#: entry is the expensive one: it pins a whole conversation's replayed messages
#: (the cache refuses one over ``REPLAY_CACHE_BYTES``). Materializing more than the
#: cache can hold would evict the early entries to make room for the late ones —
#: the queue spending its pass thrashing the LRU instead of warming a working set.
#: Reusing the cache's own bound keeps the preheat and the cache ONE policy.
PREWARM_REPLAYS = min(PREWARM_JOURNALS, REPLAY_CACHE_ENTRIES)

#: Free-space floor for the warm, in bytes (2 GiB). Deliberately much higher than
#: ``store_failures.FULL_VOLUME_FLOOR_BYTES``, which answers "can the store still
#: write at all": a cache is optional, and the margin that matters is the one
#: left for the journals, not the one left for their index.
PREWARM_MIN_FREE_BYTES = 2 * 1024**3

#: Load per CPU at or above which the warm is dropped. A host with every core
#: already queued is a host where the scan would compete with work someone asked
#: for, and this module's work is the only work here nobody is waiting on. The
#: threshold is a RATIO rather than an absolute number so the same rule reads the
#: same on a 4-core laptop and a 14-core build host.
PREWARM_MAX_LOAD_PER_CPU = 1.0

#: Journals whose anchor question was SETTLED in this process, as
#: ``{session directory: (inode, size)}``.
#:
#: WHY THIS EXISTS (review round 3, F11). ``build_anchor`` returns ``None`` for a
#: journal that already carries the checkpoint row — there is nothing to prove, so no
#: sidecar is written — which means a size-only "is the anchor current?" test answers
#: NO for that journal forever. Under the degraded pass (one journal per pass) the
#: single slot then goes to the same newest, already-warm journal every time, and the
#: journal the anchor exists for is never reached: measured on a real store as
#: ``pass 1/2/3: candidates=['new', 'old'] started=1 anchor(new)=False anchor(old)=False``.
#: Recording the settlement keeps the queue moving; the ``(inode, size)`` pair
#: invalidates it when the journal GROWS, is replaced, or is trimmed, so a row that
#: appears later (or a torn row that completes) is examined again. The limit, stated
#: rather than implied: a same-size IN-PLACE rewrite changes neither number and keeps
#: the settlement — the write path appends or replaces, never rewrites in place, so
#: this is a limit of the instrument rather than a case a writer reaches.
_NO_ANCHOR_NEEDED: dict[str, tuple[int, int]] = {}

#: Single-flight guard for the anchor write, keyed by session directory.
#:
#: WHY (review round 3, F15): the write is reachable from the startup queue AND from
#: a warm request, and its expensive half is a whole-file scan for a checkpointless
#: journal. Two concurrent callers would both scan and both write — the scan is the
#: thing the record exists to make unnecessary, so paying it twice is exactly the
#: waste this module is for.
_ANCHOR_IN_FLIGHT: set[str] = set()
_ANCHOR_LOCK = threading.Lock()

#: Strong references to the warm tasks, for the reason
#: ``transcript_index.start_refresh`` documents for its own: a bare
#: ``asyncio.create_task`` holds only a weak referent and can be collected
#: mid-flight.
_TASKS: set[asyncio.Task[Any]] = set()


def free_bytes(root: str | Path | None = None) -> int | None:
    """Free bytes on the volume holding ``root``, or ``None`` if unmeasurable.

    ``None`` is not zero: a root this process cannot stat says nothing about the
    disk, and the caller decides which way to fail (it treats it as "free
    enough", because the guard exists to protect a disk we can see).
    """
    try:
        usage = shutil.disk_usage(str(root) if root is not None else str(Path.home()))
    except OSError:
        return None
    return usage.free


def load_per_cpu() -> float | None:
    """One-minute load average divided by the CPU count, or ``None``.

    ``os.getloadavg`` is absent (and raises) on some platforms; an unmeasurable
    load means "do not drop the work", the same direction the disk guard fails.
    """
    try:
        load1, _load5, _load15 = os.getloadavg()
    except (OSError, AttributeError, NotImplementedError):
        return None
    cpus = os.cpu_count() or 1
    return load1 / cpus


def skip_reason(root: str | Path, *, free: int | None = None) -> str | None:
    """Why the warm must not run AT ALL now, or ``None`` when it may.

    DISK PRESSURE ONLY, and that is the degradation: this fleet runs at 1-8 load
    per CPU through the working day, so a load clause here refused the queue exactly
    when a store needed it and a machine that is never quiet warmed nothing (QA round
    2 measured the refusal — ``load: 1.28 per CPU ...``, 0 caches). Disk pressure is
    different in kind: it is the one condition under which the warm must not write.
    Load now PACES the queue instead — see :func:`pace_reason`.
    """
    measured_free = free_bytes(root) if free is None else free
    if measured_free is not None and measured_free < PREWARM_MIN_FREE_BYTES:
        return f"disk: {measured_free} bytes free is under {PREWARM_MIN_FREE_BYTES}"
    return None


def pace_reason(*, load: float | None = None) -> str | None:
    """Why the queue should STOP here — not why it must not start.

    Re-checked before every journal AFTER the first, so a host that becomes busy
    mid-warm keeps what it warmed and drops the rest, and a host that is already busy
    does ONE journal per pass rather than none.
    """
    measured_load = load_per_cpu() if load is None else load
    if measured_load is not None and measured_load >= PREWARM_MAX_LOAD_PER_CPU:
        return f"load: {measured_load:.2f} per CPU is at or above {PREWARM_MAX_LOAD_PER_CPU}"
    return None


def sessions_needing_warm(root: str | Path, *, limit: int = PREWARM_JOURNALS) -> list[str]:
    """The ``limit`` most recently modified journals that owe this job something.

    TWO REASONS A JOURNAL IS A CANDIDATE, and either one is enough (review round 1,
    F4): its transcript index is not current, or its TAIL ANCHOR is not current
    (``session.tail_anchor`` — the record that lets a checkpointless journal be read
    without scanning to BOF). The index probe used to decide this list alone, so a
    journal with a warm index was skipped by the refresh and silently skipped by the
    anchor pass that shares the loop — on a store whose indexes were warmed by an
    earlier run, that was every journal, and the anchors were never written at all.

    SYNCHRONOUS and it stats every journal, so its callers run it in a thread
    (``asyncio.to_thread``): 13,812 stats plus the probes for the newest few is
    tens of milliseconds of syscalls, which is a stall the event loop this daemon
    serves HTTP from does not owe anyone.

    Walks newest-first and stops at ``limit`` rather than probing the whole store:
    on this machine 0 of 13,812 journals have a current cache, so the first
    ``limit`` journals examined are the answer, and a store where that is not true
    (a warm one) is where the early exit saves the most.

    A journal with no ``transcript.jsonl`` is not a candidate, and neither is a
    directory this process cannot read: both are skipped rather than reported.
    """
    sessions_dir = Path(root) / "sessions"
    try:
        entries = list(sessions_dir.iterdir())
    except OSError:
        return []
    journals: list[tuple[float, str]] = []
    for entry in entries:
        try:
            stat = (entry / TRANSCRIPT_FILENAME).stat()
        except OSError:
            continue
        journals.append((stat.st_mtime, entry.name))
    journals.sort(reverse=True)
    candidates: list[str] = []
    for _mtime, session_id in journals:
        if len(candidates) >= limit:
            break
        needs = False
        try:
            needs = probe_index(root, session_id).state != "ready"
        except Exception:  # noqa: BLE001 — fall through to the anchor test
            logger.debug("index prewarm: probe failed for %s", session_id, exc_info=True)
        if not needs and not _anchor_current(sessions_dir / session_id):
            needs = True
        if needs:
            candidates.append(session_id)
    return candidates


def sessions_needing_replay(root: str | Path, *, limit: int = PREWARM_REPLAYS) -> list[str]:
    """The ``limit`` newest journals whose cold replay THIS PROCESS has not cached.

    A DIFFERENT QUESTION FROM :func:`sessions_needing_warm`, and it must be asked
    separately rather than by reusing that selection. That one asks whether a
    journal's ON-DISK caches are current, and a store warmed once answers "yes"
    for the journals a user is most likely to open — the exact set the replay pass
    exists for. The replay cache is PROCESS-LOCAL and empty at every core start
    (see :mod:`local_operator.session.replay_cache`), so its question is "is this
    key in THIS process's cache", and the newest journals answer it by the same
    mtime order the index pass uses.

    The key is built exactly as the reader builds it for a cold open
    (``want_checkpoint`` -- a checkpoint is REQUIRED, the spend record
    OPPORTUNISTIC -- no cursor, non-strict cut), so a materialized entry is the
    one that open will look up. A journal whose ``replay`` is already cached is
    skipped: re-materializing it would be a second parse for a hit, and
    ``load_cold_replay`` would answer from memory anyway.

    SYNCHRONOUS and it stats/keys every journal, so its caller runs it in a thread
    (``asyncio.to_thread``), for the reason :func:`sessions_needing_warm` states.
    """
    sessions_dir = Path(root) / "sessions"
    try:
        entries = list(sessions_dir.iterdir())
    except OSError:
        return []
    journals: list[tuple[float, str]] = []
    for entry in entries:
        try:
            stat = (entry / TRANSCRIPT_FILENAME).stat()
        except OSError:
            continue
        journals.append((stat.st_mtime, entry.name))
    journals.sort(reverse=True)
    candidates: list[str] = []
    for _mtime, session_id in journals:
        if len(candidates) >= limit:
            break
        key = _cold_replay_key(sessions_dir / session_id)
        if key is not None and cached_replay(key) is None:
            candidates.append(session_id)
    return candidates


def _cold_replay_key(directory: Path) -> "ReplayKey | None":
    """The key a cold open of ``directory`` looks up, or ``None`` when absent.

    ONE SPELLING, shared by the selection above and the preheat below, so the two
    cannot disagree about which read they are preparing -- and so a later edit
    cannot make the pass preheat a key no reader asks for.
    """
    return replay_key(
        directory,
        through_id=None,
        strict_cut=False,
        checkpoint_types=(FRONTEND_CHECKPOINT_CUSTOM_TYPE,),
        opportunistic_types=(SESSION_SPEND_CUSTOM_TYPE,),
    )


def start_session_warm(root: str | Path, session_id: str) -> bool:
    """Warm ONE session's index now — the per-session path — and return at once.

    WHY IT EXISTS. The queue above covers the newest journals once, at core start. A
    conversation opened minutes later has no such pass and pays the cold index scan
    on whoever opens it. A client that knows which session it is about to open can
    ask for that work now, on a background task.

    THE ANCHOR WRITE RIDES ALONG, on its own task beside the refresh, and it is the
    half that is NEW work on every open: the queue above warms the newest journals at
    core start, while the anchor a given open needs is the one for the session being
    opened. It is deferred to that task rather than skipped — the write is a
    whole-file scan for a checkpointless journal and must not run on the request's
    time.

    NEVER RAISES and NEVER BLOCKS: the caller is an HTTP handler. The scan is the
    same single-flight task ``start_refresh`` hands out, so a second request — or the
    startup queue — joins it rather than repeating it. Returns whether work was
    scheduled; a joiner, or a machine under disk pressure, gets ``False`` and no
    error.
    """
    reason = skip_reason(root)
    if reason is not None:
        logger.debug("session warm skipped (%s)", reason)
        return False
    try:
        # The module-level import, deliberately: a local one would shadow the name
        # every other call in this file goes through, and with it the seam a test
        # (or a caller) uses to see that the refresh was scheduled.
        start_refresh(root, session_id)
    except Exception:  # noqa: BLE001 — a hint must never be the failure of a request
        logger.debug("session warm could not be scheduled", exc_info=True)
        return False
    # THE ANCHOR WRITE RIDES ALONG, on its own task: it is a whole-file scan for a
    # checkpointless journal and must not run on the request's time any more than the
    # index scan does. Held in ``_TASKS`` so the loop cannot collect it mid-flight.
    anchor_task = asyncio.get_running_loop().create_task(_write_anchor(root, session_id))
    _TASKS.add(anchor_task)
    anchor_task.add_done_callback(_TASKS.discard)
    return True


async def _write_anchor(root: str | Path, session_id: str) -> None:
    """Write one session's tail anchor, with the queue's own exception contract.

    WHY IT IS WRAPPED (review round 4, F19): this runs on a task nobody retrieves, so
    a raise inside it becomes an unretrieved-task exception rather than a line in the
    log — and the queue's call of the same writer already states the intent, that "an
    unwritable anchor is a slower read, not a failure". Two anchors landing at once is
    the single-flight guard's business (``_ANCHOR_IN_FLIGHT``): the second caller
    returns without scanning. A named helper rather than a closure so the contract is
    callable in a test.
    """
    try:
        await asyncio.to_thread(write_tail_anchor, root, session_id)
    except Exception:  # noqa: BLE001 — an unwritable anchor is a slower read, not a failure
        logger.debug("tail anchor not written for %s", session_id, exc_info=True)


def _anchor_current(directory: Path) -> bool:
    """Whether this journal's tail anchor already describes its current version.

    A stat and a small read, deliberately — NOT
    ``tail_anchor.validate_anchor``, which scans everything appended since the
    record. That scan is the right price for a reader about to skip a whole-file
    walk and the wrong price for a selector that runs over the newest journals on
    every core start: a journal the selector wrongly calls current is simply
    re-checked by ``write_tail_anchor``, which does validate before writing.
    """
    from local_operator.session.tail_anchor import read_anchor

    try:
        stat = (directory / TRANSCRIPT_FILENAME).stat()
    except OSError:
        return False
    settled = _NO_ANCHOR_NEEDED.get(str(directory))
    if settled == (stat.st_ino, stat.st_size):
        # Asked and answered in this process: this journal needs no sidecar (see
        # ``_NO_ANCHOR_NEEDED``). Without this the degraded pass spins on it forever.
        return True
    anchor = read_anchor(directory)
    if anchor is None:
        return False
    return anchor.inode == stat.st_ino and anchor.size == stat.st_size


async def warm_index_cache(root: str | Path, *, limit: int = PREWARM_JOURNALS) -> int:
    """Warm up to ``limit`` journals, one at a time; returns how many were started.

    SEQUENTIAL ON PURPOSE. The scan is CPU-bound in this process and the daemon is
    also serving HTTP; one at a time is the rate that keeps the loop responsive,
    and ``await asyncio.sleep(0)`` between journals is what gives the loop its turn
    even when a scan returns without suspending.

    The guard is re-checked before EVERY journal: a host that becomes busy
    mid-warm keeps whatever was already warm and drops the rest, which is the
    right way round — the queue has no one waiting on it.
    """
    reason = skip_reason(root)
    if reason is not None:
        logger.debug("index prewarm skipped (%s)", reason)
        return 0
    candidates = await asyncio.to_thread(sessions_needing_warm, root, limit=limit)
    # DEGRADED MODE: a host that is already busy gets ONE journal this pass instead
    # of a refused queue, and the pass after it — the next core start, or a request
    # that asks for its own session (:func:`start_session_warm`) — can do one more.
    already_busy = pace_reason()
    if already_busy is not None:
        candidates = candidates[:1]
        logger.debug("index prewarm degraded (%s): one journal this pass", already_busy)
    started = 0
    for position, session_id in enumerate(candidates):
        # AFTER the first, not before it: the degraded pass is allowed the one
        # journal it came for.
        later = pace_reason() if position else None
        if later is not None:
            logger.debug("index prewarm dropped after %d journals (%s)", started, later)
            break
        try:
            await asyncio.to_thread(write_tail_anchor, root, session_id)
        except Exception:  # noqa: BLE001 — an unwritable anchor is a slower read, not a failure
            logger.debug("tail anchor not written for %s", session_id, exc_info=True)
        try:
            task, was_started = start_refresh(root, session_id)
            await task
        except Exception:  # noqa: BLE001 — one unreadable journal must not end the warm
            logger.debug("index prewarm failed for %s", session_id, exc_info=True)
            await asyncio.sleep(0)
            continue
        started += 1 if was_started else 0
        await asyncio.sleep(0)
    logger.debug("index prewarm started %d refresh(es)", started)
    # THE REPLAY PASS RAINS LAST and on purpose: the index scan is bounded and cheap
    # (single-digit to tens of ms) while the replay parse is the heavy one, so the
    # cheap work lands first and a host that turns busy mid-startup still got its
    # indexes even if the replays are dropped. Neither pass is ever on a request's
    # time; both reuse the existing single-flights.
    try:
        await warm_replay_cache(root, limit=min(limit, PREWARM_REPLAYS))
    except Exception:  # noqa: BLE001 — the preheat is an optimisation, never the answer
        logger.debug("replay prewarm unavailable", exc_info=True)
    return started


async def warm_replay_cache(root: str | Path, *, limit: int = PREWARM_REPLAYS) -> int:
    """Materialize up to ``limit`` journals' cold replays; returns how many landed.

    THE COMPANION PASS TO :func:`warm_index_cache`, over the SAME newest journals,
    for the cost that pass does not touch: the first read of a large conversation
    parses its journal end to end (323-480 ms on 6.7 MB), and the parser only ever
    reads from the newest compaction forward — so the replay is a small window that
    a later open either pays once or finds already in
    :mod:`local_operator.session.replay_cache`. The cache is process-local and empty
    at every core start, so a warm here is what makes the first click after a
    restart a hit rather than a parse.

    NO RUNTIME IS SPAWNED AND NONE IS NEEDED. The cold replay is a read of durable
    rows (:func:`replay_cache.build_cold_replay`), the same one the desktop's cold
    facade pays on its first read; nothing here dials an owner. It writes exactly
    what a reader would have written, under the same key, so a subsequent open is a
    hit.

    SEQUENTIAL and BOUNDED, for :func:`warm_index_cache`'s reasons: one parse at a
    time keeps the loop responsive while the daemon serves HTTP, and
    ``await asyncio.sleep(0)`` between journals gives the loop its turn. The DISK
    guard is asked ONCE, at the top (:func:`warm_index_cache` asks it the same way,
    through the same :func:`skip_reason`): disk pressure is the one condition under
    which the pass must not write at all, so a full volume stops the whole pass
    before the first journal. The LOAD guard is re-checked between journals, so a
    host that turns busy mid-pass keeps whatever is warm and drops the rest.

    A FAILED JOURNAL NEVER FAILS THE PASS. One unreadable or oversize journal is
    logged and skipped: the queue has no one waiting on it and the reader that later
    opens that session pays its own parse exactly as it does today.
    """
    reason = skip_reason(root)
    if reason is not None:
        logger.debug("replay prewarm skipped (%s)", reason)
        return 0
    candidates = await asyncio.to_thread(sessions_needing_replay, root, limit=limit)
    already_busy = pace_reason()
    if already_busy is not None:
        candidates = candidates[:1]
        logger.debug("replay prewarm degraded (%s): one journal this pass", already_busy)
    sessions_dir = Path(root) / "sessions"
    started = 0
    # OLDEST-FIRST, so the NEWEST journal is materialized LAST and therefore
    # survives the cache's LRU. ``sessions_needing_replay`` returns newest-first
    # (the order a user is most likely to open); inserting in that order would make
    # the newest the OLDEST entry the byte bound evicts first — measured: 8
    # materializations against a 48 MiB bound kept the OLDEST 3 and evicted the
    # newest, so the open that mattered still paid its parse. The selection order is
    # what the reader wants; the insertion order is what the bound wants, and they
    # are deliberately opposite.
    for position, session_id in enumerate(reversed(candidates)):
        later = pace_reason() if position else None
        if later is not None:
            logger.debug("replay prewarm dropped after %d journals (%s)", started, later)
            break
        directory = sessions_dir / session_id
        key = await asyncio.to_thread(_cold_replay_key, directory)
        if key is None:
            continue
        # ``load_cold_replay`` IS the reader's own single-flight: a user who opens
        # this session while its preheat is in flight JOINS this task instead of
        # starting a second parse, and a key already materialized is answered from
        # memory (the selection above skips those, so in practice this starts the
        # read). The parse itself is threaded, like the reader's.
        try:
            replay = await load_cold_replay(
                key,
                lambda directory=directory, key=key: asyncio.to_thread(
                    build_cold_replay,
                    directory,
                    key=key,
                    want_checkpoint=True,
                    through_id=None,
                    strict_cut=False,
                ),
            )
        except Exception:  # noqa: BLE001 — one unreadable journal must not end the preheat
            logger.debug("replay prewarm failed for %s", session_id, exc_info=True)
            await asyncio.sleep(0)
            continue
        # COUNT ONLY WHAT THE CACHE ACTUALLY HOLDS. ``publish_replay`` REFUSES an
        # over-budget entry by returning it with ``retained == 0`` rather than
        # storing it, so the read still answers but nothing landed — counting that
        # as a materialization would over-report a cache the bound kept empty, on
        # exactly the large journals the number is watched for. A stored entry (fresh
        # or the rare already-cached hit the selection did not skip) carries its
        # retained size, which is why the test is ``> 0`` and not "no exception".
        if replay.retained > 0:
            started += 1
        await asyncio.sleep(0)
    logger.debug("replay prewarm materialized %d replay(s)", started)
    return started


def write_tail_anchor(root: str | Path, session_id: str) -> bool:
    """Record that this journal's newest checkpoint row is ABSENT — the cold-read hint.

    Idempotent and cheap when the record already holds: the existing sidecar is
    validated first (one stat, plus a scan of anything appended since), so a
    journal warmed twice in one day costs one stat the second time. Returns
    whether a NEW record was written.
    """
    directory = Path(root) / "sessions" / session_id
    if validate_anchor(directory, read_anchor(directory)) is not None:
        return False
    # SINGLE-FLIGHT (F15): the expensive half is ``build_anchor``'s whole-file scan
    # for a checkpointless journal, and this writer is reachable from both the
    # startup queue and a warm request. The second caller leaves immediately; the
    # first one's result is the answer either way.
    key = str(directory)
    with _ANCHOR_LOCK:
        if key in _ANCHOR_IN_FLIGHT:
            return False
        _ANCHOR_IN_FLIGHT.add(key)
    try:
        anchor = build_anchor(directory)
        if anchor is None:
            # SETTLE IT (F11): ``build_anchor`` returns None for a journal that
            # already carries the checkpoint row (nothing to prove) or cannot be
            # described at all. Either way there is nothing to write, and recording
            # which keeps the degraded pass from spending every pass on this journal.
            try:
                stat = (directory / TRANSCRIPT_FILENAME).stat()
            except OSError:
                return False
            _NO_ANCHOR_NEEDED[key] = (stat.st_ino, stat.st_size)
            return False
        return write_anchor(directory, anchor)
    finally:
        with _ANCHOR_LOCK:
            _ANCHOR_IN_FLIGHT.discard(key)


def start_index_prewarm(root: str | Path) -> asyncio.Task[Any] | None:
    """Spawn :func:`warm_index_cache` off the hot path; ``None`` when skipped.

    NEVER RAISES and NEVER BLOCKS: the caller is the daemon's ``lifespan``, where
    an escaping exception fails startup and where anything awaited is startup
    latency. The DISK guard is consulted HERE as well as inside the task so a
    machine that cannot afford the writes does not even get a task; load pressure
    does not suppress the task, it shortens the queue inside it. The returned task is
    held in :data:`_TASKS` until it settles.
    """
    try:
        reason = skip_reason(root)
        if reason is not None:
            logger.debug("index prewarm not started (%s)", reason)
            return None
        task = asyncio.get_running_loop().create_task(warm_index_cache(root))
    except Exception:  # noqa: BLE001 — a warm-up must never be the failure
        logger.debug("index prewarm unavailable at startup", exc_info=True)
        return None
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return task
