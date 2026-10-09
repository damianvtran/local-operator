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
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from local_operator.session.transcript import TRANSCRIPT_FILENAME
from local_operator.session.transcript_index import probe_index, start_refresh

logger = logging.getLogger(__name__)

#: How many journals one warm covers. 32 is #2102's figure and the reason is its
#: own measurements: ~277 MB of journals and ~25 MB of cache, which is a working
#: set (the sessions a user moves between) rather than a population backfill.
PREWARM_JOURNALS = 32

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


def skip_reason(
    root: str | Path, *, free: int | None = None, load: float | None = None
) -> str | None:
    """Why the warm must not run now, or ``None`` when it may.

    Injected values keep the decision testable without a full disk or a busy
    host; the defaults are the measurements above.
    """
    measured_free = free_bytes(root) if free is None else free
    if measured_free is not None and measured_free < PREWARM_MIN_FREE_BYTES:
        return f"disk: {measured_free} bytes free is under {PREWARM_MIN_FREE_BYTES}"
    measured_load = load_per_cpu() if load is None else load
    if measured_load is not None and measured_load >= PREWARM_MAX_LOAD_PER_CPU:
        return f"load: {measured_load:.2f} per CPU is at or above {PREWARM_MAX_LOAD_PER_CPU}"
    return None


def sessions_needing_index(root: str | Path, *, limit: int = PREWARM_JOURNALS) -> list[str]:
    """The ``limit`` most recently modified journals whose index is not current.

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
        try:
            if probe_index(root, session_id).state == "ready":
                continue
        except Exception:  # noqa: BLE001 — a session we cannot read is not a candidate
            logger.debug("index prewarm: probe failed for %s", session_id, exc_info=True)
            continue
        candidates.append(session_id)
    return candidates


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
    candidates = await asyncio.to_thread(sessions_needing_index, root, limit=limit)
    started = 0
    for session_id in candidates:
        later = skip_reason(root)
        if later is not None:
            logger.debug("index prewarm dropped after %d journals (%s)", started, later)
            break
        try:
            task, was_started = start_refresh(root, session_id)
            await task
        except Exception:  # noqa: BLE001 — one unreadable journal must not end the warm
            logger.debug("index prewarm failed for %s", session_id, exc_info=True)
            continue
        started += 1 if was_started else 0
        await asyncio.sleep(0)
    logger.debug("index prewarm started %d refresh(es)", started)
    return started


def start_index_prewarm(root: str | Path) -> asyncio.Task[Any] | None:
    """Spawn :func:`warm_index_cache` off the hot path; ``None`` when skipped.

    NEVER RAISES and NEVER BLOCKS: the caller is the daemon's ``lifespan``, where
    an escaping exception fails startup and where anything awaited is startup
    latency. The guard is consulted HERE as well as inside the task so a machine
    that cannot afford the warm does not even get a task, and the returned task is
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
