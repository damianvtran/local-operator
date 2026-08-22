"""The non-blocking recorder: turns a call into a queued sample, off the loop.

This is the one object the rest of the harness touches. A session's stream
wrapper builds a :class:`CallSnapshot` on the event loop (cheap, see
``model.snapshot_component_chars``) and hands it to :func:`record_call`; that
call does a single ``queue.put_nowait`` and returns. A background daemon thread
drains the queue in batches and writes them to the shared SQLite store.

Why a thread and not an asyncio task. The write is blocking SQLite I/O, and the
sessions that produce samples run on the event loop that must never block on
disk. A daemon thread also survives across the many short-lived event loops a
process opens (each subagent run, each reload) without being re-created, and it
is shared by every session in the process, so N sessions still write through
one queue and one connection.

Why best-effort with a bounded queue. Accuracy matters, but never at the cost
of a stalled session. The queue is large enough that a normal burst never
fills it, and if it somehow does the sample is DROPPED (counted, logged once)
rather than applying back-pressure to a provider call. On a healthy machine the
drop count stays zero; when it does not, the log says analytics is losing
samples rather than the session mysteriously slowing down.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from local_operator.analytics.model import CallSnapshot
from local_operator.analytics.store import AnalyticsStore

logger = logging.getLogger("local_operator.analytics.recorder")


class _NameTask:
    """A session-name upsert queued for the writer thread.

    Routed through the SAME queue and connection as call samples rather than a
    second thread. Two threads opening their first connection to a
    freshly-created database at the same instant is a real race — observed to
    leave the writer's connection unable to see its own commits — so all writes
    funnel through one writer. Naming is infrequent, so sharing the queue costs
    nothing and removes the race by construction.
    """

    __slots__ = ("session_id", "name")

    def __init__(self, session_id: str, name: str) -> None:
        self.session_id = session_id
        self.name = name


#: Upper bound on queued-but-unwritten samples. A provider call takes seconds
#: and a batch write takes milliseconds, so this only fills if the disk is
#: wedged — at which point dropping is the correct behaviour. Sized for a
#: worst-case burst of many parallel sessions all ending turns at once.
_QUEUE_MAXSIZE = 4096

#: How long the writer waits to accumulate a batch before flushing. Short
#: enough that a report opened right after a turn sees it; long enough that a
#: tool loop's rapid calls coalesce into one transaction.
_FLUSH_INTERVAL_S = 0.5

#: Prune the retention window at most this often (seconds). Pruning is a DELETE
#: over an indexed column — cheap — but there is no reason to run it on every
#: flush; once an hour keeps the ledger bounded without touching the hot path.
_PRUNE_INTERVAL_S = 3600.0


class AnalyticsRecorder:
    """Owns the queue, the writer thread, and the store.

    One instance per process (see :func:`get_recorder`). Construction does NOT
    start the thread or open the database — the first :meth:`record` does, so a
    process that never makes a provider call pays nothing.
    """

    def __init__(self, store: AnalyticsStore | None = None) -> None:
        self._store = store if store is not None else AnalyticsStore()
        self._queue: "queue.Queue[CallSnapshot | _NameTask | None]" = queue.Queue(
            maxsize=_QUEUE_MAXSIZE
        )
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._dropped = 0
        self._last_prune = 0.0
        self._closed = False
        #: Monotonic counters used ONLY by ``flush_for_test`` as a commit
        #: barrier: ``_enqueued`` counts accepted samples, ``_committed``
        #: counts samples the writer has actually persisted. Real sessions
        #: never read these — they are how a test waits for a durable write
        #: without a fixed sleep that races the writer thread.
        self._enqueued = 0
        self._committed = 0

    # -- lifecycle -----------------------------------------------------------
    def _ensure_thread(self) -> None:
        if self._thread is not None or self._closed:
            return
        with self._lock:
            if self._thread is not None or self._closed:
                return
            thread = threading.Thread(
                target=self._run,
                name="lo-analytics-writer",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def _run(self) -> None:
        """Drain the queue on ONE thread until a sentinel arrives.

        Call samples are batched into one transaction; name tasks are applied
        as they come (rare). Handling both here — rather than a second thread
        for names — is what keeps a single write connection and avoids the
        first-connection race two writers hit on a fresh database.
        """
        while True:
            batch: list[CallSnapshot] = []
            names: list[_NameTask] = []
            try:
                item = self._queue.get(timeout=_FLUSH_INTERVAL_S)
            except queue.Empty:
                self._maybe_prune()
                continue
            if item is None:  # sentinel: flush and exit
                self._flush(batch, names)
                self._queue.task_done()
                return
            self._classify(item, batch, names)
            self._queue.task_done()
            # Opportunistically drain whatever else is already queued so a
            # burst becomes one transaction.
            while len(batch) < 256:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self._flush(batch, names)
                    self._queue.task_done()
                    return
                self._classify(item, batch, names)
                self._queue.task_done()
            self._flush(batch, names)
            self._maybe_prune()

    @staticmethod
    def _classify(
        item: "CallSnapshot | _NameTask",
        batch: list[CallSnapshot],
        names: list["_NameTask"],
    ) -> None:
        if isinstance(item, _NameTask):
            names.append(item)
        else:
            batch.append(item)

    def _flush(self, batch: list[CallSnapshot], names: list["_NameTask"]) -> None:
        if names:
            for task in names:
                try:
                    self._store.upsert_session_name(task.session_id, task.name)
                except Exception:  # noqa: BLE001 — a bad name must not kill the writer
                    logger.debug("analytics: name upsert failed", exc_info=True)
        if not batch:
            return
        try:
            self._store.record_batch(batch)
        except Exception:  # noqa: BLE001 — writer must never die on a bad batch
            logger.debug("analytics: flush failed", exc_info=True)
        finally:
            # Advance the commit barrier whether or not the write succeeded: a
            # dropped batch still "settled", and a test waiting on this count
            # must not hang because a batch failed. Counted in the finally so
            # the barrier tracks attempts, matching ``_enqueued``.
            self._committed += len(batch)

    def _maybe_prune(self) -> None:
        now = time.monotonic()
        if now - self._last_prune < _PRUNE_INTERVAL_S:
            return
        self._last_prune = now
        try:
            self._store.prune()
        except Exception:  # noqa: BLE001
            logger.debug("analytics: prune failed", exc_info=True)

    # -- API -----------------------------------------------------------------
    def record(self, snapshot: CallSnapshot) -> None:
        """Enqueue a call sample. Non-blocking; drops on a full queue.

        This is the ONLY method a provider path calls, and it does the least
        possible work: a bounded ``put_nowait``. It never raises — a recorder
        that cannot accept a sample must not turn into a failed provider call.
        """
        if self._closed:
            return
        self._ensure_thread()
        try:
            self._queue.put_nowait(snapshot)
            self._enqueued += 1
        except queue.Full:
            # Count and log ONCE per power-of-two so a wedged disk says so
            # without spamming, and never block the caller.
            self._dropped += 1
            if self._dropped & (self._dropped - 1) == 0:
                logger.warning("analytics: queue full, dropped %d samples", self._dropped)
        except Exception:  # noqa: BLE001 — recording is best-effort
            logger.debug("analytics: enqueue failed", exc_info=True)

    def note_session_name(self, session_id: str, name: str) -> None:
        """Best-effort: record a session's human name off the hot path.

        Enqueues a name task the writer thread applies on its own connection,
        so the caller (a naming callback on the event loop) never touches
        SQLite and there is only ever ONE thread writing to the database. Like
        :meth:`record`, non-blocking and never raising: a full queue drops the
        name rather than stalling the session.
        """
        if self._closed or not session_id or not name:
            return
        self._ensure_thread()
        try:
            self._queue.put_nowait(_NameTask(session_id, name))
        except queue.Full:
            logger.debug("analytics: queue full, dropped a session name")
        except Exception:  # noqa: BLE001 — naming is best-effort
            logger.debug("analytics: name enqueue failed", exc_info=True)

    @property
    def dropped(self) -> int:
        """How many samples were dropped for a full queue (0 on a healthy run)."""
        return self._dropped

    def flush_for_test(self, timeout: float = 5.0) -> None:
        """Block until the queue drains. TEST ONLY — never called on a session.

        Real sessions never wait for the writer; this exists so a test can
        assert a recorded call reached the store deterministically.
        """
        self._ensure_thread()
        deadline = time.monotonic() + timeout
        # First: the commit barrier catches up to enqueued SNAPSHOTS. This is a
        # DURABLE barrier (the writer advances ``_committed`` only after a batch
        # write returns), so it cannot return before the row is on disk the way
        # a bare ``queue.empty()`` check can — the queue empties the instant the
        # writer dequeues, before it commits.
        target = self._enqueued
        while self._committed < target and time.monotonic() < deadline:
            time.sleep(0.005)
        # Then: drain any remaining items (name tasks carry no barrier of their
        # own, and a name enqueued after the last snapshot rides a later batch).
        # ``join`` waits for every ``task_done``, which the writer calls only
        # after processing the item, so the name upsert has run when this
        # returns.
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.005)
        # A final short settle so an item dequeued-but-not-yet-committed lands.
        time.sleep(0.05)

    def close(self, timeout: float = 2.0) -> None:
        """Stop the writer and close the store (process teardown / tests)."""
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=timeout)
            self._thread = None
        self._store.close()


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_recorder: AnalyticsRecorder | None = None
_recorder_lock = threading.Lock()


def get_recorder() -> AnalyticsRecorder:
    """The process's shared recorder, created on first use."""
    global _recorder
    if _recorder is not None:
        return _recorder
    with _recorder_lock:
        if _recorder is None:
            _recorder = AnalyticsRecorder()
    return _recorder


def record_call(snapshot: CallSnapshot) -> None:
    """Module-level convenience: enqueue a sample on the shared recorder.

    The provider path calls THIS rather than reaching for the singleton, so the
    hot path is one function call with no attribute lookups on a lock.
    """
    try:
        get_recorder().record(snapshot)
    except Exception:  # noqa: BLE001 — recording is never allowed to raise
        logger.debug("analytics: record_call failed", exc_info=True)


def reset_recorder_for_test(store: AnalyticsStore | None = None) -> AnalyticsRecorder:
    """Replace the singleton with a fresh recorder. TEST ONLY.

    Lets a test point the recorder at a tmp-path store and get deterministic,
    isolated behaviour. Closes any existing recorder first so its thread and
    connection do not leak between tests.
    """
    global _recorder
    with _recorder_lock:
        if _recorder is not None:
            _recorder.close()
        _recorder = AnalyticsRecorder(store=store)
    return _recorder
