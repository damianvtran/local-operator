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

Why retention has ONE owner per host, elected on the config root. Every ``lop``
process that records a call builds a recorder, and each recorder prunes the
retention window on its own hourly timer -- so on a host with 18-22 live
processes holding one shared ``analytics.db`` there were 18-22 hourly sweeps
over the same ledger, each one paying for a backlog delete and its WAL churn.
The sweep is elected instead: the recorders race for a non-blocking ``flock``
on ``<config root>/run/analytics-maintenance.lock``, the winner runs the sweep
and stamps the hour into that file, and every loser skips it. See
:meth:`AnalyticsRecorder._run_owned_maintenance` for the mechanism and why it
cannot block the loop or a session.
"""

from __future__ import annotations

import errno
import logging
import os
import queue
import threading
import time
from pathlib import Path

from local_operator.analytics.model import CallSnapshot
from local_operator.analytics.store import SESSION_NAME_RANK_TITLE, AnalyticsStore
from local_operator.paths import config_dir

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

    __slots__ = ("session_id", "name", "rank")

    def __init__(self, session_id: str, name: str, rank: int) -> None:
        self.session_id = session_id
        self.name = name
        #: Precedence of this name, one of ``store.SESSION_NAME_RANK_*``. Carried
        #: on the task rather than resolved by the writer because only the
        #: CALLER knows which kind of label it holds — a provisional stand-in and
        #: a generated title reach this queue through the same method.
        self.rank = rank


class _ToolCallTask:
    """One tool-call sample queued for the writer thread.

    Routed through the SAME queue, thread and connection as call samples and
    name upserts, for the reason ``_NameTask`` records: two threads opening
    their first connection to a freshly-created database race in a way that
    left the writer unable to see its own commits. There is exactly one writer
    here and adding a second is forbidden.

    Fields are the store's insert tuple, flattened. The producer is the harness
    (``AgentLoop.park``), which has no analytics import and reaches this through
    a ``LoopConfig`` callback — so the shape has to be primitives.
    """

    __slots__ = ("ts_ms", "session_id", "tool_name", "origin", "fault", "duration_ms")

    def __init__(
        self,
        ts_ms: int,
        session_id: str,
        tool_name: str,
        origin: str,
        fault: str,
        duration_ms: float,
    ) -> None:
        self.ts_ms = ts_ms
        self.session_id = session_id
        self.tool_name = tool_name
        self.origin = origin
        self.fault = fault
        self.duration_ms = duration_ms

    def as_row(self) -> tuple[int, str, str, str, str, float]:
        return (
            self.ts_ms,
            self.session_id,
            self.tool_name,
            self.origin,
            self.fault,
            self.duration_ms,
        )


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
#:
#: This is now the cadence for the HOST, not for each process: the same value
#: gates the per-recorder timer (which costs no syscall) and the hour stamped
#: into the election file (which is what stops a second process from sweeping
#: the same hour). See :meth:`AnalyticsRecorder._run_owned_maintenance`.
_PRUNE_INTERVAL_S = 3600.0

#: Where the host-wide maintenance election lives, under the config root.
#:
#: ``run/`` is the directory for a root's runtime sidecars — the maintenance
#: lock is not state anybody reads back, and the root it sits under is the whole
#: point: see :meth:`AnalyticsRecorder._resolve_maintenance_root` for why an
#: isolated run elects in ITS OWN root rather than in the operator's.
_RUN_DIRNAME = "run"
_MAINTENANCE_LOCK_NAME = "analytics-maintenance.lock"

#: Whether this platform has the ``flock`` the election is built on. ``fcntl`` is
#: imported inside the helpers below rather than at module scope, the way
#: ``secrets/client.py`` does it, because a module-level import makes this module
#: — and through it the whole analytics path — unimportable on Windows.
#:
#: WHERE THERE IS NO ``flock`` THE SWEEP RUNS UNOWNED, which is the pre-election
#: behaviour: every process sweeps on its own hourly timer. That is deliberately
#: the degradation rather than "no election, no sweep", because the alternative
#: silently stops retention on Windows and lets the ledger grow without bound —
#: a worse outcome than the redundant sweep this feature removes.
_MAINTENANCE_ELECTION_SUPPORTED = os.name == "posix"

#: ``_try_lock_maintenance``'s third outcome: the kernel or the filesystem does
#: not implement the lock at all (see the errno split in that function). A
#: sentinel rather than ``None`` because the caller must tell "somebody else owns
#: this hour" — do nothing — apart from "nobody can own any hour here", which
#: sweeps unowned exactly as the no-``flock`` platform path does.
_LOCK_UNSUPPORTED = -1


def maintenance_lock_path(root: Path) -> Path:
    """The election file for a resolved config root. Pure — creates nothing.

    Split out from the recorder so a test, a doctor command or a support
    session can name the file a root would use without constructing a recorder
    and without touching the filesystem.
    """
    return Path(root) / _RUN_DIRNAME / _MAINTENANCE_LOCK_NAME


def _try_lock_maintenance(path: Path) -> int | None:
    """Take the election lock, return ``None``, or report it unsupported.

    ``None`` means DO NOT SWEEP, and it has exactly two causes — a PEER owns this
    hour, or the root will not take the lock file at all. The second is the one
    case that cannot fall back to an unowned sweep, because the sweep writes
    through the same root: if ``os.open`` fails on ``<root>/run/``, the store
    under ``<root>/`` is not writable either, so there is nothing to prune and
    skipping is the safe direction (the alternative is every process trying).

    :data:`_LOCK_UNSUPPORTED` is the DIFFERENT case: the root IS writable and
    ``flock`` specifically is refused, so the sweep can proceed and the caller
    runs it unowned. That is the distinction review R1-3 is about, and the reason
    ``None`` and the sentinel cannot share a return value. Review R3-1 found this
    docstring claiming ``None`` had only the peer cause; both causes are named
    here now. NEVER blocks.

    The mechanism is the tree's existing singleton one, copied from
    :func:`local_operator.secrets.client.ensure_broker` rather than invented:
    ``os.open(O_RDWR | O_CREAT, 0o600)``, then ``flock(LOCK_EX | LOCK_NB)``.

    THREE outcomes, not two: an ``int`` fd when this process owns the lock, ``None``
    when a peer does, and :data:`_LOCK_UNSUPPORTED` when the kernel or the
    filesystem refuses the call altogether — see the errno split below.

    **Non-blocking is the whole design, and it is #401's lesson.** A blocking
    ``flock`` in this codebase froze the TUI — a holder that wedges must cost a
    bounded delay, not a freeze — so the loser here does not wait at all.
    ``ensure_broker`` additionally polls for the winner's socket (its
    ``_LOCK_POLL_S``) because it is waiting for a daemon to come UP; this
    election has nothing to wait for, since a loser's correct answer is "not my
    hour" and the next attempt is an hour away. So the loser's whole cost is one
    ``open`` plus one refused ``flock`` — microseconds — and a holder that
    wedges inside the sweep, or dies holding the lock, costs it exactly the
    same. The lock is released by ``_unlock_maintenance`` and, if that never
    runs, by the process exiting: the kernel drops an ``flock`` with its last
    descriptor, so a crashed owner cannot wedge the host's maintenance forever.

    The file is NEVER unlinked, by anyone. Deleting a lock file is the classic
    way to hand two processes the same lock through different inodes, and a
    leftover empty file in ``run/`` is not worth that.
    """
    import fcntl

    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        # An unwritable root (a read-only home, a root somebody else owns) means
        # no election here; the sweep is skipped, which is the safe direction:
        # the alternative is every process sweeping.
        logger.debug("analytics: no maintenance lock at %s", path, exc_info=True)
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        # NOT EVERY `flock` FAILURE IS CONTENTION, and reading them all as
        # "another recorder owns this hour" wedges retention FOREVER on a root
        # that refuses the call: `ENOLCK` (lock table full), `EOPNOTSUPP`/
        # `ENOTSUP` (the filesystem does not implement it — some network and FUSE
        # mounts), `EINVAL`. Each of those leaves every process answering "not my
        # hour" for every hour, which is the unbounded-ledger outcome
        # `_MAINTENANCE_ELECTION_SUPPORTED` already refuses on Windows — reached
        # here through the filesystem rather than the platform. Reported as review
        # R1-3, with a probe showing 3 attempts and 0 prunes.
        #
        # `_LOCK_UNSUPPORTED` is therefore a THIRD outcome: sweep unowned, exactly
        # as a platform without `flock` does. The bound that makes it safe is the
        # one that shapes the whole election — `_maybe_prune` stamps
        # `_last_prune` BEFORE the attempt, so an unowned sweep still happens at
        # most once per process per hour. N processes on such a root fall back to
        # the pre-election behaviour rather than to no maintenance at all, and the
        # warning says so once per attempt rather than silently.
        # EAGAIN/EWOULDBLOCK (the same value) is the ONLY documented contention
        # signal: `flock(2)`'s ERRORS list is EBADF, EINTR, EINVAL, ENOLCK,
        # EWOULDBLOCK on Linux and EWOULDBLOCK, EBADF, EINVAL, EOPNOTSUPP on the
        # OpenBSD/macOS lineage — `EACCES` appears in neither, so classifying it
        # here would re-admit the wedge through an errno this code only guessed
        # at (an SMB/CIFS mount is exactly where an undocumented errno is most
        # likely, per the same man page's mandatory-locking note). The asymmetry
        # decides it: the two branches differ only in whether retention can
        # stall, so the default is the sweep.
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
            return None
        logger.warning(
            "analytics: flock is unusable on %s (errno %s); sweeping unowned",
            path,
            exc.errno,
        )
        return _LOCK_UNSUPPORTED
    return fd


def _unlock_maintenance(fd: int) -> None:
    """Release an election lock taken by :func:`_try_lock_maintenance`."""
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        # The close below is what matters: an unlock that fails because the
        # descriptor is already gone still drops the lock with the process's
        # last reference to it.
        pass
    os.close(fd)


def _read_sweep_claim(fd: int) -> float | None:
    """The wall-clock stamp of the last owned sweep, or ``None`` if unusable.

    Read AFTER the lock is taken, so the value cannot change under this process
    between the check and the write. Anything unparseable — an empty file, a
    truncated write from a process that died mid-stamp — reads as "no claim",
    which makes the caller sweep. That is the right direction to fail in: a
    claim that cannot be read costs one redundant sweep, while a claim that
    cannot be written but reads as fresh would stop maintenance for an hour.
    """
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 64)
    except OSError:
        return None
    try:
        return float(raw.decode("ascii", "replace").strip())
    except ValueError:
        return None


def _write_sweep_claim(fd: int, when: float) -> None:
    """Stamp the hour into the election file. Best-effort; never raises."""
    payload = f"{int(when)}\n".encode("ascii")
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, payload)
        os.ftruncate(fd, len(payload))
    except OSError:
        logger.debug("analytics: could not write the maintenance claim", exc_info=True)


class AnalyticsRecorder:
    """Owns the queue, the writer thread, and the store.

    One instance per process (see :func:`get_recorder`). Construction does NOT
    start the thread or open the database — the first :meth:`record` does, so a
    process that never makes a provider call pays nothing.
    """

    def __init__(
        self,
        store: AnalyticsStore | None = None,
        *,
        maintenance_root: Path | None = None,
    ) -> None:
        self._store = store if store is not None else AnalyticsStore()
        #: An EXPLICIT root for the maintenance election, or ``None`` to resolve
        #: one (see :meth:`_resolve_maintenance_root`). Keyword-only and normally
        #: absent: it exists so a test can point two recorders at one root and
        #: assert the election, and so a caller that already knows the root does
        #: not have to make the recorder re-derive it.
        self._maintenance_root = Path(maintenance_root) if maintenance_root is not None else None
        self._queue: "queue.Queue[CallSnapshot | _NameTask | _ToolCallTask | None]" = queue.Queue(
            maxsize=_QUEUE_MAXSIZE
        )
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._dropped = 0
        self._last_prune = 0.0
        self._closed = False
        #: Fail-soft write observability, read ONLY by ``flush_for_test``: every
        #: write the store did not make is counted here with a one-line detail.
        #: Two shapes reach it and they are the same event to a caller — the
        #: call RAISED (a bad sample, which must never kill the writer), or it
        #: RETURNED a drop, which is what the real store does on a lost lock: it
        #: retries, gives up, and returns 0 rows / ``False`` rather than raising.
        #: An item whose write was lost still settles, so the completion count
        #: alone cannot tell a test that its row landed; this is how the barrier
        #: reports the difference instead of letting a test read a missing row
        #: and fail on a bare ``None``. The swallow-and-log policy itself is
        #: unchanged and deliberate, and production never reads these: the
        #: increments happen only on the failure path.
        self._write_failures: dict[str, int] = {}
        self._write_failure_detail: dict[str, str] = {}
        #: How many failures per kind a ``flush_for_test`` has already reported,
        #: so a report names only what is NEW since the last one (and one
        #: failure is surfaced by exactly one barrier rather than by every
        #: barrier for the rest of the process).
        self._reported_write_failures: dict[str, int] = {}

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

        ``task_done()`` is called for every item ``get()`` returned — the
        sentinel included — and only AFTER the ``_flush`` carrying it has
        returned. That ordering is the whole meaning of the queue's completion
        count: ``flush_for_test`` waits on it, so a ``task_done`` taken before
        the write lets the barrier return with the row still unwritten. It used
        to be taken as each item was CLASSIFIED, with the flush at the end of
        the iteration — which is #1250 item 2, where a session-name upsert the
        writer had already dequeued could still be pending when a test read the
        ``session_names`` row. Nothing in production joins this queue, so
        settling after the flush costs a session nothing.

        If ``_flush`` ever did raise, the count would stop short and the next
        ``flush_for_test`` would report it as an unsettled item rather than
        passing — loudly, which is the point. Every store call inside ``_flush``
        is individually guarded so this stays theoretical.
        """
        while True:
            batch: list[CallSnapshot] = []
            names: list[_NameTask] = []
            tools: list[_ToolCallTask] = []
            # How many queue items this iteration took off, so the settle below
            # accounts for each of them exactly once (``unfinished_tasks`` is
            # raised by ``put`` and lowered by ``task_done``, never by ``get``).
            consumed = 0
            stopping = False
            try:
                item = self._queue.get(timeout=_FLUSH_INTERVAL_S)
            except queue.Empty:
                self._maybe_prune()
                continue
            consumed += 1
            if item is None:  # sentinel: flush, settle, exit
                stopping = True
            else:
                self._classify(item, batch, names, tools)
                # Opportunistically drain whatever else is already queued so a
                # burst becomes one transaction.
                while len(batch) < 256:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    consumed += 1
                    if item is None:
                        stopping = True
                        break
                    self._classify(item, batch, names, tools)
            self._flush(batch, names, tools)
            for _ in range(consumed):
                self._queue.task_done()
            if stopping:
                return
            self._maybe_prune()

    @staticmethod
    def _classify(
        item: "CallSnapshot | _NameTask | _ToolCallTask",
        batch: list[CallSnapshot],
        names: list["_NameTask"],
        tools: list["_ToolCallTask"],
    ) -> None:
        if isinstance(item, _NameTask):
            names.append(item)
        elif isinstance(item, _ToolCallTask):
            tools.append(item)
        else:
            batch.append(item)

    def _flush(
        self,
        batch: list[CallSnapshot],
        names: list["_NameTask"],
        tools: list["_ToolCallTask"] | None = None,
    ) -> None:
        if tools:
            # Its own transaction, not joined to the ledger insert below: tool
            # calls are produced DURING a turn and the ledger row at the end of
            # it, so the two never share a batch anyway.
            try:
                # ``as_row()`` belongs INSIDE the guard with the call it feeds:
                # the guard's contract is that no bad sample can kill the writer
                # thread, and an expression outside it is one edit away from
                # doing exactly that (a dead writer turns every later barrier
                # into a ``TimeoutError``).
                rows = [task.as_row() for task in tools]
                written = self._store.record_tool_calls(rows)
            except Exception as exc:  # noqa: BLE001 — a bad sample must not kill the writer
                logger.debug("analytics: tool-call flush failed", exc_info=True)
                self._note_write_failure("tool call", repr(exc))
            else:
                if written != len(rows):
                    self._note_write_failure(
                        "tool call", f"the store wrote {written} of {len(rows)} rows"
                    )
        if names:
            for task in names:
                try:
                    landed = self._store.upsert_session_name(
                        task.session_id, task.name, rank=task.rank
                    )
                except Exception as exc:  # noqa: BLE001 — a bad name must not kill the writer
                    logger.debug("analytics: name upsert failed", exc_info=True)
                    self._note_write_failure("session name", repr(exc))
                else:
                    if not landed:
                        # ``False`` is the store's DROP signal, not a rank-gated
                        # refusal: a refusal still ran the statement and returns
                        # True, so only a write the database never saw lands here.
                        self._note_write_failure("session name", "the store dropped the upsert")
        if not batch:
            return
        try:
            written = self._store.record_batch(batch)
        except Exception as exc:  # noqa: BLE001 — writer must never die on a bad batch
            logger.debug("analytics: flush failed", exc_info=True)
            self._note_write_failure("ledger batch", repr(exc))
        else:
            if written != len(batch):
                self._note_write_failure(
                    "ledger batch", f"the store wrote {written} of {len(batch)} rows"
                )

    def _note_write_failure(self, kind: str, detail: str) -> None:
        """Count a write the store did not make, for ``flush_for_test`` to report.

        Called on BOTH shapes of a lost write — the store raising, and the real
        store's silent drop, which it reports by returning 0 rows / ``False``
        rather than by raising (``AnalyticsStore.record_batch`` retries a lost
        lock ``_WRITE_RETRIES`` times and then gives up). The store's return
        values are the source of truth here: a store that drops a write by
        swallowing it internally would otherwise be indistinguishable from one
        that wrote it.

        Only the writer thread writes, so the read-modify-write below needs no
        lock; the barrier reads a snapshot after the queue has settled.
        """
        self._write_failures[kind] = self._write_failures.get(kind, 0) + 1
        self._write_failure_detail[kind] = detail

    def _maybe_prune(self) -> None:
        """Run the hourly maintenance attempt, if this process is due for one.

        Called from the writer thread's loop (both the idle tick and the tail of
        a batch), NEVER from the event loop — see :meth:`_run_owned_maintenance`
        for what the attempt does and why that placement is load-bearing.

        The per-process timer is checked FIRST because it is free: it is an
        attribute comparison, so 22 processes ticking twice a second do not open
        a lock file 44 times a second. Only when a process is due does it touch
        the filesystem, which bounds syscalls at one election attempt per
        process per hour.

        ``_last_prune`` is stamped BEFORE the attempt, as it always was, so a
        sweep that fails cannot retry on every tick for the rest of the hour.
        """
        now = time.monotonic()
        if now - self._last_prune < _PRUNE_INTERVAL_S:
            return
        self._last_prune = now
        try:
            self._run_owned_maintenance()
        except Exception:  # noqa: BLE001 — maintenance must never kill the writer
            # The broad guard is not decoration: an exception escaping here
            # would end the writer thread's loop, and every later
            # ``flush_for_test``/session would find a dead writer. Lock files,
            # foreign filesystems and store failures are all outside this
            # process's control, so the whole attempt is guarded rather than
            # each syscall inside it.
            logger.debug("analytics: maintenance failed", exc_info=True)

    def _resolve_maintenance_root(self) -> Path:
        """The root this recorder's election is held in.

        Three sources, in this order, and the order is the isolation rule:

        1. an explicit ``maintenance_root`` (tests);
        2. the DIRECTORY OF THE STORE'S OWN DATABASE. For the process default
           that is ``config_dir() / "analytics.db"``, so the root is the
           resolved config root — and for a store pointed anywhere else (a
           test's ``tmp_path``, an explicit ``db_path``) the election follows
           the store instead of landing in the operator's root;
        3. :func:`local_operator.paths.config_dir`, for a store that cannot name
           its database (a stub), so the behaviour is still the documented one.

        WHAT THIS FUNCTION IS CAREFUL ABOUT. It resolves the root the same way
        every other per-root thing here does — through ``config_dir()``, which
        honours ``LOCAL_OPERATOR_CONFIG_DIR`` — and it NEVER keys on
        ``Path.home()``. That distinction is not theoretical: the browser
        bridge's supervisor label was derived from ``Path.home()`` while its
        launchd domain was not, so an isolated run registered the GLOBAL label
        and evicted the operator's live daemon (see AGENTS.md, "Isolating a run";
        #1310 is the same class of mistake). An isolated run must elect within
        its own root and must not touch the operator's store, so the root here
        is always a RESOLVED root and never a home shortcut — ``config_dir()``
        itself falls back to ``~/.local-operator`` only when there is no
        override, which is the default root rather than a guess about one.
        """
        if self._maintenance_root is not None:
            return self._maintenance_root
        db_path = getattr(self._store, "db_path", None)
        if db_path is not None:
            return Path(db_path).parent
        return config_dir()

    @property
    def maintenance_lock(self) -> Path:
        """Where this recorder's maintenance election file would live.

        Pure: nothing is created until a sweep is due, which is why a recorder
        built by a test that never ticks leaves no trace in the root it names.
        """
        return maintenance_lock_path(self._resolve_maintenance_root())

    def _run_owned_maintenance(self) -> bool:
        """Elect an owner for this hour's sweep, and sweep if it is us.

        Runs on the writer thread. Returns True when THIS call swept.

        THE MECHANISM, in three steps:

        1. Take ``<root>/run/analytics-maintenance.lock`` with
           ``LOCK_EX | LOCK_NB``. A loser returns here and does nothing else: no
           polling, no waiting, no sweep. A root whose filesystem REFUSES the call
           is not a loser — it cannot have a winner — so that case sweeps unowned
           instead (see :data:`_LOCK_UNSUPPORTED`). The lock is held for the duration
           of the sweep only, so a process that is killed mid-sweep releases it
           with its descriptor.
        2. Read the hour stamped in the file. The winner of the LOCK is not
           automatically the owner of the HOUR: with this lock alone, process A
           would sweep, release, and process B — whose own hourly timer fires
           two minutes later — would take the free lock and sweep again, which
           is the 18-22-sweeps-an-hour problem wearing a different hat. The
           stamp is what makes ownership survive the winner's exit: a claim
           younger than ``_PRUNE_INTERVAL_S`` means this hour is already done
           and the caller skips it.
        3. Sweep, then stamp. The stamp is written only after the sweep
           returned, so a sweep that failed is not recorded as done and the next
           process to come due will retry it.

        WHY A LOCK AT ALL, GIVEN THAT THE SWEEP IS IDEMPOTENT. The claim in the
        file is a scheduling datum, and the two steps around it (read, decide,
        write) are a read-modify-write that must not interleave — that is all
        the lock protects. ``prune()`` is idempotent and needs no mutual
        exclusion to be CORRECT, so nothing of value crosses this lock: no
        capability, no token, no secret, and no authority over any other
        process. That is deliberate, and it is what keeps this out of the
        same-uid impostor class the secret broker's own lock is careful about
        (#1310): an attacker who takes this lock can only make analytics
        maintenance happen LATER, and every participant is a recorder of the
        operator's own uid doing the operator's own retention sweep. A lock that
        carried a capability would need the broker's peer authentication; this
        one does not, and must not grow one.

        WHY THIS CANNOT FREEZE ANYTHING. Every syscall here is bounded and none
        of them is on the event loop:

        * the lock is ``LOCK_NB`` and there is NO poll loop, so a wedged holder
          costs a loser one refused ``flock``;
        * the claim read/write is a bounded ``pread``/``write`` of 64 bytes on
          the descriptor already held;
        * the sweep itself (``store.prune`` and ``store.bound_wal``) is SQLite
          I/O on the writer thread, and the WAL half is shaped so that it can
          never wait for a reader (see ``AnalyticsStore.bound_wal``).

        So a session's provider path — which only ever does a ``put_nowait``
        on the queue — is untouched by any of it, whatever another process is
        doing under the lock.
        """
        if not _MAINTENANCE_ELECTION_SUPPORTED:
            # No ``flock`` on this platform: sweep unowned. See the constant.
            return self._sweep()
        path = self.maintenance_lock
        fd = _try_lock_maintenance(path)
        if fd is None:
            # Somebody else owns this hour. Do nothing — no polling, no waiting.
            return False
        if fd == _LOCK_UNSUPPORTED:
            # Nobody CAN own an hour on this root (see ``_try_lock_maintenance``):
            # sweep unowned rather than letting retention stop forever. The
            # per-process hourly stamp still bounds this to one sweep per process
            # per hour, which is the pre-election behaviour.
            self._sweep()
            return True
        try:
            now = time.time()
            claimed = _read_sweep_claim(fd)
            if claimed is not None and 0.0 <= now - claimed < _PRUNE_INTERVAL_S:
                # Another process swept this hour. Not a failure: the ledger is
                # already bounded, which is the whole point of the election.
                return False
            swept = self._sweep()
            if swept:
                _write_sweep_claim(fd, now)
            return swept
        finally:
            _unlock_maintenance(fd)

    def _sweep(self) -> bool:
        """The maintenance itself: prune the ledger, then bound the WAL.

        Each half is guarded on its own so a failure in one does not skip the
        other, and the return value is "the prune ran without raising", which is
        what the caller stamps into the election file: the checkpoint is a
        best-effort reclaim of space that the next sweep will try again.

        Order matters. The prune deletes rows — that is what writes the frames
        into the WAL — and the checkpoint then moves what is left back into the
        database and hands the file's space back. Checkpointing first would
        reclaim the space and then immediately re-earn it.
        """
        swept = True
        try:
            self._store.prune()
        except Exception:  # noqa: BLE001 — a failing prune must not skip the checkpoint
            logger.debug("analytics: prune failed", exc_info=True)
            swept = False
        try:
            self._store.bound_wal()
        except Exception:  # noqa: BLE001 — a store without a WAL bound is fine
            logger.debug("analytics: WAL bound failed", exc_info=True)
        return swept

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
        except queue.Full:
            # Count and log ONCE per power-of-two so a wedged disk says so
            # without spamming, and never block the caller.
            self._dropped += 1
            if self._dropped & (self._dropped - 1) == 0:
                logger.warning("analytics: queue full, dropped %d samples", self._dropped)
        except Exception:  # noqa: BLE001 — recording is best-effort
            logger.debug("analytics: enqueue failed", exc_info=True)

    def note_session_name(
        self, session_id: str, name: str, *, rank: int = SESSION_NAME_RANK_TITLE
    ) -> None:
        """Best-effort: record a session's human name off the hot path.

        Enqueues a name task the writer thread applies on its own connection,
        so the caller (a naming callback on the event loop) never touches
        SQLite and there is only ever ONE thread writing to the database. Like
        :meth:`record`, non-blocking and never raising: a full queue drops the
        name rather than stalling the session.

        ``rank`` states what KIND of label this is (see the
        ``SESSION_NAME_RANK_*`` constants). It defaults to a real title so the
        original caller — ``Session.set_conversation_name`` — keeps its exact
        previous behaviour; a stand-in label must pass the lower rank explicitly
        so it can fill an empty row without ever displacing a real title.
        """
        if self._closed or not session_id or not name:
            return
        self._ensure_thread()
        try:
            self._queue.put_nowait(_NameTask(session_id, name, rank))
        except queue.Full:
            logger.debug("analytics: queue full, dropped a session name")
        except Exception:  # noqa: BLE001 — naming is best-effort
            logger.debug("analytics: name enqueue failed", exc_info=True)

    def record_tool_call(
        self,
        session_id: str,
        tool_name: str,
        origin: str,
        fault: str,
        duration_ms: float = -1.0,
    ) -> None:
        """Best-effort: record one tool call's outcome off the hot path.

        Called from ``AgentLoop.park``, which runs ON THE EVENT LOOP inside a
        live turn, and invoked SYNCHRONOUSLY there — the loop puts no timeout
        and no thread between itself and this call, because at the measured
        0.0018 ms/call scheduling would cost more than the work. So the
        non-blocking half of the contract is load-bearing rather than advisory:
        whatever time this spends is added straight to the turn (measured
        0.002 s → 0.754 s against a hook that sleeps 0.75 s). One bounded
        ``put_nowait``, never raising, never touching disk or a lock on this
        side — analytics that can add latency to a turn or abort one is a
        defect, not a measurement.

        ``fault`` is ``""`` for a call that ran cleanly, else the classification
        set at the source (see ``store``'s ``tool_calls`` schema comment).
        ``origin`` decides which population the call is counted in and must be
        one of ``analytics.model.ORIGIN_MODEL`` / ``ORIGIN_NESTED``; only the
        former reaches a rate (see the origin partition on ``ToolCallStats``).
        """
        if self._closed or not session_id:
            return
        self._ensure_thread()
        try:
            self._queue.put_nowait(
                _ToolCallTask(
                    int(time.time() * 1000),
                    session_id,
                    tool_name,
                    origin,
                    fault,
                    float(duration_ms),
                )
            )
        except queue.Full:
            logger.debug("analytics: queue full, dropped a tool call")
        except Exception:  # noqa: BLE001 — recording is best-effort
            logger.debug("analytics: tool-call enqueue failed", exc_info=True)

    @property
    def dropped(self) -> int:
        """How many samples were dropped for a full queue (0 on a healthy run)."""
        return self._dropped

    def flush_for_test(self, timeout: float = 5.0) -> None:
        """Block until every queued item has been through the write path. TEST ONLY.

        Never called on a session: real sessions do not wait for the writer.
        It exists so a test can assert that a recorded sample reached the store
        deterministically, which is only worth having if returning MEANS the
        write happened.

        It did not mean that before #1250 item 2. The old body waited on a
        commit count that only call snapshots advanced, then polled
        ``queue.empty()`` and slept a flat 50 ms — three separate holes:

        * the queue empties the moment the writer DEQUEUES, so an item in the
          writer's hands passed the empty check; and the queued items it did
          see were counted as settled by the 50 ms sleep, not by a write;
        * an item carrying no snapshot — a session name, a tool call — never
          moved a counter at all, so the barrier never waited on it;
        * on expiry it returned as if it had succeeded.

        That is why ``tests/unit/evaluation/runner/test_provider_client.py``
        could read a ``session_names`` row the writer had not committed yet.

        The barrier is the queue's own completion count: the writer calls
        ``task_done()`` for an item only after the ``_flush`` carrying it has
        returned, so returning means every item enqueued before now has been
        through ``_flush``, names and tool calls included. That is a statement
        about the write PATH, not about the row: the writer's store calls are
        fail-soft (a bad sample must never kill the writer or stall a session),
        so an item whose write was LOST settles like any other. A lost write is
        reported two ways — the store raised, or it returned a drop (the real
        ``AnalyticsStore`` never raises on a lost lock: it retries, gives up and
        returns 0 rows / ``False``) — and the writer counts both. This method
        raises on any it has not already reported, so it cannot return clean
        while the row a caller is about to read is missing, which is the failure
        mode this whole method exists to remove.

        ON EXPIRY OR A LOST WRITE THIS RAISES rather than returning. A caller
        that asked for a guarantee must get either the guarantee or a failure
        naming what went wrong. The write-loss report names only the failures
        since the previous report, so its counts are new losses rather than a
        running total.
        """
        self._ensure_thread()
        deadline = time.monotonic() + timeout
        # ``queue.Queue.join()`` is the right barrier but takes no timeout, and
        # an unbounded wait on a wedged writer fails the suite by HANGING it
        # instead of by reporting. So the bounded wait is spelled the way
        # ``Queue.join`` itself spells it — on ``all_tasks_done``, the condition
        # ``task_done`` notifies — rather than by handing the queue to a helper
        # thread, which would leak one thread per expiry and still could not say
        # what was outstanding.
        queued = self._queue
        with queued.all_tasks_done:
            while queued.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    writer = self._thread
                    raise TimeoutError(
                        f"flush_for_test: {queued.unfinished_tasks} unsettled item(s) "
                        f"still pending {timeout:.2f}s after the call — the "
                        f"analytics writer thread has not written them "
                        f"(writer alive: {writer is not None and writer.is_alive()}, "
                        f"recorder closed: {self._closed})"
                    )
                queued.all_tasks_done.wait(remaining)
        # The items are settled; the rows are a separate question, because a
        # dropped write is not an exception here — the real store reports one by
        # RETURNING 0 rows / ``False`` (``record_batch`` gives up after
        # ``_WRITE_RETRIES`` lost lock races), and the recorder counts that shape
        # and a raising store alike. Report only what is NEW since the last
        # report, so the counts name this barrier's losses rather than a running
        # total that reads like fresh damage.
        #
        # The snapshot is taken first and it is a COPY: a producer can keep
        # enqueuing while this runs (every session records without waiting), so
        # the writer may be counting a loss at this instant, and iterating a dict
        # it is mutating is asking for "dictionary changed size during
        # iteration".
        current = self._write_failures.copy()
        new = {
            kind: count - self._reported_write_failures.get(kind, 0)
            for kind, count in current.items()
            if count > self._reported_write_failures.get(kind, 0)
        }
        if new:
            # Advance the watermark by the deltas actually REPORTED, never to the
            # live totals: a loss counted between the snapshot above and this line
            # would otherwise be marked reported without ever appearing in a
            # message, which is the same silent-success bug this method fixes.
            # Advancing by the delta leaves it greater than the watermark, so the
            # next barrier reports it.
            for kind, count in new.items():
                self._reported_write_failures[kind] = (
                    self._reported_write_failures.get(kind, 0) + count
                )
            detail = ", ".join(
                f"{kind} ×{count} ({self._write_failure_detail.get(kind, 'no detail')})"
                for kind, count in sorted(new.items())
            )
            raise RuntimeError(
                f"flush_for_test: {sum(new.values())} analytics write(s) were lost "
                f"since the last report — {detail}. The items are settled but their "
                f"rows are missing; the writer has already logged each one as "
                f"'analytics: …' at debug level."
            )

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


def record_tool_call(
    session_id: str,
    tool_name: str,
    origin: str,
    fault: str,
    duration_ms: float = -1.0,
) -> None:
    """Module-level convenience: enqueue a tool-call sample. Never raises.

    This is what ``Session`` binds into ``LoopConfig.record_tool_call``. The
    outer guard is not redundant with the recorder's own: this runs on the event
    loop during a turn, and even the singleton lookup must not be able to throw
    into it.
    """
    try:
        get_recorder().record_tool_call(session_id, tool_name, origin, fault, duration_ms)
    except Exception:  # noqa: BLE001 — recording is never allowed to raise
        logger.debug("analytics: record_tool_call failed", exc_info=True)


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
