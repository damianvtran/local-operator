"""The wake supervisor: the process that exists so a wake can fire at all.

A wake scheduled in a session whose terminal is then closed had nowhere to
fire from — the schedule was durable, but nothing was running to notice it
came due. This is the small always-on process that closes that gap: it reads
:mod:`local_operator.wakes.store`, sleeps until the earliest due time across
every session, and starts a runtime for whichever session's wake is due.

**It does not deliver the wake, and that is the design, not an omission.**
The obvious shape — a ``wake_fire`` control op naming the occurrence to
deliver — fires every wake TWICE, because a session already delivers its own
overdue wakes on load: ``WakeScheduler.load`` re-arms anything whose
``next_due_at`` has passed to ``now + LOAD_GRACE_MS`` and records it for the
resume catch-up (``harness/wake.py``). So the mere EXISTENCE of a runtime is
what fires the wake, and an op on top of that would append the occurrence a
second time.

This keeps one writer of schedule state (``Session._persist_wake_schedules``),
which is the property that matters: the supervisor never advances, retires or
persists a schedule, so it can never disagree with the session about what has
fired. Its whole job is "make a runtime exist for this session, now".

**The no-live-record rule.** A session with a live discovery record is
already running and fires its own wakes through its own scheduler. The
supervisor therefore SKIPS it entirely — engaging there would be redundant at
best, and at worst a second opinion about a schedule the live session is
actively advancing. The rule is checked at fire time rather than at scan
time, because a session can come up during the sleep.

**Self-retirement.** Nothing FIREABLE left to supervise (an empty index, or
one holding only dormant entries) means the process exits 0 and its
LaunchAgent (``KeepAlive: {SuccessfulExit: False}``) leaves it down. That is
why the exit code matters: a crash restarts, a finished job stays finished.
The next persist reinstalls it.

Retirement is the dangerous transition, because for a long time nothing
brought the supervisor back: the install hook read a loaded-but-exited job as
"already installed" (see :mod:`local_operator.wakes.install`), so an armed
wake could sit with no live process and no log line. Three things now guard
it \u2014 the install hook probes for a RUNNING process and repairs, the plist
carries a ``StartInterval`` self-heal, and retirement here re-reads the index
after a grace so the write-then-install race cannot retire into a wake that
was being armed at that moment.

Stdlib-only and import-light, like the rest of ``wakes/``: this runs as its
own supervised process and must not drag the harness in at import.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

#: A STABLE name, not ``__name__``. This module is the LaunchAgent's
#: ``python -m`` target, so ``__name__`` is ``__main__`` in the one process
#: whose output the operator actually reads — every line in
#: ``logs/wake-supervisor.log`` was tagged with the least informative name
#: available (round 1, D8). Hard-coding the dotted path makes the LaunchAgent's
#: lines and ``lop wake serve``'s identical.
logger = logging.getLogger("local_operator.wakes.supervisor")

#: Never sleep longer than this, however far away the next wake is. A long
#: sleep is not a correctness problem (the due time is recomputed on every
#: pass) but it is an OBSERVABILITY one: a supervisor asleep for nine hours
#: has not noticed a schedule cancelled eight hours ago, so its own liveness
#: and the index's state drift apart. Waking hourly to re-read is cheap.
#:
#: Since the wait is now SLICED (see :data:`SLICE_S`) this bounds the total
#: wait rather than a single uninterruptible sleep.
MAX_SLEEP_S = 3600.0

#: How long a single uninterrupted sleep may last before the index is re-read.
#:
#: The defect this closes: the loop used to compute one delay from a snapshot
#: and ``asyncio.sleep`` it whole. A session that persisted a wake due in five
#: minutes while the supervisor was already sleeping toward a wake three hours
#: out had that wake delayed by up to ``MAX_SLEEP_S`` — an hour — because the
#: new entry was not observed until the sleep expired. The install hook could
#: not help: it is idempotent by content and never nudges a running process.
#:
#: The cost is honest and small: one ``read_index`` per slice, which is an
#: ``os.listdir`` of a directory holding one small JSON file per wake-carrying
#: session (8 files, ~400 bytes each, on the machine this was diagnosed on) —
#: measured at ~0.2 ms per read there. At 10 s that is ~8,640 reads a day
#: against a wake subsystem that is otherwise idle, which buys bounded
#: lateness: a newly written earlier wake is seen within one slice instead of
#: within one hour.
SLICE_S = 10.0

#: Never sleep less than this. A schedule due in the past, or a clock that
#: jumped backwards, must not turn the loop into a spin — the floor is what
#: bounds a pathological index to one pass per second rather than thousands.
MIN_SLEEP_S = 1.0

#: How late a wake may be before the supervisor stops treating it as due and
#: leaves it to the session's own catch-up. Sessions handle arbitrarily-old
#: overdue wakes at load (that is what the resume catch-up IS), so there is no
#: value in the supervisor racing to start a runtime for one that has been due
#: for a week — the user will get the catch-up when they open it.
STALE_AFTER_S = 7 * 24 * 3600.0

#: How long a WAKE engage may take before the supervisor gives up on it.
#:
#: Deliberately not ``launch.DEFAULT_DEADLINE_S`` (30 s), which stays exactly
#: as it is for user-initiated engages. That figure is sized for a person
#: waiting at a prompt, where the alternative to waiting is telling them their
#: message went nowhere. NOBODY is waiting on a wake, and the cost of the two
#: failures is reversed: giving up early does not save anyone time, it just
#: loses the wake until the next pass.
#:
#: 30 s was simply too short here. On the machine this was diagnosed on, a
#: cold runtime (19 live sessions, large transcripts, 12 MCP servers) often
#: needs longer to publish its record, and 344 of 682 engage attempts (50.4%)
#: timed out at 30 s. Each timeout then re-entered the same path on the next
#: pass, so the deadline was manufacturing the retry storm it appeared to be
#: protecting against.
#:
#: 180 s is a generous multiple of the worst cold start observed rather than a
#: tight bound. What makes a long deadline affordable is that an engage does
#: NOT hold the loop: :func:`serve` runs engagements as background tasks and
#: returns to its slice loop immediately, so the only thing a slow engage
#: occupies is one of :data:`_MAX_CONCURRENT_ENGAGES` slots. (Round 1, R3: an
#: earlier revision awaited the whole sweep, which made a 6-session all-timeout
#: pass block for 540 s — three times worse than the serial 30 s code it
#: replaced — while this comment already claimed otherwise. The claim is now
#: true because the code changed, not because the wording did.)
WAKE_DEADLINE_S = 180.0

#: How many sessions the sweep engages at once. See :func:`fire_due_wakes`.
#:
#: KNOWN RESIDUAL (round 2, QA Q3, deferred deliberately): a freshly armed wake
#: still queues behind permanently-failing engages, because `_due_sessions`
#: sorts oldest-first and a failed engage keeps its original `due_ms`, so it
#: keeps winning the semaphore. Measured at a 10 s scaled deadline: N=0 → 8.2 s,
#: N=2 → 18.2 s, N=6 → 38.2 s (was 58.2 s before the round-1 restructure).
#: Widening this number is NOT the fix — it trades a rare lateness for cold-start
#: storms on a loaded box, where several runtimes coming up at once is exactly
#: what the original 30 s deadline was thrashing on. A real fix is a scheduling
#: question (fair queueing between fresh and retried work), not a constant.
_MAX_CONCURRENT_ENGAGES = 2

#: How many times one (session, reason) skip is logged at full volume before
#: dropping to :data:`_SKIP_HEARTBEAT_S`.
#:
#: A stale or ghost skip NEVER self-clears — a >7-day wake stays >7 days and a
#: ghost session never grows a transcript — so logging it once per slice is
#: 8,640 lines/day/entry into a file nothing rotates (``render_plist`` points
#: ``StandardErrorPath`` at a plain path). The operator's box has two such ids
#: today, so that is ~3.3 MB/day drowning the signal this observability was
#: added to create. The first occurrences are the ones that carry information;
#: after that a periodic heartbeat is enough to show the condition persists.
_SKIP_LOG_BURST = 3

#: Cadence for a skip that has already been logged :data:`_SKIP_LOG_BURST`
#: times. One line an hour per stuck entry keeps the condition visible in the
#: log without flooding it (~24 lines/day/entry rather than 8,640).
_SKIP_HEARTBEAT_S = 3600.0


class _SkipLog:
    """Throttles a skip line that would otherwise repeat every slice forever.

    Keyed by ``(session_id, reason)`` so a *change* of reason speaks up again
    at full volume: an entry that goes from "live runtime owns it" to "stale"
    is new information and must not inherit the old key's silence.

    Process-lifetime state, deliberately. It is not persisted, so a restarted
    supervisor re-announces every standing condition once — which is what an
    operator reading a fresh log wants, and it keeps this module free of any
    state file it would then have to own.
    """

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], tuple[int, float]] = {}

    def should_log(self, session_id: str, reason: str, *, now: float | None = None) -> bool:
        moment = now if now is not None else time.monotonic()
        key = (session_id, reason)
        count, last = self._seen.get(key, (0, 0.0))
        if count < _SKIP_LOG_BURST:
            self._seen[key] = (count + 1, moment)
            return True
        if moment - last >= _SKIP_HEARTBEAT_S:
            self._seen[key] = (count + 1, moment)
            return True
        self._seen[key] = (count, last)
        return False

    def forget(self, session_id: str, *reasons: str) -> None:
        """Drop a session's throttle state for the named reasons, so a
        recurrence after a genuine recovery is reported in full.

        REASONS ARE NAMED rather than clearing the session wholesale. The
        caller forgets the PRE-engage skips at the moment it gets past them —
        but "the engage itself failed" is a different condition that has not
        recovered, and clearing it there reset the throttle on every pass, so a
        permanently-failing engage logged every slice exactly as if it were
        unthrottled (found by the round-2 R8 test, which asserted the line
        count rather than the code path).
        """
        for key in [key for key in self._seen if key[0] == session_id and key[1] in reasons]:
            del self._seen[key]


#: Module-level because the throttle must outlive one sweep: every skip this
#: bounds is one that repeats on every pass for the life of the process.
_skip_log = _SkipLog()


def _schedule_due_times(entry: dict[str, Any]) -> list[int]:
    """Every readable ``next_due_at`` in an entry, in no particular order.

    Defensive like the rest of this module's index reading: the file is
    written by other processes, and a hand-edited or half-written row must
    cost one schedule, never the whole entry.
    """
    out: list[int] = []
    for raw in entry.get("schedules") or ():
        if not isinstance(raw, Mapping):
            continue
        due = raw.get("next_due_at")
        if isinstance(due, bool) or not isinstance(due, int):
            continue
        out.append(due)
    return out


def _is_stale_ms(due_ms: int, now_ms: int) -> bool:
    """Whether ONE schedule is past the staleness bound."""
    return (now_ms - due_ms) / 1000.0 > STALE_AFTER_S


def _fireable_due_ms(entry: dict[str, Any], now_ms: int) -> int | None:
    """Earliest schedule that is due NOW and not past the staleness bound.

    PER SCHEDULE, not per entry, and that distinction is the round-2 blocker
    (R7). An entry can hold a >7-day one-shot alongside a 20-minute recurring
    watch, and ``next_due_at(entry)`` answers with the EARLIEST of the two —
    so the whole entry read as stale, the live watch was never engaged, and
    once that predicate also drove retirement the supervisor exited on it.
    The one-shot only gets older, so nothing ever cleared the condition.
    """
    candidates = [
        due for due in _schedule_due_times(entry) if due <= now_ms and not _is_stale_ms(due, now_ms)
    ]
    return min(candidates) if candidates else None


def _has_fireable_schedule(entry: dict[str, Any], now_ms: int) -> bool:
    """Whether ANY schedule in this entry is still worth supervising.

    Future schedules count: they are the ordinary reason to stay up. Only an
    entry whose every schedule is past the staleness bound is work this
    process can never do (see :func:`_has_fireable_wakes`).
    """
    return any(not _is_stale_ms(due, now_ms) for due in _schedule_due_times(entry))


def _is_stale(entry: dict[str, Any], now_ms: int) -> bool:
    """Whether EVERY schedule in this entry is past the staleness bound.

    ``all``, not "the earliest one" — see :func:`_fireable_due_ms` for the
    defect that distinction caused. A mixed entry is emphatically NOT stale:
    it has work in it, and the stale sibling must not speak for the rest.
    """
    due_times = _schedule_due_times(entry)
    if not due_times:
        return False
    return all(_is_stale_ms(due, now_ms) for due in due_times)


def _due_sessions(index: dict[str, dict[str, Any]], now_ms: int) -> list[tuple[str, str, int]]:
    """``(session_id, cwd, due_ms)`` for every session with a wake due now.

    Reads the index defensively — it is written by other processes and a
    hand-edited or half-written entry must cost one session's wake, never the
    whole sweep.

    Every skip that drops a DUE wake logs, because a silent skip here is
    indistinguishable from a working supervisor: the two dead-ends below were
    invisible in the logs of a machine that had been missing wakes for a week.
    """
    from local_operator.wakes.store import next_due_at

    due: list[tuple[str, str, int]] = []
    for session_id, entry in index.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("stopped_at"):
            # Dormant: the session was deliberately stopped, and PR 3's
            # contract is that its wakes stay armed but do not fire until the
            # user reopens it. Firing here would resurrect a session the kill
            # switch ended.
            continue
        earliest = next_due_at(entry)
        if earliest is None or earliest > now_ms:
            continue
        # PER SCHEDULE (round 2, R7). `earliest` is the entry's soonest row,
        # which may be a week-old one-shot sitting beside a live recurring
        # watch; engaging on the entry's earliest meant the stale sibling
        # spoke for the whole entry and the live wake never fired. Ask instead
        # for the soonest schedule that is BOTH due and not stale.
        fireable = _fireable_due_ms(entry, now_ms)
        overdue_s = (now_ms - earliest) / 1000.0
        if fireable is None:
            # BEHAVIOUR UNCHANGED, SILENCE REMOVED. The session's own resume
            # catch-up handles arbitrarily-old overdue wakes, so racing to
            # start a runtime for a week-old one buys nothing. But skipping it
            # without a word is how a long-idle recurring watch can stop
            # firing forever with no trace: a wedged runtime starves the wake
            # past seven days, and from then on this branch drops it silently
            # on every pass. WARNING, not INFO — reaching this state means a
            # wake the user asked for is no longer being fired at all.
            # Throttled (round 1, R2/Q4): this condition never self-clears, so
            # an unthrottled line here is 8,640/day into an unrotated file.
            # The reason leads, because that is what an operator greps for.
            if _skip_log.should_log(session_id, "stale"):
                logger.warning(
                    "stale: skipping %s — its wake is %.1f days overdue (> the %.0f-day "
                    "staleness bound); it will be delivered by the session's own "
                    "catch-up when it is next opened",
                    session_id,
                    overdue_s / 86400.0,
                    STALE_AFTER_S / 86400.0,
                )
            continue
        cwd = entry.get("cwd")
        due.append(
            (session_id, cwd if isinstance(cwd, str) and cwd else os.path.expanduser("~"), fireable)
        )
    # Oldest first: if several are due at once, the one that has waited
    # longest gets its runtime first.
    due.sort(key=lambda row: row[2])
    return due


def _session_exists(config_dir: Path, session_id: str) -> bool:
    """Whether ``session_id`` has a transcript on disk.

    A GHOST GUARD. The index is derived from sessions, but nothing prunes an
    entry whose session directory has gone (a reap, a hand-deleted directory,
    or an id that never existed). Such an entry can never be engaged
    successfully, and before this guard each one consumed a full engage
    deadline on every pass, forever: the live log shows ``lr_bda7b76d34e0``
    and ``lr_28a800c6a783`` — ids with no session directory — accumulating
    30 s timeouts with zero successful starts, ever.

    The transcript rather than the directory, matching
    ``multiplexer/broadcast.py``: a directory with no transcript is not a
    session anything can resume.
    """
    from local_operator.resume import TRANSCRIPT_NAME

    try:
        return (Path(config_dir) / "sessions" / session_id / TRANSCRIPT_NAME).is_file()
    except OSError:
        # Unreadable is not proof of absence, and the cost of the two errors
        # is asymmetric: skipping a real session loses its wake, while trying
        # a ghost costs one deadline. Fail towards trying.
        logger.debug("could not check the transcript for %s", session_id, exc_info=True)
        return True


def _next_wake_ms(index: dict[str, dict[str, Any]]) -> int | None:
    """Earliest future due time across the whole index, or None."""
    from local_operator.wakes.store import next_due_at

    earliest: int | None = None
    for entry in index.values():
        if not isinstance(entry, dict) or entry.get("stopped_at"):
            continue
        due = next_due_at(entry)
        if due is None:
            continue
        if earliest is None or due < earliest:
            earliest = due
    return earliest


async def _has_live_runtime(config_dir: Path, session_id: str) -> bool:
    """Whether a runtime is already hosting ``session_id``.

    The no-live-record rule's implementation. Off the loop: it walks the
    record directory.
    """
    from local_operator.mobile.attach_client import find_runtime_record

    record, _owner = await asyncio.to_thread(find_runtime_record, config_dir, session_id)
    return record is not None


def wedged_runtime(config_dir: Path, session_id: str) -> tuple[int, float] | None:
    """``(pid, heartbeat_age_s)`` when a runtime for ``session_id`` is WEDGED.

    The last silent dead-end. A runtime whose process is alive but whose
    heartbeat has gone stale past ``HEARTBEAT_TIMEOUT_S`` is invisible to
    :func:`_has_live_runtime` (``find_runtime_record`` filters to ``live``, so
    the supervisor engages) *and* blocks that engage: ``_lease_holder`` sees
    the live pid holding the transcript lease and refuses to spawn, so every
    pass burns its whole deadline and the wake never fires while the wedge
    persists. Before this, the only trace was an ordinary timeout line,
    indistinguishable from a slow cold start.

    NOTHING HERE TOUCHES THE WEDGED SESSION. The lease is not broken and the
    process is not signalled: the safe repair belongs to the runtime's own
    watchdog, and a supervisor that killed a session's process on a heartbeat
    heuristic could destroy work a merely-slow runtime was in the middle of.

    It is NOT side-effect free, though, and round 2 (R10) was right that
    saying so was a false claim of purity. This delegates to
    ``registry.scan``, whose documented job includes reaping: it unlinks
    records whose pid is gone or whose file is torn, and ``run_dir()`` creates
    the directory if absent. That reaping is `scan`'s contract rather than
    something this function wants, and it only ever removes records for
    processes that are already dead — but it is a write, and a reader of this
    docstring must not be told otherwise. The classification is read from
    ``scan`` rather than re-derived from ``heartbeat_at`` here precisely so
    the registry stays the one owner of live/wedged/stale.
    """
    from local_operator.session.runtime.registry import scan

    try:
        records = scan(config_dir)
    except OSError:
        return None
    now = time.time()
    for record, state in records:
        if state == "wedged" and record.session_id == session_id:
            return record.pid, max(now - record.heartbeat_at, 0.0)
    return None


async def _engage_one(
    config_dir: Path,
    session_id: str,
    cwd: str,
    due_ms: int,
    moment: int,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Engage one due session. Returns whether a runtime was started.

    Split out so the sweep can run these concurrently; every skip and failure
    is contained here so one session can never fail the pass.
    """
    from local_operator.session.runtime.launch import WakeErrand, engage_runtime

    overdue_s = (moment - due_ms) / 1000.0
    async with semaphore:
        if await _has_live_runtime(config_dir, session_id):
            # The no-live-record rule: that session fires its own wakes.
            #
            # INFO with the overdue figure, not debug: a growing overdue
            # figure on a repeating "already running" line is what makes a
            # session that is up but not firing findable. Throttled, because
            # a long-lived session legitimately produces this every slice.
            if _skip_log.should_log(session_id, "live"):
                logger.info(
                    "live: skipping %s — its wake is %.1fs overdue but a live runtime "
                    "owns it (that session fires its own wakes)",
                    session_id,
                    overdue_s,
                )
            return False
        if not await asyncio.to_thread(_session_exists, config_dir, session_id):
            if _skip_log.should_log(session_id, "ghost"):
                logger.warning(
                    "ghost: skipping %s — its wake is %.1fs overdue but the session has "
                    "no transcript on disk; the index entry can never be engaged",
                    session_id,
                    overdue_s,
                )
            return False
        # Probed BEFORE the engage, not after it fails, because the engage
        # would otherwise burn its whole deadline against a lease this process
        # cannot take — and report that as an ordinary timeout.
        wedge = await asyncio.to_thread(wedged_runtime, config_dir, session_id)
        if wedge is not None:
            pid, age_s = wedge
            if _skip_log.should_log(session_id, "wedged"):
                logger.warning(
                    "wedged: skipping %s — a runtime exists (pid %d) but its heartbeat is "
                    "%.0fs stale, so it holds the transcript lease without serving; the "
                    "wake is %.1fs overdue and cannot fire until that process recovers "
                    "or is stopped",
                    session_id,
                    pid,
                    age_s,
                    overdue_s,
                )
            return False
        # Only the conditions we just cleared: reaching here means the session
        # is no longer skipped for a live/ghost/wedged/stale reason. Whether
        # the engage SUCCEEDS is decided below, and its own throttle key must
        # survive this pass — see `_SkipLog.forget`.
        _skip_log.forget(session_id, "live", "ghost", "wedged", "stale")
        logger.info(
            "engaging %s: wake due %d, %.1fs overdue (deadline %.0fs)",
            session_id,
            due_ms,
            overdue_s,
            WAKE_DEADLINE_S,
        )
        started = time.monotonic()
        try:
            # The command_id is DERIVED rather than random, but note what it
            # does and does not buy. It is NOT what dedupes a wake: `_deliver`
            # returns early for a WakeErrand (`session/runtime/launch.py`)
            # without ever sending an op, so no command_id reaches a runtime.
            # What actually keeps a retry from double-firing is record reuse
            # plus lease arbitration — an engage that finds a live record uses
            # it, and the transcript lease admits at most one SERVING runtime.
            # The id is carried for the log line and for consistency with the
            # other errands.
            await engage_runtime(
                session_id,
                cwd,
                WakeErrand(
                    schedule_id="",
                    occurrence_ms=due_ms,
                    command_id=f"wake-{session_id}-{due_ms}",
                ),
                config_dir=config_dir,
                deadline_s=WAKE_DEADLINE_S,
            )
        except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
            # One session's wake failing must not stop the others'. The
            # schedule is untouched, so the next pass tries again.
            #
            # RUNTIMEERROR IS DOCUMENTED, not defensive: `engage_runtime`
            # raises it carrying the child's own reason as soon as every
            # candidate it may start has died (a missing credential, an
            # unconstructable session). Round 2 (R8) — omitting it here sent
            # a 13-line asyncio "Task exception was never retrieved" traceback
            # into the unrotated log on every slice, ~8,640/day, bypassing the
            # throttle that the rest of this file routes repeats through.
            #
            # THROTTLED like every other repeating failure: an unconstructable
            # session fails identically on every pass, so it gets the same
            # burst-then-heartbeat treatment as the stale and ghost skips.
            if _skip_log.should_log(session_id, "failed"):
                logger.warning(
                    "failed: could not start a runtime for %s after %.1fs "
                    "(wake %.1fs overdue): %s",
                    session_id,
                    time.monotonic() - started,
                    overdue_s,
                    exc,
                )
            return False
        # RECOVERED: the failure keys are cleared only once an engage actually
        # succeeds, so the next failure after a working run is announced at
        # full volume instead of inheriting the old silence.
        _skip_log.forget(session_id, "failed", "error")
        logger.info(
            "started a runtime for %s (wake due %d, %.1fs overdue, took %.1fs)",
            session_id,
            due_ms,
            overdue_s,
            time.monotonic() - started,
        )
        return True


class _Sweeper:
    """Owns the in-flight engagements so the serve loop never waits on one.

    **Why this is not a ``gather``.** Awaiting the sweep to completion put the
    head-of-line blocking back where the slice re-read had just removed it:
    with concurrency 2 and a 180 s deadline, six all-timing-out sessions block
    for ``ceil(6/2) * 180 = 540 s``, three times worse than the serial 30 s
    code this replaced (round 1, R3), and nothing re-reads the index for the
    whole of it. Engagements are therefore *started* by a pass and awaited by
    nobody: the loop returns to slicing immediately, so a wake armed during a
    long sweep is still seen within one slice.

    **The in-flight set is what makes that safe.** Without it, a pass every
    10 s over a wake that takes 180 s to engage would start eighteen
    engagements for one occurrence. Keyed by ``(session_id, occurrence_ms)``
    rather than by session, so a *later* occurrence of a recurring wake is
    still engageable while an earlier one is somehow still in flight, and the
    same occurrence never is.
    """

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_ENGAGES)
        self._in_flight: dict[tuple[str, int], asyncio.Task[bool]] = {}
        #: Cumulative count of engagements that reported a started runtime.
        #: The loop has nothing to return any more, so the count lives here
        #: for tests and for `--once`.
        self.started = 0

    def engage(self, config_dir: Path, session_id: str, cwd: str, due_ms: int, moment: int) -> bool:
        """Start one engagement in the background. False when already running."""
        key = (session_id, due_ms)
        if key in self._in_flight:
            return False

        async def _run() -> bool:
            try:
                started = await _engage_one(
                    config_dir, session_id, cwd, due_ms, moment, self._semaphore
                )
            except asyncio.CancelledError:
                # Shutdown, not a failure: `shutdown()` cancels in-flight work
                # deliberately. Re-raised so the task settles as cancelled.
                raise
            except Exception as exc:  # noqa: BLE001 — a detached task's last resort
                # THE BELT (round 2, R8). `_engage_one` runs as a detached
                # task, so ANY raise it does not handle lands here rather than
                # at a caller — and asyncio's default handler then prints a
                # 13-line traceback per occurrence, at slice cadence, with no
                # session id and none of the `reason:`-leading format the rest
                # of this file uses. `_engage_one` already catches what
                # `engage_runtime` documents; this exists so an UNANTICIPATED
                # raise degrades to one throttled line and a retried wake
                # instead of a log flood.
                if _skip_log.should_log(session_id, "error"):
                    logger.warning(
                        "error: engaging %s raised %s: %s",
                        session_id,
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )
                return False
            finally:
                # Popped in the task itself rather than in a done-callback so
                # the key is gone the moment the work is, with no window where
                # a finished engagement still blocks its own retry.
                self._in_flight.pop(key, None)
            if started:
                self.started += 1
            return started

        self._in_flight[key] = asyncio.create_task(_run())
        return True

    @property
    def in_flight(self) -> int:
        """How many engagements are running. The loop must not retire on 0
        fireable wakes while this is non-zero: the engagement it would abandon
        is the one constructing the runtime that will advance the schedule."""
        return len(self._in_flight)

    async def drain(self) -> int:
        """Await every in-flight engagement. For ``--once`` and for shutdown."""
        while self._in_flight:
            await asyncio.gather(*list(self._in_flight.values()), return_exceptions=True)
        return self.started

    async def shutdown(self) -> None:
        """Cancel and reap every in-flight engagement.

        Cancelling rather than draining: this runs when the loop is already
        ending (retirement, or the process being torn down), and an engage can
        legitimately still have minutes of deadline left. The runtime a
        cancelled engage spawned is detached (``start_new_session=True``) and
        keeps constructing regardless — what is abandoned is the WAITING, not
        the wake, which is the same trade the deadline itself makes.
        """
        tasks = list(self._in_flight.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._in_flight.clear()

    def sweep(self, config_dir: Path, index: dict[str, dict[str, Any]], moment: int) -> int:
        """Start an engagement for every due session. Returns how many started."""
        launched = 0
        for session_id, cwd, due_ms in _due_sessions(index, moment):
            if self.engage(config_dir, session_id, cwd, due_ms, moment):
                launched += 1
        return launched


async def fire_due_wakes(config_dir: Path, *, now_ms: int | None = None) -> int:
    """Start a runtime for every cold session whose wake is due. Returns count.

    One pass, awaited to completion — this is the ``--once`` / diagnostic
    form and the seam the unit tests drive. The serve loop does NOT use it;
    it drives a :class:`_Sweeper` directly so a slow engagement cannot hold
    the loop (see that class for why).
    """
    from local_operator.wakes.store import read_index

    moment = now_ms if now_ms is not None else int(time.time() * 1000)
    index = await asyncio.to_thread(read_index, config_dir)
    sweeper = _Sweeper()
    sweeper.sweep(config_dir, index, moment)
    return await sweeper.drain()


def _has_fireable_wakes(
    index: dict[str, dict[str, Any]],
    *,
    config_dir: Path | None = None,
    now_ms: int | None = None,
) -> bool:
    """Whether anything in the index could ever cause THIS process to fire.

    Not the same question as "is the index non-empty", and the difference was
    a leak: an index holding only DORMANT entries (every session stopped) kept
    the supervisor alive forever, because the non-empty check refused to retire
    while ``_next_wake_ms`` \u2014 which skips dormant entries \u2014 returned None and
    parked it on ``MAX_SLEEP_S``. Nothing it could do would ever fire one of
    those wakes; reopening the session is what re-arms them, and that reopen
    persists, which calls the install hook and brings the supervisor back.

    Round 1 (R2/Q4) found two more entries with the same shape, and they are
    why this now takes a ``config_dir``:

    - **Stale** (>7 days overdue): the sweep refuses these permanently, and the
      condition cannot self-clear because a wake only gets older. The session's
      own catch-up delivers it on next open.
    - **Ghost** (an index entry whose session has no transcript): refused
      permanently too, and a ghost never grows a transcript.

    Both satisfied the old "any non-dormant entry with schedules" test, so the
    process was immortal AND re-logged the refusal every slice forever. None of
    the three is work this process can do, and each has its own route back, so
    retirement is the right answer for all of them.

    ``config_dir`` is optional so the pure index-shape callers need not have
    one; the ghost check is then skipped, which errs towards STAYING UP rather
    than retiring on a wake that might be real.
    """
    moment = now_ms if now_ms is not None else int(time.time() * 1000)
    for session_id, entry in index.items():
        if not isinstance(entry, dict) or entry.get("stopped_at"):
            continue
        if not entry.get("schedules"):
            continue
        # PER SCHEDULE (round 2, R7). Asking whether the entry's EARLIEST wake
        # is stale retired the process on an entry that also held a live
        # recurring watch, and `KeepAlive{SuccessfulExit:false}` then kept it
        # down while `StartInterval` re-decided the same thing every 15 min —
        # the live watch stopped firing for good, silently. One non-stale
        # schedule anywhere in the entry is work worth staying up for.
        if not _has_fireable_schedule(entry, moment):
            continue
        if config_dir is not None and not _session_exists(config_dir, session_id):
            continue
        return True
    return False


def _retirement_reason(config_dir: Path) -> str:
    """Why nothing is fireable, in the terms the operator can act on.

    Round 2 (D15). "The next persist reinstalls it" was the only explanation
    the retirement line carried, and for a stale-only store it is misleading:
    the reinstalled supervisor reads the same entry and retires again. The
    states that CAN end in retirement each have a different route back
    (reopen the session, open it for the catch-up, rewrite the index), so the
    line names which one it hit — this is the sentence ``lop wake status``
    already knows how to say.
    """
    from local_operator.wakes.store import read_index

    now_ms = int(time.time() * 1000)
    dormant = stale = ghost = 0
    for session_id, entry in read_index(config_dir).items():
        if not isinstance(entry, dict) or not entry.get("schedules"):
            continue
        if entry.get("stopped_at"):
            dormant += 1
        elif not _has_fireable_schedule(entry, now_ms):
            stale += 1
        elif not _session_exists(config_dir, session_id):
            ghost += 1
    parts: list[str] = []
    if stale:
        parts.append(
            f"{stale} past the {STALE_AFTER_S / 86400.0:.0f}d staleness bound — "
            "delivered when their sessions are next opened"
        )
    if dormant:
        parts.append(f"{dormant} dormant — reopening the session re-arms them")
    if ghost:
        parts.append(f"{ghost} with no session on disk")
    if not parts:
        return "nothing is scheduled; the next persist reinstalls it"
    return "; ".join(parts)


async def _should_retire(config_dir: Path) -> bool:
    """Whether there is genuinely nothing left to supervise.

    Re-reads after a short grace, which closes the RACE half of the
    never-restarted-supervisor defect. ``Session._persist_wake_schedules``
    writes the index entry and THEN calls the install hook, so there is a
    window where the supervisor can read an empty index while a session is
    part-way through arming its first wake. Retiring inside that window exits
    0, the hook that follows sees a job launchd still knows about, and the
    wake is left armed with nothing running — the same end state as the hard
    miss, reached by a different road.

    The grace is one slice: long enough to cover the gap between those two
    steps (they are consecutive statements), short enough that retirement on a
    genuinely empty index is still prompt.
    """
    from local_operator.wakes.store import read_index

    await asyncio.sleep(min(SLICE_S, MAX_SLEEP_S))
    index = await asyncio.to_thread(read_index, config_dir)
    return not _has_fireable_wakes(index, config_dir=config_dir)


async def serve(config_dir: Path, *, once: bool = False) -> int:
    """Run the supervisor loop. Returns the process exit code.

    Exits **0** when there is nothing fireable left, which is what lets the
    LaunchAgent's ``KeepAlive: {SuccessfulExit: False}`` leave a finished
    supervisor down instead of restarting it forever. A crash exits non-zero
    and is restarted.

    **The wait is sliced** (:data:`SLICE_S`) rather than computed once and
    slept whole: a wake persisted while the supervisor sleeps is then seen
    within a slice instead of within up to ``MAX_SLEEP_S``.

    **Engagements do not hold this loop.** A pass STARTS the engagements it
    owes and returns; the loop keeps slicing while they run (see
    :class:`_Sweeper`). Awaiting them here is what made a 6-session
    all-timeout sweep block for 540 s in round 1.
    """
    from local_operator.wakes.store import read_index

    sweeper = _Sweeper()
    try:
        while True:
            index = await asyncio.to_thread(read_index, config_dir)
            if not _has_fireable_wakes(index, config_dir=config_dir):
                if once:
                    # `--once` IS THE DIAGNOSTIC, so it must not be the quiet
                    # one (round 2, D15): a stale-only store made `lop wake
                    # serve --once` print nothing and exit 0, which reads as
                    # "everything is fine" in exactly the state where a wake
                    # the user asked for is never going to fire. Same sentence
                    # the loop logs when it retires for the same reason.
                    logger.info(
                        "nothing fireable in this store (%s)", _retirement_reason(config_dir)
                    )
                    return 0
                # In-flight work outlives a momentarily-empty index: retiring
                # under a running engagement would kill the runtime it is
                # constructing. Slice instead and re-decide next pass.
                if sweeper.in_flight:
                    await _sleep_in_slices(config_dir, SLICE_S, None)
                    continue
                if await _should_retire(config_dir):
                    # LOGGED WITH THE REASON, because this is the transition
                    # that ends the process: a supervisor that is gone looks
                    # identical to one that never started unless it says why it
                    # left. Round 2 (D15): "the next persist reinstalls it" is
                    # true and useless for a stale-only store — the reinstalled
                    # supervisor retires again on the same entry — and the
                    # `stale: skipping …` line that would have named the
                    # condition is unreachable, because retirement is decided
                    # before the sweep runs. So the retirement line carries the
                    # breakdown itself.
                    logger.info(
                        "no fireable wakes remain after a %.0fs grace re-read; "
                        "the supervisor is retiring (%s)",
                        SLICE_S,
                        _retirement_reason(config_dir),
                    )
                    return 0
                logger.info("a wake was armed during the retirement grace; staying up")
                continue

            sweeper.sweep(config_dir, index, int(time.time() * 1000))
            if once:
                await sweeper.drain()
                return 0

            # Recomputed from the index AFTER the sweep started, so a schedule
            # an already-finished runtime advanced is reflected rather than
            # re-read stale.
            index = await asyncio.to_thread(read_index, config_dir)
            if not _has_fireable_wakes(index, config_dir=config_dir):
                continue  # retirement is decided at the top, with its grace

            upcoming = _next_wake_ms(index)
            now_ms = int(time.time() * 1000)
            if upcoming is None:
                delay = MAX_SLEEP_S
            elif upcoming <= now_ms:
                # STILL DUE AFTER A SWEEP means one of two things: the wake is
                # being engaged right now (the engagement is in flight and the
                # runtime has not yet advanced the schedule), or it was
                # SKIPPED. A fired wake does not land here — the runtime
                # advances the schedule, so the next read shows a future time.
                #
                # Either way a slice is the right cadence, and MIN_SLEEP_S is
                # not: the condition can only change when an engagement
                # finishes or something else writes the index, and that is
                # exactly what the slice re-read is sized to notice. Sleeping
                # MIN_SLEEP_S here span the loop at one pass per second and
                # re-logged the same skip with it (observed against a ghost
                # entry during validation).
                delay = SLICE_S
            else:
                delay = max(MIN_SLEEP_S, min(MAX_SLEEP_S, (upcoming - now_ms) / 1000.0))
            logger.debug("sleeping %.1fs until the next wake (in %.0fs slices)", delay, SLICE_S)
            await _sleep_in_slices(config_dir, delay, upcoming)
    finally:
        # A cancelled or retiring supervisor must not leave engage tasks
        # pending: they hold a semaphore slot and an event loop reference, and
        # an un-awaited task destroyed at loop close logs a spurious error.
        await sweeper.shutdown()


async def _sleep_in_slices(config_dir: Path, delay: float, upcoming: int | None) -> None:
    """Sleep up to ``delay`` seconds, returning early if the index changes.

    Re-reads the index once per :data:`SLICE_S` and returns as soon as a wake
    earlier than the one we were waiting for appears (or the index empties),
    so the next loop iteration fires it. MIN/MAX semantics are the caller's
    and are unchanged: this only decides how the already-computed ``delay`` is
    spent.
    """
    from local_operator.wakes.store import read_index

    remaining = delay
    while remaining > 0:
        await asyncio.sleep(min(SLICE_S, remaining))
        remaining -= SLICE_S
        if remaining <= 0:
            return
        index = await asyncio.to_thread(read_index, config_dir)
        if not _has_fireable_wakes(index, config_dir=config_dir):
            return  # retirement (with its grace) is decided by the caller's loop
        earliest = _next_wake_ms(index)
        if earliest is None:
            return
        if upcoming is None or earliest < upcoming or earliest <= int(time.time() * 1000):
            # Something was scheduled EARLIER than what this sleep was sized
            # for, or is now due. Returning lets the loop re-read and fire it
            # within a slice rather than at the end of a wait sized for a
            # wake that is no longer the next one.
            logger.info(
                "an earlier wake appeared while sleeping; re-reading now instead of "
                "waiting a further %.0fs",
                max(remaining, 0.0),
            )
            return


def main(argv: list[str] | None = None) -> int:
    """``python -m local_operator.wakes.supervisor`` — the LaunchAgent entry."""
    parser = argparse.ArgumentParser(
        prog="local-operator-wake-supervisor",
        description="Fire scheduled wakes for sessions that are not running.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run a single pass and exit (diagnostics; the agent runs the loop)",
    )
    args = parser.parse_args(argv)

    # The SAME console configuration `lop wake serve` gets, so one log line
    # reads identically whichever entry point produced it. Round 1 (D8): this
    # file is what an operator is told to read, and the same skip rendered two
    # ways — `WARNING __main__:` from the LaunchAgent, `- WARNING -` from the
    # CLI — because this entry rolled its own `basicConfig` with `%(name)s`.
    # Run as a script, that name is always the useless `__main__`; the module
    # logger below is given a stable name instead (see `logger`).
    from local_operator.logger import configure_cli_logging

    configure_cli_logging()

    # launchd hands this process a bare ``PATH=/usr/bin:/bin:/usr/sbin:/sbin``
    # — the LaunchAgent sets ``LOCAL_OPERATOR_CONFIG_DIR`` and nothing else,
    # deliberately (a PATH baked into the plist render would vary per calling
    # session and turn ``ensure_supervisor_installed``'s content comparison
    # into a restart loop). That bare PATH does not stop here: every runtime
    # this supervisor engages is spawned with ``dict(os.environ)``
    # (``session/runtime/launch.py``), so a wake-driven turn inherits it and
    # loses kubectl, Homebrew, nvm, cargo, pyenv and Nix. Interactive launches
    # never showed it because a terminal already supplies a login PATH, which
    # is exactly why this was the one process entry point that skipped the
    # bootstrap every other entry (``cli.main``, ``server.app``) performs.
    #
    # Imported function-locally on purpose: importing ``helpers`` pulls in 53
    # further modules and ~11.5-12.2 ms (measured on this commit, three runs),
    # which a process whose whole justification is staying light must not pay
    # at import time, and ``wakes/`` keeps a deliberately thin import graph
    # (``tests/unit/test_import_graph.py`` pins ``wakes.store`` to stdlib-only;
    # it does not constrain this module, but the intent covers it).
    from local_operator.helpers import setup_cross_platform_environment

    setup_cross_platform_environment()

    from local_operator.paths import config_dir

    try:
        return asyncio.run(serve(config_dir(), once=args.once))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
