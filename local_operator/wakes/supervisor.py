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
from typing import Any

logger = logging.getLogger(__name__)

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
#: tight bound: the only thing a longer deadline costs is a semaphore slot
#: held by a session nobody is waiting for, and :func:`serve` keeps re-reading
#: the index on its slice throughout regardless.
WAKE_DEADLINE_S = 180.0

#: How many sessions the sweep engages at once. See :func:`fire_due_wakes`.
_MAX_CONCURRENT_ENGAGES = 2


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
        overdue_s = (now_ms - earliest) / 1000.0
        if overdue_s > STALE_AFTER_S:
            # BEHAVIOUR UNCHANGED, SILENCE REMOVED. The session's own resume
            # catch-up handles arbitrarily-old overdue wakes, so racing to
            # start a runtime for a week-old one buys nothing. But skipping it
            # without a word is how a long-idle recurring watch can stop
            # firing forever with no trace: a wedged runtime starves the wake
            # past seven days, and from then on this branch drops it silently
            # on every pass. WARNING, not INFO — reaching this state means a
            # wake the user asked for is no longer being fired at all.
            logger.warning(
                "skipping %s: its wake is %.1f days overdue (> the %.0f-day staleness "
                "bound); it will be delivered by the session's own catch-up when it is "
                "next opened",
                session_id,
                overdue_s / 86400.0,
                STALE_AFTER_S / 86400.0,
            )
            continue
        cwd = entry.get("cwd")
        due.append(
            (session_id, cwd if isinstance(cwd, str) and cwd else os.path.expanduser("~"), earliest)
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
            # INFO with the overdue figure, not debug. A WEDGED runtime (one
            # whose record is live but whose event loop is stuck) looks
            # exactly like a healthy one here, so the supervisor skips it on
            # every pass while its wake never fires. That case was invisible
            # at the default log level; a growing overdue figure on a
            # repeating "already running" line is what makes it findable.
            logger.info(
                "skipping %s: its wake is %.1fs overdue but a live runtime owns it "
                "(that session fires its own wakes)",
                session_id,
                overdue_s,
            )
            return False
        if not await asyncio.to_thread(_session_exists, config_dir, session_id):
            logger.warning(
                "skipping %s: its wake is %.1fs overdue but the session has no transcript "
                "on disk; the index entry is a ghost and can never be engaged",
                session_id,
                overdue_s,
            )
            return False
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
        except (TimeoutError, ConnectionError, OSError) as exc:
            # One session's wake failing must not stop the others'. The
            # schedule is untouched, so the next pass tries again.
            logger.warning(
                "could not start a runtime for %s after %.1fs (wake %.1fs overdue): %s",
                session_id,
                time.monotonic() - started,
                overdue_s,
                exc,
            )
            return False
        logger.info(
            "started a runtime for %s (wake due %d, %.1fs overdue, took %.1fs)",
            session_id,
            due_ms,
            overdue_s,
            time.monotonic() - started,
        )
        return True


async def fire_due_wakes(config_dir: Path, *, now_ms: int | None = None) -> int:
    """Start a runtime for every cold session whose wake is due. Returns count.

    One pass. Split out from :func:`serve` so the whole decision — which
    sessions are due, which are skipped for being live, what engaging does —
    is testable without a loop or a clock.

    **Concurrent, bounded.** This used to be a serial ``for`` loop awaiting
    each engage in turn, so a session waited behind every earlier session's
    full deadline: with two cold sessions due at once the pass took ~63 s
    (measured), and lateness grew with the number of cold sessions. Since a
    wake errand's deadline is now sized for a cold start, serial would be
    strictly worse — which is why the slice re-read and this ship together.
    """
    from local_operator.wakes.store import read_index

    moment = now_ms if now_ms is not None else int(time.time() * 1000)
    index = await asyncio.to_thread(read_index, config_dir)
    rows = _due_sessions(index, moment)
    if not rows:
        return 0
    # BOUNDED, not unbounded gather. Each engage may cold-start up to
    # `_MAX_SPAWNS` runtimes, and a runtime is a full harness process; this
    # box routinely runs ~19 sessions, so an unbounded fan-out over a backlog
    # would be a thundering herd against the very resource contention that
    # makes cold starts slow in the first place. Two at a time removes the
    # serial head-of-line blocking (the actual defect) without becoming a new
    # source of load.
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_ENGAGES)
    results = await asyncio.gather(
        *(
            _engage_one(config_dir, session_id, cwd, due_ms, moment, semaphore)
            for session_id, cwd, due_ms in rows
        )
    )
    return sum(1 for started in results if started)


def _has_fireable_wakes(index: dict[str, dict[str, Any]]) -> bool:
    """Whether anything in the index could ever cause a fire.

    Not the same question as "is the index non-empty", and the difference was
    a leak: an index holding only DORMANT entries (every session stopped) kept
    the supervisor alive forever, because the non-empty check refused to retire
    while ``_next_wake_ms`` \u2014 which skips dormant entries \u2014 returned None and
    parked it on ``MAX_SLEEP_S``. Nothing it could do would ever fire one of
    those wakes; reopening the session is what re-arms them, and that reopen
    persists, which calls the install hook and brings the supervisor back.
    """
    for entry in index.values():
        if not isinstance(entry, dict) or entry.get("stopped_at"):
            continue
        if entry.get("schedules"):
            return True
    return False


async def _should_retire(config_dir: Path) -> bool:
    """Whether there is genuinely nothing left to supervise.

    Re-reads after a short grace, which closes the RACE half of the
    never-restarted-supervisor defect. ``Session._persist_wake_schedules``
    writes the index entry and THEN calls the install hook, so there is a
    window where the supervisor can read an empty index while a session is
    part-way through arming its first wake. Retiring inside that window exits
    0, the hook that follows sees a job launchd still knows about, and the
    wake is left armed with nothing running \u2014 the same end state as the hard
    miss, reached by a different road.

    The grace is one slice: long enough to cover the gap between those two
    steps (they are consecutive statements), short enough that retirement on a
    genuinely empty index is still prompt.
    """
    from local_operator.wakes.store import read_index

    await asyncio.sleep(min(SLICE_S, MAX_SLEEP_S))
    index = await asyncio.to_thread(read_index, config_dir)
    return not _has_fireable_wakes(index)


async def serve(config_dir: Path, *, once: bool = False) -> int:
    """Run the supervisor loop. Returns the process exit code.

    Exits **0** when there is nothing fireable left, which is what lets the
    LaunchAgent's ``KeepAlive: {SuccessfulExit: False}`` leave a finished
    supervisor down instead of restarting it forever. A crash exits non-zero
    and is restarted.

    **The wait is sliced** (:data:`SLICE_S`) rather than computed once and
    slept whole: a wake persisted while the supervisor sleeps is then seen
    within a slice instead of within up to ``MAX_SLEEP_S``.
    """
    from local_operator.wakes.store import read_index

    while True:
        index = await asyncio.to_thread(read_index, config_dir)
        if not _has_fireable_wakes(index):
            if once:
                return 0
            if await _should_retire(config_dir):
                # LOGGED, because this is the transition that ends the
                # process: a supervisor that is gone looks identical to one
                # that never started unless it says why it left.
                logger.info(
                    "no fireable wakes remain after a %.0fs grace re-read; "
                    "the supervisor is retiring (the next persist reinstalls it)",
                    SLICE_S,
                )
                return 0
            logger.info("a wake was armed during the retirement grace; staying up")
            continue

        await fire_due_wakes(config_dir)
        if once:
            return 0

        # Recomputed from the index AFTER firing, so a schedule the fired
        # runtime advanced is already reflected rather than re-read stale.
        index = await asyncio.to_thread(read_index, config_dir)
        if not _has_fireable_wakes(index):
            continue  # retirement is decided at the top, with its grace

        upcoming = _next_wake_ms(index)
        now_ms = int(time.time() * 1000)
        if upcoming is None:
            delay = MAX_SLEEP_S
        elif upcoming <= now_ms:
            # STILL DUE AFTER A SWEEP means it was SKIPPED, not missed: a live
            # session owns it, the entry is a ghost, or it is past the
            # staleness bound. A fired wake does not land here — the runtime
            # advances the schedule, so the next read shows a future time.
            #
            # Sleeping `MIN_SLEEP_S` on those was a hot loop: once the skip
            # became cheap (the ghost guard answers from a stat instead of
            # burning a 30 s engage) the loop span at one pass per second,
            # re-logging the same skip forever. Observed doing exactly that
            # against a ghost entry during validation. A slice is the right
            # cadence: the condition can only change when something else
            # writes the index or a runtime exits, and that is what the slice
            # re-read is already sized to notice.
            delay = SLICE_S
        else:
            delay = max(MIN_SLEEP_S, min(MAX_SLEEP_S, (upcoming - now_ms) / 1000.0))
        logger.debug("sleeping %.1fs until the next wake (in %.0fs slices)", delay, SLICE_S)
        await _sleep_in_slices(config_dir, delay, upcoming)


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
        if not _has_fireable_wakes(index):
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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
