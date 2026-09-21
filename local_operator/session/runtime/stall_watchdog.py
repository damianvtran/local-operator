"""Bound a runtime's OWN stall: dump every thread, then leave.

**WHY THIS EXISTS, in one measurement.** On 2026-09-20 five of this machine's
session runtimes were found frozen at once, for 1.5 to 7.2 hours each, and each
had to be reaped by hand with ``lop stop --force``. ``sample <pid> 3`` on every
one of them put 100% of the event-loop main thread inside
``_sre_SRE_Pattern_search`` -> ``sre_search`` -> ``sre_ucs1_match`` — a bare
``.search()`` — while CPU kept advancing at ~0.9 core and the transcript took
zero writes. A freeze of hours is not a linear cost: the repo's own
``scrub_secrets`` walks those same transcripts in 1.75-2.07 s, so the scan in
flight was not converging on its input — and **the caller was never identified,
because nothing in the process could say which line it was stuck on**. The one
instrument that would have
(``LOP_RUNTIME_DEBUG_STACKS``' SIGUSR1 asyncio dump, ``process.amain``) was not
switched on. There is no automatic recovery path either: ``lop sessions
reclaim`` refused all five (``record-present``) and ``session.cleanup.enabled``
is off.

So a runtime now bounds its own stall. The operator's directive, verbatim: "Yes
force stop it then continue with instructions to not repeat that."

WHY A C TIMER AND NOTHING IN PYTHON
-----------------------------------
A ``threading.Thread`` watchdog, an ``asyncio`` timer and a signal handler all
fail here, and ``tests/e2e/watchdog.py`` documents the measurements: a hung
thread never releases the GIL, so a thread watchdog cannot run; a Python signal
handler runs only between bytecodes, so ``pytest-timeout``'s default method
never fires; and a loop parked in a C call never schedules the coroutine that
would report the sample. (The same GIL is why the SIGUSR1 dump above cannot
answer this class of question: its handler is an asyncio signal handler and
needs the loop that is blocked.)

``faulthandler.dump_traceback_later`` is the one mechanism that survives it. It
arms a timer in a dedicated **C** thread that writes every thread's stack with
``write(2)`` straight to a file descriptor and then calls ``_exit(1)`` — no GIL,
no Python frames and no interpreter state required.

WHAT "STALL" MEANS HERE: NO PROGRESS, NOT SLOW WORK
---------------------------------------------------
The bound is not a stopwatch on a turn. The timer is armed once and **re-armed
by the runtime's own loops as they run** (``beat``), so it measures the gap
since the last sign of life *inside this process* — the same signal the record's
heartbeat publishes. A turn that awaits for ten minutes keeps beating; a loop
parked in one C call does not.

EVERY PLANE'S PROGRESS IS TRACKED SEPARATELY, AND EITHER ONE GOING SILENT TRIPS
IT. A ``daemon``/``exec`` runtime runs its workload and its serving plane on
separate threads (``RuntimeServer.start``), so "this process is alive" is NOT the
question — the measured failure was the WORKLOAD loop parked in a synchronous
scan for hours while everything else about the process was fine, and a bound
that a healthy serving plane can keep re-arming is worse than no bound at all:
it makes the fleet look protected. Each plane therefore carries its own
last-seen stamp (:data:`WORKLOAD`, ticked by ``process._beat_stall_watchdog``;
:data:`SERVING`, ticked by ``RuntimeServer._heartbeat_loop`` — both on a
15 s cadence, see ``types.HEARTBEAT_INTERVAL_S``), and whichever tick runs next
re-arms the single process-wide C timer for the EARLIEST remaining deadline:
``min(stamp + bound) - now``. A healthy plane therefore cannot mask a silent
one — its own tick shortens the timer to the other plane's deadline rather than
pushing it out — and with BOTH planes silent no tick runs at all, so the timer
simply fires at the deadline the last tick set.

WHAT THIS BOUND MEANS, STATED PLAINLY, BECAUSE IT IS STRICTER THAN "WEDGED":
a synchronous step is indistinguishable from a wedge from outside the process,
so the bound is also a CEILING ON ONE SILENT SYNCHRONOUS STEP: a step that takes
tens of seconds passes, one that never returns is cut. What the bound
measures is the LOOPS RUNNING, never the work advancing: a plane that keeps
ticking while its work stands still is outside this design (nothing here can
see that, and inventing a second, footprint-based clock is what
``process._work_motion`` already does for the drain).

WHY THE EXIT IS THE BLUNT ONE, AND WHAT IT COSTS
------------------------------------------------
Past the bound the runtime must stop being a multi-hour freeze, and the graceful
rungs cannot be reached from the state being detected: ``_drain_for_signal``,
``_commit_to_leaving`` and ``_leave_overdue`` are all coroutines on the loop that
is blocked, and the ``SIGTERM`` handler that reaches them is a loop handler that
needs bytecodes the stuck thread never executes. That is the same fact that
makes ``lop stop`` refuse a silent-socket runtime without ``--force``. So the
timer is armed with ``exit=True``: faulthandler writes the dump and calls
``_exit(1)`` from its own C thread. Write-then-act is therefore structural
rather than something this module has to sequence.

WHAT IS LOST: the in-flight turn's uncommitted step — the same loss a SIGKILL
inflicts, because a turn commits its transcript at each step and the step in
flight has not committed. What SURVIVES: everything already committed, i.e. the
conversation, which a successor can be engaged on. What IS LEFT BEHIND: the
record, whose pid is then gone — ``registry.classify`` already reads that as
``stale`` and ``reclaim`` already sweeps it, so no new vocabulary is needed. The
one asymmetry worth stating in the PR: until this ships, the automatic paths
refuse a loop-blocked runtime (``record-present``) and the only working recovery
is a hand-run ``lop stop --pid <pid> --force``.

A HARD EXIT ALSO SKIPS THE IN-PROCESS KILL OF THIS TURN'S TOOL PROCESS GROUPS,
and that is covered rather than merely accepted: ``_exit`` from the C thread runs
no Python, so ``execute_bash``'s ``_kill`` chain cannot fire — which is exactly
the "hard death of the owning ``lop`` process" class
``local_operator/tools/group_reaper.py`` exists for. Each group is registered
with a liveness marker at spawn and :func:`sweep_orphan_groups` reaps the ones
whose owner is provably dead, at the NEXT ``lop`` startup. So a child orphaned
here is bounded by the next session start rather than by this process's death,
which is the same guarantee today's only recovery (SIGKILL) already relies on —
and a stray group is visible in the meantime exactly as it would be after any
hard death.

THE FILE IS THE EVIDENCE, AND ITS ABSENCE MEANS NOTHING FIRED
------------------------------------------------------------
The dump is written to ``<log dir>/runtime-stall-<pid>.log`` — beside
``runtime.log``, in the directory ``paths.log_dir`` already resolves and
``lop mobile logs`` already points a reader at. Its EXISTENCE carries one signal:
a clean exit (``disarm``) cancels the timer and REMOVES the file, so a header-only
file that survives means the process died without disarming (a SIGKILL, an OOM
kill, or this module's ``exit=True`` firing), and a file carrying
:data:`FIRED_MARKER` means this bound actually fired. That is the shape
``tests/e2e/watchdog.py`` uses, and for the same reason: ``faulthandler`` writes
with a raw descriptor from a C thread, so the header cannot be deferred to the
moment it fires.

WHY AN ``unlink`` HERE IS ALLOWED, AND WHERE THE ARGUMENT LIVES.
``tests/unit/session/test_no_session_deletion.py`` fails the build on any
``<path>.unlink`` under ``local_operator/`` outside ``session/cleanup.py``, because
twice a "safe" reaper deleted a real session's files. Its docstring states the
route for a call site that must exist: *"Adding a call site therefore means adding
a row HERE with a reason a reviewer can check — the point is not that the list is
short, it is that every entry was argued for."* That is what this module does,
twice (``arm``'s tidy-up of a header it just failed to arm, and ``disarm``'s
removal of a clean runtime's file), and the reason in each row is the checkable
part: :func:`dump_path` composes the path from ``paths.log_dir()`` and an int pid
ALONE — never from a session id, a session directory, or any other caller input —
so no call here can name a session file. Leaving the file behind instead (its
content, not its existence, carrying the outcome) was implemented first and
rejected on review: it accumulates one file per runtime process with nothing to
prune them, and the existence signal is worth keeping.

NO CREDENTIAL CAN LAND IN IT. ``faulthandler`` prints frames — file, line and
function name — and never local variables, which is the property this file
depends on to be written into a log directory at all. That is asserted, not
assumed: see ``test_the_dump_carries_no_local_values``.

ONE PROCESS-GLOBAL TIMER, AND THE INTERLOCK IS THE ARMING SITE
-------------------------------------------------------------
``faulthandler``'s timer is process-wide, shared with ``tests/e2e/watchdog.py``
and ``tests/shard_stall_watchdog.py``, and displacing either would silently
remove a CI stage's only bound (pinned by
``tests/unit/test_shard_stall_watchdog.py::test_no_ci_job_runs_both_watchdogs_in_one_process``).
Nothing interlocks the timers here, and nothing has to: this module is armed
ONLY from the runtime child's own entry point
(``process.__main__`` — reachable by ``python -m ...``, which is how every
spawn names it: ``launch._spawn_runtime``, ``mobile/daemon.py``), while those two
harnesses arm theirs in the pytest process. ``arm`` is deliberately NOT called
from ``RuntimeServer`` or any constructor: ``start_in_process`` is used by
in-process hosts and by the test suite, so arming there would arm a timer inside
a pytest worker — which is exactly how the e2e stage would lose its bound. That
is an invariant with a test, not a comment: see
``tests/unit/session/runtime/test_runtime_stall_watchdog.py``.

OUT OF SCOPE, NAMED SO A READER DOES NOT ASSUME COVERAGE. The TUI hosts a
``RuntimeServer(kind="tui")`` registrant of its OWN in ``tui/app.py``; that
process is a person's terminal, so whether a stalled one should exit (and what
that does to the terminal) is its own design conversation, and a bound armed
from a library constructor there is the interlock hazard above. The detached
``-m`` child is the shape this covers — 16 of 16 records in the operator's own
store were ``kind=daemon`` when this was written.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

#: The bound, in seconds, as an operator or a test overrides it. Unset means
#: :data:`DEFAULT_STALL_S`; ``0``/``off`` disables the watchdog outright.
ENV_SECONDS = "LOP_RUNTIME_STALL_SECONDS"

#: Seconds of NO PROGRESS before the runtime dumps every thread and leaves.
#:
#: Sized against the two numbers the record already publishes, and against the
#: one measured false-positive class:
#:
#: * ``HEARTBEAT_INTERVAL_S`` (15 s) is how often a healthy runtime reports;
#:   ``HEARTBEAT_TIMEOUT_S`` (45 s) is when ANOTHER process starts calling it
#:   ``wedged``. Both are too tight to exit on — a ``wedged`` reading is
#:   degraded-responsiveness EVIDENCE, and this host has produced 105.8 s and
#:   205.8 s beat gaps on sessions whose CPU time was advancing (see
#:   ``types.HEARTBEAT_INTERVAL_S`` and ``registry.classify``). Killing a
#:   session for that would be the bug this module exists to prevent, wearing
#:   the other hat.
#: * 300 s sits above the largest legitimate gap ever measured here (205.8 s,
#:   1.5x) and far below the freezes the operator actually hit (1.5-7.2 h, and
#:   four of the five with ZERO writes in that window). A single synchronous
#:   step that holds the GIL for five unbroken minutes is not slow work; it is
#:   the state that made five sessions unreachable at once.
#:
#: A runtime is expected to be armed for its whole life, so this number is a
#: CEILING on one uninterrupted stall, not a limit on how long a session may
#: live or how long a turn may take: a turn that keeps yielding keeps beating.
DEFAULT_STALL_S = 300.0

#: The largest bound ``faulthandler`` can hold. Its timeout becomes a signed
#: 64-bit count of nanoseconds, so a larger value raises ``OverflowError:
#: timestamp out of range for platform time_t`` inside a C call — and here that
#: would happen in the runtime's own entry point. Any bound a person could mean
#: is minutes; a value at or above this is a typo and is treated as one.
MAX_BOUND_S = 2**63 / 1e9

#: Filename prefix of one process's dump. Keyed by pid, like every other runtime
#: artifact here: one process writes exactly one file, and a reader holding a
#: record's pid can name the file it would have written without being told
#: anything else. A pid recycled by a later runtime truncates that file, which
#: is the correct outcome — the evidence is about whoever holds the pid now.
DUMP_PREFIX = "runtime-stall"

#: What ``faulthandler`` itself writes when the C timer actually fires
#: (``Timeout (0:05:00)!``). Presence of this line is the single definition of
#: "this file is evidence", so a header-only file left by a SIGKILL is never
#: read as a fired bound.
FIRED_MARKER = "Timeout ("

#: The header written when the timer is ARMED. It reads like a claim but is not
#: one; :func:`fired_pids` is what distinguishes the two.
ARM_MARKER = "[stall watchdog] "

#: The planes whose progress this bound tracks. Named rather than spelled at
#: each tick site: the two names are the whole vocabulary, and an unknown one is
#: a programmer error rather than a third plane (see :func:`beat`).
WORKLOAD = "workload"
SERVING = "serving"
PLANES = (WORKLOAD, SERVING)

#: The floor on the OPERATOR-facing bound, in seconds. A bound tighter than three
#: heartbeat intervals can fire on a HEALTHY runtime: both planes tick every
#: ``HEARTBEAT_INTERVAL_S`` (15 s), so a bound of one tick leaves no room for a
#: single late tick on a loaded host, and the first revision shipped without this
#: guard — review measured ``LOP_RUNTIME_STALL_SECONDS=4`` dumping and killing a
#: healthy runtime at 4.1 s. Three ticks is the smallest multiple that absorbs one
#: late tick, and it lands exactly on ``types.HEARTBEAT_TIMEOUT_S`` (45 s), the
#: threshold this fleet already uses to call a heartbeat STALE — so the floor is
#: the fleet's own existing answer rather than a new number to argue about.
#:
#: It applies to the ENVIRONMENT and not to :func:`arm`, deliberately: ``arm`` is
#: called by the entry point with whatever the environment produced, and by tests
#: with a one-second bound (see ``SHORT_BOUND_S`` there) where waiting out the
#: floor would be 45x slower for no extra coverage. The operator-facing knob is
#: the one that must not accept a value that kills healthy runtimes.
MIN_BOUND_FLOOR_TICKS = 3


def min_bound_seconds() -> float:
    """The floor above, resolved lazily so this module stays import-light."""
    from local_operator.session.runtime.types import HEARTBEAT_INTERVAL_S

    return MIN_BOUND_FLOOR_TICKS * HEARTBEAT_INTERVAL_S


#: The shortest re-arm, so an already-overdue deadline is not passed to
#: ``faulthandler`` as zero or a negative number. A fiftieth of a second is
#: short enough that "overdue" and "fires now" are the same thing to a reader.
MIN_REARM_S = 0.05


class _Armed:
    """One process's armed timer: its file, its bound, and each plane's stamp.

    The handle is held OPEN for the process's life, and that is a requirement
    rather than tidiness: ``faulthandler`` writes with the bare descriptor, so
    closing the object would leave the timer firing at a closed fd.
    """

    __slots__ = ("path", "handle", "seconds", "pid", "last_beat")

    def __init__(self, path: Path, handle: IO[str], seconds: float, pid: int) -> None:
        self.path = path
        self.handle = handle
        self.seconds = seconds
        self.pid = pid
        #: Each plane's last sign of life, in ``time.monotonic`` seconds. Seeded to
        #: the ARM time rather than left empty, so a plane that has simply not
        #: ticked yet (the 15 s between boot and its first tick) is measured from
        #: the arm rather than treated as infinitely silent — which would fire the
        #: bound on every healthy boot.
        self.last_beat: dict[str, float] = {plane: time.monotonic() for plane in PLANES}

    def deadline(self) -> float:
        """The earliest moment any plane's silence reaches the bound.

        THE EARLIEST, not the latest, and that is the whole of A2's fix: a healthy
        plane's tick must shorten the timer toward a silent plane's deadline, never
        push it out. See the module docstring.
        """
        return min(stamp for stamp in self.last_beat.values()) + self.seconds


#: The process's armed timer, or ``None``. Module state rather than an object a
#: caller threads through, because the two beats come from two different loops
#: (a serving thread and the workload's own loop) that share no owner — and
#: because "is this armed" has to be answerable without one:
#: :func:`is_armed` is what pins that no library path armed anything.
_ARMED: _Armed | None = None

#: Serializes arm/beat/disarm. ``beat`` is called from two threads — the serving
#: plane's heartbeat and the workload's tick — and each beat both stamps its own
#: plane and re-arms the ONE process-global C timer from the earliest deadline, so
#: the read-modify-write has to be atomic: two ticks interleaved could otherwise
#: leave the timer armed for the longer of the two.
_LOCK = threading.Lock()


def bound_seconds() -> float | None:
    """The configured bound, or ``None`` when the watchdog is switched off.

    Unreadable, non-numeric and out-of-range values fall back to
    :data:`DEFAULT_STALL_S` rather than raising, and a value below
    :func:`min_bound_seconds` is raised TO that floor (see the constant): this runs in the runtime's
    entry point, where an exception would take the session down over a
    diagnostic, and the honest failure of a typo is the default bound rather
    than no bound at all. An explicit ``0``/``off`` is the one spelling that
    means "do not arm", which an operator (or a test that deliberately blocks a
    loop) needs.
    """
    raw = os.environ.get(ENV_SECONDS)
    if raw is None:
        return DEFAULT_STALL_S
    text = raw.strip().lower()
    if text in ("", "off", "no", "false"):
        return None
    try:
        seconds = float(text)
    except ValueError:
        logger.warning("%s=%r is not a number; using %.0fs", ENV_SECONDS, raw, DEFAULT_STALL_S)
        return DEFAULT_STALL_S
    if seconds == 0:
        return None
    if seconds < 0:
        # A NEGATIVE bound is a typo rather than a switch: ``0`` is the written
        # spelling for "do not arm", and nothing else can be meant by -1.
        logger.warning("%s=%r is negative; using %.0fs", ENV_SECONDS, raw, DEFAULT_STALL_S)
        return DEFAULT_STALL_S
    if seconds >= MAX_BOUND_S:
        logger.warning(
            "%s=%r is beyond the C timer's range; using %.0fs", ENV_SECONDS, raw, DEFAULT_STALL_S
        )
        return DEFAULT_STALL_S
    floor = min_bound_seconds()
    if seconds < floor:
        # THE FLOOR, and clamping rather than honouring it: a bound under three
        # heartbeat intervals fires on runtimes that are not stalled at all —
        # review measured this knob at 4 s killing a HEALTHY runtime at 4.1 s —
        # so a value below it is a misconfiguration to correct, not a preference
        # to respect. Said out loud rather than silently, because an operator who
        # set 4 s and got 45 s needs to know which one they are running.
        logger.warning(
            "%s=%.3fs is below the %.0fs floor (%d heartbeat intervals); using %.0fs",
            ENV_SECONDS,
            seconds,
            floor,
            MIN_BOUND_FLOOR_TICKS,
            floor,
        )
        return floor
    return seconds


def dump_path(pid: int | None = None, directory: Path | None = None) -> Path:
    """Where this process's dump goes: ``<log dir>/runtime-stall-<pid>.log``.

    A function rather than an inline join because three readers name the same
    file — the runtime's own log line, the tests, and an operator (or an agent)
    who has a record's pid and nothing else.
    """
    from local_operator.paths import log_dir

    base = directory if directory is not None else log_dir()
    return base / f"{DUMP_PREFIX}-{pid or os.getpid()}.log"


def arm(
    *,
    seconds: float | None = None,
    directory: Path | None = None,
    pid: int | None = None,
) -> bool:
    """Arm the process's stall bound. Returns whether it is now armed.

    Called from the runtime child's entry point and NOWHERE else — see the
    module docstring for why that is the interlock with the two pytest-side
    watchdog timers. Idempotent in the sense that re-arming replaces the bound,
    but it is written to be called once: a second call leaks the first handle.

    Never raises. A diagnostic that cannot be armed must leave the runtime
    otherwise untouched: an unwritable log directory is a reading of "no dump
    possible", not a reason to fail a session's boot.
    """
    global _ARMED
    with _LOCK:
        if _ARMED is not None:
            return True
        bound = bound_seconds() if seconds is None else seconds
        if bound is None:
            logger.info("stall watchdog disabled by %s", ENV_SECONDS)
            return False
        target = dump_path(pid, directory)
        try:
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            handle = target.open("w", encoding="utf-8")
        except OSError:
            logger.warning("stall watchdog could not open %s; no dump will be written", target)
            return False
        try:
            # THE HEADER IS WRITTEN BEFORE THE TIMER IS ARMED, never deferred:
            # faulthandler writes with a raw descriptor from a C thread, so the
            # file and its header have to exist first. Everything below this
            # line is what a reader finds when the bound fires, in this order.
            handle.write(
                f"{ARM_MARKER}pid {pid or os.getpid()} armed for {bound:g}s at {time.time():.0f} "
                f"({time.strftime('%Y-%m-%d %H:%M:%S')}); the runtime's own loops re-arm this "
                f"timer, so a dump below means this process made no progress for {bound:g}s.\n"
            )
            handle.flush()
            faulthandler.dump_traceback_later(bound, file=handle, exit=True)
        except (OSError, ValueError, OverflowError, RuntimeError):
            logger.warning("stall watchdog could not arm; no dump will be written", exc_info=True)
            # The header this call just wrote would otherwise read as evidence of
            # an armed runtime, so it goes — and only ever this call's own file.
            # Allow-listed for the reason spelled out in the module docstring.
            try:
                handle.close()
                target.unlink()
            except OSError:
                pass
            return False
        _ARMED = _Armed(target, handle, bound, pid or os.getpid())
        return True


def beat(plane: str) -> None:
    """Report ONE plane's progress, and re-arm for the earliest deadline.

    ``plane`` is :data:`WORKLOAD` or :data:`SERVING` — the two loops whose
    progress this bound tracks. A stamp here means THAT LOOP RAN, and nothing
    about the work it was running: see the module docstring for what the bound
    does and does not measure.

    NO ``cancel`` BEFORE THE RE-ARM, deliberately. ``faulthandler`` replaces a live
    timer when ``dump_traceback_later`` is called again (measured: no exception,
    the new bound wins), so the cancel was two extra C calls that also opened a
    window in which the process had NO bound — a failed re-arm after a successful
    cancel would leave a runtime unbounded while the old comment claimed the
    worst case was only an early expiry. One call, one atomic replace.

    A no-op when nothing is armed, which is what keeps this safe to call from an
    in-process host (a TUI or a test) that never armed the process timer.
    """
    if plane not in PLANES:
        # A typo must not conjure a third plane that no one ever stamps: that
        # plane would look permanently silent and fire the bound on a healthy
        # runtime. Logged and ignored instead.
        logger.warning("stall watchdog: unknown plane %r; progress not recorded", plane)
        return
    with _LOCK:
        armed = _ARMED
        if armed is None:
            return
        now = time.monotonic()
        armed.last_beat[plane] = now
        try:
            faulthandler.dump_traceback_later(
                max(MIN_REARM_S, armed.deadline() - now), file=armed.handle, exit=True
            )
        except (OSError, ValueError, RuntimeError):
            # A beat that cannot re-arm must not take the loop down with it: the
            # timer is still armed — from the previous beat, or from ``arm`` — so
            # the worst case is a bound that expires sooner than intended.
            logger.warning("stall watchdog could not re-arm its timer", exc_info=True)


def disarm() -> None:
    """Cancel the bound and remove the file: this process left on its own terms.

    Removing the file is what makes its existence mean something — see "THE FILE
    IS THE EVIDENCE" in the module docstring, including why this ``unlink`` is
    allow-listed rather than replaced by an in-place rewrite. Unconditionally safe
    to call, and called on every clean exit path.
    """
    global _ARMED
    with _LOCK:
        armed = _ARMED
        _ARMED = None
        if armed is None:
            return
        try:
            faulthandler.cancel_dump_traceback_later()
        except (OSError, RuntimeError):
            logger.debug("stall watchdog could not cancel its timer", exc_info=True)
        try:
            armed.handle.close()
        except OSError:
            pass
        try:
            armed.path.unlink()
        except OSError:
            pass


def is_armed() -> bool:
    """Whether THIS process holds the C timer through this module."""
    with _LOCK:
        return _ARMED is not None


def announce() -> None:
    """Name the dump file in the runtime's log, once the log exists.

    The arming happens in the entry point, before ``configure_file_logging``
    runs, so the path cannot be logged at that moment. It is logged here
    instead — from ``main``'s first lines — so an operator reading
    ``logs/runtime.log`` after a freeze finds the file to open next to it
    rather than having to reconstruct the naming convention. A no-op when
    nothing is armed, which is the whole in-process case.
    """
    with _LOCK:
        armed = _ARMED
        if armed is None:
            return
        logger.info(
            "stall watchdog armed: %.0fs of no progress dumps every thread to %s and exits",
            armed.seconds,
            armed.path,
        )


def fired_pids(directory: Path | None = None) -> set[int]:
    """The pids in this store whose bound actually FIRED.

    The reader for the evidence this module writes, and it has a real consumer
    rather than being a helper kept warm by its own tests: ``lop sessions --json``
    carries the path per row (``info.collect.session_rows``), which is the surface
    an operator or an agent lists a fleet on after something died. The path itself
    is :func:`dump_path`, so a reader holding a pid needs nothing else.

    ``FIRED_MARKER`` at the start of a line is the test, exactly as
    ``tests/e2e/watchdog.py`` defines it, because surviving the clean exit is not
    enough to call a file a freeze report: a runtime killed without disarming
    leaves a header-only file, and that is a hard death rather than this bound
    (see "THE FILE IS THE EVIDENCE" in the module docstring).

    ONE scan for a whole listing: the marker has to be read out of each candidate,
    so a per-row call would re-glob and re-read the same directory once per
    session on a surface that renders every row.
    """
    from local_operator.paths import log_dir

    base = directory if directory is not None else log_dir()
    fired: set[int] = set()
    try:
        candidates = sorted(base.glob(f"{DUMP_PREFIX}-*.log"))
    except OSError:
        return fired
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not any(line.startswith(FIRED_MARKER) for line in text.splitlines()):
            continue
        suffix = path.name[len(DUMP_PREFIX) + 1 : -len(".log")]
        if suffix.isdigit():
            fired.add(int(suffix))
    return fired


#: The header :func:`arm` writes, in the one shape a reader parses.
#:
#: SPELLED ONCE, next to the writer, for the reason ``FIRED_MARKER`` is: a reader
#: that guessed the format would fail SILENTLY, and the failure of this reader is
#: indistinguishable from "this pid did not trip its bound" — a death that reads
#: ``unattributed`` while its own dump names the act, which is the defect this
#: reader exists to close (the 2026-09-21 ``runtime-killed`` verdicts).
#:
#: ``armed_at`` is an epoch second, and it is the run-identity half rather than
#: decoration: the file is keyed by pid alone, so a pid recycled by a LATER
#: runtime truncates and rewrites it, and a bound fired by that later process
#: must not be attributed to the earlier one (see :func:`fired_bound`).
ARMED_HEADER_RE = re.compile(
    r"^" + re.escape(ARM_MARKER) + r"pid (\d+) armed for ([0-9.]+)s at (\d+)"
)


@dataclass(frozen=True, slots=True)
class FiredBound:
    """One process's stall bound, as the artifact it left behind describes it.

    ``armed_at`` and ``seconds`` come out of the header the arming process wrote
    (so the bound is the number THAT process ran under, not this build's default),
    and ``fired_at`` is the file's mtime — the instant the C thread wrote the dump
    and left, which is the closest thing to a death time any artifact carries.
    """

    path: Path
    pid: int
    seconds: float
    armed_at: float
    fired_at: float

    def covers(self, alive_until: float | None, *, started_at: float | None = None) -> bool:
        """Whether this dump can be the one THIS run left. Bounded on BOTH sides.

        THE LOWER BOUND IS THE ROW'S OWN LAST WRITE, and it is what separates "the
        process that wrote this row tripped the bound" from "a LATER runtime was
        given the same pid". A pid-keyed file has no session in it, so the only
        ordering available is time: the dump is this run's only when the bound was
        ARMED no later than the run's last known sign of life (a process arms its
        bound at boot, before its first turn, so armed-at is always at or before
        its first write). A recycled pid arms AFTER the earlier row's last write
        and is refused.

        THE UPPER BOUND IS ``fired_at >= started_at``, and it is here because the
        LOWER BOUND ALONE IS NOT ENOUGH (review round 1, R1-3). A fired dump is
        only overwritten by the NEXT owner of that pid — and only if that owner
        manages to arm: ``LOP_RUNTIME_STALL_SECONDS=0/off`` disables arming
        entirely, and ``arm``'s own header write can fail on an unwritable log
        directory. A later run that gets the pid through either of those paths
        inherits a stale fired dump whose arm time is older than its own last
        write, and would be narrated with a cause of death that happened BEFORE IT
        STARTED. Requiring the fire to fall inside the run (start ≤ fire) refuses
        that while admitting the hours-long freeze this rung exists to name: a
        freeze means the fire is AFTER the turn began, which is the same
        direction ``journal.install_moved`` bounds its own inference in, and for
        the same reason.
        """
        if alive_until is None:
            return False
        if started_at is not None and self.fired_at < float(started_at):
            return False
        return self.armed_at <= float(alive_until)


def fired_bound(pid: int | None = None, directory: Path | None = None) -> FiredBound | None:
    """The bound that FIRED for ``pid``, or ``None`` when nothing fired for it.

    The per-pid reader next to :func:`fired_pids`' whole-store scan, and it exists
    because the two consumers want different things: a listing wants the SET (one
    glob for every row it renders), while a post-mortem verdict wants ONE process's
    evidence, with the bound it ran under, to name a death that would otherwise be
    recorded as ``unattributed``.

    ``None`` for every shape that is not this pid's own fired bound: no file (never
    armed, or a clean exit removed it), a header-only file (armed and then killed —
    a hard death, not this bound), an unreadable file, and a header naming a
    DIFFERENT pid. The caller still has to bound the fire against the run it is
    judging; see :meth:`FiredBound.covers` for the ordering that does it.
    """
    target = dump_path(pid, directory)
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
        fired_at = target.stat().st_mtime
    except OSError:
        return None
    lines = text.splitlines()
    if not any(line.startswith(FIRED_MARKER) for line in lines):
        return None
    header = next((line for line in lines if line.startswith(ARM_MARKER)), "")
    matched = ARMED_HEADER_RE.match(header) if header else None
    if matched is None:
        # A fired marker with no readable header is evidence that SOMETHING
        # tripped a bound, and no evidence of which process or which bound — the
        # one shape this reader must not guess at, because the guess would name a
        # cause on an artifact it cannot read.
        return None
    armed_pid, seconds, armed_at = matched.groups()
    if pid is not None and int(armed_pid) != int(pid):
        return None
    return FiredBound(
        path=target,
        pid=int(armed_pid),
        seconds=float(seconds),
        armed_at=float(armed_at),
        fired_at=float(fired_at),
    )
