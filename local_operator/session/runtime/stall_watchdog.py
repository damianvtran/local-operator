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
tens of seconds passes, one that never returns is cut.

A SECOND LEG: TICKING IS NOT ADVANCING, AND ONE FLEET SESSION WAS BOTH ALIVE AND
MAKING NO PROGRESS FOR SEVEN MINUTES
--------------------------------------------------------
As first shipped, this module measured the LOOPS RUNNING and nothing else, so a
loop that keeps ticking while its work stands still sat outside the design
entirely. That is not hypothetical. Session ``14066af01c7a`` on build 0.61.16 —
i.e. WITH this bound armed — was measured by another session's probe with no
progress in its transcript, its roster or its four subagent counters across 75 s,
**+14.3 s of process CPU burned in that window**, and 231 s of heartbeat age,
while its plane ticks kept re-arming this very timer. The probe's own words: no
build, no command, no work in flight. A four-way subagent batch had been launched
16 s after a sibling settled, the parent acknowledged the launch, and then it
produced nothing for the rest of the window. Its serving plane's last beat was
18:34:58 and the operator's reap landed 18:39:46 — **12 s before** the 300 s
liveness deadline of 18:39:58. Both instruments were right: the bound saw a live
loop, the probe saw a stalled session, and nothing could see "spinning without
advancing".

So the bound carries a SECOND, COMPOSITE leg, and it is deliberately NOT a
progress-only bound. A legitimate long step — a model call, a tool, a subprocess
— produces no transcript movement, and a bound keyed on movement alone would cut
it; separating WAITING from SPINNING is the whole difficulty, so three facts must
hold together for a whole window:

1. NO MOTION. The process's progress footprint has not changed since the previous
   sample — :func:`process._work_motion`, which is the DRAIN's clock and is
   REUSED here rather than re-derived. It is the one tuple in this tree whose
   every field is moved by work and not by a clock (see its docstring), and a
   second footprint clock would be a second definition of "movement".
2. NOTHING IN FLIGHT. No tool batch is executing: the live context does not end
   in an assistant message whose tool calls have no answers — the state
   ``Session._wire_legal_snapshot`` documents as holding "for the whole duration
   of every tool batch" — and no on-demand compaction is running. THIS is the
   leg that spares the legitimate long step, and the reason "CPU advancing"
   alone is not the predicate: an in-process tool (a render, a local scan) burns
   CPU with no transcript movement for as long as it runs.
3. CPU ADVANCING. This process burned at least :data:`PROGRESS_CPU_FLOOR` of one
   core across the sample. A step that WAITS burns none — a model call is a
   socket read, and a bash child's CPU belongs to the child, never to
   ``time.process_time`` — while a loop that SPINS burns it. That is the whole
   discriminator.

THE WINDOW IS THE BOUND, and the argument that sized :data:`DEFAULT_STALL_S`
sizes this one too: 300 s sits above the largest legitimate silence ever measured
on this fleet (205.8 s, 1.5x), and the measured false-positive class for a reader
calling a runtime ``wedged`` at 45 s — 105.8 s and 205.8 s of beat gap on
sessions whose CPU time was advancing — is a STARVED SCHEDULER, which is a
session DOING work: such a sample fails leg 2 (a batch is in flight) or leg 1 (a
lane is stepping, so ``_subagent_roster_generation`` moves). A run where all
three legs held for 300 continuous seconds is not a slow step; it is a process
burning a core to produce nothing.

WHAT THIS LEG STILL CANNOT SEE, named so a reader does not assume coverage. The
in-flight leg reads the PROCESS's own step and its compaction. A SUBAGENT LANE
running a long in-process tool of its own is not "in flight" by that measure, so
a lane parked in private CPU work with no step boundary for longer than the
window is cut with its parent. Widening the probe to every child session is its
own change — ``comms._records`` holds those sessions privately today — and is NOT
in this one.

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
import threading
import time
from pathlib import Path
from typing import IO, Callable

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

#: The samples one window is divided into, and so the shortest run the progress
#: leg can call a stall. A run is only ever declared after this many CONSECUTIVE
#: agreeing samples, which is what makes the leg a claim about a SUSTAINED state
#: rather than about the instant of one reading — and it is also the narrowest
#: run that can fire, so a two-sample blip cannot.
PROGRESS_SAMPLES_PER_WINDOW = 12

#: The share of ONE core this process must have burned across a sample for that
#: sample to count as SPINNING rather than WAITING. Sized well below the
#: measurement it exists for and well above an idle runtime: the incident's
#: session burned 14.3 s of CPU in a 75 s window (19% of a core, i.e. 3.8x this
#: floor), while a loop waiting on a model, a tool result or a subprocess burns
#: ~0. Set as a RATE rather than a total so a run is only accumulated while the
#: burn continues: a process that spins for a minute and then waits must not
#: carry that minute's CPU into a later window.
PROGRESS_CPU_FLOOR = 0.05

#: The line this module writes into the dump when the PROGRESS leg — not the
#: silence leg — fired. It shares :data:`ARM_MARKER`'s prefix, so one search
#: finds both of this module's own lines in a dump, and it is distinct from
#: :data:`FIRED_MARKER`, which ``faulthandler`` writes and which says only that
#: the C timer expired, never which leg set it. A reader therefore learns from
#: the file itself whether a runtime went silent or spun without advancing.
PROGRESS_MARKER = "[stall watchdog] no progress: "

#: What the first sample of a progress clock holds before it has anything to
#: compare against. A sentinel rather than ``None`` because a probe is free to
#: return ``None`` as its own motion value, and "no previous sample" and "the
#: previous sample was None" must not be the same state.
_NO_SAMPLE = object()


class _ProgressClock:
    """The progress leg's own state: the last sample, and how long a run has held.

    SAMPLED STATE RATHER THAN A SECOND C TIMER, and the asymmetry with the
    liveness leg is the point rather than an economy. That leg must survive a
    thread that holds the GIL for hours, which is why it is ``faulthandler``'s
    C thread; THIS leg is only ever about a process whose loops ARE running — a
    spinning loop reaches the sampler every interval, because a spin that
    starves the sampler is precisely the state the liveness leg already covers.
    So a plain sampler thread is sufficient here where the other leg needed a C
    timer, and the two legs cannot mask each other: whichever one's condition
    holds sets the single process-wide timer first.

    ``since`` is the instant the CURRENT run of agreeing samples began, and the
    window is measured from it; ``None`` means the last sample disagreed, so
    there is no run to measure. Storing the run's start rather than a countdown
    is what makes the window elastic in the right direction: a sample that
    arrives late (a loaded host, a GIL-hungry neighbour) does not shorten the
    run, it only delays the next look.
    """

    __slots__ = ("motion", "since", "cpu", "wall")

    def __init__(self) -> None:
        self.motion: object = _NO_SAMPLE
        self.since: float | None = None
        # Seeded at construction (the arm, or the sampler's own start) so the
        # FIRST sample already has an interval to compute a rate over; a zero
        # interval would divide by ~0 and read as an infinite spin.
        self.cpu: float = time.process_time()
        self.wall: float = time.monotonic()


class _Armed:
    """One process's armed timer: its file, its bound, and each plane's stamp.

    The handle is held OPEN for the process's life, and that is a requirement
    rather than tidiness: ``faulthandler`` writes with the bare descriptor, so
    closing the object would leave the timer firing at a closed fd.
    """

    __slots__ = (
        "path",
        "handle",
        "seconds",
        "pid",
        "last_beat",
        "probe",
        "progress_deadline",
        "clock",
        "stop",
        "thread",
    )

    def __init__(
        self,
        path: Path,
        handle: IO[str],
        seconds: float,
        pid: int,
        probe: "ProgressProbe | None" = None,
    ) -> None:
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
        #: The progress leg's probe, or ``None`` when no runtime supplied one (an
        #: in-process host, a test of the liveness leg alone, an older spawner).
        #: ``None`` means the progress leg is INERT, not that it is satisfied.
        self.probe = probe
        #: The instant the progress leg has decided to fire at, or ``None`` while
        #: its predicate does not hold. Absolute ``time.monotonic``, like the
        #: plane stamps, so that :meth:`deadline` can take one ``min`` over both
        #: legs rather than reason about each separately.
        self.progress_deadline: float | None = None
        self.clock = _ProgressClock()
        #: The sampler's own stop latch and thread, both ``None`` when no probe
        #: was supplied. The latch is a ``threading.Event`` rather than a flag the
        #: sampler polls so that ``disarm`` wakes it immediately instead of
        #: leaving a thread asleep for up to a full sample interval after the
        #: process has already decided to leave.
        self.stop: threading.Event | None = None
        self.thread: threading.Thread | None = None

    def deadline(self) -> float:
        """The earliest moment ANY leg's condition reaches its bound.

        THE EARLIEST, not the latest, and that is the whole of A2's fix: a healthy
        plane's tick must shorten the timer toward a silent plane's deadline, never
        push it out. The progress leg joins on the same rule rather than beside
        it — a beat that re-armed only the planes would push a decided progress
        fire out by a whole heartbeat, which is exactly the masking this method
        exists to prevent. See the module docstring.
        """
        earliest = min(stamp for stamp in self.last_beat.values()) + self.seconds
        if self.progress_deadline is None:
            return earliest
        return min(earliest, self.progress_deadline)


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


def _sample_interval(seconds: float) -> float:
    """How often the progress leg looks, derived from the window it measures.

    DERIVED rather than fixed, because the window is the knob an operator — and a
    test — actually sets, and the granularity has to follow it: a 1 s window
    sampled every 15 s could never accumulate a single run, while a 300 s window
    sampled every 0.05 s would spend a thread waking 6000 times to learn nothing
    the previous wake had not already said. :data:`PROGRESS_SAMPLES_PER_WINDOW`
    is the resolution the window was argued at; ``HEARTBEAT_INTERVAL_S`` is the
    ceiling, because a look coarser than the liveness beat could let a spin start
    and end entirely between two of them.
    """
    from local_operator.session.runtime.types import HEARTBEAT_INTERVAL_S

    return max(MIN_REARM_S, min(HEARTBEAT_INTERVAL_S, seconds / PROGRESS_SAMPLES_PER_WINDOW))


#: What a runtime supplies so the progress leg has facts to judge: a
#: zero-argument callable returning ``(motion, in_flight)``.
#:
#: ``motion`` is any value that is EQUAL to the previous sample exactly when the
#: work did not advance — ``process._work_motion``'s tuple is what the runtime
#: passes, and this module deliberately does not know its shape, so the clock
#: stays in the module that owns the definition of movement (see the docstring).
#: ``in_flight`` is the runtime's own answer to "is a step executing right now?".
#:
#: INJECTED RATHER THAN REACHED FOR, because the alternative is this module
#: importing ``process`` — which imports this one, from its own entry point — and
#: because a bound that can only be tested by booting a whole runtime is a bound
#: whose false-positive cases are never tested. A caller with no probe gets no
#: progress leg, which is the honest default for the in-process hosts.
ProgressProbe = Callable[[], "tuple[object, bool]"]


def _bound_class() -> str:
    """The incident class this bound's exit is recorded under, imported lazily.

    THE DUMP NAMES ITS OWN CLASS, and this is how it gets the token without
    ``stall_watchdog`` depending on the taxonomy module at import time: the file
    lives in a log directory beside ``runtime.log``, and a reader who finds a
    fired marker there should not have to know which module owns the vocabulary
    to say what happened. The import is function-local for the same reason every
    other import in this module is — this file is on the child's boot path.

    Falls back to the literal when the taxonomy cannot be imported, which is the
    one direction that cannot mislead: an unreadable class must not stop a bound
    from being armed, and a dump that says what it is in words is still evidence.
    """
    try:
        from local_operator.incidents import STALL_BOUND_CAUSE

        return STALL_BOUND_CAUSE
    except Exception:  # noqa: BLE001 — a missing label is not a reason to skip the bound
        return "runtime-stall-bound"


def arm(
    *,
    seconds: float | None = None,
    directory: Path | None = None,
    pid: int | None = None,
    probe: "ProgressProbe | None" = None,
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
                f"timer, so a dump below means this process made no progress for {bound:g}s. "
                f"IF A DUMP FOLLOWS THIS HEADER, the bound ENDED this runtime -- incidents "
                f"class {_bound_class()} -- and because the timer fires from a C thread that "
                f"runs no Python, THIS FILE IS THE ONLY PLACE THAT CLASS IS WRITTEN. A "
                f"further line below it carrying the words 'no progress' means the reason was a "
                f"loop SPINNING without advancing; its absence means the runtime went SILENT. "
                f"NEITHER MARKER IS SPELLED HERE, and that is load-bearing rather than tidy: "
                f"readers test for each as a SUBSTRING, so a header quoting one would make "
                f"every armed file -- including one left by a SIGKILL -- read as a fired bound "
                f"or as a progress fire.\n"
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
        _ARMED = _Armed(target, handle, bound, pid or os.getpid(), probe)
        if probe is not None:
            _start_sampler(_ARMED)
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


def _start_sampler(armed: "_Armed") -> None:
    """Give the progress leg the thread it samples on. Called once, from ``arm``.

    A DAEMON thread, and it is not the process's exit that makes that safe —
    ``faulthandler`` leaves through ``_exit(1)`` from its own C thread and takes
    every thread with it — but the arming sequence: a thread that outlived its
    ``_Armed`` would be a leak, and the sampler's own first act on every wake is
    to check that it is still the armed one. ``disarm`` waking it through the
    same latch is what keeps a clean exit from leaving it asleep for a whole
    interval; see ``_Armed.stop``.
    """
    stop = threading.Event()
    armed.stop = stop
    thread = threading.Thread(
        target=_progress_sampler,
        args=(armed, stop),
        name="stall-watchdog-progress",
        daemon=True,
    )
    armed.thread = thread
    thread.start()


def _progress_sampler(armed: "_Armed", stop: threading.Event) -> None:
    """Sample the progress predicate until it holds for a window, or we are disarmed.

    THE ONE LOOP the progress leg has, and it is a sleep loop rather than a call
    from the workload's own tick for the reason ``_beat_stall_watchdog``
    documents on the other side: this must keep looking at a loop that is
    spinning, and a hook the spin never reaches is a hook the spin disables.

    THE SAMPLER IS NOT A TURN'S CLOCK. Nothing here counts turns, awaits work or
    holds a reference to the session: it asks the injected probe the same two
    questions every interval and keeps the answer. That is what lets the leg be
    tested against a fake probe without a runtime, and it is why an unevaluable
    probe errs toward NOT firing — see :func:`_sample`.
    """
    interval = _sample_interval(armed.seconds)
    while not stop.wait(interval):
        with _LOCK:
            if _ARMED is not armed:
                return
            if _sample(armed):
                return


def _sample(armed: "_Armed") -> bool:
    """One progress sample. True when the bound was FIRED, so the sampler is done.

    Called with :data:`_LOCK` held, and it must be: it reads and writes the same
    ``progress_deadline`` that :func:`beat` re-arms from, on another thread, and
    the interleaving that lets slip is a beat re-arming the long deadline over a
    fire that had already decided to happen.

    EVERY LEG IS EVALUATED ON THE SAMPLE IT ARRIVES, all three to the same
    instant: reading the CPU rate at one moment and the motion at another is how
    a legitimate tool boundary gets read as a spin.
    """
    clock = armed.clock
    now = time.monotonic()
    cpu = time.process_time()
    elapsed = max(1e-6, now - clock.wall)
    probe = armed.probe
    try:
        motion, in_flight = probe() if probe is not None else (_NO_SAMPLE, True)
    except Exception:  # noqa: BLE001 — an unevaluable probe must not end a process
        logger.debug("stall watchdog: progress probe failed", exc_info=True)
        motion, in_flight = _NO_SAMPLE, True
    moved = clock.motion is not _NO_SAMPLE and motion != clock.motion
    spinning = not in_flight and not moved and (cpu - clock.cpu) / elapsed >= PROGRESS_CPU_FLOOR
    if not spinning:
        # ANY disagreement ends the run. A leg that is not true NOW is not
        # "possibly true": a window that kept accumulating across a sample where
        # the process was working would fire on a runtime that had done
        # legitimate work inside it, which is the false positive this predicate
        # exists to avoid. The deadline is cleared with it, so a runtime that
        # recovered between a decision and its timer re-arming is left running.
        clock.since = None
        armed.progress_deadline = None
    elif clock.since is None:
        clock.since = now
    elif now - clock.since >= armed.seconds:
        _fire_progress(armed, now)
        return True
    clock.motion, clock.cpu, clock.wall = motion, cpu, now
    return False


def _fire_progress(armed: "_Armed", now: float) -> None:
    """Leave through the SAME dump-and-exit path the liveness leg uses.

    Past the window the runtime must stop being a session that burns a core to
    produce nothing, and the graceful rungs cannot be reached from the state
    being detected — this is the module's oldest constraint, and the reason the
    exit is ``faulthandler``'s rather than anything Python can sequence.

    SO IT REUSES THAT PATH RATHER THAN ADDING A SECOND ONE: the dump file, the
    ``FIRED_MARKER``, the every-thread stack and the ``_exit(1)`` are all the
    liveness leg's, which is what keeps a reader's rule ("a dump with the fired
    marker is evidence a bound actually fired") true for both. What this adds is
    the ONE line above it naming WHICH leg fired — without it a reader could not
    tell a runtime that went silent from one that spun, and the two want
    different investigations.

    ARMING FOR ``MIN_REARM_S`` IS WHAT FIRES IT: ``deadline()`` takes the minimum
    over the legs, and setting the progress leg to ``now`` makes that minimum the
    present moment, so this call and every later :func:`beat` agree on when the
    timer expires. The write is before the arm, and it has to be: ``faulthandler``
    reaches its timer from a C thread that runs no Python, so this line has no
    later moment available to it.
    """
    armed.progress_deadline = now
    try:
        armed.handle.write(
            f"{PROGRESS_MARKER}{_bound_class()}: {armed.seconds:g}s of CPU with no progress "
            f"from the work -- no transcript, roster or job movement, no tool batch in "
            f"flight, and at least {PROGRESS_CPU_FLOOR:.0%} of a core burned across every "
            f"sample of the window.\n"
        )
        armed.handle.flush()
    except (OSError, ValueError):  # noqa: BLE001 — the dump below is the evidence
        logger.debug("stall watchdog could not write its progress line", exc_info=True)
    try:
        faulthandler.dump_traceback_later(
            max(MIN_REARM_S, armed.deadline() - now), file=armed.handle, exit=True
        )
    except (OSError, ValueError, RuntimeError):
        # The deadline STAYS set, deliberately. The timer already armed from the
        # last beat is at most one heartbeat away, and every beat re-arms from
        # ``deadline()``, so a failed re-arm here still ends the process within
        # that heartbeat rather than silently withdrawing a decision already
        # taken. Withdrawing it would make a transient arming failure a way for
        # this leg never to fire again.
        logger.warning("stall watchdog could not fire its progress leg", exc_info=True)


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
        sampler = armed.stop
        if sampler is not None:
            # AFTER the file is gone and the timer is cancelled, so a woken
            # sampler finds nothing armed and returns rather than reading a
            # closed descriptor as "a process that never moved".
            sampler.set()


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


#: Which of the bound's two legs ended a runtime. The vocabulary is exported
#: rather than spelled at the reader, because ``incidents``' class is one token
#: for both legs and the DETAIL is the only place they are told apart.
#:
#: ``SILENCE`` is the liveness leg: no plane ticked for the whole bound, which is
#: a loop parked in a call it never came back from. ``PROGRESS`` is the composite
#: leg: the loops ran and kept re-arming the timer, and the work still did not
#: advance while the process burned CPU.
LEG_SILENCE = "silence"
LEG_PROGRESS = "progress"


def _dump_text(path: Path) -> str:
    """One candidate dump's text, or ``""`` when it cannot be read.

    An unreadable file is NOT evidence of a fire — the reader's whole contract is
    that the marker is what makes a file mean something — so the direction is the
    quiet one.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _fires(text: str) -> bool:
    """Whether a dump's text carries the fired marker, as a whole line.

    ``FIRED_MARKER`` at the start of a line, exactly as
    ``tests/e2e/watchdog.py`` defines it, because surviving the clean exit is not
    enough to call a file a freeze report: a runtime killed without disarming
    leaves a header-only file, and that is a hard death rather than this bound.
    """
    return any(line.startswith(FIRED_MARKER) for line in text.splitlines())


def fired_leg(pid: int | None = None, directory: Path | None = None) -> str | None:
    """WHICH leg fired for this pid, or ``None`` when no bound did.

    The class a fired bound is recorded under is one token for both legs (see
    ``incidents.STALL_BOUND_CAUSE``), so this is the only thing that tells a
    reader whether a runtime went SILENT or SPUN WITHOUT ADVANCING — the two
    want different investigations, and the dump already distinguishes them: the
    progress leg writes its own line at the moment it decides, and the silence
    leg cannot, because the C thread that fires it runs no Python.

    Read from the artifact rather than from a record, because that is the only
    thing the firing path can leave: ``faulthandler`` writes the dump and calls
    ``_exit(1)`` from its own thread, so nothing that runs afterwards — no exit
    hook, no journal write, no reaper — can be relied on. ``journal.death_verdict``
    is the caller that turns this into an incident class.
    """
    text = _dump_text(dump_path(pid, directory))
    if not _fires(text):
        return None
    if any(line.startswith(PROGRESS_MARKER) for line in text.splitlines()):
        return LEG_PROGRESS
    return LEG_SILENCE


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
        text = _dump_text(path)
        if not _fires(text):
            continue
        suffix = path.name[len(DUMP_PREFIX) + 1 : -len(".log")]
        if suffix.isdigit():
            fired.add(int(suffix))
    return fired
