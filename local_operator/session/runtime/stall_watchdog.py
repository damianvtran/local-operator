"""Bound a runtime's OWN stall: dump every thread, then leave — unless work is in flight.

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

THE EXIT LEG IS DECIDED PER ARM, AND IT IS NOT THE SAME QUESTION AS THE FIRE
--------------------------------------------------------------------------
The dump and the ending were one act, and separating them is this module's
second change of policy. The operator's rule, verbatim: "Runtimes should only
update when their turn is complete (fully idle), not based on some heuristic of
inactivity that kills a running process to update it."

Measured against the fleet before this change, the two acts were not the same
event at all. Of 40 ``runtime-stall-*.log`` dumps on this machine (11 on 09-21,
29 on 09-22 up to 11:00), 37 carry the fired marker, and the event-loop thread's
own stack in those fires is a runtime DOING WORK more often than a runtime that
has stopped: ``session._run_turn`` -> ``_emit`` -> ``serving._refresh_state`` ->
``comms.roster`` (pids 78817, 30990), a boot still inside
``transcript.__init__`` on an executor thread (pid 5527), and — for six fires in
the 01:00-07:00 window — an idle loop parked in ``selectors.select`` while a
SINGLE plane's tick had gone missing. A bound that dumps every thread is worth
keeping exactly as it is in all of those cases: the stacks are the only
evidence, and one of these fires was the one that finally identified an
hours-long freeze. A bound that also ends the process is what the operator is
asking to be rid of, because the act it ends is a turn.

So the timer is armed with ``exit=`` answered at every re-arm:

* **work in flight → the dump is written and the process SURVIVES**, marked as
  STALLED rather than gone (the sampler appends :data:`HELD_MARKER` below the
  stacks, which is the only place it can be written from — see
  :func:`_record_held_fire`). The runtime keeps serving the build it loaded and
  the decision to end it is the operator's (``lop stop``);
* **idle → the bound ends it exactly as it did before**, and this is the wedge
  recovery the module was built for, preserved for the state the operator's rule
  names: nothing in flight, so there is no work a cut can lose.

WHY THE WEDGE RECOVERY IS NOT WEAKENED BY THIS. The founding measurement's five
frozen runtimes were burning ~0.9 core, i.e. a scan in flight — a state this exit
leg now holds and dumps instead of cutting. They were reaped by hand then, and
they would be reaped by hand now, with one difference: the dump says the bound
fired and held, so the hand that does it knows the runtime is stalled rather than
merely quiet. What it does NOT do is leave a wedged IDLE runtime resident: an idle
runtime answers the probe ``False``, so its bound still ends it, and a runtime
that recovers from a held stall drops the hold on the sampler's next wake
(:func:`_refresh_exit_leg`).

UNKNOWN SPLITS TWO WAYS. A probe that RAISES holds the exit (:func:`_holds_work`),
because the runtime did report a way to speak and that report could not be read; a
caller that supplied NO probe keeps today's behaviour, because it is a rig or a
reduced host with no way to speak at all — and the one production arm site always
supplies one, so holding on absence would only disarm the bound for callers that
cannot answer.

A SUBAGENT LANE IS IN FLIGHT FOR THE EXIT LEG AND STILL INVISIBLE TO LEG 2, and
the asymmetry is deliberate rather than an oversight: ``is_busy`` counts live
subagent lanes and jobs (so the exit leg holds for them — see below), while leg
2's narrower reading is left exactly as it was, because widening THAT is a change
to when the dump is written, and a dump is not destructive. Widening it is called
out as its own change below and is still not this one.

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
   core, as a **mean over the trailing window** — not a per-sample reading and not
   a cumulative mean either; see that constant for the two review measurements
   that made it neither. A step that WAITS burns none — a model call is a socket
   read, and a bash child's CPU belongs to the child, never to
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

THE TICK ITSELF CAN DIE, AND UNTIL THIS ITS DEATH WAS INVISIBLE
--------------------------------------------------------------
Everything above assumes the two ticks RUN. They are an ``asyncio`` task and a
thread loop, so they can also DIE, and on 2026-09-21 one did: the workload
tick's task raised, nothing observed it (no done-callback, no ``await``, no
reader of ``exception()``), its stamp froze, and this bound fired 300 s later on
a runtime that was otherwise perfectly healthy — killing the turn in flight.
The dump then read as :data:`LEG_SILENCE` — "a loop parked in a call it never
came back from" — which is the one explanation that was FALSE. ``faulthandler``
dumps THREADS and never tasks, so a dead tick leaves no frame at all: the file
shows what an idle, healthy process shows, and that is also what it shows when
a tick is merely late. Nothing in the artifact separated the two, and that
indistinguishability is the whole of this section.

So a tick's death is now three things, and it has to be all three: the
supervisor that drives the tick LOGS it at WARNING with the exception
(``process._watch_stall_beats``), RE-CREATES the task while the deaths stay
inside a rolling budget — ``process.STALL_BEAT_RESTARTS`` deaths inside
``process.STALL_BEAT_WINDOW_S``, and the supervisor is itself guarded so that a
fault in its own recovery path cannot end the supervision unobserved either —
and records it HERE, :func:`note_tick_death`, a line written into this
process's own dump beside the plane's own stamp, because the dump is the only
artifact that survives the exit this bound makes. A dump carrying a tick-death
line therefore says "the workload tick stopped" where the same dump used to
say nothing at all, and :func:`tick_deaths` is the reader for it — as is the
journal's incident narration, which prefers that fact to the bare silence leg
when a dump carries both.

A DEAD TICK STILL LETS THE BOUND FIRE, deliberately. Nothing here unbounds a
plane whose ticker is gone: a plane nothing can stamp is a plane whose silence
this module can no longer interpret, so the honest fail-safe is to leave on the
deadline WITH the reason in the file rather than to run on with one leg
silently switched off.

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
short, it is that every entry was argued for."* That is what this module does, at
the sites named here: ``arm``'s tidy-up of a header it just failed to arm AND of a
deadline sibling an earlier holder of this pid left (a pid is RECYCLED, and these
files are keyed by pid alone — see :func:`deadline_path`), ``disarm``'s removal of a
clean runtime's dump and of its deadline sibling, and ``_record_deadline``'s removal
of its own sidecar temp after a failed atomic replace. The reason in each row is the
checkable part: :func:`dump_path` and :func:`deadline_path` compose their paths from
``paths.log_dir()`` and an int pid ALONE — never from a session id, a session
directory, or any other caller input — so no call here can name a session file.
Leaving the file behind instead (its content, not its existence, carrying the
outcome) was implemented first and rejected on review: it accumulates one file per
runtime process with nothing to prune them, and the existence signal is worth
keeping.

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
from collections import deque
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

#: Suffix of the SIBLING that holds the deadline this process's timer is currently
#: armed for, and the leg that pinned it (see :func:`deadline_path`). A sibling rather
#: than a line in the dump because a beat rewrites it every 15 s for the process's
#: whole life: appended to the dump that would grow the artifact without bound, and
#: the number a reader needs is the LAST one written.
DEADLINE_SUFFIX = ".deadline"

#: What ``faulthandler`` itself writes when the C timer actually fires
#: (``Timeout (0:05:00)!``). Presence of this line is the single definition of
#: "this file is evidence", so a header-only file left by a SIGKILL is never
#: read as a fired bound.
FIRED_MARKER = "Timeout ("

#: The header written when the timer is ARMED. It reads like a claim but is not
#: one; :func:`fired_pids` is what distinguishes the two.
ARM_MARKER = "[stall watchdog] "

#: What the header states a FIRE IS, because this artifact is opened by a reader who
#: has nothing else in hand, and its previous wording asserted a verdict the module
#: never computes.
#:
#: THE MISFIRE IT EXISTS TO PREVENT, measured 2026-09-22: a fire landed on the build
#: carrying #1419 43 s after install, on a runtime that was MID-TURN AND WORKING,
#: running a dense loop of short tool calls -- and that pid was STILL ALIVE
#: afterwards, same pid, with no successor runtime ever coming up. An operator
#: session read the pile of those dumps as a body count, published "31 bound-kills",
#: and the count reached the v0.62.2 release notes before anyone asked whether a
#: process had died. The file did not contradict the reading: its header said a dump
#: below it meant the bound ENDED the runtime.
#:
#: WRITTEN AT ARM TIME, SO IT MUST BE TRUE OF EVERY FIRE IT CAN PRECEDE, and a fire
#: can precede a process that carries on. The timer expiring is an OBSERVATION ABOUT
#: THE TIMER; the process's death is a SEPARATE fact, and the exit that would join
#: the two may never come. A reader asking whether the runtime is still there
#: therefore reads the PID, never this file -- the same line this module's own
#: readers draw ("a bound fired" is all :func:`fired_pids` says, and
#: :func:`fired_leg` names only which leg).
OBSERVATION_NOT_VERDICT = (
    "IF A DUMP FOLLOWS THIS HEADER IT IS AN OBSERVATION, NOT A VERDICT: the watchdog's timer "
    "expired without a re-arm from this process's own loops, and that is ALL it measured. It is "
    "NOT a statement that the runtime stopped -- the timer can expire on a process that is "
    "mid-turn and working -- so THIS PROCESS MAY STILL BE ALIVE ON THIS SAME PID, and may carry "
    "on serving, AFTER this dump is written. What would turn the observation into a death is the "
    "exit faulthandler takes once it has dumped (this timer is armed with exit=True), and THAT "
    "EXIT MAY NEVER COME. So a fire is not by itself a body count, and THE PID, NOT THIS FILE, "
    "IS WHAT SAYS WHETHER THE RUNTIME IS STILL THERE."
)

#: How to read the two numbers a fire leaves behind: the value on ``faulthandler``'s own
#: fired line, and the sibling ``runtime-stall-<pid>.deadline``. Exported for the same
#: reason :data:`OBSERVATION_NOT_VERDICT` is -- the cell that pins it asserts the
#: shipped sentence rather than a fragment of it.
#:
#: THE DISTINCTION IT EXISTS TO STATE, measured over 26 retained fires: 6 read a full
#: ``0:05:00``-style value, which is :func:`arm`'s own bound and means NO beat ever
#: re-armed the timer (the never-engaged class -- a main thread idle from boot), while
#: 19 read a small value (0.4-17 s), which is a beat's recomputed remainder and means a
#: plane's stamp, not the bound, was what the timer measured. Nothing in the file said
#: which of the two a number was, so "which plane went quiet, and when" stayed
#: unreadable on a runtime whose own record is the only witness left.
HOW_TO_READ_THE_FIRED_VALUE = (
    "HOW TO READ WHAT FIRES: the value on the fired line above the stacks is the seconds the "
    "timer was LAST ARMED FOR, and which arming that was is the whole question. It is the bound "
    "above, with no beat re-arming it at all, when no loop ever reported -- a runtime that never "
    "engaged, its main thread idle from boot. It is a SMALLER value when a beat recomputed it "
    "from the oldest plane's stamp, which is how a plane that went quiet shows up as a number "
    f"below the bound. {DUMP_PREFIX}-<pid>{DEADLINE_SUFFIX} holds the deadline the timer is "
    "currently armed for and the leg that pinned it, rewritten by every beat THAT RE-ARMED, and "
    "removed when a runtime arms -- so it describes the runtime holding this pid NOW, never an "
    "earlier one (a pid is recycled). Its ABSENCE means no beat ever re-armed this timer, and "
    "its mtime is the last SUCCESSFUL re-arm: read any line below carrying the word 're-arm' "
    "with it, because where such a line says the re-arm FAILED the number and the mtime are "
    "both behind the last beat, and where it names a plane that had gone quiet it says how "
    "long. Compare that epoch with the fire to say whether the bound came due or was "
    "pre-empted.\n"
)

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

#: How many times one window is LOOKED AT, which is the leg's resolution and
#: nothing more: the decision is a claim about the trailing window (see
#: :data:`PROGRESS_CPU_FLOOR`), so the look count gates nothing and a run of two
#: samples a window apart is judged on the same terms as one of twenty. This was
#: the "twelve CONSECUTIVE agreeing samples" requirement until agent review round
#: 1 measured what a single scheduled-out sample did to it — do not restore it as
#: a gate, and do not read a shorter run here as weaker evidence: a mean over a
#: long window and a mean over a short one are the same statistic.
#:
#: It sizes the sampler's cadence (``_sample_interval``), which is why it is a
#: divisor of the bound rather than a fixed number of seconds: a 1 s bound sampled
#: every 15 s could never accumulate any run at all, and a 300 s bound sampled
#: every 0.05 s would wake 6000 times to learn nothing new.
PROGRESS_SAMPLES_PER_WINDOW = 12

#: The share of ONE core this process must have burned, as a MEAN OVER THE
#: TRAILING WINDOW, for the run to count as SPINNING rather than WAITING.
#:
#: NOT A PER-SAMPLE RATE, AND NOT A CUMULATIVE MEAN, and the two review rounds
#: are why it is neither. Round 1 measured a per-sample gate discarding a whole
#: run on one scheduled-out sample (per-sample minima of 0.0183-0.0310 against
#: medians of 0.09-0.27, an effective margin of 0.4-1.2x rather than the 3.8x the
#: justification quoted, and one rigged run in fifteen that never fired). Round 2
#: measured what the first repair still got wrong: one baseline held from the
#: run's start makes this a CUMULATIVE mean that decays as ~1/t through later
#: silence, so 4 s of full core followed by 41 s of idleness still fired at
#: sample 46 of a 45 s window, and — the shape that matters for the incident this
#: leg exists for — a session idle for an hour before it began spinning carried
#: that hour into the denominator. The mean is taken over the samples retained in
#: the trailing window (see :meth:`_ProgressClock.mean_rate`), which is the
#: statistic this justification has always described.
#:
#: Sized well above an idle runtime and well below the measurement it exists for:
#: the incident's session burned 14.3 s of CPU in a 75 s window (19% of a core,
#: 3.8x this floor), a healthy idle runtime measures 0.006-0.011 of a core, and a
#: loop waiting on a model, a tool result or a subprocess burns ~0.
PROGRESS_CPU_FLOOR = 0.05

#: The line this module writes into the dump when the PROGRESS leg — not the
#: silence leg — fired. It shares :data:`ARM_MARKER`'s prefix, so one search
#: finds both of this module's own lines in a dump, and it is distinct from
#: :data:`FIRED_MARKER`, which ``faulthandler`` writes and which says only that
#: the C timer expired, never which leg set it. A reader therefore learns from
#: the file itself whether a runtime went silent or spun without advancing.
PROGRESS_MARKER = "[stall watchdog] no progress: "

#: The line written into the dump when a plane's TICKER died, which is NOT the
#: same fact as the plane's LOOP being silent — and the difference is the entire
#: reason this line exists. Every other line in this file describes a process
#: whose loops were alive and not reporting, or were reporting and not advancing;
#: this one describes a process whose reporter is GONE, which no thread dump can
#: show (see the module docstring). It shares :data:`ARM_MARKER`'s prefix so one
#: search still finds every line this module writes, and it is written by
#: :func:`note_tick_death` at the moment the death is observed — chronologically
#: above any ``Timeout (`` dump the bound later produces, which is what lets a
#: reader attribute a fired bound to a dead reporter instead of to a parked loop.
#:
#: NOT SPELLED IN THE HEADER, for the reason that constraint exists at all:
#: readers test these as substrings, so a header quoting one would make every
#: armed file read as a fired bound or as a dead tick.
TICK_DEATH_MARKER = "[stall watchdog] tick died: "

#: The line a beat writes into the dump when a plane has gone quiet for this fraction
#: of the bound or more, naming that plane and how long it has been quiet.
#: A REPORTING POINT AND NOT A FIRING ONE: the bound still fires at the bound, this
#: only decides when the dump stops being silent about WHICH plane is behind -- the
#: question an operator asks first of a fire, and the one the artifact could not
#: answer (it carried the arm epoch and nothing else). Half the bound is where the
#: deadline being re-armed for is no longer the bound under any reading: for the 300 s
#: bound that is 150 s of silence, ten missed heartbeats, against a healthy plane's
#: 0-15 s stamp age (see ``types.HEARTBEAT_INTERVAL_S``) so a healthy runtime never
#: reaches it.
QUIET_FRACTION = 0.5

#: Written by :func:`_note_quiet_plane` when a plane crosses :data:`QUIET_FRACTION`.
#: Shares :data:`ARM_MARKER`'s prefix so one search still finds every line this module
#: writes, and is written by a BEAT -- chronologically above any dump the bound later
#: appends, which is what makes the two readable as one chronology.
#:
#: NOT SPELLED IN THE HEADER, for the reason that constraint exists at all: readers
#: test these as substrings, so a header quoting the marker would make every armed
#: file read as a runtime with a quiet plane.
REARM_MARKER = "[stall watchdog] re-arm: "

#: Written by a BEAT whose RE-ARM RAISED -- the one case where the deadline sibling's
#: mtime is not the last beat, which is the reading :data:`HOW_TO_READ_THE_FIRED_VALUE`
#: sends a reader to it for.
#:
#: THE SIBLING CANNOT SAY THIS ABOUT ITSELF: its whole content is the deadline that IS
#: in force, and a failed re-arm leaves exactly that one in force (``faulthandler``
#: keeps the previous timer when a re-arm raises, argued at :func:`_record_deadline`),
#: so the beat that failed is the only witness and the dump is the only artifact that
#: outlives the process. Without it a reader takes a frozen mtime for "this process
#: stopped beating" -- an inference, on a file whose whole purpose here is to be a
#: measurement.
#:
#: NOT :data:`REARM_MARKER`, deliberately: the reader for the quiet-plane line takes
#: lines that START WITH that marker, and this line states a different fact about a
#: different file. NOT SPELLED IN THE HEADER, for the reason that constraint exists at
#: all: readers test these as substrings, so a header quoting the marker would make
#: every armed file read as a runtime with a failed re-arm.
REARM_FAILED_MARKER = "[stall watchdog] re-arm failed: "

#: Written when a fire that DID NOT end this process has been observed — the one
#: marker here that is written AFTER the ``Timeout (`` line rather than above it.
#:
#: THE THIRD STATE, and the reason it exists. Until this marker a fired dump had one
#: reading: the bound ended that runtime, so a reader who found the file knew the
#: process was gone. A runtime with a turn in flight is not ended by this bound any
#: more (see :func:`_holds_work`), so a fired dump now has TWO readings and they want
#: opposite responses — "this runtime hit its bound and is gone" versus "this runtime
#: is still running its turn, stalled, and needs a person or a ``lop stop``". A reader
#: cannot tell them apart from the ``Timeout (`` line, which says only that the C timer
#: expired; :func:`held_fire` answers off this marker instead.
#:
#: WRITTEN BY THE SAMPLER, NOT BY THE FIRE, and that is the only place it can be
#: written: the C timer runs no Python, so the fire itself cannot leave a word about
#: what it decided to do afterwards. The sampler is the Python thread that is still
#: alive in exactly the case this marker describes, and it observes the fire as
#: appended bytes (:func:`_record_held_fire`).
#:
#: APPENDED, not written through the armed handle — see :func:`_append_dump_line` —
#: because this is the one line of ours that lands BELOW ``faulthandler``'s output.
#:
#: NOT SPELLED IN THE HEADER, for the reason every marker here states: readers test
#: these as substrings, so a header quoting it would make an armed file read as a
#: runtime whose bound already fired and held.
HELD_MARKER = "[stall watchdog] bound held: "

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

    THE RUN IS THE TRAILING WINDOW, NOT "EVERYTHING SINCE THE RUN OPENED", and
    agent review round 2 is why the two are not the same thing. Keeping one
    baseline from the run's start makes the CPU statistic a CUMULATIVE mean that
    decays as ~1/t through later silence, so how long the session happens to
    have been alive decides whether the same burn is visible: a session idle for
    an hour before it starts spinning carries that hour into the denominator. The
    samples retained here are therefore pruned to ``last - window``, which makes
    the statistic mean exactly what its justification says — the mean over the
    window — and makes the detection latency depend on the BURN rather than on
    the session's age. The retained deque is bounded by the same window it
    measures and by nothing else: the cadence is derived FROM the window
    (``_sample_interval``), so it holds about ``PROGRESS_SAMPLES_PER_WINDOW``
    entries. A COUNT cap must not be added beside the time prune — the first
    revision had one at ``3 x PROGRESS_SAMPLES_PER_WINDOW``, and it binds in the
    wrong direction: a cap can only ever hold fewer samples than the window needs,
    so any FINER interval than ``window / 36`` (a 45 s window driven a second at a
    time is 45 looks) leaves the retained span short of the window forever and the
    leg can never fire at all. The prune is exact at every cadence; a count is a
    second and wrong bound on the same thing.

    ``motion`` is the last motion tuple seen, and ``history`` is empty whenever
    the last sample disagreed with leg 1 or leg 2 — those two are the ONLY things
    that reset a run, which is what stops a scheduled-out sample from discarding
    one (review round 1) and what lets a burn/zero alternation accumulate (review
    round 2).
    """

    __slots__ = ("motion", "history")

    def __init__(self) -> None:
        self.motion: object = _NO_SAMPLE
        #: ``(wall, process_cpu)`` per sample, oldest first, pruned to the window.
        self.history: deque[tuple[float, float]] = deque()

    def restart(self, motion: object) -> None:
        """End the run: the work moved, or something is in flight.

        ONE spelling of "there is no run", used for both disagreements and by the
        construction of a new one, because two spellings of that are two chances
        for the retained samples to disagree with the stamps beside them.
        """
        self.motion = motion
        self.history.clear()

    def observe(self, now: float, cpu: float, window: float) -> None:
        """Record this sample and drop everything that has aged out of the window.

        The prune keeps the OLDEST sample at or just before ``now - window``, so
        the retained span is the trailing window and a hair of one extra interval
        — never shorter, which is what lets :meth:`mean_rate` answer at all.
        """
        self.history.append((now, cpu))
        horizon = now - window
        while len(self.history) > 1 and self.history[1][0] <= horizon:
            self.history.popleft()

    def mean_rate(self, window: float) -> float | None:
        """The mean CPU over the retained window as a share of ONE core.

        ``None`` while the retained samples do not yet span the window: a run
        younger than the window has no mean to judge, which is the honest answer
        rather than the mean of however few samples happen to exist.
        """
        if len(self.history) < 2:
            return None
        first_wall, first_cpu = self.history[0]
        last_wall, last_cpu = self.history[-1]
        span = last_wall - first_wall
        if span < window or span <= 0:
            return None
        return (last_cpu - first_cpu) / span


class _Armed:
    """One process's armed timer: its file, its bound, and each plane's stamp.

    The handle is held OPEN for the process's life, and that is a requirement
    rather than tidiness: ``faulthandler`` writes with the bare descriptor, so
    closing the object would leave the timer firing at a closed fd.
    """

    __slots__ = (
        "path",
        "deadline_path",
        "handle",
        "seconds",
        "pid",
        "last_beat",
        "seeded_at",
        "quiet_noted",
        "probe",
        "busy",
        "held",
        "arm_size",
        "seen_fires",
        "after_fire_at",
        "fired_held",
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
        busy: "BusyProbe | None" = None,
    ) -> None:
        self.path = path
        #: Where the CURRENT deadline is recorded for a reader who arrives after the
        #: process is gone (:func:`deadline_path`), rewritten by every beat.
        self.deadline_path = path.with_suffix(DEADLINE_SUFFIX)
        self.handle = handle
        self.seconds = seconds
        self.pid = pid
        #: Each plane's last sign of life, in ``time.monotonic`` seconds. Seeded to
        #: the ARM time rather than left empty, so a plane that has simply not
        #: ticked yet (the 15 s between boot and its first tick) is measured from
        #: the arm rather than treated as infinitely silent — which would fire the
        #: bound on every healthy boot.
        #:
        #: ONE ``monotonic`` CALL FOR BOTH PLANES, so the seed is a value that can be
        #: recognised: a plane still holding it has never reported at all, which is a
        #: different fact from a plane that reported and then stopped, and the quiet
        #: line has to state the one that is true.
        seeded_at = time.monotonic()
        self.seeded_at = seeded_at
        self.last_beat: dict[str, float] = {plane: seeded_at for plane in PLANES}
        #: The planes whose quiet has already been written to the dump. A transition
        #: rather than every beat, because a quiet plane can stay quiet for hours --
        #: the case that made this artifact a body count -- and a line per 15 s beat
        #: would grow the dump without bound. The LIVE number is the sibling.
        self.quiet_noted: set[str] = set()
        #: The progress leg's probe, or ``None`` when no runtime supplied one (an
        #: in-process host, a test of the liveness leg alone, an older spawner).
        #: ``None`` means the progress leg is INERT, not that it is satisfied.
        self.probe = probe
        #: The EXIT LEG's probe, and the decision it currently holds. ``held`` is
        #: recomputed at arm time and re-read on the sampler's own wakes (see
        #: :func:`_refresh_exit_leg`), and it is a FIELD rather than a fresh call at
        #: each re-arm because ``beat`` runs on two other threads and the answer must
        #: be read from one place: only the sampler and the arming thread call the
        #: probe, so a plane's tick cannot walk the session's state from a thread that
        #: does not own it (``_beat_stall_watchdog`` and ``_heartbeat_loop`` both
        #: reach :func:`beat`).
        self.busy = busy
        self.held = _holds_work(busy)
        #: The dump's size as of the last successful arm, which is how a fire is
        #: OBSERVED from Python: the C timer appends the ``Timeout (`` line and every
        #: thread's stack, so file growth after an arm is the fire's own signature and
        #: needs no parsing on the common path (see :func:`_record_held_fire`).
        self.arm_size = _dump_size(path)
        #: How many fires this arm has already ANNOTATED. A counter rather than a flag
        #: because a runtime can survive several: each held fire re-arms for the next
        #: episode, and a second one must be recorded too rather than swallowed by the
        #: first one's bookkeeping.
        self.seen_fires = 0
        #: When the last held fire was RECORDED, which is when the next episode's bound
        #: runs from. See :func:`_rearm` for why the deadline arithmetic alone cannot
        #: answer that question once a fire has been written for the episode.
        self.after_fire_at: float | None = None
        #: Set once a fire has been observed that did NOT end this process. It keeps
        #: the artifact: see :func:`disarm`.
        self.fired_held = False
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

    def pin(self) -> tuple[str, float]:
        """Which leg's deadline the timer is armed for, and when: ``(leg, monotonic)``.

        The same ``min`` :meth:`deadline` takes, kept in ONE place because the record
        this bound writes for a reader (:func:`_record_deadline`) has to name the same
        pin the timer was armed from — two spellings of "the earliest deadline" would
        be two chances for the sibling and the file to disagree about which plane was
        quiet, which is the fact the two files exist to establish.
        """
        plane = min(self.last_beat, key=self.last_beat.__getitem__)
        deadline = self.last_beat[plane] + self.seconds
        if self.progress_deadline is not None and self.progress_deadline < deadline:
            return LEG_PROGRESS, self.progress_deadline
        return plane, deadline

    def deadline(self) -> float:
        """The earliest moment ANY leg's condition reaches its bound.

        THE EARLIEST, not the latest, and that is the whole of A2's fix: a healthy
        plane's tick must shorten the timer toward a silent plane's deadline, never
        push it out. The progress leg joins on the same rule rather than beside
        it — a beat that re-armed only the planes would push a decided progress
        fire out by a whole heartbeat, which is exactly the masking this method
        exists to prevent. See the module docstring.

        Delegates the ``min`` to :meth:`pin`, which also says WHICH leg won it: the
        dump's quiet-plane line and the deadline sibling both name that leg, so the
        choice has to be made in one place rather than restated here.
        """
        return self.pin()[1]


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


def deadline_path(pid: int | None = None, directory: Path | None = None) -> Path:
    """Where the CURRENT deadline goes: ``<log dir>/runtime-stall-<pid>.deadline``.

    THE SIBLING OF THE DUMP, one number and one name on a line (~24 bytes), rewritten
    by every :func:`beat` THAT RE-ARMED -- so a reader with a dead pid has, without
    opening anything else: the deadline the timer was armed for (compare it with the
    fire to say whether the bound came due or something else pre-empted it), the leg
    whose stamp pinned it (subtract ``bound_s`` from the deadline to place that plane's
    last report), and, in the mtime, the last SUCCESSFUL re-arm (a beat whose re-arm
    raised leaves the previous number in place and writes a line saying so -- see
    :data:`REARM_FAILED_MARKER`). A sibling rather than a line in the dump because a
    beat runs for the process's whole life: appended, it would grow the artifact
    without bound and answer nothing the last number does not (see
    :data:`DEADLINE_SUFFIX`).

    ITS ABSENCE IS A SIGNAL, not a gap: nothing writes it at :func:`arm`, so a missing
    sibling means no beat ever re-armed this timer -- the never-engaged class, whose
    fire carries the ARMING value rather than a recomputed remainder
    (:data:`HOW_TO_READ_THE_FIRED_VALUE`).

    ARM CLEARS IT, AND THAT IS NOT A CONTRADICTION OF THE PARAGRAPH ABOVE. A PID IS
    RECYCLED, so a sibling in this directory may be the previous holder's, and it would
    then be read as a beat of the life in front of the reader -- a leg naming a plane of
    a life that is over, an mtime BEFORE this process's own arm epoch, on a fire whose
    own value says no beat ever re-armed the timer. :func:`arm` therefore unlinks it, so
    the rule above holds for the holder of the pid the reader is looking at (the dump is
    truncated per life for the same reason; truncating only one of the pair was the
    defect QA round 1 drove as Q1).

    WRITTEN THROUGH A SIDECAR TEMP AND ``os.replace``, never in place: ``Path.write_text``
    is open(O_TRUNC) + write + close, so a reader concurrent with a beat observes a
    TRUNCATED file -- measured at 398 beats/s, 2500 of 28,922 reads returned zero bytes
    (QA round 1, Q4), and a fire landing in that window leaves a zero-byte sibling for a
    post-mortem that has no rule for one. The replacement is atomic, so a reader sees the
    previous deadline or the new one and never a mixture of the two. The temp is
    ``<sibling>.tmp`` and lives only for the microseconds between the two calls -- a
    surviving one means the process died inside a beat, and :func:`disarm` does not remove
    it, because a process that dies mid-beat is exactly the case the sibling itself is
    left for.

    Composed from :func:`dump_path` rather than from ``log_dir()`` and a pid a second
    time, so the two files cannot drift apart, and named the same way for the same
    reason: a reader holding a record's pid needs nothing else.
    """
    return dump_path(pid, directory).with_suffix(DEADLINE_SUFFIX)


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

#: What a runtime supplies so the EXIT LEG can be decided per re-arm: a
#: zero-argument callable answering "does this process hold work a clean exit would
#: destroy?" — a turn under the lock, a live subagent lane, a background job, a gate
#: parked on the user (``ServingSessionHandle.is_busy`` is what the runtime passes).
#:
#: WHY THE EXIT LEG IS NOT THE PROGRESS LEG. ``in_flight`` above is deliberately
#: narrow — a tool batch whose results have not landed, or an on-demand compaction —
#: because it is one term of a COMPOSITE predicate whose other terms (no motion, CPU
#: advancing) are what make a fire correct. The question the exit leg asks is a
#: different one and a simpler one: would ending this process destroy work? A model
#: call, a long tool, a running subagent and a parked gate all answer yes, and the
#: operator's rule for a build move is the same rule: a runtime is replaced when its
#: turn is COMPLETE, never on a heuristic of inactivity.
#:
#: A CALLER WITH NO PROBE KEEPS TODAY'S BEHAVIOUR; one whose probe RAISES holds the
#: exit (see :func:`_holds_work`, which spells out why those are different facts).
#: The dump is written either way, which is the whole instrument; what the hold
#: protects is a runtime killed mid-turn, because that work is the one thing here
#: that nothing can reconstruct.
BusyProbe = Callable[[], bool]


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
    busy: "BusyProbe | None" = None,
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
        inherited = deadline_path(pid, directory)
        try:
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # TRUNCATE, THEN APPEND — two steps for one property, and the property is
            # the whole reason a held fire can be annotated at all. The C timer shares
            # THIS descriptor and writes at the descriptor's own offset, so a plain
            # ``"w"`` handle leaves the two writers with different ideas of where the
            # file ends: a line this module appends after a fire lands at the real end,
            # and the NEXT fire's dump then lands back at the offset the C thread still
            # holds, ON TOP of it (measured: four fires, one surviving marker, on the
            # build of this change). With ``O_APPEND`` every write from either writer
            # goes to the end of the file as it is at that instant, which is what makes
            # "the stacks, then the note about them" an ordering a reader can rely on.
            #
            # The truncation is NOT dropped in exchange: a pid is recycled and these
            # files are keyed by pid alone, so a dump left by the previous life must not
            # outlive it into this one (see the inherited-sibling paragraph below and
            # ``journal._stall_bound_evidence``, which reads that truncation as what
            # closes one half of the recycled-pid case).
            with target.open("w", encoding="utf-8"):
                pass
            handle = target.open("a", encoding="utf-8")
        except OSError:
            logger.warning("stall watchdog could not open %s; no dump will be written", target)
            return False
        # NO INHERITED STATE, and this is the same argument as the "w" above rather than a
        # second one: a pid is RECYCLED and these files are keyed by pid alone, so a sibling
        # left by the life that held this pid BEFORE would outlive it into this one. Left
        # there it reads as a beat of THIS life -- an mtime BEFORE this process's own arm
        # epoch, a leg naming a plane of a dead life, on a fire whose own value says no beat
        # ever re-armed the timer -- and that presence/absence is the ONE thing a reader has
        # for telling the never-engaged class from a stale plane (see :func:`deadline_path`),
        # so a stale file must not be able to fake it. Best-effort: a file that cannot be
        # removed must not stop the bound being armed.
        try:
            inherited.unlink()
        except OSError:
            pass
        try:
            # THE HEADER IS WRITTEN BEFORE THE TIMER IS ARMED, never deferred:
            # faulthandler writes with a raw descriptor from a C thread, so the
            # file and its header have to exist first. Everything below this
            # line is what a reader finds when the bound fires, in this order.
            handle.write(
                f"{ARM_MARKER}pid {pid or os.getpid()} armed for {bound:g}s at {time.time():.0f} "
                f"({time.strftime('%Y-%m-%d %H:%M:%S')}); the runtime's own loops re-arm this "
                f"timer as they run, so what its expiry measures is {bound:g}s WITH NO RE-ARM FROM "
                f"THEM -- a fact about the TIMER, never a reading of what this process was "
                f"doing.\n"
                f"{OBSERVATION_NOT_VERDICT}\n"
                f"{HOW_TO_READ_THE_FIRED_VALUE}"
                f"AND IF A FIRE IS THE CLASS THIS DUMP COUNTS AS, this is the only record of it: "
                f"because that exit runs from a C thread that runs no Python, the incidents class "
                f"{_bound_class()} is written here and nowhere else, which is why death "
                f"attribution opens this file for a pid that IS gone. A further line below "
                f"carrying the words 'no progress' means the re-arm stopped while the loops that "
                f"produce it were still ticking and burning CPU; its absence means no tick "
                f"reached the timer at all. WHY THIS FILE IS STILL NAMED {DUMP_PREFIX}-<pid>.log "
                f"WHEN A FIRE IS NOT A STALL: it is written at ARM time, so one exists for every "
                f"armed runtime, healthy ones included, and this fleet's incident taxonomy and "
                f"death-attribution reader are keyed to that path -- renaming it is a "
                f"cross-module change, taken whole or not at all rather than half-done here. "
                f"NO MARKER IS SPELLED HERE, and that is load-bearing rather than tidy: "
                f"readers test for each as a SUBSTRING, so a header quoting one would make "
                f"every armed file -- including one left by a SIGKILL -- read as a fired bound, "
                f"a progress fire, or a runtime with a quiet plane.\n"
            )
            handle.flush()
            to_arm = _Armed(target, handle, bound, pid or os.getpid(), probe, busy)
            _rearm(to_arm, remaining=bound)
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
        _ARMED = to_arm
        if probe is not None or busy is not None:
            # EITHER PROBE EARNS THE SAMPLER, and ``busy`` alone is a reason: the
            # sampler is what re-reads the exit leg while the timer is pending
            # (:func:`_refresh_exit_leg`), so a caller that supplies only the busy
            # probe would otherwise hold an exit leg that could never flip back — a
            # runtime that picked up a turn after its last arm would be ended by a
            # timer armed while it was idle.
            _start_sampler(_ARMED)
        return True


def _write_dump_line(armed: "_Armed", line: str) -> bool:
    """Append one line to the armed dump, with ONE retry through a fresh descriptor.

    THE RETRY IS LOAD-BEARING rather than tidy, and it lives here rather than at the
    three call sites because each of their lines is the only record of a different
    fact -- which leg fired, that a plane's reporter died, that a plane has gone quiet
    -- so an unwritable line is a lost reading and not a cosmetic failure. A closed or
    stale held descriptor is what this recovers; a permanently unwritable directory is
    not, which is why the callers keep their own warning about what is then missing.

    Written from a loop or a sampler thread, ABOVE any dump the C timer later appends:
    ``faulthandler`` writes at the descriptor's offset, so write-then-fire is what
    makes the line and the stack readable as one chronology.
    """
    try:
        armed.handle.write(line)
        armed.handle.flush()
        return True
    except (OSError, ValueError):
        logger.debug("stall watchdog could not write a dump line", exc_info=True)
        try:
            with armed.path.open("a", encoding="utf-8") as spare:
                spare.write(line)
            return True
        except OSError:
            return False


def _holds_work(busy: "BusyProbe | None") -> bool:
    """Does this process hold work a clean exit would destroy?

    THREE ANSWERS, NOT TWO, and the two negative ones are different facts:

    * **the probe answers** (``True``/``False``) — the arm holds iff it says so;
    * **the probe RAISED** → ``True``. Here the process DID tell us it has a way to
      report work and the report could not be read, so the unreadable state must not
      authorise a cut. ``True`` is also this module's fail-safe direction for the one
      decision that is irreversible: the dump is written either way, so the cost of
      holding wrongly is one ``lop stop`` against a turn that nothing can
      reconstruct;
    * **no probe was supplied** → ``False``, i.e. TODAY'S BEHAVIOUR. This is a caller
      that has been given no way to report what it is doing (a rig, an in-process
      host, a test of the liveness leg alone), and the bound's documented contract
      for such a caller is unchanged rather than silently withdrawn. The distinction
      matters because the ONE production arm site (``process._start_runtime``) always
      supplies the probe: there is no released runtime that can answer and does not,
      so holding on absence would only ever disarm the bound for the callers that
      cannot speak — the dead-instrument shape, not a safety win.
    """
    if busy is None:
        return False
    try:
        return bool(busy())
    except Exception:  # noqa: BLE001 — uncertainty keeps the runtime
        logger.debug("stall watchdog: the busy probe failed; holding the exit leg", exc_info=True)
        return True


def _dump_size(path: Path) -> int:
    """The dump's size in bytes, or ``-1`` when it cannot be read.

    ``-1`` is deliberately an answer no size can equal, so a caller comparing
    against it (see :func:`_record_held_fire`) cannot mistake an unreadable file for
    an unchanged one — an unreadable dump is not evidence that nothing was written
    to it, and the module's rule is that a dead instrument reports nothing rather
    than reporting a healthy reading.
    """
    try:
        return path.stat().st_size
    except OSError:
        return -1


def _fired_count(path: Path) -> int:
    """How many fires this dump carries, oldest first — ``_fires`` as a count.

    The text parse, reached only when the file GREW (see :func:`_record_held_fire`),
    because the count is what makes an annotation idempotent: a held fire re-arms the
    timer, so a second fire appends a second ``Timeout (`` line, and a reader that
    kept only a boolean would take the second one for the first.

    ``0`` for an unreadable file: this counts evidence, and an instrument that cannot
    be read has none to report rather than all of it.
    """
    text = _dump_text(path)
    return sum(1 for line in text.splitlines() if line.startswith(FIRED_MARKER))


def _arm_timer(handle: IO[str], remaining: float, *, exit_leg: bool) -> None:
    """Arm (or replace) the C timer, for ``remaining``, with the exit leg given.

    ONE SPELLING FOR THE THREE SITES that reach ``dump_traceback_later`` (the arm,
    a plane's beat, the progress leg's own fire) so that no site can leave the exit
    leg at a default: ``exit_leg`` is a keyword-only argument with no default, and
    the whole defect this change fixes was one literal ``exit=True`` written at a
    re-arm site that the re-arm itself had the information to answer differently.

    REPLACING IS THE ARMED STATE: a second call supersedes the pending timer rather
    than adding one (measured, and the module docstring's inventory says so), so
    there is never a second timer to reason about — which is what lets the exit leg
    be re-decided on every re-arm rather than only at the arm.
    """
    faulthandler.dump_traceback_later(max(MIN_REARM_S, remaining), file=handle, exit=exit_leg)


def _rearm(armed: "_Armed", *, remaining: float | None = None) -> None:
    """Re-arm for the earliest deadline, with the exit leg this runtime holds.

    A FIRE IS RECORDED BEFORE THE NEW TIMER REPLACES THE OLD ONE, and the order is
    load-bearing rather than tidy. A fire that lands between two beats has appended
    bytes; the next beat's re-arm resets ``arm_size`` to whatever the file now holds,
    so a fire observed only by size would be erased by the very re-arm that should
    have recorded it — and the runtime would then be re-armed from a deadline already
    in the past, writing a dump per beat. Recording first makes the growth
    un-observable to the re-arm that follows it (:func:`_record_held_fire` re-enters
    here once, with the count already advanced, and sees nothing new).

    ``remaining`` overrides the deadline arithmetic for the one case that cannot use
    it: after a fire has already landed, the stamps that produce
    :meth:`_Armed.deadline` are in the past by construction, so re-arming off them
    would fire immediately, writing a dump per re-arm instead of one per episode —
    the accumulation this module refuses everywhere else (see
    ``_note_quiet_plane``). A held fire therefore restarts the bound from the fire.
    """
    _record_held_fire(armed)
    if remaining is None:
        remaining = armed.deadline() - time.monotonic()
        if armed.after_fire_at is not None:
            # A FIRE FOR THIS EPISODE IS ALREADY WRITTEN, so the stale deadline that
            # produced it must not produce another one on every re-arm. The arithmetic
            # above reads the plane stamps, and a plane that has been silent past its
            # bound stays silent: a healthy OTHER plane beating every 15 s would each
            # time recompute a deadline in the past and fire immediately, writing a dump
            # per beat (four a minute, ~100 MB a day, on a runtime that has already been
            # reported as stalled). So the next episode's bound runs from the fire that
            # ended the last one. A plane that REPORTS again revives the ordinary
            # arithmetic on its own, because its stamp is then in the future.
            remaining = max(remaining, armed.seconds - (time.monotonic() - armed.after_fire_at))
    _arm_timer(armed.handle, remaining, exit_leg=not armed.held)
    armed.arm_size = _dump_size(armed.path)


def _append_dump_line(armed: "_Armed", line: str) -> bool:
    """Write one line to a dump that ``faulthandler`` may ALREADY have written to.

    THE SAME WRITER AS EVERY OTHER LINE, and it delegates to :func:`_write_dump_line`
    rather than opening the file itself: the descriptor is shared with the C timer and
    is opened ``O_APPEND`` (see :func:`arm`), so a write through the handle lands after
    whatever is already there — including a fire's stacks — exactly as an independent
    append would, and with one fewer writer to reason about. An earlier revision of
    this function appended by path to dodge the handle's stale position; that dodge was
    aimed at the wrong writer, and the fix it needed is the append mode rather than a
    second open (the measured failure was the NEXT fire overwriting the marker).

    Gets its own name so the call site reads as what it is — the one line here that
    lands BELOW a fire rather than above it — and so that the next reader finds this
    note instead of re-deriving the offset rules from scratch.
    """
    return _write_dump_line(armed, line)


def _record_held_fire(armed: "_Armed") -> bool:
    """Record a fire that did NOT end this process, and settle what happens next.

    THE FIRE'S OWN AFTERMATH, which until this change had no writer anywhere: the C
    thread dumps every thread and returns, and nothing in Python is told that it
    happened. What the sampler can see is that the file GREW past the size recorded
    at the last arm and that it is still running — which together are exactly "a fire
    landed and it did not end this runtime", with no parsing of the dump text on this
    path at all.

    A CONSEQUENCE WORTH STATING FOR THE CALLER: this is reached from :func:`_rearm`,
    which the BEAT path also runs, so a healthy plane's next beat records a held fire
    as surely as the sampler does. That is deliberate — the sampler is the thread that
    is guaranteed to be awake, and the beat is the one that may arrive first — and it
    is why the detection must be idempotent (``seen_fires``) rather than
    once-per-arm.

    Returns whether a fire was recorded. Order matters: the marker is written first,
    so a reader never sees the re-armed timer without the statement about the fire
    that preceded it.

    WHAT HAPPENS NEXT depends on the ONE question the exit leg asks, read again here:

    * the work is still in flight — re-arm for a full bound. The runtime is stalled
      with a turn in it, the person or the ``lop stop`` is the way out, and one dump
      per bound says so without writing a file per re-arm;
    * the work has CLEARED since — drop the hold and arm for an immediate fire, i.e.
      arm the exit leg fatally. This is the wedge recovery the bound exists for, and
      it is preserved for exactly the state the operator's rule names: nothing in
      flight, so there is no work a cut could lose. Without this the bound would
      become unfirable for a runtime that recovered from a stall it survived.
    """
    if not armed.held:
        # Not ours: a fatally armed fire leaves no process behind to annotate it.
        return False
    size = _dump_size(armed.path)
    if size < 0 or size <= armed.arm_size:
        return False
    fires = _fired_count(armed.path)
    if fires <= armed.seen_fires:
        # The file grew without a fire, which only this module's own lines can do.
        armed.arm_size = size
        return False
    armed.seen_fires = fires
    armed.fired_held = True
    armed.after_fire_at = time.monotonic()
    _append_dump_line(
        armed,
        f"{HELD_MARKER}the bound fired at "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} and did NOT end this runtime: a turn, a "
        "subagent or a job is in flight, and the runtime is now STALLED rather than "
        "gone. Every thread's stack is above. Nothing here stops it; stop it with "
        "`lop stop` if the work is not going to finish.\n",
    )
    if _holds_work(armed.busy):
        armed.held = True
    else:
        armed.held = False
    try:
        # The recursive call is the point: it re-arms for the next episode THROUGH
        # ``_rearm`` so that the count this call has just advanced (with
        # ``after_fire_at``) is what makes the re-entrant call a no-op, and so that the
        # fresh ``arm_size`` is set by the one place that arms. That is also what keeps
        # the caller's ``except`` below meaningful — a re-arm that raises does not leave
        # the file mis-described.
        _rearm(armed)
    except (OSError, ValueError, RuntimeError):
        logger.warning("stall watchdog could not re-arm after a held fire", exc_info=True)
    logger.warning(
        "stall watchdog: the bound fired for pid %s and did NOT end this runtime "
        "(work in flight); the runtime is STALLED, dump at %s",
        armed.pid,
        armed.path,
    )
    return True


def _refresh_exit_leg(armed: "_Armed") -> None:
    """Re-read the exit leg on the sampler's cadence, and re-arm when it flips.

    THE ONE THING THAT KEEPS THE EXIT LEG CURRENT. The decision is made where the
    probe can be read safely (this thread, and the arming thread), while the timer
    fires between beats: a runtime that picks up a turn after its last arm would
    otherwise be ended by a timer that was armed while it was idle, which is the
    defect in its other order. The sampler's interval is what bounds the staleness of
    the decision — the same interval the progress leg already accepts for its own
    reading (``_sample_interval``), and far inside the bound it guards.

    Only a FLIP re-arms. A re-arm on every sample would push the deadline out every
    fifteen seconds and the liveness leg could then never fire at all — a dead
    instrument, in this module's own words.
    """
    held = _holds_work(armed.busy)
    if held == armed.held:
        return
    armed.held = held
    try:
        _rearm(armed)
    except (OSError, ValueError, RuntimeError):
        logger.debug("stall watchdog could not re-arm after an exit-leg flip", exc_info=True)


def _record_deadline(armed: "_Armed") -> None:
    """Write the deadline the timer is now armed for, and the leg that pinned it.

    CALLED AFTER A SUCCESSFUL RE-ARM AND ONLY THEN. ``faulthandler`` keeps the
    previous deadline when a re-arm raises, so a record written anyway would state a
    number the timer never got -- the sibling is the one place a reader goes to settle
    "overdue, or pre-empted by something else", and it has to answer with what was
    actually in force.

    NEVER RAISES. This runs inside :func:`beat`, on a serving loop's thread, and a
    diagnostic that cannot write its own number must not take that loop down.
    """
    # WRITE-THEN-RENAME, so a reader concurrent with this beat never observes a half-written
    # sibling (see :func:`deadline_path`): the temp is this process's own pid-keyed name in
    # the same directory, and every caller holds ``_LOCK``, so the name is not contended.
    # A failure takes the temp with it, because a file per failed beat is the accumulation
    # this artifact's sibling design exists to avoid.
    temp = armed.deadline_path.with_name(armed.deadline_path.name + ".tmp")
    try:
        leg, deadline = armed.pin()
        epoch = time.time() + (deadline - time.monotonic())
        temp.write_text(f"{epoch:.3f} {leg}\n", encoding="utf-8")
        os.replace(temp, armed.deadline_path)
    except (OSError, ValueError) as exc:
        logger.debug("stall watchdog could not record its deadline: %s", exc)
        try:
            temp.unlink()
        except OSError:
            pass


def _note_quiet_plane(armed: "_Armed", now: float) -> None:
    """Name, in the dump, a plane that has reported nothing for half the bound.

    THE QUESTION THIS ANSWERS is the first one asked of a fire -- which loop stopped
    reporting, and when did it stop -- and no artifact could answer it: the dump
    carried the arm epoch and nothing else, while the stamps that do answer it
    (:attr:`_Armed.last_beat`) are memory and die with the process. So the transition
    is written where the report about that process already is, ABOVE any dump the bound
    later appends, exactly as :func:`note_tick_death` argues for its own line.

    A TRANSITION, ONCE PER PLANE, and the live number is the sibling: a plane can stay
    quiet for hours (the case that made this artifact a body count), and a line per
    15 s beat would grow the dump without bound. The threshold is :data:`QUIET_FRACTION`
    of the bound, which no healthy plane reaches -- a healthy plane's stamp age is 0-15 s
    against a reporting point of 150 s at the 300 s bound.

    A NO-OP WHEN NOTHING IS ARMED, or when the dump cannot be written: this runs on a
    beat, and a lost diagnostic must not fail the loop that reported progress.
    """
    for plane in PLANES:
        age = now - armed.last_beat[plane]
        if age < armed.seconds * QUIET_FRACTION or plane in armed.quiet_noted:
            continue
        other = SERVING if plane == WORKLOAD else WORKLOAD
        if armed.last_beat[plane] == armed.seeded_at:
            # A plane that never reported at all is a different fact from one that
            # reported and then stopped, and the line has to state the true one: the
            # stamp is the ARM seed, so there is no report to point at.
            reported = f"has reported nothing since the ARM at {time.time() - age:.0f}"
        else:
            last = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - age))
            reported = f"last reported {age:g}s ago ({last})"
        armed.quiet_noted.add(plane)
        line = (
            f"{REARM_MARKER}{plane} {reported} against a {armed.seconds:g}s bound; the {other} "
            f"plane reported {now - armed.last_beat[other]:g}s ago. The timer is re-armed from the "
            f"quiet plane's deadline until it fires, so the value on a fired line below is that "
            f"remainder rather than the bound.\n"
        )
        if not _write_dump_line(armed, line):
            logger.warning(
                "stall watchdog could not record the quiet %s plane for pid %s",
                plane,
                armed.pid,
            )


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
            _rearm(armed)
        except (OSError, ValueError, RuntimeError):
            # A beat that cannot re-arm must not take the loop down with it: the
            # timer is still armed — from the previous beat, or from ``arm`` — so
            # the worst case is a bound that expires sooner than intended.
            logger.warning("stall watchdog could not re-arm its timer", exc_info=True)
            # AND THE SIBLING IS NOW BEHIND THE LAST BEAT: a failed re-arm leaves the
            # previous deadline in force, so ``_record_deadline`` is not called and the
            # file keeps the number AND the mtime of an earlier beat -- while a reader is
            # told (HOW_TO_READ_THE_FIRED_VALUE) to read that mtime as the last beat. The
            # sibling cannot carry the correction itself (its content is the deadline in
            # force, which is exactly right), so this beat says it in the dump, beside the
            # stacks it will be read with. An unwritable dump leaves the warning above.
            if not _write_dump_line(
                armed,
                f"{REARM_FAILED_MARKER}the timer could not be re-armed at this beat, so "
                f"{DUMP_PREFIX}-<pid>{DEADLINE_SUFFIX} still holds the PREVIOUS deadline and "
                f"its mtime is behind this process's last beat\n",
            ):
                logger.warning(
                    "stall watchdog could not record a failed re-arm for pid %s", armed.pid
                )
        else:
            # AFTER a successful arm, and only then: the deadline this process is now
            # armed for, for a reader who arrives after the process is gone.
            _record_deadline(armed)
        # A stamp fact rather than a re-arm fact, so it is recorded either way: this is
        # which plane has gone quiet, and no other artifact carries it.
        _note_quiet_plane(armed, now)


def note_tick_death(plane: str, reason: str) -> bool:
    """Record durably that ``plane``'s TICKER died, in the armed dump file.

    Beside the plane's own stamp rather than in a file of its own, and that is a
    choice about WHAT A READER HAS IN HAND rather than about tidiness: the stamp
    (:attr:`_Armed.last_beat`) is memory and dies with the process, and the only
    artifact that outlives the ``_exit(1)`` this bound makes is the dump. So the
    fact that a plane stopped being REPORTED goes where the report about that
    process already is, and it is written ABOVE any dump the bound later writes,
    which is what makes the two readable as one chronology.

    Returns whether the line landed, so the caller's own WARNING can say which
    of the two shapes it is in: recorded (a reader will see it) or not (the
    caller's log line is then the only trace). A no-op when nothing is armed,
    which is the whole in-process case — a TUI host, a test — where there is no
    dump file to write into and the caller's log line is the record.

    A FAILURE HERE MUST NOT TAKE THE CALLER DOWN. This runs on the path that has
    just seen a task die, and an unwritable log directory is a reading of "no
    record possible" rather than a second failure; :func:`_fire_progress` makes
    the same call for the same reason.
    """
    line = f"{TICK_DEATH_MARKER}{plane}: {reason}\n"
    with _LOCK:
        armed = _ARMED
        if armed is None:
            return False
        if _write_dump_line(armed, line):
            return True
        logger.warning(
            "stall watchdog could not record the death of the %s tick for pid %s",
            plane,
            armed.pid,
        )
        return False


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
            # BEFORE the sample, and in this order: a fire that landed since the
            # last wake is recorded and re-armed on the same pass, and only then is
            # the next progress decision taken. The exit leg is re-read first of all
            # so that a flip made while the timer was pending is in force before
            # anything else can act on the arm.
            _refresh_exit_leg(armed)
            if _record_held_fire(armed):
                continue
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
    probe = armed.probe
    try:
        motion, in_flight = probe() if probe is not None else (_NO_SAMPLE, True)
    except Exception:  # noqa: BLE001 — an unevaluable probe must not end a process
        logger.debug("stall watchdog: progress probe failed", exc_info=True)
        motion, in_flight = _NO_SAMPLE, True
    moved = clock.motion is not _NO_SAMPLE and motion != clock.motion
    if in_flight or moved:
        # A disagreeing sample ends the run outright: a leg that is not true NOW
        # is not "possibly true", and a window that kept accumulating across a
        # sample where the process was working would fire on a runtime that had
        # done legitimate work inside it.
        clock.restart(motion)
        return False
    clock.motion = motion
    clock.observe(now, cpu, armed.seconds)
    # THE FLOOR IS TESTED ONLY HERE, against the mean over the trailing window,
    # and never as a per-sample gate — that is the whole of both review rounds'
    # predicate findings. Round 1: a per-sample test discarded a run on one
    # scheduled-out sample. Round 2: restarting the run whenever the mean dipped
    # carried an early burst through later silence without bound AND never fired
    # on a burn/zero alternation, because the run kept re-opening on the burn
    # half. A ``None`` mean means the run is younger than the window, which is
    # not yet a claim about anything.
    mean = clock.mean_rate(armed.seconds)
    if mean is not None and mean >= PROGRESS_CPU_FLOOR:
        _fire_progress(armed, now)
        # A HELD FIRE DOES NOT END THE SAMPLER, and it must not: the sampler is the
        # only Python thread left in exactly that case, and the next wake is where a
        # held fire is recorded and the bound is re-armed for the episode after it
        # (``_record_held_fire``). Returning here would leave the dump unannotated,
        # the timer unfired-again and the runtime's stall undetectable — a fire that
        # is silent about itself, which is the class of defect this module exists to
        # end.
        return not armed.held
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
    line = (
        f"{PROGRESS_MARKER}{_bound_class()}: {armed.seconds:g}s of CPU with no progress "
        f"from the work -- no transcript, roster or job movement, no tool batch in "
        f"flight, and at least {PROGRESS_CPU_FLOOR:.0%} of a core burned as a mean across "
        f"every sample of the window.\n"
    )
    # THIS LINE IS THE ONLY THING THAT SAYS WHICH LEG FIRED, and its absence is read as
    # the SILENCE leg — so a progress fire whose line could not be written would have
    # its own detail narrate the other predicate (agent review round 1, NIT 1). The
    # retry through a fresh descriptor is ``_write_dump_line``'s; the warning is here
    # because THIS caller is the one whose missing line changes a reader's verdict.
    if not _write_dump_line(armed, line):
        logger.warning(
            "stall watchdog could not record which leg fired; the dump for pid %s "
            "will read as the silence leg",
            armed.pid,
        )
    try:
        _rearm(armed)
    except (OSError, ValueError, RuntimeError):
        # The deadline STAYS set, deliberately. The timer already armed from the
        # last beat is at most one heartbeat away, and every beat re-arms from
        # ``deadline()``, so a failed re-arm here still ends the process within
        # that heartbeat rather than silently withdrawing a decision already
        # taken. Withdrawing it would make a transient arming failure a way for
        # this leg never to fire again.
        logger.warning("stall watchdog could not fire its progress leg", exc_info=True)
    else:
        # The progress leg's own deadline is what the timer is now armed for, so the
        # sibling has to be rewritten here too: leaving the last beat's number would
        # state a deadline that is no longer in force.
        _record_deadline(armed)


def disarm() -> None:
    """Cancel the bound and remove its files: this process left on its own terms.

    Removing them is what makes their EXISTENCE mean something — see "THE FILE
    IS THE EVIDENCE" in the module docstring, including why these ``unlink`` calls
    are allow-listed rather than replaced by an in-place rewrite. The deadline
    sibling goes with the dump and on the same argument: it says a beat re-armed
    this timer, and a runtime that left cleanly is not one a reader should find a
    deadline for. Unconditionally safe to call, and called on every clean exit
    path.

    ONE EXCEPTION, AND IT IS THE POINT OF THE EXCEPTION: a runtime that SURVIVED a
    fire (``armed.fired_held``) keeps both files. Its dump is not the record of an
    ending, it is the record of a stall — the stacks of a turn that was still in
    flight — and a clean exit minutes or hours later would otherwise erase the one
    artifact that says the bound fired and the runtime kept going. ``held_fire``
    reads it, and the next runtime to draw this pid truncates it at its own arm.
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
        if not armed.fired_held:
            try:
                armed.path.unlink()
            except OSError:
                pass
            try:
                # Beside the dump (see the docstring): a surviving sibling would advertise
                # a deadline for a process that disarmed, and the sibling's absence is
                # half of how a never-engaged fire is told apart from a stale-plane one.
                armed.deadline_path.unlink()
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


def held_fire(pid: int | None = None, directory: Path | None = None) -> bool:
    """Did this pid's bound fire and NOT end the runtime? The third state.

    A fired dump has two readings now, and a reader that cannot tell them apart
    gets the operator's question exactly backwards: "this runtime hit its bound and
    is gone" wants a post-mortem and a successor, while "this runtime hit its bound
    with work in flight and is still running, stalled" wants the person who owns
    that work or a ``lop stop``. `Timeout (' is written by both — it says the C
    timer expired and nothing about what the timer was armed to do — so the answer
    comes off :data:`HELD_MARKER`, which only the surviving case can write, because
    writing it takes a live Python thread after the fire.

    ``False`` for a pid with no fire at all, for a fatal fire, and for anything
    unreadable: the flag is the marker's presence, and the quiet direction is the
    one that cannot narrate a runtime as stalled when it never fired.
    """
    text = _dump_text(dump_path(pid, directory))
    return _fires(text) and any(line.startswith(HELD_MARKER) for line in text.splitlines())


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

    A MISSING PROGRESS LINE MEANS SILENCE, WITH ONE RESIDUAL, stated here rather
    than left to be discovered: the progress leg's line is written a moment before
    it arms, and :func:`_fire_progress` retries once through a fresh descriptor
    when that write fails — but a directory that is permanently unwritable, or a
    full disk, leaves a progress fire with no line, and this function then answers
    ``silence``. The class is unaffected (both legs are ``STALL_BOUND_CAUSE``);
    only the detail names the wrong predicate. The progress fire warns at WARNING
    when it cannot record its leg, so the case is not silent in ``runtime.log``.
    """
    text = _dump_text(dump_path(pid, directory))
    if not _fires(text):
        return None
    if any(line.startswith(PROGRESS_MARKER) for line in text.splitlines()):
        return LEG_PROGRESS
    return LEG_SILENCE


def tick_deaths(pid: int | None = None, directory: Path | None = None) -> tuple[str, ...]:
    """Which planes' TICKERS died for this pid, oldest first, in this artifact.

    The reader for :func:`note_tick_death`, and the counterpart of
    :func:`fired_leg`: both answer questions a THREAD dump structurally cannot,
    and a reader holding a fired bound needs both. ``fired_leg`` says the bound
    fired and which predicate it believes ended the runtime; this says whether
    the instrument that feeds that predicate was still alive when it fired — a
    ``silence`` verdict beside a non-empty tuple here means the workload plane
    did not go silent, its REPORTER did, and those two want different
    investigations (one is a parked loop, the other a defect in this module's
    own supervision).

    Empty when the file is missing, unreadable, or carries no such line — the
    quiet direction, as in :func:`_dump_text`.
    """
    text = _dump_text(dump_path(pid, directory))
    return tuple(
        line[len(TICK_DEATH_MARKER) :].split(":", 1)[0].strip()
        for line in text.splitlines()
        if line.startswith(TICK_DEATH_MARKER)
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
        text = _dump_text(path)
        if not _fires(text):
            continue
        suffix = path.name[len(DUMP_PREFIX) + 1 : -len(".log")]
        if suffix.isdigit():
            fired.add(int(suffix))
    return fired
