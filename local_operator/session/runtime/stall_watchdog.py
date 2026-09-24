"""Bound a runtime's own stall with thread dumps; production expiry never exits it.

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

So a runtime now bounds its own stall by preserving diagnostic evidence. Automatic
termination remains disabled until idle state and every work-admission path can be
linearized together; an operator must inspect and stop a runtime that stays wedged.

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
``write(2)`` straight to a file descriptor without the GIL. Production configures
it with ``exit=False`` because a Python idle sample cannot be atomic with every
work-admission path, and the native timer cannot re-check state before expiry.

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
so the watchdog is a diagnostic ceiling on one silent synchronous step: a
slow step may produce a dump, and one that never returns requires operator
intervention.

A TIMER FIRE IS NOT A RETIREMENT DECISION. Production uses the all-thread dump
as evidence, but automatic termination is disabled: a Python idleness sample is
not atomic with work admission, and the C callback cannot check again at expiry.
The operator's rule remains: "Runtimes should only update when their turn is
complete (fully idle), not based on some heuristic of inactivity that kills a
running process to update it."

Measured against the fleet before this change, the two acts were not the same
event at all. Of the 40 ``runtime-stall-*.log`` dumps on this machine (37 fired,
3 header-only; re-derived by the peer session that owns the census), 36 of the
fires are LIVENESS fires against exactly one progress fire — the common fire is a
runtime that stopped REPORTING, not one caught spinning. The event-loop thread's
stack shows a runtime DOING WORK more often than a runtime that has stopped: read
directly here, ``session._run_turn`` -> ``_emit`` -> ``serving._refresh_state`` ->
``comms.roster`` (pids 78817, 30990) and a boot still inside ``transcript.__init__``
on an executor thread (pid 5527). The stack dump is the evidence worth keeping;
the process-ending decision is unsafe because it may end a turn.

WHY THOSE PLANES WERE SILENT IS A HYPOTHESIS, NOT A READING THIS CHANGE OWNS.
The leading candidate is a reporter that stopped running rather than a loop that
stopped working — the two plane stamps are written by two different loops
(``process._beat_stall_watchdog`` and ``RuntimeServer._heartbeat_loop``), and the
record heartbeat a reader sees is NOT the same clock as either stamp, so a fresh
``beat_lag_s`` beside a stale stamp is the expected pair rather than a
contradiction. What would falsify it: a fire whose dump shows a parked turn AND an
arm that went stale at the same instant, i.e. evidence that the reported work had
genuinely stopped rather than that its reporter had. Until such a reading exists,
this module's comments do not claim the bound fired *because* a loop was parked,
and the change below does not depend on which of the two it was: an in-flight turn
must not be ended either way.

Production timer expiry is always dump-only, and the marker a surviving fire leaves
says WHICH reading it earned: :data:`HELD_MARKER` when work was in flight at the fire,
:data:`OBSERVED_MARKER` when it was idle or the work had cleared (see
:func:`_record_held_fire`). Neither a sampled busy state nor sampled idleness changes
the native timer's ``exit=False`` policy. A fired dump is evidence that the bound
expired and that this process SURVIVED it, never proof that the runtime was wedged and
never, on its own, evidence that it died. WHICH BUILD WROTE THE FILE is what decides
whether a fire could have ended anything at all, and the header states that now —
:data:`DUMP_POLICY_MARKER` here, a LEGACY fatal-policy sentence there — which is how
:func:`fire_outcome` answers without inferring anything from a missing marker.

**Why automatic idle termination is deliberately unavailable.** The probe is read
in Python, outside the admission lock; work can arrive after an idle ``False`` and
before the C timer expires, and the native callback cannot re-check that state.
Without a shared native admission/retirement barrier, the sample cannot authorize
a cut. The runtime keeps serving and the operator must inspect the dump and stop
it explicitly if it remains wedged. This loses automatic stale-idle cleanup and
successor startup; cooperative update retirement retains its separate, existing
work-aware gate.

**The diagnostic cost is explicit.** Those five measured freezes were burning
~0.9 core and had to be reaped by hand. They still require operator intervention
if wedged, but the dump now provides stacks to investigate. Every production fire
leaves the runtime resident, even if the latest sample said idle. That is the
fail-closed cost until native admission is synchronized with termination. A runtime
may recover on its own; if it stays wedged, the operator must inspect and stop it
explicitly. Cooperative update retirement retains its independent work-aware gate.

UNKNOWN SPLITS TWO WAYS. A probe that RAISES records the fire as HELD
(:func:`_holds_work`), because the runtime did report a way to speak and that report
could not be read — the answer that cannot say "nothing was in flight"; a caller that
supplied NO probe is recorded as not-held, because it is a rig or a reduced host with
no way to speak at all — and the one production arm site always supplies one, so
calling absence "held" would only ever mark states no real runtime reaches.

A SUBAGENT LANE IS IN FLIGHT AND STILL INVISIBLE TO LEG 2, and the asymmetry is
deliberate rather than an oversight: ``is_busy`` counts live subagent lanes and jobs
(so a fire over them is recorded as HELD — see below), while leg 2's narrower reading
is left exactly as it was, because widening THAT is a change to when the dump is
written, and a dump is not destructive. Widening it is called out as its own change
below and is still not this one.

AND THAT CEILING WAS BEING APPLIED TO RUNTIMES THAT WERE DEMONSTRABLY WORKING,
WHICH IS WHAT THE EXECUTION OBSERVATION BELOW FIXES
----------------------------------------------------------------
Measured on this machine's own ``runtime-stall-*.log`` store (2026-09-22, 40 dumps:
37 fired, 3 header-only): in 20 of the 37 fired dumps the loop thread's stack runs
through the runtime's own per-event subagent projection path — ``session._run_turn
-> _emit -> handler -> _refresh_state -> set_subagent_details`` and the
``serving._notify -> _publish_busy -> _publish_subagents -> subagent_counts ->
nodes -> node -> _describe`` walk — i.e. the loop was EXECUTING this tree's code
at the instant the deadline expired, not parked in a call it never returned from.
The other fires are parked-frame cases: the loop sitting in ``selectors.select``
(12, of which 11 carry a full ``0:05:00``-class arm — a plane that never stamped,
with its main thread idle since boot), or inside a GIL-holding scan
(``tools.builtin._mask_open_key_block``, 2), which is the class this bound exists
for.

A STAMP IS NOT THE ONLY SIGN OF LIFE, and that is the whole of the fix. The
liveness leg's question is "is anything running at all", and it was answering it
with one instrument — a beat — so a loop that could not deliver a beat was read as
a loop that had stopped. It had not: the WORKLOAD tick is an ``asyncio`` task on
the very loop whose frames were moving, so a single long synchronous stretch (or a
stretch of individually-costly callbacks) starves the stamp by starvation rather
than by wedging. The sampler thread runs either way, so it now also takes a second
observation per interval — the top frame of each plane's loop, as
``(filename, lineno, function)`` — and a frame that CHANGED since the previous look
is recorded as a sign of life (:meth:`_Armed.observe_execution`), extending that
plane's deadline instead of firing on it. WHICH LOOP IS WATCHED is learned, not
guessed: a beat is issued from the loop it reports for, so the THREAD OBJECT that
stamped is kept and the frame is read only while that thread ``is_alive()``. The
object rather than its ident is a correctness requirement — idents are recycled
once a thread ends (measured on this host: six sequential short-lived threads
collapsed to one ident), so a stored ident would end up naming whatever thread
holds it next and this plane would be watched through a stranger's frames.

WHY THIS DOES NOT UNBOUND ANYTHING, the argument to read before changing it. The
two legs partition the state space: the liveness leg owns "this loop never came back
from a call" and the progress leg owns "this loop is running while its work stands
still". A loop whose frames move is in the SECOND class by construction, so
abstaining on the first leg hands the observation to a progress predicate rather
than dropping it. Production timers remain dump-only in either case. Three properties
keep that diagnostic hand-off honest: (a) the extension is refused unless a probe was supplied AND
answered on that very sample, so a runtime whose progress leg is inert or unevaluable
keeps today's behaviour rather than becoming unbounded; (b) the deadline is always
``sign_of_life + bound`` and ``sign_of_life`` only moves on an extension GRANTED by
the sample in hand — the timestamp behind it has exactly ONE writer, the granted
path, so an observation alone can never move a deadline a sibling plane's ``beat``
re-arms the shared timer from — and nothing accumulates, so no counter can be left
raised; and
(c) a FROZEN frame is never extended, which is the incident this bound was written
for — a C-level matcher holding one frame for hours still fires exactly as before,
and :func:`executing_planes` is the reader that tells the two apart in the artifact.

WHAT THE SECOND OBSERVATION CANNOT SEE, MEASURED RATHER THAN ASSUMED. It sees CALLS,
not instruction-level progress: ``frame.f_lineno`` read off a FOREIGN frame is only
refreshed at a call or a suspension, so a body of bare arithmetic reads as one frozen
line however fast it runs. Measured on this host (2026-09-22, CPython 3.14.3, sampled
every 20 ms from another thread for 2 s per shape, top frame triple read through
``sys._current_frames``): a body of repeated ``total += 1`` with no call in it changed
the triple on 0 of 60 reads; ``while not stop.is_set()``, one call per iteration, on 11
of 54; a call-heavy loop on 38 of 58; and the projection walk's own shape
(``project -> describe -> describe_one -> live_twin -> <genexpr>``) on 35 of 55 — about
two reads in three. Two consequences are stated rather than papered over: the target
case (the 20 fires with the loop inside this tree's projection path) is comfortably
inside what this can see, and a RUNNING loop holding one CALL-FREE frame is still read
as silent and still cut by the liveness leg. That residual is a runaway shape rather
than a working one, and the PROGRESS leg already bounds it — it burns a core with its
work standing still — so its outcome is a fire carrying the progress marker instead of
this one. A legitimate in-memory computation that holds a single frame past the bound
while its work moves is the one case still cut that this change does not rescue; it was
cut before the change too.

THE FAILURE MODE THIS OPENS, named rather than hidden: a loop that executes AND
whose work is moving (or which holds a tool batch in flight) is bounded by neither
leg, since both legs use exactly those facts to abstain. That class is not created
here — the progress leg's in-flight clause already spares an in-process tool that
burns a core for an hour — but this change WIDENS it to any turn that keeps its
transcript or roster moving while it walks frames. The module's stated preference is
to spare a working runtime past any bound rather than cut a dead one late, and the
compensating fact is that a runtime satisfying both clauses is one whose work is
advancing by this process's own definition of advancing.

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

WHAT THIS LEG NOW READS ABOUT CHILD LANES, IN TWO HALVES AND TWO CHANGES. Both
halves are closed, and they cover different things a lane can be doing:

* a PROVIDER REQUEST outstanding in any FORKED child stream — every in-process
  lane takes its stream from ``SessionStreamFn.fork``
  (``harness.subagent._construct_child_session``), those forks share ONE counter
  with the runtime's own stream, and the probe reads it as the scalar
  ``child_model_requests_in_flight``: O(1) for the whole tree at once, and the
  manager's own request never counts;
* a STEP OPEN IN THE LANE ITSELF — its own tool batch, its own on-demand
  compaction — which no counter can express, read per live lane by
  ``process._child_lanes_in_flight`` off the roster's private records.

WHAT THE SECOND READ COSTS, MEASURED RATHER THAN ASSUMED — AND IT IS LINEAR IN
ROWS SCANNED, NOT IN LANES ALONE. It is the read the paragraph above refused as
"not cheap" before there was a number for it, and the refusal does not survive the
number; but the number has TWO axes, and quoting only the lane axis is how a
tail-dependent figure gets read as a tail-independent one. Both are below.

THE LANE AXIS. On this host (2026-09-23, CPython 3.12.13, load average 128-280 over
the sweeps — the figures are CPU, so the fleet's load does not enter them), over a
REAL ``SubagentComms`` roster of real child sessions, each holding a completed-turn
tail (four rows plus the ``user`` boundary the scan breaks on):

    live lanes   widened read (µs CPU/sample)     per lane
    0            0.19-0.21                        --
    1            0.85-0.87                        0.85-0.87
    8            6.02-8.88                        0.75-1.11
    64           59.09-62.04                      0.92-0.97
    256          290.49-331.93                    1.13-1.30

(two sweeps of ONE rig, so the ranges are this host's run-to-run spread; the
reviewer's and QA's own rigs put N = 256 at 278-315 µs, so the ORDERING and the
~1 µs per-lane constant are the stable part of this table and a single digit is
not). The sampler reads it once per ``_sample_interval`` — 15 s at the shipped
300 s bound, four samples a minute — so at that tail the widest measured roster
spends ~1.2-1.3 ms of CPU per minute: about 0.002% of one core, three orders of
magnitude below the :data:`PROGRESS_CPU_FLOOR` (5% of one core) the leg demands of
the very spin this read exists to judge.

THE TAIL AXIS IS THE REAL DRIVER, AND THE PER-ROW CONSTANT IS A PROPERTY OF THE
ROW CLASS, so it is only meaningful with the class beside it. What the scan walks
is not "a lane": it is the run of rows after the last ``user`` boundary that are
neither that boundary nor an assistant row carrying tool calls — the rows
``unanswered_tail_call_ids`` steps over one at a time. One lane, one row class per
sweep, same rig, CPU via ``time.thread_time()``, the minimum of three sweeps:

    row class                                   N=514     N=4098   marginal
    CustomMessage, the product's own ctors      849.65    6706.46   1.65 µs/row
    CustomMessage, bare                         937.03    7039.58   1.72
    CustomMessage carrying a role extra         659.00    7391.50   1.80
    Message(role='assistant') with text          38.36     305.96   0.075
    Message(role='assistant') with tool_calls     0.29       1.00   returns there
    SimpleNamespace(role='assistant')            31.81     250.38   0.061

THE CLASSES ARE 20-30x APART, so a µs/row figure means nothing without its class.
The 0.08-0.12 µs/row table this paragraph used to carry was measured on the CHEAP
class — ``Message`` and ``SimpleNamespace`` rows (0.061-0.075 here) — while the
sentence above it named the ``CustomMessage`` row the incident lands on, which is
1.65-1.80. At the class the rule names, 4098 rows cost ~6.7 ms on ONE lane rather
than ~0.46 ms. What is walked in the incident shape IS the ``CustomMessage`` class:
that is what a hub message, a job result, a compaction marker and an incident all
are.

WHAT A REAL TAIL IS, COUNTED RATHER THAN ASSUMED: over the 10,482 saved
transcripts under ``~/.local-operator/sessions``, the walk — rows from the end to
the first assistant row carrying calls, or to the last ``user`` row — is a MEDIAN
OF 5 ROWS, p95 7, p99 9, p99.9 16, and the widest is 25. Of the 43,194 rows those
walks step over, 21,809 (50.5%) are ``CustomMessage``-class entries and 21,341
(49.4%) are ``Message`` rows (44 are journal rows that never enter a live context).
A real tail is short, and about half of it is the cheap class. A WHOLE-TRANSCRIPT
COUNT IS NOT THIS NUMBER, and the gap is the reason the paragraph this one replaces
was wrong to divide by the lane count alone: the same corpus's transcripts hold a
MEDIAN OF 63 ``custom`` entries, p99 612, up to 3057 — they accumulate over the
whole conversation, while the walk stops at the last ``user`` boundary.

AT A REAL WIDTH THE READ IS CHEAP AND THE FLOOR IS UNDER NO PRESSURE. The widest
walk those transcripts show, on a 256-lane roster, is 11.7-13.5 ms of CPU per
sample: 47-54 ms of a CPU minute, ~0.08-0.09% of one core, about 55x below the
:data:`PROGRESS_CPU_FLOOR`. At the median 5-row walk it is 2.3-2.7 ms per sample,
~0.015-0.018% of a core, ~300x below it. That is the honest form of the "three
orders" claim: it holds for the tail rows a real lane carries, and it is about ROWS
more than lanes — the lane axis alone, over a real completed-turn tail (four rows
plus the boundary the scan breaks on), is 191-239 µs at 256 lanes, 0.75-0.93 µs per
lane, this round's own re-measure at the low end of the table above and inside the
rig spread that paragraph names.

THE SHAPE THAT BREACHES THE FLOOR, NAMED, BECAUSE THE FLOOR KEEPS ITS ROLE: 256
lanes whose tails are 4098 ``CustomMessage`` rows each — the shape this paragraph
used to extrapolate — costs 1.88-2.03 s of CPU per sample, 7.5-8.1 s of a CPU
minute, ~12.5-13.5% of one core. That is ABOVE the 5% floor, not below it. Even on
the real 50/50 row mix counted above the same shape is ~1.1 s per sample, ~7.5% of
a core, still above. The floor itself does not move: it is the demand the legs make
and the number this read is judged against, and what this section now says is that
at that tail the read does not meet it. Two consequences, stated rather than left
to inference:

* 4098 rows is SYNTHETIC, not a rate seen in the field — it is 164x the widest walk
  in those 10,475 transcripts, and it is structurally hard to reach, because the
  walk ends at the last ``user`` boundary: a tail that long needs thousands of
  incident / hub / job-result rows on top of one open batch. The steady state is
  the 0.08% case above.
* Should tails that long ever become real, the fix is a BOUNDED read rather than a
  smaller constant, and the bound belongs in ``protocol.unanswered_tail_call_ids``
  — the one rule this read shares with the two display predicates. That is a
  different change from this one, with a wider blast radius, and it is not taken
  here. A CHEAP BOUND WAS CONSIDERED AND REFUSED, measured rather than argued: a
  fast path that skips a row class it "knows" cannot qualify is NOT
  semantics-preserving, because ``CustomMessage`` is ``extra="allow"`` and the scan
  reads its ``role`` and ``tool_calls`` — a ``CustomMessage`` carrying
  ``role='assistant'`` and ``tool_calls`` IS the qualifying row (its calls come
  back as unanswered), one carrying ``role='tool'`` plus ``tool_call_id`` ANSWERS
  the batch below it (the answer flips from ``{'open-1'}`` to empty), and one
  carrying ``role='user'`` STOPS the walk. Short-circuiting on the first qualifying
  row is already what the loop does. A bound that can miss an unanswered call is
  not one to take.

Two structural facts keep the walk as small as it is, and both are pinned in
``tests/unit/session/runtime/test_child_lane_progress.py``: each live lane is asked
once per sample, and a lane that DOES hold a step answers on the first row the scan
reaches, so the in-flight case is the cheaper one (measured on that same 256-lane
roster: 1.5 µs per sample with an open batch on the FIRST record, against 211 µs
with one on the LAST, where every lane is asked).

N IS THE LIVE LANE COUNT, NOT ``MAX_RECORDS``, and the cap is not a budget for this
read. ``_evict_overflow`` evicts only records with no live child AND no running
job, so it bounds the map's evictable tail rather than lane concurrency: measured
here, 300 attached lanes leave 300 records, and a record detached without settling
is not evictable either. A live roster has been measured at 296 records on this
fleet. What bounds this read is how many lanes one manager is running, which is a
scheduling fact and not a constant in this module.

THE SHAPE BOTH HALVES ABOVE EXIST FOR, AND WHY ITS CPU IS THERE. A parent
driving subagents IN PROCESS, every child parked in a long model call, so no lane
closes a step and ``_subagent_roster_generation`` does not move (the roster bumps
on a completed assistant message, a model change or a lifecycle event — not on a
call being outstanding), while the PARENT's own loop burns CPU in the same
per-event projection walk the fires above were caught in. All three legs then
hold — no motion, nothing in flight by the parent's own reading, CPU advancing —
and the diagnostic timer fires while the process is working; that is the false
positive the two reads above close, one half at a time. The step half was
measured on this change's OWN child: with a lane mid-batch and nothing else in
flight, the production probe answered ``not in flight`` and the bound's fire
ended the process — the policy that build carried then, since production expiry
is dump-only now (:data:`DUMP_POLICY_MARKER`), so the same reading fires here
and this runtime SURVIVES it.
The measured cost of that walk is superlinear in the roster: ``nodes()`` calls
``node()`` per record and
``_describe()`` calls ``_live_twin()``, which re-scans ``_records`` for each one, so
the walk is O(N^2) in the lane count, and the widest roster measured on this
machine on 2026-09-22 holds 256 records (one session's
``subagent-roster.v1.json`` holds exactly 256, four more hold 146-243) — a
measured width, not the ``MAX_RECORDS`` cap, which does not bound a live roster
(see the note above). The coordinator on this change
measured one ``_refresh_state``'s subagent work at 46 ms at 64 records, 113 ms at
128, 165 ms at 192 and 422 ms at 256 on an idle box against the real
``SubagentComms`` (this change's own synthetic re-measurement, on bare records with
no ``child`` or job state, gives 0.5 ms at 64 and 2.0 ms at 256 — the shape is the
same and the constant is the real records', so quote the ordering, not the
constant).

THE RESIDUAL, STATED PLAINLY BECAUSE THE WIDENING IS A REAL TRADE, on both sides
of it. Still uncovered: a lane parked in synchronous CPU work with NO step open —
inside one C call or one long callback, no tool batch, no compaction, no provider
request outstanding — which the liveness leg's frame observation cannot see
either, because that watches only the two threads that beat. Such a lane still
holds all three legs, so the bound FIRES on a process that is working — the fatal
arms this module used to carry would have cut it — and closing the gap means
reading a lane's frames, which is a different change from this one.

AND THE MIRROR RESIDUAL, WHICH THIS WIDENING EXPOSES RATHER THAN CLOSES, named
because it is the shape closest to the incident this leg was written for. A step
that never CLOSES — a lane wedged inside an in-process tool whose results never
land, a compaction that hangs — holds its lane's answer "in flight" for as long as
it lives, which is the very property ``lane._is_streaming`` is refused for below,
reached through the narrow question instead. For that shape BOTH LEGS NOW ABSTAIN,
and the consequence is worth spelling out: the liveness leg watches the two planes'
stamps and frames and never reads CPU, so a parent that keeps looping while one
lane holds an open step is ended by nothing in this module. The run stays resident,
with no ``no progress:`` line to find. The founding incident (four running lanes
that had produced nothing for seven minutes) is this shape's neighbour: had those
lanes been parked MID-BATCH rather than idle, this leg would now spare the run. A
reader who finds a runtime alive past its bound with a lane holding a step should
read it as "possibly stalled in a lane" and stop it by hand — ``lop stop``, the
route the held arm already documents — rather than assume the bound is watching;
closing it means bounding a LANE's own step, which is a different change from this
one. The direction is still the right one (spare a process that is working past
any bound rather than cut a dead one late), and it is why the lane question is
asked narrowly, about a step being OPEN, and never as ``lane._is_streaming``: that
is true for a lane's whole turn, so one lane parked in a provider call that never
returns would keep its parent's spin unobservable forever, with not even a step to
point at.

AND IN THE SPARED SHAPE THE BOUND DOES NOT FIRE AT ALL — IT DEFERS. That is a
THIRD outcome beside the two the exit leg distinguishes, and the vocabulary is
worth keeping straight because a dump carries evidence of only two of them.
**Deferred**: the probe answers "in flight" at every sample, so :func:`_sample`
restarts the progress clock each time and no window ever closes — the file carries
the ARM-time header and neither the ``Timeout (`` fired marker nor a
``no progress:`` line, however many bounds the process outlives. That is what a
lane holding an open step produces, and it is an ABSTENTION rather than a fire
that was held. **Fired and held**: reached only once the probe STOPS answering in
flight while the exit leg still holds work (a job in flight, a live subagent); the
fire writes the fired marker, the progress line and :data:`HELD_MARKER`, re-arms
for the next episode, and leaves the process running. **Fired and observed**: a
fire on an idle runtime — the arm a LEGACY build took a native ``_exit(1)`` on,
and :data:`OBSERVED_MARKER` here — where production expiry writes the dump and
leaves the process serving. A reader looking at a header-only dump on a process
that outlived several bounds is looking at the first of the three, not at a fire
that was held.

AND NO PART OF THE FIX ABOVE DEPENDS ON THE MECHANISM BEHIND THOSE FIRES BEING
SETTLED: the abstention rests on an OBSERVED moving frame, and now on an OBSERVED
open step, not on a theory of why the stamp was late.

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
artifact of that death which outlives the process. A dump carrying a tick-death
line therefore says "the workload tick stopped" where the same dump used to
say nothing at all, and :func:`tick_deaths` is the reader for it — as is the
journal's incident narration, which prefers that fact to the bare silence leg
when a dump carries both.

A DEAD TICK STILL LETS THE BOUND FIRE, deliberately. Nothing here unbounds a
plane whose ticker is gone: a plane nothing can stamp is a plane whose silence
this module can no longer interpret, so the honest fail-safe is to leave on the
deadline WITH the reason in the file rather than to run on with one leg
silently switched off.

WHY NOTHING IS ENDED FROM HERE, AND WHAT A STOP STILL COSTS
----------------------------------------------------------
Past the bound the runtime must stop being a multi-hour freeze, and the graceful
rungs cannot be reached from the state being detected: ``_drain_for_signal``,
``_commit_to_leaving`` and ``_leave_overdue`` are all coroutines on the loop that
is blocked, and the ``SIGTERM`` handler that reaches them is a loop handler that
needs bytecodes the stuck thread never executes. That is the same fact that
makes ``lop stop`` refuse a silent-socket runtime without ``--force``. The timer
therefore dumps every thread from its own C thread and ends NOTHING: a Python idle
observation is not atomic with work admission, and native expiry cannot re-check it.
The operator must inspect the dump and explicitly stop a runtime that remains wedged;
update retirement continues to use its separate, work-aware gate.

WHAT THE STOP THAT IS ACTUALLY TAKEN COSTS, and why the residue stays bounded. These
are the costs of the recovery an operator reaches for once the dump names a loop that
is not coming back — the hand-run ``lop stop --pid <pid> --force``, i.e. a SIGKILL — and
they are the same costs a fatal fire used to impose, which is why the arguments below
are unchanged by the policy:

* the in-flight turn's uncommitted step — the same loss a SIGKILL inflicts, because a
turn commits its transcript at each step and the step in flight has not committed.
What SURVIVES is everything already committed, i.e. the conversation, which a
successor can be engaged on. What IS LEFT BEHIND is the record, whose pid is then
``stale``: ``registry.classify`` already reads it that way and ``reclaim`` already
sweeps it, so no new vocabulary is needed — and the automatic paths refuse a
loop-blocked runtime (``record-present``), which is why the stop has to be hand-run;
* the in-process kill of this turn's tool process groups, which a SIGKILL cannot run:
it runs no Python, so ``execute_bash``'s ``_kill`` chain cannot fire — which is exactly
the "hard death of the owning ``lop`` process" class
``local_operator/tools/group_reaper.py`` exists for. Each group is registered
with a liveness marker at spawn and :func:`sweep_orphan_groups` reaps the ones
whose owner is provably dead, at the NEXT ``lop`` startup. So a child orphaned
here is bounded by the next session start rather than by this process's death,
which is the same guarantee every stop already relies on — and a stray group is
visible in the meantime exactly as it would be after any hard death.

THE FILE IS THE EVIDENCE, AND ITS ABSENCE MEANS NOTHING FIRED
------------------------------------------------------------
The dump is written to ``<log dir>/runtime-stall-<pid>.log`` — beside
``runtime.log``, in the directory ``paths.log_dir`` already resolves and
``lop mobile logs`` already points a reader at. Its EXISTENCE carries one signal:
a clean exit (``disarm``) cancels the timer and REMOVES the file, so a header-only
file that survives means the process died without disarming (a SIGKILL or an OOM
kill), and a file carrying :data:`FIRED_MARKER` means this bound actually fired.
Production watchdog expiry is dump-only, so the marker proves timer expiry, not
process death. That is the shape
``tests/e2e/watchdog.py`` uses, and for the same reason: ``faulthandler`` writes
with a raw descriptor from a C thread, so the header cannot be deferred to the
moment it fires.

WHAT A READER GETS, IN THE ORDER THE FILE WRITES IT, because each line answers a
different question and none of them is a verdict: the arm line and the
:data:`DUMP_POLICY_MARKER` (which BUILD this is — what a fire could have meant at
all, see :func:`fire_outcome`), any quiet-plane or tick-death line (why a plane
stopped reporting), the :data:`TEARDOWN_MARKER` annotation (whether the process had
reached the end of ``amain`` — see :func:`fired_in_runner_teardown`), then
``Timeout (`` with every thread's stack if the timer fired, and after it, when the
process survived, exactly one line naming the reading that fire earned:
:data:`HELD_MARKER` for work in flight, :data:`OBSERVED_MARKER` for an idle or
cleared one. The last pair is what a live listing renders as ``stall_held``, which
is why only the held one may mean "a person has to unblock this".

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
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import IO, Callable

logger = logging.getLogger(__name__)

#: The bound, in seconds, as an operator or a test overrides it. Unset means
#: :data:`DEFAULT_STALL_S`; ``0``/``off`` disables the watchdog outright.
ENV_SECONDS = "LOP_RUNTIME_STALL_SECONDS"

#: The BOOT bound's own knob, parallel to :data:`ENV_SECONDS` and resolved by the
#: same rules (same spellings, same floor, same ceiling). Unset means
#: :data:`DEFAULT_BOOT_STALL_S`; ``0``/``off`` means "no boot phase" — arm
#: straight at the steady bound — rather than "do not arm". The two knobs read
#: their spellings identically and mean different things by them, which is worth
#: the sentence: only the STEADY knob can disable the watchdog, because it is the
#: one that governs the life a runtime spends after it has engaged.
#:
#: Deliberately NOT a registry key (``settings_io``): the bound has always been
#: env-only, and registering one drags in a ``Setting``, a ``Scope`` and the
#: every-default-matches-its-consumer gate for a knob no operator asked to see.
ENV_BOOT_SECONDS = "LOP_RUNTIME_BOOT_STALL_SECONDS"

#: How many doublings a survived fire's interval may take, and the ceiling it stops
#: at (agent review round 1, MINOR-5). Two figures rather than one so the growth is
#: written down where a reader asks "how fast does this grow": 4 steps of a 300 s
#: bound reaches 4800 s, and the cap holds it there, so a runtime held for a day
#: writes ~24 dumps instead of ~288.
HELD_FIRE_BACKOFF_STEPS = 4
HELD_FIRE_BACKOFF_MAX_S = 3600.0

#: Seconds of NO PROGRESS before the runtime dumps every thread. NOTHING LEAVES:
#: the bound has no authority over the process (see the module docstring), so what
#: this number sizes is when the EVIDENCE is written.
#:
#: Sized against the two numbers the record already publishes, and against the
#: one measured false-positive class:
#:
#: * ``HEARTBEAT_INTERVAL_S`` (15 s) is how often a healthy runtime reports;
#:   ``HEARTBEAT_TIMEOUT_S`` (45 s) is when ANOTHER process starts calling it
#:   ``wedged``. Both are too tight to exit on — a ``wedged`` reading is
#:   degraded-responsiveness EVIDENCE, and this host has produced 105.8 s and
#:   205.8 s beat gaps on sessions whose CPU time was advancing (see
#:   ``types.HEARTBEAT_INTERVAL_S`` and ``registry.classify``). Firing on a
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

#: Seconds of NO PROGRESS allowed BEFORE THE RUNTIME HAS ENGAGED, and the bound the
#: single :func:`arm` call arms with.
#:
#: A SEPARATE NUMBER FROM :data:`DEFAULT_STALL_S` BECAUSE THE WINDOW IT COVERS HAS
#: NO CLOCK OF ITS OWN. ``arm`` runs from the ``__main__`` guard, before ``main()``,
#: and it seeds both planes to that instant — but NOTHING IN THE BOOT PATH CAN MOVE
#: A STAMP: the WORKLOAD beat's only driver starts after publication and sleeps a
#: heartbeat before its first stamp, and the SERVING beat starts with the serving
#: thread. So from the arm to the first post-publication beat, every second a
#: legitimate boot spends — session construction, lease arbitration, MCP bring-up,
#: the inbox drain, the socket bind, publication — is spent against a clock no boot
#: code can re-arm, and the earliest deadline is ``arm + bound``.
#:
#: MEASURED over 68 retained dumps on the build that carries the observation header:
#: 17 fires, and 10 of them carried the full ``Timeout (0:05:00)`` value — i.e. NO
#: beat ever re-armed the timer, the never-engaged class and the plurality of that
#: build's fires. (THOSE 10 ARE AN UPPER BOUND ON THAT CLASS, not an exact count: a
#: failed re-arm is indistinguishable from a never-engaged runtime by the armed value
#: alone -- see :func:`engage` -- which is why the note below and
#: :data:`HOW_TO_READ_THE_FIRED_VALUE` both qualify the reading.)
#: (The other 7 show the planes' boot latencies differ: the
#: re-arm line names WORKLOAD "has reported nothing since the ARM" while SERVING
#: "reported 0 s ago" — the workload plane is the long pole, which is why
#: :func:`engage` stamps BOTH.)
#:
#: 900 s is loose ON PURPOSE, and it is not the same kind of number as the steady
#: bound: that one is a ceiling on one silent synchronous step in a runtime that is
#: running, and this one is room for a boot whose latency is unbounded and
#: legitimate (there is no boot-side false-positive class to size against, only the
#: unbounded-tail risk of cutting a slow-but-working boot). The steady bound takes
#: over the moment the runtime engages, so the whole of this number is spent on boot.
#:
#: IT BUYS EVIDENCE, NOT A CUT, and the distinction is the one thing the entry point's
#: side of the split is easiest to misread (agent review round 1, R1-1). The busy
#: probe answers through ``process._busy_probe``, which reports WORK IN FLIGHT for the
#: whole pre-publication window — ``True`` while ``_live_handle`` is None, because a
#: runtime still constructing itself is not idle in any sense this bound may act on —
#: so ``arm`` seeds ``_Armed.held`` True here and every fire in this stretch is recorded
#: as HELD (:data:`HELD_MARKER`). A boot that hangs is therefore DUMPED at this bound
#: and HELD, and one that never reaches publication keeps answering "in flight" for the
#: rest of its life, so nothing in this module ends it — no timer armed here has ever
#: had authority over the process, whatever the sample says. THE WAY OUT IS MEASURED RATHER THAN
#: ASSUMED, because the first version of this paragraph named one that does not
#: reach this class (agent review round 2, Q-3):
#:
#: * the boot's OWN FAILURE PATH is the exit that is automatic — the construction
#:   error that ends the runtime child (measured: rc 2 and the cause on stderr, 1.4 s);
#: * ``lop stop`` is NOT one. It resolves its target through session RECORDS
#:   (``mobile.peer_send.resolve_peer_target``) and a boot that never published has
#:   none, so ``lop stop --pid <pid>`` answers ``no session found with pid <pid>``
#:   and the process carries on (measured, as is the ``--session`` form);
#: * an operator ends it by SIGNALLING the pid this dump is named for — the header
#:   above carries it, and ``kill -TERM <pid>`` ends the process (measured: the
#:   signal is not gated on any record).
#:
#: What this bound contributes is the ATTRIBUTION: the fired value is this number
#: and the deadline sibling is absent — together the never-engaged class, WHEN THE RE-ARM
#: SUCCEEDED. An engaged runtime whose timer replacement FAILED has no sibling either
#: (nothing re-armed it, so no beat ever wrote one) and can fire at this value, which is
#: why :func:`engage` records that failure in the dump rather than letting the value and
#: the missing sibling be read as never-engaged. A fire under THIS bound ends nothing
#: either: the boot phase has its own bound, not its own fate, and the entry point arms
#: with ``exit=False`` exactly as every re-arm does. The armed value is the whole of what
#: this number decides, and it decides it in the dump, never over the process.
DEFAULT_BOOT_STALL_S = 900.0

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

#: The dump-only statement every header of this era carries, and one of the phrases
#: :func:`fire_outcome` reads as PROOF that the build which wrote the file could not
#: end a runtime from its timer. Named rather than inlined because a reader keys on
#: it (see :data:`DUMP_ONLY_POLICY_PHRASES`) and because a header and its reader
#: drifting apart is the class of defect this whole convergence round is about.
DUMP_ONLY_STATEMENT = "PRODUCTION EXPIRY IS DUMP-ONLY"

#: What a dump's header states when the build that wrote it is this one.
#:
#: THE ARTIFACT, RATHER THAN AN INFERENCE FROM THE ARTIFACT'S SILENCE. Finding A of
#: the 2026-09-23 convergence round: attribution used to read "fired, and no post-fire
#: held marker => the bound ended the runtime", which is false on a dump-only build
#: (a GIL-held fire cannot append a Python marker at all, and any later stop or crash
#: is a separate cause). The fix is a POSITIVE statement, so a reader no longer has to
#: reason from what is absent -- the same inversion :data:`OBSERVATION_NOT_VERDICT`
#: makes in prose, made here in a token a parser can key on.
#:
#: THIS IS THE ONE MARKER THE HEADER SPELLS, and that is safe for exactly the reason
#: the note at the foot of :func:`arm` gives for the others: no reader infers "a fire
#: happened", "a plane is quiet" or "the bound held" from it, and the fire markers
#: (``Timeout (``, :data:`HELD_MARKER`, :data:`OBSERVED_MARKER`, :data:`PROGRESS_MARKER`)
#: are what those questions are asked of.
DUMP_POLICY_MARKER = "[stall watchdog] policy: dump-only (exit=False)"

#: Every phrase a header can carry that proves the build which wrote it was dump-only.
#: Two, because the statement and this marker are both spellings of the SAME policy: a
#: header that says the timer is dump-only proves it however it is worded, and a reader
#: keying on one spelling alone would narrate a dump-only fire as unknown for no reason.
#:
#: NO RELEASED BUILD WRITES THE STATEMENT ALONE (agent review round 3, NIT-1). The
#: era sentence shipped only in the branch commit that introduced it (`2555c7a21`, never
#: on ``main``), so this alias is DEFENSIVE rather than a reader for artifacts in the
#: wild: it costs a substring test and covers a header that states the policy without
#: the marker, which is the shape a build BETWEEN the two spellings would have written.
DUMP_ONLY_POLICY_PHRASES = (DUMP_POLICY_MARKER, DUMP_ONLY_STATEMENT)

#: The sentences a LEGACY (pre-dump-only) header carried that state the OPPOSITE
#: convention: an IDLE fire takes a native exit.
#:
#: THE ONLY THING THAT MAY MAKE A FIRE A WATCHDOG DEATH (see
#: :func:`local_operator.session.runtime.journal.death_verdict`). Both spellings
#: shipped on ``main``: ``this timer is armed with exit=True`` is the all-fires-fatal
#: build, and the second is the one that followed the work-aware change and still
#: ended an idle runtime. They are quoted rather than paraphrased because they are
#: what is ON DISK in the retained artifacts, and a reader matching a paraphrase
#: would classify nothing.
LEGACY_FATAL_POLICY_PHRASES = (
    "this timer is armed with exit=True",
    "an IDLE one is dumped and exited",
)

#: What :func:`fire_outcome` answers with. THREE STATES, because a fired dump has
#: three readings and two of them want opposite responses (see :func:`held_fire`).
#:
#: ``FIRE_SURVIVED`` -- the artifact proves the fire did not end this runtime (a
#: dump-only build, or a legacy build whose fire was held). ``FIRE_LEGACY_FATAL`` --
#: the artifact states the legacy convention, so an unheld fire ended it.
#: ``FIRE_UNKNOWN`` -- no fire, or an artifact whose own words prove neither; the
#: last one is the state that must never be rounded to "the bound killed it".
FIRE_SURVIVED = "survived"
FIRE_LEGACY_FATAL = "legacy-fatal"
FIRE_UNKNOWN = "unknown"

OBSERVATION_NOT_VERDICT = (
    "THIS IS A TIMER OBSERVATION, NOT A VERDICT.\n\n"
    "THIS PROCESS MAY STILL BE ALIVE ON THIS SAME PID. Check the PID; it may still serve.\n\n"
    f"{DUMP_ONLY_STATEMENT}: the native timer is armed with exit=False.\n\n"
    "The timer expired without a re-arm from this process's own loops, and that is ALL it "
    "measured. It is NOT a statement that the runtime stopped -- the timer can expire on a "
    "process that is mid-turn and working, and it may carry on serving AFTER this dump is "
    "written.\n\n"
    "This header is written at ARM time, BEFORE any of those re-arms, so it cannot name the "
    "answer that will hold at the fire -- and so the file cannot say whether the runtime is "
    "gone either. A fire is not by itself a body count: THE PID, NOT THIS FILE, IS WHAT SAYS "
    "WHETHER THE RUNTIME IS STILL THERE."
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
#: plane's stamp, not the bound, was what the timer measured. THE FIRST READING IS
#: CONDITIONAL ON THE RE-ARM, for the same reason the boot note is (design review round
#: 1, D1): :func:`engage` moves the bound in memory BEFORE it replaces the timer, so an
#: engaged runtime whose replacement FAILED keeps the original C timer and can fire at
#: the armed value -- a member of this 6, not of the never-engaged class, and the class
#: the paragraph below states as an exception rather than hiding. Nothing in the file
#: said which of the two a number was, so "which plane went quiet, and when" stayed
#: unreadable on a runtime whose own record is the only witness left.
HOW_TO_READ_THE_FIRED_VALUE = (
    "HOW TO READ WHAT FIRES: the value on the fired line above the stacks is the seconds the "
    "timer was LAST ARMED FOR, and which arming that was is the whole question. It is the bound "
    "above, with no beat re-arming it at all, when no loop ever reported -- a runtime that never "
    "engaged, its main thread idle from boot, PROVIDED THE RE-ARM SUCCEEDED. An ENGAGE whose "
    "timer replacement FAILED is the exception: the bound moves in memory while the original "
    "timer stays in force, so a fire can carry this value on a runtime that DID engage -- the "
    "dump then carries a 're-arm failed' line, and that line's absence is what makes the "
    "never-engaged reading safe. It is a SMALLER value when a beat recomputed it "
    "from the oldest plane's stamp, which is how a plane that went quiet shows up as a number "
    f"below the bound. {DUMP_PREFIX}-<pid>{DEADLINE_SUFFIX} holds the deadline the timer is "
    "currently armed for (as an epoch), the leg that pinned it and the same deadline on the "
    "process's monotonic clock, rewritten by every beat THAT RE-ARMED, and "
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

#: Written when a fire that DID NOT end this process has been observed AND work was in
#: flight — the one marker here that is written AFTER the ``Timeout (`` line rather than
#: above it.
#:
#: THE LIVE STATE, and the reason it exists. A fire over work that is in flight (a turn
#: under the lock, a live subagent lane, a job) is the case this marker names, and its
#: two readers — :func:`held_fire` and :func:`held_pids` — are what a listing renders as
#: a runtime that is STILL STALLED with that work inside it: that wants the person who
#: owns the work or a ``lop stop``.
#:
#: WORK IN FLIGHT, NOT MERELY A SURVIVED FIRE (finding B of the 2026-09-23 convergence
#: round). Until the split below, this marker was appended for EVERY fire the sampler
#: observed, so an idle runtime that survived a dump-only fire was reported through those
#: readers as ``stall_held`` — a permanent false "stalled with work in flight" cell on a
#: process with nothing in flight, which is a state the artifact cannot establish. The
#: neutral case has its own marker (:data:`OBSERVED_MARKER`) and does not set it.
#: FOR A HISTORICAL ARTIFACT NOTHING CHANGES, and that is why the split is safe to make
#: now: on the legacy (fatal) builds an idle fire could not append anything at all — it
#: had already exited the process — so this marker's historical meaning was already
#: "work was in flight", and the absence of a post-fire marker carried the fatal reading
#: the dump-only builds falsified.
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
#: WHY THIS LINE MAY SAY "STALLED" WHEN THE ONLINE VOCABULARY MAY NOT (agent review
#: round 1, NIT-1): the word is barred from the drain's prose because a runtime cannot
#: establish that a slow turn is stuck, and a sentence that says so would assert a half
#: it cannot know. The bound's own premise is narrower and provable — 300 s in which
#: NEITHER plane reported and, if the progress leg fired, no CPU moved either — so for
#: the bound "stalled" names the measurement it just made rather than a guess about
#: someone else's work. The dump is also the artifact an operator reads while looking
#: for exactly this, which is why the word is worth keeping here and nowhere else.

HELD_MARKER = "[stall watchdog] bound held: "

#: Written when a fire that DID NOT end this process has been observed and work was NOT
#: established — the sample said idle, or the work had cleared before the fire was seen.
#: The neutral half of the split above.
#:
#: WHAT IT IS FOR: it keeps the fire on the record — the timer expired, here is the
#: moment and every thread's stack — without setting the live ``stall_held`` state, and it
#: is the phrase :func:`fire_outcome` reads as :data:`FIRE_SURVIVED` when the header says
#: the build was dump-only. An idle fire is worth investigating (a plane stopped
#: reporting, or the process spun without advancing) and it is not a stall that has to be
#: unblocked, and those two are different rows on the surface a person scans.
#:
#: NOT A PREFIX OF ANY OTHER MARKER and contained in none, for the reason every marker
#: here states: readers test these as substrings.
OBSERVED_MARKER = "[stall watchdog] bound fired: "

#: Written, once per arming, when the asyncio runner that owns ``amain`` starts its OWN
#: teardown: ``[stall watchdog] asyncio runner teardown: asyncio runner teardown started;
#: executor join may follow``.
#:
#: WHY THE WORDING IS "teardown started; executor join MAY follow" rather than "executor
#: joining": task cancellation and async-generator shutdown precede the executor join, so
#: a reader that took this line for "the loop is in the join" would name the wrong phase.
#: The fires this exists for DO sit in the join, but the line is written before the phase
#: is entered — the only moment at which a Python thread is still alive to write it.
#:
#: WHAT IT IS FOR, and the class it closes: the 17-27 dumps in the operator's store whose
#: loop sits in ``asyncio`` shutdown and whose death nothing attributed (2026-09-23). The
#: annotation is a POSITION in the artifact, and the classification is read off the
#: ordering alone (:func:`fired_in_runner_teardown`): a fire below this line was taken
#: while the runner tore down, a fire above it was not, and an ordering that cannot be
#: established is reported as unknown rather than inferred.
#:
#: SET ONCE AND NEVER CLEARED: the annotation describes a phase this process reached, and
#: a phase is not un-entered.
TEARDOWN_MARKER = "[stall watchdog] asyncio runner teardown: "

#: The sentence :func:`note_runner_teardown` appends after :data:`TEARDOWN_MARKER`.
TEARDOWN_NOTE = "asyncio runner teardown started; executor join may follow"

#: Written when the liveness leg OBSERVES a plane's loop executing Python and therefore
#: abstains from firing on that plane's silence: ``[stall watchdog] loop executing: <plane>
#: in <file>:<line>:<function>``.
#:
#: WHY IT IS IN THE DUMP RATHER THAN ONLY IN THIS PROCESS. A fire is the artifact an
#: operator reads to find out why a runtime left, and the fairness fix makes the
#: interesting case one where the bound did NOT fire — so the line has to be written
#: when the ABSTENTION happens and not at the fire, or the evidence for the decision
#: would only exist in the runs where nothing needed explaining. It also has to survive
#: the sampler, because the loop it describes is by construction not running: this is
#: written by the SAMPLER thread, under the same lock that serialises the deadline.
#:
#: The frame is included because "which line was the loop walking" is the question this
#: line exists to answer, and a reader cannot ask the process afterwards — it is gone.
#: It is the TOP frame of that thread's stack, so it names the innermost call, not the
#: turn above it.
#:
#: NOT SPELLED IN THE HEADER, for the reason that constraint exists at all: readers
#: test these as substrings, so a header quoting the marker would make every armed file
#: read as a runtime whose loop was executing.
EXECUTING_MARKER = "[stall watchdog] loop executing: "

#: What the first sample of a progress clock holds before it has anything to
#: compare against. A sentinel rather than ``None`` because a probe is free to
#: return ``None`` as its own motion value, and "no previous sample" and "the
#: previous sample was None" must not be the same state.
_NO_SAMPLE = object()

#: A caller that has NOT evaluated the probe passes this, and
#: :func:`_sample` then reads it itself; :data:`_NO_SAMPLE` and ``None`` are the two
#: answers that mean "there is no live progress leg for this sample". The three are
#: distinct on purpose: "nobody asked me to decide" (this), "the probe is absent"
#: (``None`` from :func:`_read_probe`), and "the probe is there and useful".
_UNSET = object()


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
        "boot_seconds",
        "steady_seconds",
        "pid",
        "last_beat",
        "seeded_at",
        "quiet_noted",
        "probe",
        "busy",
        "held",
        "arm_size",
        "seen_fires",
        "held_fires",
        "after_fire_at",
        "fired_held",
        "progress_deadline",
        "clock",
        "stop",
        "thread",
        "loop_thread",
        "last_frame",
        "last_move",
        "executing_at",
        "executing_noted",
        "runner_teardown",
    )

    def __init__(
        self,
        path: Path,
        handle: IO[str],
        seconds: float,
        pid: int,
        probe: "ProgressProbe | None" = None,
        busy: "BusyProbe | None" = None,
        *,
        steady_seconds: float | None = None,
    ) -> None:
        self.path = path
        #: Where the CURRENT deadline is recorded for a reader who arrives after the
        #: process is gone (:func:`deadline_path`), rewritten by every beat.
        self.deadline_path = path.with_suffix(DEADLINE_SUFFIX)
        self.handle = handle
        #: THE LIVE BOUND, and the only one the timer arithmetic reads
        #: (:meth:`deadline`, the quiet threshold, the progress window). It starts at
        #: the bound this process was ARMED for -- the BOOT bound, because nothing in
        #: the boot path can stamp a plane -- and :func:`engage` moves it DOWN to
        #: :attr:`steady_seconds` once and once only.
        self.seconds = seconds
        #: What ``seconds`` was at arm time: the bound the header named and the timer
        #: was armed with, kept because ``seconds`` no longer holds it after an
        #: engage, and both a reader of the dump and the failure line in :func:`engage`
        #: are about THAT number rather than the live one.
        self.boot_seconds = seconds
        #: The bound :func:`engage` moves to, resolved from the steady knob. Defaults
        #: to the armed bound, so an ``_Armed`` built with one number (every cell that
        #: constructs one directly, and every arm that resolved a single value) has no
        #: boot phase to end and :func:`engage` is a documented no-op on it.
        self.steady_seconds = seconds if steady_seconds is None else steady_seconds
        #: How many fires this process has SURVIVED. Only the held-fire backoff reads
        #: it (see ``_rearm``): an all-thread dump is ~14 KB, so a runtime wedged with
        #: work in flight would otherwise write one every bound for the rest of its
        #: life — measured at ~4 MB/day for a 300 s bound, which is the accumulation
        #: this module refuses everywhere else (see ``_note_quiet_plane``).
        self.held_fires = 0
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
        #: Which THREAD owns each plane's loop, learned from the beat itself: a
        #: plane's tick is called FROM the loop it reports for, so
        #: ``threading.current_thread()`` at stamp time IS the thread whose frame
        #: says whether that loop is executing (see
        #: :meth:`_Armed.observe_execution`). Learned rather than assumed —
        #: hardcoding the main thread would be wrong for the serving plane, whose
        #: heartbeat runs its own loop.
        #:
        #: THE OBJECT AND NOT ITS IDENT, and that is a correctness requirement rather
        #: than a convenience. ``get_ident()`` values are RECYCLED by CPython once a
        #: thread ends — measured on this host: six sequential short-lived threads
        #: collapsed to ONE distinct ident — so an ident kept past its thread's death
        #: starts naming a DIFFERENT, LIVE thread, whose moving frames would then be
        #: read as this plane still executing. A plane whose reporter is gone must be
        #: ended one deadline after its last stamp, and the ident shape defeats exactly
        #: that. Holding the object lets :meth:`_Armed.observe_execution` ask
        #: ``is_alive()`` instead, which is the fact the ident cannot carry.
        #:
        #: A PLANE THAT NEVER STAMPED HAS NO ENTRY, and that is load-bearing rather
        #: than an empty case: an unlearned plane means this loop can never be
        #: OBSERVED executing, so it cannot be extended either. The
        #: never-engaged class (a runtime whose main thread never reported at all)
        #: therefore behaves exactly as it did before this
        #: (``test_a_runtime_that_never_engaged_fires_at_the_armed_value_with_no_sibling``).
        self.loop_thread: dict[str, threading.Thread] = {}
        #: Each plane's loop frame as of the previous observation, as
        #: ``(filename, lineno, function)``, or :data:`_NO_SAMPLE` before the
        #: first look. A frame that DIFFERS from this is a loop executing Python.
        self.last_frame: dict[str, object] = {}
        #: When each plane's frame was last SEEN to move — the observation, and
        #: nothing more. It drives the per-STRETCH record in the dump
        #: (:attr:`executing_noted`); it is deliberately NOT what the deadline is
        #: measured from, because an observation is not a decision. See
        #: :attr:`executing_at`.
        self.last_move: dict[str, float] = {}
        #: When each plane's execution was last GRANTED as a sign of life, which is
        #: the whole of the fairness fix (see the module docstring): the plane's
        #: liveness deadline is measured from the later of this and its last beat.
        #: WRITTEN ONLY BY :func:`_extend_for_execution`, ON THE SAMPLES IT GRANTS —
        #: the single writer, so a sample that did not earn the extension cannot move
        #: the deadline a sibling's :func:`beat` re-arms the shared C timer for. That
        #: split is the fix for the refused hand-off still extending: an unevaluable
        #: probe used to move the deadline through the observation alone, which made
        #: an executing loop with a broken probe unbounded on a runtime whose other
        #: plane kept stamping. Absent until an extension has been granted, so nothing
        #: is ever extended on a plane whose loop has not been observed executing.
        self.executing_at: dict[str, float] = {}
        #: The planes whose execution is already written into the dump: a TRANSITION
        #: rather than every observation, for the reason :attr:`quiet_noted` gives — a
        #: loop that executes for hours would otherwise grow the artifact one line per
        #: sample. Emptied for a plane when its loop stamps again, which is what makes a
        #: later stretch its own transition.
        self.executing_noted: set[str] = set()
        #: The progress leg's probe, or ``None`` when no runtime supplied one (an
        #: in-process host, a test of the liveness leg alone, an older spawner).
        #: ``None`` means the progress leg is INERT, not that it is satisfied —
        #: and it is also what REFUSES the liveness leg's abstention, because a
        #: hand-off with nothing to hand to is just an unbounded loop (see
        #: :func:`_extend_for_execution`).
        self.probe = probe
        #: The WORK-IN-FLIGHT probe, and the decision it currently holds. ``held`` is
        #: recomputed at arm time and re-read on the sampler's own wakes (see
        #: :func:`_apply_exit_leg`), and it is a FIELD rather than a fresh call at
        #: each re-arm because ``beat`` runs on two other threads and the answer must
        #: be read from one place: only the sampler and the arming thread call the
        #: probe, so a plane's tick cannot walk the session's state from a thread that
        #: does not own it (``_beat_stall_watchdog`` and ``_heartbeat_loop`` both
        #: reach :func:`beat`).
        #: THE ``..._exit_leg`` NAMES AROUND IT ARE HISTORICAL: this decision used to
        #: authorize a native exit, and no timer in this module has that authority any
        #: more (finding C of the 2026-09-23 convergence round). What it decides today
        #: is which READING a fire carries, and therefore which line records it —
        #: :data:`HELD_MARKER` when it says work is in flight, :data:`OBSERVED_MARKER`
        #: when it does not.
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
        #: Whether the asyncio runner that owns ``amain`` has started its OWN teardown
        #: (:func:`note_runner_teardown`). AN EVENT RATHER THAN A FLAG because the writer
        #: is the main thread's coroutine wrapper and the READER can be the SAMPLER
        #: thread: the set is the happens-before edge between them, and it is also what
        #: makes the annotation idempotent without a second field to keep in step.
        #: It lives on ``_Armed`` rather than as module state so a later arming starts
        #: un-annotated, and it is never cleared: see :data:`TEARDOWN_MARKER`.
        self.runner_teardown = threading.Event()

    def sign_of_life(self, plane: str) -> float:
        """The last moment this plane showed ANY sign of life: a beat, or a frame that MOVED.

        THE FAIRNESS FIX LIVES IN THIS ONE ``max``. The liveness leg's question is
        "is anything running at all", and a beat is not the only way a loop answers
        it — a loop executing Python code is running, and the WORKLOAD tick is an
        ``asyncio`` task on the very loop whose frames are moving, so the state a
        starved tick produces is *indistinguishable from a parked loop* to anything
        that only watches stamps. ``executing_at`` is that missing observation, and
        taking the later of the two is how a moving frame stands in for the stamp
        the loop could not deliver.

        ABSENT MEANS NO EXTENSION, deliberately: ``0.0`` rather than the seed time,
        so a plane whose loop has never been OBSERVED executing is measured from its
        beat exactly as before. The seed is not usable as a default here — it would
        grant every plane a full bound of extension on its first sample, which is
        the unbounded reading this module must not have.

        ``executing_at`` IS WRITTEN ONLY BY THE GRANT, so this ``max`` can only move
        on a sample that EARNED the extension — a probe supplied, answering, and a
        frame that moved, all on that same sample. Reading a timestamp the
        OBSERVATION had written would let a refused hand-off move this deadline
        anyway, through the ``pin``/``deadline`` a sibling plane's :func:`beat`
        re-arms the shared timer from: the process would live on with a deadline that
        never counted down, which is the unbounded reading arriving by a side door.
        """
        return max(self.last_beat[plane], self.executing_at.get(plane, 0.0))

    def observe_execution(self, now: float) -> "tuple[tuple[str, ...], tuple[str, ...]]":
        """Look at each known plane's loop frame; name the planes whose frame MOVED.

        THE SECOND OBSERVATION THE LIVENESS LEG WAS MISSING, and the reason it needs
        the sampler to carry it: the whole starved state is one where the loop cannot
        stamp, so nothing that runs ON that loop can report for it. This runs on the
        sampler's own thread, which is exactly why it can still see the answer.

        THE DISCRIMINATOR IS MOVEMENT, NOT WEDGEDNESS, and that is the only claim made
        here. A loop parked in a call it never returns from holds ONE frame for the
        whole wait — a C-level matcher (``_sre_SRE_Pattern_search``, the 2026-09-20
        incident), ``select``, ``flock``, a child's ``wait`` — so its frame repeats and
        this reports nothing. A loop EXECUTING Python reaches this through its CALLS, so
        the top frame's ``(filename, lineno, function)`` differs from the last look. Both
        halves are load-bearing: the frozen case is the incident this bound exists for and
        must keep firing, and the moving case is the false positive that killed sessions
        which were demonstrably working.

        WHAT IT SEES IS CALLS, NOT INSTRUCTION-LEVEL PROGRESS, and that is a MEASURED
        property of CPython rather than a design choice — see the module docstring's
        paragraph on it. ``f_lineno`` on a FOREIGN frame is only refreshed at a call or a
        suspension, so a body of bare arithmetic reads as one frozen line however fast it
        runs, while a body that calls something (the projection walk is all calls) shows a
        change on about two reads in three. The limit that follows is stated there with
        the numbers, and it is why this is a conservative half of the fix.

        Returns ``(moved, noted)``. ``moved`` is every plane whose frame changed on THIS
        look, which is what the abstention needs. ``noted`` is the subset that STARTS a
        stretch — a plane whose execution was not already recorded — which is what the
        dump line needs, so the artifact grows per stretch rather than per sample. A
        stamp NEWER than the last observation ends a stretch: the loop reported for
        itself again, so what follows is a new one. That is decided from the stamps
        rather than from a still look, because the top frame alternates between a call
        and its caller and a rule keyed on one still look re-records the SAME stretch
        several times a second — measured as 8 identical dump lines across a 6 s
        stretch, which is the artifact growth this avoids.

        IT IS AN OBSERVATION AND NOT A DECISION, and the split is structural rather
        than stylistic: this method writes :attr:`_Armed.last_move` (which drives the
        dump's per-stretch record) and never :attr:`_Armed.executing_at` (which is
        what the deadline is measured from). The only writer of the latter is
        :func:`_extend_for_execution`, on the samples where the hand-off is GRANTED.
        See :attr:`_Armed.executing_at` for the defect that split exists to close.

        Does the state update and the dump transition under the caller's ``_LOCK``:
        :meth:`sign_of_life` reads ``executing_at`` while deciding the deadline a beat
        re-arms, so a write that was not serialised against that read could arm the
        timer one way and describe it another.
        """
        if not self.loop_thread:
            return (), ()
        # One snapshot for every plane rather than one lookup each: this is a
        # foreign-thread read of interpreter state, so it is taken once per sample.
        frames = sys._current_frames()
        moved: list[str] = []
        noted: list[str] = []
        for plane, thread in self.loop_thread.items():
            # A THREAD THAT HAS ENDED IS NOT A SIGN OF LIFE, and the check has to be on
            # the OBJECT: an ident outlives the thread that owned it (CPython recycles
            # them), so ``frames.get(ident)`` would hand back some OTHER live thread's
            # frames and this plane would read as executing forever. A plane whose
            # reporter is gone is exactly the case the tick supervision promises the
            # bound will end, one deadline after its last stamp.
            #
            # THE RESIDUAL IS ONE SAMPLE WIDE, stated rather than hidden: a thread can
            # die between this check and the snapshot below, and if its ident is
            # recycled inside that window one look reads the wrong thread. The next
            # sample re-checks and skips, so the extension that can be granted this way
            # is bounded by one sample interval and then counts down — it cannot become
            # the open-ended abstention the ident-keyed lookup produced.
            if thread is None or not thread.is_alive():
                continue
            ident = thread.ident
            if ident is None:
                # A LIVE thread always has an ident, so this cannot be reached for a
                # running loop; it is here because ``Thread.ident`` is typed optional,
                # and a ``None`` key would look up nothing (or, worse, read as an
                # unmatched plane). Skipping is the same answer either way.
                continue
            frame = frames.get(ident)
            seen: object = _NO_SAMPLE
            if frame is not None:
                seen = (frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
            previous = self.last_frame.get(plane, _NO_SAMPLE)
            self.last_frame[plane] = seen
            if self.last_beat[plane] >= self.last_move.get(plane, 0.0):
                # The loop stamped for itself since the last look, so the stretch this
                # note describes is over and a later one is its own transition.
                self.executing_noted.discard(plane)
            # No frame at all is NOT movement: the sample caught an interpreter state
            # that cannot be read. Extending on that would be inventing a sign of life,
            # so the plane falls back to its stamp and the bound keeps counting.
            if seen is _NO_SAMPLE or previous is _NO_SAMPLE or seen == previous:
                continue
            self.last_move[plane] = now
            if plane not in self.executing_noted:
                self.executing_noted.add(plane)
                noted.append(plane)
            moved.append(plane)
        return tuple(moved), tuple(noted)

    def pin(self) -> tuple[str, float]:
        """Which leg's deadline the timer is armed for, and when: ``(leg, monotonic)``.

        The same ``min`` :meth:`deadline` takes, kept in ONE place because the record
        this bound writes for a reader (:func:`_record_deadline`) has to name the same
        pin the timer was armed from — two spellings of "the earliest deadline" would
        be two chances for the sibling and the file to disagree about which plane was
        quiet, which is the fact the two files exist to establish.
        """
        plane = min(self.last_beat, key=self.sign_of_life)
        deadline = self.sign_of_life(plane) + self.seconds
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


def _bound_from_raw(env_name: str, raw: str | None, default: float) -> float | None:
    """One bound from one RAW environment value, with the shared floor and ceiling.

    BOTH KNOBS PARSE HERE rather than in two near-copies, because the rules are the
    same rules: the same spellings mean "off", the same typos fall back to the knob's
    own default, and both clamp to :func:`min_bound_seconds` and the C timer's range.
    Two copies would be two chances for the boot knob to accept something the steady
    one refuses, and a bound that is only wrong during boot is the one nobody measures.

    The ``os.environ.get`` itself stays at each caller, one line per knob, and that is
    load-bearing: the ambient-environment audit
    (``tests/unit/test_ambient_env_isolation.py``) resolves a read's key through
    module-local constants, so a read through a PARAMETER would leave both knobs
    invisible to it and quietly unaccounted for.

    Unreadable, non-numeric and out-of-range values fall back to ``default`` rather
    than raising: this runs in the runtime's entry point, where an exception would
    take the session down over a diagnostic, and the honest failure of a typo is the
    default bound rather than no bound at all.
    """
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in ("", "off", "no", "false"):
        return None
    try:
        seconds = float(text)
    except ValueError:
        logger.warning("%s=%r is not a number; using %.0fs", env_name, raw, default)
        return default
    if seconds == 0:
        return None
    if seconds < 0:
        # A NEGATIVE bound is a typo rather than a switch: ``0`` is the written
        # spelling for "do not arm", and nothing else can be meant by -1.
        logger.warning("%s=%r is negative; using %.0fs", env_name, raw, default)
        return default
    if seconds >= MAX_BOUND_S:
        logger.warning("%s=%r is beyond the C timer's range; using %.0fs", env_name, raw, default)
        return default
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
            env_name,
            seconds,
            floor,
            MIN_BOUND_FLOOR_TICKS,
            floor,
        )
        return floor
    return seconds


def bound_seconds() -> float | None:
    """The configured STEADY bound, or ``None`` when the watchdog is switched off.

    This is the bound a runtime is judged against once it has ENGAGED, which
    :func:`engage` moves the timer to; the stretch before that is covered by
    :func:`boot_bound_seconds`. An explicit ``0``/``off``/``no``/``false`` (or an
    empty value) is the one spelling that means "do not arm", which an operator (or
    a test that deliberately blocks a loop) needs.
    """
    return _bound_from_raw(ENV_SECONDS, os.environ.get(ENV_SECONDS), DEFAULT_STALL_S)


def boot_bound_seconds() -> float | None:
    """The configured BOOT bound (:data:`DEFAULT_BOOT_STALL_S`), or ``None`` for
    "no boot phase".

    Read by :func:`arm` and only there: it is the deadline covering the stretch in
    which nothing can stamp a plane, and :func:`engage` is what ends it.

    ``None`` here means the boot phase is DISABLED, not that the watchdog is: the
    entry point arms straight at the steady bound, exactly the behaviour that existed
    before the two bounds were split. The distinction runs opposite to the steady
    knob's, where ``0``/``off`` means "do not arm at all" — worth the paragraph
    because the two knobs accept identical spellings and mean different things by
    them.
    """
    return _bound_from_raw(ENV_BOOT_SECONDS, os.environ.get(ENV_BOOT_SECONDS), DEFAULT_BOOT_STALL_S)


def dump_path(pid: int | None = None, directory: Path | None = None) -> Path:
    """Where this process's dump goes: ``<log dir>/runtime-stall-<pid>.log``.

    A function rather than an inline join because three readers name the same
    file — the runtime's own log line, the tests, and an operator (or an agent)
    who has a record's pid and nothing else.
    """
    from local_operator.paths import log_dir

    base = directory if directory is not None else log_dir()
    return base / f"{DUMP_PREFIX}-{pid or os.getpid()}.log"


def dump_candidates(pid: int | None = None, directory: Path | None = None) -> tuple[Path, ...]:
    """Every place this pid's dump MAY have been written, this process's dir first.

    A WRITER takes :func:`dump_path` — one path, its own — and a READER takes this,
    because a reader holding a dead pid is routinely not the process that wrote the
    file: the directory comes from ``paths.log_dir()``, which honours
    ``LOCAL_OPERATOR_CONFIG_DIR`` PER PROCESS, and the two processes need not agree.
    Measured on this fleet (2026-09-22): the runtimes carry that variable and the
    ``lop serve`` daemon that classifies their deaths does not, so a fired dump sat
    in ``~/.local-operator/logs`` while the classifier resolved
    ``~/Library/Logs/local-operator`` — and the death of a runtime that had dumped
    every thread was narrated ``unattributed`` (``pid 96510``, the card reproduced
    in ``tests/unit/session/runtime/test_turn_journal.py``).

    ``directory`` SUPPRESSES THE SEARCH rather than seeding it: an explicit
    directory is how a test or an isolated run states which store it means, and a
    reader that widened that to the host's real log directories would escape the
    isolation it was handed. One candidate, exactly as before, when it is given.
    """
    from local_operator.paths import log_dirs

    if directory is not None:
        return (dump_path(pid, directory),)
    return tuple(dump_path(pid, base) for base in log_dirs())


def deadline_path(pid: int | None = None, directory: Path | None = None) -> Path:
    """Where the CURRENT deadline goes: ``<log dir>/runtime-stall-<pid>.deadline``.

    THE SIBLING OF THE DUMP, ``<epoch> <leg> <monotonic>`` on one line (~40 bytes), rewritten
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
    (:data:`HOW_TO_READ_THE_FIRED_VALUE`). THE ONE EXCEPTION IS A FAILED RE-ARM: a
    replacement that raises leaves the previous timer in force and writes
    :data:`REARM_FAILED_MARKER` instead of a sibling, so absence-plus-an-armed-value is
    the never-engaged class only when that line is absent (see :func:`engage`).

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

#: What a runtime supplies so the WORK-IN-FLIGHT reading can be decided per re-arm: a
#: zero-argument callable answering "does this process hold work a stop would
#: destroy?" — a turn under the lock, a live subagent lane, a background job, a gate
#: parked on the user (``ServingSessionHandle.is_busy`` is what the runtime passes).
#:
#: WHY IT IS NOT THE PROGRESS LEG. ``in_flight`` above is deliberately
#: narrow — a tool batch whose results have not landed, or an on-demand compaction —
#: because it is one term of a COMPOSITE predicate whose other terms (no motion, CPU
#: advancing) are what make a fire correct. The question this probe asks is a
#: different one and a simpler one: is there work in progress right now? A model
#: call, a long tool, a running subagent and a parked gate all answer yes, and the
#: operator's rule for a build move is the same rule: a runtime is replaced when its
#: turn is COMPLETE, never on a heuristic of inactivity.
#:
#: A CALLER WITH NO PROBE IS RECORDED AS NOT-HELD; one whose probe RAISES is recorded
#: AS HELD (see :func:`_holds_work`, which spells out why those are different facts).
#: The dump is written either way, which is the whole instrument, and the reading is
#: the only thing this decision changes: no timer here can end a process, so what a
#: wrong answer costs is a misleading cell rather than a runtime killed mid-turn —
#: which is exactly the cost the answer is nonetheless sized to avoid.
BusyProbe = Callable[[], bool]


def _bound_class() -> str:
    """The incident class a fire of this bound is recorded under, imported lazily.

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
    boot_seconds: float | None = None,
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

    THE BOUND IT ARMS IS THE BOOT BOUND (:data:`DEFAULT_BOOT_STALL_S`), not the
    steady one, and that is the whole of the entry point's side of the split: the
    entry point runs before anything can stamp a plane, so the deadline it sets is
    measured from here and only :func:`engage` can move it — and ``engage`` is called
    FROM the boot path, at the publication boundary inside ``process.amain``, so what
    cannot reach it is nothing before that boundary rather than the boot path as a
    whole. ``boot_seconds`` overrides the boot bound for a caller that has a
    reason (the suite's children pass one to exercise the mechanism without waiting
    out 900 s); an explicit ``seconds=`` is a statement about THIS process's whole
    arming and therefore stands as both bounds unless ``boot_seconds=`` says
    otherwise.

    Never raises. A diagnostic that cannot be armed must leave the runtime
    otherwise untouched: an unwritable log directory is a reading of "no dump
    possible", not a reason to fail a session's boot.
    """
    global _ARMED
    with _LOCK:
        if _ARMED is not None:
            return True
        steady = bound_seconds() if seconds is None else seconds
        if steady is None:
            logger.info("stall watchdog disabled by %s", ENV_SECONDS)
            return False
        # WHICH BOUND IS ARMED HERE, and why the rule is not simply "the boot knob":
        # the split exists because an operator's steady bound is measured from
        # ENGAGEMENT and a boot cannot be. So the boot bound is the boot knob's when
        # this call resolves its own numbers (the production path — the entry point
        # passes only ``probe=``), and an explicit ``seconds=`` supplies both when a
        # caller has stated the bound for the whole arming (every child in the suite
        # passes one at 1-2 s, far below the floor the environment is held to, and a
        # boot bound of 900 s there would mean no cell could reach a fire).
        if boot_seconds is not None:
            boot = boot_seconds
        elif seconds is not None:
            boot = seconds
        else:
            boot = boot_bound_seconds()
        if boot is None:
            # The boot knob spelled ``off``: no boot phase, so the steady bound from
            # the first instant. The watchdog is NOT disabled here — only the steady
            # knob says that (see :func:`boot_bound_seconds`).
            boot = steady
        # NEVER TIGHTER THAN THE STEADY BOUND, even for the boot stretch: an operator
        # who set the steady knob above the boot default asked for MORE silence than
        # the boot bound allows, and arming shorter than that would cut their runtime
        # during boot and then judge it more leniently afterwards. ``max`` also makes
        # :func:`engage` a downward move by construction, which is the property its
        # cells pin.
        bound = max(boot, steady)
        if boot < steady:
            # THE `max` ABOVE DROPS THE BOOT BOUND HERE, and until agent review round 1
            # (Q-1) it did so SILENTLY: every other unusable value of either knob is
            # announced by ``_bound_from_raw`` (non-numeric, negative, beyond the C
            # timer's range, below the floor), while a boot bound below the steady one
            # left the operator's only artifact naming a number they never configured.
            # The arithmetic is intended — the boot stretch is never judged more
            # tightly than the steady bound, which is also what makes :func:`engage` a
            # downward move by construction — so this is visibility, not a different
            # bound, and a boot window longer than the operator asked for is exactly
            # the kind of thing they need to be told about.
            logger.warning(
                "%s=%gs is below the steady bound of %gs; the boot window is %gs (boot is "
                "never armed tighter than the steady bound)",
                "boot_seconds" if boot_seconds is not None else ENV_BOOT_SECONDS,
                boot,
                steady,
                steady,
            )
        # THE ONE SENTENCE THAT MAKES THE SPLIT READABLE FROM THE ARTIFACT. A fired
        # value on its own cannot say which of the two bounds produced it, and the two
        # classes have different causes: a value at the steady bound is a runtime that
        # ENGAGED and then went silent (the case the steady bound was sized for), while
        # a value at the boot bound is a runtime that never engaged at all — no session
        # judged, no beat ever landing — which is the class 10 of the 17 measured
        # current-build fires belonged to. THAT READING IS CONDITIONAL ON THE RE-ARM, which
        # is why the note below qualifies it rather than stating it flatly (design review
        # round 1, D1): :func:`engage` moves the bound in memory BEFORE it replaces the
        # timer, so a runtime that DID engage but whose replacement failed keeps the
        # original C timer and can still fire at the boot bound — a member of the 10/17
        # class the attribution above would misread as never-engaged. Written only when
        # the two bounds differ: with one bound there is no split to explain, and a header
        # sentence about a phase that cannot exist is how a header starts lying.
        boot_note = ""
        if bound != steady:
            boot_note = (
                f"THIS IS THE BOOT BOUND, covering the stretch before the runtime has a "
                f"session to judge: nothing can stamp a plane in it, so the value above is "
                f"the whole of what a fire in that stretch measured. The runtime's first "
                f"engagement moves the bound to {steady:g}s and stamps BOTH planes. "
                f"When re-arming succeeds, a fired value of {bound:g}s -- this number -- means "
                f"the runtime never engaged. If engagement could not re-arm the timer, see the "
                f"re-arm-failed message in this dump; the timer may still fire at the boot bound.\n"
            )
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
        # for telling the never-engaged class from a stale plane (given a fire at the armed
        # value with no 're-arm failed' line; see :func:`engage`), so a stale file must not
        # be able to fake it. Best-effort: a file that cannot be removed must not stop the
        # bound being armed.
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
                f"{DUMP_POLICY_MARKER}\n"
                f"{boot_note}"
                f"{OBSERVATION_NOT_VERDICT}\n"
                f"{HOW_TO_READ_THE_FIRED_VALUE}"
                f"A FIRE IS AN OBSERVATION AND NEVER AN EXIT: the C timer runs no Python, so "
                f"when it fires the incidents class "
                f"{_bound_class()} is written here and nowhere else, which is why death "
                f"attribution opens this file for a pid whose death nothing else recorded -- "
                f"and whether that runtime is still there is read off the PID and its own "
                f"record, never off this file. A further line below "
                f"carrying the words 'no progress' means the re-arm stopped while the loops that "
                f"produce it were still ticking and burning CPU; its absence means no tick "
                f"reached the timer at all. WHY THIS FILE IS STILL NAMED {DUMP_PREFIX}-<pid>.log "
                f"WHEN A FIRE IS NOT A STALL: it is written at ARM time, so one exists for every "
                f"armed runtime, healthy ones included, and this fleet's incident taxonomy and "
                f"death-attribution reader are keyed to that path -- renaming it is a "
                f"cross-module change, taken whole or not at all rather than half-done here. "
                f"NO FIRE MARKER IS SPELLED HERE, and that is load-bearing rather than tidy: "
                f"readers test for each as a SUBSTRING, so a header quoting one would make "
                f"every armed file -- including one left by a SIGKILL -- read as a fired bound, "
                f"a progress fire, or a runtime with a quiet plane. {DUMP_POLICY_MARKER} ABOVE IS "
                f"THE ONE EXCEPTION AND IT IS DELIBERATE: it is a statement about the BUILD, no "
                f"reader infers a fire or a stalled plane from it, and it is how a dump-only "
                f"fire is told from a legacy fatal one without reasoning from what is absent. "
                f"NO LEGACY POLICY SENTENCE IS EVER SPELLED HERE for the same substring reason: "
                f"those phrases are exactly what proves the OPPOSITE convention, so a header "
                f"quoting one would make every dump this build writes read as a fatal-build "
                f"artifact.\n"
            )
            handle.flush()
            to_arm = _Armed(
                target, handle, bound, pid or os.getpid(), probe, busy, steady_seconds=steady
            )
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
            # sampler is what re-reads the work-in-flight reading while the timer is
            # pending (:func:`_apply_exit_leg`), so a caller that supplies only the busy
            # probe would otherwise hold a reading that could never flip back — and a
            # runtime that picked up a turn after its last arm would then be recorded as
            # an idle observation for the rest of that arm.
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
    * **the probe RAISED** → ``True``, and this is an INVARIANT rather than a detail:
      the process told us it has a way to report work and the report could not be read,
      so the unreadable state must not authorise a CLAIM OF IDLENESS and must not
      suppress the fire either. ``True`` means the FIRE still happens — the timer
      expires, the dump is written, the class recorded, the held line appended — and it
      means the fire is filed as held, which is the answer that cannot say "nothing was
      in flight". That is pinned by
      ``test_an_UNREADABLE_work_report_still_fires_and_holds``, which drives a real
      child with a raising probe and asserts the fired marker is in the dump. The cost
      of holding wrongly is one ``lop stop`` against a turn that nothing can
      reconstruct, and the way out is the control ladder's SIGKILL rung (driven against
      a real wedged process in ``test_the_escape_hatch_reaches_a_runtime_wedged_in_a_c_call``);
    * **no probe was supplied** → ``False``, i.e. TODAY'S BEHAVIOUR. This is a caller
      that has been given no way to report what it is doing (a rig, an in-process
      host, a test of the liveness leg alone), and the bound's documented contract
      for such a caller is unchanged rather than silently withdrawn. The distinction
      matters because the ONE production arm site (``stall_watchdog.arm`` in
      ``process.py``'s ``__main__`` guard, the only place a real runtime child arms
      at all) always supplies the probe: there is no released runtime that can answer
      and does not, so calling absence "held" would only ever mark a state no real
      runtime reaches — the dead-instrument shape, not a safety win.
    """
    if busy is None:
        return False
    try:
        return bool(busy())
    except Exception:  # noqa: BLE001 — uncertainty keeps the runtime
        logger.debug(
            "stall watchdog: the busy probe failed; the fire is recorded as held", exc_info=True
        )
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
    """Arm (or replace) the C timer for diagnostics, never process termination.

    ``exit_leg`` remains an input because callers compute and record their local
    decision, but a sampled idle state is not atomic with every work-admission path.
    No production timer may use it to authorize a native process exit.

    ONE SPELLING FOR EVERY PRODUCTION TIMER SITE (the arm, a plane's beat, the
    progress leg's own fire). ``exit_leg`` remains explicit because callers record
    their Python-side sample, but the native call below is ALWAYS ``exit=False``: no
    value a caller can pass reaches it, which is why the argument can no longer decide
    anything about a process (finding C of the 2026-09-23 convergence round, and
    ``test_no_production_timer_site_can_arm_a_fatal_expiry`` pins both halves of it).
    A Python sample cannot be re-checked at native expiry, so a sampled idle state can
    never be treated as authority to end a process that may already be serving again.

    REPLACING IS THE ARMED STATE: a second call supersedes the pending timer rather
    than adding one (measured, and the module docstring's inventory says so), so
    each re-arm replaces the prior diagnostic deadline without creating a second
    timer.
    """
    faulthandler.dump_traceback_later(max(MIN_REARM_S, remaining), file=handle, exit=False)


def _rearm(armed: "_Armed", *, remaining: float | None = None) -> None:
    """Re-arm for the earliest deadline, with the work-in-flight reading this runtime holds.

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
            # THE HELD-FIRE BACKOFF (agent review round 1, MINOR-5). While the work
            # stays in flight the episode repeats, and every repeat writes another
            # all-thread dump: at the shipping bound that is ~14 KB every 300 s, i.e.
            # ~4 MB a day, on a runtime a person has already been told about. So the
            # interval between dumps DOUBLES per survived fire up to an hour, which
            # keeps "the bound fired and it is still stalled" on the record at a rate a
            # person can read and bounds the artifact at tens of KB a day instead of
            # megabytes. The budget is a CAP, never a stop: a held runtime keeps
            # producing evidence, and nothing here turns the bound into a monitor that
            # has given up — the fail-safe for a plane that truly stopped reporting is
            # still the ordinary liveness leg.
            # ...AND NEVER SHORTER THAN THE BOUND IT STARTS FROM (agent review round 2,
            # MINOR-1): the cap is a ceiling, so an operator whose bound is ABOVE it
            # (``LOP_RUNTIME_STALL_SECONDS=7200`` is accepted) would otherwise have got
            # half the bound they configured — a "backoff" that fires twice as often.
            interval = max(
                armed.seconds,
                min(
                    armed.seconds * (2 ** min(armed.held_fires, HELD_FIRE_BACKOFF_STEPS)),
                    HELD_FIRE_BACKOFF_MAX_S,
                ),
            )
            remaining = max(remaining, interval - (time.monotonic() - armed.after_fire_at))
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

    Returns whether a fire was recorded. Order matters: the line is written before the
    re-arm, so a reader never sees the re-armed timer without the statement about the
    fire that preceded it.

    WHICH LINE, AND WHY IT IS NOT ALWAYS THE SAME ONE: :data:`HELD_MARKER` when work
    was in flight at the fire and :data:`OBSERVED_MARKER` when it was not (idle, or
    cleared). Only the first one is the live stalled state a listing reads off
    (:func:`held_pids`), so a fire the runtime survived with nothing in flight stays a
    diagnostic observation instead of turning that state on — see the markers'
    docstrings for the incident that produced the split.

    WHY THE SIZE AND THE COUNT ARE READ TOGETHER, and not just the size: growth on
    its own cannot distinguish a fire from this module's own lines. The pair can,
    because the C handler writes the stack dump and appends the marker in ONE call
    with no scheduling point between them (the probe it consults is pure Python), so a
    sampler can never observe the growth first and re-baseline over a fire it should
    have annotated — and the count comparison is what makes a repeat observation of
    the same fire a no-op rather than a second report.

    WHAT HAPPENS NEXT depends on the ONE question the busy probe answers, read again
    here, and on it alone — never on the sampled state alone, and never as
    authorization for a native exit (every timer this module arms is dump-only):

    * the work is still in flight — re-arm for a full bound. The runtime is stalled
      with a turn in it, the person or the ``lop stop`` is the way out, and one dump
      per bound says so without writing a file per re-arm;
    * the work has CLEARED since, or was never there — drop the sampled hold and keep
      the diagnostic timer re-armed. A later fire still records evidence, under
      whichever line its own reading earns. A recovered runtime remains alive; a
      persistent wedge requires the operator to inspect its dump and stop it
      explicitly.
    """
    # Every timer is dump-only, so a fire always survives long enough to be
    # observed here, even when the last Python sample said the runtime was idle.
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
    fired_mono = time.monotonic()
    armed.after_fire_at = fired_mono
    # WHICH MARKER GOES IN IS DECIDED BEFORE EITHER LINE IS WRITTEN (finding B of the
    # 2026-09-23 convergence round). The order used to be append-then-decide, which is
    # how a fire over an IDLE runtime came to carry ``HELD_MARKER`` — the very word the
    # live ``stall_held`` state is read off (:func:`held_pids`) — and so reported a
    # process with nothing in flight as permanently stalled. The probe is read here, on
    # the sampler's or the beat's thread, exactly as it was before; what moved is only
    # that its answer now chooses the LINE rather than trailing it.
    held = _holds_work(armed.busy)
    if held:
        armed.held = True
        armed.held_fires += 1
        _append_dump_line(
            armed,
            # THE MONOTONIC STAMP IS THE MACHINE-READABLE HALF; the local time beside it is
            # for a person. :func:`_holds` compares it with the sibling's monotonic
            # deadline, both written by THIS process on ONE clock, because a wall-clock
            # comparison broke three ways (agent review round 1 M1/m1, QA Q-1): macOS
            # pauses ``monotonic`` across sleep so a monotonic-derived epoch drifts
            # forward, the repeated DST hour parses an hour early, and a reader in another
            # zone parses by its offset — each reading a stuck runtime as recovered.
            f"{HELD_MARKER}{_FIRED_AT_PHRASE}"
            f"{time.strftime(_FIRED_AT_FORMAT)} ({_MONOTONIC_TAG}{fired_mono:.3f}) "
            "and did NOT end this runtime. Work was "
            "in flight when the fire was observed, so the runtime is stalled with it. "
            "Every thread's stack is above. Inspect this dump and stop the runtime "
            "explicitly if it remains stuck.\n",
        )
    else:
        armed.held = False
        # THE COUNTER IS PER EPISODE, NOT PER PROCESS (agent review round 2, MINOR-1).
        # The docstring's subject is "while the work stays in flight the episode
        # repeats", and a lifetime counter outlived that: a runtime that survived a
        # fire, cleared its work and wedged AGAIN later re-armed at the backed-off
        # interval — up to twelve times the configured bound — which would also delay
        # the next diagnostic dump after work clears. The hold clearing is exactly where
        # an episode ends; no timer fire authorizes production termination.
        armed.held_fires = 0
        # THE NEUTRAL LINE, and the difference is what a LISTING shows: no work in
        # flight means this is not the state a person has to unblock, so it must not
        # raise ``stall_held``. The fire is still on the record in full — the moment,
        # and every thread's stack above — because "the bound fired on something that
        # was not a turn" is itself the observation worth keeping.
        _append_dump_line(
            armed,
            f"{OBSERVED_MARKER}the bound fired at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} and did NOT end this runtime. No work "
            "was in flight when the fire was observed, so this is a diagnostic "
            "observation rather than the held state: the sample was idle, or the work "
            "had cleared. Every thread's stack is above.\n",
        )
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
    if held:
        logger.warning(
            "stall watchdog: the bound fired for pid %s and did NOT end this runtime "
            "(work in flight); the runtime is STALLED, dump at %s",
            armed.pid,
            armed.path,
        )
    else:
        # THE SAME FIRE, WITHOUT THE CLAIM THE MARKER NO LONGER MAKES: the log line is
        # read by an operator next to the listing, so a sentence saying STALLED about a
        # runtime with nothing in flight would put the false reading in two surfaces.
        logger.warning(
            "stall watchdog: the bound fired for pid %s and did NOT end this runtime "
            "(no work in flight); the dump is at %s",
            armed.pid,
            armed.path,
        )
    return True


def _apply_exit_leg(armed: "_Armed", held: bool) -> None:
    """Re-arm for the work-in-flight reading just taken, when it differs from the one
    in force.

    THE NAME IS HISTORICAL (see :data:`HELD_MARKER`): this used to be the leg that
    decided whether a native exit was authorized, and what it decides now is which
    marker a fire is recorded under — :data:`HELD_MARKER` or :data:`OBSERVED_MARKER`.

    THE READING HAPPENS OUTSIDE :data:`_LOCK` AND THE DECISION INSIDE IT, which is a
    constraint rather than a preference: ``beat`` takes that lock on the serving
    plane's thread and on the workload loop, and a probe that takes a while to answer
    would hold the bound's own bookkeeping against the very ticks it exists to time
    (the same property the progress leg's probe call is being moved out of the lock
    for, in the change that owns that leg). Splitting the call from the mutation is
    what lets both happen without either one widening the critical section.

    Only a FLIP re-arms. A re-arm on every sample would push the deadline out every
    interval and the liveness leg could then never fire at all — a dead instrument,
    in this module's own words.
    """
    if held == armed.held:
        return
    armed.held = held
    if not held:
        # The other place an episode ends: the sampler watches the work clear while no
        # fire is pending (see ``_record_held_fire``'s counter note).
        armed.held_fires = 0
    try:
        _rearm(armed)
    except (OSError, ValueError, RuntimeError):
        logger.debug("stall watchdog could not re-arm after a work-in-flight flip", exc_info=True)


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
        # THE THIRD FIELD IS THE SAME DEADLINE ON THE CLOCK IT WAS COMPUTED ON. The epoch
        # is for a reader holding a dead pid (compare it with the fire's wall time); the
        # monotonic value is what :func:`_holds` compares with the held marker's own
        # stamp, because the epoch is derived through ``time.time() - monotonic()``, an
        # offset that grows across every host sleep (agent review round 1, M1). Appended
        # rather than inserted, so ``epoch leg`` stays the first two tokens it always was.
        temp.write_text(f"{epoch:.3f} {leg} {deadline:.3f}\n", encoding="utf-8")
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

    AND IT LEARNS WHICH THREAD TO WATCH, which is the one fact the liveness leg's
    fairness fix cannot get any other way. A beat is issued FROM the loop it
    reports for, so this call site IS the identification: for the WORKLOAD plane
    that is the runtime's own event loop (``process._beat_stall_watchdog`` is an
    ``asyncio`` task on it), and for SERVING it is the serving plane's own loop
    thread. Assuming the main thread instead would mis-name the serving plane, and
    assuming nothing (leaving the field empty) would silently disable the fix on a
    runtime whose loop is on a thread this module never guessed — so it is
    observed here, where it is definitionally true.
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
        # The THREAD OBJECT rather than its ident, and the difference is not
        # cosmetic: idents are recycled once a thread ends, so a stored ident starts
        # naming whoever holds it next and this plane would be watched through
        # another thread's frames. Keeping the object is what lets the observation ask
        # ``is_alive()`` — see :attr:`_Armed.loop_thread`.
        thread = threading.current_thread()
        if armed.loop_thread.get(plane) is not thread:
            # A NEW LOOP THREAD FOR THIS PLANE, and the recorded baseline goes with the
            # old one: the previous thread's last frame says nothing about where this
            # one is, so comparing across the change would report a movement that never
            # happened — a sign of life invented out of a thread swap, which is exactly
            # what ``observe_execution`` must never do.
            armed.loop_thread[plane] = thread
            armed.last_frame.pop(plane, None)
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


def engage() -> bool:
    """End the BOOT phase: move the bound to the steady one and stamp BOTH planes.

    THE RESET IS ON ENGAGEMENT, NOT ON ARM, and this function is the whole of the
    difference. :func:`arm` runs before the runtime has a session, a loop or a
    socket, so the deadline it sets is measured from the entry point — a clock no
    boot code can re-arm (:data:`DEFAULT_BOOT_STALL_S`). This call belongs where the
    runtime HAS something to judge, so from here the deadline must be measured from
    NOW: it stamps both planes to this instant and moves the bound down to the steady
    one, which is what makes that bound mean what it was sized for (a ceiling on one
    silent synchronous step in a runtime that is running) rather than an accident of
    how long boot happened to take.

    BOTH PLANES, and that is not symmetry: in the measured fires where a beat HAD
    re-armed the timer, the re-arm line names WORKLOAD as "has reported nothing since
    the ARM" while SERVING "reported 0 s ago". The workload plane is the long pole of
    boot — its tick starts last, and it sleeps a heartbeat before its first stamp —
    so a reset that stamped only the plane that happens to be beating would leave the
    workload plane's seed at arm time and re-arm the very deadline this exists to move.

    IT NEVER WIDENS THE BOUND. The move is downward by construction (:func:`arm`
    seeds the armed bound at the LARGER of the two), and when the armed bound is
    already at or below the steady one there is nothing to move: that is the state of
    an explicit ``arm(seconds=...)``, of a boot knob spelled ``off``, and of a steady
    knob set at or above the boot bound. In all three this is a documented no-op
    rather than an extension, and a cell
    (``test_engage_never_widens_the_steady_bound``) goes red on any edit that lets it
    raise the bound a silent runtime is judged against.

    Idempotent, and a no-op when nothing is armed — the same contract as
    :func:`beat`, and for the same reason: an in-process host (a TUI, a test) calls
    this on a boot path that never armed the process timer, and it must cost it
    nothing.

    IT RE-ARMS THROUGH :func:`_rearm`, NOT WITH A TIMER CALL OF ITS OWN, and that is
    a requirement rather than a refactor: ``_rearm`` is the one place that decides how
    a fire is recorded (the fired-held bookkeeping, which is what
    :func:`_record_held_fire` and the held/neutral split hang off) and the one that
    passes the exit leg. A timer call written here would silently BYPASS that
    bookkeeping for the one arming that starts the steady phase — the arming whose
    bound the rest of the design rests on. The remaining time it is
    handed is derived from the stamps this call just moved, so the timer holds
    exactly the steady bound measured from NOW.

    Returns whether THIS call moved the bound, which is what lets a caller — and a
    cell — tell the boot-to-steady transition from a call that had nothing to do. It
    is a diagnostic and never a control: every failure below leaves the runtime with a
    live bound and the caller with nothing to handle.
    """
    with _LOCK:
        armed = _ARMED
        if armed is None:
            return False
        if armed.seconds <= armed.steady_seconds:
            # Nothing to move, which is also the SECOND call's path: the bounds are only
            # equal before an engage when there is no boot phase to end, and equal after
            # one because this is what made them equal.
            return False
        now = time.monotonic()
        armed.seconds = armed.steady_seconds
        for plane in PLANES:
            armed.last_beat[plane] = now
        try:
            # ONE CALL, NO CANCEL FIRST, for the reason :func:`beat` gives: the replace is
            # atomic, and a cancel-then-fail would leave the process with no bound while
            # the steady bound is what everything below assumes is in force. Both planes
            # have just been stamped to ``now``, so the remaining time ``_rearm`` derives
            # is the steady bound exactly, and the work-in-flight reading comes from the
            # probe as it does on every other re-arm.
            _rearm(armed)
        except (OSError, ValueError, RuntimeError):
            logger.warning("stall watchdog could not re-arm its timer on engage", exc_info=True)
            # AND THE FAILURE IS NAMED, because the attribution a reader gets otherwise is
            # the WRONG CLASS: the sibling is absent (no beat has ever written it), which
            # is the signature of the never-engaged runtime, while the value on a fired
            # line would be the boot bound. The bound did move in memory, so the dump has
            # to carry both facts. THIS LINE IS ALSO WHAT KEEPS THE HEADER'S NOTES TRUE:
            # the qualified never-engaged reading is safe exactly when this line is absent
            # (see :data:`HOW_TO_READ_THE_FIRED_VALUE`), so a reader who finds it in the
            # dump must not make that reading.
            if not _write_dump_line(
                armed,
                f"{REARM_FAILED_MARKER}the engage could not re-arm the timer, so a fired line "
                f"below carries the BOOT bound ({armed.boot_seconds:g}s) rather than the "
                f"steady one ({armed.steady_seconds:g}s) this runtime moved to\n",
            ):
                logger.warning(
                    "stall watchdog could not record a failed engage for pid %s", armed.pid
                )
        else:
            # AFTER a successful re-arm, and only then, exactly as :func:`beat` does: the
            # sibling must carry the deadline the timer really holds. Its EXISTENCE is
            # also the reader's half of the split — a runtime that re-armed has one, and
            # the never-engaged class has none (see :func:`deadline_path`) — and it is the
            # successful-re-arm half of the qualification the header notes carry, which is
            # why the failure above writes a line rather than leaving absence to speak.
            _record_deadline(armed)
        return True


def note_tick_death(plane: str, reason: str) -> bool:
    """Record durably that ``plane``'s TICKER died, in the armed dump file.

    Beside the plane's own stamp rather than in a file of its own, and that is a
    choice about WHAT A READER HAS IN HAND rather than about tidiness: the stamp
    (:attr:`_Armed.last_beat`) is memory and dies with the process, and the only
    artifact that outlives a watchdog fire is the dump. So the fact that a plane
    stopped being REPORTED goes where the report about that
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

    A DAEMON thread, and its lifecycle does not depend on a fatal watchdog exit:
    production timers are dump-only. The arming sequence ensures a thread that
    outlived its
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


def _read_probe(probe: "ProgressProbe | None") -> "tuple[object, bool] | None":
    """Read the injected probe, or ``None`` when there is no LIVE answer for this sample.

    TWO ``None`` SOURCES, ONE MEANING. There is no probe at all (an in-process host,
    a test of the liveness leg alone, an older spawner), or the probe RAISED. Both
    mean the progress leg cannot speak for this sample, which matters beyond the
    progress leg: :func:`_extend_for_execution` refuses to abstain when there is no
    live second leg to hand the process to, so an unevaluable sample must not read as
    an evaluated one.

    CALLED OUTSIDE :data:`_LOCK`, and that is the point of it existing separately
    from :func:`_sample`: the production probe reaches LIVE session state from a
    thread that is not the session's — transcript footprints, job rows, the
    tool-batch tail of a context another thread is appending to — and it can be slow
    for as long as that state is contended. Holding ``_LOCK`` across it makes every
    :func:`beat` queue behind a diagnostic read, so a slow probe delays the stamp
    that is supposed to prove the runtime is alive. What stays under the lock is the
    DECISION, because that is what has to be serialised against a beat's re-arm.
    """
    if probe is None:
        return None
    try:
        return probe()
    except Exception:  # noqa: BLE001 — an unevaluable probe must not end a process
        logger.debug("stall watchdog: progress probe failed", exc_info=True)
        return None


def _note_executing(armed: "_Armed", noted: "tuple[str, ...]") -> None:
    """Record, in the dump, that these planes STARTED a stretch of executing code.

    WRITTEN UNCONDITIONALLY when a stretch opens, including on runs where the hand-off
    below is REFUSED: the question a reader of this file is asking is "what was the
    loop doing when the bound fired", and that question does not stop being worth
    answering because no probe was supplied. :func:`executing_planes` is the reader, and
    it means "observed executing", NOT "the abstention was granted" — the granting is
    visible in the deadline this module did or did not move, never in whether the
    process left.

    ``noted`` IS ALREADY THE TRANSITION SET, decided by
    :meth:`_Armed.observe_execution` rather than here, because the rule that ends a
    stretch is a property of the stamps: one line per STRETCH, not per sample.
    """
    for plane in noted:
        frame = armed.last_frame.get(plane)
        if isinstance(frame, tuple):
            where = ":".join(str(part) for part in frame)
        else:  # pragma: no cover - a moved plane always has a frame to name
            where = "unknown frame"
        _write_dump_line(
            armed,
            f"{EXECUTING_MARKER}{plane} in {where} -- this plane's loop is executing code, "
            f"so its silence is not a park; the bound abstains for it while its frames "
            f"keep moving, and the progress leg still bounds a loop that runs without "
            f"advancing.\n",
        )


def _extend_for_execution(armed: "_Armed", now: float, moved: "tuple[str, ...]") -> None:
    """Do NOT fire on a plane whose loop was just observed EXECUTING; re-arm instead.

    THE ABSTENTION, and the one place this module deliberately declines to enforce its
    own bound. The liveness leg's question is "is anything running at all"; a plane
    whose frames are moving has answered that question YES without a stamp, and the
    state that starves a stamp (the loop busy in its own code, which is what the
    2026-09-22 fires show: the loop inside the session's per-event subagent projection
    at the instant the deadline expired) is a loop that is WORKING. A fire there is a
    FALSE reading — it calls a loop that was making progress one that stopped reporting
    — which is a strictly worse failure than a late bound.

    WHY THIS DOES NOT UNBOUND A RUNAWAY, which is the argument to read before changing
    it. The two legs partition the state space: the liveness leg owns "this loop never
    came back from a call" and the progress leg owns "this loop is running while its
    work stands still". A loop whose frames move is in the second class by
    definition, and the progress leg judges it on the same bound — so abstaining
    here hands the diagnostic observation over rather than dropping it. The hand-off is
    why the extension is refused outright unless the progress leg can actually speak
    (see below), and why nothing here touches the CPU floor or the motion tuple: a
    frames-moving runaway that burns a core with no work moving still fires, on the
    other leg, with its own marker naming it.

    THE EXTENSION IS BOUNDED, and by structure rather than by a number, because a
    second magic constant is the failure mode this module has already paid for twice:
    the deadline is always ``sign_of_life + bound``, and ``sign_of_life`` is only ever
    moved by an observation made ON THIS SAMPLE. So the bound can be pushed no more
    than one interval past the last moment the loop was SEEN to execute, and it is
    re-derived rather than accumulated — there is no counter that a stuck sample could
    leave raised. Concretely, extending requires, on the same sample: a probe supplied,
    a probe that ANSWERED, and a frame that MOVED. Any one of them failing leaves the
    deadline counting down from the last observation, and the bound fires within
    ``bound`` of it.

    ITS FAILURE MODE, stated rather than hidden: a loop that executes AND whose work is
    moving (or which holds a tool batch in flight) is bounded by neither leg, since
    both legs use exactly those two facts to abstain. That class is not created here —
    an in-process tool that burns a core for an hour is already spared by the progress
    leg's second clause, by design — but this change WIDENS it to any turn that keeps
    its transcript or roster moving while it walks frames. The compensating fact is
    that such a runtime is doing work, and the module's stated preference is to spare
    a working runtime past any bound rather than cut a dead one late.

    A PLANE WITH NO LEARNED THREAD IS NEVER EXTENDED (``observe_execution`` returns
    nothing for it), so the never-engaged class and every plane that never stamped
    behave exactly as before.

    AND THIS IS THE ONE WRITER OF THE EXTENSION ITSELF: ``moved`` is an OBSERVATION
    that ``_sample`` has already gated (a probe supplied and answering on this same
    sample) before passing it here, and the timestamp that moves the deadline is
    written below and nowhere else. The observation must not write it: a refused
    hand-off that had already moved the deadline would leave an executing loop with a
    broken probe unbounded on any runtime whose OTHER plane kept stamping, since
    ``beat`` re-arms the shared timer from ``pin``/``deadline`` for the whole
    process. Gating the observation's write is what makes the refusal total.
    """
    if not moved:
        return
    if armed.probe is None:
        # No receiver for the hand-off, so no hand-off: with the progress leg inert,
        # abstaining would leave the process bounded by NOTHING, and a moving frame
        # alone is not a claim that the work is advancing. The observation has still
        # been recorded by ``_note_executing``. NOTHING IS WRITTEN HERE — not the
        # extension, not the deadline — which is the whole of the refusal.
        return
    for plane in moved:
        # BEFORE the re-arm, because the re-arm reads it back through ``deadline()``.
        armed.executing_at[plane] = now
    try:
        # THROUGH ``_arm_timer``, the one production policy boundary. This sample
        # extends the diagnostic deadline while a loop frame moves; it must not bypass
        # the shared wrapper, which guarantees ``exit=False`` on every re-arm.
        _arm_timer(
            armed.handle,
            max(MIN_REARM_S, armed.deadline() - now),
            exit_leg=not armed.held,
        )
    except (OSError, ValueError, RuntimeError):
        # A re-arm that cannot happen leaves the timer on its previous deadline, which
        # is SHORTER than this extension. That direction is safe — the bound fires
        # early rather than late — so it is reported and not escalated, and the line
        # ``_note_executing`` wrote is what keeps the run from being silent. Driven by
        # ``test_a_failed_re_arm_leaves_the_timer_on_the_shorter_deadline``.
        logger.warning("stall watchdog: could not re-arm for an executing loop", exc_info=True)
    else:
        _record_deadline(armed)


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

    AND IT IS ALSO THE LIVENESS LEG'S ONLY WITNESS, which is why the interval is
    the right cadence for the frame observation too: this thread runs whatever the
    loop is doing, so a loop too busy to stamp is exactly the loop this thread is
    still able to look at.

    THE PROBE IS READ BETWEEN TWO LOCK HOLDS, not inside one (see
    :func:`_read_probe` for why the read must not hold ``_LOCK``). The armed-ness is
    re-checked after the unlocked section rather than trusted from before it: that
    section can take arbitrarily long, and ``disarm`` can run inside it.
    """
    # THE FIRST CADENCE, and only the first: it is re-derived from the LIVE bound on
    # every wake below, because ``armed.seconds`` is no longer a value fixed at the arm
    # (agent review round 1, R1-3 / QA Q-2).
    interval = _sample_interval(armed.seconds)
    while not stop.wait(interval):
        # THE WHOLE BODY IS GUARDED, and it is the same rule ``server._heartbeat_loop``
        # now follows one plane over (agent review round 1, MINOR-1). This thread is the
        # ONLY thing that re-reads the work-in-flight reading after the arm, and a raise from the
        # sample or from the fire bookkeeping would end it silently: ``armed.held``
        # would then stay frozen at whatever it last read, and a frozen ``False`` with
        # work in flight is precisely why production expiry remains diagnostic-only;
        # the native timer fire needs no Python at all. A reporter that dies takes
        # its plane's evidence with it; the fix is to keep it running.
        try:
            # #1438's probe read, UNCHANGED: the progress probe is taken from under
            # the lock and CALLED outside it, then the armed-ness is re-checked after
            # the unlocked section rather than trusted from before it (that section can
            # take arbitrarily long, and ``disarm`` can run inside it).
            with _LOCK:
                if _ARMED is not armed:
                    return
                probe = armed.probe
            observation = _read_probe(probe)
            # ...and the WORK-IN-FLIGHT probe is read out here for the same reason: this
            # thread is the only one that re-reads that reading after the arm, and
            # holding the lock across a probe would hold it against the very ticks the
            # bound exists to time (see ``_apply_exit_leg``).
            held = _holds_work(armed.busy)
            with _LOCK:
                if _ARMED is not armed:
                    return
                # THE CADENCE FOLLOWS THE LIVE BOUND, re-derived here on every wake
                # rather than captured once before the loop. :func:`engage` moves
                # ``armed.seconds`` DOWN (900 -> 300 at the shipped defaults) while the
                # window :func:`_sample` accumulates reads that same attribute, so a
                # cadence taken at the arm would stay the BOOT bound's for the rest of
                # the process's life — a sampler looking LESS often than the window it
                # measures implies. Measured before this line existed: probe-call gaps
                # of 7.504 s both before and after an engagement moved a 90 s bound to
                # 45 s, where ``_sample_interval(45)`` is 3.75 s. The direction was
                # safe (``_sample_interval`` is monotone in the bound, so the frozen
                # value can only be too coarse) and capped at ``HEARTBEAT_INTERVAL_S``,
                # which is why this was a MINOR and not a mechanism defect — but at a
                # steady bound under 180 s it is the difference between the window's
                # ``/PROGRESS_SAMPLES_PER_WINDOW`` resolution and the liveness beat's
                # ceiling, i.e. a leg deciding on fewer looks than it intends.
                # Re-derived BEFORE the fire bookkeeping, so the ``continue`` on a
                # recorded held fire waits on the fresh value too.
                interval = _sample_interval(armed.seconds)
                # BEFORE the sample, and in this order: a fire that landed since the
                # last wake is recorded and re-armed on the same pass, only then is the
                # work-in-flight reading applied — so a flip made while the timer was
                # pending is in force before anything else acts on the arm — and only then
                # is the next progress decision taken.
                _apply_exit_leg(armed, held)
                if _record_held_fire(armed):
                    continue
                if _sample(armed, observation):
                    return
        except Exception:  # noqa: BLE001 — a diagnostic never ends a reporter
            logger.warning(
                "stall watchdog: the work-in-flight sampler raised; it keeps running so the "
                "reading stays fresh",
                exc_info=True,
            )


def _sample(armed: "_Armed", observation: object = _UNSET) -> bool:
    """One progress sample. True when the bound was FIRED, so the sampler is done.

    Called with :data:`_LOCK` held, and it must be: it reads and writes the same
    ``progress_deadline`` that :func:`beat` re-arms from, on another thread, and
    the interleaving that lets slip is a beat re-arming the long deadline over a
    fire that had already decided to happen.

    EVERY LEG IS EVALUATED ON THE SAMPLE IT ARRIVES, all three to the same
    instant: reading the CPU rate at one moment and the motion at another is how
    a legitimate tool boundary gets read as a spin.

    ``observation`` is the probe's answer, already read OUTSIDE the lock by
    :func:`_progress_sampler`; leaving it at :data:`_UNSET` makes this read the probe
    itself, which is what the predicate cells that drive ``_sample`` directly want.
    The default is not a second implementation of the leg — :func:`_read_probe` is the
    one reader either way — it is where the read happens, and the sampler's cell proves
    the production path does not hold the lock across it.
    """
    clock = armed.clock
    now = time.monotonic()
    cpu = time.process_time()
    # THE FRAME OBSERVATION COMES FIRST, before any leg can decide to leave: its
    # result is what the abstention below needs. THE INSTANTS ARE NOT ALWAYS THE SAME
    # ONE, and it is worth saying rather than implying: on the production path the
    # MOTION below was read before this sample took ``_LOCK`` (the probe is
    # deliberately allowed to be slow — see :func:`_progress_sampler`), so the run is
    # judged on an answer that may be older than this frame read. The direction is
    # safe — a stale motion or in-flight answer is the run's state as of an earlier
    # instant, which can only DELAY the progress leg, the leg that ends a runtime —
    # but the claim is "no earlier than", never "the same instant".
    executing, noted = armed.observe_execution(now)
    # Recorded BEFORE the progress leg can decide to leave, and regardless of whether
    # the hand-off below is granted: the reader's question is what the loop was DOING
    # (see :func:`_note_executing`). Only planes that OPENED a stretch are written.
    # A REFUSED HAND-OFF STILL WRITES IT, so this line can appear on a run that then
    # FIRES at the bound: it is a record of an observation, and
    # :func:`executing_planes` reads it as one — "this loop was seen executing", never
    # "this runtime was spared".
    _note_executing(armed, noted)
    if observation is _UNSET:
        read: object = _read_probe(armed.probe)
    else:
        read = observation
    # A LIVE ANSWER IS A TUPLE. An unevaluable read (``None``) and an absent probe
    # collapse to the same thing here — "no live second leg for this sample" — which is
    # why the abstention's refusal (see :func:`_extend_for_execution`) keys on this
    # rather than on the probe merely being present.
    probe_answered = isinstance(read, tuple)
    motion, in_flight = read if probe_answered else (_NO_SAMPLE, True)
    moved = clock.motion is not _NO_SAMPLE and motion != clock.motion
    if in_flight or moved:
        # A disagreeing sample ends the run outright: a leg that is not true NOW
        # is not "possibly true", and a window that kept accumulating across a
        # sample where the process was working would fire on a runtime that had
        # done legitimate work inside it.
        clock.restart(motion)
        _extend_for_execution(armed, now, executing if probe_answered else ())
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
    _extend_for_execution(armed, now, executing if probe_answered else ())
    return False


def _fire_progress(armed: "_Armed", now: float) -> None:
    """Record a progress-leg fire in the same diagnostic dump used by the liveness leg.

    The progress predicate still writes its identifying line and sets an immediate
    diagnostic deadline, but this function does not terminate the runtime. Production
    ``_arm_timer`` always passes ``exit=False`` because a Python sample cannot be
    atomic with every work admission before native expiry.

    THE SAME ARTIFACT, rather than a second one: the dump file, the ``FIRED_MARKER``
    and every-thread stack are shared by both legs, which keeps a reader's rule
    ("a dump with the fired marker is evidence a bound actually fired") true for both.
    What this adds is
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
                # half of how a never-engaged fire is told apart from a stale-plane one
                # (the other half being the absence of a 're-arm failed' line).
                # WHEN A HELD FIRE IS KEPT the pair is kept with it, deliberately: the
                # sibling names the deadline the fire was waiting for, so a reader
                # comparing "was due at" against "fired at" reads one episode rather
                # than two, which is exactly the cross-check a survived bound needs
                # (agent review round 1, NIT-2).
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

    NEITHER LINE SAYS THE BOUND ENDS THE PROCESS (finding C of the 2026-09-23
    convergence round): this is the line an operator reads FIRST after a freeze, so a
    sentence promising an exit here would be the stale claim of the policy this module
    removed, in the one place a tired reader is most likely to believe it.
    """
    with _LOCK:
        armed = _ARMED
        if armed is None:
            return
        if armed.seconds == armed.steady_seconds:
            logger.info(
                "stall watchdog armed: %.0fs of no progress dumps every thread to %s; the "
                "bound is dump-only, so it ends nothing -- a fire over work in flight (a turn, "
                "a subagent or a job) is recorded as held and the runtime keeps serving",
                armed.seconds,
                armed.path,
            )
            return
        # BOTH BOUNDS, when they differ, and for the same reason the dump header spells
        # the split: this line is the artifact an operator reads FIRST after a freeze,
        # and one claiming the boot bound for the whole life would be a log that
        # misstates the bound the runtime is actually judged against from publication on.
        # The held-fire sentence carries over unchanged: the in-flight reading applies to
        # a fire under EITHER bound.
        logger.info(
            "stall watchdog armed: %.0fs of no progress during BOOT (before the runtime has a "
            "session to judge), then %.0fs once it engages -- either one dumps every thread to "
            "%s; neither ends the runtime (dump-only), and a fire over work in flight is "
            "recorded as held",
            armed.seconds,
            armed.steady_seconds,
            armed.path,
        )


def note_runner_teardown() -> bool:
    """Annotate the dump: the asyncio runner is about to begin its OWN teardown.

    CALLED FROM THE ONE PLACE THAT OWNS THAT BOUNDARY: the coroutine wrapper around
    ``amain`` in ``process.main``, whose ``finally`` runs as the coroutine completes and
    BEFORE ``asyncio.Runner`` starts cancelling tasks and joining the default executor.
    The ordering is the annotation's entire value: the fires this exists to attribute
    are taken while the runner is parked in ``shutdown_default_executor``, and by then
    no Python thread of ours is left to write anything — so a line written LATER would
    describe a phase this process had already left (:func:`fired_in_runner_teardown`
    reads the ordering alone).

    NAMED FOR WHAT IS KNOWN, not for what usually comes next: task cancellation and
    async-generator shutdown run before the executor join, so the line says the runner
    "started" its teardown and that the join MAY follow. It is a phase marker, never a
    diagnosis.

    A NO-OP WHEN NOTHING IS ARMED, which is every in-process host and every TUI: the
    annotation is a line in a dump, so a caller with no dump has nothing to annotate.

    IDEMPOTENT ON THE EVENT, so the line states a phase once however many paths reach
    this call. Written under ``_LOCK`` like every other line this module appends — the
    sampler is a separate daemon thread reading the same ``_Armed`` — and it never
    raises: a diagnostic that cannot write its own line must not fail a runtime's
    shutdown, which is the last moment anything could go wrong for the operator.
    """
    with _LOCK:
        armed = _ARMED
        if armed is None:
            return False
        if armed.runner_teardown.is_set():
            return False
        armed.runner_teardown.set()
        return _append_dump_line(armed, f"{TEARDOWN_MARKER}{TEARDOWN_NOTE}\n")


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


def dump_evidence(pid: int | None = None, directory: Path | None = None) -> tuple[Path, str] | None:
    """``(this pid's evidence dump, its text)``, or ``None`` — ONE search, ONE read.

    THE ONE PLACE THE FILE IS CHOSEN, for every reader in this module and for the
    callers that judge which LIFE a dump belongs to. ``fired_dump`` asks it for the
    path, ``_evidence_text`` for the text, and ``journal._stall_bound_evidence`` for
    both plus the arm epoch its header carries — a caller that read the file a second
    time for that epoch would be two reads that can disagree about which store they
    are describing, which is the class of error the pair-of-facts functions in this
    module already exist to avoid.

    The choice itself: the first candidate that CARRIES THE FIRE — a file the pid's
    dump was written into but never fired in is not evidence about this bound, and
    preferring it would let a header-only file left by a SIGKILL mask a fire in the
    other store — else the first candidate that EXISTS, because "no dump" and "a dump
    with no fire" are different facts (a hard kill writes the file and nothing else)
    and the mtime fence has to be able to see the second one. Deterministic in
    :func:`dump_candidates`' order, so two readers cannot disagree about which life
    they are describing.
    """
    first_existing: tuple[Path, str] | None = None
    for candidate in dump_candidates(pid, directory):
        text = _dump_text(candidate)
        if _fires(text):
            return candidate, text
        if first_existing is None and candidate.exists():
            first_existing = (candidate, text)
    return first_existing


def fired_dump(pid: int | None = None, directory: Path | None = None) -> Path | None:
    """THE dump that fired for this pid, wherever it was written, or ``None``.

    The file a reader must REPORT, not a path it composes: a surface that published
    ``dump_path(pid)`` handed a caller this process's own log directory, which for
    exactly the rows this search makes visible is a file that does not exist
    (review round 1, MAJOR-1 / QA Q-1 — ``info.collect``'s ``stall_dump``).
    """
    evidence = dump_evidence(pid, directory)
    return evidence[0] if evidence is not None else None


def _evidence_text(pid: int | None, directory: Path | None) -> str:
    """The text of the dump that is this pid's evidence, or ``""``.

    ONE CHOICE OF FILE FOR THE FOUR READERS, because they are four questions about
    the same artifact — which leg fired, which tickers died, which loops were seen
    running, whether the fire ended the runtime — and two of them picking different
    files would be two chances for the pair to describe two different lives.
    """
    evidence = dump_evidence(pid, directory)
    return evidence[1] if evidence is not None else ""


def armed_at(text: str) -> float | None:
    """The epoch the dump's header names as its ARM, or ``None`` when unreadable.

    THE OTHER END OF THE FENCE, and the artifact's own answer to "which life is
    this": ``arm`` writes ``… armed for <bound>s at <epoch>`` as the header's first
    line, so a DUMP that names an epoch LATER than the turn whose death a reader is
    judging was written by a LATER process that drew the same pid — the recycling
    case the mtime check cannot see, because a successor's dump is newer than an
    earlier life's row by construction (review round 1, MAJOR-2).

    ``None`` for a header that cannot be read, and the caller decides what that
    means: this module reports evidence, never a verdict. Kept free of ``re`` — the
    module is on the child's boot path and its imports are held minimal — by
    partitioning on the two literals :func:`arm` writes them with.
    """
    for line in text.splitlines():
        if not line.startswith(ARM_MARKER) or " armed for " not in line:
            continue
        _, _, rest = line.partition(" armed for ")
        _, found, rest = rest.partition("s at ")
        if not found:
            return None
        try:
            return float(rest.split(" ", 1)[0])
        except ValueError:
            return None
    return None


def dump_is_current(path: Path | None, started_at: float) -> bool:
    """Whether ``path`` can belong to a life that was already running at ``started_at``.

    ONE SPELLING OF THE LIVENESS FENCE, for the two publish sites in
    ``info.collect`` and the narration in ``journal``: a dump written BEFORE this
    life began belongs to a predecessor that drew the same pid, and publishing it as
    this row's stall evidence is how a live, healthy session came to be painted
    ``bound held; lop stop`` off a recycled pid's leftover artifact (QA round 1,
    Q-2). ``False`` for no path, and for a stat that raises: an unreadable artifact
    is not evidence of a fire, the same quiet direction :func:`_dump_text` takes.
    """
    if path is None:
        return False
    try:
        return path.stat().st_mtime >= started_at
    except OSError:
        return False


def held_fire(pid: int | None = None, directory: Path | None = None) -> bool:
    """Is this pid's LIVE state "stalled with work in flight"? The third state.

    A fired dump has two readings for a listing, and a reader that cannot tell them
    apart gets the operator's question exactly backwards: "this runtime hit its bound
    and is gone" wants a post-mortem and a successor, while "this runtime hit its
    bound with work in flight and is still running, stalled" wants the person who owns
    that work or a ``lop stop``. `Timeout (' is written by both — it says the C
    timer expired and nothing about what the fire was over — so the answer
    comes off :data:`HELD_MARKER`, which only a fire over WORK can write, because
    writing it takes a live Python thread after the fire AND a busy probe that said so.

    IT IS THE HELD READING ONLY, and :data:`OBSERVED_MARKER` must never satisfy it: a
    dump-only fire on an idle (or recovered, or GIL-held) runtime is an observation,
    and answering ``True`` for it put a permanent false "stalled" cell on a process
    with nothing in flight (finding B of the 2026-09-23 convergence round). Whether a
    fire ENDED the runtime is a different question with its own reader,
    :func:`fire_outcome`, because on a dump-only build the answer is no for every fire.

    PRODUCTION ASKS THIS PER-PID QUESTION IN BULK, THROUGH :func:`held_pids` — named so
    the answer is not read as dead weight (agent review round 3, NIT-2). The listings
    behind ``/info``, the TUI panel and ``lop sessions --json`` scan every live dump for
    the held reading, so the readers a person sees go through ``held_pids`` while the
    verdicts go through :func:`fire_outcome`; this function is the one-pid spelling of
    exactly that reading, used by the cells and by any future caller asking about a
    single pid.

    ``False`` for a pid with no fire at all and for anything unreadable: production
    watchdog fires are never fatal, and the quiet direction is the
    one that cannot narrate a runtime as stalled when it never fired.
    """
    evidence = dump_evidence(pid, directory)
    if evidence is None:
        return False
    return _holds(*evidence)


#: The phrase both post-fire markers carry before the moment they were written, and the
#: local-time format of that moment. Shared by the writer (:func:`_record_held_fire`) and
#: the reader (:func:`_held_fired_mono`), so the two cannot drift apart.
_FIRED_AT_PHRASE = "the bound fired at "
_FIRED_AT_FORMAT = "%Y-%m-%d %H:%M:%S"
#: The tag the held marker writes its observation's ``time.monotonic()`` behind, in the
#: parenthesis after the local time: ``… fired at <local> (monotonic 1234.567) and …``.
_MONOTONIC_TAG = "monotonic "


def _held_fired_mono(text: str) -> float | None:
    """The ``time.monotonic()`` the LAST :data:`HELD_MARKER` was written at, or ``None``.

    The LAST one, because a runtime that stays stalled re-fires on the held backoff and
    every repeat appends its own marker: the question a listing asks is whether the
    runtime has recovered since the MOST RECENT fire, not since the first.

    THE MONOTONIC STAMP AND NEVER THE LOCAL TIME, for the reasons the writer states:
    the local time has no zone and no DST flag, so turning it back into an instant is
    wrong by an hour in the repeated fall-back hour and by the zone offset for a reader
    east of the writer, and both errors read a stuck runtime as recovered (QA round 1,
    Q-1). ``None`` for a marker without the stamp — every dump written before it existed,
    or a torn line — which the caller reads as "cannot show a recovery" and so keeps the
    held reading the marker states.

    Searched within the marker's OWN line only: the marker can land mid-line when
    ``faulthandler``'s buffered write interleaves (Q-4), and a stamp found further down
    the file would belong to some other line.
    """
    at = text.rfind(HELD_MARKER)
    if at < 0:
        return None
    line = text[at + len(HELD_MARKER) :].split("\n", 1)[0]
    if not line.startswith(_FIRED_AT_PHRASE):
        return None
    _, found, rest = line.partition(f"({_MONOTONIC_TAG}")
    if not found:
        return None
    try:
        return float(rest.split(")", 1)[0])
    except ValueError:
        return None


def _rearmed_since(dump: Path, fired_mono: float) -> bool:
    """Has EVERY plane reported since the fire, per this dump's deadline sibling?

    THE SIBLING'S CONTENT, NOT ITS MTIME, and the difference is the whole predicate. The
    mtime moves on every successful re-arm, and a runtime whose WORKLOAD loop is parked
    keeps re-arming off its healthy SERVING plane every 15 s — measured on this fleet:
    pid 65820's dump reads "workload last reported 4932.88s ago … the serving plane
    reported 0s ago" — so a fresh mtime is exactly what a genuinely stuck runtime shows.
    The content is :meth:`_Armed.pin`: the EARLIEST plane's last sign of life plus the
    bound (or the progress leg's decided deadline). It therefore moves past the fire
    only when EVERY plane has shown life after it, which is the definition of a
    recovered runtime; while any plane is still silent it stays at or before the fire,
    because that plane's deadline is what fired.

    ONE CLOCK: the sibling's THIRD field (the pinned deadline in ``time.monotonic()``)
    against the held marker's own monotonic stamp. Both are written by the runtime that
    owns the pid, from the one clock ``pin`` computes on, so a host sleep, a wall-clock
    step, the DST fall-back hour and the reader's zone cannot move one side without the
    other (agent review round 1 M1/m1, QA round 1 Q-1).

    THE MARGIN IS HALF THE SMALLEST BOUND (``min_bound_seconds() / 2``, 22.5 s), sized
    against both sides of the comparison rather than tuned: a stuck runtime's pinned
    deadline is at or BEFORE the fire's observation (that plane's deadline is what
    fired), while a recovered one is at least a whole bound (>= 45 s) past it. Half the
    floor separates the two with room on either side.

    THE PROGRESS LEG STAYS HELD HERE, stated because it is the conservative direction:
    its decided deadline is the fire's own moment and nothing in the watchdog clears
    it, so a sibling naming ``progress`` never reads as recovered. That is the reading
    the markers themselves keep giving, since that leg keeps re-firing.

    Unreadable, absent, malformed or OLD-FORMAT (two-field) siblings answer ``False`` —
    the evidence of a recovery is missing, so the held reading the dump states stands.
    """
    try:
        raw = dump.with_suffix(DEADLINE_SUFFIX).read_text(encoding="utf-8").split()
        deadline_mono = float(raw[2])
    except (OSError, ValueError, IndexError):
        return False
    return deadline_mono - fired_mono >= min_bound_seconds() / 2


def _holds(dump: Path, text: str) -> bool:
    """THE ONE HELD PREDICATE: fired, held over work, and not superseded since.

    Every surface that renders the third state reads it through here —
    :func:`held_fire`, :func:`held_dumps`/:func:`held_pids` (and so ``info.collect``'s
    ``stall_held``, the ``/info`` panel's ``bound held`` and ``lop sessions``'
    ``bound held; lop stop`` cell) and the kill ladder's drain check — so no two of them
    can disagree about one dump.

    SUPERSESSION IS WHAT THIS ADDS, and the defect it closes was a sticky flag: the
    marker is appended once per fire and nothing ever withdrew it, so a runtime that
    recovered minutes after a fire read ``bound held; lop stop`` for the rest of its
    life (pids 65820 and 83160 on 2026-09-24, 5 and 14 hours after their fires, while
    heartbeating every 15 s — an operator nearly stopped both). Two later facts end it:

    * a later :data:`OBSERVED_MARKER` — the most recent fire found NO work in flight, so
      whatever the earlier fire held has cleared;
    * the deadline sibling showing every plane re-armed after the last held fire
      (:func:`_rearmed_since`).

    A runtime that is genuinely stuck has neither, and keeps the held reading.
    """
    # A SUBSTRING TEST, NOT A LINE-START ONE (QA round 2, Q-4). This marker is the one
    # line of ours written into a file ANOTHER WRITER IS STILL FLUSHING: ``faulthandler``
    # writes its dump from its own thread with a buffered handle, we append this from
    # Python on the same ``O_APPEND`` fd, and the two can interleave so the marker lands
    # MID-LINE (measured on this fleet: one of four real fires, in the shape
    # ``  File "[stall watchdog] bound held: …``). A line-start test then reads a genuine
    # held dump as one that was NOT held, which through the product's own readers means
    # ``held_fire=False``, ``held_pids()`` empty, no STALLED cell — and ``death_verdict``
    # narrating a STILL-ALIVE runtime as "the runtime ended ITSELF". The module's own
    # header states the rule this restores: readers test these markers as substrings,
    # which is exactly why no header quotes one. ``rfind`` keeps that rule for the
    # ordering test too: an interleaved marker is still found, wherever it landed.
    if not _fires(text) or HELD_MARKER not in text:
        return False
    if text.rfind(OBSERVED_MARKER) > text.rfind(HELD_MARKER):
        return False
    fired_mono = _held_fired_mono(text)
    return fired_mono is None or not _rearmed_since(dump, fired_mono)


def held_now(pid: int, started_at: float, directory: Path | None = None) -> bool:
    """Is THIS life of ``pid`` held right now? :func:`held_fire` behind the pid-reuse fence.

    The per-pid spelling of what ``info.collect`` does for a whole listing (the held
    scan plus :func:`dump_is_current`), for a caller holding one record — the kill
    ladder, deciding whether a draining runtime can still reach the turn boundary it
    is waiting for. One spelling, so the ladder and the listing cannot disagree.
    """
    evidence = dump_evidence(pid, directory)
    if evidence is None or not dump_is_current(evidence[0], started_at):
        return False
    return _holds(*evidence)


def fire_outcome(pid: int | None = None, directory: Path | None = None) -> str:
    """How this pid's fire ended the runtime, as far as the ARTIFACT proves it.

    THE POSITIVE READING THAT REPLACES AN INFERENCE (finding A of the 2026-09-23
    convergence round). Attribution used to reason from what a dump did NOT contain —
    "it fired and carries no post-fire marker, therefore the bound ended this runtime" —
    which was true only of the legacy (fatal) builds and is FALSE of every build since
    the timer went dump-only: a fire held by the GIL cannot append a Python marker at
    all, and any later stop or crash is a different cause with its own rung. So the
    answer is now read off what the header STATES (:data:`DUMP_ONLY_POLICY_PHRASES`), and an
    artifact that proves neither convention answers :data:`FIRE_UNKNOWN` rather than
    being rounded to a death.

    * :data:`FIRE_SURVIVED` — the build was dump-only, OR the fire was a legacy HELD
      one (on a fatal build the held leg is precisely the one that did not exit, so the
      marker is proof by itself). Checked BEFORE the legacy phrases, because a legacy
      held artifact carries both and the CHECK order is what keeps its meaning;
    * :data:`FIRE_LEGACY_FATAL` — the header states the legacy convention and no held
      marker is present: an idle fire under such a build took the native exit, which is
      the reading those artifacts were written to support;
    * :data:`FIRE_UNKNOWN` — no fire (an armed header a SIGKILL left, an unreadable
      file, no file), or a header that proves neither convention. ABSENCE IS NOT PROOF:
      the quiet answer here is what keeps a still-alive runtime from being narrated as
      a watchdog death, which is the whole defect this closes.

    Substring tests, like every other marker reader here, and for the same reason (see
    :func:`held_fire`): the two writers share one ``O_APPEND`` descriptor, so a marker
    can land mid-line.
    """
    text = _evidence_text(pid, directory)
    if not _fires(text):
        return FIRE_UNKNOWN
    if any(phrase in text for phrase in DUMP_ONLY_POLICY_PHRASES):
        return FIRE_SURVIVED
    if HELD_MARKER in text:
        return FIRE_SURVIVED
    if any(phrase in text for phrase in LEGACY_FATAL_POLICY_PHRASES):
        return FIRE_LEGACY_FATAL
    return FIRE_UNKNOWN


def fired_in_runner_teardown(pid: int | None = None, directory: Path | None = None) -> bool | None:
    """Was this dump's fire taken while the asyncio runner was tearing down?

    ``True`` only when the artifact PROVES the ordering: the :data:`TEARDOWN_MARKER`
    line stands above the fire's own ``Timeout (`` line. Everything else answers
    ``None`` — the marker is absent, the marker is present but not above that fire, or
    the text is there mid-line rather than on a line of its own (the two writers share
    one buffered ``O_APPEND`` descriptor, so an interleave is a real state; see
    :func:`held_fire`). It is ``None`` rather than ``False`` because the class is
    ENTERED by proof and never by inference: an unmarked artifact cannot say whether
    the runtime was still inside ``amain`` (the annotation had not run) or reached the
    teardown by a path that could not write (a GIL-held stall never returns from
    ``amain``, so it never gets the marker at all — the case the annotation must NOT
    claim).

    THE ORDERING IS THE WHOLE MECHANISM, so nothing here looks at mtimes or at which
    phase it thinks a stack is in. The dump's fire is teardown-attributed exactly when
    the annotation stands above it; where the annotation is absent, or the artifact
    cannot place it above that fire, the answer is unknown rather than inferred.

    THE ANNOTATION ATTRIBUTES ANY FIRE IT STANDS ABOVE, AND ONE DUMP CAN HOLD SEVERAL
    (agent review round 3, MINOR-1). A dump-only build does not end on a fire, so a
    runtime can take one, carry on, return from ``amain`` and then be annotated before
    the runner parks in its executor join — the artifact then reads fire, annotation,
    fire, and the fire the annotation belongs to is the one BELOW it. Reading the FIRST
    fire's index answers ``None`` for that shape: the annotation is at or below it, so
    the one artifact that proves this class cannot claim it. The predicate is therefore
    "does any fire stand below the annotation", which is the LAST fire's index — a
    fire taken after the runner began tearing down can only be a teardown fire, since
    ``amain`` has already returned. Every single-fire row answers exactly as it did.
    """
    text = _evidence_text(pid, directory)
    if not _fires(text):
        return None
    lines = text.splitlines()
    fire_at = max(index for index, line in enumerate(lines) if line.startswith(FIRED_MARKER))
    marker_at = next(
        (index for index, line in enumerate(lines) if line.startswith(TEARDOWN_MARKER)),
        None,
    )
    if marker_at is None:
        return None
    # STRICTLY ABOVE, and a marker at or below EVERY fire is NOT the opposite answer:
    # the annotation lands after an earlier fire's line on the path where the coroutine
    # returns and the runner then joins a parked executor worker — which is exactly the
    # fire this exists to attribute, so a reader that read "after" as "not teardown"
    # would deny the class it was built for. Unknown, never inferred.
    return True if marker_at < fire_at else None


def _scan_marked(directory: Path | None, *, held_only: bool) -> dict[int, Path]:
    """One pass over every candidate store for the dumps whose text this accepts.

    :func:`fired_dumps` (every fire) and :func:`held_dumps` (the fires that did NOT
    end the runtime) differ only in that test, and the store search made them two
    copies of one loop the moment the path became part of the answer. ONE SPELLING
    keeps the pair from drifting: the two sets are nested by definition, and a reader
    comparing them — "which of the fired ones is still alive" — is comparing exactly
    this scan's two spellings.


    FIRST STORE WINS for a pid that fired in more than one, mirroring
    :func:`dump_evidence`, so a listing and a narration cannot point at two different
    files for one pid. Unreadable entries are skipped rather than raising: a
    diagnostic must never take the listing down.
    """
    from local_operator.paths import log_dirs

    # EVERY CANDIDATE STORE, not just this process's: the daemon that renders a
    # listing does not carry the override the runtimes that wrote these files do, so
    # a single-directory scan answers "nothing fired" for a fleet of fired dumps (see
    # ``dump_candidates``). An explicit ``directory`` still means exactly one store.
    bases = (directory,) if directory is not None else log_dirs()
    found: dict[int, Path] = {}
    for base in bases:
        try:
            candidates = sorted(base.glob(f"{DUMP_PREFIX}-*.log"))
        except OSError:
            continue
        for path in candidates:
            text = _dump_text(path)
            if not _fires(text):
                continue
            if held_only and not _holds(path, text):
                # the one held predicate, so a superseded fire leaves the set too
                continue
            suffix = path.name[len(DUMP_PREFIX) + 1 : -len(".log")]
            if suffix.isdigit():
                found.setdefault(int(suffix), path)
    return found


def held_dumps(directory: Path | None = None) -> dict[int, Path]:
    """The pids whose bound FIRED AND DID NOT END THEM, mapped to their dump.

    ``fired_dumps``' companion on every listing that shows both: a fired dump whose
    ``HELD_MARKER`` is present says the runtime was STALLED with work in flight, and
    that wants a person (``lop stop``), while one without it says the runtime is gone
    and wants a successor. The two sets are nested — held is a subset of fired — and
    the reader needs both because the useful question is "which of the fired ones is
    still alive", which is exactly what a listing of live rows is.

    THE PATH IS PART OF THE ANSWER, for :func:`fired_dumps`' reason, and it is what a
    publish site needs for the liveness fence: a held dump's mtime is whether it
    belongs to the life in front of the reader at all (QA round 1, Q-2).
    """
    return _scan_marked(directory, held_only=True)


def held_pids(directory: Path | None = None) -> set[int]:
    """The pids of :func:`held_dumps`, as a set — the third state's membership test.

    The set is this function's whole answer, so a caller that already holds the map,
    or that needs the artifact, takes :func:`held_dumps` instead of re-scanning.

    THE HELD READING ONLY, and :data:`OBSERVED_MARKER` must never satisfy it: a
    dump-only fire on an idle (or recovered, or GIL-held) runtime is an observation,
    and answering membership for it put a permanent false "stalled" cell on a process
    with nothing in flight (finding B of the 2026-09-23 convergence round). A fired
    dump outside this set says either that the runtime is gone (a LEGACY fatal-policy
    artifact, see :func:`fire_outcome`) or that the fire was survived with nothing in
    flight — and neither of those is the held state, which is why the two readings are
    separate functions rather than one predicate.

    IT SCANS EVERY CANDIDATE STORE, through :func:`held_dumps`/:func:`_scan_marked`,
    and needs the artifact for that reason as much as this one: the marker has to be
    read out of each candidate file, and a listing that rendered a whole fleet from a
    single directory answered "nothing held" for dumps written under another
    ``LOCAL_OPERATOR_CONFIG_DIR`` (see :func:`dump_candidates`).
    """
    return set(held_dumps(directory))


def fired_leg(pid: int | None = None, directory: Path | None = None) -> str | None:
    """WHICH leg fired for this pid, or ``None`` when no bound did.

    The class a fired bound is recorded under is one token for both legs (see
    ``incidents.STALL_BOUND_CAUSE``), so this is the only thing that tells a
    reader whether a runtime went SILENT or SPUN WITHOUT ADVANCING — the two
    want different investigations, and the dump already distinguishes them: the
    progress leg writes its own line at the moment it decides, and the silence
    leg cannot, because the C thread that fires it runs no Python.

    Read from the artifact rather than from a record: ``faulthandler`` writes the
    dump from its own thread without the GIL, while production configures it with
    ``exit=False``. The marker proves a timer fire, not runtime death;
    ``journal.death_verdict`` consults the record only when the runtime is actually gone.
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
    text = _evidence_text(pid, directory)
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
    fired and which predicate it fired on; this says whether
    the instrument that feeds that predicate was still alive when it fired — a
    ``silence`` verdict beside a non-empty tuple here means the workload plane
    did not go silent, its REPORTER did, and those two want different
    investigations (one is a parked loop, the other a defect in this module's
    own supervision).

    Empty when the file is missing, unreadable, or carries no such line — the
    quiet direction, as in :func:`_dump_text`.
    """
    text = _evidence_text(pid, directory)
    return tuple(
        line[len(TICK_DEATH_MARKER) :].split(":", 1)[0].strip()
        for line in text.splitlines()
        if line.startswith(TICK_DEATH_MARKER)
    )


def executing_planes(pid: int | None = None, directory: Path | None = None) -> tuple[str, ...]:
    """Which planes' loops this artifact recorded as EXECUTING, oldest first.

    The reader for :data:`EXECUTING_MARKER`, and it answers the one question a fired
    dump raises about this module's own fairness fix: *did the leg know this loop was
    running?* A ``silence`` verdict beside a plane named here is a bound that fired on
    a loop its own sampler had already observed executing — which is the state
    :func:`_extend_for_execution` writes the line for, so finding one means the
    abstention did not carry (an unevaluable probe, a missing plane entry, or a
    deadline that expired between two moving samples). The recovery for those is not
    the same, and without this reader they are indistinguishable in the file.

    Empty when the file is missing, unreadable, or carries no such line — the same
    quiet direction as :func:`tick_deaths`.
    """
    text = _evidence_text(pid, directory)
    return tuple(
        line[len(EXECUTING_MARKER) :].split(" ", 1)[0].strip()
        for line in text.splitlines()
        if line.startswith(EXECUTING_MARKER)
    )


def fired_dumps(directory: Path | None = None) -> dict[int, Path]:
    """The pids whose bound actually FIRED, mapped to the dump that fired.

    The reader for the evidence this module writes, and it has a real consumer rather
    than being a helper kept warm by its own tests: ``lop sessions --json`` publishes
    the path per row (``info.collect.session_rows``, ``stall_dump``), which is the
    surface an operator or an agent lists a fleet on after something died.

    THE PATH IS THE FILE THE SEARCH FOUND, and that is the whole reason this returns a
    map rather than a set. A publish site that answered ``stall_dump`` with
    ``dump_path(pid)`` composed a path out of ITS OWN log directory, so for exactly the
    cross-store rows the search makes visible it handed out a path that does not exist
    (review round 1, MAJOR-1 / QA round 1, Q-1: ``fired_pids()`` contained the pid
    while ``dump_path(pid).exists()`` was False). Which store a dump is in is a fact
    only the search knows, so the search is what answers.

    ``FIRED_MARKER`` at the start of a line is the test, exactly as
    ``tests/e2e/watchdog.py`` defines it, because surviving the clean exit is not
    enough to call a file a freeze report: a runtime killed without disarming leaves a
    header-only file, and that is a hard death rather than this bound (see "THE FILE IS
    THE EVIDENCE" in the module docstring).

    ONE scan for a whole listing, which is why the publish sites take this map rather
    than asking per row: the marker has to be read out of each candidate, so a per-row
    call would re-glob and re-read every store once per session on a surface that
    renders every row.
    """
    return _scan_marked(directory, held_only=False)


def fired_pids(directory: Path | None = None) -> set[int]:
    """The pids of :func:`fired_dumps`, as a set — membership without the artifact.

    A caller that needs WHICH FILE fired — every publish site does — takes
    :func:`fired_dumps`: the path is a fact only the search holds, and asking
    ``dump_path(pid)`` for it answers from this process's own log directory instead.
    """
    return set(fired_dumps(directory))
