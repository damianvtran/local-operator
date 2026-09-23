# Bounding a runtime's own stall

Status: implemented (the instrument). The bound on the scan's cost is a separate
change — see "PR-B" at the end, which this note exists to inform.

## The failure, measured

On 2026-09-20 five session runtimes on the operator's machine were found frozen
at once, for **1.5 to 7.2 hours** each, and a sixth (a manager session) had been
frozen 6.9 h before it was reaped by hand. On every one of them:

- `sample <pid> 3` put **100% of the event-loop main thread** inside the
  credential-shape pass's matcher — `_sre_SRE_Pattern_search` →
  `sre_search` → `sre_ucs1_match` on one round, `sre_ucs2_*` on another (so the
  string being searched carries non-ASCII);
- heartbeats were stale by hours against a 45 s timeout, while CPU kept
  advancing at ~0.9 core, and the transcript took **zero writes**;
- `lop sessions reclaim` refused all five (`0 would reclaim (refused
  [record-present 16])`), `session.cleanup.enabled` is off, and `lop stop`
  refuses a silent-socket runtime without `--force` (which waits ~2.5 min and
  then SIGKILLs).

The linear cost model does not explain hours: the repo's own `scrub_secrets`
over those exact transcripts costs 1.75-2.07 s for 1.16-1.45 MB. The scan was
not converging, and **nothing in the process could say which line it was on**,
because the one instrument that would have answered (`LOP_RUNTIME_DEBUG_STACKS`,
the SIGUSR1 asyncio dump) was not set on the launcher. That is the defect this
change fixes first: a wedged runtime now names itself, and then it leaves.

## No caller is named, and this note does not pretend otherwise

An earlier revision of this note named the trigger: session `e837562a4c28` froze
twice, both times ~30 s after start and both times right after `hub resume` of a
subagent, and `harness/comms.py::resume` replays the stopped child's stored
messages and tool results — so the note said a resumed child was re-scrubbing
32.6 MB of stored text. **Review measured that away, and the claim is withdrawn
rather than softened:**

- booting a real runtime over a 20 MB credential-shaped store costs **1.27 s of
  CPU against 0.97 s for a 1-row store** — a resume does not scrub the store;
- **six real runtime passes** (runtime → real OpenAI-compatible provider → real
  `bash` → real hook, stores of 0.5-48 MB, bounds of 20 s and 30 s) never came
  near the bound: `fired=False`, turn CPU up to **60.4 s**, `beat_lag_s` steady at
  **15.00 s** throughout.

So what the five frozen processes proved is exactly two things, and this change
addresses both: the loop was parked inside a C-level matcher
(`_sre_SRE_Pattern_search`), and **nothing in the process could name the line** —
the SIGUSR1 dump that would have was not switched on. The caller is unidentified,
and no claim here depends on identifying it. Bounding the cost of a scan that a
caller has not been shown to run is not this change's job; naming and bounding the
STALL is, and the instrument below is what will name the caller next time.

One measurement from those probes stands on its own merit, as a property of the
shape pass itself rather than of a caller: 0.86 MB of credential-shaped text with
6270 hits costs **1.97 s** (2.30 µs/byte) because
`redaction_shapes._credential_fragments_survive` full-text-scans per hit —
`hits × bytes` ≈ 5.4 G byte-scans for that one child. Wherever that pass runs, its
cost is superlinear in the input. PR-B's subject; not this PR's.

## The instrument

`local_operator/session/runtime/stall_watchdog.py`.

**Why a C timer.** A `threading.Thread` watchdog, an `asyncio` timer and a
signal handler all fail here: a hung thread never releases the GIL, a Python
signal handler only runs between bytecodes, and a loop parked in a C call never
schedules the coroutine that would report the sample. `faulthandler`'s timer
runs in a dedicated **C** thread, needs no GIL and no interpreter state, writes
every thread's stack to a file descriptor and then `_exit`s.

**What the bound measures: no progress on a plane, not slow work.** Each plane
carries its own last-seen stamp — the workload tick (`process._beat_stall_watchdog`)
and the serving plane's heartbeat (`RuntimeServer._heartbeat_loop`) — and whichever
tick runs next re-arms the single process-global C timer for the EARLIEST
remaining deadline: `min(stamp + bound) - now`. A turn that awaits for ten minutes
keeps beating; a loop parked in one C call does not. **A healthy plane therefore
cannot mask a silent one**: its own tick shortens the timer toward the other
plane's deadline instead of pushing it out, and with both planes silent no tick
runs at all, so the timer fires at the deadline the last tick set.

That was the first shape of this change's ONE real defect, and it is worth
recording because the fix is what makes the bound mean anything. The first
version treated "the process is making progress" as one fact — any tick re-armed
the full bound — which cannot fire on the failure this PR exists for: a
`daemon`/`exec` runtime serves on its own thread, so the plane that stalls is
usually not the whole process, and the serving plane's healthy heartbeat kept the
timer fresh for hours. Measured on the committed head (`15ec2c63`) with a
one-second bound: a rig with the workload plane 100% busy in the matcher and the
serving plane beating every 0.2 s ran **8 s — 8x the bound — and was killed by an
external timeout, with ZERO fired markers and a header-only dump**. After the fix
the same rig leaves at the bound (`rc=1`) with every thread's stack.

**What it is stricter than, stated plainly.** A synchronous step is
indistinguishable from a wedge from outside the process, so the bound is also a
ceiling on ONE SILENT SYNCHRONOUS STEP: a step that takes tens of seconds passes,
one that never returns is cut. What the bound measures is the
loops RUNNING, never the work advancing — a plane that keeps ticking while its
work stands still is outside this design (nothing here can see that; inventing a
second, footprint-based clock is what `process._work_motion` already does for the
drain).

**The bound: 300 s.** Above the largest legitimate beat gap ever measured on this
host (205.8 s, on a session whose CPU time was advancing) and far below the
freezes actually suffered (1.5-7.2 h, four of five with zero writes). A single
synchronous step that holds the GIL for five unbroken minutes is not slow work.
`LOP_RUNTIME_STALL_SECONDS` overrides it; `0` disables it.

**The bound during BOOT: 900 s, and 300 s starts at engagement.** The steady bound
is measured from the last sign of life, so it can only be honest once something is
in a position to write one — and nothing in the boot path can. `arm` runs from the
`__main__` guard, before `main()`, and seeds both planes to that instant; the
workload beat's only driver starts after publication and sleeps a heartbeat before
its first stamp, and the serving beat starts with the serving thread. So from the
arm to the first post-publication beat, every second of a legitimate boot
(session construction, lease arbitration, MCP bring-up, the inbox drain, socket
bind, publication) was spent against a clock no boot code could re-arm, and the
only possible deadline was `arm + 300 s`. Measured over 68 retained dumps on the
build carrying the observation header: 17 fires, 10 of them at the full
`Timeout (0:05:00)` value, i.e. no beat ever re-armed the timer at all — the
never-engaged class, and the plurality. (An UPPER bound on that class, not an exact
count: a failed re-arm is indistinguishable from it by the armed value alone; see the
exception below.) `DEFAULT_BOOT_STALL_S`
(`LOP_RUNTIME_BOOT_STALL_SECONDS`, same spellings and floor/ceiling as the steady
knob, `off` meaning "no boot phase") is what `arm` arms with; the runtime's first
ENGAGEMENT — the publication boundary in `process.amain`, where `_live_handle` is
set — calls `stall_watchdog.engage()`, which moves the bound down to the steady one
and stamps BOTH planes, so the steady bound is measured from engagement rather than
from boot. That is also what makes the never-engaged class NAMEABLE: a fire that
engaged has the deadline sibling `engage` writes, and one that never engaged has
none. Engagement never widens the bound.

*Exception, measured (design review round 1, D1):* engagement is not atomic. `engage()`
moves the bound in memory and stamps both planes, THEN replaces the C timer, so a
replacement that FAILS leaves the original boot-bound timer in force while the bound
reads steady. A single fire can therefore carry the boot value on a runtime that DID
engage. The dump names this (`[stall watchdog] re-arm failed: the engage could not
re-arm the timer, ...`), which is why every statement of the never-engaged reading —
the boot note, the comment above it, and `HOW_TO_READ_THE_FIRED_VALUE` — is qualified
on the re-arm having SUCCEEDED rather than stated flatly.

`engage()` is also **not idempotent**: an unguarded second call after a successful
first one would compute a negative remainder from the already-reset stamp and fire
early. `engage()` returns early when the bound is already at steady, so only the first
call moves it.

**What that bound does with a hung boot: it DUMPS it and HOLDS it, it does not cut
it.** The exit leg answers through `process._busy_probe`, which reports work in
flight for the whole pre-publication window — `True` while `_live_handle` is `None`,
because a runtime still constructing itself is not idle in any sense that bound may
act on — so `arm` seeds `_Armed.held` true and every fire in this stretch is
non-fatal: the fire writes its dump and its `bound held:` marker, and the process
carries on. A boot that never reaches publication therefore keeps answering "in
flight" for the rest of its life, and nothing in the module ends it. **What does end
it, measured rather than assumed** (the first version of this paragraph named an
escape that cannot reach this class — agent review round 2, Q-3): the boot's **own
failure path** is the automatic exit (the construction error that ends the runtime
child: `rc 2` with the cause on stderr, 1.4 s here); **`lop stop` is not one**, because
it resolves its target through session records (`mobile.peer_send.resolve_peer_target`)
and a boot that never published has none — `lop stop --pid <pid>` answers `no session
found with pid <pid>` and the process goes on; and an operator ends it by signalling
the pid the dump is named for, which its header carries (`kill -TERM <pid>`, not gated
on any record). What the boot bound contributes in that window is the ATTRIBUTION — the
fired value is the boot bound and the deadline sibling is absent, together the
the never-engaged class — *provided the re-arm succeeded*; an engaged runtime whose
timer replacement failed can fire at the boot value too (D1) — while the exit leg only becomes fatal once the runtime has
published and its work has cleared (a property of the in-flight prohibition, #1439,
not of the boot phase). Pinned as a pair, because the two arming shapes answer
different questions:
`test_a_hung_boot_with_the_production_probes_is_dumped_and_HELD` (the entry point's
own arming, `busy=process._busy_probe`, held at `rc 0`) and
`test_a_never_engaging_boot_with_NO_work_in_flight_is_still_cut` (a caller with no
probe at all, armed fatally, `rc 1`).

**Still open in that window, and NOT closed here.** A tick that returns early
without raising is never re-created by `_watch_stall_beats`, whose restart fires
only on an exception (`_do_shutdown` is the same class: a plane is unreported
because the runtime is ending). Both are real — dump 24646 (19 threads, main idle
in `select`, no `tick died:` line) and 75019 (mid-shutdown) — and both need their
own fix (a restart on early return, and a shutdown state the bound recognises).

**What the exit costs, beyond the turn.** A hard exit runs no Python, so the
in-process kill of this turn's tool process groups cannot fire (`execute_bash`'s
`_kill` chain) — which is exactly the "hard death of the owning `lop` process"
class `tools/group_reaper.py` exists for. Each group is registered with a
liveness marker at spawn and `sweep_orphan_groups` reaps the ones whose owner is
provably dead at the **next `lop` startup**, so a child orphaned here is bounded
by the next session start rather than by this process's death — the same
guarantee today's only recovery (SIGKILL) already relies on.

**Why the exit is the blunt one.** Past the bound the process must stop being a
multi-hour freeze, and the graceful rungs cannot be reached from the state being
detected: `_drain_for_signal` and `_commit_to_leaving` are coroutines on the loop
that is blocked, and the `SIGTERM` handler that reaches them needs bytecodes the
stuck thread never executes — the same fact that makes `lop stop` refuse a
silent-socket runtime. The timer therefore dumps every thread from its own C
thread, which is structural rather than sequenced, and THEN decides what to do
with the process: `exit=not held` is re-decided at every re-arm from the same
work probe the reaper's WORK signal reads, so a runtime with nothing in flight
`_exit(1)`s exactly as before, while a runtime with a turn, a subagent or a job in
flight keeps the dump, records the held state (``stall_watchdog.HELD_MARKER``) and
stays ALIVE — stalled, marked, and ended by a person (`lop stop`, whose SIGKILL
rung is the only one that reaches a wedge; see `control.py`). Evidence is never
withheld either way: the bound still fires, still writes the dump and still records
its class; only `_exit` is refused.

**What is lost, what survives.** On the fatal arm (nothing in flight) the
in-flight turn's uncommitted step is lost — the same loss a SIGKILL inflicts,
because a turn commits its transcript at each step and the step in flight has not
committed. On the HELD arm nothing is lost and nothing is cut: the runtime keeps
serving on the build it loaded, the turn inside it is still running, and the row
plus the dump say so. Everything already committed survives on both arms, i.e. the
conversation, which a successor can be engaged on. The record is left behind with
a dead pid on the fatal arm, which `registry.classify` already reads as `stale`
and `reclaim` already sweeps: no new vocabulary, and `live`/`wedged`/`stale` is
untouched.

**The file's content is the evidence, not its existence.**
`<log dir>/runtime-stall-<pid>.log`, beside `runtime.log`. Only a file carrying
`faulthandler`'s own `Timeout (` line is a freeze report; a runtime's file is
header-only while it is armed, and a clean exit truncates that header and writes
`[stall watchdog] clean exit` over it. **Nothing is deleted**, and that is a
decision rather than an oversight: existence is the *weaker* signal (a SIGKILL
leaves exactly what an armed runtime has), and this repository's session-deletion
guard fails the build on an `unlink` whose call shape it cannot tell apart from a
"safe" reaper — twice a reaper like that deleted a real session's files. The cost
is one small self-describing file per runtime process, which nothing prunes.
`faulthandler` prints frames and never locals, so no credential can land in a
file written into a log directory.

**Who reads it.** The path is printed by the runtime's own log line at boot
(`stall_watchdog.announce`), and `lop sessions --json` carries a `stall_dump` key
per row — the path when that pid's bound fired, `null` otherwise
(`stall_watchdog.fired_pids`, read once per listing). That is the surface an
operator or an agent lists a fleet on after something died, and the runtime that
wrote the file is gone by definition, so the reader has to reach it from a
listing rather than from the process.

**The interlock.** `faulthandler`'s timer is process-global and shared with
`tests/e2e/watchdog.py` and `tests/shard_stall_watchdog.py`. The arming therefore
lives in `process.__main__` — reachable by `python -m` alone, i.e. by a real
runtime child — and never in a constructor or in `main()`, both of which are
called in-process by the suite. Arming a library path would let an in-process
boot (or a `start_in_process` host) displace a CI stage's only bound. Pinned by
`tests/unit/session/runtime/test_runtime_stall_watchdog.py`.

**Honest records while nothing is beating.** The heartbeat now publishes the gap
it measured and this process's CPU time over that same gap
(`SessionRecord.beat_lag_s`, `cpu_since_beat_s`, published in `lop sessions
--json`). `wedged` alone is one word for three situations: starved by its own
work (CPU advanced), starved by the host (CPU did not), not running (pid gone).
Both readings are in-process — no `ps`/`lsof` fork per tick.

## The second leg: spinning without advancing

**What the first release could not see, and the measurement that exposed it.** The
bound above measures the LOOPS RUNNING. On 2026-09-21 a session on build 0.61.16 —
with the bound armed — was measured by another session's probe with no progress in
its transcript, its roster or its four subagent counters across 75 s, **+14.3 s of
process CPU** burned in that window, and 231 s of heartbeat age, while both planes'
ticks kept re-arming the timer. The probe's own words: no build, no command, no work
in flight. A four-way subagent batch had been launched 16 s after a sibling settled,
the parent acknowledged the launch, and then produced nothing.

**The timing, because it settles what was at fault.** Its serving plane's last beat
was 18:34:58, so the 300 s liveness deadline was 18:39:58. The operator's reap landed
at **18:39:46 — twelve seconds short**. Both instruments were right, and neither was
broken: the bound saw a live loop, the probe saw a stalled session, and nothing in the
design could see "spinning without advancing".

**Why the fix is composite.** A bound keyed on movement alone would cut every
legitimate long step — a model call, a tool, a subprocess all produce no transcript
movement — which is the false positive the liveness design was chosen to avoid. Three
facts together separate WAITING from SPINNING, and all three must hold for a whole
window:

1. **no motion** — `process._work_motion`, the drain's clock, REUSED rather than
equalled by a second footprint clock;
2. **nothing in flight** — no tool batch executing (the live context does not end in
   an assistant message whose tool calls have no answers, the state
   `Session._wire_legal_snapshot` documents, plus the `_compacting` flag);
3. **CPU advancing** — at least `PROGRESS_CPU_FLOOR` (5%) of one core as a MEAN
   OVER THE TRAILING WINDOW. Not a per-sample reading: agent review round 1 measured
   that a single scheduled-out sample discarded a run, leaving an effective margin of
   0.4-1.2x against a documented 3.8x, and one rigged run in fifteen never fired. And
   not a cumulative mean either: round 2 measured that one baseline held from the run's
   start decays as ~1/t through silence, so a burst is carried and the detection
   latency depends on how long the session has been alive. The mean is taken over the
   samples pruned to the window, so the latency follows the BURN. A model call is a
   socket read; a bash child's CPU belongs to the child and never reaches
   `time.process_time`.

**The window is the bound**, and the argument that sized 300 s sizes this one: it sits
above the largest legitimate silence measured on this fleet (205.8 s, 1.5x), and the
measured false positive for calling a runtime `wedged` at 45 s — 105.8 s and 205.8 s of
beat gap on sessions whose CPU was advancing — is a starved scheduler, i.e. a session
DOING work: such a sample fails leg 2 or leg 1.

**The leg is live in production, and that is proven on a spawned child.** The PR's
first version proved the predicate behaviourally and the wiring only textually —
three text-preserving mutants (the publication inside `if False:`, an early
`return (), True` before the probe reads the handle, a probe-less arm site in another
module) left that pin green with the leg inert. The acceptance evidence is now a real
`python -m …process` child spawned through `launch._spawn_runtime`, spinning because a
`PYTHONPATH`-supplied `sitecustomize.py` starts a CPU-burning thread at interpreter
start (no model, no turn), with `LOP_RUNTIME_STALL_SECONDS=45`: it exits `rc=1` with
the progress line in its dump and `fired_leg → "progress"`. The same rig with `probe=`
dropped inside the child does not fire. A predicate that ships silently disabled
with green tests would make the fleet *look* protected, which is the failure this
whole design exists to avoid.

**The firing path names its class.** `faulthandler` reaches its timer from a C thread
and calls `_exit(1)` there, so no exit hook, no journal row and no reaper runs after a
fire — the dump is the only place a class can be written, which is why the header and
the progress line carry `incidents.STALL_BOUND_CAUSE` and `fired_leg()` reads the leg
back out of the file. Without that token the loudest ending in the fleet — a runtime
that dumped every thread and killed itself — was narrated as `unattributed`, i.e. "no
act was recorded", which is the one reading that is worse than not knowing.

## Not in this change

**Reclaim cannot help here.** `reclaim`'s rung 5 refuses any candidate with a
record (`record-present`), and it is right to: a stale beat covers a frozen
process *and* a healthy one starved by a long turn. Widening it to admit a
loop-blocked runtime means teaching that ladder a distinction the fleet
vocabulary does not carry yet, and the ladder is what stops a kill switch
killing the wrong process. It gets its own round; until then the only working
recovery for this class is a hand-run `lop stop --pid <pid> --force`.

**The TUI's own registrant is out of scope.** `tui/app.py` hosts a
`RuntimeServer(kind="tui")` in the user's terminal process, so a bound there has
to decide what happens to that terminal — a different design question, and
arming it from a library path is exactly the interlock hazard above. The
detached `-m` child is the shape covered: 16 of 16 records in the operator's own
store were `kind=daemon` when this was written.

**A subagent lane's own in-process tool is not covered by the progress leg.**
The in-flight fact reads the PROCESS's own step and its compaction. A lane running
a long in-process tool of its own is not in flight by that measure, so a lane
parked in private CPU work with no step boundary for longer than the window is
cut with its parent. A lane that is STEPPING is covered from the other end (its
step boundaries move the roster generation, so leg 1 fails); closing the gap
needs a per-child probe, and `comms._records` holds those sessions privately
today. Widening the probe to every child session gets its own round.

## PR-B: bounding the scan itself

Candidates, in the order this note argues for them.

1. **Skip the re-scan of stored text (the principled fix, IF the pass runs there
   at all).** Review has already shown that a resume does not scrub, so this
   starts from an unverified premise: the pass must first be shown to run on
   stored text at all (a profile, not an argument), and only then is a cache or a
   skip worth designing.
   `session/transcript.py` contains no redaction at all: what reaches a
   transcript is masked upstream, on the tool-result path. If the stored bytes
   are already masked, then re-scrubbing them on replay is pure waste, and the
   right repair is to **skip or cache** rather than only to bound bytes. The
   caveat is the whole difficulty, and it must be discharged before this ships:
   a skip that is wrong is a credential leak, so it needs proof that the stored
   bytes are already masked (per surface that can write a transcript, including
   ones added later) **plus** a byte-equality corpus asserting the skip changes
   no output. Treat "already masked" as a claim to be tested, never as an
   assumption to be inherited.

2. **Attack the quadratic term.** `_only_fully_masked` → `_credential_fragments_survive`
   runs `value in text` over the whole text for every hit. A single pass that
   answers "which of these values survive in this text" for all hits does the
   same work in one traversal; the byte-equality corpus from (1) is what proves
   it equivalent.

3. **Bound the bytes** (`_PipeRedactor`/`scrub_shapes_with_hits` bounds, offload
   to `asyncio.to_thread`) — necessary as a backstop, not sufficient on its own:
   it stops the freeze without reducing the cost, and it trades a stall for a
   partial scan. A preserved, unfinished diff for this is on branch
   `fix/bounded-credential-shape-scan` (commit `f4ab88a4`, worktree
   `loop-scan-bound`); it was deliberately not extended here.

Any of these lands together with a regression test on a synthesized store of the
shape used above, because the cost model — not a stopwatch — is what has to
change.
