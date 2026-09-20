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

## The trigger, named and quantified

Session `e837562a4c28` ("Optimize time to first token latency") froze twice, both
times ~30 s after start and both times immediately after `hub resume` of a
subagent, at 73.5 s of CPU in a 75 s window. `harness/comms.py::resume`'s own
docstring says why that action is heavy: the new child reads the old one's
session directory, so it replays every message and tool result the stopped run
produced.

That session is a manager with **22 child transcripts, 32,621,531 bytes** in
total (its own transcript is another 1.69 MB). Scrubbing those child transcripts
in sequence costs **49.66 s of CPU** (~1.5 µs/byte); the largest single child is
6.83 s for 4,547,831 bytes.

Re-measured for this change on a synthesized store of comparable shape
(`tests/unit/session/runtime/test_runtime_stall_watchdog.py`), the cost is worse
than a flat per-byte rate, and the reason is visible in the stack the new dump
names:

- 0.86 MB of replayed text with **6270 hits** costs **1.97 s** (2.30 µs/byte),
  i.e. `hits × bytes` ≈ **5.4 G byte-scans for that one child**;
- the dump's innermost frame is
  `redaction_shapes.py:1963 in _credential_fragments_survive` — `if value in
  text` — reached from `_only_fully_masked(hits, text)` for **every hit**.

So the loop occupation is quadratic in storage: each credential-shaped hit
re-scans the whole replayed text. That is what turns a 50-second replay into a
scan that never finishes, and it is PR-B's target.

## The instrument

`local_operator/session/runtime/stall_watchdog.py`.

**Why a C timer.** A `threading.Thread` watchdog, an `asyncio` timer and a
signal handler all fail here: a hung thread never releases the GIL, a Python
signal handler only runs between bytecodes, and a loop parked in a C call never
schedules the coroutine that would report the sample. `faulthandler`'s timer
runs in a dedicated **C** thread, needs no GIL and no interpreter state, writes
every thread's stack to a file descriptor and then `_exit`s.

**What the bound measures: no progress, not slow work.** The timer is armed once
and **re-armed by the runtime's own loops as they run** — the serving plane's
heartbeat (`RuntimeServer._heartbeat_loop`) and the workload loop's own tick
(`process._beat_stall_watchdog`). A turn that awaits for ten minutes keeps
beating; a loop parked in one C call does not. Both loops beat deliberately: the
timer is process-global, so its honest reset is progress from *any* of the
process's own loops. The accepted residual is the mirror case — a stall confined
to one plane (say a serving plane blocked while a turn steps) does not trip this
bound, because the other loop is still proving the process is alive.

**The bound: 300 s.** Above the largest legitimate beat gap ever measured on this
host (205.8 s, on a session whose CPU time was advancing) and far below the
freezes actually suffered (1.5-7.2 h, four of five with zero writes). A single
synchronous step that holds the GIL for five unbroken minutes is not slow work.
`LOP_RUNTIME_STALL_SECONDS` overrides it; `0` disables it.

**Why the exit is the blunt one.** Past the bound the process must stop being a
multi-hour freeze, and the graceful rungs cannot be reached from the state being
detected: `_drain_for_signal`, `_commit_to_leaving` and `_leave_overdue` are
coroutines on the loop that is blocked, and the `SIGTERM` handler that reaches
them needs bytecodes the stuck thread never executes — the same fact that makes
`lop stop` refuse a silent-socket runtime. The timer is therefore armed with
`exit=True`, so faulthandler writes the dump and `_exit(1)`s from its own C
thread: write-then-act is structural rather than sequenced.

**What is lost, what survives.** The in-flight turn's uncommitted step is lost —
the same loss a SIGKILL inflicts, because a turn commits its transcript at each
step and the step in flight has not committed. Everything already committed
survives, i.e. the conversation, which a successor can be engaged on. The record
is left behind with a dead pid, which `registry.classify` already reads as
`stale` and `reclaim` already sweeps: no new vocabulary, and
`live`/`wedged`/`stale` is untouched.

**The file is the evidence.** `<log dir>/runtime-stall-<pid>.log`, beside
`runtime.log`. Its existence means the bound fired: a clean exit cancels the
timer and removes it, so a header-only file left by a SIGKILL is not read as a
freeze, and only a file carrying `faulthandler`'s own `Timeout (` line is.
`faulthandler` prints frames and never locals, so no credential can land in a
file written into a log directory.

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

## PR-B: bounding the scan itself

Candidates, in the order this note argues for them.

1. **Skip the re-scan of stored text (the principled fix).**
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
