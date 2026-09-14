# The TUI against N attached sessions — the instrument, and the before-numbers

`scripts/bench_tui_sessions.py`. Measured on an **unmodified `origin/main`**
checkout (`1296cda41`, `/tmp/lo-tui-baseline`) with the harness from this branch,
so every number below is a before-number: nothing in `local_operator/` was
changed to produce it.

## Why the instrument exists

The central claim under review — that the TUI's render cost grows with the number
of sessions it has attached — had no instrument behind it. The only TUI load
harness in the tree, `scripts/bench_tui_background.py`, varies `--children`
*within one session*, and `scripts/sidebar_performance.py`'s session-count axis
is catalog *rows*, not attached viewers. So the N-session figure in
`/tmp/tui-coupling-audit.md` §7 was a linear extrapolation of a single-session
constant, flagged there as UNVERIFIED. This is the harness that turns it into a
measurement, and the same script is the before/after instrument a fix will be
measured with.

## What it measures

`--sessions N` boots the real `OperatorApp` under `app.run_test`, attaches N
viewers, streams canonical deltas at the producer's own cadence, injects input
from a separate OS thread, and times **the compositor's paint boundary** —
Textual's `_display` receiving a `LayoutUpdate`/`ChopsUpdate` — never the
`pilot.press`/`pause` round trip and never the terminal's presentation latency.

The load model, in the production order:

| element | how it is modelled |
|---|---|
| each session | a real `AttachedSession` viewer with its own `FrontendStateStore`, seeded with `--children` running task jobs retaining `--rows` trajectory rows each |
| each delta | produced by the production *writer* — `FrontendStateStore.refresh_jobs` → `mutate` — so the frame carries `changes['jobs']` summaries plus trailing `job_trajectory_appends`, and an assertion in `ProducerRoster.tick` fails the run if a frame carried anything else |
| delivery | `AttachedSession._on_frontend_update`, the decoded-callback boundary the socket codec hands the viewer (`attach_client.py`), one delta per session every 50 ms — the producer's real coalescing cadence (`session/session.py`), i.e. 20 deltas/s per attached session |
| session 0 | also the app's current session, so `OperatorApp._on_frontend_update` → `_apply_frontend_state` (which reads the whole state) is live |
| sessions 1..N-1 | registered as leased sidebar sources (`SessionInteraction` + `_watch_source_frontend`), so the per-source fan-out and the retention predicate that clones the state for a boolean are live |
| animation | production shimmer is left ON; the measurement must include the work it studies |

Frames are pre-built before the window and recycled (sequence rewritten). The
owner's roster rebuild happens in *another process* in production; running it on
this loop would charge the viewer for the owner's work. The build cost is
reported separately (`pool_build_s`, 4-26 s per cell here).

The delta is 6,540 B against a 3,000-row retained roster per session
(`6 children × 500 rows`) — quoted so a reader can see the fixture did not move
when the numbers did. The audit's reference frame at the same roster shape is
17.8 KiB.

### What it does not measure

- **`job_todo_updates`, `job_trajectory_replacements` and descendant usage.** The
  fixture's children have no todos and the window never rotates past the cap, so
  those parts of the delta vocabulary ride empty. The audit measured the todos +
  descendant-usage shape at +1.1 ms/delta over the base.
- **The sidebar catalog poll and prewarm** (`/tmp/tui-coupling-audit.md` F5/F6).
  Those scale with store size, not with attached sessions.
- **A live socket.** The decode (`json.loads`) is not in the window; the ingest
  measured starts at the boundary the codec produces, which is the same start
  point `bench_tui_background.py` documents.

## Basis: host, and why wall is not the verdict

Load average on this 14-core host was **76.5 / 84.9 / 103.4** (1/5/15 min) at the
end of run A and **73.2 / 81.7 / 98.9** at the end of run B. The loop thread's
CPU is **8-17 % of the wall window** in every cell — i.e. the process is
descheduled for most of it — so wall percentiles here are observations, not
ceilings: on an idle box they are smaller and the achieved frame rate higher. Per
AGENTS.md ("If you must measure, measure CPU, not wall time") the CPU columns are
`time.thread_time()` on the loop thread, and they are the load-robust half; the
wall columns move up to ~2x between the two runs while the CPU columns agree to
within 5 %. Both are reported because the user-visible question is a wall one.

## Canaries

Every run asserts, before any cell is measured:

- a normal character **paints** (input → compositor frame recorded);
- a deliberate `time.sleep` on the loop **appears in wall latency** (measured
  169-194 ms against a 120 ms floor) while the CPU gap stays flat — the CPU clock
  is blind to a blocking sleep, and this is the assertion that keeps that
  blindness from silently becoming a result;
- an absent marker **stays undetected** (`absent_marker_detected: false`) — a
  probe that matched anything would report a paint for input the app never
  rendered;
- the loop gap probe sees CPU work, a blocking sleep, and an idle loop as three
  distinct readings.

Per-cell, the run also asserts that every injected character reached the
composer, that each viewer's sequence advanced by exactly the number of frames
delivered (a delta refused by the follower's exact-sequence check would fail
here rather than quietly shrink the load), and that every appended row landed in
the retained window.

## Commands

```sh
# A pristine, unmodified origin/main tree to measure — not a working checkout.
git -C ~/workspace/repos/lo-frame-bench worktree add --detach /tmp/lo-tui-baseline origin/main

cd ~/workspace/repos/lo-frame-bench
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/bench_tui_sessions.py \
    --source-root /tmp/lo-tui-baseline --output /tmp/tui-sessions-baseline.json
# -> 4 cells: 0, 2, 6, 12 attached sessions, 6 children x 500 retained rows each,
#    120 injected characters, 50 ms delta cadence. ~4 min per sweep.

# Repeat, same command, for reproducibility:
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/bench_tui_sessions.py \
    --source-root /tmp/lo-tui-baseline --output /tmp/tui-sessions-baseline-repeat.json
```

The JSON's `provenance.modules` asserts the app resolved from the named
`--source-root` (here `/private/tmp/lo-tui-baseline/local_operator/...`): a
before-run that silently measured the changed tree would fail that assertion
instead of reporting the change as a no-op.

## The numbers

Run A, canonical, `--sessions 0 2 6 12` (all CPU columns are `time.thread_time()`
per delivered delta, on the loop thread):

| attached sessions | window s | deltas applied | applied delta/s | offered delta/s | input→frame p50 ms | p95 ms | achieved FPS (wall) | FPS per loop-CPU s | loop CPU s | loop CPU % of wall | loop CPU gap p50 ms | ingest p50 ms | ingest mean ms | attributed CPU ms/delta | frames painted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 10.4 | 0 | 0 | 0 | **12.1** | 99.5 | **14.18** | 181.2 | 0.81 | 7.8 % | 0.2 | — | — | — | 147 |
| 2 | 17.5 | 94 | 5.4 | 40 | **304.5** | 913.1 | **3.88** | 29.6 | 2.30 | 13.1 % | 6.4 | 9.61 | 18.52 | 19.31 | 68 |
| 6 | 23.8 | 174 | 7.3 | 120 | **749.4** | 1802.1 | **1.76** | 11.3 | 3.72 | 15.6 % | 6.3 | 9.17 | 19.08 | 19.38 | 42 |
| 12 | 37.9 | 240 | 6.3 | 240 | **1489.3** | 4249.5 | **0.71** | 5.5 | 4.91 | 13.0 % | 14.0 | 8.31 | 19.22 | 19.45 | 27 |

Run B, the same command repeated (the reproducibility claim):

| attached sessions | window s | deltas applied | applied delta/s | offered delta/s | input→frame p50 ms | p95 ms | achieved FPS (wall) | FPS per loop-CPU s | loop CPU s | loop CPU % of wall | loop CPU gap p50 ms | ingest p50 ms | ingest mean ms | attributed CPU ms/delta | frames painted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 10.3 | 0 | 0 | 0 | 14.2 | 107.0 | 14.89 | 188.2 | 0.81 | 7.9 % | 0.2 | — | — | — | 153 |
| 2 | 19.5 | 82 | 4.2 | 40 | 341.5 | 1091.0 | 3.07 | 30.5 | 1.97 | 10.1 % | 6.1 | 8.72 | 18.24 | 18.93 | 60 |
| 6 | 27.9 | 120 | 4.3 | 120 | 1454.3 | 4007.7 | 1.08 | 11.2 | 2.68 | 9.6 % | 7.1 | 9.04 | 18.52 | 18.85 | 30 |
| 12 | 30.4 | 240 | 7.9 | 240 | 1917.8 | 4791.3 | 0.92 | 5.3 | 5.25 | 17.3 % | 14.5 | 8.96 | 20.48 | 20.70 | 28 |

What repeats exactly: the per-delta CPU (p50 8.3-9.6 ms, mean 18.2-20.5 ms,
attributed total 18.9-20.7 ms/delta across every cell and both runs), the
monotone collapse of achieved FPS and FPS-per-loop-CPU-second, the rise in the
loop's CPU gap, and the direction and rough size of every wall column. What moves
with the weather: the absolute wall percentiles and the frame counts, because the
window is sample-driven and the host's scheduling is not under our control.

## The control: is the per-delta cost the fixture, or the app?

The per-delta CPU above is measured *inside* a running TUI under load. The same
fixture, the same tree and the same reducer, with no app, no loop and no other
session (the `frontend_state` import must follow `probe_isolation`, as in the
harness):

```python
# /tmp/control_apply.py -- run as: .venv/bin/python /tmp/control_apply.py
import copy, os, statistics, sys, time

TREE = "/private/tmp/lo-tui-baseline"      # the pristine origin/main checkout
sys.path.insert(0, TREE)
sys.path.insert(0, os.getcwd())            # the branch holding the harness
for key in [k for k in os.environ if k.startswith("CMUX_")]:
    del os.environ[key]
import scripts.probe_isolation  # noqa: F401  -- before any local_operator import

from local_operator.session.frontend_state import FrontendStateStore, FrontendUpdate
from scripts.bench_tui_sessions import ProducerRoster, build_pool

roster = ProducerRoster("0000000000cf", 6, 500)          # the sweep's default shape
frames = build_pool(roster, 12)
follower = FrontendStateStore(roster.store.state)
samples = []
for frame in frames:
    payload = copy.deepcopy(frame)
    payload["sequence"] = follower.state.sequence + 1     # the follower is exact-sequence
    started = time.thread_time()
    follower.apply_update(FrontendUpdate.model_validate(payload))
    samples.append(time.thread_time() - started)
ordered = sorted(samples)
print(f"apply_update: p50={statistics.median(samples) * 1000:.2f} ms "
      f"p95={ordered[int((len(ordered) - 1) * 0.95)] * 1000:.2f} ms "
      f"mean={statistics.fmean(samples) * 1000:.2f} ms n={len(samples)} "
      f"retained={len(follower.state.jobs[0].trajectory)} rows/child")
```

```
$ .venv/bin/python /tmp/control_apply.py
apply_update: p50=6.84 ms p95=22.75 ms mean=10.51 ms n=12 retained=500 rows/child
$ .venv/bin/python /tmp/control_apply.py          # an earlier run, same command
apply_update        : p50=7.20 ms  p95=12.30 ms  mean=9.45 ms  n=12
```

This reproduces the audit's own follower-side figure (**6.8 ms per delta** at a
6-child roster with a 500-row retained window) to within 6 %. So the in-app
`ingest p50` of 8.3-9.6 ms is the code's cost plus ~1-2 ms of running-app
overhead, and the in-app *mean* of ~19 ms is a heavy tail — the p95 in run A
reaches 75-127 ms at 2-6 sessions — not a heavier fixture.

## What the numbers say about the claim

**The audit's extrapolation is confirmed, and the mechanism is sharper than
"linear".**

1. **The per-delta cost does not grow with N.** Ingest p50 is 9.61 / 9.17 / 8.31
   ms at 2 / 6 / 12 sessions (run A) and the attributed total is 19.31 / 19.38 /
   19.45 ms/delta. One delta costs what one delta costs, however many sessions
   are streaming; the per-session constant is a constant.
2. **The aggregate does grow, linearly and by construction.** Each attached
   session with a moving roster offers 20 deltas/s (the producer's 50 ms
   coalescer), so the loop's per-second cost is `N × 20 × ~0.01945 s` ≈ 0.39
   CPU-seconds per second **per session** — and there is exactly one loop thread
   to spend it. That is a single-core ceiling of `1.0 / 0.39` ≈ **2.6 attached
   sessions at this roster shape**, before any painting, ingest or timer work.
3. **The loop does not keep up, and the shortfall is user-visible.** At 12
   sessions the loop applied 6.3 deltas/s against an offered 240 (2.6 %), and the
   achieved frame rate fell 14.18 → 3.88 → 1.76 → **0.71 frames/s** while
   input→painted-frame p50 rose 12.1 → 304.5 → 749.4 → **1489.3 ms**. Frames per
   loop-CPU second fell 181 → 5.5, i.e. the loop became, almost entirely, a delta
   reduction engine.
4. **Which mechanism?** Ingest dominates the app-side handlers at the same
   cadence: 9.2 ms of `_on_frontend_update` per delta against ~0.9 ms for the
   current session's `_apply_frontend_state` (10x) and 91 µs of
   `_source_frontend_changed`, the retention predicate's full-state copy (100x).
   All three are linear in N; the first is the one that matters at this roster
   depth.

A saturated loop necessarily services fewer frames, and it does: this is the
measured size of that effect, not a hidden one.

**One caveat on the absolute rates.** The loop here is both busy *and* starved —
it burned only 8-17 % of each wall second — so the achieved delta and frame rates
are host-amplified: on an idle box the same code would service more deltas per
second and paint more frames. What is host-independent is the per-delta CPU, and
that is what the ceiling in (2) is computed from. The shape of the curve, not its
exact position on this host, is the finding.
