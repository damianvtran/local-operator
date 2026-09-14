# frame cost / loop starvation — measured evidence

The operator's report: *"can't load some sessions in the TUI; desktop send fails
with 'The backend could not complete this request'"*. The architect's diagnosis
(`arch-resume-wedge`) found a live runtime whose **event loop** is the resource
being starved — the record's heartbeat is written by an asyncio task on that same
loop, so "the loop has not run" is reported to every surface as "this session is
dead" (`registry.scan` → `wedged` → the sidebar, the picker, the phone, and the
TUI's 27 s dial budget).

Two defects were measured. **This change fixes the second one only**: the
frontend-state layer re-materialises every job's whole retained trajectory on
every roster tick (`JobState.from_job` deepcopy + `_freeze_job` re-freeze +
`_jobs_equal` deep compare, ~20×/s while a turn streams), and `sync_wire_payload`
serializes ~100 MiB of retained tool results to ship a 16.7 KiB frame.

**Change 1a** — `FrontendStateStore` keeps a per-job memo of the frozen retained
row window (`_TrajectoryWindows`), so a tick freezes only the appended tail; a
fingerprint over the writer's monotone append stamp, the raw row list identity
and the tail row identity decides when the memo can be reused, and every
invalidation path is enumerated in the class docstring and covered by a test.

**Change 1b** — the sync frame's dump leaves the retained rows out
(`_SYNC_OMIT_TRAJECTORY_KEY` serialization context) instead of dumping and then
stripping them. The frame's **bytes are unchanged**; only its cost is.

## Reproducing

```sh
# Both halves of every A/B below run the SAME file, same host, same interpreter;
# cwd is /tmp so PYTHONPATH decides which tree's `local_operator` is imported
# (sys.path[0] is the script's directory -- the first line of output names the
# tree that actually ran, and that must be checked).
PY=<frame-cost-loop-starvation>/.venv/bin/python
EV=<frame-cost-loop-starvation>/docs/evidence/frame-cost-loop-starvation

cd /tmp && PYTHONPATH=<tree> $PY $EV/frame_cost_harness.py all            # §1, §2
cd /tmp && PYTHONPATH=<tree> $PY $EV/canonical_state_equivalence.py      # §3
cd /tmp && env -u CMUX_WORKSPACE_ID -u CMUX_SESSION_ID -u CMUX_SURFACE_ID \
  -u LOP_MOBILE_CHILD_PROVIDER -u LOP_RUNTIME_ADOPT_SESSION \
  HOME=/tmp/frame-cost-demo-home PYTHONPATH=<tree> $PY \
  $EV/loop_responsiveness_demo.py                                       # §4
```

Baseline tree: `~/local-operator-worktrees/arch-resume-wedge` pinned at
`origin/main` @ `9dc694995` (the ref the architect measured). Post-change tree:
the PR branch. Interpreter for both sides: `Python 3.12.13`, `gc` thresholds
`(700, 10, 10)` — the numbers are wall cost on a **loaded** shared host, which is
the environment the symptom occurs in, and the load average is printed on every
run.

## 1. One roster tick: 5 subagents × 500 retained rows × 8 KiB

Two rounds, interleaved, on the same host (load averages 160-171 / 14 cores):

| per tick | baseline (round 1 / 2) | after (round 1 / 2) |
|---|---|---|
| idle (window not moving) | **58.6 / 76.0 ms** | **0.99 / 0.13 ms** |
| idle GC collections | **87.2** | **0.0** |
| idle rows frozen | **2,500** | **0** |
| streaming (1 row/job/tick at the cap) | **79.6 / 76.4 ms** | **13.0 / 13.9 ms** |
| streaming GC collections | **106.2** | **18.1** |
| streaming rows frozen | **2,500** | **5** |

At the pump's 50 ms cadence the baseline demands 117-160 % of a core *to publish
nothing*; after the change the idle tick is 0.13-0.99 ms (0.3-2 % of a core). The
streaming column now freezes exactly the appended tail: **5 rows/tick for 5
children**.

**Where the residual 13 ms of the streaming tick goes** (attributed with the
architect's `Chrono` wrapper, 20 ticks): `mutate` 11.8 ms of 12.7 ms, of which the
store's own re-materialisation is 0.2 ms. The rest is the reducer's PRE-EXISTING
at-cap delta: when the window rotates past the cap it ships the whole 500-row
replacement per child to subscribers (`mutate`'s `trajectory_appends`), which the
architect measured at 17.5 ms and which this PR does not change. It is the
follow-up candidate the report's "serve rows on demand through `job_trajectory`"
direction addresses; out of scope here.

## 2. `sync_wire_payload` (the desktop's `frontend_sync`, on the loop)

| roster | baseline (round 1 / 2) | after (round 1 / 2) | wire payload |
|---|---|---|---|
| 5 jobs × 500 × 2048 B (13.4 MiB) | 25.4 / 18.3 ms | 0.3 / 0.3 ms | 4.9 KiB |
| 5 jobs × 500 × 8192 B (28.0 MiB) | 20.0 / 19.6 ms | 0.2 / 0.2 ms | 4.9 KiB |
| 22 jobs × 500 × 2048 B (58.9 MiB) | 96.3 / 96.8 ms | 0.6 / 0.6 ms | 17.3 KiB |
| 22 jobs × 500 × 8192 B (123.3 MiB) | 102.0 / 108.7 ms | 0.6 / 1.7 ms | 17.3 KiB |

22 jobs is the operator's wedged session `835fbcafdc27` (22 job rows, 256 comms
records); 5 is `ec6291a4700d`. The payload is byte-identical to the baseline's in
every row — the frame's size is the point of the strip, and it did not move. The
architect measured the same shape at 460-491 ms on the reference host under
heavier load; the direction and the mechanism are the same, the absolute number
is host-dependent.

## 3. Canonical-state equivalence (§3 = the correctness claim)

`canonical_state_equivalence.py` drives one deterministic scenario list through
both trees and prints a SHA-256 digest of `store._state.model_dump(mode="json")`
after every step. Shapes covered: idle ticks; one-row appends; a fill to the cap;
cap rotation; a 40-row burst; a front-only trim; the tail row replaced by an equal
copy; an unstamped row; a status move; all rows dropped; the row **list** replaced
by a second attempt with the stamps restarting at 0; an epoch move; a job leaving
and joining the roster; `refresh_from_session` plain and `initial=True`.

**20 of 21 digests are identical between the two trees.** The 21st is printed
last, labelled, and deliberately divergent: a row revised **in place** with its
stamp untouched (`201464c6…` baseline vs `3eddea88…` here). The writer never does
this — `subagent._make_relay` appends a fresh dict per event and never rewrites a
retained row, and `subagent.runner` **rebinds** the list rather than editing it —
and it is pinned by `test_the_relay_writer_only_appends_and_trims_the_front`, which
drives the real relay past the cap and asserts contiguous monotonically-increasing
stamps, front-only trimming, and no replacement of a retained row object.

The scenario list is also the shape the memo's own tests cover
(`tests/unit/session/test_frontend_row_window.py`): each guard is
mutation-checked — removing it makes a named test fail.

## 4. End to end: does the loop stay answerable?

`loop_responsiveness_demo.py` stands up, in one isolated config dir
(`LOCAL_OPERATOR_CONFIG_DIR` + redirected `HOME`, synthetic session id, no TUI, no
fork, nothing of the operator's touched):

* a real record published through `registry.publish`, heartbeated by a task
  shaped exactly like `RuntimeServer._heartbeat_loop`;
* a real `asyncio.start_server` control port answering an auth frame with a
  welcome — both halves loop callbacks;
* the repaint pump faithfully: a 50 ms `loop.call_later` tick calling the real
  `FrontendStateStore.refresh_jobs`, appending one row per child per tick (the
  shape a roster change schedules) at the 500-row cap;
* a sibling thread reading the record/`registry.scan` every 5 s, and one dialling
  the port the way the attach path does.

| 60 s run, 5 children × 500 rows × 8 KiB, pump at 50 ms | baseline | after |
|---|---|---|
| tick cost | **mean 107.5 ms**, p50 85.2, p95 246.2, **max 496.8 ms** | **mean 20.4 ms**, p50 17.9, p95 33.6, max 158.0 ms |
| dial welcome after auth | 28, 87, 105 ms | 1, 1, 30 ms |
| dial connect | 8, 9, 24 ms | 0, 1, 5 ms |
| max heartbeat age (timeout 45 s) | 15.1 s | 15.1 s |
| `registry.scan` verdicts | `live` throughout | `live` throughout |

**Reported honestly: the heartbeat stayed inside the timeout on BOTH sides at
this scale.** That is the architect's §B5 negative result reproduced: defect B
alone did not produce a stale heartbeat here. What the pump stops doing is
demanding 215 % of a core on a 50 ms cadence, and what improved with it is the
pool of loop time a dial competes for (welcomes 28-105 ms → 1-30 ms). The
minutes-long silences need the third, measured, non-code term (app-spawned
runtimes at `pri 20` getting ~0.4 % of a core) — out of scope, the operator's call.

## What this change does NOT fix

* **The macOS scheduling band** — session runtimes spawned by the desktop app run
  at `pri 20` and get ~40× less CPU than a terminal-spawned sibling under the same
  load. Not a code defect in this repo.
* **The at-cap replacement delta** (§1's residual) — a single appended row at the
  cap still ships a full 500-row replacement to subscribers per tick.
* **"Alive but slow" vs "dead"** — `loop_tick_at` + an off-loop record writer +
  `registry.classify()` are a separate workstream (peer-owned), and the TUI dial
  budget and kill-switch change (the architect's 2b) depend on that record field,
  so it is deferred rather than duplicated here.
