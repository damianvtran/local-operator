# Background activity and the viewer input loop

The normal terminal frontend is an `AttachedSession`: runtime deltas arrive on
its event loop, which is also Textual's keyboard and compositor loop. A scalar
activity update must not serialize, validate or freeze every retained child
trajectory. It must also not rebuild unrelated compatibility collections.

## Ownership and ordering constraints

- `FrontendStateStore.apply_update` validates the supplied fields through an
  ephemeral `FrontendSessionState`, then installs only its explicitly supplied
  fields. This preserves model field validators and `extra="allow"` across mixed
  runtime/viewer versions. Unchanged owned job payloads remain shared and frozen.
- Job deltas retain the existing append/replacement and todo-plan merge rules.
  Todo sequence watermarks are staged with the update, so an invalid later field
  cannot partially consume a plan. Explicit `jobs=[]` clears; `jobs=null` remains
  invalid. A degraded frame advances sequence without pretending its omitted
  body was unchanged; resynchronization still belongs to the attached subscriber.
- `AttachedSession` replaces the jobs/comms, wakes and MCP compatibility
  collections only when their fields changed. A full snapshot always replaces
  all collections, including after reconnect and sequence reset.
- `SnapshotJobs.get` indexes IDs, but returns a detached Pydantic shell just as
  before. Its first-duplicate-ID semantics and the ordered list view are retained.
- The TUI coalescer retains the authoritative session until its scheduled paint,
  then reads one latest public snapshot. Session-generation invalidation happens
  before that read, so queued callbacks cannot paint or copy a retired session.

None of these changes caches a mutable job by its identity, trajectory length,
status or current progress string. Those are not revisions: a bounded trajectory
can rotate at constant length, and nested usage/todos can change independently.

## Reusable deterministic benchmark

```sh
.venv/bin/python scripts/bench_tui_background.py \
  --attached --children 0 10 50 --samples 120 \
  --capture /tmp/tui-after --output /tmp/tui-after.json

# The same harness and dependencies against an exact baseline source archive:
.venv/bin/python scripts/bench_tui_background.py \
  --source-root /tmp/tui-baseline-source \
  --attached --children 0 10 50 --samples 120 \
  --capture /tmp/tui-before --output /tmp/tui-before.json

# Keep a long transcript scrolled away from the live tail while slow backend
# pricing work completes. This records each composed frame's scroll offset.
.venv/bin/python scripts/bench_tui_background.py \
  --attached --children 50 --samples 120 --scroll --slow-stats-ms 50 \
  --output /tmp/tui-scroll-io.json
```

The script isolates HOME/config before any application import, scrubs every
`CMUX_*`, and uses synthetic session IDs. Production shimmer remains enabled;
the capture helper's usual no-animation setting would remove part of the load
being measured. No provider requests or operator sessions are needed.

The owner mode (without `--attached`) uses the real job ledger, canonical state
projection and progress event fanout. Attached mode executes the real
`AttachedSession._on_frontend_update` callback, including validation, facade
updates and subscriptions, with deterministic wire-shaped deltas. It starts at
the decoded socket callback, not at a model or an actual transport connection;
independent end-to-end QA must exercise the latter separately.

An OS thread injects keys independently of the event loop. Timing ends only when
Textual's actual compositor update contains the new marker in the composer's
rows. This is **input injection to composed frame**, not `pilot.press`/`pause`
roundtrip time, and not a claim about a physical terminal's display latency.
Wrapping is exercised by 120 deterministic characters. Positive/negative
canaries require an injected blocking sleep to appear in wall latency, a normal
character to paint, and an absent marker to remain undetected.

A burst is offered every 50ms after the previous burst finishes. Load continues
until every key has been injected, with `--edges` setting only a minimum number
of bursts. This prevents the fixed build from finishing its work early and
measuring most keys on an idle screen. Report achieved event/burst counts and
throughput: a saturated baseline necessarily services fewer bursts. All offered
edges must reach the correct final state and every injected character must
appear in a composed update. Per-operation counts/latencies and loop CPU/wall
gaps distinguish loop work from ambient host contention. No numeric latency
ceiling belongs in CI.

Run paired/interleaved baseline and fixed cells on the same host. JSON records
application source paths/hashes, interpreter and harness hash; verify an archive's
files against the baseline commit before relying on it. Generated JSON, logs,
SVGs, geometry and PNGs are external evidence, never repository contents.
