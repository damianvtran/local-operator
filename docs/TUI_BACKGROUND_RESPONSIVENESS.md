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

## The runtime-side retained-window memo

The writer side of the same value is held to a stricter bar, because there the cost
was measured (`scripts/bench_roster_tick.py`): a roster refresh
re-froze every retained row of every job on a 50 ms coalescer and demanded more
than one core to publish nothing. `FrontendStateStore` therefore keeps a per-job
memo of the frozen retained window (`_TrajectoryWindows`,
`session/frontend_state.py`), keyed on the raw row list's OBJECT identity, the
count and the stamps at both ends (which the writer never revises), and the tail
row object. Those are the facts the paragraph above rules out as sufficient on
their own; used together they prove a window is UNCHANGED, which is the only
claim a memo may make. A rotation at constant length moves the end stamps, a
rebuild replaces the list or its tail object, and an unstamped row is never
memoised at all; anything it cannot prove is re-frozen. `replace()`,
`refresh_from_session(initial=True)` and an epoch move drop the memo, and a job
leaving the roster drops its entry. This memo guards the writer's window only;
the viewer's `apply_update` path carries its own, described next.

## The viewer-side memo and the producer's capped tail

The reader's half of the same value was tied to the retained window by
construction: `FrontendStateStore.apply_update` re-materialised every job's whole
retained trajectory on every canonical `jobs` delta, so an attached viewer paid
the window's depth per delta however small the delta was. The two mechanisms
below mirror the writer-side memo, and both are proven rather than assumed:
`_FollowerTrajectoryWindows` proves a window is UNCHANGED, `_capped_overlap_tail`
proves the appended rows before it ships them.

- `_FollowerTrajectoryWindows` (`session/frontend_state.py`) memoises the frozen
  window a follower has installed, per job, and extends it by only the delta's
  appended rows. The proof a memo applies is the previous canonical job's
  `trajectory` being the very object the memo installed, which is conclusive
  because the frozen containers are immutable and the reducer installs windows
  with `model_copy`, so no re-coercion can smuggle an equal-but-different list. A
  `(count, last_seq)` fingerprint is deliberately **not** the proof: a second
  attempt at the same job id presents the same count and the same stamps, and the
  window is a bounded page, so its length is not a function of the runtime's
  history. Retained rows never reach pydantic, and `trajectory_length` is derived
  from the window actually installed, keyed on **presence** exactly as
  `JobState._derive_trajectory_length` keys it, so an explicit-but-invalid count
  still reaches pydantic and is refused.
- `_capped_overlap_tail` is the producer's second classifier arm. When the new
  window is exactly the cap and a nonempty overlap of the old window's back and
  the new trajectory's front is proven **row for row**, the appended rows are the
  new trajectory's tail and the receiver's own `(old + tail)[-CAP:]` lands on the
  new window exactly. The stamp is a *candidate*, never the proof: `_lo_seq`
  counts relays, so it can be reset, repeated or non-integer, and an interior
  edit can leave both endpoints agreeing. Every returned tail is backed by one
  element-wise comparison; anything unprovable returns `None` and keeps the
  replacement the owner has always sent. No new wire field is needed: the
  follower already trims at the cap, so a receiver that does not know the owner
  evicted anything reconstructs the rotated window all the same.
- `FrontendStateStore.has_running_job` is the clone-free sibling of
  `pending_gate`, because the retention predicate asked "any child running?"
  through a full deep copy of canonical state, on the loop, once per delta of
  every leased source.

A tail is only shipped when it is proven, because a wrong tail sends the wrong
rows to a viewer. The cases that deliberately fall back to a replacement are
below-cap front deletion, a new window shorter than the cap, an interior edit
whose endpoints agree, reordered rows, missing / boolean / non-integer / reset
stamps, an empty prior window, and a delta with no append. A replacement frame
and a proven tail are asserted to land on the same canonical state, so the fast
arm is equivalent rather than merely green; a hydrated follower is compared row
for row through rotations and across the cap after every tick. An older receiver
with no follower memo still reconstructs by tail, which is the compatibility path
this design keeps open on purpose.

## The per-source subscription coalescer

`OperatorApp._watch_source_frontend` installs one subscription per leased source,
and the callback that subscription installs coalesces the source's deltas into
ONE change callback per source per loop turn, matching the guard
`_on_frontend_update` has carried for the current session. The scheduled bit
lives on the SOURCE rather than on the app — the app has one current session,
while N leased sidebar sources each carry their own pending callback — and it is
cleared by the callback alone, as that callback's first statement and never
behind a guard.

The bit is claimed only when Textual ACCEPTED the callback. `call_later` returns
False on a closing or closed pump, and latching on a refused schedule would
strand the source permanently deaf to its owner: the callback that clears the bit
is never going to run, and the next delta finds the bit set and returns. A refused
schedule therefore leaves the bit unset, so the owner's next delta tries again.
The residual window belongs to Textual: a Callback queued successfully but
dropped uninvoked by `on_callback` when the app is closing or has no screen —
shutdown, not a live lease. The direct
`call_later(self._source_frontend_changed, source)` sites (subagent events, gate
transitions, the close drain) keep their own cadence deliberately: the coalescer
sits on the subscription, so an event-driven change cannot be swallowed by a
source that already has a queued callback.

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

## Workload semantics, and where the coalescer pays

The bench's numbers mean nothing without the workload that produced them, and two
axes carry a trap:

- `--workload capped` (the default) mirrors runtime append/front eviction and
  reports replacement job counts per frame. `--workload suffix` disables owner
  eviction to isolate append cost, and its results must **not** stand in for
  long-running runtime performance: a suffix fixture never reaches the
  replacement arm, so a frame rate taken there measures the fixture.
- The decoded-callback instrument bypasses the transport's watched-job filter and
  byte budget, so an all-detail cell is reducer stress, not default parked socket
  traffic. Real-socket QA must cover both watched and unwatched connections. A
  frame rate from this harness is a capacity figure rather than natural demand,
  and the per-delta CPU column — not the wall percentiles — is the load-robust
  half.

To those two axes this change adds a third, and it is about the coalescer itself:
**the coalescer pays only on bursts.** The bench offers one delta per 50 ms, the
producer's own cadence, so each delta gets its own loop turn, the scheduled bit is
always clear when the next one arrives, and the coalescer is a no-op there — 440
deltas cost 440 callbacks. A production-shaped burst of 3 deltas per 150 ms
engages it: 480 deltas cost 160 callbacks, 320 suppressed. Reading the coalescer's
effect out of the one-delta-per-cadence cells would report it absent for a reason
that belongs to the fixture, not to the code.

The parked lane and the watched lane are separate findings, and only one of them
is the render path. A parked (unwatched) leased source costs a constant ~3,785
B/delta of owner relay, so twelve parked sources stay under 1 % of a loop;
watching all six children instead costs ~207 KB and ~6.4 ms per delta, i.e. ~2.5
MB and ~77 ms per wave at twelve sources.
`RuntimeServer._relay_frontend_to_on_loop` already filters through the
connection's watched jobs before serialization, so the parked lane never carries
the detail the all-detail bench hands over unconditionally. That is why this work
fixes the producer's window and the follower's reduce instead of adding a
summary/presentation subscription: the parked lane was never the problem, and a
second stream would have duplicated a projection the transport already performs.
The before/after tables, the exact commands and the reducer control that produced
them are on the PR, not in the tree.
