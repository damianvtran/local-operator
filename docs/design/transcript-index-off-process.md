# ADR: build the transcript index in a spawned worker process, not on a serve thread

Status: **proposed — architecture decision plus an env-gated POC (default off).**
Written by the architect against `origin/main` `d4d27b625977` (v0.107.1). The repo
keeps its decision records under `docs/design/` without numbers (there is no
`docs/adr/`), so this one follows that convention. Every `file:line` below is at
that revision. Every number carries the host's load average and swap use: on
this machine (14 cores, load 18–55, swap 12–15 GB used during the runs) a wall
time with no load figure means nothing.

## The problem as found

The operator reports 5–10 s sidebar opens. In 13 traced opens on v0.106.7 the
median click→snapshot was 1,054 ms and the max 4,491 ms, and the server-side
snapshot build was 67% of that (`f66ed86b7244/scratchpad/traced-opens-phases.json`).
#4143 and #4184 removed the bounded-wait causes. What is left is not a wait. It
is **GIL contention from transcript-index work running inside `lop serve`**:

* `start_refresh` runs `refresh_index` with `asyncio.to_thread`
  (`session/transcript_index.py:4459`). That function covers the whole
  invalidation ladder: `_scan_full` (`:3438`), `_scan_incremental` (`:3491`), and
  the hydration of the persisted publication via `_read_indexed` (`:1565`).
  All of it is pure Python, or single C calls such as `json.loads` on a whole
  snapshot, made while holding the GIL.
* The existing mitigation, "Yielding to other opens" (`:3903–4046`), pauses at row
  boundaries only while a paint hold younger than 5 s is live
  (`_YIELD_HOLD_STALE_S`, `:3990`). It stops pausing after 2 s in total
  (`_YIELD_BUDGET_S`, `:3984`), and it cannot split one C call. Between holds, a
  scan competes with every loop step at Python's 5 ms switch interval.
* The resident and parse-memo LRUs hold **4** sessions each (`_RESIDENT_SESSIONS`,
  `:252`; `_PARSED_SESSIONS`, `:1149`). An operator switching among more than 4
  large conversations therefore re-pays hydration on most opens, even when the
  persisted index is current.

### The anchor journal, measured

`439818272d84/transcript.jsonl` is now **808 MB and 137,555 rows**. The figures
below come from row heads only (`d4c94fe751cf/scratchpad/shape.py`, no content read out). Its
persisted index snapshot is **84.2 MB**
(`cache/transcript_index/439818272d84.snap.*.snapshot`).

| row class | rows | MB | note |
|---|---|---|---|
| `custom/frontend_state_checkpoint_v1` | 1,915 | 373.0 | ~195 KB each; **parsed in full** by the scan |
| `compaction` | 173 | 152.9 | ~884 KB each; parsed (below the 2 MB skip) |
| `message/assistant` | 39,037 | 123.9 | |
| `message/peer_message` | 14,847 | 48.1 | |
| `message/tool` | 35,153 | 45.7 | |
| attention_started / completion_attention | 12,911 / 12,903 | 6.6 | |

`_MAX_KEPT_LINE_BYTES` is 2 MB (`:241`), so `_skip_eligible` (`:2213`) never applies
to any of these rows. About 65% of the anchor's bytes are bookkeeping rows that
the scan `json.loads` in full and then reduces to `{id, seq, "bookkeeping"}`
(`source_projection_fact`, `:634`).

### Who triggers a scan on the serve process (question 1)

| caller | file:line | waits? |
|---|---|---|
| open frame `_frame_index` | `server/utils/desktop_sessions.py:5234–5265` | up to the paint deadline |
| journal-order correction `correct()` | `server/utils/desktop_sessions.py:5463–5512` | shielded, after paint |
| companion-answer resident warm `_warm_resident_scan` | `server/utils/desktop_sessions.py:5395–5425` | never |
| peer-display records | `server/utils/desktop_sessions.py:5775–5789` | never |
| checkpoint manifest `checkpoints_view` | `session/transcript_index.py:4554–4610` (route, `mobile/checkpoints.py:67`, `session/transcript_find.py:394`) | up to `_FIRST_PAINT_WAIT_S` when payable |
| boot prewarm | `session/index_prewarm.py:374, 539` (skipped at load ≥ 1.0/CPU, `:126,217`, so usually off on this host) | never |
| relay page loop | `network/relay.py:8662, 11447` | bounded |
| **synchronous** `refresh_index`, outside the single-flight | `harness/peer_cycle_display.py:463, 729, 821`, via `session/history_window.py:522`, `mobile/durable.py:819`, `mobile/projection.py:2345`, `tui/app.py:13668`, `tui/session_presentation.py:1255` | blocks its caller |
| session runtime's journal-order emitter | `session/runtime/serving.py:4103–4116` (runtime process, not serve) | shielded |

**Can two opens scan the same journal twice?** Within one loop, no:
`_IN_FLIGHT` single-flights per `(root, session)` (`:4421–4429`). Across loops
and processes, yes, and the code says so itself. `start_refresh` joins only a
task recorded on the same loop. The relay's page loop "is duplicated, not
joined" (`network/relay.py:11381–11400`). The runtime's own emitter scans the
same journal in its own process. The synchronous `peer_cycle_display` callers
bypass the single-flight entirely. The publish lock (`_publish_lock`, `:1649`)
serialises only the writes, not the scans.

**Trace evidence of the rate.** In the operator's 13-hour v0.106.6 desktop trace
(rows after epoch-ms 1794940906000) there are **14,483 `index.refresh_start`
rows**, about 18.6 per minute across sessions. That includes 579 for the anchor.
The marks carry no pid, so runtime children that inherited `LOP_OPEN_TRACE` may
contribute to the count. Adding `pid` to `open_trace.mark` would settle this.

## Options (question 2)

The comparison numbers are in "Measurements" below.

**(a) Spawned worker process.** `start_refresh` dispatches the same
`refresh_index` to a spawn-context `ProcessPoolExecutor`. The worker also
publishes the SQLite read-model companion and returns only a receipt (sig, rows,
token).
* Correctness: the invalidation ladder, the crash-safe store and the naming merge
  are unchanged, because the worker runs the same function. They were already
  cross-process: session runtimes publish through the same flock'd store. A
  worker killed mid-publish is exactly today's "serve crashed mid-publish". A
  torn delta past the committed length is ignored (`:1078–1090`), and a staged
  `.<pid>.tmp` snapshot is reclaimed once its pid is gone (`_staged_is_orphan`,
  `:1396`), and the worker's pid works there unchanged.
* Cost: the pickle hop is two strings and a small dict. **Hydration stays in
  serve** if serve keeps a resident `TranscriptIndex`.

**(b) A scan that yields.** This is already built: per-row pauses gated on paint
holds. Measured (base arm below), it still leaves p95 loop lag of 44–111 ms
in-process while three builds run. It cannot split one C call (an 884 KB
compaction `json.loads`, or the 84 MB snapshot decode on hydration). It does not
reduce CPU; it only reorders it. `json.loads` is 43% of a cold scan's CPU (256 of
594 ms on the 70 MB fixture), so even a perfect C parser leaves the other ~57% as
GIL-held Python classification. orjson is not a dependency.

**(c) Index at write time.** This mostly exists already. The runtime's emitter
(`serving.py:4103–4116`) refreshes after durable commits in the runtime process,
which is why the anchor's publication was 13 minutes old with a 2 KB delta when
inspected. Extending it does not reach:
* idle sessions with no runtime;
* compaction's inode swap, which forces a full rescan (`_sig_matches`, `:3588`);
* serve-side hydration;
* the incremental re-derive from the resume window. That re-derive can be tens of
  MB for one appended row (37.9 MB on the 51.7 MB fixture, `:3783–3790`), and
  moving it into the writer stalls the agent's turn.

**(d1) Head-classified bookkeeping rows.** Give `custom` non-attention rows the
same `_Row` and source fact from the head, without `json.loads`. Their fact needs
only `id`, the type, and that `payload` is an object. This cuts about 46% of the
anchor's parse bytes. It is a worker-CPU win that is complementary to (a), not a
substitute. Not built here.

**(d2) Serve answers from the companion, never hydrates.** The SQLite read model
already serves `summary`, `frame_facts`, `order_facts` and `checkpoint_manifest`
(`session/index_read_model.py:737–1215`), bounded and off-loop. Two things
block it from carrying the open today. It is published only after 2 s of quiet
(`_MATERIALIZE_IDLE_S`, `:96`), so a live session never gets one. And a
companion for a grown journal answers `stale` (`:311`), which falls back to
the in-serve refresh. Option (a) removes the first blocker, because the worker
publishes the companion at once. The second needs an **append-prefix rule**: on
the same inode, with size ≥ the witnessed size and a verified tail, rows below
the witnessed row count keep their facts, and only the tail is unknown.

**(d3) Free-threaded CPython or subinterpreters** (`InterpreterPoolExecutor`,
3.14). The repo supports `>=3.12` (`pyproject.toml:168`), and C-extension
dependencies are not free-threading-ready. Revisit later.

## Measurements (question 3)

**Fixture.** `d4c94fe751cf/scratchpad/rig/gen.py` is synthetic lorem text in the anchor's byte
mix at 1/10 scale. The big session is **70.0 MB / 9,850 rows**, with 54% bookkeeping
+ compaction bytes, 33.5 MB of bookkeeping rows at ~52 KB each, and 17
compactions at ~800 KB. There are two busy sessions of 34.2 MB each and one
small session of 1.2 MB. Cold full scan: 594–606 ms CPU, 1.0 s wall at load 22,
6.7 MB snapshot. Cold hydration (`read_index`, fresh process): 86 ms CPU.

### Mechanism: in-process loop lag, 3 cold builds at once

`d4c94fe751cf/scratchpad/rig/loopbench.py` runs a 1 ms asyncio ticker on the loop that
calls `start_refresh` for all three (70 + 34 + 34 MB). It runs in a fresh process
per run, four runs per arm, interleaved. The ticker's own lag is the loop-lag
measure. "serve CPU" is `process_time` of the serving process (loop plus threads,
excluding the worker). The figures are medians over 4 runs. Load 21.7–32.1, swap
14.54–14.55 GB used.

| arm | pool | build wall | serve CPU | lag p95 | lag p99 | lag max (median / worst) | loop ticks/s |
|---|---|---|---|---|---|---|---|
| base: thread (`to_thread`) | — | 1,144–1,700 ms | 849–1,026 ms | 44–111 ms | 102–135 ms | 123–181 / 217 ms | 34–126 |
| (a) worker + serve hydrates (`=1`) | warm | 2,241 ms | 196 ms | 10.9 ms | 25.3 ms | 84 / 100 ms | 287 |
| (a) worker + serve hydrates (`=1`) | cold spawn | 3,233 ms | 209 ms | 15.5 ms | 39.4 ms | 156 / 202 ms | 224 |
| (a)+(d2) worker, no hydration (`=2`) | warm | 1,971 ms | 49 ms | 8.0 ms | 14.1 ms | 26 / 33 ms | 369 |
| (a)+(d2) worker, no hydration (`=2`) | cold spawn | 4,021 ms | 88 ms | 13.5 ms | 23.2 ms | 86 / 140 ms | 226 |

The thread ranges are the cold-start and warm medians: the base arm has no pool,
so its two halves differ only by order. Raw rows are in `d4c94fe751cf/scratchpad/rig/lb2.jsonl`.

### Real path: isolated `lop serve` over HTTP

`d4c94fe751cf/scratchpad/rig/drive.py` boots a fresh `lop serve` per cell (`env -i`, scratch
`HOME` and config, `LOP_OPEN_TRACE`, standby disabled, port 18761) and reaps it
by exact process group (no leftovers in any cell). For each cell:

1. Warm-open the small session.
2. Trigger cold builds of all three large sessions through
   `GET /v1/desktop/sessions/<id>/checkpoints`. This uses no acquire, so the only
   load is the index work.
3. While they build, re-open the small session
   (`GET …?open_frame=1&journal_order=1&limit=100`) every ~150 ms, and probe
   `/health` every ~10 ms.

There were 5 interleaved reps per arm, at load 18.9–37.0 and swap 13.36–14.72 GB.
The samples below are pooled over each arm's build windows.

| arm | `/health` n | p50 | p95 | max | small re-open `a.snap.total` p50 / p95 / max | small re-open wall p50 / max | index ready (head) | serve+workers RSS peak |
|---|---|---|---|---|---|---|---|---|
| base: thread | 75 | 13 ms | 233 ms | 518 ms | 71 / 281 / 281 ms | 255 / 961 ms | 0.6–1.7 s | 225–260 MB |
| (a) `=1` | 189 | 29 ms | 290 ms | 967 ms | 78 / 233 / 270 ms | 189 / 1,051 ms | 2.3–7.4 s | 350–439 MB |
| (a)+(d2) `=2` | 145 | 25 ms | 118 ms | 421 ms | 97 / 177 / 217 ms | 161 / 494 ms | 2.0–3.1 s | 346–402 MB |

Reading it honestly:
* Over HTTP on this host, `/health` latency is dominated by host scheduling.
  Settled-idle `/health` p50 alone is 5–33 ms. The `=1` arm's window is
  2–5× longer, so it collects more of the host's tail.
* At 70 MB the open itself is already under 500 ms in **every** arm: small
  re-open `a.snap.total` has p95 ≤ 281 ms.
* The process boundary shows clearly in the mechanism table. Serve-side CPU fell
  from about 0.85–1.0 s to 0.2 s with hydration and 0.05 s without. Loop lag p99
  fell from about 100–135 ms to 25 ms and 14 ms.
* Over HTTP, only the no-hydration arm moves the tail: p95 halves and the max
  falls 19%.
* The worker arms finish **later** (2–7 s against 0.6–1.7 s). Three reasons: two
  workers queue three builds; the worker also builds and fsyncs the companion,
  which the thread path defers to an idle timer; and the first build pays spawn
  plus import under swap.

### Extrapolation to the 808 MB anchor

Everything scales linearly in bytes parsed, which is what the code does: one pass,
no quadratic step. The anchor's mix is ~12× the fixture's bytes and ~14× its rows.

* A cold anchor scan is about **7–8.5 s of GIL-held CPU in serve** today
  (594 ms × 11.5–14). With `=1` or `=2`, that moves to the worker.
* Hydrating the anchor's 84 MB snapshot costs about **1.1 s** of serve CPU (86 ms
  × 12.5). That is consistent with the in-code figure of 0.54 s for a 62.9 MB
  index (`:1570–1572`). One such call is a single uninterruptible `json.loads` of
  the snapshot plus typing, so a mode-1 serve still holds the GIL for ~0.5 s
  stretches on every LRU miss of the anchor.
* At anchor scale, therefore, **(a) alone is not enough**. The 0.2 s → 1.1 s
  hydration residual sits in the same band as the 1–4.5 s opens the operator
  sees. **(a)+(d2)** keeps serve at the 49–88 ms floor, independent of journal
  size, because companion queries are bounded by the page.
* Not measured at anchor scale: worker peak RSS. In-serve today, the scan's
  transient heap for an 808 MB journal is held by serve itself, and pymalloc
  rarely returns it. In a worker it returns on recycle (`max_tasks_per_child`).
  The rig measured +100–180 MB for two idle-plus-busy workers at 70 MB.

### Found, not in this decision

A first open after a serve restart with a metadata **miss** pays `deferred_proof`
(`session/attached.py:2258`): 450–530 ms on the 70 MB fixture, which extrapolates
to ~6 s on the anchor. That is a second whole-journal pass in serve, and it is the
likely source of the 39 s first-ever acquire. The same worker is the natural home
for it, as a follow-up.

## Decision

**Adopt (a) + (d2): every transcript-index derivation runs in a spawned worker
process, and serve answers opens from the read-model companion the worker
publishes.** Ship it in three steps:

1. **Worker build (this POC's `=1`, made production).** `start_refresh` dispatches
   to a per-process spawn pool. Changes needed to make it production:
   * Enable it by an explicit call in serve's startup (`server/app.py`), not by an
     environment variable. Runtimes inherit serve's environment, and a pool per
     runtime is N×2 interpreters. The POC already gates on both: the env var
     selects the mode only in a process that called `host_scan_pool()` (serve's
     lifespan), and `_runtime_env` scrubs `LOP_INDEX_SCAN_PROCESS` from every
     runtime (review R1-1).
   * Drop the pool at serve's lifespan shutdown without joining it
     (`shutdown_scan_pool()`: `wait=False`, `cancel_futures=True`). Otherwise
     interpreter exit waits out an in-flight anchor cold scan (~7–8.5 s) before
     serve can stop. Done in the POC (R1-4).
   * Single-flight across loops: share a `concurrent.futures.Future` per
     `(root, session)` that any loop awaits through `asyncio.wrap_future`. This
     closes the relay page-loop duplicate (`relay.py:11381–11400`).
   * Take a per-session build flock in the worker, then re-check the signature
     after acquiring it, so a runtime and serve never scan the same bytes
     concurrently. The second caller gets an exact HIT. **Delivered by #4224**,
     which adds the per-session `<sid>.build.lock` inside
     `transcript_index.refresh_index`. The worker calls that same
     `refresh_index`, so it inherits the lock; this PR deliberately adds no
     second lock around the call.
   * Drop `_defer_for_paint` and the yield machinery for worker builds; the POC
     already bypasses the deferral.
   * The synchronous `peer_cycle_display` callers (`:463, 729, 821`) take a
     current index or `None`; they never build. This matches their own docstring,
     "never a scan", which the code does not honour today.
2. **Companion-first open (d2).** Publish the companion from the worker
   immediately; that is done in the POC. Add the append-prefix rule to `header()`
   so a live session's companion answers `ready` for rows up to its witness. Above
   a size threshold, `_wrapped` keeps no resident copy, and the
   `fresh_resident` consumers move to companion queries: `correct()`, peer display,
   open-frame runs, `transcript_find`, relay.
3. **(d1)** Head-classify bookkeeping rows in `_classify`, with a byte-identical
   golden test against the parsed path. This cuts worker CPU and every runtime's
   own refreshes by about half on journals shaped like the anchor.

**Rejected.** More yielding (b): it is measured and already shipped, and its
residual is structural. Extending write-time indexing (c): it already exists in
the runtime, and the remainder would move cost onto the agent's turn.

### Invalidation contract (unchanged, restated)

* The ladder is the one in `refresh_index` (`:3615–3621`), executed in the
  worker: an exact `(size, mtime, inode)` match reuses; the same inode, grown,
  with a verified tail, increments; anything else rescans.
* Serve trusts a resident copy only through `fresh_resident`'s two stats, and a
  companion only through `header()`'s journal witness. The append-prefix rule
  only widens `ready` for rows the witness covers.
* The journal remains the authority, and the JSON head remains the commit point.
  The companion is still derived and disposable.
* A worker failure is one failed refresh: `_FAILURES`, a logged warning, and the
  previous state is served. A `BrokenProcessPool` drops the pool that broke
  (never a replacement another refresh created meanwhile), and the next refresh
  respawns it. The POC tests both paths.
* A companion publish failure in the worker (`OSError`, `ValueError`,
  `sqlite3.Error`, the classes the thread path memoises) returns in the receipt,
  and serve records it with `index_read_model.remember_failure`. Serve passes
  its memoised failed source into each dispatch, and the worker skips the
  publish while the journal is unchanged. The memo therefore lives in serve and
  survives worker recycling, as on the thread path.

## Rollout and the risks to watch

* **The flag stays default-off** until step 2 lands. At anchor scale, `=1` trades
  ~8 s scans for ~1 s hydrations, and the HTTP rig did not show `=1` beating base
  at 70 MB.
* **Freshness lags.** Builds finish later (median 2.0–2.2 s against 1.1–1.7 s for
  three concurrent builds in-process, and longer on the first spawn). Watch the
  delay from `ready` to first paint in `index.worker_done` against
  `index.refresh_start`. Consider pre-spawning one worker at boot (import cost
  ~0.5 s).
* **Memory.** Two workers add ~100–180 MB at idle and scan. Watch serve+worker
  RSS under the memory guard. `_SCAN_TASKS_PER_CHILD` recycles workers. For a
  memory-pressured host, consider one worker plus a priority for the session
  being opened.
* **Spawn hygiene.** Spawn re-imports the parent's `__main__`. The `lop` console
  script is guarded and was verified in the serve rig. An unguarded script
  failed loudly with `RuntimeError: ... before the current process has finished
  its bootstrapping phase`, which is a loud failure, not a hang. The packaged
  desktop backend runs under a named launcher. The rig's serve showed
  `Path: …/Local Operator`, `com.local-operator.runtime`, and spawn worked
  through it, but the bundled app build must be checked once.
* **Other consumers.** The TUI and the mobile projection call `refresh_index`
  synchronously in their own processes. Step 1 leaves them as they are, apart
  from the "never build" change for `peer_cycle_display`. Their freshness then
  depends on serve's or the runtime's builds, so check that the TUI's peer-cycle
  display still fills on a cold session. The relay gains the cross-loop
  single-flight. The native app reads through the same routes and is unaffected.
* **Evidence instrumentation.** Add `pid` to `open_trace.mark` before the next
  live trace, so serve-side refreshes can be told from runtime-side ones.
