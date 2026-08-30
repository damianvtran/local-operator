# TUI responsiveness benchmarks (S1/S2/S3)

`scripts/bench_tui_lag.py` reproduces the three reported symptoms with a
loop-lag monitor (5 ms tick probe, stalls recorded from 30 ms). Run:

```sh
PYTHONPATH=. .venv/bin/python scripts/bench_tui_lag.py --json out.json
```

`PYTHONPATH=.` matters: the script must import THIS checkout's
`local_operator`, and a plain run from another directory can resolve an
editable install pointing elsewhere.

## before.json / after.json

Captured on this machine (macOS/APFS) against `origin/main` @ `f2a52b53`
(before) and the same tree with the fixes (after), 1,000-session synthetic
store, 8 MB bash output, discovery stubbed to 400 ms.

| Scenario | Metric | Before | After |
| --- | --- | --- | --- |
| S1 boot scans | first pass (writes markers) | 184 ms | 358 ms (one-time: writes 1,000 sentinels) |
| S1 boot scans | second pass (steady state) | 178 ms | **40 ms** (stat-only) |
| S1 boot scans | loop stalls > 30 ms | n/a (was on the loop: 460-490 ms warm, 2 s cold, per the design audit) | 0 |
| S2 send (pricing) | `turn_cost` from the loop, cold memo, hostile listing | **807 ms loop block, 812 ms max stall** | 0.6 ms, 0 stalls |
| S3 bash emit | per-tick body cost, 8 MB accumulated | 1.7 ms median (O(total); 32 MB measures 4.7 ms and grows) | 0.03 ms, flat |
| S3 bash emit | live payload per update | 8,323,237 chars | 131,110 chars (bounded) |

The before/after first-pass inversion in S1 is expected and correct: the
fixed tree's first pass performs the one-time migration (writes one
`title-scan.json` sentinel per answered directory). Every later boot is the
second-pass number, which is the steady state a user lives in.

The design audit's own measurements on the real store (1,365 sessions,
431 MB) for the same code paths: `_prepare` 519-544 ms with a 460-2,000 ms
loop stall; `turn_cost` cold-miss 418 ms (2 s budget branch) to 13 s
(unlisted model); boot to model label delayed by the 251 ms MCP gate. The
synthetic numbers above are the reproducible proxy for the same invariants.

## Boot-to-label with deferred MCP wiring (fix B)

Measured with the real `create_session` (hosting `test`) and a 500 ms
discovery stub: with `defer_mcp_wiring=True` the factory returned in
**269 ms** while wiring was still in flight (the stub had not settled), so
the TUI can adopt the session and paint the model label before the MCP
segment fills. On the unfixed tree the same call blocked for the stub's
full 500 ms plus the real 250 ms gate before returning.

## Parallel instances (cross-instance scope)

`p_parallel_boot`: five concurrent scan passes over one shared 1,000-session
store — the shape of five lop instances booting together.

| Metric | Before | After |
| --- | --- | --- |
| First boot (one-time migration) | 2,276 ms | 1,415 ms |
| 5-parallel steady-state wall | 656 ms | 391 ms |
| Slowest instance (steady) | 605 ms | 363 ms |

Every instance's steady-state pass drops to stat-only (A2 sentinels, both
title and origin), so N parallel instances cost N x ~40 ms of stats instead
of N x full-store reads. The remaining wall time is the shared-directory
I/O itself, not redundant parsing.

Cross-instance findings (DA-rows in the remediation):

- **DA1 (fixed, A1+A2)**: O(N x store) redundant boot scans eliminated.
  Sentinel writes are plain `write_text`+`replace` — no fsync, so no
  cross-instance fsync amplification. The to_thread'ed scans run in the
  default executor (no shared asyncio.Lock between them; the only
  serialization is per-instance sequential awaits, preserved deliberately).
- **DA2 (verified, no change)**: C2's background refresh resolves through
  `available_models` -> `cached_listing`, which reads the SHARED disk cache
  first; a peer's fresh document satisfies the TTL and no network call
  happens. Per-machine fetch frequency is bounded by the 24 h TTL.
- **DA3 (measured, no change)**: registrant projection push serializes the
  ~87 KB frame per client at 0.18 ms each, capped at 20 pushes/s by the
  0.05 s debounce — five followers cost ~1.8% of the registrant loop. Not
  a bottleneck.
- **DA4 (verified, no change)**: daemon scan of 20 live records measures
  0.82 ms per 2 s cycle; stale-reap `_durable_projection` (~287 ms for a
  500-entry transcript) runs exactly once per stale transition (the stale
  record file is unlinked by `registry.scan`; the vanished-pid arm is
  gated by `entry.ended`).
- **DA5 (fixed)**: `cached_listing` now takes a best-effort cross-process
  fetch lease (lockfile with pid+token and 60 s expiry, `O_CREAT|O_EXCL`
  take, stale-steal, holder-only release). Five concurrent cold misses
  measured: exactly ONE live fetch, all five served. Degrades to
  fetch-anywhere on any coordination failure — a read-only cache dir can
  never block a session start.

## resume-picker-before.json / resume-picker-after.json

The `/resume` picker reach fix (uncapping the row list, plus the scan, index
and search-tier work that makes uncapping affordable). Captured on this machine
(macOS/APFS) against `origin/main` @ `fa62d097` (before) and the same tree with
the fixes (after), on a synthetic three-month store built by
`scripts/bench_resume_picker_store.py`: 2,700 user sessions and 29,000 subagent
sessions, 31,700 directories, 584 MB. That 10.6:1 subagent-to-user ratio is
what the real store shows, and it is the ratio that drives the scan cost.

```sh
.venv/bin/python scripts/bench_resume_picker_store.py /tmp/synth-store 2700 29000
PYTHONPATH=. .venv/bin/python scripts/bench_resume_picker.py /tmp/synth-store after --json out.json
```

| Stage | Before | After |
| --- | --- | --- |
| `recent_sessions`, uncapped | 1,073 ms | **314 ms** |
| `recent_session_rows`, uncapped | 1,256 ms | **409 ms** |
| `build_index`, warm cache | 62 ms | 60 ms |
| `build_index` after the daemon's narrow call | **768 ms** | **94 ms** |
| exact search, steady keystroke | 3.4 ms | **0.8 ms** |
| soft search, first keystroke | 161 ms | 154 ms (now deferred; see below) |
| **picker open, total** | **1,355 ms** | **494 ms** |

The operator's real store (230 user / 2,451 subagent), same command, is
`picker open 102 ms -> 34 ms` while listing 30 MORE rows than before.

Reading the table:

- The **scan** win is the origin verdict cache (`resume.ORIGIN_CACHE_NAME`).
  The listing must read and parse every `origin.json` that exists — existence
  alone must never be read as "subagent" — and the markers that exist are the
  subagent ones, so that rule costs one file read per subagent directory:
  1,127 ms over 31,700 dirs, of which 639 ms is reads and only 17 ms parsing.
  Skipping the read for unmarked directories alone saves ~8% and cannot fix it.
  Memoising the parsed verdict on the marker's own `(mtime, size)` is sound
  because the marker is written once at directory creation and never rewritten.
- The **`build_index`** win is cache preservation. Before, any narrow caller
  (the mobile daemon asks for 200 or 100 ids) pruned the on-disk index to its
  own ids and the next picker open re-digested the whole store. After, a
  wide → narrow → wide sequence re-digests **zero** transcripts, verified by
  counting `digest_transcript` calls, not by timing.
- The **soft-search** row is unchanged per call and that is the point: the tier
  is now deferred behind the exact tier, so it does not run at all for a query
  the cheap tiers already answer with a screenful. Its cost on the operator's
  real 2,681-digest / 10.04 MB corpus is 324 ms and 95 MB resident, which is
  what used to land on the first character typed once the picker was uncapped.

Cold-cache numbers are reported separately and never averaged into the warm
ones: the first scan after the cache is deleted is 1.6-5.2 s on this store,
dominated by reading 29,000 marker files, and it writes the cache as it goes.

## chrome-paint-before.json / chrome-paint-after.json

`scripts/bench_tui_chrome_paint.py` measures what one running subagent page
costs per spinner tick, and what an idle session writes to its terminal focused
versus blurred. Run:

```sh
env -u NO_COLOR TERM=xterm-256color \
  .venv/bin/python scripts/bench_tui_chrome_paint.py --focus 8 --json out.json
```

Everything except the focus block is a load-invariant COUNT rather than a
duration, and deliberately so: the machine this was diagnosed on sat at loadavg
260-307 on 14 cores from unrelated harnesses, where a fixed work quantum held
its CPU time (81.6 -> 101.5 ms) while wall time inflated 4.6-7.8x. Wall time is
printed but is not a signal.

Captured against `origin/main` @ `3c79116c` (before) and the same tree with the
fixes (after), 50 driven ticks, 160-block transcript, one running child.

| Per spinner tick | Before | After |
| --- | --- | --- |
| `messages.Layout` posted | 4.54 | **0.02** |
| compositor reflows | 4.42 | **0.00** |
| `Screen._refresh_layout` | 4.42 | **0.00** |
| `messages.Update` posted | 27.98 | **8.82** |
| title rewrites | 4.60 | 2.56 |
| breadcrumb rewrites | 4.60 | **0.00** |
| rule rewrites | 4.60 | **0.00** |

Both columns are the committed artifacts in this directory
(`chrome-paint-before.json`, `chrome-paint-after.json`), captured at the same
tick count so the two are directly comparable. The harness names every metric
even when it is zero, so a column that reads 0.00 is a counted zero in the
artifact, not a missing key. Absolute counts drift a little between runs
because the 1 Hz job poll also refreshes the page; the columns that go to
exactly zero are the ones this work is about, and those are structural rather
than sampled.

The breadcrumb and the rule are pure functions of `_ancestors`/`_label` and of
width; neither can change on a spinner tick, and both were being rewritten with
byte-identical strings 12.5 times a second. The reflow column is the missing
`layout=False` on the breadcrumb, which negated the deliberate `layout=False`
its two siblings already carried — one defaulted `update` on the same tick
relayouts the same screen.

### Focus gating

Terminal bytes per second, splash up and a turn running (the shape of a session
in a window the user has tabbed away from), 8 s windows:

| | Before | After |
| --- | --- | --- |
| focused | 24,039 B/s | 24,109 B/s |
| blurred | 23,824 B/s | **862 B/s** |
| ratio | 1.01x | **28.0x** |

The before column is the diagnosis restated: on `origin/main` a real `AppBlur`
posted to the app changes nothing, because Textual's own `_on_app_blur` only
sets a flag and refreshes bindings. The focused row is unchanged by design —
this gates on focus, it does not slow down animation anyone can see.
