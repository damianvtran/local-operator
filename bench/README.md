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

## info-snapshot-before.json / info-snapshot-after.json

`scripts/bench_info_snapshot.py` measures one `/info` open — the
`collect_snapshot` call the TUI's `/info` screen and the desktop's
`GET /v1/desktop/info` both make, the probes underneath it, and the route itself
(through the real FastAPI app, and with `--http` through a real uvicorn daemon on
a loopback socket). Run:

```sh
.venv/bin/python scripts/bench_info_snapshot.py --runs 5 --json bench/info-snapshot-after.json
.venv/bin/python scripts/bench_info_snapshot.py --http --json /tmp/info-after.json  # adds a socket
.venv/bin/python scripts/bench_info_snapshot.py --no-route --json /tmp/info-quick.json
```

`--tree <checkout>` measures another checkout — `git worktree add --detach
/tmp/lo-base <sha>`, with its own venv — which is how the two columns below were
captured. The script prints the `local_operator/__init__.py` it resolved, checks
that the daemon it boots for `--http` imports that same tree, and writes that
path into each artifact: a benchmark that measured the wrong tree is worse than
none. The `check:` lines are verdicts on the measurement (the fixture's sessions
were listed and measured, the two readers agree, the route's response carried
the fixture), not on taste; the millisecond ceilings are applied only when the
machine is not oversubscribed (both load averages are in the artifact), while the
probe's own ceiling is applied either way, since the regression it guards is two
orders of magnitude deep.

The fixture is 12 spawned processes with real session records published into a
temp config root — a dozen live sessions is the reported case. The records are
re-published before every section, because a record past `HEARTBEAT_TIMEOUT_S`
(45 s) classifies as `wedged` and wedged rows are not probed at all: without that
refresh the later sections measure a read with an empty fleet in it, which looks
fast and means nothing. It is also why every route row reports how many fixture
rows came back in the response. `~/.local-operator` is never touched.
`--extra-pids` adds real long-lived session processes to the equivalence section,
so the two readers are compared over 170-500 MB footprints and not only over
parked interpreters. The process names in those rows are recorded REDACTED
(`Local Operator [session]` rather than the session id, a path's basename rather
than the path): the artifacts are committed, and a session id names one of the
operator's own sessions.

The two committed artifacts were captured before the redaction existed in the
script, and are the recorded runs with that transformation—and only that
one—applied to them, through the script's own `_process_label` / `_display_path`:
every measured field in them is byte-identical to the run that produced them
(verified field by field at capture time, since a re-run on this machine would
have measured a different load regime rather than the same one twice).

| | Before (`ddd213ae7`) | After |
| --- | --- | --- |
| `session_resource_usage`, 12 pids, median | 5,068 ms | **41.8 ms** |
| `collect_snapshot`, median of 5 | 5,758 ms | **221 ms** |
| `collect_snapshot`, first call in a cold process | 7,447 ms | 6,366 ms |
| `GET /v1/desktop/info`, median (real daemon) | 5,444 ms | **171 ms** |
| `GET /v1/desktop/info`, first request after boot | 6,065 ms | **907 ms** |
| fixture sessions with a footprint in the response | 0 / 12 | 12 / 12 |
| `proc_pid_rusage`, per pid | absent | 5.7 us (median; max 25 ms under this load) |
| `top -l1` dump, measured for reference | 9,532 ms | 15,017 ms |
| worst \|delta\| against the `top` MEM column (parked fixture pids) | not comparable | 0.005 % |

Both columns were captured minutes apart on this machine at **19x and 35x the
CPU count** (1-minute load 267 and 488 on 14 CPUs), which is the regime this
repo is worked in rather than a fault in either row: every wall time here is
inflated by whatever else the box was running, and the after column was measured
under the heavier load of the two.

### The same read, measured independently

An independent QA pass on the same head — its own fixtures, its own `ctypes`
binding, its own daemon, loads recorded per sample across this box's 29 → 639
range — reproduced the direction and reached the quieter regime this artifact
cannot:

| | base | head | load (base / head) |
| --- | --- | --- | --- |
| `session_resource_usage`, 12 parked pids, median | 1,090.5 ms | **14.0 ms** | 195 / 166 |
| `collect_snapshot`, 12 sessions, steady median | 1,164.8 ms | **12.5 ms** | 31.6 / 31.6 |
| `GET /v1/desktop/info`, first request | 2,387.0 ms | **148.6 ms** | 216 / 231 |
| `GET /v1/desktop/info`, steady median | 2,113.1 ms | **42.3 ms** | 216 / 231 |
| TUI `/info` row, one session's memory cell | `17 MB` (RSS, footprint lost) | **`8 MB`** (footprint) | — |

**So the before column is bimodal, and one ratio would describe a machine nobody
has.** At load ~195 the base answered correctly, about 1.1 s late. At load ≳470
the dump outran the module's own 5 s subprocess timeout — 13 of 15 raw samples
measured 5.01-11.61 s at load 470-640, and two came in under it at 1.81 s and
2.37 s — and the read was pinned at ~5.02-5.23 s with no footprint at all. The
speedup is therefore ~93x where the base was still answering (12.5 ms against
1,164.8 ms of `collect_snapshot`) and ~26x against the loaded base on this
artifact's own columns (221 ms against 5,758 ms); the dump itself ranges 8 ms to
17 s here on the same command, which is scheduling rather than work.

What the rows say, in order:

- The **read** is the whole story. `session_resource_usage` went from 5,068 ms to
  41.8 ms because the macOS footprint no longer comes from one `top -l1` dump of
  the entire process table: `proc_pid_rusage` answers per pid in microseconds
  (5.7 us median) and the batched `ps` for RSS (25.9 ms) is now the only
  subprocess on the path. The dump survives for a MISSING READER rather than for
  any unknown pid: only pids that still exist and that the direct read could not
  answer — another account's process, which returns `EPERM`, and which
  `/usr/bin/top` can read because it is setuid root — or every pid on a host
  where libproc did not load, are sent to it. A pid that is GONE earns nothing:
  no reader can answer it, so the dump would spend its whole sampling interval
  and return nothing, and that is the module's documented normal case (a session
  dying between the registry scan and the read). Measured through the module's
  own runner on **twelve live pids plus one reaped pid**: **50.6 / 69.2 /
  140.2 ms** with the gate, against **1,259.6 / 1,334.3 / 1,403.9 ms** with it
  removed — one whole-system dump per read, whose ceiling is the runner's own 5 s
  timeout, which is the coarse part of the range (the same command returned in
  8 ms on an idle box and past the timeout under load). A pid the kernel answers
  with a footprint of zero (a zombie) is likewise left unknown: `0` is what the
  payload would carry where `info/model.py` requires the sentinel (`null`), and
  what `lop sessions` would print in its FOOTPRINT column where `—` is the
  unknown. The TUI's memory cell is a third rule —
  `format_bytes(footprint_bytes or rss_bytes)`, and a zombie's RSS is 0 too —
  and renders `0 MB` on this and every earlier release, so that surface is left
  as it was. The zombie spends no dump either. The reference row above is what
  the dump costs when a pid really does need it.
- **`collect_snapshot`'s first call in a cold process is unchanged**, and that is
  the honest reading rather than a regression: a bare interpreter pays the whole
  `/info` import graph plus the agent/config metadata scans (~1.4-2.1 s on this
  machine in a calmer window), and at 35x load that is 6.4 s. It is not what the
  desktop pays, because the backend daemon has the application imported before it
  answers anything — the row that describes the real first open is the route's
  first request, 907 ms on the same loaded box.
- **The before column lost the FOOTPRINT — silently, for every session, and only
  the footprint.** Its 0 / 12 row is not a rendering difference: the dump outran
  the module's 5 s subprocess timeout (5,068 ms of that read *is* the timeout),
  so `session_resource_usage` degraded to "unknown". What a reader SAW depends on
  the surface, and it is worth being exact about which: `lop sessions` prints `—`
  in its FOOTPRINT column and the desktop payload carries `footprint_bytes: null`,
  while the TUI row renders `format_bytes(footprint_bytes or rss_bytes)` and
  therefore fell back to **RSS** — QA captured a degraded base frame reading
  `17 MB` where the footprint was ~8.5 MB, against the head's `8 MB`. The RSS
  column survived throughout. The after column answers for all 12 sessions, for
  the quantity the column claims.
- The **equivalence** rows are what make this a speed change and not a different
  number, and they are stated against the reader this change REPLACES — `top`'s
  MEM column — rather than against a third one: `/usr/bin/footprint` reports
  1.8 % less for the same pid (884,736 B against 901,312 B read together here),
  which makes it a differently-defined reader rather than a tie-breaker. On the
  same parked pids `ri_phys_footprint` agrees with the `top` MEM column to within
  0.005 % (both report the kernel's phys-footprint accounting; `top` prints three
  significant digits, which is the entire difference). The
  structural proof that the hardcoded struct offset is the right field is the
  `rusage layout` check — `ri_resident_size` sits at offset 64 and equals `ps`
  RSS byte for byte on every parked fixture pid (worst delta 0 bytes), so the
  footprint at 72 is where the v1/v2 layout puts it. On the running sessions
  (`--extra-pids`) the two readers differ by up to 4.2 %, which is the session
  allocating between `top`'s sampling instant and the direct read, and is why
  that column is reported but not gated.
- **Overlap is not worth a thread pool here.** The `blocks` section times each
  collector on its own: the ceiling such a pool could buy is the sum of the
  independent ones minus the longest (69.5 ms at 35x load, 26.8-43.7 ms in
  calmer windows), and two of the five blocks cannot overlap anyway because
  `process` and `agents` are built from the sessions block's output. That is why
  the read is still serial.
