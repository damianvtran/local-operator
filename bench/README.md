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
