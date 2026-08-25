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

Captured on this machine (macOS/APPS) against `origin/main` @ `f2a52b53`
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
