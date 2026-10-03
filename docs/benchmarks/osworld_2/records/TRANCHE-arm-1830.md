# TRANCHE — arm 1830 (harness 0.64.10 @ 302a061e5), frozen build, ten tasks × two runs

Owner: coder, lopdev. Purpose: the paired-run measurement on the build carrying all seven fixes —
a variance read on the same ten tasks, with every zero classified apparatus-vs-capability, and a
cost distribution to size the next wave. Companion records: `COST-FORENSICS-1830-task_003.md`,
`EVAL-JUDGE-FINDING-1830-task_003.md`, `ZERO-CLASSIFICATION-1830-009-013.md`.

## Arm identity

**1830** — harness **0.64.10** @ `302a061e5d3f29c952c99cb7109adf9358e85ca1` (all seven fixes verified
ancestor: 46eb2d5c docs, baf8dbf3 image budget, fbc2c858 action field, 5bfff4a6 input-refusal,
63e23894 newest-frame, cac531d6 bridge wedge, 2602e502 provider recovery).
`package_digest b993bb22ec814ca6490fbb038328ef2d06118564258bc602c7c3b6b043472d91`;
`workspace_digest 146c7da73a64234e9034f0e7d729f8ce63075599092dddd7a9fb39f4e039c4de`;
`release_digest a942b9602f82bbd2d6f25eb6632fbf912847c7f31e9c2bab864f521346354c47`; adapter wheel
sha256 `15cba89d…` (member-identical to arm 1796's wheel). Build record `venvs/1830/build-record.json`.

**Protocol.** Route `openrouter/qwen/qwen3.8-max-0902` primary, `deepseek/deepseek-flash` provider-failure
fallback, usd basis `deepseek/deepseek-v4.1-flash`; engagement **session** (`SESSION_WALL=18000`);
500 steps; `--max-usd 3.00`; wall 18000 s; TTL 18900 s; settle throughput; judge
`qwen/qwen3.8-max-0902`. One episode per launchd job. Engagement verified every run by the four-leg
check (launch line, live runner argv `/venvs/1830/`+`/build-1830/`+`--engagement session`, scratch
`mcp.json` episode-actions command, sealed `-session` dir + `arm=session`).

## The ten pairs

| task | r1 bin/ppm | r2 bin/ppm | flipped? | partial Δ | r1 $ | r2 $ |
|---|---|---|---|---|---|---|
| 001 | **1** / 1,000,000 | 0 / 777,778 | **YES ↓** | −222,222 | 1.67 | 5.33 |
| 003 | 0 / 0 | 0 / 0 | no (apparatus both) | 0 | 14.32 | 5.24 |
| 004 | 0 / 516,800 | 0 / 700,000 | no | +183,200 | 1.47 | 0.60 |
| 005 | 0 / 0 | 0 / 0 | no | 0 | 1.03 | **18.40** |
| 006 | 0 / 666,667 | 0 / 444,444 | no | −222,223 | 5.41 | 4.31 |
| 009 | 0 / 0 *(apparatus)* | 0 / 574,359 | no (r1 apparatus) | +574,359 | 1.68 | 2.41 |
| 010 | **1** / 1,000,000 | **1** / 1,000,000 | no | 0 | 2.25 | 0.57 |
| 013 | 0 / 0 | **1** / 1,000,000 | **YES ↑** | +1,000,000 | 0.56 | 0.44 |
| 016 | 0 / 830,435 | 0 / 778,261 | no | −52,174 | 6.10 | **24.72** |
| 017 | 0 / 142,857 | 0 / 142,857 | no (identical) | 0 | 0.76 | 0.70 |

**All ten pairs complete.**
**r1 (n=10): mean partial 415,676 · binary 2/10 · $35.26.**
**r2 (n=10): mean partial 541,770 · binary 2/10 · $62.70.**
**2 binary flips in 10 pairs (001 ↓, 013 ↑).** Partial deltas: 013 +1,000,000, 009 +574,359,
004 +183,200, 001 −222,222, 006 −222,223, 016 −52,174; stable: 003, 005, 010, 017 (017 byte-identical).

### The read

The **arm-level means are close and the per-task behaviour is not**: two of eight paired tasks flipped
their binary outcome, in opposite directions, on identical build/route/budgets. 013 went 0 → a full
solve; 001 went a full solve → 0/777,778. Five tasks moved only in partial; 003/005/010 were stable.

**This is the campaign's answer at ten pairs: the arm-level result is indistinguishable given its own
spread, and any single-run comparison against another arm or a published baseline has no power.** The
flip rate (2/8) is the number that matters, and it is high enough that a ten-task arm cannot carry a
capability claim.

## Cost distribution (the sizing input)

All 20 completed episodes, sorted: `0.44, 0.56, 0.57, 0.60, 0.70, 0.76, 1.03, 1.47, 1.67, 1.68, 2.25,
2.41, 4.31, 5.24, 5.33, 5.41, 6.10, 14.32, 18.40, 24.72` → **median $2.25, max $24.72, total $97.96.**
Note many runs are provider-priced on a subset of calls (e.g. 001 r1 on 62/144), so figures are
lower bounds where noted.

The tail is **prefix re-billing**, not output: 005 r1 $1.03 → 005 r2 **$18.40** (cache_write 6.71M vs
0.11M); 016 r1 $6.10 → 016 r2 **$24.72** (cache_write 9.00M); 003 r1 **$14.32** (cache_write 5.02M).
The tail is the single largest cost term and is **episode-dependent, not task-dependent** — 005 and 016
straddle it across their own two runs. Full forensics in `COST-FORENSICS-1830-task_003.md`.

## Fix-exercise status (what this arm did and did not measure)

State plainly: **the arm does not validate a fix that never fired.**

| fix | exercised? | evidence |
|---|---|---|
| #1796 input-refusal recovery | **NO** | no `data_inspection_failed` / input refusal in any record |
| #1828 bridge-wedge re-bind | **NO** | no `no observation to bind to` / `RpcRemoteError` |
| #1830 provider recovery (504 / media 400) | **NO** | provider failures seen were HTTP 400 timeouts handled by the older fallback chain (`model_change`), never a 504 idle timeout or a media-download 400 |
| #1805 newest-frame rung ladder | **YES** | rung-ladder messages present (task_003 ×12, task_001 ×4) |
| #1788 image-budget downscale | **YES** | `downscal` in task_001 |
| #1790 action-field (dropped sibling) | **partial** | task_006 logged `Invalid arguments: missing required argument 'actions'` (adjacent, not the dropped-sibling path) |
| #1768 prose-completion challenge | **YES** | `completion_challenged` in task_001 and task_003 |
| 46eb2d5c docs | n/a | — |

Signal scan covered the r1 records (and live notices); not an exhaustive pass over all 20 episodes.

## Zero classification (every zero labelled)

| zero | class | reason |
|---|---|---|
| 003 (both runs) | **apparatus** | evaluator judge provider `openrouter` is not a registered backend → `LLM: False` unconditionally. `EVAL-JUDGE-FINDING-1830-task_003.md`. Fixed by #1887. |
| 009 r1 | **apparatus** | evaluator matched a directory-listing tab whose URL equals the bare `tab_prefix`; `getJSON is not defined`. Proven racy by 009 r2 (`research-output.html` → `getJSON executed successfully` → 0.574359). Fixed by #1894. |
| 005 (both) | capability | completed cleanly, scored 0 |
| 013 r1 | capability/variance | checked+failed (both `state_gt.json` and `state_fetched.json` present, comparison ran); **013 r2 solved it (binary 1)**, so r1's zero is variance, not a fixed limit |
| 001 r2, 006, 004 | capability | scored partials / misses with the evaluator reading the right surface |

## Incidents

- **Disk-stop void:** the first 005/006/009 attempt was killed mid-flight on a peer disk alert (and its
  3 guests left billing); swept (`instance-terminated, schedule-deleted`), `VOID-DISK-STOP.txt` in each
  root, excluded from all aggregates. Relaunched cleanly. Cost write-off ~20 min each.
- **Fleet ENOSPC during 001 r1:** the sink hit `ENOSPC` (53 failed writes) but the run completed and
  scored **through** it (`record_incomplete: true`, labelled) — the #1715 sink-durability path working.
- **Floor holds:** most of the tranche ran under a ≥8–10 GiB floor; the last two pairs were held ~3 h at
  3–5 GiB.
- Hygiene throughout: per-wave sweeps `[]`, audit `[]` after every wave, all launchd registrations
  cleaned up, no processes left.

## What would be needed to claim parity with a published figure

**Nothing here does.** These are ten tasks of 108; OSWorld 2.0's leaderboard figures are full-suite
(Claude Opus 5 31.43% binary / 68.31% partial; MiniMax M3 4.6% / 22.3%). A domain-selected subset can
outscore a full-suite average by drawing easier tasks — `docs/benchmarks/osworld_2/MODEL_PILOT.md` says
so explicitly. A parity claim needs the full suite or a properly sampled subset reported with variance.
Protocol notes that travel with any comparison: the cheap-model baselines used **standard** tooling and
sit on the June-24 release; the frontier runs used **batch tool**; ours is the August-8 release. Our
metric is OSWorld's own evaluator return (`binary = 1 iff value == 1.0`), so we are not self-scoring,
and 500 steps with no wall clock is the standard (our 18000 s wall is a guard sized so the step budget
binds first).

## Open items

- **All ten pairs are sealed, swept and closed**; audit `[]`, no launchd registrations or processes left.
  Total provider spend for the 20 episodes ≈ **$97.96** (plus the probe and the disk-stop void wave).
- **Fixed-build arm** (next): re-run the same ten on the build carrying #1891 (cache breakpoint), #1887
  (judge guard), #1894 (tab-shadow un-score) — expect the cost tail to compress (fit walk was ~49% of
  spend; this arm's tail is $14.32 / $18.40 / $24.72) and 003/009 to stop being guaranteed zeros. **A null
  result is a real result**: if the tail does not move, that is the finding. Classify 003/009 zeros afresh;
  do not assume the merges fixed them.
