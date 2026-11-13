# CI unit duration weights

The unit matrix has **six** shards and retains its **38-minute** job timeout,
coverage commands, complete file selection, serial-control receipts and watchdogs.
Only scheduling weights and the number of partitions change. More shards spend
additional fleet capacity. The 2026-11-13 refresh below projects every shard
under 38/1.15 = 33.0 minutes of job wall; that projection is the margin claim,
and the PR gate run of the refresh is its first post-change observation.

## Why the weights were refreshed (2026-11-13)

Shard 0 was cut at the 38-minute ceiling with every test green: #3864's gate run
43470745098 (job 132063792182, run_bounded rc=0 at 38.18 min, then cancelled) and
#3898's gate run 43495430540 (job 132157719392, rc=0 at 38.07 min). Neither cut
was runner variance or the serial controls (their step runs 12–15 s). The prior
manifest, measured on **five-shard** runs from 2026-11-01, planned every shard at
110.2 test-minutes, yet the 3.12 JUnit artifacts of both cut runs measured
shard 0 at 122.1 and 124.5 test-minutes against 61.9–116.2 for the others, at a
steady 0.29–0.30 minutes of pytest wall per test-minute on every shard. The
measured weights had gone stale: `tests/unit/session/test_attach_frame_size.py`
was committed at 1315.7 s and measured 2185.6 s and 2415.6 s in those runs (it
changed in #3236, #3251, #3269 and #3697 after the old sample). Unmeasured files
were NOT the cause: they take a pessimistic p90 fallback that over-budgets them.
The fix is fresh weights from complete six-shard runs, keeping the 38-minute cap.

## Reproduce the committed manifest

Run from the repository root with Python, the repository's existing PyYAML
dependency and authenticated `gh` (no pytest run, dependency install or build):

```sh
python scripts/gen_test_durations.py --cache-dir "$LOCAL_OPERATOR_SCRATCHPAD/timing-cache" \
  --ci-run 43461090162:1 --ci-run 43426228826:1 --ci-run 43329220708:1
```

The raw XML cache is ~190 MB; point `--cache-dir` outside a size-capped
scratchpad (for example a worktree's ignored `.rig/`).

The generator fetches attempt-specific jobs and downloads artifacts by immutable
ID. It requires a completed successful CI run and the **exact** complete Python
3.12 shard matrix declared by its immutable execution-source workflow, then binds
artifact creation time to each job's execution interval. A six-shard source with
only jobs 0–4 is incomplete, not a historical five-shard run. Missing or ambiguous
artifacts are fatal.
New artifacts include `-attempt-${{ github.run_attempt }}` in their names; old
names remain readable only with the interval check. Raw XML is stored under the
cache; run/head/attempt/job/artifact IDs and per-file observation counts are
committed in `tests/durations.json` under `_provenance`.

### Supported execution-source contract

A code checkout is not automatically the revision that created a workflow job.
Online ingestion supports only `pull_request` and `push`, with this repository's
first, unmodified `actions/checkout@v7` step (no custom name, inputs, condition,
repository/ref, or subsequent checkout). Unsupported forms reject rather than
attempt a general workflow interpretation.

The attempt's Jobs API must identify exactly one successful checkout action at
step 2 with timestamps. Only server-timestamped log lines inside that step's
window are eligible, including its action marker and identity command. Test-only
checkout-shaped output is not source evidence. The logged checkout command must
name the **default execution ref**: `refs/remotes/pull/<PR>/merge` for a PR, or the
normal remote branch checkout for push. Explicit alternate checkout refs reject.
PR number comes from authenticated run metadata; closed PRs may disappear there,
so a unique same-repository association at `commits/<immutable-head>/pulls` is
also supported. Missing or ambiguous associations reject; log text never supplies
the PR identity.

For PR events, API `head_sha` is the branch head, not the execution merge. Verify
merge ancestry, fetch workflow bytes at that immutable merge, and additionally
require them to be byte-identical (including blob ID) to workflow content fetched
independently at the immutable API head. Divergent head/merge workflows are
unsupported—even if valid—and fail closed. Thus printed checkout text cannot
substitute a five-shard source for the authenticated six-shard head schema.
For push, the default branch checkout SHA must equal the recorded execution head.
In both cases, the fetched source must actually declare the supported default
checkout action. No mutable main/PR merge lookup is used.

Source YAML is safely composed without expression evaluation: duplicate keys,
aliases, dynamic shard expressions and matrix include/exclude fail closed. Only
literal contiguous five/six matrices are supported. This contract depends on
retained job logs and source content; expired/unreadable logs, absent action/step
metadata, missing/multiple checkout identities, wrong ancestry, incompatible
head/merge source, or failed API fetches abort without weakening the contract.

| Run | Executed workflow SHA | Workflow blob |
|---|---|---|
| 43461090162 | `be6f138ee3a3ceedbacaf02aa9b527d056ad3d78` | `dfc0a8dc5f8f38bb1250894fa184dea4d79ccef5` |
| 43426228826 | `abf20525b2cc47b986f9e14d6a398ae2ed0c086a` | `dfc0a8dc5f8f38bb1250894fa184dea4d79ccef5` |
| 43329220708 | `73dc41570c94ff22db15c4a6721f458614de2998` | `dfc0a8dc5f8f38bb1250894fa184dea4d79ccef5` |

These source identities were verified separately for every admitted run. Their
identical workflow blob (`full-suite.yml`) declares six shards; no observed-job
count is used to infer the expected matrix.

Only exact known ordinary and shard-zero serial-control XML members are read.
Classnames resolve against real source files, trimming nested test classes;
parameterized testcase names do not change the module. Empty classname module
skips resolve through `name`. No old-manifest lookup is used. An unresolved
mapping or invalid/negative/nonfinite duration aborts before writing.

Per-file testcase seconds are summed **within each run** before taking inclusive
p75 (linear interpolation; with three samples, halfway between the two largest).
Files absent from another run are not zero-cost observations. Each positive
weight and the p90 fallback has a one-second floor. Unobserved current files
still run with the fallback (31.0 s); the regenerated manifest has 2,312
observed files (2,297 in all three runs, 13 in two, 2 in one), and 7 files of
the 2,319-file tree at this head are unobserved. Source changes after this
snapshot can add more.

## Source selection and limitations

The generator admits only completed, successful `full-suite.yml` runs from
`push`, `schedule` or `workflow_dispatch` (see `SHARD_WORKFLOWS`), so the inputs
are the three latest such runs with a complete six-shard 3.12 matrix, all
`push` to `main`:

| Run | Attempt | Created | Head |
|---|---:|---|---|
| 43461090162 | 1 | 2026-11-13 16:16Z | `be6f138ee3a3ceedbacaf02aa9b527d056ad3d78` |
| 43426228826 | 1 | 2026-11-13 12:33Z | `abf20525b2cc47b986f9e14d6a398ae2ed0c086a` |
| 43329220708 | 1 | 2026-11-12 23:29Z | `73dc41570c94ff22db15c4a6721f458614de2998` |

The two cut runs (43470745098, 43495430540) are labelled-PR runs whose shard 0
was cancelled, so they are inadmissible as inputs; they serve below as the
out-of-sample check instead. Most `push` runs that day ended `cancelled` (pending
runs superseded in the `full-suite-main` concurrency group) and are likewise
excluded. The three inputs span
about seventeen hours and differ slightly in test content.

## Projections against the 38-minute cap

`python scripts/shard_tests.py --report` at this head (2,319 files):

| Manifest | Unmeasured (fallback) | Projected test-minutes per shard 0…5 |
|---|---|---|
| prior (five-shard, 2026-11-01) | 308 (30.0 s) | 110.2 ×6 |
| refreshed (six-shard, 2026-11-13) | 7 (31.0 s) | 104.2 ×6 |

Manifest test-minutes are p75 sums, not measurements. To test the refresh,
both LPT assignments are rescored with each run's **measured** per-file totals
(ordinary plus serial-control XML; files a run did not observe contribute 0 and
are counted). This is a rescoring of today's tree, not that run's historical
assignment. Job wall is projected as test-minutes × 0.306 + 1.10, the worst
pytest-step wall per test-minute and the worst job-minus-step overhead measured
across all 30 3.12 shard jobs of these five runs (range 0.286–0.306 and
0.62–1.10 minutes, from the Jobs API step 8 `Run test shard with coverage`):

| Run | Role | Unobserved | Prior assignment, test-min (job wall) | Refreshed assignment, test-min (job wall) |
|---|---|---:|---|---|
| 43470745098 | cut, out-of-sample | 6 | 115.7 / 68.5 / 95.5 / 79.1 / 91.2 / 79.0 (36.5 / 22.1 / 30.3 / 25.3 / 29.0 / 25.3) | 92.6 / 87.5 / 85.1 / 89.6 / 83.4 / 90.6 (29.4 / 27.9 / 27.1 / 28.5 / 26.6 / 28.8) |
| 43495430540 | cut, out-of-sample | 3 | 115.8 / 98.1 / 116.2 / 74.5 / 79.5 / 85.6 (36.5 / 31.1 / 36.7 / 23.9 / 25.4 / 27.3) | 99.5 / 96.7 / 99.9 / 90.1 / 91.8 / 91.5 (31.5 / 30.7 / 31.7 / 28.7 / 29.2 / 29.1) |
| 43461090162 | input, in-sample | 7 | 117.5 / 70.8 / 115.3 / 96.4 / 69.2 / 68.0 (37.1 / 22.8 / 36.4 / 30.6 / 22.3 / 21.9) | 95.1 / 97.3 / 88.5 / 92.3 / 83.3 / 80.8 (30.2 / 30.9 / 28.2 / 29.3 / 26.6 / 25.8) |
| 43426228826 | input, in-sample | 9 | 73.1 / 97.9 / 118.8 / 75.0 / 92.4 / 100.8 (23.5 / 31.1 / 37.5 / 24.1 / 29.4 / 31.9) | 78.2 / 94.2 / 99.1 / 95.2 / 97.0 / 94.4 (25.0 / 29.9 / 31.4 / 30.2 / 30.8 / 30.0) |
| 43329220708 | input, in-sample | 22 | 110.4 / 80.4 / 72.0 / 80.4 / 86.2 / 101.1 (34.9 / 25.7 / 23.1 / 25.7 / 27.5 / 32.0) | 94.5 / 81.6 / 87.0 / 87.9 / 83.1 / 96.3 (30.0 / 26.1 / 27.7 / 28.0 / 26.5 / 30.6) |

Shard 0's worst projected job wall under the refreshed weights is 31.5 minutes
(run 43495430540, out-of-sample), 38/31.5 = 1.21× the cap, and the worst shard
of any row is 31.7 (shard 2, same run); both are inside the 33.0-minute (15%)
bar. The p75 manifest load itself (104.2 test-minutes) projects to 33.0 at the
worst ratio and 31.8 at the median 0.295, so the bar holds even if a shard
lands its full pessimistic budget. These are projections from five runs; if a
post-change shard exceeds 33.0, regenerate from fresh runs first, and report
evidence for a seventh shard rather than raising the timeout or dropping tests.

## Measured per-shard wall and testcase totals (same run/job)

Job and pytest-step walls come from the attempt-specific GitHub Jobs API
(`started_at` to `completed_at`); testcase minutes are the shard's ordinary
JUnit XML (shard 0's serial-control XML excluded, as its step is separate).
Each run's own historical assignment, under the prior manifest:

| Run | Shard | Conclusion | Job wall min | Pytest step wall min | Testcase min | Wall per test-min |
|---|---:|---|---:|---:|---:|---:|
| 43470745098 | 0 | cancelled | 38.27 | 37.17 | 122.1 | 0.304 |
| 43470745098 | 1 | success | 19.03 | 18.02 | 61.9 | 0.291 |
| 43470745098 | 2 | success | 27.88 | 26.92 | 91.2 | 0.295 |
| 43470745098 | 3 | success | 24.20 | 23.27 | 81.0 | 0.287 |
| 43470745098 | 4 | success | 28.43 | 27.65 | 93.6 | 0.295 |
| 43470745098 | 5 | success | 24.42 | 23.55 | 79.1 | 0.298 |
| 43495430540 | 0 | cancelled | 38.13 | 37.07 | 124.5 | 0.298 |
| 43495430540 | 1 | success | 28.73 | 27.90 | 96.0 | 0.291 |
| 43495430540 | 2 | success | 35.02 | 34.22 | 116.2 | 0.294 |
| 43495430540 | 3 | failure | 20.83 | 20.13 | 70.4 | 0.286 |
| 43495430540 | 4 | success | 24.13 | 23.18 | 79.6 | 0.291 |
| 43495430540 | 5 | failure | 24.63 | 23.85 | 82.8 | 0.288 |
| 43461090162 | 0 | success | 37.62 | 36.60 | 126.7 | 0.289 |
| 43461090162 | 1 | success | 19.07 | 18.45 | 62.4 | 0.296 |
| 43461090162 | 2 | success | 35.38 | 34.65 | 117.7 | 0.294 |
| 43461090162 | 3 | success | 29.73 | 29.12 | 100.6 | 0.289 |
| 43461090162 | 4 | success | 19.45 | 18.83 | 64.6 | 0.291 |
| 43461090162 | 5 | success | 20.02 | 19.38 | 65.2 | 0.297 |
| 43426228826 | 0 | success | 22.07 | 21.32 | 69.7 | 0.306 |
| 43426228826 | 1 | success | 28.85 | 28.07 | 96.2 | 0.292 |
| 43426228826 | 2 | success | 36.12 | 35.33 | 123.6 | 0.286 |
| 43426228826 | 3 | success | 19.92 | 19.08 | 66.3 | 0.288 |
| 43426228826 | 4 | success | 29.27 | 28.50 | 96.5 | 0.295 |
| 43426228826 | 5 | success | 31.87 | 31.22 | 105.7 | 0.295 |
| 43329220708 | 0 | success | 35.85 | 34.88 | 116.7 | 0.299 |
| 43329220708 | 1 | success | 23.47 | 22.53 | 75.2 | 0.300 |
| 43329220708 | 2 | success | 20.33 | 19.67 | 64.5 | 0.305 |
| 43329220708 | 3 | success | 23.52 | 22.88 | 79.6 | 0.288 |
| 43329220708 | 4 | success | 26.87 | 26.17 | 88.7 | 0.295 |
| 43329220708 | 5 | success | 31.42 | 30.65 | 105.6 | 0.290 |

The two `failure` rows in 43495430540 each record one failing test
(`failures="1"` in their JUnit XML), not a timeout; the shards ran to
completion, so their timings are used only for the ratio and overhead.
