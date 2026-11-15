# CI unit duration weights

The unit matrix has **six** shards and retains its **38-minute** job timeout,
coverage commands, complete file selection, serial-control receipts and watchdogs.
Only scheduling weights and the number of partitions change. More shards spend
additional fleet capacity. The 2026-11-13 refresh below (re-weighed over both
Python legs on 2026-11-14) projects every shard
under 38/1.15 = 33.0 minutes of job wall; that projection is the margin claim,
and the PR gate run of the refresh is its first post-change observation.

## 2026-11-14 regeneration: #3941's runs, max over both Python legs

Two changes landed in parallel: #3941 (2026-11-13, below) refreshed the weights
from three six-shard `main` runs using the 3.12 leg only, and #3843 (2026-11-12,
below) taught the generator to weigh each file by the **max of the per-leg p75**
over every `--python` leg it is given. Converging #3843 onto `main` regenerated
`tests/durations.json` with #3843's generator from #3941's same three runs, both
legs (the command under "Reproduce"); nothing was hand-edited.

Checks on that regeneration (all on the merged tree):

- Run with `--python 3.12` alone, the converged generator reproduces #3941's
  committed manifest **byte-identically** (`cmp`), so the only change is the
  second leg.
- Every committed weight equals `max(p75 3.12, p75 3.13)` for its file (0
  mismatches over 2,312 files); no weight fell below #3941's. 349 files take
  their weight from the 3.13 leg. `test_attach_frame_size.py` stays 2337.1 s
  (3.12; its 3.13 p75 is 1342.4 s).
- `samples_per_file` counts (run, leg) observations: 2,297 files have all 6,
  13 have 4, 2 have 2. A second regeneration was byte-identical.
- `python scripts/shard_tests.py --report`: 2,334 files, 22 unmeasured
  (fallback 31.6 s), 106.5 test-min per shard (#3941's manifest: 105.4–105.5).

Both LPT assignments of the merged tree, rescored with each input run's
measured per-file totals per leg (in-sample; unobserved files contribute 0).
Job wall uses #3941's 3.12 conversion, test-min × 0.306 + 1.10; it was measured
on 3.12 jobs only, so the 3.13 walls are indicative:

| Run | Leg | #3941 assignment, test-min (job wall) | Converged assignment, test-min (job wall) |
|---|---|---|---|
| 43461090162 | 3.12 | 94.3 / 96.2 / 88.7 / 91.8 / 84.6 / 81.7 (30.0 / 30.5 / 28.2 / 29.2 / 27.0 / 26.1) | 94.9 / 93.6 / 86.1 / 91.8 / 82.2 / 88.7 (30.1 / 29.7 / 27.4 / 29.2 / 26.3 / 28.2) |
| 43426228826 | 3.12 | 77.5 / 93.8 / 99.1 / 95.9 / 97.4 / 94.4 (24.8 / 29.8 / 31.4 / 30.4 / 30.9 / 30.0) | 77.5 / 95.1 / 99.9 / 94.9 / 97.1 / 93.5 (24.8 / 30.2 / 31.7 / 30.2 / 30.8 / 29.7) |
| 43329220708 | 3.12 | 94.2 / 81.6 / 87.8 / 87.7 / 83.6 / 95.6 (29.9 / 26.1 / 28.0 / 27.9 / 26.7 / 30.3) | 95.5 / 83.9 / 86.4 / 88.1 / 84.7 / 91.8 (30.3 / 26.8 / 27.5 / 28.1 / 27.0 / 29.2) |
| 43461090162 | 3.13 | 71.4 / 86.1 / 79.6 / 88.4 / 84.1 / 83.1 (22.9 / 27.5 / 25.5 / 28.1 / 26.8 / 26.5) | 72.3 / 85.6 / 79.0 / 86.0 / 82.9 / 86.8 (23.2 / 27.3 / 25.3 / 27.4 / 26.5 / 27.7) |
| 43426228826 | 3.13 | 64.9 / 73.1 / 74.8 / 72.8 / 70.2 / 70.6 (21.0 / 23.5 / 24.0 / 23.4 / 22.6 / 22.7) | 64.9 / 73.1 / 73.1 / 73.3 / 69.1 / 72.9 (21.0 / 23.5 / 23.5 / 23.5 / 22.3 / 23.4) |
| 43329220708 | 3.13 | 72.8 / 85.5 / 79.8 / 83.2 / 71.6 / 76.9 (23.4 / 27.3 / 25.5 / 26.6 / 23.0 / 24.6) | 75.5 / 84.8 / 77.9 / 81.3 / 69.3 / 80.9 (24.2 / 27.1 / 24.9 / 26.0 / 22.3 / 25.8) |

In these three runs 3.13 was the faster leg overall, so the max-over-legs
weights change little: the worst shard of any row moves from 31.4 to 31.7
minutes projected, inside the 33.0-minute bar. The method matters when a file
is slower on 3.13, as `test_attach_frame_size.py` was in #3843's sample; it
plans for whichever leg is slower per file instead of trusting 3.12.

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

## #3843 record (2026-11-12): why a file weighs the max over both Python legs

This section records the input that introduced the max-over-legs method. Its
manifest was superseded on 2026-11-14 (above); the method and its rationale stand.

**Why.** One partition serves both matrix legs, but #3284 weighed files with
3.12 data from three PR runs of 11-01. By 11-11 shard `test (3.12, 0)` sat at the
cap (walls 38:09 cancelled with every test passed, 34:21, 35:27; PR #3719
diagnostic). The largest drift is `tests/unit/session/test_attach_frame_size.py`,
weighted 1315.7 s while these six runs measured it at 1165-2170 s on 3.12 and
1207-2491 s on 3.13 (p75 2071.1 s and 2180.8 s). 261 current files were also
absent from the old manifest and ran at the 30.0 s fallback.

**Method.** Six successful `main` full-suite runs (all twelve shard jobs of
attempt 1 green), both legs of each. Each leg takes its own inclusive p75 of
per-run file totals (as before); a file weighs the **max of the two legs'
p75**, so whichever leg is slower for that file is the one the partition plans
for. Pooling both legs into one p75 would let the faster leg dilute the slower.
`samples_per_file` counts (run, leg) observations: 2,237 files have all 12, and
the 22 with fewer are files that were added or removed during the window.

| Run | Attempt | Event | Head | Workflow blob |
|---|---:|---|---|---|
| 43040596560 | 1 | push | `d756665e377df9b5500eeb348d2b5246aa487b91` | `18002e9d72f97a79f8adf5f8d1d1373ba7939391` |
| 42978914635 | 1 | schedule | `ec4b40fa656a8000524c1234b6c8e28206f86a4b` | `ada0d8f35f1a3473e25a479f574802e9577c0cee` |
| 42932909817 | 1 | push | `966b5954a794204e861a3a055e109bdd6c6aecb9` | `5fe5313ec8ec24d827db89da9cace528c3bfaf2b` |
| 42862349942 | 1 | push | `e457f8424e0c32a76566ddd1bb7be5a4d5c2cabb` | `37c237dc40f124d1d97c0a8ff0320d08a7685074` |
| 42852373405 | 1 | push | `6f2cee87a4d11d5888b2e8dd21163003456b6a76` | `37c237dc40f124d1d97c0a8ff0320d08a7685074` |
| 42831360242 | 1 | push | `c9ad434a03b6684ef6ef73646f1a75bb75cc0846` | `37c237dc40f124d1d97c0a8ff0320d08a7685074` |

**Projection, both legs, on the 2,272-file tree at `dee55f697`.** The table scores
each manifest's LPT assignment three ways: by the manifest itself (what
`shard_tests.py --report` prints), by each leg's measured p75, and by the worst
single run of that leg. Values are test-minutes. The wall column turns the
largest shard into minutes using the lowest test-seconds-per-wall-second seen in
the ordinary pytest step across all 72 shard jobs of these runs (3.23; median
3.41), plus the largest non-pytest job time seen (81 s). It is a projection, not
a measurement.

| Assignment | Scored by | s0 | s1 | s2 | s3 | s4 | s5 | max | wall of max | vs 38:00 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| before (#3284) | manifest | 106.3 | 106.3 | 106.3 | 106.3 | 106.3 | 106.3 | 106.3 | 34.3 | +3.7 |
| before (#3284) | p75 3.12 | 106.6 | 83.2 | 99.3 | 101.3 | 96.7 | 96.0 | 106.6 | 34.4 | +3.6 |
| before (#3284) | worst run 3.12 | 111.5 | 97.4 | 105.7 | 101.2 | 96.1 | 97.1 | 111.5 | 35.9 | +2.1 |
| before (#3284) | p75 3.13 | 108.8 | 100.7 | 107.6 | 98.4 | 95.6 | 93.6 | 108.8 | 35.1 | +2.9 |
| before (#3284) | worst run 3.13 | 112.6 | 102.2 | 108.5 | 98.3 | 97.8 | 101.2 | 112.6 | 36.2 | +1.8 |
| after | manifest | 105.4 | 105.4 | 105.4 | 105.4 | 105.4 | 105.4 | 105.4 | 34.0 | +4.0 |
| after | p75 3.12 | 100.0 | 94.0 | 95.1 | 97.0 | 98.2 | 98.9 | 100.0 | 32.3 | +5.7 |
| after | worst run 3.12 | 96.0 | 94.6 | 92.6 | 93.9 | 95.2 | 95.0 | 96.0 | 31.1 | +6.9 |
| after | p75 3.13 | 100.3 | 100.5 | 102.2 | 98.5 | 101.5 | 101.8 | 102.2 | 33.0 | +5.0 |
| after | worst run 3.13 | 98.2 | 99.2 | 101.5 | 94.7 | 98.6 | 107.8 | 107.8 | 34.7 | +3.3 |

Shard 0 holds `test_attach_frame_size.py` in both assignments. At the highest
3.12 cost the diagnostic saw for that file (2619 s), shard 0's 3.12 p75 load
becomes 115.7 test-min before (37.2 min projected) and 109.1 after (35.2 min).

**Limits.** These numbers are in-sample: the same runs supply the weights and
the scores. The suite is ~100 test-minutes per shard at about 3.2-3.4x
effective concurrency, so even a perfect balance leaves six shards at about 32
minutes. Reweighting removes the skew; it does not create more than about 4-7
minutes of margin. If CI keeps landing near the cap after this change, the next
lever is the per-test cost (the year-long state in `test_attach_frame_size.py`)
or a seventh shard, not the timeout.

## Reproduce the committed manifest

Run from the repository root with Python 3.11+ (the job-log timestamps carry
seven fractional digits, which 3.9's `datetime.fromisoformat` rejects), the
repository's existing PyYAML dependency and authenticated `gh` (no pytest run,
dependency install or build). The cache holds raw junit XML, ~45 MB per run and
leg, so point it outside a size-capped session scratchpad:

```sh
.venv/bin/python scripts/gen_test_durations.py --cache-dir /path/outside/scratchpad/timing-cache \
  --python 3.12 --python 3.13 \
  --ci-run 43461090162:1 --ci-run 43426228826:1 --ci-run 43329220708:1
```

The generator fetches attempt-specific jobs and downloads artifacts by immutable
ID. It requires a completed successful CI run and, for every `--python` leg
(default 3.12 alone), the **exact** complete shard matrix declared by its
immutable execution-source workflow, then binds
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
still run with the fallback (31.6 s since 2026-11-14; 31.0 s in #3941's 3.12-only
manifest); the manifest has 2,312 observed files (2,297 in all three runs, 13 in
two, 2 in one), and 22 files of the 2,334-file tree at the 2026-11-14 merge are
unobserved (7 of 2,319 at #3941's head). Source changes after this
snapshot can add more.

## #3941 record: source selection and limitations

The generator admits only completed, successful `full-suite.yml` runs from
`push`, `schedule` or `workflow_dispatch` (see `SHARD_WORKFLOWS`), so the inputs
are the three latest such runs with a complete six-shard 3.12 matrix, all
`push` to `main` (both their 3.12 and 3.13 legs feed the 2026-11-14 manifest):

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

## #3941 record: 3.12-only projections against the 38-minute cap

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

## #3941 record: measured per-shard wall and testcase totals (same run/job)

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
