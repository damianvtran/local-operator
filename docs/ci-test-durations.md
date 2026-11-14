# CI unit duration weights

The unit matrix has **six** shards and retains its **38-minute** job timeout,
coverage commands, complete file selection, serial-control receipts and watchdogs.
Only scheduling weights and the number of partitions change. More shards spend
additional fleet capacity; six is authorized here, not a claim that any timeout
margin has been demonstrated.

## 2026-11-12 regeneration: the max over both Python legs

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

Below: how to reproduce the committed manifest, the source contract the
generator enforces, and the record of the #3284 (2026-11-01) regeneration this
one replaced (its "Source selection" and projection tables describe that
earlier input, not the committed one).

## Reproduce the committed manifest

Run from the repository root with Python 3.11+ (the job-log timestamps carry
seven fractional digits, which 3.9's `datetime.fromisoformat` rejects), the
repository's existing PyYAML dependency and authenticated `gh` (no pytest run,
dependency install or build). The cache holds raw junit XML, ~45 MB per run and
leg, so point it outside a size-capped session scratchpad:

```sh
.venv/bin/python scripts/gen_test_durations.py --cache-dir /path/outside/scratchpad/timing-cache \
  --python 3.12 --python 3.13 \
  --ci-run 43040596560:1 --ci-run 42978914635:1 --ci-run 42932909817:1 \
  --ci-run 42862349942:1 --ci-run 42852373405:1 --ci-run 42831360242:1
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

| Run | Executed merge workflow SHA | Workflow blob |
|---|---|---|
| 41295059480 | `88239c28f29018a7ea73b1e1dd569d429498865f` | `5212079c36d92854bba386bfdb1b18975e14a6f7` |
| 41294499173 | `1ea0054d488d8acfb8bf54aa26859234fbd23cb4` | `5212079c36d92854bba386bfdb1b18975e14a6f7` |
| 41294469202 | `e88665a2ee82ee964dedba8b64a26b4a161f606b` | `5212079c36d92854bba386bfdb1b18975e14a6f7` |

These source identities were verified separately for every admitted run. Their
identical workflow blob declares five shards; no observed-job count is used to
infer the expected matrix.

Only exact known ordinary and shard-zero serial-control XML members are read.
Classnames resolve against real source files, trimming nested test classes;
parameterized testcase names do not change the module. Empty classname module
skips resolve through `name`. No old-manifest lookup is used. An unresolved
mapping or invalid/negative/nonfinite duration aborts before writing.

Per-file testcase seconds are summed **within each run** before taking inclusive
p75 (linear interpolation; with three samples, halfway between the two largest).
Files absent from another run are not zero-cost observations. Each positive
weight and the p90 fallback has a one-second floor. Unobserved current files
still run with the fallback; the #3284 manifest had 2,011 observed files
(2,008 observed in all three runs, three in one). On the implementation tree,
`team_bridge/test_reply_windows.py` and the new generator regression file are
unobserved and remain selected. Source changes after this snapshot can add more.

## #3284 record: source selection and limitations

All three inputs are complete PR runs from the same eight-minute creation window
on 2026-11-01, not independent days or post-change observations:

| Run | Attempt | Head |
|---|---:|---|
| 41295059480 | 1 | `4e3ec5e5f1fb4e52daadeaaaecd8744ac41d3e50` |
| 41294499173 | 1 | `67b0f1d6e8698ef76d62e3bea28c9cbc8a408bf1` |
| 41294469202 | 1 | `243b033d81f78930812e63574a765f9bc950c7f7` |

`41295058879` is deliberately excluded: its run-level status is success only
after attempt 2, while shard 1 of attempt 1 was cancelled. Combining its latest
shard-1 artifact with four attempt-1 artifacts would masquerade as one complete
run. Other incomplete/cancelled inputs were not used. The independent extraction
initially admitted this mixed run; fresh attempt-specific API reads exposed it.

These runs share runner/job configuration but differ slightly in test content.
This is within-window calibration, not proof of out-of-sample wall performance.

## #3284 record: assignment projections — test-time, NOT wall

Both rows below partition the SAME 2,013-file implementation tree with LPT. Old
means the prior manifest with five shards; new means p75 with six. Values are
summed manifest test-seconds, **not predicted elapsed seconds**:

| Assignment | Files per shard | Projected test-seconds per shard |
|---|---|---|
| old5 | 402, 403, 402, 403, 403 | 5747.5, 5747.5, 5746.5, 5747.5, 5747.5 |
| new6 | 335, 335, 335, 336, 336, 336 | 5081.2, 5081.2, 5080.2, 5081.2, 5081.2, 5081.2 |

Rescoring these fixed assignments with each input run's measured file totals
is an **in-sample test-time projection**, not that run's actual historical
assignment. Unobserved current files contribute no measured value in this table;
they are counted explicitly and are NOT removed from the real partition:

| Source run | Unobserved current files | Old five test-seconds | New six test-seconds |
|---|---:|---|---|
| 41295059480 | 4 | 4548.7, 6561.3, 6102.4, 4879.8, 5921.9 | 4852.1, 4508.5, 4827.6, 4727.0, 4468.4, 4630.5 |
| 41294499173 | 5 | 6108.8, 5446.9, 4343.1, 6153.1, 6386.3 | 4632.4, 4994.4, 4552.5, 4579.8, 4752.9, 4926.1 |
| 41294469202 | 3 | 4141.1, 5334.7, 6085.9, 5468.7, 4232.5 | 4182.9, 4055.1, 4364.7, 4326.9, 4484.2, 3849.2 |

## #3284 record: actual historical wall and testcase totals (same run/job)

The following are the original five-shard jobs, not six-shard observations.
Job and pytest-step walls come from the attempt-specific GitHub Jobs API
(`started_at` to `completed_at`). The ordinary step is `Run test shard with
coverage`. Testcase sums come directly from each exact artifact; shard zero
includes its separate serial-control XML (so it is not identical in scope to
its ordinary pytest step). No worker count is inferred and no sum is divided
by a guessed concurrency.

| Run | Shard | Job wall s | Ordinary pytest step wall s | Summed testcase s |
|---|---:|---:|---:|---:|
| 41295059480 | 0 | 1233 | 1193 | 4154.799 |
| 41295059480 | 1 | 1821 | 1784 | 6411.532 |
| 41295059480 | 2 | 1918 | 1883 | 6505.235 |
| 41295059480 | 3 | 1437 | 1404 | 4870.455 |
| 41295059480 | 4 | 1828 | 1792 | 6072.083 |
| 41294499173 | 0 | 1822 | 1773 | 6146.084 |
| 41294499173 | 1 | 1551 | 1517 | 5406.706 |
| 41294499173 | 2 | 1245 | 1202 | 4181.199 |
| 41294499173 | 3 | 1848 | 1806 | 6266.607 |
| 41294499173 | 4 | 1912 | 1871 | 6437.536 |
| 41294469202 | 0 | 1178 | 1136 | 3936.288 |
| 41294469202 | 1 | 1489 | 1463 | 5066.853 |
| 41294469202 | 2 | 1953 | 1918 | 6637.043 |
| 41294469202 | 3 | 1671 | 1636 | 5613.851 |
| 41294469202 | 4 | 1228 | 1194 | 4008.939 |

The gap cannot be labelled collection overhead from these data: parallel worker
imbalance, idle time, setup/teardown and collection may all contribute. Historical
success within 38 minutes is not a six-shard margin forecast. Observe at least
three complete post-change CI runs before assessing a ten-minute ceiling margin;
if six shards are insufficient, report evidence for a seventh-shard decision
rather than raising the timeout or dropping selected tests. No extra full CI
workflow is dispatched to manufacture those observations.
