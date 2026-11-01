# CI unit duration weights

The unit matrix has **six** shards and retains its **38-minute** job timeout,
coverage commands, complete file selection, serial-control receipts and watchdogs.
Only scheduling weights and the number of partitions change. More shards spend
additional fleet capacity; six is authorized here, not a claim that any timeout
margin has been demonstrated.

## Reproduce the committed manifest

Run from the repository root with Python, the repository's existing PyYAML
dependency and authenticated `gh` (no pytest run, dependency install or build):

```sh
python scripts/gen_test_durations.py --cache-dir "$LOCAL_OPERATOR_SCRATCHPAD/timing-cache" \
  --ci-run 41295059480:1 --ci-run 41294499173:1 --ci-run 41294469202:1
```

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

For PR events, the Actions API's `head_sha` is the branch head, not the execution
merge. The generator reads the full identity SHA immediately following the
attempt job's actions/checkout group (`git log -1 --format=%H`), verifies that
merge has the run head as a parent, and fetches the workflow at that immutable
SHA. It never resolves today's mutable PR merge ref or main. For push events,
checkout must equal the recorded head. Source YAML is safely composed without
expression evaluation: duplicate keys, aliases, dynamic shard expressions and
matrix include/exclude fail closed. Only literal contiguous five/six matrices
are supported; broader workflow forms need an explicitly validated schema.

This depends on retained job logs and source content. Expired/unreadable logs,
missing/multiple checkout identities, wrong ancestry, or failed source fetches
abort; falling back to branch HEAD could certify the wrong matrix.

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
still run with the fallback; the regenerated manifest has 2,011 observed files
(2,008 observed in all three runs, three in one). On the implementation tree,
`team_bridge/test_reply_windows.py` and the new generator regression file are
unobserved and remain selected. Source changes after this snapshot can add more.

## Source selection and limitations

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

## Assignment projections — test-time, NOT wall

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

## Actual historical wall and testcase totals (same run/job)

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
