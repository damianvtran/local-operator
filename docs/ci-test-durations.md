# CI unit duration weights

The unit matrix has **seven** shards and retains its **38-minute** job timeout,
coverage commands, complete file selection, serial-control receipts and watchdogs.
This capacity experiment changes only partition count, not the committed weights
or selected set. More shards spend fleet capacity; seven is not a claim of
measured wall-time improvement, guaranteed timeout margin or a native-crash fix.

## Seventh-shard experiment: natural six-shard observations

Three strict-compatible natural post-PR3284 attempt-1 matrices completed with
all configured unit/coverage jobs successful. These observations justify trying
another partition while retaining the timeout, not extrapolating wall time from
summed test weights. Job wall and ordinary pytest-step wall are distinct:

| Run | Unit jobs | Max job wall min | Max ordinary step wall min | Margin to 38 min |
|---|---:|---:|---:|---:|
| 41324404334 (main) | 12 | 31.283 | 30.417 | 6.717 |
| 41326832044 (PR3292) | 6 | 27.183 | 26.550 | 10.817 |
| 41335594592 (PR3277) | 6 | 30.967 | 30.117 | 7.033 |

All three use deployed workflow blob `cafae6f6e88cbb71cf2f91766a40d7c389fb9619`,
the unchanged duration manifest and default checkout/source binding. Two of three
leave less than ten minutes of job margin. PR3290/run41327449686 is supplementary
only: its actual execution merge used six shards, but its branch-head workflow
and manifest predate PR3284 and fail the strict head/merge compatibility contract.
It is excluded from this qualifying denominator; premerge run41319197549 is also
not included. Do not weaken source binding to make additional samples qualify.

The unit Python-version expression gives one version on PRs and two on main
pushes, so six-to-seven adds one unit job per PR and two per main push. Coverage
and e2e axes are unchanged. Observe complete natural post-seven matrices before
claiming wall-time/margin improvement; no artificial workflow replay is needed.
The historical five/six calibration and projection tables below remain history,
not forecasts for this seven-shard experiment. The manifest is not regenerated.

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
literal contiguous five/six/seven matrices are supported. This contract depends on
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
success within 38 minutes was not a six-shard margin forecast. The six-shard
repair required at least three complete natural post-change matrices before a
seventh-shard decision; that evidence is recorded above. The same discipline
applies to the seventh-shard experiment: require actual post-change observations,
never raise the timeout or drop selected tests to manufacture margin. No extra
full CI workflow is dispatched to manufacture those observations.
