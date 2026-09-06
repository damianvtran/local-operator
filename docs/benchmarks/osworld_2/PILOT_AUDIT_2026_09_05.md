# Pilot audit: 2026-09-05

This is an apparatus incident and exploratory-run report, **not a leaderboard
score or a model ranking**. Original sealed outcomes must remain unchanged.
An internally `comparable`/`reportable` receipt does not certify compliance with
an external benchmark protocol.

## Observed runs

The inspected runs used the clean source checkout at `299ec72d8`, an isolated
Python 3.12.13 wheel installation, and adapter 0.1.2 with explicit `paste_text`
and ASCII-only `type`. The launch snapshot specified 150 action batches,
$3 per-episode model admission budget, 2400-second episode wall budget,
2700-second cloud lease, and three recent frames. Exact package, workspace,
and environment identities remain in each original bundle.
Do not confuse `harness_git_revision` with a raw Git SHA: this launcher hashes
its source checkout's HEAD (falling back to a version digest outside Git).
That field alone does not attest every byte of the installed harness wheel.

| Route via OpenRouter | Task | Binary | Model requests | Recorded model USD | Stop |
| --- | --- | ---: | ---: | ---: | --- |
| `meta/muse-spark-1.3` | 010 | 0 | 101 | 2.307911 | budget-cap |
| `google/gemini-3.8-flash` | 010 | 0 | 157 | 2.860948 | max-steps |
| `google/gemini-3.8-flash` | 080 | 0 | 80 | 1.041983 | repeated-batch |
| `google/gemini-3.8-flash` | 103 | unscored | 0 | 0 recorded | outer process killed during setup |

Costs are sums of `usage_cost.cost_microusd`, not retail-price estimates, and
exclude infrastructure. A budget reservation can stop before the allowance is
fully spent. `completed` is the runner lifecycle, not task success. The empty
outcome file for the interrupted task is not JSON and must not be converted to
score zero. Gemini's completed attempts cost $3.902931 in total; with no accepted
completion, cost per successful task and full-suite competitiveness are unknown.

Local raw evidence locations (not public redistribution of gated task assets):

- `~/worktrees/osworld/runs/batch-spark-m1-010/task_010/evidence/ep-0a52bce248bd/`
- `~/worktrees/osworld/runs/batch-gem38-3t/task_010/evidence/ep-290e1c3f86b7/`
- `~/worktrees/osworld/runs/batch-gem38-3t/task_080/evidence/ep-a3bd6e25062f/`
- `~/worktrees/osworld/runs/batch-gem38-3t/task_103/evidence/ep-ee26a47e0bf6/`

## Repeated input was mistaken for lack of progress

Gemini task 080 ended with four consecutive Right-arrow batches. Inspection of
stored screenshots shows the image viewer advancing through document pages;
its last two observations display `view_3.png` and `view_4.png` with different
content. The old `RepeatedBatchGuard` compared only action bodies (excluding
observation IDs). It stopped legitimate pagination even though observations
changed. This is a benchmark-neutral false positive, not evidence the model
was stuck at that point.

The last viewed screenshot is artifact
`4de4c87e80223993f58e34a05966a289c59571b047b29fc699d757c2bc90b1c7`.
The sealed `environment_step` records `truncation_reason: repeated-batch`.
A correction must require evidence of unchanged observable state, not merely
repeated input, and must inspect the post-action observation. Text progress,
multiple displays, and intervening waits matter too. Reuse the existing guard
boundary rather than introducing task IDs, benchmark answers, a second agent
loop, or an image-processing service. Pixel-identical state is a conservative
signal, not a proof of semantic stagnation; animations may defeat it. Existing
step, wall, and spend budgets remain the backstop.

Correcting this decision cannot retroactively turn the historical zero into a
success. An offline replay proves only the corrected stop decision. A fresh,
separately identified run is needed to measure task completion.

### Offline replay actually performed

The parent replay reconstructed guard-relevant state from retained observations
75–79 and the four corresponding ActionBatch artifacts, verifying each consumed
artifact's byte count and SHA-256. It used real `GuardInput`/`EpisodeTurn`
objects and the production guard, with the old guard loaded from base
`547136ee8` for comparison. No task setup, evaluator, VM, or model was called.
The observed sequence was not replaced with invented identical frames.

```text
observation sequence: 75, 76, 77, 78, 79
old RepeatedBatchGuard: truncate / repeated-batch
new RepeatedBatchGuard: continue / ok
new NoChangeGuard: continue / ok
```

Consumed frame digests, in order:

```text
3073980d95e5a6141465680c07f4d8498beec8a56dec6d7c4b62937fd0a5ea9f
8514cde4b0ce80942e30764f02ca73c79643b2019f83a601a4366028f1ca1376
45e377e4b494b6be5dcec048272e9557ca9a402891ec4562748c1d6958cf66d8
05218df81995a7dcd56816318f1a82f1e9552a3691529e2061b4277275ca6f7d
4de4c87e80223993f58e34a05966a289c59571b047b29fc699d757c2bc90b1c7
```

This is a real-data decision replay, not re-execution of the task. Synthetic
regression fixtures complement it with stationary loops, text-only progress,
changed secondary frames, missing observable state, and intervening waits.

## Memory and interpretation corrections

All 327 retained parseable accepted replies across these three scored attempts
have empty `public_observations` (98 Spark, 150 Gemini task 010, 79 Gemini task
080). Their 34 context-compaction events are not evidence that useful public
facts survived compaction. The memory mechanism being installed is different
from the model supplying facts to it. Do not claim proven efficiency gains from
these unsuccessful runs or silently turn private reasoning into public notes.

Terminal commands entered through permitted computer I/O are not inherently a
wrong application surface. The [OSWorld 2.0 paper](https://arxiv.org/html/2606.29537v1)
analyzes terminal use. This does not authorize direct host filesystem/API access
or exposing evaluator internals to the evaluated agent. The agent must remain
separate from the setup/evaluation controller. No hidden task solutions were
used to design the guard correction.

## Outer-timeout incident and future admission

A single tool invocation wrapped three sequential episodes in a 3600-second
outer timeout. Tasks 010 and 080 took approximately 2321 and 1168 seconds,
respectively; the third was killed during setup. Per-episode wall limits do not
bound the sum of setup, all episodes, scoring, rescue and cloud deletion.

Before future paid work:

1. Launch one episode per outer command unless a campaign supervisor explicitly
   budgets the entire lifecycle. Never wrap several 2400-second episode budgets
   in one 3600-second process timeout.
2. Leave time for setup, scoring, and cleanup; an outer SIGKILL is not graceful
   cancellation. Verify actual resource deletion rather than treating a cleanup
   request or a terminated local process as success.
3. If interrupted, inspect persisted rescue descriptors before any next launch.
   The existing same-selector rescue sweep requested termination and schedule
   deletion here. A follow-up AWS read confirmed the instance terminated, its
   attached volume absent, and the benchmark tag audit `[]` in `us-east-1`,
   account `212841448981`. Do not rewrite the interrupted bundle as complete.
4. Do not run independent batch scripts concurrently while their safety check
   requires a globally empty audit: another owned live episode will appear as
   dirty. Use sequential pilots until ownership-aware orchestration is verified.
5. Enforce the existing free-disk admission threshold (21 GiB for these pilots),
   not a lowered threshold to get a run started. The host fell below it during
   this audit; no further paid episodes were started.
6. Check costs, stop reasons and rendered progress after each pilot. Expand only
   after apparatus defects are resolved and completed-task evidence improves.

## Recovery and evidence retention

Earlier cleanup treated an absent remote ref as zero commits ahead of main and
deleted local branches/worktrees. That is not a merge check. Local recovery refs
now preserve `8afbba9fe`, `d173ef6cd`, and `6c8470285`; no unrelated work was
changed during this recovery. A missing ref or failed Git command means unknown,
not permission to delete. Verify local commit identity, the all-state PR record,
patch equivalence where squash merging applies, cleanliness including untracked
work, and active ownership before removing a worktree.

The native-X11 candidate `8afbba9fe` is unmerged and intentionally rejected:
retained records report silent Unicode loss with zero-delay xdotool. It is not
part of the installed runtime. The reviewed replacement is the explicit paste
path merged in [PR #651](https://github.com/damianvtran/local-operator/pull/651).
Do not revive the rejected candidate merely because its branch was recovered.

Some original native-input/capture diagnostic run directories were deleted.
Surviving scripts, hashes, audit summaries and published #651 evidence are not
a replacement for missing raw reports/PNGs; no claim of full raw retention is
made. The separately retained capture audit found no demonstrated production
screenshot defect. Do not repeat paid native experiments solely to reconstruct
lost evidence or change screenshot code without a current reproduction.
