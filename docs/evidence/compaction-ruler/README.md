# Evidence: compaction token accounting on one ruler

Captured 2026-08-29 on macOS 25.6.0 against the real session transcript at
`~/.local-operator/sessions/bda7b76d34e0/transcript.jsonl` (read-only), which
contains ten real compaction passes across three models.

| File | What it shows |
|---|---|
| `validate_fix.txt` | Receipt accuracy per pass, shipped formula vs proportional, against the provider's own next-reported context |
| `retention_real2.txt` | Active-task spans per pass and the retention the preserve-window cap allows, old cap vs new |
| `controlflow_real.txt` | Whether the three `tokens_after` consumers change any DECISION under the fix |
| `frames/before.svg`, `frames/after.svg` | The rendered receipt line, before and after |

All three tables are the verbatim stdout of the committed scripts beside them
and need no credential (they read a local transcript):

```sh
.venv/bin/python docs/evidence/compaction-ruler/validate_fix.py
.venv/bin/python docs/evidence/compaction-ruler/retention_real2.py
.venv/bin/python docs/evidence/compaction-ruler/controlflow_real.py
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
  docs/evidence/compaction-ruler/shot_compaction.py out.svg <repo-root>
```

## The defect: two comparisons each mixed two token rulers

The local estimator (`compaction/tokens.py`, cl100k_base) and a provider's
reported `context_tokens` are not the same scale. Fitting
`provider = a * local + b` inside contiguous model-homogeneous runs of this
session:

```
run          model                      n    slope   intercept   mean fit err
7740-7975    anthropic/claude-opus-5   117   1.728     48,341        241
8990-9400    anthropic/claude-opus-5   202   1.649     32,731        320
4670-5430    anthropic/claude-opus-4-8 259   1.633     43,193      1,354
4670-5430    openai/gpt-5.6-sol         50   1.030     37,065         59
```

A mean fit error of 241 tokens over 117 points is structural, not noise, and
the OpenAI control in the same session with the same tool schemas sits at
1.03 where Anthropic sits at ~1.7. It is genuine provider-side tokenizer
divergence on code/JSON-dense content, not content the estimator fails to see:
the wire body tokenizes to 295k here against a provider-reported 304k.

## 1. The receipt understated every pass by ~140k tokens

`tokens_after = context_tokens - (tokens_before - history_after)` subtracts a
LOCAL saving from a PROVIDER total, which assumes slope 1. Scaling the
provider figure by the ratio the history actually shrank keeps both ends on
the provider's ruler and needs neither ruler's constants:

```
  pass   provB REAL_after |  current      err | proposed     err
  9954  546458     311220 |   418953  +107733 |   327409  +16189
mean abs error  current=  139,406   proposed=   14,616
```

Pass 9954 is the screenshot the operator reported: `546.5k → 419.0k
(23% smaller)` for a pass whose true after-figure was 311.2k, a ~43%
reduction. MAE across all ten passes falls **139,406 → 14,616**.

A tuned-slope variant (`provB - slope * (lb - la)`) was measured and is
strictly worse even at its best value (19,512 at slope 1.75) while carrying a
constant that rots on the next retokenization.

## 2. The preserve-window cap had the same bug, in the cut path

`Session._advisor_floor_cap` returned `max(keep_recent, threshold // 2)` =
300,000 on a 1M window. It caps `task_boundary_floor`, whose span is summed in
LOCAL units — while `threshold` is a PROVIDER-unit number.

```
  pass hist_before  task_span | keep_old keep_new | retain_old retain_new
  5439     338,613    129,660 |  129,660  100,000 |      38.3%      29.5%
  7979     322,812    113,835 |  113,835  100,000 |      35.3%      31.0%
  9954     318,084    131,376 |  131,376  100,000 |      41.3%      31.4%
```

Three of ten passes retained 35-41% of history where the other seven retained
4-19%. Expressing the cap in `keep_recent` units (`* 5` = 100,000 at the
20,000 default) bounds those three and leaves the other seven untouched.

**Why 5 and not 4.** The multiple is sized by the measured active-task spans
recorded at `cutpoint.task_boundary_floor` (p50 32k, p90 99k over an earlier
7-pass session): 4x is 80,000, which clips that p90 and starts severing the
long agentic turns the floor exists to protect; 5x is the smallest whole
multiple clearing it. On this later 10-pass session both 4x and 5x clip the
same three outlier spans, so the choice costs nothing here and buys headroom
on the p90 case.

## 3. No decision changes — only the printed number

`tokens_after` has three consumers. `_held_context_tokens` is a display
carrier with no branch on the value; the other two are gates, checked against
the real `resolve_threshold_tokens` / `RECOVERY_BAND` / `cleared_headroom`:

```
(2) auto-continue decision flips: 0/10
(3) advisor kill-switch flips:    0/10
```

The proposed after-figure is always <= the shipped one on this data, so both
gates can only move toward "the pass helped": the fix can neither newly
disable the advisor nor newly suppress a continuation.

Nothing in this change touches `compaction_context_tokens`,
`resolve_threshold_tokens`, or `EDGE_WINDOW_FRACTION`, so **when** compaction
fires is unchanged.

## The rendered receipt

`shot_compaction.py` drives a real `Session` through a real `compact_now()`
with the provider's `context_tokens` pinned at the screenshot's 546,458, then
renders the resulting event through the real `OperatorApp` (which loads the
stylesheet). The frame moves only because the arithmetic moved.

```
before:  context compacted · 546.5k → 542.5k tokens (1% smaller), via summary
after:   context compacted · 546.5k → 223.7k tokens (59% smaller), via summary
```

Layout is identical between the two frames; the status band's context reading
follows the receipt down, which is the same figure rendered in a second place.
