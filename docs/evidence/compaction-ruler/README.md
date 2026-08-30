# Evidence: compaction token accounting on one ruler

Captured 2026-08-29 on macOS 25.6.0 against the real session transcript at
`~/.local-operator/sessions/bda7b76d34e0/transcript.jsonl` (read-only), which
contains ten real compaction passes across three models.

| File | What it shows |
|---|---|
| `slope_fit.txt` | The provider-vs-local token slope, fitted per model and per epoch — the measurement the whole PR rests on |
| `span_percentiles.txt` | Pooled active-task spans and what each candidate cap multiple would clip — the basis for choosing 5 |
| `cap_by_window.txt` | The preserve cap across the registry's context windows: why the capacity term must stay (round 1, blocker-1) |
| `fallback_reach.txt` | How often the receipt's no-shrink fallback fires on real passes: 0/10 (round 2, blocker-1) |
| `validate_fix.txt` | Receipt accuracy per pass, shipped formula vs proportional, against the provider's own next-reported context |
| `retention_real2.txt` | Active-task spans per pass and the retention the preserve-window cap allows, old cap vs new |
| `controlflow_real.txt` | Whether the three `tokens_after` consumers change any DECISION under the fix |
| `frames/before.svg`, `frames/after.svg` | The rendered receipt line, before and after |

All three tables are the verbatim stdout of the committed scripts beside them
and need no credential (they read a local transcript):

```sh
.venv/bin/python docs/evidence/compaction-ruler/slope_fit.py
.venv/bin/python docs/evidence/compaction-ruler/validate_fix.py
.venv/bin/python docs/evidence/compaction-ruler/retention_real2.py
.venv/bin/python docs/evidence/compaction-ruler/controlflow_real.py
.venv/bin/python docs/evidence/compaction-ruler/span_percentiles.py   # no transcript needed
.venv/bin/python docs/evidence/compaction-ruler/cap_by_window.py      # no transcript needed
.venv/bin/python docs/evidence/compaction-ruler/fallback_reach.py
env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
  docs/evidence/compaction-ruler/shot_compaction.py out.svg <repo-root>
```

## The defect: two comparisons each mixed two token rulers

The local estimator (`compaction/tokens.py`, cl100k_base) and a provider's
reported `context_tokens` are not the same scale. Fitting
`provider = a * local + b` per model over this session (`slope_fit.py`):

```
  anthropic/claude-opus-5            n= 32 slope= 1.685 inter=   24,341 err=   7,173 ratio p50=1.82
  anthropic/claude-opus-4-8          n= 24 slope= 1.622 inter=   49,623 err=   5,151 ratio p50=1.96
  openai/gpt-5.6-sol                 n= 20 slope= 1.036 inter=   32,758 err=   6,972 ratio p50=1.15
  zai/glm-5.3                        n= 11 slope= 1.019 inter=    8,854 err=     297 ratio p50=1.06
  openrouter/z-ai/glm-5.3            n=  5 slope= 1.030 inter=    5,835 err=      87 ratio p50=1.05
```

Anthropic sits at ~1.65 where OpenAI and GLM sit at ~1.03, in the same session
with the same tool schemas. That is provider-side tokenizer divergence on
code/JSON-dense content, not content the estimator fails to see.

**The slope is a per-model property, not a session constant**, which is why the
fix scales proportionally instead of baking in a number. The same fit run per
inter-compaction epoch is the identifiability check:

```
       epoch    n  slope  fiterr  models present
   1051-2123   15  1.634     653  HOMOGENEOUS: None/None x15
   2124-2929   15  0.806  18,999  HOMOGENEOUS: None/None x15
   2930-3998   15  1.054   9,913  MIXED(3): zai/glm-5.3 x7, openai/gpt-5.6-sol x6, xai/grok-4.6 x2
   3999-4660   15  0.919  47,197  MIXED(2): anthropic/claude-opus-4-8 x9, openai/gpt-5.6-sol x6
   4661-5439   15  1.384  44,716  MIXED(3): anthropic/claude-opus-4-8 x12, openai/gpt-5.6-sol x2, kimi/k3 x1
   5440-6865   14  0.913  39,031  MIXED(4): alibaba-token-plan/qwen3.8-max x4, openai/gpt-5.6-sol x4, ...
   6866-7979   14  1.860  71,026  MIXED(4): openrouter/z-ai/glm-5.3 x5, zai/glm-5.3 x4, anthropic/claude-opus-5 x3, ...
   7980-8982   14  1.622   1,415  HOMOGENEOUS: anthropic/claude-opus-5 x14
   8983-9954   15  1.638     419  HOMOGENEOUS: anthropic/claude-opus-5 x15
```

Every tight fit (419 / 653 / 1,415) is a single-model stretch; every loose one
(9,913 to 71,026) switches models mid-epoch, because one line cannot describe
two tokenizers. The intercept swings from -119,224 to +122,208 across those
mixed stretches, which is the concrete reason the provider's fixed overhead is
not a quantity this code could estimate and subtract.

**Provenance note.** An earlier draft of this PR quoted a narrow-window fit
(117 points at slope 1.728, mean error 241) and a "wire body tokenizes to 295k
against 304k" comparison. Both were dropped: the narrow-window fit is
collinear in (slope, intercept) over a 15% span, and the wire figure needs a
request capture not reproducible from the transcript. Every number now in the
code comments and in this README is the output of a script committed beside it.

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

**Why 5.** Pooling these ten spans with the seven recorded at
`cutpoint.task_boundary_floor` gives 17 measurements: p50 47.4k, p75 53.7k,
p90 125.9k, max 131.4k. The distribution is **bimodal** — thirteen spans under
54k, four between 113k and 132k — so the real question is where between the
clusters the bound sits. 5x the 20,000 default puts it at 100,000: about 2x
headroom over the longest ordinary task (53,732) and below the outlier cluster,
which is precisely the set of passes that retained 35-41% of history.

**The evidence does not separate 5 from 4, and the code says so.** 4x (80,000)
clips the same four pooled spans; the case for 5 is margin over the observed
ordinary maximum, not a different clipping outcome. 6x (120,000) *would* differ
— it lets the 113,835 span through, which is one of the passes this fix exists
to bound — so the multiple should not be raised without new evidence
(`span_percentiles.txt`):

```
candidate multiples against the pooled spans:
  3x -> cap  60,000  clips  4/17 [113835, 123400, 129660, 131376]
  4x -> cap  80,000  clips  4/17 [113835, 123400, 129660, 131376]
  5x -> cap 100,000  clips  4/17 [113835, 123400, 129660, 131376]
  6x -> cap 120,000  clips  3/17 [123400, 129660, 131376]  <- lets an OUTLIER through
  7x -> cap 140,000  clips  0/17 []  <- lets an OUTLIER through
```

An
earlier draft justified 5 as "the smallest multiple clearing a p90 of 99k";
that p90 does not reproduce from the cited spans (they give 78.8k interpolated
or 123.4k nearest-rank, and a p50 of 46.9k rather than 32k), so the argument
was replaced with the one above rather than restated.

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

## 4. The receipt never reports growth, and the fallback is rare

The proportional form assumed `history_after <= history_before`. That holds on
the text-model passes above but **not** on snapcompact, which replaces history
with verbatim edges plus archive text plus images the local ruler prices at a
flat `IMAGE_TOKEN_ESTIMATE`: the saving is real on the provider ruler while the
local estimate of the replacement can be larger. The ratio then exceeded 1,
multiplied an already ~1.7x provider total, and the receipt reported a
compaction that *grew* the context (measured 70,888 -> 111,594 on a real
over-threshold pass), at which point `compaction_receipt` drops both numbers.

The fix applies the ratio only when the pass shrank local history, and clamps
the result to `context_tokens` — the upper bound the old subtraction form had
for free.

`fallback_reach.txt` measures how often that no-shrink branch actually fires,
because an earlier revision of this PR claimed it was the path vision models
always take:

```
  pass  hist_before  hist_after  shrank   ratio  branch
  1050      246,989      52,729    True    4.7x  proportional
  ...
  6865      533,774      60,947    True    8.8x  proportional
  9954      318,084     173,835    True    1.8x  proportional

0/10 real snapcompact passes take the fallback; 10/10 get the full proportional receipt.
```

**That claim was wrong and is corrected in the code comment.** It came from
synthetic 10-70 turn fixtures. The archive's plain-text edges are sized by frame
shape (`HQ_EDGE_FRAMES`), not by how much history was removed — about 20,900
tokens at the shipped shape, which is 116% of an 18k history and 3.9% of a 534k
one. So a near-threshold toy history can come out larger on the local ruler
while a real pass never does. The fallback is a property of unusually small
histories, not of vision models.
