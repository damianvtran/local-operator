# Step budget, wall clock, and where a step's time goes

Why the run limits are what they are, and what the per-step latency is made
of. Written after a 26-task Kimi K3 cohort spent 22.8 h of wall time and had
9 of 30 episodes truncated by a limit this harness invented.

## The standard is 500 steps and NO wall clock

OSWorld 2.0 bounds an episode by **model steps only**. There is no
time limit anywhere in the upstream harness:

- `lib_run_single.py` bounds the loop with `while not done and step_idx <
  max_steps`, and `step_idx += 1` happens once per `agent.predict()` call,
  outside the loop that executes that call's actions. One step = one model
  call, however many actions it carries.
- The official Claude script passes `--max_steps 500`; the paper's headline
  metric is binary completion "at 500 steps", with 150/300 as documented
  reduced budgets.
- `wrapt_timeout_decorator` is imported in `lib_run_single.py` but **no
  function in the file carries a decorator** — dead code inherited from
  WebArena. The `"Time limit exceeded"` string in `run.py` is the generic
  `except Exception` message, not a timeout path.
- The argparse default of `--max_steps 15` is a vestigial OSWorld-1.0 value;
  the documented run commands, not the defaults, carry the standard.

Our runner already counts steps the same way: `_steps_taken += 1` once per
action batch, with `_guest_actions` tracked separately. So `--max-steps 500`
is directly comparable to upstream.

### What the old 2400 s wall did

At the cohort's median pace a 40-minute wall bought **~87 steps**, against a
standard budget of 500 and reference agents that use 190–318. It was not
measuring capability, it was measuring who fit in 40 minutes:

| | n | mean partial |
|---|---|---|
| Hit the wall | 9 | **13.5%** |
| Finished in time | 17 | **28.8%** |

The truncated episodes were still working when they were cut off — 2–6 actions
per batch, zero errors, compaction running normally.

### What the limits are now

`--max-wall-s` defaults to **18000** (5 h) and exists only as a runaway guard;
it is sized so the 500-step budget always binds first. `OSWORLD_TTL_SECONDS`
defaults to **18900** — the wall plus 900 s of slack — because
`providers/aws.py` falls back to `DEFAULT_TTL_SECONDS = 7200` when no TTL is
supplied (the wall budget is not carried on the adapter wire), and an instance
reclaimed at 2 h kills an episode that the harness would have let run.

**Any wall-truncated episode is a deviation from the standard.** The runner
records `truncation_reason`; a non-zero wall-truncation rate belongs in any
reported score, because the benchmark specifies no time limit and a
time-limited number is not comparable to a leaderboard entry.

## Where a step's 27 s actually goes

Measured from `monotonic_ns` gaps across 64 evidence bundles / 22.8 h. Note
that all four model events are journalled *after* the provider returns, so
`observation -> model_request` is the provider call and the sub-second gaps
after it are just journalling.

| transition | share | median | what it is |
|---|---|---|---|
| `observation -> model_request` | 52.6% | 9.21 s | the provider generating |
| `usage_cost -> action_batch` | 25.1% | 5.21 s | guest execution + screenshot |
| `lifecycle_transition -> observation` | 8.1% | 91.6 s | per-episode startup, once |
| `observation -> context_compaction` | 7.0% | 8.91 s | **not compaction cost** — see below |

Three findings worth recording, because each one closed an avenue that looked
promising:

1. **Compaction is not a cost centre.** `context_compaction` is journalled
   after `decide()` returns, so on those steps it lands where `model_request`
   normally would; both gaps measure the same model call. All 268 records are
   `strategy="prune"` with `summary_artifact=None` — no summarising LLM call
   is ever made. Compaction steps are marginally *faster* than ordinary ones.
2. **Harness CPU on the model hot path is ~12 ms**, against a 9.21 s gap.
   Measured on real 600 KB frames: base64 2.45 ms, artifact verification
   1.89 ms, token estimation 0.34 ms warm, body building 0.43 ms. The
   memoised tokenizer and the `to_thread` around image bounding already work.
   There is nothing to win here.
3. **Frames on the wire are already bounded.** The archived native artifacts
   in a bundle are 1920×1080, which makes it look as though full-size frames
   are sent; the frame the model actually receives is 1280×720 at ~4 KB,
   produced by `bound_screen_frame` through the same
   `local_operator.imaging.bound_image_for_model` ladder the interactive
   session uses. Verified by driving `ObservationBuilder` directly.

So ~59% of the run is the provider generating tokens, and the largest
remaining harness-side lever is **output volume**, not harness CPU.

### The 3.0 s settle is deliberate and stays

`providers/aws.py` sleeps `action_delay_s = 3.0` after each batch. Upstream's
`sleep_after_execution` defaults to `0.0` and the official script does not pass
it, but the paper states "A 3 s pause is inserted after each action", so 3.0 s
is the documented behaviour rather than an accident. Cutting it would save
about a second per step and risks screenshotting a half-painted UI — a score
risk taken to chase a latency win that the wall-clock fix dwarfs. Not changed.
If it is ever revisited, ship it as its own run and watch the
`UNCHANGED_FRAMES_NOTE` rate.

## Known remaining inefficiency: the echoed observation_id

Every action carries `observation_id`, a 64-hex-char digest, and
`ActionBatch` requires each action's copy to equal the batch's. The runner
already injects the batch-level value from the current observation, so the
model's copies are never read as data — they exist to expose staleness.

Cost, measured: 43 tokens per action × 2.88 actions per batch = ~124 output
tokens per call, ~2.2 s of generation at the fitted rate, **1.56 h across the
cohort — 6.8% of total runtime**.

This is a real win but it changes a frozen protocol invariant with strict
validators, so it is deliberately NOT bundled with the budget fix. It needs
its own change and its own review round.
