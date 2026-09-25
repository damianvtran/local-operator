# Step budget, wall clock, and where a step's time goes

Why the run limits are what they are, and what the per-step latency is made
of. Written after a 26-task Kimi K3 cohort spent 22.8 h of wall time and had
9 of 30 episodes truncated by a limit this harness invented.

## OSWorld v2.1 comparison protocol: 500 steps, no official wall cap

OSWorld v2.1 is the current recommended release. Its headline result is at
**500 decision steps** (the paper also reports 150/300-step results). The
upstream benchmark loop bounds an episode by model steps only; there is no
official wall-clock limit. One step is one model call, however many actions it
carries.

This checkout's ordinary `run_episode.py` defaults are 25 steps and the older
`osworld-v2-2026.08.08` release, so those defaults do not reproduce the current
headline protocol. For a comparable headline run, select the release and budget
explicitly with `--benchmark-release osworld-v2.1 --max-steps 500`.

The runner's five-hour `--max-wall-s` default is an operational watchdog, not an
OSWorld rule. Any episode truncated by it is non-comparable to an uncapped
benchmark result and must be disclosed as a protocol deviation. Do not confuse
an operational cloud lease/TTL with a benchmark wall-time limit.

OSWorld 2.0 also bounds an episode by **model steps only**. There is no
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
- The vendored adapter's upstream argparse default is 15 (a vestigial
  OSWorld-1.0 value); the Local Operator wrapper separately defaults to 25.
  Neither default substitutes for the current v2.1/500-step comparison command.

Our runner already counts steps the same way: `_steps_taken += 1` once per
action batch, with `_guest_actions` tracked separately. So `--max-steps 500`
is directly comparable to upstream.

### What the old 2400 s wall did

At the cohort's median pace of 27.4 s per model call a 40-minute wall bought
**~88 steps**, against a standard budget of 500 and reference agents that use
190–318. It was not measuring capability, it was measuring who fit in 40
minutes.

Four of the 26 tasks have more than one scoring run, so the exact means depend
on which run is picked. The split is reported as a range because the
CONCLUSION is what is robust, not the decimals:

| pick | capped | finished | capped mean | finished mean |
|---|---|---|---|---|
| first scoring run | 9 | 17 | 13.5% | 28.8% |
| last scoring run | 8 | 18 | 15.2% | 30.0% |
| best partial | 8 | 18 | 15.2% | 30.6% |

Under every selection the episodes that ran out of time score roughly half of
those that finished. The truncated ones were still working when they were cut
off — 2–6 actions per batch, zero errors, compaction running normally.

Two caveats on attribution, both of which a reader needs in order to
reproduce the table rather than conclude it is wrong:

- The grouping rule is **elapsed >= 2280 s**, i.e. within 5% of the 2400 s
  ceiling, not elapsed >= 2400 s. The margin matters: `task_098` finished at
  2331 s, under the ceiling but plainly pressed against it, and it belongs
  with the truncated group. Using the ceiling itself gives 8/18 and
  9.7% vs 29.7% -- the same conclusion, different decimals.
- These episodes are recorded with `truncation_reason=budget-cap`, not a
  wall-specific reason. No episode carries a reason naming the wall, so "hit
  the wall" is inferred from elapsed time, not read from the journal.

### What the limits are now

`--max-wall-s` currently defaults to **18000** (5 h) in
`scripts/run_episode.py` and exists only as a local runaway guard; it is not an
official OSWorld budget. A run that reaches this watchdog must be reported as
wall-truncated and is not directly comparable to an official uncapped score.
For an official-style comparison, keep the step budget explicit and report any
watchdog truncation rather than treating it as an official stopping condition.

The cloud lease is now **derived from the wall** in `run_episode.py`
(`_ensure_lease_outlasts_wall`, wall + 900 s) rather than left to the
provider's fixed fallback. This is load-bearing: raising the wall alone would
have inverted the previous ordering, because
`providers/aws.py` falls back to `DEFAULT_TTL_SECONDS = 7200` when no TTL is
supplied — the wall budget is not carried on the adapter wire. With an 1800 s
wall that fallback was harmless; with an 18000 s wall an episode past two
hours would die on a TERMINATED INSTANCE rather than at a budget boundary,
which loses the episode instead of ending it and reads as an infrastructure
fault rather than a deliberate cap. An explicit `OSWORLD_TTL_SECONDS` still
wins.

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

### Settle policy: throughput default, paper opt-in

The runner exposes `OSWORLD_ACTION_SETTLE_POLICY` as an episode-scoped,
manifest-sealed infra value. `throughput` is the default and preserves the
existing behavior: ordered guest runs are grouped where possible and one 3.0 s
settle is paid per model batch. This is the policy behind the historical cohort
measurements above.

For paper-style pacing, opt in with
`--infra OSWORLD_ACTION_SETTLE_POLICY=paper`. It applies the same fixed 3.0 s
settle after each compiled nonterminal semantic action. A `WAIT` first receives
its requested duration, then its separate 3.0 s settle before the next action.
A compound compiled `ScrollAction` remains one semantic action. This can add
several seconds and provider executes to a multi-action batch, so paper-mode
latency/cost is expected to be higher than throughput mode.

The adapter still performs exactly one readback/observation after the whole
model batch. Upstream OSWorld's `env.step` can return/log intermediate states
between primitive actions, so this is settle-timing parity, **not exact
trajectory or observation parity** with upstream. Keep the policy sealed with
results and do not compare throughput and paper runs as if their timing
apparatus were identical.

The 3.0 s value follows the paper's stated pause; policy selects its frequency,
not its duration. Upstream's separate `sleep_after_execution` argument defaults
to 0.0 and is not the source of this adapter's settle.

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
