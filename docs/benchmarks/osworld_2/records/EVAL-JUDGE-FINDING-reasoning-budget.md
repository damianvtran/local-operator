# EVALUATOR-SIDE FINDING — a mandatory-reasoning judge cannot answer inside the checkers' token caps

Measured 2026-11-11 by replaying real judge payloads offline (no VM, no episode) through
the vendored `model_client.generate_text` with each checker's own prompt, system text and
`options`, and the checker's own parser. task_025: its own `_score_form_with_model`,
with the union JSON the episode wrote; task_003: its `_llm_judge` prompt rebuilt from the
checker source. 58 paid calls in total. Raw responses (content, reasoning, finish_reason,
usage) were kept in the investigating session's scratchpad.

## Trigger

`t30a-t025-20261111-222346` finished its agent work, then `score` raised
`Task025EvaluationError: 8843: no valid model judgment after 3 attempts … caused by
ValueError: expected exactly one complete JSON judgment object`. The episode is
`unscored`, not zero.

## Cause

The campaign judge `qwen/qwen3.8-max-0902` (OpenRouter → Alibaba) **always reasons, and
the reasoning counts against `max_completion_tokens`**. The checkers pass upstream's caps,
sized for upstream's default judge `gpt-5.2`, which does not reason at those caps:

- task_025, cap 1500: `finish_reason=length`, `reasoning_tokens=1500/1500`, content `""`
  (2/2 calls). With the cap raised to 8000 (measurement only, checker unchanged) the same
  payload judged correctly 2/2, using 3114 and 3869 reasoning tokens. So the cap, not
  the payload, is the defect.
- task_003, cap 5: `finish_reason=length`, `reasoning_tokens=5/5`, content `""`.
  `generate_text` returns `""` and the checker reads it as NO. That is a **silent** NO,
  every call, regardless of the image.

Disabling the reasoning is not available. `reasoning_effort=none` returns HTTP 400
`Reasoning is mandatory for this endpoint and cannot be disabled`, and
`reasoning_effort=low` is accepted but does not bound it (1500/1500 again). Nor can the
apparatus raise the cap. The worker environment is built from a closed allowlist
(`adapters/supervisor.py` `minimal_environment`), and only provider, model and base URL
are injectable judge settings (`vendor_bridge._ENV_INJECTABLE`).
`OSWORLD_EVAL_MODEL_MAX_OUTPUT_TOKENS` would lose to the checker's `options["max_tokens"]`
anyway (`model_client._build_config` picks options first).

Scope: of the 108 corpus tasks, 21 are judged according to the adapter's own
`requirements.is_judged`. Eighteen of those pass an explicit cap: 5, 10, 16, 64, 128, 200,
400, 1024, 1500, 2048 or 4096. The other three (007, 092, 093) pass no literal cap, so the
cap comes from a metric helper or `generate_text`'s 512 default. Fifteen of the 21 pass
images directly. Every small-cap task grades a silent NO under this judge. In the 20-task sample, task_003 (cap 5) and task_025
(cap 1500) are judged. Their sealed results:

| run | judge reached | judge outcome | decided the score? |
|---|---|---|---|
| t29a-t003 | yes, 2 calls | silent NO (cap 5) | no — `Transp: False` (alpha 0.4 vs 0.6) failed both halves independently |
| t30a-t003 | yes, 2 calls | silent NO (cap 5) | no — `Filter: False` failed both halves independently |
| t29a-t025 | no | all four result forms blank, model call skipped by the checker | no |
| t30a-t025 | yes | empty content ×3 → `Task025EvaluationError` | yes — unscored (`rpc-remote-error`) |

No other judge-parse failure appears among the 31 sealed results in `RESULTS.jsonl`.

## Candidate judges (non-Western providers only)

Reference for 8843: under the checker's own rules, three findings are correct —
`f1_13` wrong (passport typo), `f1_26` wrong (school ZIP 92037 vs 92093) and `c1_1[1]`
unexpectedly filled. That gives 14/17 = 0.8235. The uncapped qwen3.8-max reached the
same three findings on both calls. `=ref` marks an exact match. `r=` is reasoning tokens.
task_003's real call is the t30a rainy background; the control compares the golden image
with itself, so the answer must be YES.

| judge | task_003 cap 5 (agent image) | golden-vs-golden control | task_025 8843 cap 1500 |
|---|---|---|---|
| qwen/qwen3.8-max-0902 (current) | length/'' r=5 | — | length, r=1500: FAIL empty content; length, r=1500: FAIL empty content |
| qwen/qwen3.8-max-0902, reasoning_effort=low | — | — | length, r=1500: FAIL empty content |
| qwen/qwen3.7-plus | — | — | length, r=1500: FAIL empty content |
| qwen/qwen3.8-flash | — | — | length, r=1500: FAIL empty content |
| qwen/qwen3-235b-a22b-2507 | error 404 (no image endpoint) | — | stop, r=0: FAIL wrong category; stop, r=0: 0.8235 =ref; stop, r=0: FAIL wrong category; stop, r=0: FAIL wrong category; stop, r=0: FAIL wrong category |
| moonshotai/kimi-k2-0905 | error 404 (no image endpoint) | — | stop, r=0: FAIL wrong category; stop, r=0: 0.8824; stop, r=0: 0.8824 |
| qwen/qwen3-vl-30b-a3b-instruct | stop/'NO'; stop/'NO'; stop/'NO' | stop/'YES' | stop, r=0: FAIL wrong category; stop, r=0: FAIL wrong category; stop, r=0: FAIL wrong category |
| qwen/qwen3-vl-235b-a22b-instruct | stop/'NO'; stop/'NO'; stop/'NO' | stop/'YES' | stop, r=0: 0.6471; stop, r=0: 0.6471; stop, r=0: 0.6471 |
| deepseek-chat (api.deepseek.com) | stop/'NO'; stop/'NO'; stop/'NO' | stop/'YES' | stop, r=None: 0.7647; stop, r=None: 0.8824; stop, r=None: 0.8824 |
| deepseek-flash (api.deepseek.com, thinking default) | stop/'NO' r=180; stop/'NO' r=201; stop/'NO' r=220 | stop/'YES' r=143 | stop, r=8081: 0.8235 =ref; stop, r=3421: 0.8235 =ref; stop, r=2283: 0.8235 =ref; stop, r=1880: 0.8235 =ref; stop, r=2944: 0.8235 =ref |
| qwen/qwen3.8-max-0902, cap raised to 8000 (measurement only) | — | — | stop, r=3869: 0.8235 =ref; stop, r=3114: 0.8235 =ref |

Two other task_025 forms were built with the checker's own `_pdf_to_canonical_json` and
`_build_union_eval_json` from the real ground-truth PDFs, because t30a fetched no other
result PDF before it crashed: 1040NR filled exactly (reference 1.0), and 540NR with 3 wrong
values and 1 blank seeded (reference 66/70 = 0.9429). deepseek-flash, deepseek-chat and
qwen3-vl-235b-a22b-instruct each matched both exactly, with the right field IDs.

Readings:

- Text-only models (Kimi K2-0905, Qwen3-235B-2507) get a 404 on any image task, and the
  checker's bare `except` turns that into a silent NO. They are disqualified for the 15
  judged tasks that pass images.
- **deepseek-flash** is the only candidate that is correct on every row. On
  api.deepseek.com its reasoning does **not** count against `max_tokens`: it spent
  180–220 reasoning tokens under a cap of 5, and 1880–8081 under a cap of 1500, and still
  returned the full answer. It is the study route's own model, so for that route the judge
  is same-model, where the campaign previously recorded a deliberate cross-family judge.
- **qwen3-vl-235b-a22b-instruct** is deterministic (3/3 identical), multimodal and
  non-reasoning, but it is over-strict against task_025's explicit equivalence rules. It
  flags `F-1` vs `F-1 Student` and similar formatting differences, scoring 0.6471
  against the 0.8235 reference: a systematic deflation.
- **deepseek-chat** fits the caps by design, but on 8843 it missed `c1_1[1]` in 2 of 3
  calls (0.8824), so it is less accurate than deepseek-flash.

## Wiring

- An OpenRouter model (e.g. qwen3-vl-235b-a22b-instruct) needs only
  `JUDGE_MODEL=<id>` on `launch_episode_21.sh`, which passes it through to the runner
  unchanged. No new code.
- DeepSeek direct needs a runner variant (`RUNNER_SUBDIR`), not passthrough alone.
  `JUDGE_PROVIDER=openai_compatible` and `JUDGE_MODEL=deepseek-flash` already pass
  through. What is missing is an `OSWORLD_EVAL_MODEL_BASE_URL=https://api.deepseek.com`
  infra value (seeded, plus `--infra-ref benchmark_judge:OSWORLD_EVAL_MODEL_BASE_URL=…`).
  The adapter also requires the key as the secret named exactly
  `OSWORLD_EVAL_MODEL_API_KEY` (`adapter._JUDGE_KEY`, `vendor_bridge.JUDGE_KEY_ENV`), so
  `JUDGE_ENV_NAME` cannot point it at another row. The DeepSeek key lives in lop's login
  store (`auth.db`, provider `deepseek`), so a scratch-store-only alias is needed, in the
  same shape as `alias_fixture_gitlab_pat.py`. No adapter or checker change is needed.
