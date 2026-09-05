# Matched-model development pilot

This is an exploratory protocol and run ledger, **not a leaderboard submission**.
The purpose is to separate model limitations from apparatus failures before paying
for a full 108-task evaluation. A domain-selected pilot cannot establish parity
with a published full-suite average, even if its percentage is higher.

## Reference selection

Primary sources checked during the September 2026 pilot:

| Model | OpenRouter ID | USD per million input / output tokens | OSWorld 2.0 reference and limitations |
| --- | --- | --- | --- |
| GPT-5.6 Luna | `openai/gpt-5.6-luna` | 0.20 / 1.20 | OpenAI reports 45.6; fetched release text does not establish the precise metric, release, reasoning, or repeat protocol. Do not label it binary completion. |
| MiniMax M3 | `minimax/minimax-m3` | 0.30 / 1.20 | Benchmark-team reference: 4.6% binary, 22.3% partial at 500 steps on the **June 24** release, standard tools, reasoning enabled. |
| Muse Spark 1.3 | `meta/muse-spark-1.3` | 1.25 / 4.25 | Existing local control. No exact OSWorld 2.0 reference established by this research. |
| GPT-5.6 Terra | `openai/gpt-5.6-terra` | 2.00 / 12.00 | Optional mid-price follow-up, not in the initial pilot. OpenAI reports 50.2 with the same unresolved protocol qualifications as Luna. |

Prices are catalog rates, not task costs. Cache discounts, service-tier routing,
long-context premiums, reasoning, retries, and compaction affect actual bills.
The pilot must report observed usage rather than price-table extrapolations alone.

Sources:

- [Official leaderboard data](https://osworld-v2.xlang.ai/static/data/leaderboard/official-results.json)
  separates binary accuracy, partial score, step budget, and benchmark release.
- [OSWorld 2.0 paper](https://arxiv.org/html/2606.29537v1) documents the original
  benchmark-team runs. MiniMax uses screenshot observations and PyAutoGUI actions;
  terminal/file workflows are discussed as solution styles, not automatically
  prohibited because the observation arrives as a screenshot.
- [OpenAI release](https://openai.com/es-ES/index/gpt-5-6/) and model documentation
  for [Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna.md) and
  [Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra.md).
- [MiniMax model card](https://huggingface.co/MiniMaxAI/MiniMax-M3).
- [OpenRouter catalog](https://openrouter.ai/api/v1/models).
- [Google evaluation methodology](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-8_flash_model_evaluation.pdf):
  Gemini 3.8 Flash's reported 59.0 is **partial score**, maximum of three runs,
  500 steps, screenshots and batched tools, **before the August 8 patch**. It is
  not a 59% completion target for this apparatus.

For current-release context, the official data checked here lists Opus 5 max at
31.43% binary / 68.31% partial and GPT-5.6 Sol max at 27.34% / 62.72%, on the
August release, full scope, 500 steps with batched tools. Those figures are not
substitutes for a matched-model control on our pilot tasks.

## Frozen initial design

The controller recorded `trend-pilot-a-plan.json` before the first model result.
The selected tasks are `task_010`, `task_080`, and `task_103`: respectively an
Office/mail workflow, a local spreadsheet/PDF workflow, and a CAD/image workflow.
Selection was based on application diversity and available prerequisites, not
observed success or evaluator answers. These three tasks are not a random sample.
Previously explored `task_001` is excluded from clean trend evidence.

Initial arms: Luna, MiniMax M3, and Muse Spark 1.3. Every arm has the same:

- 150-step ceiling, $3 provider-spend ceiling, 2,400-second wall ceiling;
- 2,700-second resource lease and three recent screenshot frames retained;
- existing generic cost, repeated-action, unchanged-screen, and ask-loop guards;
- pinned `osworld-v2-2026.08.08` tasks/assets/code and isolated adapter environment;
- one episode at a time, distinct output root, cleanup audit before and after;
- no out-of-band controller interaction with the guest during the episode.

The first applicable cap ends the episode. A wall-limited episode is **not** a
completed 150-step test. Nine episodes would authorize at most $27 in provider
spend; infrastructure and any evaluator/simulator charges are separate. The
controller inspects each outcome before launching the next. Infrastructure
failures and sustained non-progress stop expansion; failed attempts remain in the
ledger rather than disappearing from the denominator.

Guest preparation holds/aborts snap refreshes and clears download cache. This
must be disclosed even on otherwise default hardware. The evidence library's
`comparable` or `reportable` labels attest its internal checks, **not benchmark
organizer approval or exact equivalence to reference protocols**.

## Apparatus checks and observed attempts

The starting runner is Local Operator 0.46.21 at
`2f3d7e9470b9b1c35737bdabf9e63e99beeac153`, with explicitly selected adapter
workspace `selector-0.1.5.json`. Do not use the stale default selector in an older
operator script. Verify installed library paths, distribution hashes, runner
source revision, and workspace digests before any run; version strings alone
previously concealed a checkout/wheel mismatch. Historical hyg1 event timestamps
fall on September 5 UTC, while the research context supplied September 4; preserve
the raw timestamps rather than silently asserting a consistent chronology.

The evaluated control loop is not the full interactive session: it deliberately
has no host shell/file tools, uses a custom JSON action contract, and sends
`tool_choice="none"`. It does reuse the central model/provider request path and
compaction engine. This distinction must remain explicit: shared primitives are
not proof of identical TUI prompts, skills, tools, or overall behavior.

The current action protocol lacks drag and explicit pointer-motion operations.
That is a capability limitation for the CAD domain, not grounds to remove a task
after observing failure. Add missing generic capabilities only against measured
need and independent regression tests.

| Attempt | Model / task | Outcome | Provider spend | Interpretation |
| --- | --- | --- | ---: | --- |
| Existing `batch-hyg1` | Spark / `task_001` | 60 steps, scored 0, no environment errors | $1.194133 | Environment endurance evidence only. Information gathering progressed, but the final calendar was byte-identical to its initial fixture. Not a completed task and not a 500-step comparison. |
| `batch-trend-luna-a01` | Luna / `task_010` | Setup failure before model call: missing `evaluation_examples` import | $0 | Packaging defect, not a Luna failure. Cleanup audit empty; batch expansion paused for repair. |

The expanded-set failure demonstrates why a one-task smoke test is insufficient:
the isolated wheel supplied `desktop_env` but not a runtime helper namespace
imported by another task. The fix must cover the pinned dependency graph and be
validated before cloud allocation, not special-case one task or add the controller
checkout to `PYTHONPATH`.

The repair's offline census found 16 imports of the omitted helper namespace.
After packaging the three exact release helper files, all 108 task modules loaded
in an isolated interpreter with network/process execution denied by an audit hook.
No task constructors, setup methods, or evaluators ran in that census. Static
mandatory dependency resolution reported no missing modules; `torch` and `lpips`
were absent only in a guarded optional path. This proves import coverage, **not**
that every task's live environment or evaluator works. The original task_010
`instantiate_task` reproduction separately changed from `ModuleNotFoundError` to
`Task010`, without allocating a VM.

## Promotion and reporting gates

1. Require observable target-side edits and at least one official-evaluator
   completion before increasing the sample. Information extraction alone is not
   completion.
2. Preserve binary and partial metrics according to the actual evaluator return
   contract. Do not infer one from an unrelated field or throw away the detailed
   evaluation artifact needed to interpret a zero.
3. Compare models on the same task IDs, apparatus revision, and budgets; distinguish
   setup failures, agent failures, truncations, and completed evaluations.
4. Generalizable fixes are developed against synthetic or held-out regression
   cases. Never inject benchmark answers, task-specific click scripts, or evaluator
   feedback into model context.
5. After tuning, freeze the harness and evaluate untouched tasks. Report all
   attempts and any stopping/selection rules. A promising small pilot supports a
   decision to spend on a full run; it cannot justify claiming equality with the
   published benchmark.
6. A leaderboard claim requires the release/protocol-specific full evaluation and
   the organizers' verification process, not only a successful local bundle seal.
