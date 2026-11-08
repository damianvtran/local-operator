# EXPERIMENTAL V2 preregistration — draft pending

> Public-port status (2026-11-03): DRAFT, NOT FROZEN, NOT READY FOR HEADLINE RUNS.
> The original 60-task OSWorld quota below is infeasible under current screening;
> 83 metadata-selected candidates are not 83 oracle-feasible tasks. Retained quotas
> and budget calculations are proposed design arithmetic, not an executable allocation.
> Amend selection and allocation before freezing. Terminal-Bench has 21/30 passing
> oracles in the private development report; six Linux runtime toolchains were
> verified, Hermes remains incomplete, and there have been no paid coding pilots.
> Those private artifacts are not shipped or independently revalidated here.
> (2026-11-03 note: the runner now pins released v0.94.22 `500883ef`; the
> text below records the earlier preregistration and is not updated.)
> LOP requires the reviewed source head `07c324e9` and its own source install;
> installed `0.94.22` is distinct and not evidence of that head. Never update a
> global runtime for this experiment. Runner source is ported after immutable
> author handoff, but paid/native-wire integration remains pending; see README.
> `../public_loadout/` is the sole shared public loadout, not a private profile.
> No quality parity or performance win is established by this protocol.

**Status: DRAFT / NOT AUTHORIZATION TO START HEADLINE RUNS.**
Date: 2026-11-03. This protocol is written before v2 headline runs. No model or API calls were made to prepare it. All pending checks in §12 must be resolved, the manifest completed, and this document committed and timestamped before the first headline trial. Any amendment must retain the previous version, its reason and timestamp. No public performance claims yet.

## 1. Question, population and scope

Does Local Operator (LOP), signed into its hosted resource classifier, reduce **total billed USD per successful long-horizon task without a material loss in task success**, relative to six alternative CLI harnesses driving the same main model?

Harnesses: Claude Code, Codex, LOP, omp, OpenCode, Hermes and dsh. Codex's primary configuration is the best-config arm: a verified 1M model catalog/context configuration, high reasoning and the tool-search bridge. Coding-lane rule (R14): no arm gets a server-side (Anthropic-executed) tool, because such a tool runs outside the network fence and is unpriced; client-side web tools stay available to every arm and fail identically for lack of a route. Each arm's server tools are denied natively: codex-best `web_search = "disabled"`, Claude Code `--disallowedTools WebSearch` (its WebSearch would send `web_search_20250305`), omp `web_search.enabled: false`; the byte-level audit found Local Operator's and dsh's DeepSeek search providers (a server tool at api.deepseek.com, unreachable: no DeepSeek credential, and deepseek.com is a fence model host) and Claude Code's advisor tool (inactive without an advisorModel setting, which the bench never sets); OpenCode and Hermes send none; the meter still refuses any server tool as a backstop (per-arm audit table: coding/README.md). Its historical as-shipped configuration is supplemental, never substituted into the primary comparison.

The primary population is offline-adapted OSWorld long-horizon tasks under the public loadout. It is not all desktop usage, live MCP usage or native-provider performance. The cache/workload policy, tool-search bridge and other interventions are part of the evaluated configuration, not invisible corrections. Report all seven harnesses, including failures and unfavorable comparisons. The benchmark author develops one evaluated harness; disclose this conflict and use an independent reviewer for eligibility, exclusions and interpretation.

Terminal-Bench 2.1 is a separate exploratory replication, not pooled into the primary OSWorld result. The clean/no-loadout arm and historical v1 are supplemental only. A public loadout does not reproduce private v1 exactly.

## 2. Freeze configuration and keep development separate

Before headline runs publish a machine-readable freeze manifest containing:

- This document's commit and hash; runner, proxy, translator, evaluator, aggregation and figure script SHAs; OSWorld full SHA corresponding to `8903ebe`; Terminal-Bench 2.1 repository/release and full commit; OS, Python, package lockfiles and toolchain versions.
- Exact installed executable path relative to the installation root, executable SHA256 and version for every harness. For source builds, source SHA **and rebuilt executable hash**; a source fix alone is insufficient.
- Main model routing/provider and resolved served model; reasoning/thinking/effort mapping, context-window metadata, output limits and context-compaction thresholds. Main-model intent is Sonnet 5.5, with Anthropic routing pinned and provider fallbacks disabled. Preserve and meter documented side-call models rather than silently forcing them to the main model.
- Permission modes (equivalent unattended approval), command line, allowlisted environment variable **names**, timeout, isolation and dependencies. No credential values. List every proxy mutation and native tool-search override; verify actual wire behavior.
- Public loadout hashes, per-harness configuration adapters and instruction discovery evidence. Use a single canonical instruction file per harness's effective discovery path; identical copies must not be loaded twice.
- LOP classifier endpoint/model/version if exposed, policy and account billing semantics; classifier enabled on every primary LOP run. Its timing/asynchronous recommendation policy is part of the frozen runtime.

**LOP replay defect:** the thinking-block replay fix must be independently tested and its runtime frozen before the holdout is revealed to implementers. Preliminary replay-fix outcomes (1/10 vs 0/11 baseline) do not establish a causal quality improvement and must not appear as effectiveness evidence. All tasks used to discover, triage or validate that fix are development tasks, even if the fix is general. Register them before selection. Do not tune the benchmark adapter or runtime against a failed headline task. See §10 for changes after freezing.

Existing v1 tasks (all 24 historical IDs, to be entered in an external exclusion ledger; the private source task archive is not distributed), every later triage/pilot task and any previously inspected outcome task enter `development_exclusions.json`. The exclusion ledger identifies task IDs, dates, reason and who had access; it contains no private content. The selector/oracle reviewer may see candidate inputs, but implementers must not see headline outcomes before the freeze.

## 3. Public loadout and equality checks

Use `../public_loadout/README.md`, its pinned public skills, uniform instruction text (`../public_loadout/public_user_instructions.md`), and `../public_loadout/public_mcp_schemas.json`. Historical private-loadout counts are not specifications for this public port. Record exact byte counts and JSON serialization, not just rounded KB. Recompute and freeze hashes; these counts are initial specifications, not a substitute for the actual manifest.

Install the same source skills, instruction text and MCP surface for all seven harnesses through their native discovery mechanisms. Native differences in catalog rendering, lazy discovery or skill reads are measured behavior, not grounds to pad smaller prompts. Verify:

1. Native discovery exposes all intended source skills/tools, and the canonical instructions are actually included **once**. Document catalog names, description UTF-8 bytes, bodies, instruction bytes, server/tool counts, tool description/schema bytes and per-tool size distribution.
2. Capture first/main/side request shapes and a synthetic final transcript. Count listed versus deferred tools, skill-body reads, tool-search calls and actual schema expansion. Never infer loading from files merely existing in a home.
3. Public office skills are relevant to these tasks; they may be read, unlike the v1 private skills. Include every resulting cost and report reads. No "none relevant" assumption.
4. MCP replay refuses service actions; tool discovery must work, calls must fail consistently, and no real service is contacted. Correct task solutions need only local tools. Codex's bridge is separately pinned, published and exercised with positive and negative discovery controls, including salt-on/salt-off behavior.
5. Hermes and dsh adapters receive exactly the same source loadout and have wire-level evidence before entry. Failure to implement an adapter is a pending gate, not permission to omit the harness after observing results.

Publish these as new public-loadout results; do not claim byte-matched replication of v1. Existing v1 figures remain historical and clearly labeled.

## 4. Eligibility and pre-specified sampling

### 4.1 OSWorld pool and task complexity

Distinguish three manifests: (1) the complete pinned source universe, (2) metadata-eligible candidates after pre-outcome exclusions and complexity classification, and (3) oracle/checker-feasibility-cleared headline candidates. A metadata-eligible task is NOT yet headline eligible. The selection lane reports 368 definitions in pinned `test_all.json` and example files, including 101 `multi_apps` tasks; freeze and verify these counts from its explicit selection manifest, not from the incorrect v1 review count of 369/102. Exclude all 24 v1 tasks as development, regardless of pass/fail, plus every additional task on the development ledger.

Start with all task definitions in pinned OSWorld domains `multi_apps`, `libreoffice_calc`, `libreoffice_writer`, `libreoffice_impress` and `vs_code`. Publish the entire candidate list before eligibility screening. Exclude the development ledger. No exclusion depends on harness outcomes, token costs, expected winner or historical v1 pass/fail.

Classify a task as **long-horizon** before running any harness if its reference dependency graph requires at least three ordered substantive operations and either:

- two or more independently supplied content-bearing input files must be combined/read to produce a result; or
- two or more separately testable content transformations must be composed, with a required intermediate artifact/state consumed by a later operation.

Merely opening, saving, importing a library, or inspecting a file is not substantive. One spreadsheet formula filled down one column is not long-horizon. Two reviewers resolve classification from task/input/reference evidence, without model outcomes. Publish the dependency graph and classification. Wall time, model call count and harness failures must not be used to retrospectively promote a task to "long".

Target 60 feasibility-cleared tasks (not a forced minimum if the source cannot support it), with the following proposed fixed strata. Do not declare the sample selected until actual source capacity and references support these quotas:

| Domain | Long | Short | Total |
|---|---:|---:|---:|
| multi_apps | 24 | 6 | 30 |
| libreoffice_calc | 6 | 4 | 10 |
| libreoffice_writer | 4 | 4 | 8 |
| libreoffice_impress | 4 | 4 | 8 |
| vs_code | 2 | 2 | 4 |
| Total | **40** | **20** | **60** |

These counts define the target mix, not the prevalence in OSWorld. Long tasks are the primary analysis; short tasks are a separate regression/context-overhead stratum. Generalization is to this eligible mix only.

### 4.2 Deterministic selection

Seed string: `harness-cost-v2-20261103-osworld-holdout-01`.
For each candidate compute lowercase SHA256 hex of UTF-8 `seed + "\n" + full_task_id`. Within each `(domain, complexity)` stratum sort by `(hash, full_task_id)` ascending. Screen candidates in that order until the stratum quota is filled; retain all later candidates as an ordered reserve. No Python salted `hash()`, unpinned RNG or filesystem enumeration order.

Publish the rank, eligibility decision and reason for **every screened candidate**, including rejects. If the eligible pool cannot fill any quota, STOP BEFORE RUNS: amend the target mix with independent review and a new version of this preregistration. Do not silently weaken complexity criteria, duplicate tasks or select alternatives based on harness performance. Feasibility of 40 truly long offline tasks is explicitly pending.

### 4.3 Offline feasibility and checker validity

For each selected task and reserve candidate, use no model/API calls. A reviewer builds a local reference solution from the **original inputs**, with precisely the offered offline dependencies and adapter, without consulting a harness output. The reference may use upstream gold to audit the checker, but must not copy gold into the deliverable or make gold-only edits not inferable from the instruction.

Require all of:

- A dependency inventory proves no GUI, network service, hidden VM state, missing application or unprovided file is needed. All GUI setup/getter/postconfig steps are either faithfully adapted and documented or make the task ineligible.
- Fresh reference execution passes; untouched input and missing output fail. Include negative controls that deliberately violate each requested property, not just a self-comparison of gold. For rule-based tasks create both a satisfying reference and meaningful failing examples.
- Gold identity, offered-library round-trip, correct reference and deliberately altered-output scores are recorded. **Round-trip failure is a diagnostic, not automatic impossibility:** investigate lost formula caches, harmless geometry, formatting and evaluator assumptions.
- A reviewer reads both task and checker to ensure it measures the requested operation, with appropriate numerical tolerances. Correct full-precision versus rounded values, formula versus stored value, notes paragraph boundaries, presentation geometry and uppercase instructions need explicit policies.
- Non-target properties must not reject a substantively correct solution due solely to a GUI application's save normalization. Fix an adapter/checker only before outcome runs; publish both upstream and adapted scores and explain the changed construct. If correctness is ambiguous or the reference cannot pass using information in the prompt, reject now, not after harness results.
- Reference execution is rerun twice from clean inputs. Task and evaluator hashes, expected result and negative controls are frozen.

Prompt adapter is uniform and supplies only necessary environment/path/dependency facts. It must not leak the reference solution. Instructions about formulas/precision must agree with the checker. The absence of a GUI is a change of benchmark modality and must be stated prominently.

### 4.4 Terminal-Bench 2.1

Independent seed: `harness-cost-v2-20261103-terminalbench21-01`.
Use the pinned task manifest, exclude all development/known-outcome tasks, and apply the same SHA256 ranking rule. Stratify by the dataset's published task categories, preserving its category names. For a target of 30 tasks allocate quotas proportional to candidate category counts using Hamilton largest remainder (ties by category name); screen each category in hash order and publish rejects. No cross-harness outcomes during screening.

Require the bundled oracle to pass in its pinned container twice and the unmodified environment to fail. Review each test against the instruction and apply controlled-negative tests for major requirements; oracle success alone does not establish validity. Reject tasks needing unavailable external services or unbudgeted resources. Oracle runs incur no model calls. Container/toolchain/oracle hashes and exact test commands are published.

Target approximately 30 tasks, one cold rep per harness under the exploratory budget. Admit tasks only as complete seven-harness blocks; fewer complete tasks are reported as such. Do not pool this evidence with OSWorld or claim three-rep Terminal-Bench coverage. Additional reps require a prospective budget amendment, never selection of just promising tasks.

## 5. Workloads and a practicable cache experiment

**Primary workload: an isolated cold-start session.** Each run begins with no reusable provider prompt prefix from another trial, but its own earlier requests may be cached and reused. This measures the observed cost of solving a new isolated task, not the cost of disabling caching entirely.

**Secondary workload: a steady-state batch of independent task sessions.** Same fresh session/home per task, but stable harness/loadout/cache namespace within a batch and consecutive tasks close enough to share eligible prefixes. Include the startup/warm-up cost in the batch total. Show both batch-inclusive cost and the post-warm-up subset; never headline the subset as the total workload cost.

### 5.1 Preferred isolation: empirically validated proxy namespaces

No assumption that a nonce in the user message, run ID, headers or a late system block invalidates the earlier tool/system prefix. Anthropic's documented conceptual cache order is tools → system → messages, but provider routing and tool deferral may change the effective serialized prefix. The main lane must archive authoritative documentation and **verify the actual route** before relying on this order.

Proposed reversible transformation, identical code for all harnesses:

- Create a fixed-length, inert, public namespace tag for the trial. Derive its content from the schedule, not task contents or results; use no secrets. State it is administrative cache metadata and must not affect task behavior.
- Prefix the **first actual serialized tool description** with the tag (not names, argument schemas, enums or required fields), after final tool ordering. Do not salt every tool description: eager-schema harnesses would otherwise incur hundreds of copies of the instrumentation overhead. Also prefix the first system text block with the same tag, covering no-tools/title/classifier-like main-provider calls. If the first tool has no description, an additive administrative description requires separate behavior qualification; otherwise use the TTL fallback. Do not add a fictitious tool or alter tool-use IDs, results, thinking blocks or message roles.
- The main provider proxy must see the final post-LiteLLM/tool-search transformation. When the first serialized tool changes or deferred schemas expand, the new first descriptor receives the same tag; do not tag every revealed descriptor. Keep native discovery and bridge indexing on original descriptions; verify that model-visible administrative tags do not alter search/selection.
- For isolated cold sessions, a unique namespace per `(task, harness, rep, workload)`, stable across all requests in that run. For the batch, a stable namespace per `(harness, batch_id)`, never shared with another harness/batch or development runs.
- Tag length and token footprint must be measured for every served model. Equal character length does **not** imply equal token length. Accept only a fixed-format tag set whose token overhead is equal or bounded by a frozen small tolerance (at most 2 tokens per tagged location and 4 tokens per request across the registered tag set); balance tag assignment across harnesses. Record total added input tokens. Same semantic intervention everywhere; byte-identical tags across harnesses are not required and would undermine isolation.
- This is an instrumented workload, not an unmodified native prompt. Show its overhead separately and publish the mutation. Compare to TTL-isolated unmodified sessions on development tasks. Do not subtract the cost from the observed primary metric.

**Preflight manipulation checks (paid only from the pilot allocation, never headline data):**

1. A repeated identical request/prefix in the **same** namespace shows real cache reads after a write, at token lengths exceeding the model's cache minimum. Confirm accounting on both streaming and non-streaming paths.
2. Change only namespace; cache reads from the prior namespace disappear. Reuse the previous namespace; eligible reads return. Run this A/A/B/A test with each harness's actual tool/system layout, with and without tool search, and each distinct side-call served model. Test a no-tools call as well as main requests, assistant-history reuse, compaction and deferred-schema expansion.
3. First requests and **first requests of each distinct side-call prefix** show no cross-trial cache reuse. A cold run's later cache reads must be attributable to a compatible prefix written earlier in the same run. Usage totals alone do not prove lineage: retain sanitized serialized prefix hashes/breakpoint lengths and request timing. Treat unexplained reads above a documented unavoidable wrapper allowance as a failed cold manipulation.
4. Tool selection and valid/invalid argument behavior are preserved on synthetic controls; no tag enters tool names/arguments or creates search hits. Same task/prefix behavior under salt vs no salt is checked on development tasks; do not demand identical stochastic output or use headline tasks for this check.
5. The batch's repeatable common main prefix has at least one verified cross-session read; retain measured read/write shares, timestamps, actual TTLs and the gap distribution. A batch with no hits is "attempted steady-state / no observed sharing", not warm by fiat.
6. Predeclare acceptable first-request residual reads per layout from the negative controls, including wrapper caching if unavoidable. Tolerance cannot be invented after headline inspection. If an unmodified tool prefix remains reusable, salts do not pass merely because most tokens miss.

**Do not run headline trials with salts until all applicable controls pass.** A namespace may also be needed for any separately cacheable classifier request; if its service cannot expose/control caching, report its observed behavior separately and do not claim that subsystem is cold. An external classifier returning no usage is a metering gate failure, not zero cost.

### 5.2 Valid fallback: TTL isolation without salts

If namespace isolation cannot be proved, use **unmodified** prompts. Inventory cache TTL markers and provider/account cache scope first. Do not silently change a native one-hour TTL to five minutes. Before an isolated session, wait longer than the maximum applicable TTL since the **last response completion** of any potentially sharing request in that account/model/cache scope, plus a documented safety margin (at least 60 seconds). For exclusively five-minute caches this is at least six minutes; for one-hour caches at least 61 minutes. Hold the account quiet or use a demonstrably isolated scope; unrelated traffic resets the timer.

Seven different harnesses are not automatically seven cache namespaces: common side calls may share. Until non-sharing is proved, isolate sessions serially across that scope. Within-session caching remains enabled. This can take days; wall time is a real tradeoff, not a reason to call a repricing "measured cold". Parallel execution is allowed only in proven separate scopes. Record every wait and last activity. Initial observed reads inconsistent with isolation stop the scheduler for diagnosis. A provider with opaque/uncontrollable caching cannot support a measured-cold claim.

For steady-state, run each harness's task batch contiguously in the same scope with no deliberate TTL wait and no intervening other harness runs. All batch task order is randomized in advance. Let natural within-run long pauses expire caches if they do; count the actual cost rather than synthetically restoring hits.

Freeze **one** successful cache method and its overhead/limitations before data collection. Do not pool salt and TTL measurements under one primary estimate. If neither works, pause: observed-order cost and a repriced sensitivity may be reported descriptively, but there is no measured cold/steady-state headline.

### 5.3 Repriced sensitivity, never a measured arm

Optionally price the same observed usage under a common frozen rate card and under a common cache rule (e.g. no cross-session reuse or no cache reuse at all). Show a bridge from observed billed cost, by model and cache tier, including side calls, cache writes, assistant/history reads and classifier charges where available. Prefix-identical validity cannot be reconstructed by taking the minimum token count alone. Do not "correct" only a high-cost run. Apply the rule to all harnesses and all runs. Label it **counterfactual repricing of a fixed trace; no behavioral rerun**. It does not estimate how changed latency/context/cache behavior would alter the trace.

## 6. Repetitions, order and scheduling

OSWorld primary: 60 tasks × seven harnesses × **three cold reps** = 1,260 trials, of which 840 are in the 40-task long primary stratum.

Secondary batch: the same 60 tasks × seven harnesses × **one batch rep** = 420 trials. Three batch reps are desirable but not promised within $800. This choice preserves three primary reps and a measured second workload instead of pretending a repriced column is another experiment. Batch evidence has no within-cell replication; label its uncertainty accordingly.

Use ordering seed `harness-cost-v2-20261103-schedule-01`. Generate and publish the full schedule before runs, using SHA256 rank keys for rep/block ordering, seven-harness order and within-batch task order. Keys concatenate seed, workload, rep, full task ID and harness with newline separators. For cold trials a block is all seven harnesses for one task and rep; prefer executing complete tasks (all three blocks) before budget admission advances. Randomize rep-block order within tasks. Balance harness positions across blocks via a seeded cyclic Latin rotation, with seed-ranked harness labels; publish actual position counts. Do not pick arbitrary batches at execution time.

Batch scheduling: seed-rank all 60 tasks once, use that order for each harness to make the batch workload paired; seed-rank the seven harness-batch orders. Run each complete 60-task harness batch contiguously. Cross-harness time confounding remains with one batch rep; state it, log service time, and do not claim a fully replicated crossover. If additional reps are funded prospectively, rotate batch order and use new namespace IDs. Include one prespecified discarded-for-quality warm-up development task per harness, but **charge it to batch cost** and the budget; exclude it only from the success denominator.

One trial uses a fresh home/workdir/session and fresh inputs. No surviving agent memory, compaction summaries or previous-task artifacts. Only the deliberate batch provider cache is reused. Concurrency is bounded by disk/resources and cache-scope evidence; contention and provider throttling are logged. Maintain the 15 GiB disk floor, shared dependency trees and exact-PID cleanup. No live service restarts are implied by this protocol.

## 7. Outcomes, quality gate and estimands

Let `C_thr` be all attributable billed USD for task t, harness h, rep r (including retries and failed calls under §9). Let `Y_thr` be the frozen checker success indicator, 0 or 1. Publish partial checker scores separately but do not redefine success after inspection.

For a stratum S with n tasks and R reps:

- `mean_cost_h = (1/n) sum_t [(1/R) sum_r C_thr]`.
- `quality_h = (1/n) sum_t [(1/R) sum_r Y_thr]`.
- **Cost per successful task opportunity:** `E_h = mean_cost_h / quality_h`. Equivalently total spend divided by total successful attempts when cells are complete. If quality is zero, E is infinite. This includes the cost of every failure; it is not the mean cost conditional on success and it is not a retry-until-success expectation.

Primary: `E_LOP / E_comparator` in the **40 long tasks, isolated cold workload**, with an explicit quality non-inferiority gate for each of the six comparisons. No pooled seven-harness winner metric and no substitution of the easiest stratum. Primary mixture weights are fixed by the long quotas in §4 (24/40, 6/40, 4/40, 4/40, 2/40).

Quality non-inferiority margin: **5 percentage points absolute**, `quality_LOP - quality_comparator > -0.05`. A cost-effectiveness superiority claim additionally requires the cost-per-success ratio's upper confidence bound <1. Both conditions must hold for that comparator. A low cost caused by errors, early abandonment or caps cannot earn a "more cost-effective at comparable quality" claim without passing the quality gate.

Secondary (clearly labeled): 60-task mix, short stratum, steady-state batch, per-task arithmetic mean USD, geometric and median paired cost ratios, win counts, cost components, tokens, context growth, wall time, first-request tokens and failure categories. Provide each comparator's quality and spending, not only ratios against LOP. Never remove all-fail or all-pass tasks because they do not discriminate. A zero-cost crash is not a win.

Batch cost accounting: charge its startup/warm-up and any between-task assistant calls. Allocate batch-only overhead equally across its 60 tasks for per-task ratios; publish that allocation and the unallocated batch total. Distinguish cost per pass of a complete batch from any conditional post-warm-up table.

## 8. Statistics and power limits

Resample **tasks, not 180 independent runs**. Use 100,000 deterministic hierarchical paired bootstrap draws, seed `20261103` with the recorded generator/library version:

1. Resample tasks with replacement within each fixed domain/complexity stratum, preserving stratum quotas/weights.
2. Within each sampled task, resample its three rep blocks with replacement, using the same selected rep labels for all seven harnesses. Preserve the pairing and repeated-measures clustering. Each independently drawn copy of a task receives its own inner resample.
3. Recompute quality, cost, E and paired ratios in each draw. Preserve infinite E/ratios, report how often denominators vanish, and do not discard them to manufacture finite bounds. For the one-rep batch and Terminal-Bench exploratory sample, task bootstrap only; it cannot measure within-cell variability.

Publish ordinary two-sided 95% paired intervals for descriptive estimates. For six preregistered cost-plus-quality claims, use conservative Bonferroni bounds across **12 one-sided gates** (six NI and six cost gates), alpha `0.05/12` for each bound: lower quality-difference bound >−0.05 and upper E-ratio bound <1. This is an approximate bootstrap family-wise procedure; show finite-sample sensitivity and do not present it as exact small-sample coverage. Do not replace with a more permissive test after seeing data.

Report supplementary paired Wilcoxon signed-rank tests on per-task mean log-cost ratios, sign/win counts and a Friedman rank summary across seven harnesses. Apply Holm across the six Wilcoxon comparisons; these do not replace the primary quality gate. Exact McNemar on per-task majority-of-three success is a sensitivity analysis; primary quality uses attempt fractions, not majority votes. Binomial/Wilson intervals on majority counts may be shown with that denominator explicitly, not as CIs for 180 independent attempts. No inference that p>0.05 establishes parity/equivalence.

**Power constraint:** 40 primary long tasks is small for a 5-point NI margin, especially with six comparisons. Rough paired binary planning with discordance 0.10 gives about `0.10 * (2.64+0.84)^2 / 0.05^2 ≈ 485` independent tasks for 80% power at the registered one-sided alpha 0.05/12, before accounting for heterogeneous repeated attempts; discordance 0.30 requires about 1,450. These are illustrative assumptions, not a post-hoc power claim. Three reps improve estimates within tasks but do not create three times as many independent tasks. Conduct a frozen simulation using development-only variance/discordance if available and publish its assumptions before runs. This budget may produce **inconclusive quality parity**, even with a clear cost difference. Report that honestly; never widen the NI margin to get a winner.

## 9. Caps, failures, retries, attribution and budget

### 9.1 Common trial limits

Initial preregistered proposal: **$0.30 billed USD per OSWorld trial**, 20-minute wall limit, 1,000 metered main/side provider requests maximum. Same limits in both cache workloads and for all harnesses. Limits include all attributable side calls and LOP classifier charges; count repeated tool/model calls. Terminal-Bench proposal: $0.45/trial, 30-minute wall limit, same request ceiling. Task-complexity eligibility is independent of these caps.

These monetary caps are a feasibility/budget tradeoff, not proof that a trial can exercise a genuinely long horizon. Pilot projections must establish whether they are useful before freezing (§12). If the cap is usually binding, pause and replan rather than publish "long-horizon effectiveness" from mostly truncated traces. Do not raise caps for only one harness or for tasks it failed. Any revision is uniform, prospective and recorded before holdout outcomes.

Stop admitting calls using a reservation ledger for **worst-case billable in-flight exposure**, not just spend already returned. Reserve according to model/input/output limits and known cache-write maximum rates before dispatch, plus explicit bounded classifier exposure. With unknown provider billing or unbounded side calls, a hard cap is not enforceable: this is a launch blocker. A denied reservation ends the attempt as cost-capped even if it stops below $0.30. Publish actual spend and binding frequency; never bill a synthetic full cap for unspent tokens. Lowering requested output limits to fit reservations must be applied by an explicit uniform pre-frozen policy, not opportunistically per harness.

### 9.2 Classifying failures without favorable reruns

- Checker failure, invalid output, tool misuse, refusal/abandonment, cost cap, request cap, timeout and **harness-internal crashes** are primary failures (Y=0). Cost and wall time remain. A file may still be scored for diagnostics; an externally capped attempt does not become a primary success after cutoff. An unbilled, **complete** response whose token counts price at the run's frozen rate card settles from those tokens instead of reading as unknown spend (ADAPTERS.md): it is a bill, so it neither voids the attempt nor blocks later admissions, and a stated `usage.cost` is used unchanged wherever the provider gives one. A response that ended before completion never settles this way: its partial usage is a floor, not the charge, and it keeps the unknown-spend latch.
- An identifiable external infrastructure defect (input copy corruption, evaluator service outage, proxy transport failure, host termination unrelated to harness, provider 5xx/429) is marked invalid with evidence and independently reviewed, without looking at comparative scores. Maximum **one** infrastructure rerun of a cell; no rerun because its cost is high or its score is low.
- Do not automatically call every nonzero exit an infrastructure defect. In particular v1 exit 143 does not establish who sent SIGTERM. Preserve exit signal/parent/process evidence.
- For invalid infrastructure attempts publish both original and rerun rows. The scheduled opportunity's primary **paid cost** includes all actually billed attempts, even when quality is assessed on the valid retry. Provide a valid-attempt-only sensitivity separately. If no valid retry completes, classify as unresolved; a primary comparative claim is blocked unless a frozen conservative missingness analysis also supports it. Do not silently drop just one harness's cell.
- Native retry behavior within a session is included and metered. Proxy retries must be idempotency-aware and logged individually. No free or invisible retries.
- A discovered broken checker is a dataset incident: freeze the affected task across **all harnesses**, retain original data, independently adjudicate, and report with/without it. Any replacement comes only from the frozen reserve, with complete reruns for all seven harnesses and all reps; it is new data, never swapped in just for a losing harness. No replacement because a task has all zeros/ones.

### 9.3 Incremental budget: maximum $800

| Envelope | USD maximum | Scope |
|---|---:|---|
| Development/preflight | 60 | Cache validation, model metadata/reasoning/bridge and metering controls; no headline tasks |
| OSWorld headline workloads | 540 | 1,260 cold + 420 batch trials at $0.30 = $504 maximum nominal spend, plus batch warm-up/exposure headroom |
| Terminal-Bench exploratory | 100 | Up to 30 tasks × 7 harnesses × 1 cold rep × $0.45 = $94.50, plus exposure headroom |
| Infrastructure retry envelope | 80 | Evidence-backed scheduled-cell retries, not feature tuning |
| Uncommitted safety reserve | 20 | Exposure/ledger discrepancy only, not automatic spending authority |
| **Total incremental maximum** | **800** | Do not reset or reinterpret historical spend |

Record starting cumulative spend and separate campaign-ledger costs; shared live spend is not attribution. Reconcile actual request charges + classifier charges + other billed routes to the campaign total, and enumerate development/invalid/headline subsets. Poll/account for concurrent in-flight calls atomically. Never consume another lane's budget.

Admit an OSWorld task only when the remaining envelope can reserve a complete seven-harness × three-cold-rep block (up to $6.30), plus its planned seven batch opportunities (up to $2.10), retries/exposure margin and committed obligations. The target is 60 if supported by the source; if feasibility capacity or funding cannot assure the frozen target before starting, STOP and amend. Do not force ≥60 with unsuitable tasks or advertise ≥60 and silently deliver fewer. During collection an unexpected safety stop leaves a clearly **incomplete exploratory** campaign, never a reduced confirmatory sample chosen from successful completed cells.

For the batch phase reserve the entire remaining seven-harness cohort before starting any harness's batch (at most $126 nominal for 60 tasks). This avoids a budget stop leaving only cheap harnesses completed. Preserve complete admitted cohorts; budget is not a criterion for selecting completed wins.

Terminal-Bench is lower priority: only start a task when all seven runs can be reserved. Stop at the last complete block in the preregistered order. No minimum 30 guaranteed. No transfer that jeopardizes the complete OSWorld design. Envelope changes or extra reps require a dated amendment before the affected outcomes; report the original plan too.

## 10. Changes and fresh-holdout rule

No runtime/provider/configuration changes inside a primary campaign. Reconcile expected and actual executable SHAs at each run. A version drift or bug discovered mid-run pauses collection. Do not patch the failing harness, rerun only its failures and merge those outcomes into the frozen comparison.

If a change is necessary, retain the entire old campaign and its incidents. Any task whose outcome informed the fix becomes development data. Start a new pre-registered version with a new seed, remaining eligible unseen tasks, renewed feasibility checks and all seven frozen configurations. If insufficient unseen tasks/budget remain, the result is an engineering case study or incomplete benchmark, not a fresh validation. Generic correctness fixes may be justified, but performance after such a fix still needs a fresh holdout. Triage results must never be selected as public wins.

## 11. Publication and audit trail

Publish raw sanitized request/run ledgers (all attempts, not only successes), evaluator scores/reasons, kept artifacts, oracle scripts/tests, ordered task manifest/rejections/reserve, schedule, rate card and dated provenance, classifier tokens/charges, environment and executable hashes, cache manipulation controls, namespace/prefix metadata and actual timing. Include all error/cap/timeout/compaction counts. Retain served/requested models, response status, usage, cache tiers/TTLs, tool/message counts and matched billable request identifiers. Missing usage is an incident, not zero.

Use observed provider billed usage as primary cost; when a hosted classifier bills a separate charge, include that charge once in total and show it separately, together with its tokens. Do not double-count vendor-estimated model cost and the service charge. If prepaid credits or subscriptions make marginal charge zero, report actual billed cash and attributable credit/list-price consumption separately; establish the primary convention prospectively. An unpriced classifier prevents a complete all-in cost claim. Include all assistant side calls, titles, search, summaries/compaction and retries, even when they have no tools.

One pinned command must derive every table and figure from the published immutable snapshot, with unique attempt IDs, no last-write-wins hiding reruns, explicit joins, missingness checks and totals reconciled by model/workload. Include token units, task and rep denominators, uncertainty, workload labels and quality-gate outcome in figures themselves. Context-growth curves must show how many runs survive to call k; do not imply a fixed cohort when short runs drop out. No synthetic corrected Codex row in the observed-cost headline.

Never publish credentials, private loadout text or user home paths. Use public inputs and kit-relative paths/environment variables. Review request/response content and artifacts for private leakage before committing. Original v1 is frozen; corrections are versioned addenda, not retroactive data edits.

## 12. Pending checks / decisions before freeze (main-lane gate)

- [ ] Confirm that the 40-long/20-short quotas can be filled by **previously unseen**, oracle-feasible tasks. Publish pool, exclusion ledger, ranked eligible list, references, negative controls and adjudication. If not, amend before outcomes; do not relax silently.
- [ ] Approve the workload allocation: three cold reps primary, one measured batch rep secondary, approximately 30 Terminal-Bench tasks with one cold rep as budget permits. Three reps per cache regime and three Terminal-Bench reps are **not** funded promises.
- [ ] Validate whether $0.30/20-minute OSWorld and $0.45/30-minute Terminal-Bench caps meaningfully exercise long-horizon work. If conservative request reservation cannot admit ordinary calls, or dev trials are mostly capped, revise scope/budget before freeze. Do not hide a capped-toy experiment behind the long-horizon label.
- [ ] Choose **validated salt namespaces OR measured TTL isolation** after development controls. Archive official cache ordering/minimum/TTL documentation and route-specific empirical evidence. Freeze tag serialization/tolerance or exact quiet-scope/wait policy. Accept potentially multi-day TTL wall time if salt isolation fails.
- [ ] Ensure classifier-on LOP has separate usage and a defensible billable USD mapping, cache-policy qualification, and bounded reservation; freeze asynchronous recommendation behavior. No metering path may silently omit it.
- [ ] Finish replay fix independent QA; freeze deployed LOP runtime SHA/executable hash. Freeze all seven runtimes and the Codex catalog/bridge/high-reasoning mappings; verify no pathological compaction and preserved thinking on the wire.
- [ ] Verify public loadout exposure once per harness, especially omp instruction duplication, Hermes/dsh discovery, native skills and deferred/search tools. Record exact public schema and skill manifest hashes.
- [ ] Verify all seven main-model and side-call routes, streaming usage merging, provider costs, classifier attribution and atomic maximum-exposure reservation. Record negative controls for missing usage, cap denial, retry and crash classification.
- [ ] Freeze checker adaptations/precision policies and privacy-safe output pipeline; independently review the NI margin, multiplicity and power caveat. Accept that outcome may be "cost differs; quality parity inconclusive".
- [ ] Complete remaining manifest fields; commit/hash protocol, schedule and task selection before headline starts. Main lane records each check's evidence and decision, not a blanket "done".

**Protocol verdict: draft complete, experiment NOT READY until these gates are evidenced. No public claims are authorized by this document.**
