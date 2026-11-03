# Historical independent rigor review — EXPLORATORY v1

Publication edits: removed private paths; corrected upstream universe count to 368 (multi_apps 101), per pinned task-manifest audit; commands/paths adapted for the public kit. Review findings remain historical; see ../CORRECTIONS.md for derivation clarifications. Original-source hash is recorded in ../PROVENANCE.json.

# Rigor review, round 1: harness cost benchmark v1

Reviewer: independent data-science reviewer (agent `reviewer`, model claude-opus-5-5). Read-only, and $0 of model spend.
Scope: the frozen v1 data in `v1-snapshot/` (SHA256SUMS verified OK, 13/13 files), plus the rig scripts that produced it.
All numbers below were recomputed from `v1-snapshot/requests.jsonl` and `v1-snapshot/runs.jsonl`, rep 0, with the scripts in `reviews/`. Run them from the kit root with `the optional review environment Python`:

| Script | What it does |
|---|---|
| `reviews/recompute.py` | per-run list price, plus OpenRouter billed cost |
| `reviews/cold.py` | reverse-engineers the published cold column and computes a stricter alternative |
| `reviews/stats.py` | paired bootstrap, sign, Wilcoxon, McNemar, Wilson and Friedman tests (output in `stats.out`) |
| `reviews/feasibility.py` | gold, gold round-trip, input and missing-file scores per checker (output in `feasibility.out`) |
| `reviews/rescore.py` | re-scores every kept artefact against runs.jsonl |

**What reproduces.**
- Every headline warm mean in figure_data.json, to the last digit:
  - LOP 4.344¢
  - omp 6.823¢
  - OpenCode 7.923¢
  - Claude Code 9.238¢
- Pass counts.
- Cost per pass.
- All 180 rep-0 logged scores are reproducible from the kept outputs under the same evaluator: `rescore.py` gives agree=180, disagree=0. This establishes score reproducibility, NOT semantic checker validity or feasibility.

The methodology problems below matter more than any arithmetic error.

Severity key:
- **blocker:** the published claim as worded is wrong or unsupported.
- **major:** a real defect a fair critic will land.
- **minor:** improves rigour or clarity.

---

## R1 — blocker — Codex's full-arm cost measures a compaction loop, not just schema bloat

**Evidence.**
- All 234 Codex full-arm main calls have `n_messages` of 1 or 3. In the clean arm the counts grow 1→3→5→7→9.
- Main calls (496 tools, one message) strictly alternate with one-tool calls whose prompt is the handoff template "Another language model started to solve this problem and produced a summary…". See `logs/bodies/full__codex__aceb0368__r0.last.json`.
- Codex's own stdout for every full run:
  - It warns `Model metadata for 'claude-sonnet-5-5' not found. Defaulting to fallback metadata`.
  - It emits one compaction warning per cycle: 9, 21, 59 and 26 compactions on eb303e01, aceb0368, 81c425f5 and bc2b57f3. Count with `grep -c 'compactions can cause' runs/full__codex__*/stdout.log`.
  - None of the 16 clean Codex runs compacts.
- So the "11.7M input tokens per run" (REPORT.md:27, fig2 C) comes from two things multiplied together:
  - the 376k prefix;
  - a compaction after almost every step, associated with missing model metadata; its exact causal mechanism was not isolated.
- That is a property of running an unknown model id through LiteLLM, not of Codex shipping schemas alone.

**Other Codex confounds.**
- **Thinking.** Codex reached the model with no `thinking` on 100% of calls. The other four harnesses used adaptive thinking, at effort `high` except OpenCode.
- **Cache injection.** The proxy injected a top-level `cache_control` on all 320 Codex calls.
- **Correction.** Reconstructed: it reprices calls after the first in eb303e01 as reads of the 374.7k minimum prefix. That reproduces $1.8708 against $9.965 measured.
  - It is applied to 1 of 4 runs, chosen after seeing the data.
  - The root cause of the misses was not established. One detail: eb303e01's first body has 494 tools and later bodies have 496, and the system hashes differ.
- **Subset.** Codex ran on 4 tasks, the first 4 in file order, before a budget stop. n=4 supports no ratio. The "≈41×" (3.735/0.0918) has no uncertainty attached.

**Remediation.**
- Withdraw the Codex headline row, the 41× callout and fig2 C's 11.7M bar. Alternatively, relabel them "Codex + LiteLLM with an unrecognised model id: compaction every turn".
- For v2, either:
  - give Codex correct model metadata (context window and the reasoning mapping) and verify on the wire that `n_messages` grows and thinking is on; or
  - drop Codex and say why.
- Never apply a post-hoc correction to one run. If the cache miss recurs, report it as measured, alongside a pre-declared counterfactual applied to every run.

## R2 — blocker — the warm-cache headline is order- and timing-dependent; cold-start reverses the ranking vs OpenCode

**Evidence.**
- **The published cold definition.** It is cost + (first *main-call* `cache_read`) × (2.50 − 0.10)/M. That formula reproduces 180/180 `cold_cost` rows.
  - "First call" (REPORT, fig1 footnote) is wrong: it matches only 47/180, because 4 harnesses open with a tool-less side call.
- **The arithmetic is sound.** A stricter definition prices every cross-run read, side calls included, per served model. It agrees within 0.01¢ for LOP, omp and OpenCode, and lowers Claude Code from 12.59¢ to 12.28¢.
- **Cold reverses the headline vs OpenCode.** Recomputed full arm, harness/LOP paired ratios, task-bootstrap 95% CI:

| vs LOP | warm geo-mean ratio | cold geo-mean ratio | LOP cheaper (warm / cold) |
|---|---|---|---|
| omp | 1.63 [1.52, 1.74] | 1.45 [1.41, 1.48] | 24/24 / 24/24 |
| OpenCode | 2.03 [1.76, 2.29] | **0.81 [0.77, 0.87]** | 23/24 / **3/24** |
| Claude Code | 2.18 [1.60, 2.71] | 1.03 [0.76, 1.21] (median 1.18) | 23/24 / 23/24 |

- So REPORT.md:61's "LOP's advantage narrows there" is false for OpenCode: cold, OpenCode is cheaper on 21/24 tasks (Wilcoxon p = 2e-5).
- **Where LOP's warm advantage comes from.**
  - Its first main call reads a median 95% of its prompt from cache: 27.8k cross-run tokens per run. The figures are 85% for omp, 41% for Claude Code and 26% for OpenCode.
  - Those reads exist only because the same harness ran a run with an identical prefix about 1.1 minutes earlier (median gap; 23 of 24 gaps under the 5-minute TTL).
  - The early tasks include cold first-main requests after the disk pause; this is not universal (Claude Code eb303e01 has a 0.41 read share). That is the same mechanism in reverse.
- **The clean arm, which REPORT omits.**
  - Warm: omp 2.04¢ < OpenCode 2.39¢ ≈ LOP 2.40¢ < Codex 5.75¢ < Claude Code 5.92¢.
  - Cold: LOP 7.94¢ is second most expensive. omp/LOP 0.59, OpenCode/LOP 0.45, Codex/LOP 0.75, Claude Code/LOP 1.56.

**Remediation.**
- Report cold-start and steady-state as two pre-declared headline metrics with equal prominence. Do not present one as a footnote.
- Rewrite "narrows" to "reverses vs OpenCode".
- Publish the clean-arm results.
- v2: isolate caches. Use a per-run unique prefix nonce in the first user turn? No: that breaks the harness's own caching. Instead, run each harness's cold run first in a fresh 6+ minute window, then N warm reps back-to-back, and report both.

## R3 — major — n=1 per cell: cost ratios survive task sampling, pass-rate differences do not, and run-to-run variance is unmeasured

**Evidence (from `stats.out`).**
- **Cost.** The paired cost ordering survives task resampling:
  - Friedman χ² = 52.2, p = 3e-11, over 4 harnesses and 24 tasks.
  - Mean ranks: LOP 1.08, omp 2.29, OpenCode 2.96, Claude Code 3.67.
- **Pass rates** (Wilson 95% CI):
  - LOP 16/24 = 67% [47, 82]
  - omp 17/24 = 71% [51, 85]
  - OpenCode 17/24 = 71% [51, 85]
  - Claude Code 14/24 = 58% [39, 76]
- **No pass-rate difference is detectable.** Exact McNemar p is 1.00, 1.00 and 0.50; discordant pairs number 0/1, 1/2 and 2/0.
  - "±1–2 tasks of noise" (REPORT.md:59) understates it: a Wilson half-width of about 18 pp is about ±4 tasks.
- **Cost per pass**, bootstrap ratio vs LOP:
  - omp [1.28, 1.66]
  - OpenCode [1.29, 2.28]
  - Claude Code [1.95, 3.18]
- **Run-to-run variance is not in any of these intervals.** The only repeats are smoke reps r8/r9 on 1e8df695, and they show large swings:

| Harness | r0 | r9 |
|---|---|---|
| omp | $0.048, fail | $0.143, pass |
| LOP | $0.027 | $0.101 (cold) |
| OpenCode | pass | fail |

  So both cost (×3–4 from cache state) and outcome flip within a cell.

**Remediation.**
- Publish the CIs above.
- Drop or soften every pass-rate comparison ("no detectable difference in pass rate at n=24").
- v2: at least 3 reps per cell, with rep-level variance reported (see the v2 design).

## R4 — major — task selection is undocumented and half the tasks cannot discriminate between harnesses

**Evidence.**
- **The 24 tasks discriminate poorly.** 14 pass for all four full-arm harnesses and 6 fail for all four; only 4 separate any harness from another (corrected by recomputation).
  - All-pass: 37608790, abed40dc, 01b269ae, 7a4e4bc8, 51b11269, 0cecd4f3, e528b65e, edb61b14, 276cc624, 930fdb3b, aceb0368, bc2b57f3, 51f5801c, 20236825.
  - All-fail: eb303e01, 5bc63fb9, 8df7e444, d53ff5ee, 73c99fb9, 9cf05d24.
- **Selection is undocumented; post-hoc selection cannot be ruled out.** The 24 were picked by the author from the pinned 368-task source universe (OSWorld @ 8903ebe: calc 47, impress 47, writer 23, multi_apps 101, vs_code 23, …; exact selection manifest is authoritative).
  - There is no recorded selection rule.
  - The only validation was gold = 1 and input/missing = 0, and that check passes for all 21 file checkers.
- **The author develops one of the harnesses.** A hostile reviewer will raise this conflict of interest.
- **Offline checker sensitivity, not a proof of impossibility.**
  - 73c99fb9 and 9cf05d24: `compare_pptx_files` defaults `examine_shape=True` (OSWorld slides.py:234). The LibreOffice-saved gold shifts shape geometry relative to the input on 5/5 and 6/7 shapes, e.g. size (8227800, 1141200) vs (8227440, 1140840).
  - A simple python-pptx edit preserves input geometry, so a content-correct edit can fail for unrelated geometry. This does NOT prove that every possible offline solution is impossible: low-level geometry normalization remains possible, though not justified by the task instruction alone. All four harnesses wrote "Page 1" correctly into slide 2.
  - eb303e01: REPORT claims it is unpassable (paragraph merge on save); that claim was not independently proven by this review. Note its options already set `examine_shape: false`, so the cause there differs.
- **Ambiguous instruction.**
  - d53ff5ee: the gold lowercases *all* text ("presentation instruction"). Every harness lowercased only all-caps runs; Claude Code's answer says so explicitly.
  - That is an instruction-ambiguity task and is insensitive to the harness.
- **Formatting-dependent.**
  - 1e8df695: the gold holds `=B2-C2` formulas with cached full-precision values. Claude Code, LOP and omp wrote rounded values (11577.18) and fail; OpenCode wrote full precision (11577.1772459512) and passes.
  - The pass/fail is a rounding choice under a prompt that says "store computed values".
- **A terminated run was scored as a task fail without resolved cause.**
  - `full__claude-code__eb03d19a__r0` exited 143 after 2.6 s with `[claude-code:unrecognized_model] {"query_source":"generate_session_title"}`. Its stdout is empty.
  - It is scored 0 and is the *one* task where Claude Code was "cheaper" than LOP ($0.0038). It is the 23 in "23/24", and it pulls Claude Code's geo-mean CI down to 1.60.
  - Claude Code passed the same task in the clean arm.

**Remediation.**
- Pre-register a selection rule: stratified random sample from the offline-feasible pool, fixed seed.
- Run an offline-feasibility gate: an oracle or reference solution built with the offered toolchain must score 1.
- Report the all-pass and all-fail split.
- Adjudicate termination causes: harness-native crashes are outcome failures and stay counted; only independently evidenced external infrastructure faults qualify for a bounded retry, with original paid cost retained. Exit 143 and a warning alone do not establish external causation. Record exit codes and a `valid` flag in runs.jsonl.
- Disclose the conflict of interest.

## R5 — major — derived numbers have no producing script

**Evidence.**
- `figure_data.json`, `cold_cost` (in results.json), `call_sequences.json`, `cost_components_long.json` and the Codex "cache-corrected" $3.74 are read by `figure_cost.py` and `chart.py`.
- No script in the rig writes them: `grep -n "figure_data\|cold_cost\|call_sequences\|cost_components" *.py` finds only readers.
- report.py's results.json also lacks `cold_cost`, which the snapshot's results.json carries.
- I reverse-engineered the cold definition (R2) and the Codex correction (R1). A reader cannot.

**Remediation.**
- Before publishing, add one `derive.py` that turns logs/ into every JSON the figures read.
- Make figures and tables regenerate from raw logs in one command, and check the output into CI.

## R6 — major — "full loadout" is not matched across harnesses; other configuration asymmetries

**Evidence.**
- **Instruction files.** Measured in each harness's last request body on 51f5801c (aceb0368 for Codex):
  - The 3,480-byte CLAUDE.md/AGENTS.md text is fully present (13/13 lines) for Claude Code, Codex and omp.
  - OpenCode gets none (0/13; its config is a 50-byte opencode.jsonc).
  - LOP gets a different, 19,814-byte `system_prompt.md` (25/25 of its own lines) and only 4/13 of the shared lines.
- **Skill roots.** SKILL.md counts per root: LOP 30, `.claude` 27, `.codex` 33, `.agents` 28, OpenCode 15, omp 15. The fig1 subtitle says "15 skills".
- **Tool counts.** "480 tools" in fig1 vs 482 in REPORT.md:27 vs 496 on Codex's wire, which includes plugin tools.
- **Thinking, effort and output budget.**
  - Thinking adaptive with effort `high`: Claude Code, LOP, omp.
  - Adaptive without effort: OpenCode.
  - No thinking: Codex (R1).
  - `max_tokens`: Claude Code 32,000; the others 128,000.
- **Side calls** in the full arm, rep 0:
  - omp: 408 Haiku calls (17/run, $0.023 total).
  - OpenCode: 24 Haiku.
  - Claude Code: 24 Sonnet title calls ($0.091).
  - LOP: 48 Sonnet ($0.029).
  - Defensible as "harness defaults", but they must be disclosed.
- **The proxy's cache injection also hit Claude Code.** It injected `cache_control` into 42 of Claude Code's tool-less title calls, about 0.38¢/run in the full arm. That contradicts REPORT.md:55, "in practice only Codex".
- **The hosted classifier (new in v2).** v1's LOP ran without the signed-in account classifier (`runner.py.bak-v1` has no token injection; v1 rows lack `lop_radient`).
  - From the r50 probe (`full__local-operator__37608790__r50`), the v2 LOP path adds:
    - per-turn classifier calls to a hosted model (e.g. `tokens=4795/477 cost=$0.000201`), billed outside the proxy;
    - a `<resource_recommendations>` block in the transcript (2 occurrences in the last body);
    - timing nondeterminism (`no recommendation within 50 ms; the turn continues without one`).
  - In that probe, the first main call wrote 31.7k to cache (cold), against v1's 1.5k write plus 30.3k read.
- **ENABLE_TOOL_SEARCH=true for Claude Code.** Defensible, but the "first-party default" claim needs a citation. The same principle (each harness at its documented direct-API default) must be applied and stated for every harness.
- **Concurrency.** The clean arm ran 5 harnesses concurrently (`matrix.sh:21`); the full arm ran sequentially (`matrix_full.sh`). Wall-time comparisons across arms are not like-for-like.

**Remediation.**
- Define the loadout as a harness-neutral spec: N skills, one instruction text, M MCP servers. Render it into each harness's native location with identical content.
- Verify on the wire that the instruction text and skill names appear in every harness's first body, and publish that check.
- Disclose effort, thinking, max_tokens and side-call models per harness in a table.
- Never mix runs with and without the classifier: run LOP in both modes as two labelled rows, and add the classifier's own cost and token count as a line item.
- Fix the "only Codex" sentence.

## R7 — major — reproducibility and privacy of the published artefacts

**Evidence.**
- The full arm depends on private skills, instruction text and cached MCP schemas.
- The raw request bodies hold that private text verbatim. Examples:
  - The Codex first body opens with the operator's personal AGENTS.md.
  - The handoff summary carries absolute home-directory paths (`…/workspace/harness-cost-bench/bin/python3`).
- So "publish raw logs" and "keep the loadout private" conflict. `requests.jsonl` itself carries only usage and sizes, which is fine; `logs/bodies/` is not.

**Remediation.** A public synthetic loadout is acceptable *if* it matches, per harness, what drives cost.
- **Required matches:**
  - skill count;
  - catalog bytes as rendered (names, descriptions, paths);
  - SKILL.md body sizes, in case any are read;
  - instruction-file bytes;
  - MCP server count;
  - tools per server;
  - per-tool schema-bytes distribution, not just the total (Codex's cost scales with schema bytes; deferred harnesses scale with names and descriptions);
  - description bytes.
- **Acceptance test:** re-run both loadouts and show first-request token counts within ±2% per harness, and per-task cost within the rep-level CI.
- Publish only the synthetic bodies, and ship v1's private-loadout numbers alongside as "measured on a private loadout of matched size".

## R8 — minor — pricing provenance: reconciled, with two caveats

**Evidence.**
- `price_row` equals OpenRouter's `usage.cost` on all 1,533 priced rows. The maximum absolute difference is 6.9e-18; the sum is $36.9097 both ways.
- `spend.json` at $39.3027 equals $36.9097 + $2.3929, the smoke and aborted logs:
  - `old/requests.real-smoke` $0.7359
  - `aborted1` $1.0711
  - `aborted2` $0.4515
  - `aborted3` $0.1344
- The snapshot also holds $4.14 of r8/r9 smoke runs, so the rep-0 data behind the headline cost $32.77. "Total spend $39.30" is correct but should be broken down.
- **Caveat 1.** OpenRouter's figure is OpenRouter's own computation from the same usage block, not an independent invoice. It confirms the rate card, not the token counts.
- **Caveat 2.** No row exercises the Haiku >100k tier or 1h writes, so those branches of `price_row` are untested.
- **Stale docstrings.**
  - `analyze.py:9-10` says the runs were served by a subscription (OAuth).
  - `meter_proxy.py:10-16` describes OAuth and identity injection.
  - Both contradict REPORT.md.

**Remediation.**
- Break the spend down.
- Fix the docstrings.
- Name the rate-card source and date (the docstring does) in REPORT.

## R9 — major — wording and figures that overreach

| Where | Claim | Problem |
|---|---|---|
| REPORT.md:59 | "cost ratios hold task by task (23–24 of 24 paired tasks)" | Holds warm only. Cold vs OpenCode it is 3/24 (R2), and Claude Code's 1 loss is a crash (R4). |
| REPORT.md:61 | "LOP's advantage narrows" (cold) | It reverses vs OpenCode (R2). |
| REPORT.md:59 | "±1–2 tasks of noise" | Wilson ±18 pp ≈ ±4 tasks (R3). |
| REPORT.md:27, fig2 C | Codex "11.7M input tokens per run" caused by schemas | Mostly the compaction loop (R1). |
| REPORT.md:21, fig1 | Codex "≈41×", "$3.74 cache-corrected" | n=4, with a post-hoc correction on 1 run (R1). |
| REPORT.md:48 | "Each checker was validated" | Gold = 1 / input = 0 only. Two pptx tasks reject ordinary content-correct edits for unrelated geometry (R4); impossibility was not demonstrated. |
| REPORT.md:55 | auto cache "in practice only Codex" | Claude Code had 42 injected calls (R6). |
| fig1 subtitle | "15 skills … 480 tools" | Per-harness skills 15–33, tools 480–496 (R6). |
| fig1 footnote | Cold "reprices each run's first-call cache reads" | It is the first *main* call (R2). |
| fig1 table | Point estimates only | No CIs. Pass column is shown as if it ranks harnesses. |
| REPORT.md:31 | LOP "stable cache prefix" gives 95% hits | 95% is mostly cross-run warmth from back-to-back runs (R2). Own-run hit share is not separated out. |

**Remediation.**
- Make each of the edits in the table.
- Add CIs to fig1, using error bars from the task bootstrap.
- Put the cold column next to warm in the bar chart, not only in the table.

## R10 — minor — smaller rigour items

- **Mixed run order.** The fixed task order is the same for every harness, with harnesses interleaved per task. That is fair in timing but confounds task identity with position (the first 2 tasks are always cold). In v2, randomise the order per rep.
- **Disk pause.** One full-arm run per harness sat behind a 32-minute disk pause and was cold. Log `gap_since_prev_same_harness_s` per run.
- **Scores.** `score >= 1.0` is binary. Fine, but state it. OSWorld checkers can return fractional scores, though none here did.
- **Smoke reps.** r8/r9 smoke reps sit in the frozen `requests.jsonl` and `runs.jsonl`. report.py filters `rep == 0`, but figure_cost.py filters `run_id.endswith("r0")`, which would also match a future r10, r20 and so on. Use `rep == 0` everywhere.

---

## Prioritised v2 design: what a hostile but fair reviewer would demand

1. **Pre-register before any v2 run.** Commit, with a timestamp and hash, the following:
   - the headline metric, paired geometric-mean cost ratio per task vs a named reference harness, reported **separately for cold-start and steady-state**;
   - the secondary metrics: pass rate and cost per pass;
   - the task list and selection seed;
   - the rep count;
   - the exclusion rules (infrastructure errors are rerun up to 2×, then marked invalid and excluded *pairwise*);
   - the analysis script.
   Report deviations.
2. **Task selection.**
   - From the OSWorld @ 8903ebe pool, drop GUI-only domains.
   - Apply the offline-feasibility gate: a reference solution built with the offered toolchain scores 1.
   - Then draw a stratified random sample (by domain and short/long) with a published seed. Target 48–60 tasks.
   - Report how many pass for all harnesses and how many fail for all. Exclude all-fail tasks from pass-rate comparisons, but keep them in cost.
3. **Reps.**
   - At least 3 reps per (harness, task). Five is better for the pass rate.
   - The observed sd of log cost ratios is 0.17 (omp), 0.33 (OpenCode) and 0.69 (Claude Code), so 48 tasks × 3 reps resolves ratios to about ±5–10%.
   - The pass rate needs about 100 paired tasks before a ±9 pp half-width; say so rather than rank on it.
4. **Order.** Randomise the harness × task order within each rep, using a block design so every harness sees every position. Log the timestamp and the previous same-harness run end.
5. **Cache policy.** Report both regimes.
   - **Cold:** the first request of each run cannot read a prior run's cache. Either wait more than 6 minutes since that harness's last request, or reprice with the stricter definition in `reviews/cold.py`, which must agree with the measured cold runs.
   - **Steady-state:** the same harness runs back-to-back within the TTL, after one discarded warm-up.
   - Publish own-run vs cross-run cache-read shares.
6. **Configuration.**
   - Each harness at its documented direct-API defaults, with any overrides listed.
   - A per-harness table of model, effort/thinking, max_tokens, side-call models, permission flags, tool search and proxy interventions (cache injection count).
   - LOP both with and without the hosted classifier, with classifier tokens and cost metered.
   - Codex either with correct model metadata (verified: no compaction, thinking on) or excluded.
7. **Statistics to report.** For each paired comparison:
   - the geometric-mean ratio and median ratio, with task-cluster bootstrap 95% CIs (resample tasks, keep reps nested);
   - a Wilcoxon signed-rank test on log ratios;
   - win counts;
   - Holm correction across comparisons.
   For pass:
   - Wilson CIs;
   - McNemar on per-task majority outcome, or a mixed logistic model with task random effects.
   Also: cost per pass with a bootstrap CI, and a Friedman test with mean ranks.
8. **Loadout.** The public synthetic loadout from R7, rendered identically into every harness, with a wire-level check that each harness received it. Optionally a private-loadout replication, numbers only.
9. **Publish.**
   - Raw `requests.jsonl` and `runs.jsonl`, with all reps including invalid ones and a `valid` flag.
   - Synthetic-loadout request bodies.
   - Kept artefacts.
   - The tasks.json subset and seed.
   - Harness versions and the OSWorld commit.
   - The proxy, runner, derive and figure scripts. One command, logs → every table and figure.
   - The pre-registration.
   - A conflict-of-interest statement.

## Top 10 (summary)

1. **R1 (blocker).** Codex full-arm cost is a compaction loop caused by an unknown model id, plus no thinking. The 41×, $3.74 and 11.7M figures do not measure schema bloat alone.
2. **R2 (blocker).** The cold-start ranking reverses LOP vs OpenCode: geo ratio 0.81 [0.77, 0.87], OpenCode cheaper on 21/24 tasks. LOP vs Claude Code is about even, 1.03 [0.76, 1.21]. The warm advantage is cross-run cache warmth.
3. **R3 (major).** Pass rates are statistically indistinguishable (McNemar p ≥ 0.5; CIs ±18 pp). Cost ratios are robust to task sampling, but within-cell variance is unmeasured.
4. **R4 (major).** The Claude Code eb03d19a "fail" is a terminated run (exit 143, 2.6 s) whose cause needs adjudication, not automatic invalidation. It is also the one task in Claude Code's 23/24.
5. **R4 (major).** 20/24 tasks do not discriminate in the observed run. 73c99fb9 and 9cf05d24 reject ordinary content-correct edits because of examine_shape geometry; this is not an impossibility proof. d53ff5ee is ambiguous. 1e8df695 hinges on rounding. Selection is undocumented by an interested author; outcome-based selection is unproven.
6. **R5 (major).** figure_data.json, cold_cost and the Codex correction have no producing script.
7. **R6 (major).** The loadout is not matched: OpenCode got no instruction text, LOP got a different file 5.7× larger, and skills range 15–33. The classifier-on v2 LOP path must never be mixed with v1.
8. **R7 (major).** Request bodies contain private instruction text and home paths. A public synthetic loadout must match per-harness skill count, catalog bytes, instruction bytes, tool count and per-tool schema bytes, validated by first-request tokens within ±2%.
9. **R9 (major).** Wording overreaches: "narrows", "23–24/24", "±1–2 tasks", "each checker validated", "only Codex", "15 skills / 480 tools".
10. **R8 (minor).** Pricing reconciles exactly ($36.9097 both ways; spend $39.30 = data + $2.39 smoke/aborted). Stale OAuth docstrings remain.


## Correction addendum — 2026-11-03, before v2 preregistration

The 180/180 result proves reproducibility of recorded scores, not that all 180 solutions are correct or the checkers feasible. Gold-round-trip failures diagnose evaluator/toolchain sensitivity; they do not prove no agent can solve the task offline. Geometry evidence proves ordinary correct edits can fail, not that every offline normalization is impossible. The universe is 368, multi_apps 101, per the pinned selection manifest (the earlier 369/102 was wrong). Recount of four full-arm harnesses gives 14 all-pass, 6 all-fail and 4 mixed tasks. Undocumented sampling does not prove outcome-based cherry-picking. Exit 143 does not establish external infrastructure fault or its cause; no automatic invalid/rerun ruling is justified. R4 remains major for missing eligibility validation, semantic validity and termination adjudication, not a claim of proven unsolvability.

Codex's repeated compaction and missing-metadata warning are observed; the exact causal contribution of fallback context metadata versus translation/bridge behavior was not isolated. V2 must test the corrected catalog/bridge configuration, not treat the inferred mechanism as established. Likewise LOP replay-fix pilot outcomes 1/10 versus 0/11 cannot demonstrate causal quality improvement.
