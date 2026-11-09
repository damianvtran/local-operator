# Headline preregistration v2 — frozen 2026-11-09 before headline dispatch

Version: `HEADLINE-2026-11-09-v2`. This is a new version, not an edit of v1.
The retained v1 text below is the unchanged baseline; this prospective amendment
supersedes its admission, execution-count/order and obsolete qualification-status
statements only. Model, seven harnesses, price card, cache rules, $5/trial and
$100 campaign caps, metrics, statistical plan, refusal policy, measurement
conditions and disclosures remain identical. DEV pilot data remains excluded.

## V2 admission and sample

All 28 tasks in the frozen 30-task subset minus the two reserved DEV tasks are
in the sample and eligible, subject to the driver preflight. The 27 strict
admissions and sanitize-git-repo's disclosed-graded oracle are recorded in
`coding/container-s4/dev-attest-255f6035/oracle-admission-28-a3f8f1f8.md/.json`
and `coding/container-s4/ORACLE-SWEEP-a3f8f1f8.json` in the retained campaign workspace.
The admission predicate, applied identically to every arm and every task, is:

> admitted if the oracle row is graded with reward 1.0 AND post_agent_ok AND every disclosed hidden match is class expected-output-fixture AND the host pre-inference receipt is clean for that trial AND the egress fence is clean; isolation_ok need not be true for that disclosed-graded path.

This gives up the strict guarantee that no hidden digest appears anywhere after
agent execution. It does not give up pre-inference isolation: the clean host
pre-inference receipt and clean egress fence are the safeguards. Only disclosed
expected-output fixtures may match; refusing matches remain disqualifying.
For sanitize-git-repo the strict clause cannot be satisfied by any successful
arm, including the reference solution (reward 1.0). Keeping it would exclude the
task for policy reasons rather than capability. No task-specific payload or
arm-specific exception is introduced. Qualification must be attested, not
inferred from a successful reference score alone.

## V2 execution and checkpoint contract

Rep 1 label 60: 28 tasks × 7 harnesses = 196 cells. Task order is the frozen
28-task subset order below; inside each task use the seven-harness order in §3.
Dispatch through the reviewed driver **one trial at a time**, never parallel
trials. Order is fixed, never outcome-dependent. Append each trial
and checkpoint its next (task, harness); never redispatch an existing run ID.
A permitted infrastructure rerun is retained as a separate row and attempted
once immediately after its void; refusal is an outcome, not a rerun.

Only after rep 1 completes, decide whether to run rep 2 (label 61) in the same
fixed order using **cost and wall clock only**, never rewards. Record arithmetic
before rep 2: observed rep-1 mean × 196 projects rep-2 cost; require the total
projection plus rerun headroom to fit $100. Reserve a further rep-1 observed
cost as headroom (projected total = 3 × rep-1 spend); require projected remaining
wall clock to fit the available execution lease. Otherwise retain rep 1 only.
No second-rep decision may be selected using outcomes. The statistical methods
remain the v1 methods; one rep cannot estimate within-cell variability.

Start heavy work only with at least 21 GiB free, abort below 17 GiB, hard floor
15 GiB. Treat disk stalls as infrastructure rather than task outcomes. Stop on
version drift, source bugs, 401/402/403, or unsafe known-plus-reserved spend.
Never patch reviewed executable source mid-run.

## V2 source binding and publication

Reviewed source: `ee5647c8c28105002692b324280ccbb7a499c7a6`, clean and terminal
through `radient-dev/LOP-EDGE-REVIEW-R22.md`. R22 establishes the 19-member
CONTAINER_CLOSURE is unchanged and publication.py lies outside it; verify this
again when binding the source. Bind a new r4 oracle attestation for all 28 and
refresh runtime/toolwire only if their inputs changed. Preflight must pass for
all seven arms against this JSON and a live /health showing caps 100/5 before
any paid call. Any refusal stops this lease.

Publication must call the R22-clean `benchmarks/harness_cost/publication.py::publish_request`
projection (or the identical reviewed contract) over raw meter records. It must
project the S13 provenance fields, including settlement derivations, no-charge
markers, rejection metadata and conversation shaping. After the first completed
trial, diff one published row against its raw log row and retain the comparison;
do not defer this check to the end of the matrix. Never publish raw error text.

## Retained v1 baseline (verbatim; overrides above take precedence)

# Headline preregistration — Terminal-Bench 2.1 seven-harness cost/quality comparison (frozen 2026-11-08)

**Version:** `HEADLINE-2026-11-08` — a NEW versioned document. It does not rewrite
`PREREGISTRATION.md`; the amendment audit trail R20 asked for is preserved in
[Appendix A](#appendix-a--retained-prior-text-superseded-92-and-72-text). The four R20
publication minors were closed after this freeze and before any headline trial, without touching a
metered byte — [§13](#13-publication-record-amendments-s13-and-its-r21-remediation-2026-11-08).

**Status:** frozen for **planning and admission only**, amended at §13 (S13, 2026-11-08). No
headline trial has run at this version. Every number below is derived from a retained artifact
named beside it; none is typed from memory.

| Item | Value | Source |
|---|---|---|
| Bench source head | `a3f8f1f88e7dad179d09a6ebce65f5d98494df28` on `bench/harness-cost-lop-edge` | `git rev-parse` in the lane worktree |
| Amended after freeze | S13 (§13), the child of `3f6b257854d` | `publication.py`, `test_kit.py` and docs only; no `CONTAINER_CLOSURE` member touched, so the S4 record and preflight HARDLOCK still bind |
| Reviews | clean and terminal through R20 | `radient-dev/LOP-EDGE-REVIEW-R20.md` |
| Frozen 30-task subset | `97aac6092add53b8c65232f69e0d373eb94fb39c072b63eadef752707c020459` | `sha256 subset.json` (`complete: true`, `target_count: 30`) |
| Oracle sweep record bound | `428dd2c7322455ea2d6e15d0176d9d5f433548d3241c31b2391b15393fc096f7` | `sha256 container-s4/ORACLE-SWEEP.json`, `source_commit 255f6035…` |
| Machine-readable preregistration | `PREREGISTRATION-HEADLINE-2026-11-08.json` (this directory) | built by `container-s4/dev-attest-255f6035/build_headline_prereg.py` |
| Subject of the comparison | the seven harnesses in §3, one workload (Terminal-Bench 2.1), one model (§4) | `driver.HARNESS`, live price card |

---

## 1. Sample: 30 minus 2 DEV tasks = 28, frozen

The scored population is the frozen 30-task Terminal-Bench 2.1 subset (`subset.json`,
`complete: true`, seeded `20261103`, `candidates: 57`) **minus the two prospectively reserved
development tasks**:

* `git-leak-recovery`, `vulnerable-secret` — reserved development tasks
  (`coding/DEV-HOLDOUT-RESERVATION.json`, `reserved_development_tasks`). They are **excluded
  from the headline sample** and may not be relabelled held-out. `DEV-HOLDOUT-RESERVATION.json`
  records `held_out_task_count: 28` and lists the same 28 identifiers this document freezes.

**The 28-task headline sample (frozen; subset order):**

```
password-recovery, dna-insert, feal-linear-cryptanalysis, dna-assembly,
schemelike-metacircular-eval, large-scale-text-editing, torch-pipeline-parallelism,
circuit-fibsqrt, headless-terminal, financial-document-processor, kv-store-grpc,
fix-code-vulnerability, nginx-request-logging, build-pmars, overfull-hbox,
sparql-university, largest-eigenval, make-doom-for-mips, extract-moves-from-video,
llm-inference-batching-scheduler, merge-diff-arc-agi-task, tune-mjcf, build-pov-ray,
adaptive-rejection-sampler, sanitize-git-repo, regex-log, pytorch-model-cli, code-from-image
```

Derivation, reproducible in one command:

```
python3 -c "import json;s=json.load(open('coding/subset.json'));d=set(json.load(open('coding/DEV-HOLDOUT-RESERVATION.json'))['reserved_development_tasks']);print([t['name'] for t in s['tasks'] if t['name'] not in d])"
```

### 1.1 Admission status at freeze (decided before any headline data exists)

The driver admits a task only with a current passing paid-topology oracle row
(`coding/driver.py` preflight: `reward == 1`, `graded`, `isolation_ok`, matching
`image_digest`/`task_checksum`). Seven of the 28 do not have one **today**:

| Task | Sweep status | reward | isolation_ok | Attempts (run ids) | Why it is not admissible |
|---|---|---:|---|---|---|
| `sparql-university` | FAIL | 0.0 | true | `requal__sparql-university__1794127472`, `…1794128930` | oracle scored 0 twice at `255f6035`; no exception, exit 0, so not an infrastructure void |
| `extract-moves-from-video` | FAIL | 0.0 | true | `requal__extract-moves-from-video__1794127855`, `…1794129001` | same |
| `merge-diff-arc-agi-task` | FAIL | 0.0 | true | `requal__merge-diff-arc-agi-task__1794127978`, `…1794129075` | same |
| `build-pov-ray` | FAIL | 0.0 | true | `requal__build-pov-ray__1794128126`, `…1794129254` | same |
| `regex-log` | FAIL | 0.0 | true | `requal__regex-log__1794128522`, `…1794129403` | same |
| `code-from-image` | FAIL | 0.0 | true | `requal__code-from-image__1794128834`, `…1794129464` | same |
| `sanitize-git-repo` | PASS | 1.0 | **false** | `requal__sanitize-git-repo__1794128459` | `hidden-digest-appeared-after-agent-phase`: the post-agent scan would refuse **every** successful arm (model or oracle), so the task is unmeasurable in this topology (SLICE-S9 §1b) |

**Admitted at freeze: 21 tasks × 7 harnesses = 147 trials.**
The other seven stay in the sample (they are not removed), are non-dispatchable until
re-qualified, and their re-entry needs a new versioned preregistration (§8.4).

## 2. What is being compared

One workload, seven harnesses, one model, one rep: the **cost per successful task
opportunity** and the **quality (checker success) fraction** of each harness on the same 21
admitted tasks. No pooled seven-harness winner metric, no per-harness cap or prompt change,
no task substitution after outcomes.

## 3. Harnesses: seven, pinned

Pinned in `driver.HARNESS` (reviewed source at the frozen head); runtime attestation
`runtime-attestation.json` (this freeze) records each binary's version, Linux verification and
per-file sha256 over the frozen toolchain; `toolwire-attestation.json` records the observed
wire for all seven in both capture rounds (A and B).

| Harness | Adapter | Pinned version | Source revision |
|---|---|---|---|
| `claude-code` | `FrozenClaudeCode` | 2.1.274 | vendor binary, frozen toolchain |
| `codex-best` | `FrozenCodex` | 0.147.0 | vendor binary, frozen toolchain |
| `opencode` | `OpenCodeV2` | 2.0.27 | vendor binary, frozen toolchain |
| `local-operator` | `LocalOperator` | `500883ef-v0.94.22` | released v0.94.22, git `500883ef56ca504325ffd80c5b90bf2435d00e3f` |
| `omp` | `Omp` | 18.11.2 | vendor binary, frozen toolchain |
| `hermes` | `FrozenHermes` | 0.22.6 | verified source `f0d94882e89ab8d4098b5bb2f166adfb9ee0a0ab` (peeled by `v0.22.6` and `rc.3-v0.22.6`) |
| `dsh` | `Dsh` | 0.2.0-rc.2 | frozen toolchain |

All seven share one metering path (`meter_proxy`), one public loadout, one classifier-ledger
route, and one tool-output cap (§5). The `local-operator` arm is the released v0.94.22 build,
not this benchmark fork; the fork only supplies the measurement harness.

## 4. Model, price card and cache rules in force

* **Model:** `claude-sonnet-5-5` (upstream `anthropic/claude-sonnet-5.5`), billing tier
  `default`, output limit 128,000 tokens.
* **Price card captured at freeze time:** `https://openrouter.ai/api/v1/models`, fetched
  `fetched_at 1794196080.7145338`, response sha256
  `d94d8370997bb6a32c8265d0c5368d98c121355ebed0c51698539d81e32bdc13`; converted by the head's
  own `meter_proxy.rate_card_from_models_listing`.
* **Rates (USD per million tokens):** input **2.0**, output **10.0**, cache write **2.5**,
  cache read **0.1**; reservation input rate **2.5**. Identical to the card the pilot ran
  under (`price-dd0cb631/rate-card.json`) — the live card is unamended, so the pilot's costs and
  this freeze's prices are on one basis.
* **Cache rules in force:** each trial runs in a fresh container with a fresh home/workdir;
  no namespace salt and no cross-trial cache namespace are used, so a trial's cache reads can
  only come from its own session. Within-session caching stays enabled and is priced from the
  card (write 2.5, read 0.1). A stated `usage.cost` always wins; a complete response with no
  stated cost settles from its own token counts at this card (§7). The shared prefill
  continuation (§7.3) may create a fresh cache write on the affected request and is priced
  normally.

## 5. Repetitions, order and trial limits

* **Reps: one** (`--rep 60`, the rep label in every run id). 21 tasks × 7 harnesses × 1 =
  **147 trials**. One rep means the task is the resampling unit only: within-cell variability
  cannot be estimated from this sample (`PREREGISTRATION.md` §8, Terminal-Bench clause).
* **Order:** the frozen subset order for tasks, `driver.HARNESS` order for arms inside one
  invocation; a task block is all seven arms for one task. Admission is per block: a block
  starts only while the campaign ledger's remaining balance covers it (§6.3).
* **Per-trial limits (all arms identical):** campaign and trial money caps of §6; wall clock
  per request `BENCH_MAX_REQUEST_S=900`; request body ≤ 262,144 bytes input; output ≤ 128,000
  tokens; one fresh home/container/session per trial; 15 GiB disk floor.
* **Metering:** every request on the served route goes through the one meter; classifier
  traffic is on the same shared ledger. No free or invisible retries.

## 6. Cost caps and the arithmetic behind them

### 6.1 Per-trial cap: **$5.00** (hard, meter-enforced)

The pilot's costliest light trial was **$0.106588** (codex-best / vulnerable-secret, 14 valid
trials, `coding/PILOT.md`). $5.00 is **46.9×** that maximum. The pilot measured *light* tasks,
so its maximum is a floor for a long task, not a ceiling: a task 10× heavier than the pilot's
worst would cost ≈$1.07 and stay well inside $5. The cap was never approached in the pilot
(no trial came within an order of magnitude of it: the observed maximum is 2.1% of the cap), and
$5.00 is the value the meter, the container-S4 `/health` parity row and the pilot all ran 
under, so changing it would break that parity without evidence.

### 6.2 Campaign cap: **$100.00** (hard, meter-enforced total)

It is the Terminal-Bench envelope already frozen in `PREREGISTRATION.md` §9.3, and it fits the
operator's authorisation:

```
operator authorisation (programme)             $800.00
  − development/preflight envelope               $60.00
  − OSWorld headline workloads                  $540.00
  − Terminal-Bench exploratory  ← this run      $100.00
  − infrastructure retry envelope                $80.00
  − uncommitted safety reserve                   $20.00
                                              = $800.00
spent to date ≈ $1.40 → remaining ≈ $798.60; this run's cap = $100.00 (12.5% of remaining)
```

Measured spend behind the ≈$1.40: the 14-trial DEV pilot's lane total is **$0.728638**
(`coding/PILOT.md`, 14 valid trials; $0.683756 of it is the 14 rows' billed cost and $0.044881
the adapter/unscored qualification rows sharing the request log), plus the earlier zero-paid
qualification probes; the operator's figure for everything spent at freeze time is ≈$1.40. The
$60.00 development envelope therefore still holds ≈$58.60 unspent, and this run needs none of it
(a preflight is zero-paid).

### 6.3 What the cap binds, and the headroom it leaves

| Scenario | 147 trials (21 admitted) | 196 trials (all 28) |
|---|---:|---:|
| pilot pooled mean $0.048840/trial | **$7.18** | $9.57 |
| every trial at the pilot's max $0.106588 | **$15.67** | $20.89 |
| every trial at the frozen $0.45 planning figure (§9.1) | **$66.15** | $88.20 |
| mean trial cost at which the $100 cap binds | $0.6803 (13.9× the pilot mean) | $0.5102 (10.4× the pilot mean) |

* **Rerun headroom, stated arithmetically:** the §9.2 rule allows one infrastructure rerun of a
  cell (≤7 trials). At the pilot's mean a full-cell rerun costs $0.34; at the pilot's observed
  maximum $0.75; at the frozen $0.45 planning figure $3.15. Against a worst-case nominal $66.15
  for the whole block, five full-cell reruns still fit inside $100. A rerun is admitted from the
  *actual* remaining balance, never from a nominal reservation.
* **Stop rule:** if the ledger's remaining balance cannot cover a complete seven-arm block at
  the frozen $0.45 planning figure (7 × $0.45 = $3.15), the campaign stops at the last completed
  block and publishes an explicitly **incomplete exploratory** matrix (`PREREGISTRATION.md`
  §9.3). A budget stop is never a reason to drop one arm's cell.
* **If the cap binds in practice:** if the realised mean exceeds ≈$0.68/trial, the campaign
  pauses and reports rather than truncating a comparison (`PREREGISTRATION.md` §9.1: a cap that
  is usually binding means replan, not truncate).

## 7. Failure, refusal and settlement rules (S10–S12), as measurement conditions

These are properties of the measurement, applied identically to all seven arms; they are
frozen here as the conditions under which the headline numbers are produced.

### 7.1 Unbilled but complete responses settle from their own tokens

A **complete** response with no `usage.cost` settles deterministically at this run's frozen
rate card from its own counts (`input_tokens`, `output_tokens`, plus any cache counts), marked
`settled_from_tokens: true` with the derivation on the row. It is a bill: it neither voids the
attempt nor blocks later admissions. A response that ended before completion never settles this
way — its partial usage is a floor and keeps the unknown-spend latch. A stated `usage.cost`
always wins.

### 7.2 A rejected request is $0, not an unknown charge — 400/422 only

A request the provider **rejects at validation** — status **400 or 422 only**, stating no cost
and no token counts, with a readable error body naming its type — settles at exactly **$0**
(`settled_no_charge: true`), neither voids nor blocks, contributes $0 to the trial's cost, and
the trial's quality outcome is exactly what native grading says. The settled set is 400 and 422
and nothing wider.

**Infrastructure statuses 401, 402, 403 and 429 never settle at $0.** A revoked key, an
exhausted balance, a lost permission and a provider rate limit are the campaign's environment
failing, so with no stated bill all four latch **and** void for one rerun exactly as a 5xx: a
mid-run credential or credit failure must stop the campaign rather than let it finish with an
unmeasured `$0` ledger reported as complete. One that **does** state a cost or token counts is
priced at that amount and voided for one rerun **without** latching (S13 correction of R20
MINOR 2; the superseded wording is retained in Appendix A): the classification decides the
void, the amount decides the latch, so a bill the meter can read never blocks the campaign. The
bounds keep the pre-S11 latch: a 2xx without a priceable bill, a
400/422 that *does* state a cost or token counts, any unpriceable usage field, a 5xx, any other
4xx (404, 413, …), an unreadable error body, and a 4xx delivered as a stream.

Every money reader (`meter_proxy`, `runner_v2.stated_amount`, `coding/driver.row_cost`, and the
publishing rig's `coding/billing.py`) carries the same literal `frozenset({400, 422})`, so no
lane can read an infrastructure row as a zero. **The headline run must be pointed at the rig
tree that holds this `billing.py`** (`~/workspace/harness-cost-bench/coding/`): a run against
another rig checkout would republish rejections as unknown spend.

### 7.3 One shared prefill continuation for an assistant-ended conversation

The served route refuses assistant prefill for this model (the pilot's measured 400). When an
outgoing `v1/messages` body would end on an assistant turn, the meter's single shaping path
appends exactly one turn `{"role": "user", "content": "Continue."}` and flags the row
`prefill_continuation: true`; a body already ending on a user turn is forwarded unchanged. No
harness branch exists: every arm's request passes the same code, and the change converts only
dead requests into live ones. It is **an input rewrite and is disclosed as such**; it is not
an unmodified native prompt. Assumption recorded: this is safe only because the served route
rejects prefill, so a request that deliberately prefilled cannot succeed either way.

### 7.4 Other primary-failure classes

Checker failure, invalid output, tool misuse, refusal/abandonment, cost cap, request cap,
timeout and harness-internal crashes remain primary failures (Y=0) with cost and wall time
retained. An identifiable external infrastructure defect voids at most **one** rerun of a cell;
no rerun because a cost is high or a score is low. A discovered broken checker is a dataset
incident: freeze the task across all arms, retain the data, adjudicate independently, report
with and without it.

## 8. Prospective policies fixed before any headline data

### 8.1 A task where every arm is refused stays in the sample

If all seven arms fail a task — including by refusal, abandonment, cap or timeout — the task
**stays in the sample**, is published as a dataset/refusal finding with its per-arm evidence,
is **never removed after seeing outcomes**, and counts **Y = 0 for every arm**. It is not a
reason to drop the task, reweight the stratum, or rerun it for a better score.

### 8.2 Quality gate and cost claim

`Y_thr ∈ {0,1}` from the frozen checker. `quality_h = (1/n) Σ_t (1/R) Σ_r Y_thr`;
`mean_cost_h` likewise; **cost per successful task opportunity** `E_h = mean_cost_h / quality_h`
(infinite when quality is 0, never discarded). Primary: `E_LOP / E_comparator` per comparator,
with the quality **non-inferiority** gate `quality_LOP − quality_comparator > −0.05`
(**5 percentage points absolute**). A cost-effectiveness claim additionally needs the upper
bound of the cost-per-success ratio below 1; both conditions must hold per comparator. A low
cost caused by errors, abandonment or caps cannot earn a "more cost-effective at comparable
quality" claim.

### 8.3 Statistics (unchanged from the frozen plan — no new statistics)

Task-level paired bootstrap, **100,000 deterministic draws**, seed `20261103`, with the
generator/library version recorded; one rep means task resampling only (no within-cell
resampling), so this sample cannot measure within-cell variability. Ordinary two-sided 95%
paired intervals are descriptive. For the six preregistered cost-plus-quality claims use
conservative Bonferroni bounds across **12 one-sided gates** (six NI, six cost) at
`alpha = 0.05/12`: lower quality-difference bound > −0.05 and upper E-ratio bound < 1. Show
finite-sample sensitivity; never substitute a more permissive test after seeing data.
Supplementary: paired Wilcoxon signed-rank on per-task mean log-cost ratios with Holm across
the six comparisons, sign/win counts, Friedman rank summary; exact McNemar on
majority-of-three is a sensitivity analysis only (and is not available at one rep).
**Power:** 21 (or 28) tasks is small for a 5-point NI margin; the frozen planning figures put
80%-power requirement at ≈485 (discordance 0.10) to ≈1,450 (0.30) independent tasks. **A null
result is expected to be inconclusive on quality parity.** That is reported honestly; the NI
margin is never widened to manufacture a winner.

### 8.4 How the seven qualification-gap tasks can join

Their entry is prospective and specified now: a **new versioned preregistration** after a
zero-paid re-qualification round that produces a passing paid-topology oracle row for the task.
It enters as new data with complete seven-arm blocks; no arm's cell is ever filled in later, and
no task is swapped in because a losing arm failed elsewhere. Until then the published analysis
is the 21-task admitted block, with the gap and its receipts disclosed.

### 8.5 DEVELOPMENT data is never mixed into headline results

The 14-trial development pilot (2 DEV tasks × 7 harnesses, `coding/PILOT.md`,
`coding/pilot_results.json`) is **development data**. It is never pooled, appended or compared
with headline rows; it informs cost planning only (§6.3). Its two tasks are excluded from the
sample by construction (§1). The pilot's `pilot_results.py` lane total ($0.728638, including
$0.044882 of adapter/unscored spend) is a DEVELOPMENT figure and appears in no headline cost
cell.

## 9. Known documentation gaps carried into this freeze (disclosed, not fixed here)

R20 recorded four publication minors and left them open. Three involve reviewed source or
published text that this lease must not change; they are disclosed here so a public reviewer
sees them rather than discovering them. **Amendment at S13 (2026-11-08, before any headline
trial): items 1–3 are fixed in reviewed source and item 4's retention is complete — see §13,
which supersedes this section as a statement of current state. The text below is retained as
the freeze-time record.**

1. **Discovery-row attribution is wrong.** `ADAPTERS.md` (and `SLICE-S9.md` §S12 deviation 1)
   say the 37 typed-404 discovery rows "carry `run_id: "probe"`". Measured over
   `logs/requests.jsonl`, only **7** of the 37 carry `probe`; the rest are
   `clean__hermes__37608790__r40` (8), `full__hermes__37608790__r41` (8),
   `coding__probe-hermes__hostcheck__r0` (7) and `dev__hermes__lifecycle__r43` (7). The claim the
   sentence supports — none of the 37 is `write_row` output, since none carries `channel`,
   `priced_model`, `reserve_usd` or `n_messages` — **is** true across all 37 rows and five run
   ids. Only the attribution is false.
2. **"Latched … never settles" is imprecise.** For an infrastructure status (401/402/403/429)
   that *states* a cost, the stated cost is priced and the row does not read as zero — the
   infrastructure classification governs the void, not the amount. The correct statement is
   "never settles at $0", not "never settles".
3. **`n_messages` counts the injected continuation.** The published `n_messages` counts the
   prefill continuation turn (§7.3), while `prefill_continuation` is **not** in the publication
   allowlist (`publication.REQUEST_FIELDS`/`REQUEST_SCHEMA`). A published-snapshot reader
   therefore sees a message count one higher than the harness sent, on rows nothing in the
   public metadata marks as shaped. The disclosure exists only in the raw log. The two-entry fix
   (`prefill_continuation=(bool,)` plus its name in `REQUEST_FIELDS`) is a reviewed-source change
   and was **not** made in this lease.
4. §9.2's in-place rewrite (R20 MINOR 4) is handled here by **retaining** the superseded text in
   Appendix A rather than editing `PREREGISTRATION.md`.

Adjacent items recorded, not fixed: `lop_edge.py`'s classifier channel still latches `None` on a
bill-free 4xx (deliberate, disclosed); `meter_proxy.py`'s `is_count` branch is unreachable dead
code after the S12 path gate; the rig tree holding `billing.py` is untracked, so the S12 reader
fix does not travel with the commit (the run must be pointed at
`~/workspace/harness-cost-bench/coding/`).

## 10. Freeze register

Every artifact below was regenerated or rebuilt for this head; nothing is reused by assertion. Paths
are relative to `~/workspace/harness-cost-bench/` unless they name the benchmark worktree.

| Artifact | Path | sha256 |
|---|---|---|
| Headline preregistration (machine-readable) | `<worktree>/benchmarks/harness_cost/experimental/PREREGISTRATION-HEADLINE-2026-11-08.json` | `0440a2cede8e20f3a495505f9744c52a8b053cd5cec041acafc55b80e5a4a8d0` |
| Container S4 record (zero-paid, 33 rows, all PASS) | `coding/container-s4/coding-edge-evidence.json` | `270e9954f2fc696726f3e97117f00698f75657bb68711b208753f44f1c1e4544` |
| Container S4 evidence artifact (what the record hashes) | `coding/container-s4/CONTAINER-S4-EVIDENCE.md` | `fa740e7d19e6e645f74592ecc499529a7bcbb637f9be366470c9f9998c28b4b6` |
| Runtime attestation (7 harnesses) | `coding/container-s4/dev-attest-255f6035/attestations-headline/r2/runtime-attestation.json` | `08fd9a3d80d71c8021260a182593bf4a38e8005b9fb62ccb79ee4150b1148223` |
| Toolwire attestation (7 harnesses, rounds A+B) | `…/attestations-headline/r2/toolwire-attestation.json` | `5891a84821e29242b23d37ec606f17cf9006cb237d403dfb0d70817b885b2b1a` |
| Preregistered tool list | `…/attestations-headline/r2/prereg-tools.json` | `32e745a34473a6a9538d49550050525e0facdef0242ef8ce0ef413615bc5be6b` |
| Oracle attestation (21 admitted tasks, bound to sweep `255f6035`) | `…/attestations-headline/r2/oracle-attestation.json` | `b0c841a0f249bad02e7a9e96431a7a051ed6e29e452be4d8f476e13494993fec` |
| Live price card (freeze time) | `…/price-a3f8f1f8/rate-card.json` | `a85beb459fdf8d0f6054524c6d4617c954b27c1312b02301e4eadc7323b8b969` |
| Preflight transcript (no `--execute`) | `…/preflight-headline-2026-11-08.txt` | `25a69d1598c31f6d66702078d3a3c537302096cd2c50833555f6b6aea33e9017` |
| Oracle attestation refusal for the full 28-task sample (retained) | `…/oracle-28-refusal-2026-11-08.txt` | `8778af1a67f609cc46fce60c345237c23d4f5c259eaab1fcde660fecc7c02065` |

## 11. Preflight result and the honesty preconditions, as checks

**Preflight passes on the 21-task admitted block, all seven arms, at the frozen caps**
(`preflight-headline-2026-11-08.txt`):

```
meter /health: {"ok":true,"models":["anthropic/claude-haiku-5.5","anthropic/claude-sonnet-5.5"],"caps":{"total_usd":100.0,"trial_usd":5.0}}
Preflight passed; dispatch requires --execute
preflight exit=0
```

Negative control, same command before the S4 record was regenerated:
`driver.py: error: HARDBLOCK PAID CODING: no zero-paid container S4 record
(--coding-edge-evidence) matches the current metering closure` (exit 2). The two closure files
that differ between the archived `dd0cb631` record and this head are exactly `../meter_proxy.py`
(`de90279a…` → `e5d5d626…`) and `driver.py` (`dc7f0b90…` → `d78cd7d7…`) — the S11/S12 changes.

| Precondition | Result | Evidence |
|---|---|---|
| No DEVELOPMENT task in the sample | **PASS** | `set(prereg["tasks"]) ∩ {git-leak-recovery, vulnerable-secret} = ∅`; the 28 identify exactly `subset.tasks − reserved_development_tasks` (checked true) |
| Prices match the frozen card | **PASS** | prereg `input_usd_per_m 2.0`, `output_usd_per_m 10.0` = live card `prompt 2.0`, `completion 10.0`, cache write 2.5, cache read 0.1 |
| Caps match `/health` | **PASS** | prereg `trial 5.0 / total 100.0` = meter `/health` `{"total_usd":100.0,"trial_usd":5.0}` = the CLI args preflight compared |
| Publication allowlist covers the results schema | **PASS at S13** | `publication.py` publishes `n_messages` beside `prefill_continuation` and the other measurement-provenance fields (§13, R20 MINOR 3). At freeze this row read **FAIL — disclosed** |
| Per-call context capture is on | **PASS** | every meter row carries `usage.{input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens, cost}` (sampled from `container-s4/runs-a3f8f1f8/lop/meter.jsonl`) |
| Survivors and flags recorded | **PASS** | S4 rows `gold sentinel invisible before inference` (5 digests, 2 planted, `matched []`), `token-leak scan` (0 artifacts); every trial record carries `post_agent_flags`; every requalify row carries `survivors` (keep-alive `94c2168f…`, recorded never killed) |
| Egress rows complete | **PASS** | all 30 sweep rows `egress_rows_complete_with_end: true`; S4 rows `every package-egress connection has exactly one row with an end reason` PASS for LoP/omp/claude-code |

## 12. How the paid lease invokes the matrix (not run in this lease)

A near-copy of the DEV launcher `coding/container-s4/pilot-dd0cb631/pilot_run.py` (rig-side, outside
this repo) is the whole invocation: it starts the meter, the LiteLLM translator for the codex arm and
the driver, and reaps all three by exact pid. Change exactly five things in the copy:

```python
SRC  = S / "src-a3f8f1f8/benchmarks/harness_cost/experimental"   # this head's verified extract
A0   = DA / "attestations-headline/r2"                            # the freeze's four inputs
TASKS = tuple(json.load(open(PREREG))["tasks"])                   # 21, read from the freeze
menv["BENCH_BUDGET_USD"] = "100"                                  # was 40
cmd[cmd.index("--total-budget-usd")+1] = "100"                    # was 40
```

then

```
python3 headline_run.py --execute-paid --prereg \
  <worktree>/benchmarks/harness_cost/experimental/PREREGISTRATION-HEADLINE-2026-11-08.json
```

The launcher resolves `OPENROUTER_API_KEY` and the Radient bearer from the credentials store into
process memory (never argv, never an artifact), fetches a live card and refuses if the prices moved
off this freeze, and passes `--execute`. Env it sets for the meter (the single budget authority):
`BENCH_BUDGET_USD=100`, `BENCH_TRIAL_BUDGET_USD=5`, `BENCH_RATE_CARD=<freeze card>`,
`BENCH_MAX_REQUEST_S=900`, `BENCH_MAX_INPUT_BYTES=262144`, `BENCH_MAX_OUTPUT_TOKENS=128000`,
`BENCH_SERVICE_TIER=default`, `BENCH_MODEL=claude-sonnet-5-5`, `BENCH_SPEND_FILE=<run>/spend.json`,
`BENCH_LOG=<run>/meter.jsonl`, `BENCH_ADMISSION_WAIT_S=30`,
`BENCH_CURRENT_RUN_FILE=<run>/current_run`, `BENCH_PORT=<free loopback port>`.

* **Expected trials:** 21 tasks × 7 harnesses × 1 rep = **147** (196 when all 28 are admitted).
* **Worst-case cost:** bounded by the campaign cap at **$100.00** by construction (the meter refuses
  admission past it); nominal worst case at the frozen $0.45 planning figure $66.15; expected
  **$7.18** at the pilot's pooled mean.
* **Publish from the rig tree that holds the S12 reader:** `~/workspace/harness-cost-bench/coding/`
  (`billing.py`), or rejections republish as unknown spend (R20, §7.2 above).

## 13. Publication-record amendments (S13, and its R21 remediation), 2026-11-08

R20 left four MINOR documentation/publication items open. They are fixed in one commit on
`bench/harness-cost-lop-edge` — `fix(bench): publish what the meter measured, and correct three
false details`, the child of `3f6b257854d` — which changes **no metered byte**: no member of
`coding/driver.py`'s `CONTAINER_CLOSURE` is touched (closure diff empty), so the zero-paid
container S4 record and the preflight HARDLOCK stay valid for this head. The sample (§1), the
caps (§6), the measurement conditions (§7–§8), the metrics and the statistics are unchanged.

1. **A published row can now show how it was measured** (MINOR 3). `publication.REQUEST_FIELDS`
   and `REQUEST_SCHEMA` gain `prefill_continuation`, `settled_from_tokens`, `settled_no_charge`,
   `upstream_status`, `upstream_error_type`, `upstream_error_code`, `cost_derivation` and
   `incomplete`, so the published `n_messages` can be read for what it counts and a derived charge
   can be recomputed from the record. The derivation is the producer's own shape —
   `{tokens, rates_usd_per_m, usd}`, with `rates_usd_per_m` keyed by the same token fields it
   priced — closed and numbers-only, and a derivation that cannot be recomputed (a priced count
   with no rate, or a rate for a count the row does not state) is refused. The two upstream error
   strings are **not** published as text: the publish step maps a value from a reviewed vocabulary
   (`invalid_request_error`; no provider code is reviewed yet) and otherwise publishes
   `unclassified:<sha256 prefix>` of the raw value, keeping the raw string in the run's private log
   for out-of-band audit. `test_kit`'s `test_measurement_provenance_fields_publish` and
   `test_upstream_error_text_cannot_publish` assert both directions against the real producer.
2. **The discovery-row attribution is corrected** (MINOR 1). 30 of the 37 typed-404 rows do not
   carry `run_id: "probe"`; the measured attribution is in §9.1. `ADAPTERS.md` now states the
   field-shape evidence — no `write_row` field on any of the 37, across five run ids — instead of
   the false attribution, and the same sentence's path list now matches the seven paths measured.
3. **The latch sentence is corrected** (MINOR 2). An infrastructure status that *states* a cost or
   token counts is priced as stated, voided for one rerun and **not** latched. §7.2,
   `ADAPTERS.md` and `PREREGISTRATION.md` §9.2 now say so, and §7.2's superseded wording is
   retained verbatim in Appendix A.
4. **Retention is complete** (MINOR 4). Appendix A quotes the superseded §9.2 and §7.2 sentences
   verbatim with the revision, reason and timestamp of each, and `PREREGISTRATION.md`'s
   front-matter note carries the same retention in the document whose own rule demands it.

**R21 remediation** (the child of `12aae454867`, 2026-11-08). R21 blocked on two MAJOR findings
against the paragraph above, and both are fixed at the publication boundary: the derivation schema
was keyed by the card's own rate names (`prompt`, `completion`, …) that the producer never writes,
so no genuine settled row could publish; and the two upstream error strings were published
verbatim, so provider text could carry request content. The schema now matches
`settled_charge` exactly, and the publish step classifies the error values. Both change no metered
byte, task, cap, metric or statistic, and the container S4 closure still matches 19/19. R21's two
minors are closed as well: the S11 amendment now cites the commit that introduced its text, with
that commit's timestamp, and the retained quotations that are excerpts say so.

Why amend a frozen document at all: a disclosure that has since become false is a false public
statement, and §9 read as current state would publish one. Nothing here is a post-outcome change —
no headline trial has run at any version.

## Appendix A — retained prior text (superseded §9.2 and §7.2 text)

R20 MINOR 4: `PREREGISTRATION.md` §9.2 was rewritten in place, where the document's own rule
(`PREREGISTRATION.md:36`) requires that any amendment retain the previous version, its reason
and its timestamp. This appendix carries the audit trail in the frozen document; S13 completed it
with the S11 and S12 wordings verbatim and with this document's own superseded §7.2 sentence.

**Pre-S11 text** (as at `277d14b9adf`, before the S11 change), last sentence of the paragraph:

> "A response that ended before completion never settles this way: its partial usage is a floor,
> not the charge, and it keeps the unknown-spend latch."

— and, at that revision, §9.2 had **no** validation-rejection rule at all; any unbilled request
kept the latch.

**S11 amendment** (`c18c1dc1cf9`, 2026-11-08T20:41:37-05:00, prospective; the text stands
unchanged at the later revision `611ac9825bc`, 21:21:25), verbatim:

> "A request the provider **rejects outright** — a 4xx that states no cost and no token counts,
> with a readable error body naming its own type — settles at exactly **$0**, is recorded as
> `settled_no_charge: true` with that status and error type, and neither voids nor blocks: it
> contributes $0 to the trial's cost and the trial's quality outcome is exactly what native
> grading says. The bounds are the meter's, unchanged: a 2xx without a priceable bill, a 4xx that
> does state a cost or token counts, any unpriceable usage field, a 5xx, an unreadable error body
> and a 4xx delivered as a stream all keep the unknown-spend latch (ADAPTERS.md)."

**S12 amendment** (`a3f8f1f88e7`, 2026-11-08T22:14-05:00, prospective, before any headline data):
the settled set is narrowed to **400 and 422 and nothing wider**; 401/402/403/429 become
infrastructure faults that latch and void exactly as a 5xx. Same amendment: one shared prefill
continuation for an assistant-ended turn (§7.3) and the rig reader that reads the $0 marker. Its
superseded wording, verbatim:

> "…so all four are external infrastructure — latched and voided for one rerun exactly as a 5xx,
> never settled at $0, because a mid-run credential or credit failure must stop the campaign
> rather than let it finish with an unmeasured `$0` ledger reported as complete."

**§7.2 superseded wording** (as frozen at `3f6b257854d`), verbatim:

> "A revoked key, an exhausted balance, a lost permission and a provider rate limit are the
> campaign's environment failing, so all four latch **and** void for one rerun exactly as a 5xx:
> a mid-run credential or credit failure must stop the campaign rather than let it finish with an
> unmeasured `$0` ledger reported as complete."

Corrected at S13 (see §13): an infrastructure status that states a cost or token counts is priced
at that amount and voided for one rerun **without** latching.

Reason for all of them, as recorded: task-dependent refusal rates would otherwise penalise one arm
unevenly; a mid-run credential or credit failure must stop the campaign rather than let it finish
with an unmeasured `$0` ledger published as complete; and (S13) a bill the meter can read is not
unknown spend, so it must not block later admissions.
