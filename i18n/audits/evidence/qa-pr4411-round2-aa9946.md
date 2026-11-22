### QA report — round 2

**Verdict: PASS. Q1 is fixed on this head. No new FAIL/BLOCKED finding.**

QA: qa-tester, actual served model **openai/gpt-6.1-sol** (same disclosed quota fallback as round 1; no model/role changes).
Scope: `21ce2ba4663bcda18c4a947d5567ad3e18f83343..97cf0a4aef6818cce20b3faab8fc49f81d4740ab`.
Worktree: `/Users/damian/local-operator/.worktrees/cos-compaction`; HEAD verified before/after. Read-only product/test checkout. Independent fresh isolated HOME/config/cwd cells, env-i removing CMUX/LOP/secrets, primary venv plus worktree PYTHONPATH. Serial, each command <=590s and sampled 512MiB policy. No live Aida/session/secret/provider API access, install or new worktree.

#### Delta surface map and preflight
- `session/session.py`: real summary cost/route/output allowance, tier-aware rates, exact replay residual and prepared-summary reuse, optional-vs-mandatory fallback -> real-path + direct controls + scoped Session tests.
- `compaction/economics.py`: remove downward calibration cap and clarify cheap marker hint -> residual/regression neighbors; numerical EOQ formula unchanged.
- `memory_capture.py`, memory guide: bounded procedural hint, deliberate candidate review/source/subject, revision checked correction/supersession -> direct persisted-store flow plus four focused memory tests. This tests contracts, not that a live model will follow prose.
- Changed tests exercised; committed round1 QA receipt is non-runtime evidence.

Preflight: `/Users/damian/.local-operator/sessions/aa9946e6d094/scratchpad/qa-plan-round2.md`.

#### Exact command pattern
`S=/Users/damian/.local-operator/sessions/aa9946e6d094/scratchpad`, `W=/Users/damian/local-operator/.worktrees/cos-compaction`, `P=/Users/damian/local-operator/.venv/bin/python`.
From each fresh `$S/{r2prod,r2controls,r2tests}` cell:
```sh
env -i HOME="$CELL/home" LOCAL_OPERATOR_CONFIG_DIR="$CELL/home/config" \
 PATH="$PATH" TERM=xterm-256color PYTHONPATH="$W" PYTHONDONTWRITEBYTECODE=1 \
 GIT_CONFIG_SYSTEM=/dev/null GIT_TERMINAL_PROMPT=0 \
 "$P" "$W/scripts/run_bounded.py" --contract command-status \
 --timeout 590 --memory-mb 512 -- <payload>
```
Payloads:
1. `$P -u $S/r2_real_path.py` — independent copy of disclosed coder rig; strengthened next-request reduction and captured-input/output quote assertions, real resolver with explicitly synthetic tariff.
2. `$P -u $S/r2_controls.py` — independent direct unknown route, same-Session tier repricing, cost refusal, actual archive prepare/commit, persisted candidate correction/replacement flow.
3. With `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`: `$P -m pytest -p pytest_asyncio.plugin -p xdist.plugin -o addopts='' -n 0 -q $W/tests/unit/session/test_compaction_economic.py` plus these four nodes in `$W/tests/unit/session/test_memory_capture.py`: `test_post_compaction_hint_is_bounded_review_not_automatic_confirmation`, `test_reviewed_candidate_correction_keeps_source_and_refuses_stale_revision`, `test_suggestion_guard_suppresses_generic_one_word_and_filters_scope`, `test_suggestion_slot_is_one_per_fact_and_budget_is_the_rendered_line`.

#### Fresh remediation / direct-neighbor matrix
| Surface | Command / ACTUAL response and side effect | Result |
|---|---|---|
| Q1 conservative quote matches actual summary request | `r2_real_path.py`: captured summary input **310,727 ruler tokens**, configured max output **16,384**, quote **$1.684707**. Asserted quote >= input*4/M + output*8/M; real full-write allowance makes quote larger. Synthetic tariff/provider accounting, NOT actual billing. | PASS |
| Q1 discriminating refusal through production economic gate | `r2_controls.py`: real planned 170k-token synthetic operator span, C=170000/R=100000, cold-next. Former net-gain quote **$0.28**, avoided rewrite **$0.56**, actual planned conservative quote **$0.982987**. Gate wrote `recheck:below_eoq`, summaries=0 and no marker. This is independently derived, not copied from reviewer. Scoped regression also rejects original 160k/92k counterexample, including previous-summary/system/template/output. | PASS Q1 fixed |
| Economic gate, request reduction, no sawtooth after cost fix | Real Session with production summary/preservation settings: 110 seed requests, then **180 deliveries / 45 accepted quiet turns**, all wake/peer/bridge paths. Exactly one marker at wake0, C=**330,665** <400k; exact residual projection **38,332**; first synthetic provider request after pass **21,956**. summaries=1, provider turn requests=290 (no auto-continue), ask/decision carried, cap=100000. No repeated pass. | PASS |
| Exact snapcompact preparation/replay and reuse | Direct control used **real compact_to_archive**, not a renderer substitute: prepared once, actual economic commit ran, no LLM summary; conservative projected **2,067** equals converted committed replay ruler **2,067**, same prepared tuple retained, archive function call count remains1. This is backend archive execution, not a rendered UI/pixel review. | PASS |
| Large residual counterexample / no clipped downward calibration | Scoped R2 regression constructs >32k archive and verifies final residual >.60, not false fixed32k authorization, and prepared tuple identity at commit. Large fixture substitutes raster bytes; separate small real archive path above exercises actual preparation/commit. No claim of native large-frame billing validation. | PASS (contract qualification) |
| Actual-size same-Session pricing / tier crossing | Direct real resolver, synthetic tier catalogue. Anthropic rates $/M `(read,write1h,input)` 100k `(0.2,8,4)` ->300k `(0.4,16,8)` ->100k `(0.2,8,4)`; Google `(0.2,5,4)` -> `(0.4,10,8)` ->original. Inclusive/disjoint accounting and no stale size cache. Scoped real Session tariff-crossing regression declines `recheck:tariff_change` before summary. | PASS |
| Unknown route / unknown serving provenance | Direct Session constructed without workload make helper: real unrecognized resolver returns None at170k. Scoped unknown-service-tier/unknown-write-tariff tests refuse safely. | PASS |
| Optional archive failure versus mandatory fallback | Scoped regression forces archive failure: optional economic summary forbids LLM fallback (0 summaries), ordinary mandatory path retains successful fallback (1 summary). | PASS |
| Exact refusal, restart baseline, growth, cap, quiet scheduling and ask/decision neighbors | Scoped Session file executed including existing wake/quiet/exact-refusal/restart/carry/cap tests, now against new cost/residual machinery. | PASS |
| New bounded hint, no body injection or auto-confirmation | Direct persisted capture produced one candidate. Hint **109 tokens <=180**, no synthetic Nova candidate body, store fact before/after identical. Explicit source/subject/revision/authoritative-recall review instructions present; other candidates stay unconfirmed. | PASS |
| Reviewed correction / source / stale revision / explicit replacement | Direct production MemoryStore: inspect candidate, revision-checked correction to active preserves source/subject; stale write raises `Memory changed concurrently. Read its current revision and retry.` Explicit separate replacement names inspected predecessor; bounded1200 search returns replacement and excludes predecessor. | PASS |
| Scoped automated regressions | **23 passed, 2 warnings in 14.36s**, serial, rc0. No full suite or whole-directory run. | PASS |
| Isolation / source / process census | Correct worktree source in actual log, HEAD remains97cf0a4, clean before receipt creation. All3 commands rc0; no remaining r2 script Python/scoped pytest processes. No browser/dev server launched. | PASS (runner qualification below) |

#### Retained round1 rows — not re-run wholesale
Retained SHA: `21ce2ba4663bcda18c4a947d5567ad3e18f83343`. Evidence: `r2-compatibility.log` compares Git blobs and exact AST source segments; this is source/byte compatibility, not an invented new execution.
- Existing explicit memory **store -> compact -> dispose/restart -> recall** four directive/decision/routing/delegation facts; **650/1200** recall and no wholesale injection: **retained (21ce2ba)**. `memory.py`, `tools/memory_tools.py`, transcript, marker/API/cutpoint and `_finish_compaction` are byte-identical. New procedural hint separately checked above.
- Original method-isolated base causal control: **retained (21ce2ba)**; no need to repeat baseline dispatch demonstration for cost-only remediation.
- Provider failure durable journal/no immediate retry, numerical low-yield doubling/cap/reset, min-gain/cold-cost/sawtooth predicate table: **retained (21ce2ba)** with unchanged formula and byte-identical `_economic_usage_scan`, `_economic_declined_at`, `_idle_growth_baseline`; direct economic-neighbor tests and new full wake sequence additionally exercise changed callers.
- Ordinary size-trigger unknown-route real-path row: **retained (21ce2ba)**, supplemented by fresh mandatory-fallback positive control above. `_preserved_turns_cap`, API/cutpoint unchanged; cap/carry additionally exercised freshly.
- Original 1300-operator seed counts and four individually fresh per-kind measured outputs remain historical **retained (21ce2ba)**, not reasserted as current-head occupancy. New caller/cost behavior is covered by the fresh110+180 run and scoped quiet/wake tests.
- Round1 scoped44-test result remains historical, not substituted for round2's23 tests. Original Q1 FAIL is superseded only by fresh Q1 rows above.

#### Qualifications / gaps
- **All token occupancy and dollar figures are synthetic/provider-script/catalogue-model receipts, not actual billing or fleet savings.** No live Aida compaction/restart, no unsafe cap reduction, no paid live-model behavior validation.
- Automatic memory checkpoint/ack-before-trim remains OPEN. No guarantee of arbitrary eviction/candidate-budget exhaustion, semantic promotion, model adherence to new prose or preservation under a tighter cap.
- Large archive residual test suppresses raster bytes; small real archive path does not certify large visual-frame/provider billing. No UI changed; no browser/TUI screenshots needed for this backend/procedural delta.
- Runner `command-status` on Darwin enforces sampled memory policy but explicitly does not certify descendant cleanup. Executable-scoped census found no remaining processes. Initial broad census matched its own bash command text; refined executable check confirmed it was not an orphan.
- Same known quiet `lo-analytics-writer` thread diagnostic after250ms and two Pydantic deprecations as round1; no process remains, no new gating finding.
- CI read once, head97cf0a4: completed checks successful; type-check IN_PROGRESS and multiple checks QUEUED at observation. **Not claiming merge-time CI green.** Auth/wrong-tenant/expired-auth remote surfaces N/A for isolated local Session/store.

Absolute evidence root: `/Users/damian/.local-operator/sessions/aa9946e6d094/scratchpad/`:
`qa-plan-round2.md`, `r2_real_path.py`, `r2prod.log`, `r2_controls.py`, `r2controls.log`, `r2tests.log`, `r2-compatibility.log`, `r2-ci.json`, `r2-cleanup.log`, `qa-report-round2.md`.
