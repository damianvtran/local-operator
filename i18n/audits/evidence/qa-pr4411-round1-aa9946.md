### QA report — round 1

**Verdict: FAIL — Q1 (MAJOR, same root cause as reviewer R1).** The turn-end behavior, guards, and existing bounded memory path passed independent execution; the honest pass-cost acceptance criterion does not.

QA: qa-tester, actual served model **openai/gpt-6.1-sol**, temporary quota fallback from anthropic/claude-opus-5-5. No model/role settings changed.
Scope: `666963a3ed17..21ce2ba4663bcda18c4a947d5567ad3e18f83343`.
Worktree: `/Users/damian/local-operator/.worktrees/cos-compaction`; HEAD verified before/after, no product/test edits. Primary repo venv, `PYTHONPATH` resolves `local_operator.__file__` inside the briefed worktree. All executions serial, synthetic Session IDs and isolated HOME/config/cwd, `env -i` stripping CMUX/LOP/secrets; no live Aida access, compaction, restart or real provider calls.

#### Commands / environment

Evidence root `S=/Users/damian/.local-operator/sessions/aa9946e6d094/scratchpad`; `W=/Users/damian/local-operator/.worktrees/cos-compaction`; `P=/Users/damian/local-operator/.venv/bin/python`.

Each successful command ran from its own `$S/{prod,scenarios,memory,tests}` with:
```sh
env -i HOME="$CELL/home" LOCAL_OPERATOR_CONFIG_DIR="$CELL/home/.local-operator" \
 PATH="$PATH" TERM=xterm-256color PYTHONPATH="$W" \
 GIT_CONFIG_SYSTEM=/dev/null GIT_TERMINAL_PROMPT=0 \
 "$P" "$W/scripts/run_bounded.py" --contract command-status \
 --timeout <590|300|180> --memory-mb 512 -- <command>
```
Actual payloads:
- Prod: `$P -u $S/qa_rig.py $S/prod text 180` (590s bound).
- Scenarios: `$P -u $S/qa_scenarios.py` (300s bound).
- Memory: `$P -u $S/qa_memory.py` (180s bound).
- Targeted tests (590s bound; additionally `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`): `$P -m pytest -p pytest_asyncio.plugin -p xdist.plugin -o addopts='' -n 0 -q $W/tests/unit/compaction/test_economics.py $W/tests/unit/session/test_compaction_economic.py`.

`qa_rig.py` is an independent copy of coder's disclosed scripted-provider rig, amended to include wake_prompt and assert quiet acceptance, exactly one pass and next-request reduction. `qa_scenarios.py` is independently authored, reusing the repository's scripted Provider and scaled Session construction helpers; real planning, pass, transcript replay and commit remain enabled. `qa_memory.py` independently exercises the real memory tool executor and Session restart.

#### Matrix — actual results

| Surface | Command / evidence | ACTUAL output / side effect | Result |
|---|---|---|---|
| Production-default economic gate below size threshold | prod / `prod.log` | 1,300 ordinary operator seed turns, compaction disabled only while seeding; restored production defaults. Seed 387.0s; measured high-wake=true; catalogue prices `(2e-7,8e-6,4e-6)` dollars/token. Wake 0 economic marker at provider context **253,707**, below 400,000 size threshold. | PASS |
| Per-request context reduction | prod | Next request **125,194** vs 253,707 before, ratio `125194/253707` (about 0.4935). Receipt projection 140,868, local after-estimate 103,690: neither is mislabeled provider occupancy. | PASS |
| Long repeated wake/peer/bridge/quiet path | prod | 180 deliveries cycling wake_prompt, peer_message, team_bridge_message, accepted quiet/no_reply. `pass wakes: [0]`, `summaries: 1`, `turn requests: 1480`; no extra continuation, no second pass. Final synthetic request context 195,932. | PASS |
| Economic pass specifically at each normal turn kind | scenarios / JSON rows | Independently fresh scaled Session for each wake, peer and bridge: before 21,038 -> next request 3,089; quiet accepted=true, before 21,038 -> next request 3,106. Each marker trigger=economic, one summary, exactly one ordinary provider request at that turn end. | PASS |
| Existing cap safety | prod + targeted tests | Production marker preserved 1,185 operator turns, `preserved_turns_cap=100000`; targeted high/low-wake assertions both retain 100,000 cap. No unsafe 24k tightening. Does not claim all seeded transcript content is carried forever. | PASS |
| Open ask carry | scenarios | A real queued synthetic approval ask was appended through asks store; all four economic markers carried `Keep synthetic deck approval open?`; assertions passed. | PASS |
| Decisions carry / restart baseline / required regrowth | targeted tests | Real Session tests for carried decision `#ops`, open ask, restart every 15 wakes, and >=6k growth between passes all passed. | PASS |
| Exact residual negative control | scenarios | Patched exact residual to 1,000,000,000: durable decline `why=recheck:residual_ratio`, no marker/summary, context kept, no immediate retry after small regrowth. | PASS |
| Provider failure negative control | scenarios | Scripted summarizer raised `RuntimeError: summarizer provider is down`; durable `why=pass_failed`; markers=0, context kept, no immediate retry. Expected error logged, assertions/command rc=0. | PASS |
| Unknown pricing refusal, real resolver | scenarios | Real unknown `opus-like` route, not patched prices: `prices=None markers=0 summaries=0` above economic growth floor. | PASS |
| Ordinary size trigger unaffected by unpriced economics | scenarios | Same unknown route with threshold_tokens=10000: ordinary compaction marker (no economic trigger), summaries=1. | PASS |
| Low-yield, min-gain, cold-cost refusal and anti-sawtooth | targeted tests | Gate reason table, low-yield doubling/cap/reset, cold shortcut refusing cost > saving, five synthetic sawtooth seeds, and exact/restart Session guards passed. Low-yield numerical/backoff coverage here is predicate/test coverage, not a live provider billing experiment. | PASS |
| Causal baseline control | scenarios | Executed actual base `_maybe_compact` method from `git show 666963a3ed17:.../session.py` in compatible current Session globals: same accumulated small history had markers=0, summaries=0. Restored head method, next wake produced marker=economic. This is a method-isolated base control, not a whole-base build. | PASS |
| Session memory store -> compact -> restart -> recall | memory / `memory.log` | Four real `execute_fact_memory(store, scope=session)` calls returned active, for instruction, decision, routing and in-flight delegation. Actual `compact_now` ran, then dispose/reconstruct same transcript. Recall returned all four exact bodies, active/origin=stated/source.author=main. | PASS |
| Bounded recall / no bulk prompt injection | memory | Recall **650 tokens <=1200 ceiling**; advisory tail **0 tokens**; next actual scripted turn request contained none of the four stored fact bodies wholesale. | PASS |
| Honest context-full pass-cost accounting | scenarios plus `session.py:29943-29947` | Gate uses `(C-R)*p_input`; observed wake Session C=21,038 and projected R=4,001 imply 17,037 modeled input tokens, but captured actual summarizer request has **21,422** ruler message tokens, before system/output charges. Thus input-only pass cost is understated on this real path. Q1 below. | **FAIL Q1** |
| Scoped automated regressions | targeted tests / `targeted-tests.log` | **44 passed, 2 warnings in 46.30s**, serial, rc=0. | PASS |
| Isolation / cleanup / source identity | logs + `cleanup.log` | Correct worktree source; HEAD unchanged; no scratch-script Python processes remain; production leader PID77748 `kill -0` returned ESRCH. No browser/server/native surface launched. | PASS (cleanup qualification below) |

**All usage/token occupancy supplied by the scripted provider is synthetic. Prices are catalogue/modeled prices, not actual billed spend. No measured fleet savings or Anthropic billing claim.**

#### Q1 — MAJOR: context-full pass cost prices removed context, not the summarization request

Code path: `Session._maybe_economic_compact`, `judge`, lines 29943–29947. `pass_cost=0` for snapcompact, otherwise `max(0,C-R)*p_input`. The summarizer consumes the span being folded (and prompt/system overhead), not only tokens finally saved; preserved turns and the resulting summary reduce gain without reducing summarizer input correspondingly. Summary output is also omitted.

Independent repro: run `qa_scenarios.py` at this head. Its captured `wake_prompt` JSON reports `context_tokens=21038`, `projected_residual=4001`, `summary_request_ruler=[21422]`. Pinned input price is `1e-6`, so modeled pass input cost is `(21038-4001)*1e-6 = $0.017037`; captured message-input-only cost is `21422*1e-6 = $0.021422`, excluding system/output. These are synthetic costs, not billing. This independently corroborates reviewer R1's root cause. The manager additionally relayed reviewer's decision-flip counterexample (C=160000,R=92000, summarized span150000, input4/M, write8/M, cold-next: modeled .272 vs input-only .600 vs rewrite saving .544); that counterexample is **reviewer-sourced, not rerun by QA**.

Requested remediation: price the real planned context-full summary request conservatively, including output allowance and its actual route; preserve cheap-before-exact behavior, but recompute honest pass cost with the exact plan before committing. Add a discriminating negative control where the former underestimated pass would be due but correctly priced pass is refused. This belongs in the manager's unified remediation with R1, not a separate wave.

#### Gaps / qualifications / non-gating diagnostics

- Automatic pre-eviction memory checkpoint remains explicitly OPEN, not tested/claimed implemented. Existing explicit-store retention path passes; arbitrary eviction, quota exhaustion, automatic promotion of all old instructions and indefinite preservation are not proved.
- Main production-default run uses text/context-full strategy. Snapcompact exact residual/pricing tier issues reported by reviewer are not independently re-executed here; QA does not claim them covered by this production run. Whole suites, actual billing, live Aida, restart cache-miss and clearance costs are outside scope. UI screenshots not applicable: no UI changes exercised.
- No remote auth/tenant surface exists in this local synthetic Session slice; unauthorized/wrong-tenant/expired-auth requests are N/A, not silently passed.
- Scoped tests emitted one `lo-analytics-writer` thread-leak diagnostic after 250ms on the quiet test (same diagnostic already disclosed in PR body). Non-gating/undetermined causality; no process survived the run. Two Pydantic deprecation warnings also occurred.
- Initial setup attempts: invalid bounded-runner argument corrected before spawn; default cleanup-required runner then refused this Darwin backend before spawn. Successful runs used its prescribed `--contract command-status`, which enforces sampled 512MiB policy but explicitly **does not certify descendant cleanup**. Exact scratch-process census and production leader ESRCH check are recorded, not represented as stronger certified cleanup. First pytest collection attempt with plugin autoload disabled failed on the xdist hook before tests; explicitly loading xdist and running `-n 0` repaired it.
- CI read once during collection: all completed checks successful; type-check IN_PROGRESS and packaging-smoke(3.14) QUEUED then. Not claiming CI green or merge-ready.

Absolute evidence files:
- `/Users/damian/.local-operator/sessions/aa9946e6d094/scratchpad/qa-plan.md`
- `/Users/damian/.local-operator/sessions/aa9946e6d094/scratchpad/qa_rig.py` and `prod.log`
- `/Users/damian/.local-operator/sessions/aa9946e6d094/scratchpad/qa_scenarios.py` and `scenarios.log`
- `/Users/damian/.local-operator/sessions/aa9946e6d094/scratchpad/qa_memory.py` and `memory.log`
- `/Users/damian/.local-operator/sessions/aa9946e6d094/scratchpad/targeted-tests.log`
- `/Users/damian/.local-operator/sessions/aa9946e6d094/scratchpad/cleanup.log`
