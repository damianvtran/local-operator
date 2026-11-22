# Four-key model persistence notice — integration handoff

## Disposition
Implemented privately; NOT pushed, no PR/review approval. Normal base-aware checker is **FAILED**, not waived. Broader styled QA remains unperformed pending Root's authorization at a pushed head.
Worktree `/Users/damian/local-operator/.worktrees/i18n-model-persist-notice`, entry HEAD `912e165fd76e217b5cd5d212a59e7e1e906fa6fd`. Actual served fallback reported by host: openai/gpt-6.1-sol after anthropic/claude-opus-5-5 quota failure. No model/config pins changed.

## Admission routing correction and evidence
The earlier `admission-preflight/receipt.md` is historical and superseded in its diagnosis. French B `receipt.json` is an artifact manifest, not the canonical row; Root supplied the correct existing `current-ledger.json`. Urdu A `receipt.json` is a schema1 one-entry wrapper. Their initial failed CLI attempts remain `fr-b-dry-run.log` / `ur-a-dry-run.log`; these are rig input-routing failures, NOT linguistic failures.

All 14 canonical records were checked for current input and declared artifact SHA equality, key_count=4, passed verdict and auditor-session != translator-session. Canonical routing manifest `admission-preflight/canonical-routing.json`. All14 real CLI dry-runs passed before ANY apply (`canonical-dry-runs.json`). Then serial append used fresh history-sha, dry-run and apply for each: Hindi repaired A2/B2 first, then fr, zh-CN, es, ru, vi, ur A/B. Actual append receipts `applied.json`.

Urdu wrapper preserved byte-for-byte. The current append CLI does not unwrap schema1: main `_read` passes payload directly to `append`/`_rows`. Its SOLE existing entries[0] was mechanically JSON-serialized into own scratch and submitted unchanged to normal enforcement; no fields/provenance/values invented. Root explicitly accepted this conditional on equality/hash evidence after the late wrapper note. `ur-adapter-manifest.json` proves source/output raw SHA and structural equality. Parent's earlier assertion of direct wrapper CLI support was corrected against code. No source record/receipt edits.

Current supported exported view: all7 latest rows passed, 2 history rows each except Hindi 4. Hindi original 2-row prefix (including failed old B) structurally preserved. `current-ledger.json` exported through ledger.py, `current-status.json` lists current auditor sessions. No self-native-audit or eligibility/global completion claim.

## Product change
Only app.py imports and `_persist_hint_notice` changed. Generated typed `tui_model_persistence` factory resolves settings/persist/saved clauses and model carrier through existing Msg.envelope path using ONE current active locale snapshot on each opening. Four state branches, no-label routes, separators and English exact bytes retained. Opaque label unchanged. Existing plain-string `_system_notice`/NoticeBlock and pinned-tail text dedup unchanged. No historical transcript rewrite or translated-prefix parsing.

Legacy PERSIST_HINT, model picker/footer English parser, providers, raw copy/context, immutable ledger snapshots and shipping eligibility untouched. Visible English footer is out of scope, NOT an allowed human-English exception or whole-picker completion; follow-on needs machine-prefix migration, not translation of its global constant.

## Execution
All tests/captures/checks used primary repo .venv, PYTHONPATH leased worktree, isolated scratch HOME/config, env -i, TERM=xterm-256color. Git normal HOME. No install/new worktree/fetch/rebase/native/browser/server/full suite. Module path printed by guarded capture proves worktree import.
- Targeted pytest: new tests/unit/tui/test_model_persistence_notice.py plus exact five named existing nodes from source receipt (one parametrized, six cases): **39 passed, 2 warnings, 10.31s**, rc0. `tests/targeted.log`.
- Forced-English negative control monkeypatched production active_locale to en and called real live-locale regression: **1 failed in 1.36s**, rc1 as EXPECTED, at `assert french != english`; `tests/force-english.log`. Product not edited for control.
- Supported generate.py rc0; generate.py --check **201 artifacts byte-identical**, rc0 (`checks/generate-check.log`). Only generated __init__ and new four-function module differ.
- `check.py --update local_operator/tui/app.py` rc0, no baseline diff; no ceiling raised.
- Normal `check.py` ONCE: **rc1**, 0 enforced ratchet findings, 1 catalogue problem, 114 advisory outside; `audit history: audit suffix file deletion is forbidden` (`checks/normal-check.log`). Read-only follow-up: origin/main contains seven tui.editor_readonly suffixes absent from frozen old branch, listed in `checker-base-missing.json`. No fetch/rebase/base substitution/narrow-green rerun; needs manager convergence decision.
- Existing ruff/black neither executable/module available; checked availability only, no implicit wrapper/install.

## Rendered evidence and limitations
`before/` captures actual unchanged method from frozen entry HEAD BEFORE consumer edit (no function reconstruction). `after/` uses changed real styled OperatorApp. Same existing hermetic_tui_env.__wrapped__ executed before startup, counted update no-op exactly1; socket audit + httpx send guards each negative-controlled before app import. Unexpected attempt list empty through context teardown; no successful fake HTTP. First rig attempted await on synchronous slash method and failed; corrected script only, retained before/capture.log. Successful before/capture-fixed-rig.log and after/capture.log rc0.

Each directory contains first/settled SVG, PNG and geometry sidecars. **first frames are pre-notice/splash**, not valid same-state animation controls. Guard script's inherited startup summary still says screenshots=0; that field is stale rig metadata, not actual count: each successful run wrote two SVG captures, externally rasterized to two PNGs. No log edited to hide this. Root viewed all four PNGs and cleared ONLY settled content LOOK; no stable-pixels/no-reflow claim.
Settled French before notice English, after French; exact model ID unchanged. Actual NoticeBlock width75, height3→4, virtual size matches, scrollbar false at 100x30. I viewed settled pair; Root independently confirmed no visible clipping. Wider locales/states/widths, live styled reopen and native RTL/shaping remain unqualified; existing English dedup/width nodes passed, unit branch/locale assertions are not a rendered matrix.

All run_bounded completions report cleanup=UNKNOWN; preserve UNKNOWN. No server/browser/native app started. Scoped final ps census found no capture.py/negative-control/run_bounded model-notice processes (grep rc1 means none), not a kernel-wide proof. Latest df showed 13 GiB available (>12 GiB floor); no further capture dispatch.
