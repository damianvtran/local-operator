# Read-only settings real-path rig preflight

Date: 2026-11-21. Owner: independent qa-tester ff0e39d33743. Actual served model: openai/gpt-6.1-sol after host-reported quota fallback from anthropic/claude-opus-5-5. Allocation: read-only preparation <=15 min (fix/patch sublane); no execution allocation consumed.

## Disposition

**PLAN ONLY / TESTED NONE.** No App, test, capture, server, browser, native lane, build, dependency installation, repository edit, PR comment or publication was dispatched. Metadata inspection is not a QA PASS. Current head observed twice: `e1c00f8c6bc4c06f2519c63d0ac406e31628dcf8` (K2), in manager-owned `/Users/damian/local-operator/.worktrees/i18n-approval-choice-copy`.

DO NOT execute this plan until Root supplies a successful ledger/generation integration receipt from owner `5c4b7ba1256f` and an updated pinned generated head. The brief's all-seven A/B semantic PASS is relayed, not independently retested here. Future formal QA/code/design/UX must share one PUSHED head after transport recovery; this document is neither a formal round nor pushed-head clearance.

Operator intent as briefed: all UI French / all seven non-English locales without ordinary English; this bounded lane covers real settings lexical copy. Global native Urdu shaping proof remains open, and this plan does not close the whole programme.

## Existing rig and exact source coupling

Prior QA provenance: `/Users/damian/.local-operator/sessions/527208343088/scratchpad/qa-report-r1.md`, lines 9–47. Existing reusable drivers in that same directory:
- `qa_capture.py`: real OperatorApp + application CSS, FakeSession from `tests/unit/tui/test_app_pilot.py`, dropdown expansion, checked member, Escape/config-byte check, first/settled/expanded captures and geometry.
- `qa_api.py`: real FastAPI settings router and TestClient; synthetic ConfigManager; overrides `get_config_manager` and `require_desktop`. GET typed wire projections, PATCH persistence and invalid human-label rejection. **No authentication coverage** is inferable from these overrides.
- `qa_warm_controls.py`, `negative_control.py`, and old SVG/PNG/geometry artifacts exist, but old results are not candidate evidence.

**Neither old driver is runnable unchanged against this candidate.** `qa_capture.py:4–5` and `qa_api.py:4` prepend the sibling `i18n-settings-leftovers` worktree. Capture lines 12–13 can exec an old complete settings_view module when QA_BASE=1; that mode is forbidden here. Capture line 23 only supports three fixed seed keys. The old helper/default-summary correction was PR4428 (`ccb3c388…` in prior report), disjoint from this candidate. Do not import its base-settings-view.py, helper functions, or old negative-control adapter. No hybrid product modules.

Reuse means a later scratch-only adaptation of these OWN QA driver files: replace only driver source root/seed table/state coverage, and use the candidate's unmodified production modules. Do not execute original hard-coded drivers. Prior frames show the reusable rig existed, not that the candidate works.

Verified via git blob IDs: `90009effa833` and K2 have identical `settings_io.py`, `tui/widgets/settings_view.py`, `tui/app.py`, `tui/local_operator.tcss`, `scripts/visual_capture.py`, `tests/unit/tui/test_app_pilot.py`. Thus before/after can use the SAME App/CSS/consumers while varying only catalogue source bytes. SettingsView blob: `144b307af8c4a65d47ea4d6c8d2b6ad62a1d989f`; settings_io: `3cc9bab6fc07d17527370c15193dd1e1c90161b6`; CSS: `ffe7dc9400edfa00558865161d1d01a7587ea8c0`. Recheck compatibility against the new generated pin before dispatch.

### Actual render chain, not assumed helper behaviour

At K2 the registry uses `settings_io._LocalizedText(str)` retaining `message` (`settings_io.py:177–217`), NOT a class named FrameMsg in the inspected registry/i18n source. `_t` resolves active locale through namespace caches and ICU; `_refresh_registry_copy` is registered at line 6805. Choice.label/description and Setting.help are frozen registry strings refreshed in place.
- TUI choices: `settings_view.py:4396–4421` paints `row.choice.label`, then description. Non-BOOL registry prose is explicitly projected with source_locale='en' in this baseline; do not claim this candidate repairs that existing provenance choice.
- TUI help/detail: `settings_view.py:5283–5284` paints `row.setting.help`; `_paint` at 255 delegates to localized_display.
- API: `server/routes/settings.py:211–227` projects setting.label/help and c.value/label/description. Same registry sources, raw typed values separate from human copy.
- Existing summary/default paths use `_render_value` (`settings_view.py:4554,5695`) rather than PR4428's helper. Raw summary/default enum words can therefore remain pre-existing: report them accurately without attributing the sibling fix to K2 or rewriting the whole-interface acceptance claim.

Native labels/help CAN be validated independently of PR4428 using these consumers. Retain exact output/provenance, including existing limitations; do not silently treat a raw summary as translated.

## Dependency/source preflight findings (metadata only)

Shared primary interpreter exists: `/Users/damian/local-operator/.venv/bin/python`, symlink to CPython 3.12. Installed dist metadata: Textual 8.2.8, FastAPI 0.143.0, httpx 0.28.1, pytest 9.1.1. `rsvg-convert` exists at `/opt/homebrew/bin/rsvg-convert`. No tool was installed or launched beyond stdlib/metadata inspection.

The requested matching lock/project metadata is **NOT established**: candidate has no `uv.lock`; primary has one; pyproject files differ (candidate version 0.112.0 vs primary 0.92.1; dependencies differ). A second check parsed candidate requirements with the primary interpreter, `importlib.metadata` + `packaging.Requirement`, without importing app: 28 applicable installed requirements checked, no version-spec conflicts among them; **missing pypdf, watchdog, backports.zstd**. Primary installed editable dist is local_operator-0.91.8 metadata. This is not a claim that all three packages lie on the settings capture import path, nor an observed App import failure.

Minimum prerequisite: after new head arrives, establish that the shared interpreter can import that exact candidate's settings_io, OperatorApp, FakeSession and FastAPI router under isolated env, and record module paths + relevant dependency versions. If import fails, stop at the exact missing module/API; Root supplies/authorizes the minimum shared-dependency repair. No install/new venv/worktree/source copies/pyright bootstrap in this lane. No whole-programme hold follows merely from optional/lazy missing metadata. Python source does not need a native build; no compiled native client identity is available or required for Textual compositor evidence. Native Urdu is a different unallocated lane.

## Frozen execution envelope (future only)

Use own new synthetic root under this session scratch, NEVER old HOME/config files and NEVER env fallback for fixture location. Each run starts from isolated cwd (not home or repo). Launch with allowlisted env, conceptually:

```sh
cd /Users/damian/.local-operator/sessions/ff0e39d33743/scratchpad/<owned-case-root>
env -i HOME=<owned-case-root>/home \
 LOCAL_OPERATOR_CONFIG_DIR=<owned-case-root>/home/config \
 PATH=/opt/homebrew/bin:/usr/bin:/bin TERM=xterm-256color \
 PYTHONPATH=/Users/damian/local-operator/.worktrees/i18n-approval-choice-copy \
 LOCAL_OPERATOR_NO_NOTIFICATIONS=1 LOCAL_OPERATOR_NO_DESKTOP_LAUNCH=1 \
 /Users/damian/local-operator/.venv/bin/python <owned-adapted-driver> <case>
```

This strips all inherited CMUX_*, LOP_*, NO_COLOR, XPC_FLAGS and live credentials; set only deliberate fixture controls later. No git/gh/keychain clients under redirected HOME. Override tempfile scratch root to owned scratch before capture helper isolation, or use candidate `scripts.probe_isolation` inspected before dispatch, so helpers do not create scratch outside owned root. Existing `visual_capture.isolate_capture` creates another temporary HOME/config (lines 47–49); record its actual final paths and assert they are under owned scratch. Do not merely assert the launch env stayed authoritative.

Before app imports, assert HOME/config/cwd are synthetic and notifications/desktop launch off. After imports, record `local_operator.__file__`, settings_io.__file__, settings_view.__file__, app.__file__, CSS_PATH resolved absolute and FakeSession.__module__ source path. All must resolve within pinned candidate except third-party shared packages. Print actual interpreter and active locale; assert locale equals requested, and no CMUX/LOP variables beyond deliberately chosen controls. Use synthetic session IDs only. No OS clipboard operation, desktop notification, live sessions, MCP, provider credentials or user data.

Before/after each run: `git rev-parse HEAD` equals NEW_PIN; record tracked diff/status without modifying concurrent owner artifacts. Config-byte snapshots are from explicit synthetic config paths only. No global config readback or hashing. No dev server needed for in-process ASGI or Textual. If a later authorized lane needs a server, start exact process-group leader, trap kill of that group, verify PID gone; harness browser only, close owned tab. No browser proposed for this bounded plan, hence none opened.

## Written matrix — all rows PLANNED / NOT RUN

Order: fr, zh-CN, hi, ur, then es, ru, vi. Use English as semantic/control baseline, not another native attestation.

| ID / surface | Future command / driver action | Expected positive and observed evidence to collect | Negative / refusal | Status |
|---|---|---|---|---|
| P0 source + eligibility | NEW_PIN check, isolated import preflight; inspect generated data/shipped.json and current wire.settings ledger/history; run generator check only when separately leased | Root integration success receipt; current English hash and seven native 895-key files bound to new generated head; active locale == request | No private eligibility override, no false passed ledger/hash, no activation by forcing locale | NOT RUN; awaits Root receipt + new pin |
| P1 original vs repaired native choices | adapted capture, same consumer/App/CSS; baseline catalogues from git show 90009effa833, repaired from NEW_PIN | exact raw old enum labels vs native labels, choices/check marker; first/settled/expanded frames and geometry | Restoring selected OLD label in OWN synthetic catalogue control must fail exact native-label assertion, not bless fallback | NOT RUN |
| P2 seven help/description flows | select settings via real settings interaction, expand dropdown, Tab and sidepanel/detail states; all seven locales | selected native help, native Choice.label/description agree with current message text, no stale previous locale; full selected text recorded with bounded clipping explicit | Compare current source expected; English/source code fallback fails native row (technical names remain allowed) | NOT RUN |
| P3 compact French | at actual measured 31-cell label budget, select `agents.auto_update_seeds` label `Mise à jour auto des rôles intégrés`; normal/narrow App sizes, Tab/pane state | true content budget reported; truncation/wrap/ellipsis plus full accessible detail assessed; no overlap, no invented geometry | Longer real label is control; no shortened injected product string | NOT RUN |
| P4 compact Hindi | actual 46-cell description budget for `providers.openrouter.enforce_distillable_text`, true choice | current native description `सिर्फ़ ऐसे endpoint जिनके टेक्स्ट से मॉडल डिस्टिलेशन की अनुमति हो`; first/settled and expanded/pane captures; full raw source vs display recorded | old description `सिर्फ़ distillable-text endpoint` must fail native-copy predicate; no substring-only false pass | NOT RUN |
| P5 Urdu/code islands | populated Urdu choice/help with real technical tokens, plus raw technical-ID controls | literal source tokens remain exact; balanced isolate/control evidence and frame clipping recorded; raw wire/config free of paint controls | code/enum/technical controls stay unchanged; no global stripping that masks leakage | NOT RUN; compositor only, NOT native shaping |
| P6 no-write | initial/config snapshot, paint, expand, navigate Tab/detail, Escape/reopen | synthetic config byte-for-byte unchanged throughout all read-only states | cancelled edit must not persist; reading config only inside owned synthetic root | NOT RUN |
| P7 real ASGI presentation | adapted qa_api.py, current router, synthetic ConfigManager; GET /v1/settings in seven locales + en | status/body native labels/help/choices, exact typed value/default/choices.value equality to en, unchanged config bytes | reject human labels as values; no auth/tenant/expired-auth claim due capability override | NOT RUN |
| P8 raw write semantic control | narrow PATCH tool_approval_mode='auto', readback; one applicable changed enum; tri-state only if relevant | 200; persisted raw str/bool/None as seeded; human prose never stored | localized human label PATCH 422 and exact refusal envelope; invalid key/value unchanged bytes | NOT RUN |
| P9 eligibility switch control | separate isolated subprocess with deliberate LOP_I18N=0 | English is expected only under explicit switch; native assertions must fail here, proving checker sensitive to fallback | Do not conflate this with production locale success | NOT RUN |

`31` and `46` above are briefed target budgets, not measured in this run. Driver must expose measured content widths and select real App sizes that produce them; if not reachable, report actual geometry and retain the target as unverified rather than clamp widgets or fake layout. P2 covers all changed help families, prioritizing approval/role/voice/OpenRouter/shell and exact repairs from the ledger. Derive finite key list from 90009..NEW_PIN catalogue diff before execution; capture representatives of each distinct rendering/state budget, record full source-output mapping for every changed key. No huge full backend matrix.

### Before fixture and counterfactual

Later only: git show 90009effa833:<catalogue path> to OWN scratch baseline catalogues; do not move worktree HEAD or load sibling product helpers. Provide owned catalogue fixture root containing required baseline namespace files and unchanged current namespaces; use established catalogue-root seam in QA process only, without eligibility bypass. Hold generated eligibility constant from accepted NEW_PIN, prove the only varied bytes are target wire.settings catalogues; label this a controlled same-consumer lexical reproduction, not a historically shipped full build. Both baseline and repaired share same accepted product consumer/CSS. If runtime source-current verification rejects baseline fixture, stop and report exact refusal; do not alter ledger/eligibility to force rendering. No archive/copy/capture created during preflight.

Counterfactual must independently show native-copy oracle fails with old native enum text and with explicit English kill switch. Do not run PR4428's `_render_value` injection: it targets another defect. Existing raw summary/default limitation is neither hidden nor repaired by a hybrid fixture.

## Remaining limits and release boundary

- All behaviour, imports of application modules, frames and API requests remain untested this turn by explicit preparation-only instruction.
- Generated `shipped.json` currently lists all eight locales, but this alone is not source-current eligibility proof. New integrated ledger/generated receipt is mandatory before execution.
- No native Urdu glyph shaping proof; Textual SVG/librsvg cannot establish physical terminal bidi/shaping. No Mac/browser/native budget requested or used.
- No ordinary-English-free whole UI claim; this lane does not cover unrelated namespaces or unmerged sibling fixes.
- Authentication, authorization, tenant and expired credentials N/A to overridden synthetic ASGI fixture; do not report them PASS.
- No PR exists/was supplied for this preflight; no formal QA round/comment or merge/publication approval.

Authorities read: guide://delivery; required browser-qa/design-qa/minerva-software-development/verification skills; AGENTS visual/isolation sections 2738–2947 and 977–1047; docs/I18N.md 1–90. Repo remains untouched by QA. Final observed head remains K2, not a generated/pushed clearance head.
