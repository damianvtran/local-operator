# Private non-native settings QA — bounded execution report

**Disposition: BLOCKED for complete P1 before/after visual proof; remaining observed rows below pass within their stated scope. No product FAIL derived. NOT formal QA clearance, publication permission, whole-settings/no-English clearance, or native Urdu proof.**

Worktree: `/Users/damian/local-operator/.worktrees/i18n-approval-choice-copy` @ `fad3ab5703fc5bb159f48cb43755c8d9e1323b6c`. Head checked before import, after capture, and at completion; unchanged. Final `git diff --name-only` empty. QA wrote only its own session scratch; no repo modifications, tests, installs, builds, browser/native/hardware lanes, git transport or PR comment.

Continuation model: current host header identifies **anthropic/claude-opus-5-5**. Earlier preflight receipt declared an openai/gpt-6.1-sol quota fallback; that prior declaration is historical provenance, not evidence of a model switch in this continuation. No model/config/tool changes made.

## Environment and actual launch

`R=/Users/damian/.local-operator/sessions/ff0e39d33743/scratchpad`. Drivers and results below are beneath R. Each real driver was launched with:

```
bash "$R/run-private-case.sh" <case> <driver.py> [stage]
```

Launcher creates a unique owned cwd/HOME/config/TMPDIR, uses `env -i`, `TERM=xterm-256color`, `PYTHONPATH=<candidate worktree>`, `PYTHONDONTWRITEBYTECODE=1`, notification and desktop-launch disable switches, primary `/Users/damian/local-operator/.venv/bin/python`. No inherited CMUX_*, LOP_*, XPC_FLAGS, NO_COLOR or credentials. Capture/API drivers import candidate `scripts.probe_isolation` first: its nested HOME/config were printed and asserted under owned scratch via owned TMPDIR. Synthetic FakeSession and in-process ASGI fixture only; no live server, OS clipboard or user session mutation.

Actual import prerequisite succeeded before any App/capture dispatch: local_operator, settings_io, OperatorApp, FakeSession and FastAPI settings router all printed module paths inside the pinned candidate. CSS_PATH `local_operator.tcss`. Versions: Textual 8.2.8, FastAPI 0.143.0, httpx 0.28.1, pydantic 2.14.1, Rich 15.0.0. The previously missing three distribution metadata names did **not** block these imports; no install or workaround was needed. `shipped.json` printed eight locales; independently computed English wire.settings SHA-256 `d9498d844e7924e2eba039d627b67a02725c5cc23c6a6020c823ce4db6a38bf1`.

Root integration evidence, read rather than rerun: `approval-native-admission-generate-check.json` says exit 0, `200 artifact(s) byte-identical`; `approval-native-admission-runtime-catalogue-cache.json` records seven 895-key target hashes/cache bindings. These are Root's results, not my generator execution. Root's strict-origin-main remote_compute gap remains open; fork-scope success is not strict/merge clearance.

## Written matrix — actual results

| Row / surface | Actual command | Actual output / side effect | Result |
|---|---|---|---|
| P0 isolated imports/source | allowlisted env runs `candidate-import-prereq.py`; log `candidate-import-prereq.log` | all five candidate module imports succeeded; eight shipped locales; English source hash above; exit 0 | PASS — import/source prerequisite only |
| P1 old vs repaired native labels | `run-private-case.sh before-complete private-settings-capture.py before`; current counterpart `after-complete … after` | Old wire.settings catalogue accepted 895 keys. Actual old shell choices `inherit/allowlist` become fr `hériter/liste d’autorisation`, zh-CN `继承/允许列表`, hi `मौजूदा परिवेश/अनुमति-सूची`, ur `موجودہ ماحول اپنائیں/اجازت کی فہرست`, es `heredar/lista de permitidos`, ru `наследовать/список разрешений`, vi `Kế thừa/Danh sách cho phép`. BUT unchanged namespace fixture symlinks were refused, producing unrelated message-code chrome in baseline frames. Those frames are NOT valid complete before/after visual comparisons. | **BLOCKED — fixture context**, not product defect |
| P2 seven-locale live consumers | `run-private-case.sh after-complete private-settings-capture.py after`; plus `api-complete private-settings-api.py` | One real CSS-loaded App switches fr→zh-CN→hi→ur→es→ru→vi. 21 selected setting/dropdown/help/Tab/pane flows plus six compact flows completed. 27/27 first/settled export pairs equal; all 27 config-byte assertions hold. API/registry mapping proves 362 changed live slots exactly equal current raw catalogue; three other changed slots resolve correctly but lack live registry consumers (listed below). | PASS for exercised live consumers; resolver-only gap explicit |
| P3 French role label | same capture driver, message-identity lookup | `wire.settings.team_bridge_session.label` derives actual key **team_bridge.session**. Source `Session de la cheffe de cabinet` measures **31 Rich cells**. At 100×30: list width60, detail94, pane visible; at80×24: list/detail74, pane hidden. Row label is visibly marked with ellipsis at both sizes; full help is visible (two lines at80). No overlap or reflow seen. Full unellipsized label accessibility not demonstrated. | PASS — bounded geometry/marked-ellipsis observation; accessibility gap |
| P4 Hindi OpenRouter description | same capture driver | `providers.openrouter.enforce_distillable_text`, raw enum member **string `"true"`**, current description `सिर्फ़ ऐसे endpoint जिनके टेक्स्ट से मॉडल डिस्टिलेशन की अनुमति हो` measures **46 cells**, complete composed row57 cells (fits measured list width60 at100×30 and74 at80×24). Expanded current frames viewed; no new clipping/overlap demonstrated. | PASS — compositor/geometry only |
| P5 Urdu technical controls | capture plus `run-private-case.sh controls private-controls-diagnostic.py` | Actual captured Urdu summary/detail/pane isolates balanced, including zero-isolate existing registry paints explicitly recorded. Four technical controls (`shell_environment.mode`, OpenRouter key, PATH/HOME/SHELL/TERM/USER/LOGNAME, `/login <provider>`) recover byte-identical raw text after LRI/PDI removal and balanced counts. Current Urdu frames viewed. Existing `_paint(...source_locale='en')` registry behaviour is not certified as complete bidi projection. | PASS — limited raw-token/balance controls; **not native shaping or all-code-island clearance** |
| P6 browse/cancel no-write | capture driver snapshots config, paints, expands, Escape, Tab, pane switches, resize | `config_unchanged=true` on all27 current records; baseline also unchanged. Configuration path is synthetic owned nested HOME. | PASS |
| P7 real ASGI GET presentation | `run-private-case.sh api-complete private-settings-api.py` | Eight GET `/v1/settings` =200;278 settings each. Raw typed value/default/choices.value projections identical to English. Every served label/help/choice label/description agrees with current registry. No config-byte change during GETs. | PASS — synthetic desktop capability; no auth claim |
| P8 actual PATCH persistence/refusals | same API driver | Five200 writes/readbacks: approvals auto(str), OpenRouter true(str), nerd-icons True(bool), False(bool), None(NoneType). French `demander` as raw approval value =422, `wire.errors.settings.invalid_value`, expected-one-of message; bool True for string OpenRouter enum =422. Both rejections preserve config bytes. No paint controls enter raw tested values. | PASS |
| P9 discriminating controls | same API driver | Old native-copy predicate rejects34/50/63/75/39/47/57 changed values in fr/zh-CN/hi/ur/es/ru/vi (365 total). Deliberate `LOP_I18N=0` selects en, GET200, native predicate rejects34 French slots. This is an intentional English negative control, not native success. | PASS |

No pytest/full-suite run. No authentication/authorization/wrong-tenant/expired-auth result: fixture overrides `require_desktop` and `get_config_manager`. No loading branch changed/exercised; synchronous populated settings, empty providers/teams pane state and populated synthetic packaged-agent pane were observed. Error proof is actual API422, not a fabricated unrelated visual error screen.

## P1 exact reproduction and minimum remaining fixture requirement

`private-controls-diagnostic.py` reproduced the surprising baseline chrome one step further with the production loader:
- `fr/wire.settings`: **accepted895 keys** (owned old JSON bytes).
- `fr/tui.small`: `CatalogueNotFound: 'fr/tui.small escapes the catalogue root'`.
- `fr/tui.settings_view.chrome`: same escape refusal.

Reason is production `catalogues._file` at lines313–324 resolving fixture symlinks outside the owned catalogue root. It intentionally refuses them. I did **not** weaken that check, override eligibility, replace product modules, or retry a different loading mechanism after diagnosing the refusal. Current-head native captures use the real root and are unaffected. The baseline's lexical wire labels are actual old runtime output, but baseline overall frames cannot support a clean visual diff.

Minimum remaining requirement is a separately accepted **self-contained data-only baseline fixture** with unchanged namespace bytes physically inside its root (no escaping symlinks), then repeat only P1's representative before frames with the exact same consumer/CSS and compare them against retained current frames. No package installation, new worktree, source/helper injection, native budget or whole-programme hold is needed. This report does not silently grant/retry that fixture repair under the explicit stop-on-refusal instruction.

## Other observations / explicit non-findings

- All365 changed catalogue slots were checked:362 map to live frozen registry fields; Hindi `wire.settings.hub_auto_update_agents.help` and Urdu that help plus `.choice.true.description` are resolver-only. A second source search found no corresponding registry consumer in settings_io; no UI execution claimed for those three historical slots.
- Current raw summaries/defaults still show `auto`, `inherit`, `true`, `warm`, etc.; packaged synthetic agents contain English descriptions; the transcript fixture says Read-only. These were visible and are not hidden. This candidate does not include sibling PR4428/local1974 presentation adapter repairs and does not establish no-English whole Settings.
- Early harness errors, all fixed only in OWN driver: bool True seed rejected for an actual string enum; `.plain` incorrectly requested on string pane output; guessed shell.environment.mode corrected to production `shell_environment.mode`; mapping originally asserted unused historical slots had live consumers. Their logs retained, no product FAIL manufactured.
- French role marked ellipsis is a static-width observation, not a new blocking defect or copy-polish request. Hindi row57<=60 provides measured support beyond the frame. No new concrete BLOCKER/MAJOR established.
- Current source uses `_LocalizedText` registry refresher; no PR4428 product code or base helper injected.

## Rendered evidence personally viewed

Rasterized with existing `/opt/homebrew/bin/rsvg-convert`; no browser/native engine launched. Viewed `comparison-1.png` through `comparison-8.png` (French/Chinese shell choices, Hindi100/80, Urdu120/80, French role100/80) and `other-three-locales.png` (Spanish/Russian/Vietnamese current shell flows). The eight comparison images **contain invalid baseline chrome on their left sides**, retained as evidence of the fixture defect, not endorsed before/after proof. Current right sides and the three-locale sheet were inspected for actual presentation. All frame SVGs plus geometry sidecars retained under R with `after-*` / `before-*` prefixes. No native Urdu glyph-shaping claim can follow from these Textual/librsvg images.

## Cleanup and receipt paths

Each launcher started an exact process-group leader, waited and TERM-reaped its OWN group on exit. Final independent `kill -0` checks show PID AND group gone for all eight driver attempts:49936,50203,50743,51400,51656,52733,52849,53827. No server/browser started. Import prerequisite was synchronous, no child service left running.

Primary receipts under R:
- `private-settings-qa-report.md` (this report), `private-qa-digest.json`.
- `candidate-import-prereq.py/.log`; `private-settings-capture.py`; `private-settings-api.py`; `private-controls-diagnostic.py`; `run-private-case.sh`.
- `after-capture-result.json`, `before-capture-result.json`, `private-api-result.json`, `private-controls-result.json`, `catalogue-changes.json`.
- Successful command logs: `after-complete.log`, `before-complete.log`, `api-complete.log`, `controls.log`; attempt/PID logs retained.

Formal pushed-head QA/PR comment remains intentionally absent: private pre-PR authorization only. Global native Urdu and strict-origin-main/publication gates remain open independently of this bounded result.
