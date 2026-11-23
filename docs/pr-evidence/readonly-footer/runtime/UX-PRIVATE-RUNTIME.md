# Private UX runtime proof — read-only footer

- Round: private-runtime1; scope: ONLY `tui.editor_readonly.hint` and existing read-only/back interaction.
- Head: `3fca695e29bafbf0ea8539dd7a147f2f87183c6d`; worktree `/Users/damian/local-operator/.worktrees/i18n-readonly-footer` remained clean before/after.
- Reviewer: ux-reviewer session `a7646aff80aa`. Guide/skill initial read on anthropic/claude-opus-5-5; execution/analysis on actual temporary fallback **openai/gpt-6.1-sol**, following provider quota failure. No model/tool/policy changes.
- Allocation: bounded private hotfix/runtime lane, <=20 minutes.
- Verdict: **TERMINAL for this bounded Textual compositor interaction proof; runtimeProofReadyForPR=true with explicit native-renderer qualification.** No U-BLOCKER/MAJOR found in this tested surface. Not native-terminal approval, whole-app localization approval, or a substitute for formal independent code/QA/design rounds when a PR opens.

## Instrument and isolation

Primary interpreter `/Users/damian/local-operator/.venv/bin/python`, Python 3.12.13, loaded `local_operator` from the exact manager worktree. Real `OperatorApp.run_test(size=(40,24))`, production CSS, existing `FakeSession` and `_factory` fixtures. Opened production SettingsView, whose real mode entry owns the read-only composer, rather than assigning translated text. App `_apply_language` drove the mounted repaint; supplementary `_cmd_language` drove the public command handler's actual config-write/resolution/repaint path, not a mocked locale resolver.

HOME/config/CWD were explicit own scratch `readonly-runtime/home`; environment replaced by a narrow whitelist, all inherited CMUX/LOP/provider keys removed, intentional `LOP_LANG=en` bootstrap only. Desktop notification/launch/title switches disabled. Network socket connects and subprocess creation during app execution raised assertions; no server/browser/native app/provider/runtime was launched. Font provenance was pre-cached through the unchanged capture helper before HOME isolation; rasterization ran afterwards under original HOME. No dependency installs, copies, new worktree, source edits, commits, forge comments, release or actual-runtime config changes.

All app contexts exited. Foreground commands exited; final process query for the two exact rig script names returned no matching process. No process kill was needed. All generated private fixtures remain under this session's scratchpad.

## Actual command matrix

Here `$S` is `/Users/damian/.local-operator/sessions/a7646aff80aa/scratchpad`; `$W` is the worktree above; `$PY` is the primary interpreter above.

| Check | Command / actual result | Status |
|---|---|---|
| Copy scan | `$PY $S/copy_scan.py local_operator/i18n/catalogues/{en,fr,es,zh-CN,ru,vi,hi,ur}/tui.editor_readonly.json --check` from W: `scanned 8 files, 8 candidate strings; 0 findings`; exit 0 | PASS |
| Mounted compositor all8 | `env -u NO_COLOR TERM=xterm-256color PYTHONPATH=$W $PY $S/readonly_runtime.py`, log `readonly-runtime-attempt3.log`: `PASS mounted all8 runtime; app context exited`; exit 0 | PASS |
| Return | Same log: `PASS Escape: closed settings, restored editable composer and focus` | PASS |
| Custom mode negative | Same log: `PASS custom-placeholder negative control`; existing credential-prompt sentinel survived a locale change after leaving read-only mode | PASS |
| Persisted locale and per-locale Esc | `env -u NO_COLOR TERM=xterm-256color PYTHONPATH=$W $PY $S/readonly_supplement.py`, log `readonly-supplement.log`: `PASS all8 persisted language + actual Escape; context exited`; exit 0. Each actual config `values.language` matched requested locale; all eight actual Esc presses closed SettingsView and restored editable composer focus | PASS |
| Rasterization | Existing `rsvg-convert` on 16 calibrated first/settled SVGs, exit 0 | PASS |
| First/settled pixels | PNG byte comparisons: `en`, `zh-CN`, `fr`, `es`, `ru`, `vi`, `hi`, `ur`: all `PNG first==settled True size (320, 408)` | PASS |
| Placeholder contract | Each catalogue contained exactly one `{key}`; actual supplied literal `esc` occurred exactly once in mounted text and painted strip. No hand-edited generated/catalogue data | PASS |
| Identity/hygiene | `git rev-parse HEAD` stayed 3fca695…; `git status --porcelain` empty before/after | PASS |
| Native Urdu/Hindi glyph shaping | Textual compositor + SVG/librsvg is NOT a native terminal renderer. Urdu screenshot visibly uses disconnected/cell-projected glyphs. Hindi appearance/cluster shaping cannot establish native fidelity either | UNVERIFIED, not waived |

### Preserved setup failures, not product findings

1. `readonly-runtime.log`, exit 1: the process-denial control caught macOS locale auto-detection attempting `defaults` during startup. No CLI was spawned. Bounded repair: explicit supported `LOP_LANG=en` bootstrap, not removal of the guard.
2. `readonly-runtime-attempt2.log`, exit 1: fixture expected short JSON key `hint`; actual catalogue uses qualified `tui.editor_readonly.hint`. Bounded repair: read the actual qualified key. The app had mounted; no translation assertion was weakened.
3. Final attempt above passed with all real assertions retained. A separate fingerprint shell command initially looked for absent root `uv.lock`; corrected by inspecting tracked lockfile paths. This repo has no root Python lockfile. Installed dependency metadata hashes, not a nonexistent lock, are recorded.

## Measured locale matrix at a requested 40x24 terminal

Every row: actual screen/content [38,22], screen virtual [38,22], screen scrollbars [false,false]; editor size/content [32,1], editor virtual [0,1], editor region [4,19,34,1], editor scrollbars [false,false]; focus SettingsView. First and settled editor strips and geometry matched, and PNG bytes matched. `x` input never changed the read-only editor draft. There was no key wrapping or clipped hint at this tested size.

| Locale | Actual painted hint | Cells | Free editor cells |
|---|---|---:|---:|
| en | Read-only · esc back | 20 | 12 |
| zh-CN | 只读 · esc返回 | 14 | 18 |
| fr | Lecture seule · esc retour | 26 | 6 |
| es | Solo lectura · esc volver | 25 | 7 |
| ru | Только чтение · esc назад | 25 | 7 |
| vi | Chỉ đọc · esc quay lại | 22 | 10 |
| hi | केवल पढ़ने के लिए · esc लौटें | 21 | 11 |
| ur | صرف پڑھنے کے لیے · esc واپس | 27 | 5 |

Cell counts come from `rich.cells.cell_len` of actual text, backed by mounted widget geometry; they are terminal-cell accounting, not native font-advance/shaping proof. Target hint prose matches the approved catalogue in each case. The only common English-looking hint token is the intentionally technical literal `esc`.

## Viewed evidence and limits

Captured English baseline first, then zh-CN (prioritized) and French, followed by the remaining targets on the same mounted page/app. All eight individual `*-settled.png` frames were actually opened with the image read tool, as was `all8-contact.png` showing first/settled pairs. Equivalent first frames are pixel-byte-identical. Files `en-first.svg`, `zh-CN-first.svg`, `fr-first.svg` and all other locale first/settled SVGs use the calibrated helper; each has adjacent `.geometry.json` and PNG. Separate `*-rich.svg` files are raw `app.save_screenshot` exports, NOT geometry authority.

Visual result: translated footer stays one line, with its existing dim inert affordance and visible `esc` token. English/fr/zh/es/ru/vi compositor evidence is readable. Urdu glyph ordering/shaping in the SVG rasterizer is visibly not credible native RTL proof; literal `esc` itself is preserved, but correct native bidi projection/opposite-direction placement is NOT certified. Hindi native cluster shaping remains unverified. Do not close either native-programme gate from these images. The 40-column Settings page's own pre-existing clipped body rows/scrollbar are outside the changed footer scope and were not audited.

CLDR: N/A for this hint (no plural). No new target-language authorship/audit was performed. Pseudolocale/missing-namespace/LOP_I18N=0 optional cases were not rerun; this lane exercised genuine admitted locales, not synthetic success markers. No broader themes/viewports/whole-app language matrix is implied.

## Provenance files

- `runtime.json`: exact mounted strings, cells, geometry, input-refusal assertions and per-locale catalogue SHA-256 map.
- `config-escape-supplement.json`: actual notices and all-eight persisted-locale/Esc results.
- `fingerprints.json`: exact source/CSS/generated-key/catalogue hashes and installed dependency metadata hashes (Textual 8.2.8, Rich 15.0.0, regex 2026.9.29, Pillow 12.3.0).
- `../readonly_runtime.py`, `../readonly_supplement.py`: executable bounded rigs, with no product source modifications.
- Existing integration and resolver receipts were read as context only; current mounted/painted and config/Esc claims were independently derived in this lane.
