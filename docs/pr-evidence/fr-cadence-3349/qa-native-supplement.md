### QA report — round 1 — supplemental native evidence

Reviewer: qa-tester / anthropic/claude-sonnet-5-5 (actual model identified by this resumed runtime; no model/tool/default changes or fallback invoked during this native pass)
Head: dcdb0b280553861b72b225695497810d201ef890
Control: 029c0ec7ade8be8bdb6625a5816b4e23fc0a2902
Verdict: PASS for actual-app editor interaction, validation/recovery and frame stability; FAIL for full-copy recovery on the tested settings surface. Designer must judge the measured fit/truncation below; no clean visual sign-off claimed.

This supplements, without resetting, the unchanged runtime/manifest/generation evidence in the initial QA report. Operator request: “Make sure to continue the work all the way for all languages to get the entire UI completely translated”. Scope stays exactly the two French cadence keys, 0 daily-local / 1–168 whole elapsed hours / 4 example / already-owed semantics. No UI refactor, behavior change, other namespaces or version bump.

#### Admission, setup and actual runs

The manager explicitly cleared prior native hold: UI job `16a2cc0b2f6e` stopped/reaped. Admission measured here before head: load `10.79 15.58 18.89`, memory free `78%`, disk available `19 GiB`; before base: load `10.56 14.98 18.53`, memory free `78%`, disk `19 GiB`. Both satisfy requested load <=20 / memory >=25% / disk >=15 GiB.

No installation. Primary existing `.venv/bin/python` plus explicitly prepended feature-tree/snapshot source. Source identity and production `OperatorApp.CSS_PATH=local_operator.tcss` passed preflight on both trees before dispatch. Base is an immutable `git archive 029c0ec7ade8 local_operator scripts tests pyproject.toml` source snapshot, not an English catalogue mock. `scripts.visual_gallery.py --list` was consulted first; actual app instrument is `OperatorApp`, `FakeSession/_factory`, `isolate_capture()` before any app import, and `scripts.visual_capture.save_capture()` with `.geometry.json`. No bare/CSS-less panel host. Synthetic fixture session; no operator state.

Exact run form, sequential head then base (the script and all evidence remain in the scratch path below):

```sh
P=/Users/damian/local-operator/.venv/bin/python
R=/Users/damian/local-operator/.worktrees/french-cadence
S="$LOCAL_OPERATOR_SCRATCHPAD"
# base snapshot already extracted from git archive; preflight passed on both.
cd "$S/native-evidence"
env -i HOME="$S/native-evidence" PATH="/opt/homebrew/bin:/usr/bin:/bin" TERM=xterm-256color \
 "$P" "$R/scripts/run_bounded.py" --timeout 600 --memory-mb 1024 -- \
 "$P" "$S/cadence_native.py" "$R" "$S/native-evidence/head" head
# [run_bounded] memory budget 1024 MiB (explicit); sampled owned child group only
# PREFLIGHT head source .../.worktrees/french-cadence/local_operator/__init__.py CSS local_operator.tcss HOME .../native-evidence/lop-visual-gv6yh9v8
# CAPTURE PASS head 52 frames
# [run_bounded] python rc=0 wall=25.6s

env -i HOME="$S/native-evidence" PATH="/opt/homebrew/bin:/usr/bin:/bin" TERM=xterm-256color \
 "$P" "$R/scripts/run_bounded.py" --timeout 600 --memory-mb 1024 -- \
 "$P" "$S/cadence_native.py" "$S/native-base-029c0ec7ade8" "$S/native-evidence/base" base
# [run_bounded] memory budget 1024 MiB (explicit); sampled owned child group only
# PREFLIGHT base source .../native-base-029c0ec7ade8/local_operator/__init__.py CSS local_operator.tcss HOME .../native-evidence/lop-visual-4s4h433e
# CAPTURE PASS base 52 frames
# [run_bounded] python rc=0 wall=27.6s
```

`env -i` strips inherited CMUX/LOP flags and XPC_FLAGS; `NO_COLOR` absent, `TERM=xterm-256color`. Isolation helper creates HOME/config beneath this session's scratch, CWD moved there before app construction, notifications/desktop launch disabled. Both runs printed the local operator-authority file-mode diagnostic in their isolated sandbox; no external auth/provider calls or OS window were launched. Owned process-group bound enforced by run_bounded; no measured peak-RSS claim. Follow-up process inspection found no remaining `cadence_native.py` process. Feature tree remains clean at pinned head.

#### Actual-app matrix

| Surface / state | Actual result | Verdict |
|---|---|---|
| French settings selected row at 60x20, 80x24, 150x40, dark + light | Real production CSS rendered the French label (ellipsized) and French help; matching base source renders English fallback | PASS capture/provenance |
| Editor at 60x20 + 80x24, both themes | Default `0`, typed `4`, typed `168` remain visible in the real inline editor; hint/cancel/focus retained | PASS |
| Rejected `169`, both narrow grids/themes | Actual Enter: editor stays open, buffer remains `169`, rendered error `must be at most 168`; isolated config assertion confirms no stored `169` | PASS |
| Correction and cancellation | Actual backspace/`8`/Enter corrects `169 → 168`, editor closes and row shows `168`; reopening/Escape returns to selected row without leaving settings | PASS |
| Entry/error first vs settled | Rendered all 104 SVGs to PNG using `rsvg-convert`; pixel-difference comparisons: **22 first/settled pairs per revision, 0 mismatches** | PASS |
| Theme geometry | Dark/light widget+screen geometry byte/value-equal at each tested grid; colors visibly change as expected | PASS geometry retention; no invented contrast ratio |
| Full label / complete help recovery | No full-copy recovery on tested selected/editor surface; measured truncation below. No tooltip/help binding found in the actual SettingsView bindings/detail code | FAIL recovery; designer disposition needed |

#### Q1 — MINOR: measured full-copy recovery gap (not an inferred clipping report)

Looked at rendered PNG comparisons in both themes at all three grids. The 36-cell French label displays `Intervalle entre les points…` in the fixed label column at all three widths; base label is complete. Enter opens the inline editor but does not reveal a full label. Help detail uses a one-text-line content region within a two-row band, not general wrapping for this cadence row.

Observed default/selected help, both themes:

- **60x20:** `0 conserve l’heure locale du point quotidien. De 1 à…`
- **80x24:** `0 conserve l’heure locale du point quotidien. De 1 à 168, l’intervalle se…`
- **150x40:** `0 conserve l’heure locale du point quotidien. De 1 à 168, l’intervalle se compte en heures entières écoulées ; 4 correspond à un point toutes l…`

Consequences measured, not assumed: at 60/80 the elapsed-hour clause/example/owed clause are not fully visible; at 150 French still loses the owed clause whereas the actual base English help fits completely there. Base already truncates its help at 60/80, so that narrow mechanism is pre-existing; French newly exceeds the wide default detail capacity. Stored off-default state sheds help in favor of `default: 0` and key path in both revisions. Classified MINOR as copy-fit degradation within unchanged, explicitly ellipsizing chrome; designer owns final visual disposition against D1. This is not a claim that the translation text is semantically wrong or a request for unrelated global settings redesign.

Production path corroboration: `SettingsView` `_detail_clause`/detail rung renderer uses `truncate_cells` when help alone exceeds width; the two-line wrap slot is restricted to other special rows. `SettingsView.BINDINGS` offers movement/change/reset/escape, not a full-help reveal. Thus this was taken one step beyond the initially surprising ellipsis rather than reported from pixels alone.

Measured geometry (screen sizes are content size, region equals requested terminal grid):

| Grid | Screen size = virtual | Settings content size = virtual | Settings border region | Detail border / content region | Screen/settings scrollbar flags |
|---|---|---|---|---|---|
| 60x20 | 58x18 | 56x11 | x1 y1 w58 h13 | x2 y9 w56 h2 / x2 y10 w56 h1 | false,false |
| 80x24 | 78x22 | 76x15 | x1 y1 w78 h17 | x2 y13 w76 h2 / x2 y14 w76 h1 | false,false |
| 150x40 | 148x38 | 146x31 | x1 y1 w148 h33 | x2 y29 w146 h2 / x2 y30 w146 h1 | false,false |

The list itself is a deliberately scrolled 293-row content widget (not a terminal overflow); screen and SettingsView virtual sizes match their actual content sizes, with no screen/settings scrollbar introduced. Every capture retains all visible-widget geometry including list/scroll containers for independent inspection.

#### Evidence handoff

All actual outputs, not repository additions:

`/Users/damian/.local-operator/sessions/b65f5adc1dfb/scratchpad/native-evidence/`

- `head/` and `base/`: each 52 SVG + PNG + `.geometry.json` + `.state.json`, plus `matrix.json`.
- `head-run.log`, `base-run.log`.
- `compare-{dark,light}-{60x20,80x24,150x40}.png`: six base-left/head-right sheets, all actually viewed here. Narrow sheets show selected, editor0/4/168/169, corrected168, cancelled states.
- Repro script: sibling `cadence_native.py`.

Native frame capture is no longer BLOCKED. Designer should consume these artifacts and judge Q1 fit/recovery; independent QA does not silently convert missing full-copy recovery into visual approval. No repo edits, full suite, dependency install, browser/Electron/window launch, merge, review request or handle tag.
