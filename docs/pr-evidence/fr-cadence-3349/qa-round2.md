### QA report — round 2

Reviewer: qa-tester / anthropic/claude-sonnet-5-5 (actual resumed runtime model; no fallback/model/default/tool changes invoked in this pass)
Scope: dcdb0b280553861b72b225695497810d201ef890..005d1eed1dbb1de84db78398ab188ec3df9b821d
Head: 005d1eed1dbb1de84db78398ab188ec3df9b821d
Verdict: PASS — terminal remediation/direct-regression QA; Q1 fixed for translated-label fit and lost wide help. Existing narrow single-line help truncation remains disclosed below, not a new defect. Designer retains independent visual judgment.

Operator request verbatim: “Make sure to continue the work all the way for all languages to get the entire UI completely translated”. Bounded acceptance remains these two French cadence values, including daily-local 0, whole elapsed-hour 1–168, example 4, and owed-point preservation. No new UI/behavior, other namespaces/refactors/version changes.

#### Scope and preserved evidence

Only the remediation delta was examined: two shortened French values plus new audit evidence/append-only ledger rows. Actual `git diff --name-only dcdb0b280553..HEAD -- local_operator/tui scripts/visual_capture.py local_operator/settings_io.py local_operator/i18n/messages.py local_operator/i18n/text.py local_operator/i18n/resolve.py` returned empty. Thus unchanged runtime/registry validation and original generator/manifest checks remain sourced to round 1; no full runtime/test matrix or full suite was repeated. The actual earlier EN control frames at `029c0ec7ade8be8bdb6625a5816b4e23fc0a2902` and prior French frames are retained, not recreated.

#### Setup and commands actually executed

Admission measured here: load `9.58 15.23 16.63`, memory free `72%`, disk available `20 GiB`. This satisfies the manager's load <=20 / memory >=25% / disk >=15 GiB admission. No overlapping UI native dispatch.

Reused prior rig with only expected French label and state list changed: selected row, default editor, rejected 169 and correction to 168 at all three grids, both themes. Real `OperatorApp`, actual `local_operator.tcss`, `FakeSession/_factory`, `isolate_capture()` before imports, explicit feature-tree source identity, isolated HOME/config/CWD, env cleared of CMUX/LOP inheritance, TERM set, NO_COLOR absent. Existing primary venv, no installation/build. Import-only preflight passed before actual app dispatch.

```sh
P=/Users/damian/local-operator/.venv/bin/python
R=/Users/damian/local-operator/.worktrees/french-cadence
S="$LOCAL_OPERATOR_SCRATCHPAD"
cd "$S/native-r2"
env -i HOME="$S/native-r2" PATH="/opt/homebrew/bin:/usr/bin:/bin" TERM=xterm-256color \
 "$P" "$S/cadence_native_r2.py" "$R" "$S/native-r2/head" head --preflight
# PREFLIGHT head source .../.worktrees/french-cadence/local_operator/__init__.py CSS local_operator.tcss HOME .../native-r2/lop-visual-6pg1m4_a

env -i HOME="$S/native-r2" PATH="/opt/homebrew/bin:/usr/bin:/bin" TERM=xterm-256color \
 "$P" "$R/scripts/run_bounded.py" --timeout 600 --memory-mb 1024 -- \
 "$P" "$S/cadence_native_r2.py" "$R" "$S/native-r2/head" head
# [run_bounded] memory budget 1024 MiB (explicit); sampled owned child group only
# PREFLIGHT head source .../.worktrees/french-cadence/local_operator/__init__.py CSS local_operator.tcss HOME .../native-r2/lop-visual-7u_6eqvk
# CAPTURE PASS head 42 frames
# [run_bounded] python rc=0 wall=30.8s

for f in "$S"/native-r2/head/*.svg; do
 /opt/homebrew/bin/rsvg-convert "$f" -o "${f%.svg}.png"
done
```

All six comparison sheets were ACTUALLY VIEWED with the image read tool: three selected-row sheets (base EN / old FR / current FR, dark and light) and three current editor/rejection/correction sheets (dark/light). Not XML-only inspection.

| Surface/command | Actual result | Verdict |
|---|---|---|
| `rich.cells.cell_len` on committed French values | Label **25** cells, help **133** cells; within supplied target budgets 29/137. No `maxCells` claim invented: unchanged English context sidecar has prose constraints but no explicit maxCells field | PASS |
| Actual selected row and editor, 60x20 / 80x24 / 150x40, dark/light | Full `Intervalle des points (h)` including hours unit visible without ellipsis; state assertions and viewed PNG agree | PASS, Q1 label fixed |
| Actual default detail and editor at 150x40, dark/light | Full exact help visible: `0 : point quotidien à l’heure locale. 1–168 : heures entières écoulées (4 : toutes les 4 h). Un point dû le reste après modification.`; equality asserted against committed catalogue | PASS, Q1 wide-help/owed clause fixed |
| Actual 169 input and Enter, each grid/theme | Buffer remains `169`, editor stays on `aida.cadence.interval_hours`, visible `must be at most 168`; isolated config check excludes saved 169 | PASS |
| Correct rejected 169 → 168 and Enter | Editor closes, selected actual row displays 168 in every grid/theme; no error remains in captured result | PASS |
| First/settled SVG → raster comparison (`PIL.ImageChops.difference`) | **18 pairs, 0 pixel mismatches** across selected/editor/error states | PASS |
| French catalogue bytes | SHA-256 `42333772890dd59733f0bcff715c9d608ff1f7c069c8dbb025ee3b2f0548ed63`, matches supplied remediation audit hash | PASS |
| Cleanup/source | Exact script-path process query returned no surviving capture process; `git status --porcelain` empty, HEAD still pinned | PASS |

#### Narrow geometry and remaining baseline limitation

Same measured geometry as round 1, dark/light:

| Grid | Screen size = virtual | Settings content size = virtual | Detail border / content region | Screen scrollbars |
|---|---|---|---|---|
| 60x20 | 58x18 | 56x11 | x2 y9 w56 h2 / x2 y10 w56 h1 | false,false |
| 80x24 | 78x22 | 76x15 | x2 y13 w76 h2 / x2 y14 w76 h1 | false,false |
| 150x40 | 148x38 | 146x31 | x2 y29 w146 h2 / x2 y30 w146 h1 | false,false |

No wrapping/reflow regression introduced; selected label now fits even in 60x20's editor. The unchanged one-text-line detail still visibly ellipsizes long help at narrow widths:
- 60x20: `0 : point quotidien à l’heure locale. 1–168 : heures…`
- 80x24: `0 : point quotidien à l’heure locale. 1–168 : heures entières écoulées (4…`

This is the previously demonstrated EN baseline single-line limitation, not newly claimed complete narrow-help recovery. No full-help reveal was added or tested; no general settings redesign requested. The actual remediation repairs complete label/unit visibility at every tested width and restores complete semantic help at the wide grid where the EN control was complete. That closes Q1's PR-introduced degradation; no blocker/major or new Q-finding from this delta. No new contrast ratio or unrelated theme/accent claims.

Artifacts for designer, all outside repo:
`/Users/damian/.local-operator/sessions/b65f5adc1dfb/scratchpad/native-r2/`
- `selected-comparison-{60x20,80x24,150x40}.png` (retained base/old-FR versus current-FR)
- `editor-comparison-{60x20,80x24,150x40}.png`
- `head/`: 42 SVGs + raster PNGs + per-frame geometry/state JSON and `matrix.json`
- `run.log`; repro script is sibling `cadence_native_r2.py`.

No repo edits, full suite, rebuilt baseline, install, browser/Electron/window launch, merge, review request or handle tag. Existing round-1 runtime dimensions retained with source compatibility proof, not rerun.
