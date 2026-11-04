# Independent audit — `tui.import` (es) — round 2

**Namespace:** `tui.import` · 87 keys · English source 87 keys · **no English context sidecar**
**Audit id:** `w6a2` · round 2 · 2026-10-30
**Auditor:** independent (authored none of this copy, applied none of the remediation)
**Verdict:** `passed` — 0 blocker, 0 major, 0 minor, 4 nit (3 deferred from round 1, 1 recorded now)

Round 1 (`w6a1`) requested changes on this namespace: one major (I1, the
sessions-state hint row overflowed the documented 76-cell row at 80×24) and two
minor. The implementer applied all three, to exactly the wording round 1 asked
for:

| finding | key | round-1 ask | now | measured |
|---|---|---|---|---|
| I1 | `hint_select` | `elegir` in `space elegir` | applied | sessions row **75** / 76 |
| I2 | `hint_list` | `listar` | applied | result row 25 |
| I3 | `error_select` | `… con la tecla espacio …` | applied verbatim | 75 cells, complete |

## I1 — closed. The row now fits, verified by re-assembling and re-measuring

This is the finding the manager asked me to re-measure rather than accept, so I
did it from the current bytes in two independent ways.

**1. The assembly rule is proven, not assumed.** `HintButton._build`
(`subagent_view.py`) prepends `" · "` to every non-lead hint and joins key and
label with one space; `_hint_labels` (`harness_import_view.py:505-529`) fixes
button order and the per-state subset. I mounted the **real widget** headless at
80×24 with the two stylesheet paddings applied (`Screen { padding: 1 }`,
`.settings-view { padding: 1 1 1 1 }`) and let it paint its own sessions row: it
produced `↑↓ move · space select · a all/none · enter preview · esc back`, 62
cells, in a 76-cell row. The assembler in `checks.py` reproduces that string
byte-for-byte from the English catalogue — so the same rule applied to the
Spanish values is measuring the row a user sees.

**2. The rows, measured with `rich.cells.cell_len`:**

| state | es | en | cells | budget 76 |
|---|---|---|---|---|
| sources | `↑↓ mover · enter abrir · r recargar · esc volver` | 42 | 48 | fits |
| **sessions** | `↑↓ mover · space elegir · a todas/ninguna · enter vista previa · esc volver` | 62 | **75** | **fits, 1 spare** |
| preview (import) | `enter importar · esc volver` | 23 | 27 | fits |
| preview (confirm) | `enter confirmar · esc volver` | 24 | 28 | fits |
| progress | `c cancelar · esc volver` | 19 | 23 | fits |
| result | `enter listar · esc volver` | 21 | 25 | fits |
| loading / error / empty | `esc volver` | 8 | 10 | fits |

The sessions row ends with the complete `esc volver` hint (10 cells) at offset
65, inside the budget — the way out is painted, which was the whole defect.
80 cells would have been the row with `seleccionar`; `elegir` is the 5 cells
that make it fit, and it is the *same* word the error path now uses
(`Elige claude, codex u omp`), so the screen teaches one verb.

**Residual, measured and stated rather than hidden (I4 below):** `_hint_labels`
gates the move hint on `self.size.width >= 74`, and in Textual `Widget.size` is
the *content* width. So the narrowest box in which all five hints paint is 74
cells — a 78-column terminal — where this 75-cell row is **1 cell over** and the
final character of `volver` clips (`… · esc volve`). At 79 columns the row and
the box are both 75; at 80 it has 1 spare. This is a one-column band, the key
itself stays visible, and it is a consequence of a gate constant derived from
the English row (62 cells). It does not reopen I1, which was about the
documented 80×24 row.

## I2 — closed

`listar` — infinitive, matching the hint family (`mover`, `abrir`, `recargar`,
`cancelar`, `confirmar`, `importar`, `volver`) and the guide's action-label rule.
The result row reads `enter listar · esc volver`, 25 cells. Round 2 asserts the
family mechanically now: `[register]` requires an infinitive on all nine verb
hints, and the control that restores `lista` trips it.

## I3 — closed, and the longer message was checked for truncation

`Selecciona al menos una sesión con la tecla espacio antes de previsualizar.`
exactly as asked; the ambiguity of `con espacio` ("with room") is gone, and the
referent matches the hint row's `space`. The wording grew by 17 cells, so I
checked where it lands: `error_select` feeds only `self._empty`, which
`_body_rich` appends without `truncate_cells` into the scrollable body. The rig
injects the actual es value into the empty state at 80×24: the body is 76 cells
wide, the message is **75 cells**, and the widget holds the **complete string**
(`body_text_plain_contains_full_message=True`). No truncation, no dependence on
the fix being the shorter variant.

## I4 — nit (new): the gate-boundary column

- Key: the sessions-state row (`hint_move` + `hint_select` + `all_none` +
  `hint_preview` + `hint_back`).
- Measured: row 75 cells; documented 80×24 budget 76; `show_move` box floor 74
  (a 78-column terminal) → **1 cell clipped, the final `r` of `volver`**.
- Why not higher: the documented budget passes with a cell to spare, the band is
  one terminal column wide, the `esc` key remains visible, and the English row
  is 12 cells under the same floor (so the floor was set for English).
- If the owner wants every gated box to fit, the copy route is
  `a todo/nada` in place of `a todas/ninguna` (row **71** cells — fits the
  74-cell floor, at the cost of the feminine agreement with *sesiones*); the
  code route is raising the gate to `>= 77`. I do not request either: the copy
  is correct as it stands and the second option is out of translation scope.

## Deferred round-1 nits — dispositions, not repeats

- **N1 `result_memory`** (`ocultados` for "redactions") — not applied. Judged
  acceptable: `ocultados` describes what the reader sees (content hidden from
  the import), the count and the fact are right, and `censurados` is a
  refinement rather than a correction. Deferred.
- **N2 `selected_count`** (`seleccionadas: {count}` vs `{count} seleccionadas`)
  — not applied. Still the weaker shape (the count sits where `_detail_text`'s
  truncation would eat it), but the fact is right and it is a nit. Deferred.
- **N3 `metadata_only_explain`** (`rollout` kept English) — recorded-only in
  round 1, no wording requested; nothing to apply. The kept machine term is
  correct under the glossary's product-name policy.

## Regression sweep — nothing else moved

I re-read all 87 values, not only the three that changed. The destructive paths
round 1 verified are byte-identical (`preview_confirm`,
`preview_confirm_metadata` and both narrow forms keep the past/future negative
distinct; `rerun_hint` still names all three source behaviours under the safety
claim; `subtitle` keeps the read-only source; `metadata_only_explain` keeps
*"no se puede importar nada"*). The state/outcome gender split
(`cancelado/terminado/fallido` for a *trabajo*, `cancelada/importada/fallida`
for a *sesión*) is unchanged and principled. `u omp` euphony intact in both
keys. All 15 plural messages keep `one`/`many`/`other`, render correctly at
0 / 1 / 2 / 10⁶ / 2×10⁶ / 1.5×10⁶, and lose no branch content.

## Mechanical evidence

`checks.py` in this bundle, run as

```sh
env PYTHONPATH=$PWD /Users/damian/local-operator/.venv/bin/python \
  i18n/audits/evidence/es/tui.import/w6a2/checks.py
```

result: **`PASS — all checks clean, every negative control tripped`** (0 blocking
failures; 14 negative controls, including the three round-2 ones: the restored
`seleccionar` row at 80 cells, a grown sibling term at 91 cells, and the noun
`lista` in the infinitive family).

The repository validator (`python3 scripts/i18n/check.py --catalogues-only`)
reports **0 problem(s)** on the current tree. `audit.log` is the execution log
for every command; `checks.py` is the script that produced it.

## Provenance limit stated plainly

The production resolver serves English for `es` until this namespace has a
passing ledger row; at audit time this working tree's
`local_operator/i18n/data/shipped.json` also listed locales WITHOUT `es` (a
concurrent edit by another session in this shared checkout, not mine), so the
widget answered `en` and could not paint the Spanish row through the live
resolver. The EN row was painted by the real widget (rig block
in `audit.log` §7) and the assembler reproduces it; the Spanish row was then
measured from the catalogue values with that proven rule. No rendered Spanish
frame is claimed, and no native-review sign-off is implied.

## Ledger data (for the coordinator — I do not edit the ledger)

- locale `es` · namespace `tui.import` · status **passed**
- `source_sha256` (en) `932e0f87210ff80950346c684afb000907610c9ff35388125977182120b136c2`
- target sha256 `17a4dec148f5de8bfb07e79c35e0b84eaa02da13bbf713676c42dbaa4d840725`
- auditor `i18n-auditor` · model actually running `radient/auto` (fallback chain
  disclosed in `record.json`) · 2026-10-30 · round 2
- unresolved blocker/major: **0**
