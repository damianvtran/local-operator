# Independent audit — es `tui.small` (round 2, audit id w6a2)

Auditor: translation-auditor (lopdev team), independent of the translator. I
authored none of this copy, repaired none of it, and edited nothing outside
`i18n/audits/evidence/es/`. Model actually running: **radient/auto** (the
session header is `anthropic/claude-sonnet-5-5`; provider-quota failures moved
this run through `openai/gpt-6-astra` and `kimi/k3.1-pro` first). No catalogue,
sidecar, ledger or allowance file was touched.

- English source: `local_operator/i18n/catalogues/en/tui.small.json` (137 keys,
  sha256 `16d49c4ba2a514bbde7da494443abade26a3b28c0389e6bca124d00aa7d0ed75`) —
  **byte-identical to the round-1 source**.
- Context sidecar: `.../en/tui.small.context.json` (137 entries, sha256
  `1e3d6785f78126b8389b2acbcc081d5bf782fbaf744114e47611fbff3c45dde1`) — unchanged.
- Target: `local_operator/i18n/catalogues/es/tui.small.json` (137 keys, sha256
  `e374959d70bb379a37e3f02e8c69edeea168b7edbd11107e4131f41fe116b9af`, UNTRACKED
  at HEAD `b925c2e`). Round 1 audited `c48357ff…763ed145`.

## Verdict

**passed** — 0 blocker, 0 major, 0 minor, 0 nit, 0 new findings. Four of the
five round-1 findings are closed with the requested wording; the fifth (F4) is a
recorded constraint that is still intact and still enforced. The remediation is
**exactly** the five recorded values: reverting them reproduces the round-1 file
digest byte-for-byte (`checks.py` C10), so no value outside the recorded delta
moved.

## What round 2 judged

1. **Each fix against the round-1 ask** — every one is the proposed wording,
   character for character, and none lost a placeholder, an indent or a nuance.
2. **The values the fix touches indirectly** — the spend ladder, the goal-panel
   heading and its two overflow rows, and the network member row. C10 proves the
   first point formally: only the recorded values differ from round 1.
3. **Widths, re-measured** — `rich.cells.cell_len` on the string the renderer
   *builds* (placeholders substituted with the values the code supplies), not on
   the raw `{name}` token, which overstates a digit slot by five cells.

## Round-1 findings on the current bytes

| Id | Sev | Key | Round-1 ask | Current es | Status |
|---|---|---|---|---|---|
| F1 | minor | `info_panel.deeper_below_depth` | `(bajo la profundidad {depth})` | `  +{deeper} más abajo (bajo la profundidad {depth})` | **closed** |
| F2 | minor | `spend.unpriced_partial` | `algunas sin precio` | `algunas sin precio` | **closed** |
| F3 | nit | `network_panel.suspect_tag` | `  [sospechoso: clave quizá copiada]` | `  [sospechoso: clave quizá copiada]` | **closed** |
| F4 | nit | `copy.code_count` | no change requested (constraint) | `{shown} código` | **held — intact** |
| F5 | nit | `goal_panel.more_settled` (+`_indent`) | `… {hidden} más finalizados` | `… {hidden} más finalizados` / `  … {hidden} más finalizados` | **closed** |

### F1 — `tui.small.info_panel.deeper_below_depth`

`bajo` + a bare noun was the defect; `bajo la profundidad {depth}` restores the
article and the row now reads as Spanish. Measurement, substituted:

| Form | Cells | Why |
|---|---|---|
| round 1 (`bajo profundidad`) at `{deeper}`=3 | 35 | pre-fix |
| **current** at `{deeper}`=3, `{depth}`=3 | **38** | `_MAX_TREE_DEPTH = 3` (info_panel.py:145) |
| current at `{deeper}`=1200 (raw, ungrouped) | 41 | the sidecar's raw-digit contract |
| en at the same values | 27 / 30 | for comparison |

No `maxCells` is declared. The row is appended to the info card, whose content
width is `max(_MIN_CARD_WIDTH, …)` — 38 cells at the floor
(`info_panel.py:141,1520`) and 88 by default. Measured against those: the
common-case row (38) fits the narrowest possible card exactly, and only the
ungrouped-`1200` row (41) crops there. Pre-fix that row was 35, so the fix adds
three cells of exposure in a regime that already cropped below 35; at the
default 88-cell card the common-case row has 50 cells of slack. Accepted. (If a
future card shrinks below 38, `(por debajo de {depth})` is the tighter natural
form, at 45 cells.)

### F2 — `tui.small.spend.unpriced_partial`

`algunas sin precio` restores the quantifier the rung exists to carry: the row's
figure is a **lower bound**, and without it the note read as a flat claim about
the whole row. Feminine plural agrees with the implicit *peticiones/llamadas*
(both feminine), so no reader is left with a dangling masculine. **18 cells
against the 24 declared** (en 22), so the widest rung still fits its budget.

The neighbours did not move and still read as four distinct states: `precio
parcial` (14 cells, budget 15), `incompleto` (10, budget 12), `sin precio` (10,
budget 15), `sin precio del servidor` (23, budget 24). No other key in the
namespace contains `algunas`.

### F3 — `tui.small.network_panel.suspect_tag`

**50 → 35 cells** (en 30), keeping every element: the two-space indent, the
brackets, the colon, and the deliberately cautious hedge. Agreement is right —
*clave* (f.) → *copiada* (f.). The tag is appended inline after the member row's
fixed 29-cell prefix (indent 2 + state 7 + id 12 + role 5 + spaces) plus the
capability list, so at the panel's measured 56x20 frame (card 43 cells) it still
wraps; it is 15 cells shorter than the round-1 marker, which recovers a
continuation line wherever the row had been two lines over.

### F4 — `tui.small.copy.code_count` — held, not changed

The value is still exactly `{shown} código`. Round 1 recorded this as a
**constraint**, not an oversight, and round 2 re-proves all three legs of it
(C13): the sidecar still says the key deliberately has NO plural branch; the
runtime's own argument parser still reports one `simple` slot; and a plural probe
still parses to a different kind, so the parity rule still refuses the upgrade.
Nothing to fix, and the next person to touch it will trip C13 deliberately
rather than silently.

### F5 — `tui.small.goal_panel.more_settled` / `more_settled_indent`

`… {hidden} más finalizados` is the natural order (trailing `más` was the nit),
and the participle now agrees with the heading it sits under. The goal panel's
own vocabulary is masculine singular/plural throughout — `settled_heading`
`finalizados`, `none_yet` `aún ninguno`, `done_suffix` `hecho` — so `más
finalizados` agrees with the noun the section has already established
(*objetivo*). In the "no settled row fits" branch the card reads:

```
finalizados
  … 3 más finalizados
```

which mirrors en's `settled` / `… 3 more settled` rather than inventing a shorter
Spanish form. The indent variant is exactly the base form plus two spaces
(asserted by C12, not eyeballed). No change.

## Sibling sweep

- **Spend ladder (16 declared budgets):** all within, the tightest two unchanged
  from round 1 — `spend.no_charge` 9/9 and `spend.unpriced_cancelled` 30/30.
  The remediated rung is the only one that moved and it went *down* (21 → 18).
- **Goal panel:** heading, `none_yet` and `done_suffix` are byte-identical to
  round 1 (proved by C10, not by trust).
- **Network panel:** the `relé:`/`comprobando…` rows keep their indent and
  newline (C5), and the changed tag does not touch them.
- **`copy.*` block:** `code_count` verified, everything else untouched.

## Mechanical floor

`checks.py` (this bundle, run on these exact bytes) reports **C1–C8 pass, C10–C13
pass**, the repo validator reports `0 problem(s)` tree-wide, and **15 negative
controls are all CAUGHT** — the nine carried from round 1 plus six round-2
controls: an edit outside the recorded delta (scope drift), a recorded key left
at its round-1 value, a dropped placeholder, a budget breach on the remediated
rung, an ASCII ellipsis reintroduced, and a goal-panel overflow row losing
agreement with its heading.

## Environment note (not a property of these bytes)

While this audit ran, the working tree's generated
`local_operator/i18n/data/shipped.json` did **not** list `es` (`M` relative to
HEAD, where the list is `en, fr, es, zh-CN, ru, vi, ur, hi`). Consequently
`resolve.eligible_locale("es")` returns `"en"` and `messages.render(…,
locale="es")` renders **English** — the ship gate doing its job mid-wave, since
the ledger has no passed row for `es` yet. C9 therefore renders through the same
catalogue lookup + ICU formatter `render_record` uses *after* the gate, and says
so in `audit.log`. The gate state is reported, never judged, and this audit did
not modify the file. The plural selection itself is exercised and correct:
`1 miembro` / `2 miembros` / `1.000.000 de miembros`, `1 imagen` / `2 imágenes`
/ `1.000.000 de imágenes`, and the same for every plural-bearing key in the
namespace.

## Limitations

- LLM semantic judgement, no native-speaker attestation; the team's gate is this
  independent audit.
- No rendered TUI frames: this host is loaded and heavy runs are forbidden.
  Widths are `rich.cells.cell_len` on the built strings; the panel, card and
  status measurements are read from the code that pins them.
- The target file is untracked at `b925c2e`; this verdict binds the sha256 above,
  not a commit.
- C9's ship-gate bypass is disclosed above; it proves the ICU layer, not a
  widget-level render.
