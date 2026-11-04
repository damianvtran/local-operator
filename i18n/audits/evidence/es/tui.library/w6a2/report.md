# Independent audit — es `tui.library` (round 2, audit id w6a2)

Auditor: translation-auditor (lopdev team), independent of the translator. I
authored none of this copy, repaired none of it, and edited nothing outside
`i18n/audits/evidence/es/`. Model actually running: **radient/auto** (the
session header is `anthropic/claude-sonnet-5-5`; provider-quota failures moved
this run through `openai/gpt-6-astra` and `kimi/k3.1-pro` first). No catalogue,
sidecar, ledger or allowance file was touched.

- English source: `local_operator/i18n/catalogues/en/tui.library.json` (108 keys,
  sha256 `5a389b6a7d8de06621462f2685eff6dc95cd4cd6584d35dd6c26a0a19ad50b11`) —
  **byte-identical to the round-1 source**.
- Context sidecar: **none** — this namespace has no `en/tui.library.context.json`
  (unchanged from round 1); fit is judged from the render paths, not a budget.
- Target: `local_operator/i18n/catalogues/es/tui.library.json` (108 keys, sha256
  `81cbce350526ce25b68aeebd09175c747716dc8324e9277b424ec44cf6b3f205`, UNTRACKED
  at HEAD `b925c2e`). Round 1 audited `4b3e5982…ac4bc62`.

## Verdict

**passed** — 0 blocker, 0 major, 0 minor, 0 nit, 0 new findings. Both round-1
findings are closed, and reverting them reproduces the round-1 digest
byte-for-byte (`checks.py` C10), so nothing else in the namespace moved.

## Round-1 findings on the current bytes

| Id | Sev | Key | Round-1 ask | Current es | Status |
|---|---|---|---|---|---|
| F1 | minor | `rail_hidden` | `Los filtros aparecen a partir de {width} columnas.` | `Los filtros aparecen a partir de {width} columnas.` | **closed** |
| F2 | nit | `form_name_placeholder` | `si lo dejas vacío, se usa el nombre del archivo` | `si lo dejas vacío, se usa el nombre del archivo` | **closed** |

### F1 — `tui.library.rail_hidden`

The round-1 defect was structural, not stylistic: the string is painted into
`.library-view-status` (`height: 2; padding: 1 0 0 0` — **one** content row),
and it is painted *exactly when the rail is hidden*, i.e. below `RAIL_MIN_WIDTH
= 90` (library_view.py:95, 1443). The old wording was 76 cells, so on a 61–75
column terminal the Spanish notice was cut where the English still fitted.

Measured at `{width}` = 90 with `rich.cells.cell_len`, placeholder substituted:

| Form | Cells | Verdict |
|---|---|---|
| round 1 | 76 | clipped below 76 columns |
| **current** | **45** | fits from 45 columns up |
| en | 61 | clipped below 61 |

So the remediated line survives *every* width at which it can be shown down
to 45 columns: 16 columns narrower than the English and 31 narrower than what
it replaced. The claim is unchanged: the rail appears at `{width}` columns.
`a partir de 90 columnas` reads as "from 90 columns [wide]" in a terminal; the
dropped `de ancho` and the dropped actor are what bought the width. Accepted.

### F2 — `tui.library.form_name_placeholder`

`si lo dejas vacío, se usa el nombre del archivo` (47 cells, en 27) states the
*condition* first, which is what the old fragment (`si no, el nombre del
archivo`) made the reader infer. It agrees with the field's own label
(`form_name` = `nombre`), stays lowercase like its siblings
(`para qué sirve este recurso`, `separadas por comas`), and uses the guide's
`tú` (`dejas`), not `usted`.

Fit: the field is a full-width `Input` inside the detail pane
(`.library-view-form-input` has no width of its own; `.library-view-form` is
`1fr`), so 47 cells fits any pane wider than 47 and clips gracefully below that.
The clipped prefix is `si lo dejas vacío, se usa el no…` — the condition, which
is the load-bearing half, survives. Accepted. No budget is declared.

## Sibling sweep

- The rail's labels (`Hojas de cálculo` 16 cells, `sidebar_categories`
  `categorías` 10) are untouched and unchanged from round 1 (C10).
- `form_cancel` `cancelar` and `hint_cancel` `cancelar` still agree with each
  other; the remediated placeholder does not introduce a competing verb.
- The three prompt-shaped strings (`ask_prompt`, `ask_prompt_digest`,
  `ask_seeded`) are byte-identical to round 1 — C10 proves it, and C2/C4 keep
  their `{path}`/`{kind}`/`{session}`/`{id}` slots intact.

## Mechanical floor

`checks.py` (this bundle): **C1–C8 pass, C10–C13 pass**; the repo validator
reports `0 problem(s)` tree-wide; **13 controls all caught** (the nine from
round 1 plus four for round 2). This namespace still has no `maxCells` to
enforce, and C10 is what makes "nothing else moved" a fact rather than a claim.

Plural rendering (C9) exercises the es catalogue and formats correctly:
`0 recursos` / `1 recurso` / `2 recursos` / `1.000.000 de recursos`, and
`1 conversación` / `2 conversaciones` / `1.000.000 de conversaciones`. See the
audit.log note on the locale gate.

## Note on the run environment (not a property of these files)

`local_operator/i18n/data/shipped.json` in the working tree (not HEAD) omits
`es`, so
`resolve.eligible_locale('es')` returns `'en'` and the default `render()` path
would answer in English. This audit did not touch that file; C9 renders through
the same catalogue lookup + ICU formatter that `render_record` uses after the
gate, and prints the gate state in `audit.log`. The Spanish bytes are what was
exercised.

## Limitations

- No sidecar means no per-key role or budget: fit rests on the render paths read
  (`library_view.py`, `local_operator.tcss`).
- No rendered TUI frames (loaded host, heavy runs forbidden); widths are
  `rich.cells.cell_len` on the built strings.
- LLM semantic judgement, no native-speaker attestation.
- The target file is untracked at `b925c2e`; this verdict binds the sha256 above,
  not a commit.
