# Independent audit — `tui.import` (es) — round 1

**Namespace:** `tui.import` · 87 keys · English source 87 keys · **no English context sidecar**
**Auditor:** independent (authored none of this copy, repaired none of it)
**Verdict:** `changes_requested` — 0 blocker, **1 major**, 2 minor, 3 nit

## What I read

`i18n/glossary.md` and `i18n/style-guides/es.md` in full, then the shipped es TUI
catalogues for register precedent (`tui.status`, `tui.transcript`, `tui.viewer`,
`tui.palette`), then the English source with its keys, then all 87 Spanish
values. Because there is **no sidecar for this namespace**, I took the register
question to shipped copy and to the widget that consumes these keys
(`local_operator/tui/widgets/harness_import_view.py`), and judged `maxCells`
from the code's own documented budgets rather than from a hint that is absent.

I then re-read every **destructive and data-loss-adjacent key** as a second
pass: `preview_confirm`, `preview_confirm_metadata`, `preview_narrow`,
`preview_narrow_metadata`, `rerun_hint`, `metadata_only_explain`,
`outcome_metadata_only`, `availability_metadata_only`, `subtitle`,
`error_busy`, `error_cancel`, `result_memory`.

## The two questions the task asked me to settle

### Does `preview_confirm` really say *nothing has been* imported and nothing *will be*?

Yes, and both halves survive:

| key | English | Spanish | reads as |
|---|---|---|---|
| `preview_confirm` | `Nothing has been imported. Enter imports the selected sessions.` | `No se ha importado nada; enter importa las sesiones seleccionadas.` | nothing has been imported **yet**; `enter` is what performs the import |
| `preview_confirm_metadata` | `Nothing will be imported; the source index has no conversation body.` | `No se importará nada; el índice de la fuente no tiene cuerpo de conversación.` | nothing **will** be imported, and why |
| `preview_narrow` | `… · Nothing imported yet. Enter imports.` | `… · Aún no se ha importado nada; enter importa.` | same, narrow form |
| `preview_narrow_metadata` | `… · Nothing will be imported.` | `… · No se importará nada.` | same, narrow form |

The Spanish keeps the **present perfect** (`no se ha importado nada`) for the
past-negative and the **future** (`no se importará nada`) for the
future-negative. The distinction the English draws is the distinction the
Spanish draws. The `;` instead of `.` is correct es typography and does not
chain the clauses into a promise.

### Are all three `rerun_hint` source behaviours named?

Yes. `Repetir es seguro: las fuentes sin cambios se omiten, las que han crecido
se reanudan y las reescritas reciben una copia nueva.`

| behaviour | English | Spanish | present |
|---|---|---|---|
| unchanged → skipped | `unchanged sources are skipped` | `las fuentes sin cambios se omiten` | ✓ |
| grown → resumes | `grown sources resume` | `las que han crecido se reanudan` | ✓ |
| rewritten → new copy | `rewritten sources get a new copy` | `las reescritas reciben una copia nueva` | ✓ |

The safety claim itself (`Re-run is safe`) is preserved and placed first, where
it does the reassuring. Nothing here implies the re-run overwrites or loses
data.

### Is every destructive path as explicit as the English?

- `metadata_only_explain` — `no se puede importar nada` (nothing **can** be
  imported) plus the reason (`no el cuerpo de la conversación`). Explicit.
- `outcome_metadata_only` — `solo índice — nada que importar`. Explicit.
- `subtitle` — `La fuente sigue siendo de solo lectura` keeps the read-only
  reassurance that makes the whole import flow safe. Explicit.
- `error_busy` — `Ya hay una importación en curso` blocks a second concurrent
  import. Explicit.
- `error_cancel` — `No hay nada en curso que cancelar`. Explicit.
- `result_memory` — `memoria importada: {imported} · omitidos: {skipped} ·
  ocultados: {redactions}`. No data-loss misstatement (see N1 on the word
  choice).
- `review_memory` / `review_instructions` — both keep `guardada para revisar en
  {path}`, i.e. the content still exists somewhere.

No destructive action is described more weakly than the English describes it.

## What I found

### Gender of the state vocabulary — correct, and worth recording

`state_*` renders masculine (`cancelado`, `terminado`, `fallido`) and
`outcome_*`/`phase_cancelled` render feminine (`cancelada`, `importada`,
`fallida`). I checked the consumers rather than assuming drift: `state_*` fills
`Job {job} · {state}` where "job" is **trabajo** (masculine); `outcome_*` fills
`{outcome} · {messages…}` on a per-**sesión** result (feminine); `phase_*` fills
`{completed}/{total} · {phase}` where "phase" is **fase** (feminine). The split
is principled and consistent, not two renderings of one word.

### Terminology against shipped precedent

- `harness` kept English (`Harnesses`, `de otro harness`, `el harness`) — the es
  guide's wave-1 kept-English list. ✓
- `job` → `trabajo` (`Trabajo {job}`, `ningún trabajo de importación`) — the
  guide's recorded `job` decision. ✓
- `caché`, `coste`, `tenant`, `daemon`, `plantilla incluida` do not occur here.
- `solo índice` and `no se puede importar nada` are spelled without the
  pre-2010 accent on `solo`, which is the current RAE rule. ✓
- `o omp` / `u omp` — `u` before an `o` sound (`Codex u omp`) is correct
  Spanish; the catalogue gets it right in both `error_source` and `no_sources`.
- `rollout` stays Latin; it is the Codex artefact's own name (see N3).

### Width — one real overflow

There is no sidecar here, so I measured the hint row the widget actually
assembles. `harness_import_view.py` documents the row's budget in its own
comment: *"The hint row still fits at 80x24 (76 cells)"*, and `show_move` keeps
the `↑↓` hint whenever `size.width >= 74`, so at a standard 80×24 all five
session-state hints paint. Hints render as `key label` joined by ` · `
(`HintButton._build`).

| locale | assembled sessions-state row | cells |
|---|---|---|
| en | `↑↓ move · space select · a all/none · enter preview · esc back` | 62 |
| es | `↑↓ mover · space seleccionar · a todas/ninguna · enter vista previa · esc volver` | **80** |

80 cells into a 76-cell row: the row overflows by 4 and the trailing hint is
clipped — that is `esc volver`, the way out of the screen. Spanish runs long
here because four labels each pay 1–5 cells (`seleccionar` +5, `vista previa`
+5, `todas/ninguna` +5, `volver` +2, `mover` +1). A compliant row is available
without losing information: `space elegir` instead of `space seleccionar`
measures **75** cells; `a todo/nada` instead of `todas/ninguna` measures **76**.
This is finding I1 and it is the one thing that must change.

### Register of the hint family

Seven of the ten hint labels are infinitives (`volver`, `cancelar`,
`confirmar`, `importar`, `mover`, `abrir`, `recargar`, `seleccionar` — eight,
counting `seleccionar`), matching the guide's "Action labels: infinitive" rule
and the shipped `tui.viewer` legend (`r recargar · o abrir externamente · esc
cerrar`). Two are nouns: `lista` and `vista previa`. In the result state
`hint_list` sits directly beside `esc volver`, so one row mixes a noun and a
verb. See finding I2.

## Findings

### I1 — MAJOR — import hint row overflows its documented budget

- Keys at fault: `tui.import.hint_select`, `tui.import.hint_preview`,
  `tui.import.all_none`, `tui.import.hint_move`, `tui.import.hint_back`
- Assembled row (sessions state): `↑↓ mover · space seleccionar · a todas/ninguna · enter vista previa · esc volver` = **80 cells**
- English row: 62 cells · budget documented in the view: **76 cells at 80×24**
- Problem: the row cannot fit, so the trailing hint (`esc volver`) is clipped at
  the default terminal size. The user loses the visible way out of the screen
  on the screen's main state.
- Achievable: `space elegir` for `seleccionar` → **75 cells** (measured);
  or `a todo/nada` for `todas/ninguna` → **76 cells** (measured, exactly at the
  limit). Both keep every hint and every key named.
- Why major, not minor: overflow is a defect in this surface, not a preference —
  the copy is read in a dense panel where a clipped hint is invisible.

### I2 — minor — `hint_list` is a noun in an infinitive hint family

- Current: `lista` · English: `list` · suggested: `listar`
- Problem: the other hint labels are infinitives per the guide's action-label
  rule and the shipped `tui.viewer` precedent; `lista` names a thing. In the
  result state it is painted next to `esc volver`, so the row reads
  `enter lista · esc volver` — a noun beside a verb.
- `hint_preview` (`vista previa`) is the same class of deviation but I do not
  gate it: `vista previa` is the settled Spanish noun for a preview screen and
  this catalogue already uses it consistently as a noun (`preview_header`,
  `loading_preview`, `outcome_metadata_only`). Recorded, not required.
- Suggested: `listar`

### I3 — minor — `error_select` says "con espacio", which is ambiguous in Spanish

- Current: `Selecciona al menos una sesión con espacio antes de previsualizar.`
- English: `Select at least one session with space before previewing.`
- Problem: `con espacio` in Spanish reads first as "with room / with space".
  The key is the space bar (the hint row names it `space seleccionar`). This is
  the message a blocked user is given to unblock themselves, so an ambiguous
  instruction costs an extra failed attempt.
- Suggested: `Selecciona al menos una sesión con la tecla espacio antes de
  previsualizar.` (or `con la barra espaciadora`)

### N1 — nit — `result_memory` renders "redactions" as `ocultados`

- Current: `memoria importada: {imported} · omitidos: {skipped} · ocultados: {redactions}`
- English: `{imported} memory imported · {skipped} skipped · {redactions} redactions`
- Problem: `ocultados` reads as "hidden ones". What happened is that content was
  scrubbed for safety, which Spanish expresses as `censurados` or `redactados`
  rather than `ocultados`. The count and the fact are right; the word is the
  plain-language approximation. Not a misstatement — hidden is what the reader
  sees — so it does not gate.
- Suggested: `censurados`, if a precise term is wanted.

### N2 — nit — `selected_count` inverts the count order used by its sibling

- Current: `seleccionadas: {count}` · English: `{count} selected`
- Problem: the same fact is rendered as `{count} seleccionadas` by
  `preview_narrow`, so one screen shows two shapes. The count also sits last,
  where `_detail_text`'s `truncate_cells` would eat it; the English leads with
  the number.
- Suggested: `{count} seleccionadas`

### N3 — nit — `rollout` kept English in `metadata_only_explain`

- Current: `Puede que el harness haya eliminado el rollout.`
- Problem: the source keeps `rollout` as the Codex artefact's own name; a
  Spanish reader outside Codex may not know it. Acceptable as a kept machine
  term, but it is a term the reader must decode from context.
- Suggested: leave as is, or `el volcado de sesión` if the reader is not
  expected to know Codex internals. Recorded, not required.

## Mechanical evidence

`checks.py` in this bundle, run as

```sh
PYTHONPATH=$PWD /Users/damian/local-operator/.venv/bin/python \
  i18n/audits/evidence/es/tui.import/w6a1/checks.py
```

result: `PASS — all checks clean, every negative control tripped` (the script
also asserts the positive semantic rules the task asked about: the three
`rerun_hint` behaviours, the past-negative/future-negative split in the four
preview keys, the read-only claim, and the absence of any past-tense import
claim or data-loss claim).

It checks key parity, ICU parse, argument-tuple parity with EN, Spanish CLDR
categories, branch losslessness, real branch selection at six sample counts,
protected machine tokens (`<source>`, `[id]`, `/import`, `→`, backticks, flags),
negation parity, glossary rules and byte-identical-to-EN prose. There is no
context sidecar, so there are no `maxCells` budgets to assert — the script says
so and the width finding above is measured from the assembled hint row instead.

Eleven negative controls each break one thing and assert a check catches it;
all eleven caught — including a deleted `many` branch, an emptied plural
branch, an injected forbidden `few`, a translated `<source>`, a dropped
`nothing has been imported`, a dropped `rerun_hint` behaviour, and an injected
`maxCells` budget that the English fits and the Spanish does not.

## Limitations

- No context sidecar exists for this namespace. I did not treat that as the
  translator's fault; it means register was judged against shipped TUI copy and
  the consumer widget, and widths against the widget's own documented budget
  rather than per-key `maxCells`.
- LLM semantic judgement, no native-speaker attestation (not a gate).
- No rendered TUI frames. The overflow finding is arithmetic on the exact
  strings the widget concatenates plus the view's own stated 76-cell row, not a
  captured frame; a rendered frame at 80×24 would confirm the clipped tail and
  is the natural next evidence step if the fix is disputed.
- The target catalogue is **untracked** at the audited revision; the binding is
  to its on-disk sha256.
