# Independent audit — `tui.analytics` (es) — round 1

**Namespace:** `tui.analytics` · 98 keys · English source 98 keys · context sidecar 98 entries
**Auditor:** independent (authored none of this copy, repaired none of it)
**Verdict:** `passed` — 0 blocker, 0 major, 2 minor, 3 nit

## What I read

In this order: `i18n/glossary.md` (all 173 lines), `i18n/style-guides/es.md`
(all 114 lines), the shipped es TUI catalogues already in the tree
(`tui.status`, `tui.transcript`, `tui.viewer`, `tui.palette`) for register
precedent, then the English source with every context note and all five
`maxCells` budgets, then all 98 Spanish values key by key.

I then re-read, as a second pass, every key that states a **money figure**, an
`unpriced`/`priced`/`paid` distinction, or a failure/empty distinction — the
concentration of harm in this block.

## What I found

### Money and the price distinctions — clean

This is a cost/usage panel, so I checked every figure-bearing key against the
English promise, not just its shape:

- `legend.cost` keeps the `+ lower bound` mark, the parenthetical
  `(algunas llamadas sin precio)`, the `$— no published price` mark **and** the
  deliberate three-space alignment run (asserted mechanically, not by eye).
- `search.incl_money_read` / `_search` keep `{money}` pre-formatted and add
  `en lecturas` / `en búsquedas`, which states what the money was spent on
  rather than understating it.
- The four `unpriced` rungs (`incl_unpriced_*`, `note_*_unpriced`) render
  `sin precio` — never `gratis`. Asserted as a **forbidden** rule, with a
  negative control that proves the assertion fires.
- `summary_free` renders `{count} gratis` and never `sin precio`; the two are
  different facts and the translation keeps them apart.
- `summary_free_paid` keeps the denominator (`{priced}`, the priced
  operations) and the money as `pagado`; the figures reconcile with the row
  above, which is the whole point of that key.
- `model.capped`, `rate.legend`, `table.calls` / `coverage` / `tokens` /
  `tok_s` / `cache` all keep their parameters and units.

No money figure is understated, re-derived, or re-formatted. No `+` floor and
no `$—` was dropped.

### State distinctions — clean

`chart.daily_empty` / `monthly_empty` ("no usage recorded yet") are kept
distinct from `model.failed` ("the per-model read failed") and `model.pending`
("reading the ledger…"). The scoped reassurance in `model.failed` survives as
`el resto del informe no se ve afectado` — the failure is not broadened to the
screen. `model.empty` stays parenthetical. This is the section where a
translation quietly collapses "empty" into "failed"; it does not.

### Plural agreement — correct

14 plural messages, each carrying exactly `one` / `many` / `other` (no `few`,
no `zero`). I did not take the branch table on trust: `check_render` drives the
real `render_message` at 1, 0, 2, 10⁶, 2×10⁶ and 1.5×10⁶ and asserts the
*shape* the branch produces. Spanish rules hold throughout:

| count | renders | category |
|---|---|---|
| 1 | `1 sesión` | `one` |
| 0 | `0 sesiones` | `other` (0 is never `one` in es) |
| 10⁶ | `1.000.000 de sesiones` | `many` |
| 1.5×10⁶ | `1.500.000 sesiones` | `other` — the guide's settled rule |

Table-cell agreement follows the noun: `({count} fallidas)` agrees with
`llamadas`, `· medidas` with the same, `{fresh} nuevos` with `tokens`.
The one singular `totals.calls_failed` reference (`fallidas`) is correct
because the count it renders is a call count.

### Width — one avoidable omission

All five declared budgets hold, measured with `rich.cells.cell_len` (the
function `_row_overhead` measures with):

| key | es | en | cap |
|---|---|---|---|
| `hint.expand.compact` | 28 | 29 | 31 |
| `hint.expand.full` | 39 | 40 | 42 |
| `hint.expand.full_metric` | 56 | 56 | 57 |
| `hint.expand.keys` | 24 | 22 | 25 |
| `hint.expand.keys_bare` | 14 | 13 | 12 |

`keys_bare` is over its cap **by construction**: English is 13 against the same
12. The question there is whether 14 is the best achievable Spanish, and it is
— `enter · e todo` is the floor once both key names must be named, and `e todo`
is already the shortest form of "all". I record it as unattainable, not a
defect.

`hint.expand.compact` is the one real width finding: it drops `fila` while
sitting three cells *under* its cap, and the narrower `hint.expand.keys` tier
(24 of 25) keeps it. See finding A2.

`hint.expand.full` and `.full_metric` render a bare `esc` where the sibling
tiers say `esc volver`. I checked whether that is avoidable and it is not:
`esc volver` costs 7 cells and `full` has 3 spare (`full_metric` has 1), and
every shorter back-word (`atrás`, 5) still overruns. Dropping the *action word*
rather than the *verb meaning* (`expandir`) or the *noun* (`fila`) is the right
trade — `esc` is universal and is taught by every other tier. Not a finding.

### Terminology checked against shipped precedent

- `caché` in all six cache keys (`table.cache`, `totals.cache_hit`,
  `cache_read`, `cache_write` + notes) — matches the W1 amendment and the 19
  shipped keys.
- `coste` in `t coste/tokens` — matches the wave-1 `coste` decision; `costo`
  does not appear.
- `tokens` kept English throughout, including `{value} tokens` and
  `tokens de contexto`.
- `sesión` / `subagente` / `registro` (ledger) / `proveedor` — all consistent
  with shipped `tui.*` copy.
- `decode` and `wall` stay Latin in `model.legend` while `decode` is translated
  to *decodificación* in `totals.decode`. That is **correct, not drift**: the
  legend decodes two *column headers* that render literally as `decode` and
  `wall`, so a translated header word would break the link between legend and
  column. The sidecar's rule ("`wall = whole call` must not be shortened to a
  rate word") is honoured by `llamada completa`.
- `daemon`, `tenant`, `plantilla incluida` do not occur in this namespace.
- `esc volver` matches the shipped `esc <infinitive>` pattern
  (`tui.viewer`: `r recargar · o abrir externamente · esc cerrar`).

## Findings

### A1 — minor — `tui.analytics.totals.output_note`

- Current: `{generation} generación, {thinking} pensamiento`
- English: `{generation} generation, {thinking} thinking`
- Problem: `pensamiento` is a literal calque of "thinking". Spanish developer
  usage for a model's reasoning tokens is `razonamiento`; `pensamiento` reads
  as "thought", not as the token class the row is counting.
- Suggested: `{generation} generación, {thinking} razonamiento`

### A2 — minor — `tui.analytics.hint.expand.compact`

- Current: `↑↓ · enter expandir · e todo` (28 cells, cap 31)
- English: `↑↓ row · enter expand · e all` (29 cells)
- Problem: the noun is dropped, so the arrows no longer say *what* they move.
  This ladder exists precisely to keep the row cursor (`hint.rows.*` →
  `↑↓ fila`) apart from scrolling (`hint.scroll.*` → `↑↓ desplazar`); a bare
  `↑↓` reintroduces the ambiguity the tiers resolve. It is also internally
  inverted: the *narrower* `hint.expand.keys` tier (cap 25) keeps `fila` and
  drops `expand`, while this wider tier (cap 31) does the opposite. The sidecar
  authorises dropping only `esc back` at this tier.
- Achievable inside the cap: `↑↓ fila · enter abrir · e todo` = 30 cells
  (measured). `abrir` is the settled Spanish UI verb for expand/open, so
  neither the noun nor the action is lost.
- Suggested: `↑↓ fila · enter abrir · e todo`

### A3 — nit — `tui.analytics.search.meta_live`

- Current: `todo el proceso · en vivo` · English: `process-wide · live`
- Problem: `todo el proceso` parses as "the entire process" rather than "across
  the process". The English scope word is a qualifier.
- Suggested: `en todo el proceso · en vivo`

### A4 — nit — `tui.analytics.search.summary_free_paid`

- Current: `{free} de {priced} con precio fueron gratis · {money} pagado`
- English: `{free} of {priced} priced free · {money} paid`
- Problem: the finite verb `fueron` (past) breaks the noun-phrase register of
  every sibling rung — `{free}/{priced} gratis · {money} pagado`,
  `{count} gratis`, `{count} de pago ({money})` — and puts the line in the past
  tense on a live ledger. Meaning is right; the tense is not.
- Suggested: `{free} de {priced} con precio gratis · {money} pagado`

### A5 — nit — `tui.analytics.rate.legend`

- Current: `tasa de decodificación en {shown, number} de {total, number} llamadas`
- English: `decode rate over {shown} of {total} calls`
- Problem: `en` = "in"; the English preposition is "over" (covering this many
  calls). Correct but slightly off the source's frame.
- Suggested: `tasa de decodificación sobre {shown, number} de {total, number} llamadas`

## Mechanical evidence

`checks.py` in this bundle, run as

```sh
PYTHONPATH=$PWD /Users/damian/local-operator/.venv/bin/python \
  i18n/audits/evidence/es/tui.analytics/w6a1/checks.py
```

result: `PASS — all checks clean, every negative control tripped`.

It checks key parity, ICU parse, argument-tuple parity with EN, Spanish CLDR
categories, branch losslessness, real branch selection at six sample counts,
protected machine tokens (`$—`, `↑↓`, `→`, `≈`, backticks, flags, `/command`,
`<angle>`, `[bracket]`), the five declared budgets via `rich.cells.cell_len`,
negation parity, glossary rules and byte-identical-to-EN prose.

Eleven negative controls each break one thing and assert the check catches it;
all eleven caught. Notably: a deleted `many`, an emptied plural branch, a
`many` → `many` mutation of the CATEGORY (an injected `many` is not enough —
it is already present, so the controls target deletion, emptiness and a
forbidden `few` instead), a renamed argument, a rewritten `$—`, a padded
hint, a dropped negation, an `unpriced`→`gratis` swap and an untranslated
prose value.

## Limitations

- LLM semantic judgement, no native-speaker attestation (per programme policy,
  it is not a gate).
- No rendered TUI frames: the host was not put under a heavy run for this
  audit. Every width claim is a `rich.cells.cell_len` measurement of the exact
  string, which is the same function the widget layer measures with, not an
  estimate.
- The target catalogue is **untracked** at the audited revision; the binding is
  to its on-disk sha256 (recorded in `record.json`), not to `HEAD`.
