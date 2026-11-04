# Independent audit — `tui.analytics` (es) — round 2

**Namespace:** `tui.analytics` · 98 keys · English source 98 keys · context sidecar 98 entries
**Audit id:** `w6a2` · round 2 · 2026-10-30
**Auditor:** independent (authored none of this copy, applied none of the remediation)
**Verdict:** `passed` — 0 blocker, 0 major, 0 minor, 5 nit (3 deferred from round 1, 2 recorded now)

Round 1 (`w6a1`) passed this namespace with 2 minor and 3 nit. The
implementer applied exactly the two minor findings:

| finding | key | round-1 suggested | now | cells |
|---|---|---|---|---|
| A1 | `totals.output_note` | `{generation} generación, {thinking} razonamiento` | applied verbatim | 36 at sample figures (A7) |
| A2 | `hint.expand.compact` | `↑↓ fila · enter abrir · e todo` | applied verbatim | 30 / cap 31 |

## What I did

I re-read the style guide and glossary, then the English source and its sidecar,
then **all 98 Spanish values again** — not only the two that changed, because a
verdict on the current bytes is a verdict on the whole file. I re-ran every
mechanical family from round 1 (key parity, ICU parse, argument tuples, Spanish
CLDR categories, real renders at six counts, protected machine tokens, the five
declared `maxCells` budgets, polarity, glossary rules, semantic assertions) and
added the round-2 families: tier order, the compact counterfactual, and a
per-finding disposition check. Then I re-measured every declared budget with
`rich.cells.cell_len` rather than trusting the implementer's arithmetic.

## A1 — closed

`{generation} generación, {thinking} razonamiento`

The word is right, and it is right *for this reason*: the row counts reasoning
**tokens**, not the model's streamed thought. Spanish shipped copy already uses
`razonamiento` for exactly that concept — `cli.exec_help` ("esfuerzo de
razonamiento"), `relay.chrome` ("Esfuerzo de razonamiento"), `wire.settings`
("Esfuerzo de razonamiento predeterminado", "razonamiento profundo/ligero/
moderado", "ocultar el razonamiento"), `tui.notices` ("eco de razonamiento"),
`wire.errors.desktop` ("niveles de esfuerzo de razonamiento"). The surviving
`pensamiento` occurrences are a *different* referent: `wire.settings` uses it
for the model's private thought text that streams on screen. The change therefore
resolves a split in the correct direction rather than creating one.

Nothing else moved: the shape (`{value} noun, {value} noun`) matches its sibling
`totals.billed_note` (`{context} entrada · {output} salida`), both placeholders
are untouched and stay pre-formatted, and no agreement is at stake (both nouns
are masculine singular, both arguments are numbers).

Fit: no `maxCells` is declared for this key, so I measured the row the card
actually composes. `ReportView` CROPS an overrunning line rather than wrapping
it, and `kv()` composes `"  " + name:<22 + value:<11 + "  " + note`, i.e. **37
cells plus the note**. The card's body measures **66 cells at 80×24** (rig: 90%
of the 78-cell screen content, less `padding: 1 2`). So at a session-scale figure
the Output row is 69 cells in English (`12.3k generation, 12.3k thinking`) and
73 in Spanish (`12.3k generación, 12.3k razonamiento`): **the English note is
already cropped by 3 cells there, and the Spanish by 7**. The Spanish costs one
cell more than the round-1 wording (`razonamiento` 12 vs `pensamiento` 11) and
four more than the English. There is no shorter faithful Spanish word for this
token class, and the crop is source-side and pre-existing, so this is recorded
as A7 (a fit note against the card, not against this copy), not raised against
the translator.

## A2 — closed

`↑↓ fila · enter abrir · e todo` = **30 cells**, cap **31** (`hint.expand.compact`
sidecar note: "drops 'esc back' but keeps both expand keys named").

The noun is back, so the tier no longer contradicts the ladder's whole purpose —
telling the row cursor (`↑↓ fila`, shared with `hint.rows.*`) apart from
scrolling (`↑↓ desplazar`, `hint.scroll.*`). It also repairs the inversion
round 1 flagged: the narrower `hint.expand.keys` tier (24 cells, cap 25) keeps
`fila`, and now so does this wider one.

### The question the manager asked: did the ladder's verbs become misleading?

**No — and here is the measurement that decides it.** `expandir` cannot fit this
tier: `↑↓ fila · enter expandir · e todo` measures **33 cells against a 31-cell
cap**, so the two options at compact are "a different verb" or "no verb". The
chosen verb is not a different meaning: `abrir` is the shipped Spanish action
label for "enter opens this row's content" (`tui.import.hint_open`,
`tui.notifications`, `tui.review`, `tui.library` all use it), and this screen has
no second open-like action a reader could confuse it with — the only other keys
are `e`, `t` and `esc`. The ladder already varies its vocabulary by tier for the
same reason (`esc` vs `esc volver`), the key it names is byte-identically
`enter` in every tier, and a reader sees one tier at a time. Recorded as A6
below as a consistency note, not a defect.

The consequence of 28 → 30 cells is real but bounded: a box narrower than 30
cells now falls to the `keys` tier (`↑↓ fila · enter · e todo`) instead of
painting the compact tier. That is the documented degradation ("the first tier
that FITS"), it loses the action name only in the 28–29-cell band, and at the
default 80×24 the card's body is 66 cells (measured), so the widest tier that
fits — `full_metric`, 56 cells — is the one that paints. No wording that keeps
both key names and a verb is shorter than 30.

## Deferred round-1 nits — dispositions, not repeats

- **A3 `search.meta_live`** (`todo el proceso · en vivo`) — not applied.
  Judged acceptable: as section meta it names the same scope the English does
  ("process-wide"), and the semantic assertion for the process scope passes.
  Left as the owner's deferred nit.
- **A4 `search.summary_free_paid`** (`… fueron gratis · {money} pagado`) — not
  applied. Meaning is correct (the free/paid split and the denominator both
  survive); only the finite past-tense verb is off-register. A nit by
  construction, so not applying it is a disposition I accept.
- **A5 `rate.legend`** (`tasa de decodificación en … llamadas`) — not applied.
  `en` vs `sobre` is a preposition preference; the coverage claim reads
  correctly. Accepted as deferred.

## A6 — nit (new, recorded, no change requested)

- **`hint.expand.compact` vs `hint.expand.full` / `full_metric`:** the same
  `enter` key is labelled `abrir` at the compact tier and `expandir` at the two
  wider tiers. Forced by the cap (33 > 31, measured above), not misleading (see
  the section above), but a future wave that widens the compact cap should
  unify the ladder on `expandir`.

## A7 — nit (new, fit note against the CARD, not against this copy)

- **`totals.output_note` in the composed Output row.** At 80×24 the analytics
  card's body is 66 cells; `kv()` spends a fixed 37 cells before the note, so
  the note has 29 cells and a session-scale note needs 32 in English, 36 in
  Spanish. The tail is cropped in **both** locales (measured: en 69, es 73).
  Consequence for this audit: the Spanish crop is 4 cells longer, but it is not
  caused by the A1 word choice (that cost 1 cell) and no shorter faithful
  wording exists. The fix belongs to the row's column budget (name/value cells)
  or to the note's own line, not to this catalogue. Recorded; no translation
  change requested.

## Mechanical evidence

`checks.py` in this bundle, run as

```sh
env PYTHONPATH=$PWD /Users/damian/local-operator/.venv/bin/python \
  i18n/audits/evidence/es/tui.analytics/w6a2/checks.py
```

result: **`PASS — all checks clean, every negative control tripped`** (0 blocking
failures; 15 negative controls, including the two round-2 ones).

Widths, re-measured on the current bytes (`rich.cells.cell_len`):

| tier | es | en | cap | state |
|---|---|---|---|---|
| `hint.expand.full_metric` | 56 | 56 | 57 | within |
| `hint.expand.full` | 39 | 40 | 42 | within |
| `hint.expand.compact` | 30 | 29 | 31 | within |
| `hint.expand.keys` | 24 | 22 | 25 | within |
| `hint.expand.keys_bare` | 14 | 13 | 12 | cap already exceeded by EN (13) — unattainable, not a defect |

Ladder order widest-first is strictly decreasing (56 > 39 > 30 > 24 > 14), which
is the property the widget's "first tier that fits" selection depends on; the
round-2 control that inverts it trips `[ladder]`.

The repository validator (`python3 scripts/i18n/check.py --catalogues-only`)
reports **0 problem(s)** on the current tree.

`audit.log` is the execution log for every command above; `checks.py` is the
script that produced it.

## Provenance limit stated plainly

The production resolver serves English for `es` until this namespace has a
passing ledger row (`i18n.resolve.resolve_language` filters to the shipped set).
At audit time `local_operator/i18n/data/shipped.json` in this working tree
listed locales WITHOUT `es` — a concurrent working-tree edit by another session
in this shared checkout, not mine — so the resolver answered `en` and the card
could not be painted in Spanish through the live resolver. The en row *was* painted
by the real widget (see the `tui.import` bundle's rig block, EN control), and
this bundle's assembler reproduces it — the same `HintButton._build` seam rule
applies here. No rendered Spanish frame is claimed.

## Ledger data (for the coordinator — I do not edit the ledger)

- locale `es` · namespace `tui.analytics` · status **passed**
- `source_sha256` (en) `aa5221066aa84eedf0f924628d51bc8f08ccf5f26b5075638b9d40062f7b617e`
- target sha256 `0496f1b123ab06e0aee3c991f8551c0013f77b3b1fff77230b876ec4e9143657`
- auditor `i18n-auditor` · model actually running `radient/auto` (fallback chain
  disclosed in `record.json`) · 2026-10-30 · round 2
- unresolved blocker/major: **0**
