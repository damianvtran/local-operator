# Audit — es / tui.usage / w6a1

- **Verdict: changes_requested** — 0 blocker, **1 major**, 3 minor, 0 nit
- Revision `b925c2e11be36653dfe88ac78f627bc2531427df`; the es target is **untracked** at that
  revision: es `3cc8ea06555b0bf…`, en `5aeaec2543528961…`, context `bf599f5f161087b8…`
  (full digests in `record.json`).
- Keys read: **21** en / 21 es / 21 context, all with sidecar roles — the sidecar is where this
  block states its own constraints (uncapped allowance must invent no number; eligibility
  unknown must not imply a side; lowercase matches the panel).
- Auditor: independent translation auditor (lopdev team), session `6a2f6fa808a7`; model
  **radient/auto** as actually run (fallback chain in `record.json`).

## What I read

All 21 values against en and the sidecar, then `tui/widgets/usage_panel.py` (the footer tally,
the paid annotations, the title row's counters, panel geometry), and — because the brief asks
whether the reuse of `relay.shell.usage` is real — the **whole committed `relay.shell` es
catalogue**, which is `passed` in the es ledger at the *current* en hash `f53b23560b88`, i.e.
approved, shipped copy for the same readout on the relay surface.

## Mechanical result — clean

`scripts/i18n/check.py` exits 0; section A passes everything. Preserved exactly:

- the two-space indents (`  ·  usage unavailable`, `  no windows reported`);
- the leading comma in `, se reinicia en {value}` and the trailing space in `mostrando ` —
  they compose with ` de ` into `mostrando 1–10 de 21`;
- `…` in `obteniendo…` / `actualizando…`;
- ` — ` with spaces in `error al actualizar — se muestran los últimos datos conocidos`;
- both plurals (`windows_count`, `reported_balance`) with `one`/`many`/`other`, no empty
  branch, `{shown}` kept in every branch (per-branch check), and `1 ventana` / `2 ventanas` /
  `1000000 de ventanas` / `1500000 ventanas` verified through the shipped runtime (the count is
  passed as a raw string, so digits stay unformatted by design in both locales).

## Findings

### US-1 · major · `tui.usage.paid.continue`

> `cuota incluida agotada; continuará con créditos`
> en: `included quota used; credits will continue`

This is the same sentence, about the same allowance, that already ships on the relay surface as
`Asignación incluida agotada; continuará con créditos.` (`relay.shell.usage.continue`, committed
and ledger-`passed`). So the reuse the brief asks about is **not** consistent — and it is not a
one-off: this very catalogue says `Asignación de pago ilimitada` for the paid allowance
(`paid.unlimited`, identical to `relay.shell.usage.unlimited`). The es readout therefore names
one money object two ways depending on which surface and which state the user is looking at.

The rest of the family does match, which is what makes this one visible as drift:

| this file | relay.shell (shipped) | match |
|---|---|---|
| `siguen aplicándose los límites por familia de modelos` | same (capitalised sentence) | yes |
| `Gasto no disponible` | same | yes |
| `Elegibilidad para gasto desconocida` | same | yes |
| `Asignación de pago ilimitada` | same | yes |
| `cuota incluida agotada; …` | `Asignación incluida agotada; …` | **no** |

Recommended: align to the shipped term —
`Asignación incluida agotada; continuará con créditos` — so one concept keeps one name.
If the programme would rather standardise on `cuota` (it is the word `wire.settings` and
`wire.slash.usage` use for quota), then `relay.shell` has to move in the same wave, and that is
a re-audit of a shipped file, not a copy tweak. Either way the split should be recorded in the
es style guide, which currently has no row for *quota* / *allowance*.

### US-2 · minor · `tui.usage.exhausted` and `tui.usage.stale`

> `{value} agotadas` · `{value} obsoletas` (en `{value} exhausted` · `{value} stale`)

`{value}` is a bare count of **windows** (`usage_panel.py` tallies limits, and the footer's own
`windows_count` prints `3 ventanas`). The participle assumes a plural noun, so at one window
the footer reads `1 agotadas` and `1 obsoletas`. en is invariant and cannot show this.

Suggested, count-neutral and shorter: `agotadas: {value}` and `obsoletas: {value}` — or name
the noun (`{value} ventanas agotadas`) if the width allows.

Same row, noted not charged: `sign_in_expired` picks a masculine anchor
(`con inicio de sesión caducado`) while its siblings agree with *ventanas*. Each parses on its
own, but the row mixes referents.

### US-3 · minor · `tui.usage.sign_in_expired` (width)

> `{value} con inicio de sesión caducado` (31 cells rendered; en `2 sign-in expired` = 17)

The mixed-state title is composed as `<count>  · <count>` and **truncates from the right**
(`usage_panel.py:1789–1800`), and the code comment says both counts are kept precisely because
they ask the user for opposite things. At the supported 40×24 floor the panel body is 31 cells:
en's mixed row measures 28 and fits; this one measures **46**, so the stale count is cut and the
row becomes the `…`-truncated single fact the split exists to prevent.

Suggested: `{value} sesión caducada` — the mixed row drops from 46 to **32**, i.e. it fits at
every normal width and misses the 31-cell floor by one cell (so at 40 columns the last
character of the second count clips). If the floor must stay whole, the terse form is
`{value} caducadas` (mixed row 26). Either removes 14 of the 15 cells the current phrase adds.

### US-4 · minor · `tui.usage.paid.family_caps` and `tui.usage.paid.eligibility_unknown` (width)

> `siguen aplicándose los límites por familia de modelos` (53 cells against en's 29)
> `Elegibilidad para gasto desconocida` (35 against en's 23)

Paid annotations render as `TIER_INDENT + note` on one line truncated with an ellipsis
(`usage_panel.py:1039–1047`). At the 31-cell floor body both clip where en does not:
`… por familia de mod…` (the noun is lost) and `Elegibilidad para gasto des…` (the *unknown*
qualifier — the one the sidecar says must not imply a side — is lost). Only the narrow floor is
affected; at 60+ columns both fit.

Suggested if the floor matters: `los límites por familia siguen` (30) and
`elegibilidad de gasto: desconocida` (34) — or record the truncation as a width exception, the
precedent `cli.services_status` set in this locale's guide.

## Verified, not flagged

- `paid.blocked` = `Gasto no disponible`: states unavailability only, keeps the funded balance
  visible, never claims a zero.
- `paid.eligibility_unknown` = `Elegibilidad para gasto desconocida`: does not imply enabled or
  disabled, as its sidecar requires.
- `paid.unlimited` = `Asignación de pago ilimitada`: no invented percentage or denominator.
- `paid.continue` — despite US-1 — still makes the money claim: the included quota is spent and
  billing continues on credits. A control replaces that clause with `se detendrá` and the check
  rejects it, which is the failure mode this string must never have.
- `paid.family_caps` is a fragment, lowercase, exactly the register its sidecar asks for.
- `refresh_failed` = `error al actualizar — se muestran los últimos datos conocidos`: keeps the
  honesty clause that the numbers are the last known ones.
- Title/counters, `no_windows`, `fetching…`/`refreshing…`, `title` = `Uso` (glossary `usage` →
  `uso`): all clean.

## Limitations

LLM judgement, no native attestation. Widths measured with the repo's own ruler
(`rich.cells.cell_len`) against the geometry constants in `usage_panel.py`
(`PANEL_MIN_WIDTH` 32, `PANEL_PADDING_CELLS` 4, `SCROLLBAR_GUTTER_CELLS` 1, `TIER_INDENT` 2);
**no rendered frame was captured** — the clipping above is derived from the truncate call and
the arithmetic, not photographed. No catalogue, sidecar, ledger or allowance was edited.

## Ledger row — prepared, NOT written

Instruction for this audit was *no ledger writes*, so nothing was appended to
`i18n/ledger/es.json` (nor to the aggregate). The row below is what the ledger owner
should append, per the retention rule that the note stays ≤280 characters and points at
the tracked record:

```json
{
  "locale": "es",
  "namespace": "tui.usage",
  "source_sha256": "5aeaec254352896101bf2825ac03d06f74c95e3598cda23536c5fe28507a2f8b",
  "translator": "<from the wave's translation record — not restated here>",
  "auditor": {
    "agent": "i18n-auditor (independent; audited, did not translate)",
    "model": "radient/auto",
    "date": "2026-10-30",
    "result": "changes_requested",
    "findings": 4
  },
  "status": "changes_requested",
  "note": "changes_requested (0 blocker/1 major/3 minor). Evidence i18n/audits/evidence/es/tui.usage/w6a1/record.json; es sha256 3cc8ea06…, en 5aeaec25…; independent auditor, model radient/auto, 2026-10-30. Major: `cuota incluida` vs shipped relay.shell `Asignación incluida`."
}
```

Note length: 265 characters. `translator` is left as a placeholder on purpose:
that block belongs to the wave's translation record, and inventing it here would be a
false provenance claim.
