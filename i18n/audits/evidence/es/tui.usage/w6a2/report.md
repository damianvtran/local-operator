# Audit — es / tui.usage / w6a2 (round 2)

- **Verdict: passed** — 0 blocker, **0 major**; 2 minor and 1 nit remain and are recorded below
  (both minors are side effects of the remediation, not regressions of meaning).
- Round 1: `i18n/audits/evidence/es/tui.usage/w6a1/` (`changes_requested`, 0 blocker, 1 major,
  3 minor, 0 nit).
- Revision `b925c2e11be36653dfe88ac78f627bc2531427df`; es target **untracked**, so the on-disk
  sha256 binds it: es `e2ab2abd3d581eb5…`, en `5aeaec2543528961…`, context `bf599f5f161087b8…`
  (full digests in `record.json`).
- Branch: `feat/i18n-es-tui` — the brief named `tui-wip`; `git rev-parse --abbrev-ref HEAD` says
  otherwise, and the log records what is actually checked out.
- Auditor: independent translation auditor (lopdev team), session `7fa3a39af5d9`; model
  **radient/auto** as actually run (fallback chain in `record.json`). Round 1 was a different
  session (`6a2f6fa808a7`), so this pass is independent of it as well as of the translator.

## What I re-read

All 21 values, the sidecar for each, and the two render sites in `tui/widgets/usage_panel.py`
(`UsageStats.describe()` for the footer tally, `_title_row()` for the mixed-state title, and
`build_usage_body`'s paid-note rows), then the whole committed `relay.shell` es catalogue for the
continuation sentence it now cites.

## Mechanical result — clean

`scripts/i18n/check.py` exits 0; probe section A passes on the current bytes: parity 21/21, ICU
argument tuples identical, both plurals (`windows_count`, `reported_balance`) carry
one/many/other with no empty branch (rendered 1 / 2 / 1000000: `1 ventana`, `7 de ventanas`,
`2 saldos informados`), the code span set is empty on both sides, and the layout carriers survive
byte-exactly: `  ·  ` separator, two-space indents (`  ·  uso no disponible`,
`  sin ventanas informadas`), the leading comma in `, se reinicia en {value}`, the trailing space
in `mostrando `, `…` in both progress words, and ` — ` in `refresh_failed`.

## Fit — re-measured with `rich.cells.cell_len`, not taken on trust

Geometry read from the source: `PANEL_MIN_WIDTH 32`, `PANEL_PADDING_CELLS 4`,
`SCROLLBAR_GUTTER_CELLS 1`, `PANEL_WIDTH_MARGIN 4` → at a 40-column terminal the panel is 36 wide,
the **body** composes to **31** cells and the chrome (title/rule/footer) to 32.

**Mixed-state title row** (`Usage` + age + `sign-in expired` + `stale`), expired 2 / stale 1:

| | counters segment | composed row incl. title and `  2m ago` |
|---|---|---|
| en | 28 | 45 |
| es (current) | **27** | **42** |
| es, with the `caducadas: {value}` form proposed below | 29 | 44 |
| 31-cell floor / 32-cell chrome | 27 ≤ 31 ✓ | — |

The counters now fit the floor body and the composed row is **3 cells shorter than en**, so es
never truncates a fact that en shows. The whole row still truncates at 40 columns in *both*
locales (45 and 42 against 32) — that is the design's existing limit, not a translation defect,
and es is on the better side of it.

**Paid annotation rows** compose as `TIER_INDENT + note` (2 cells of indent) and are ellipsised by
design; the defect to catch is es clipping where en fits:

| key | en composed | es composed | body at 60 cols = 51 |
|---|---|---|---|
| `paid.continue` | 44 | **54** | **es clips (→ `…continuará con crédit…`), en fits** |
| `paid.family_caps` | 31 | 55 | es clips (accepted exception, US-4) |
| `paid.eligibility_unknown` | 27 | 37 | es clips (accepted exception, US-4) |
| `paid.unlimited` | 26 | 30 | fits |
| `paid.blocked` | 19 | 21 | fits |

## Round-1 findings

### US-1 · major · CLOSED · `paid.continue`

> `Asignación incluida agotada; continuará con créditos`

The allowance is now named the way the shipped relay surface names it
(`relay.shell.usage.continue` = `Asignación incluida agotada; continuará con créditos.`) — the
money-object split the major was about is gone, and the es guide recorded the `cuota` vs
`asignación` rule on the strength of it. The only difference from relay is the final period, which
the TUI row correctly omits because en has none. **The sentence still makes the money claim**
(included allowance spent, billing continues on credits); a control that swaps that clause for
`se detendrá` is caught by the probe.

### US-2 · minor · CLOSED · `exhausted` / `stale`

> `agotadas: {value}` · `obsoletas: {value}`

Count-neutral: `1 agotadas` and `1 obsoletas` are gone, so the title reads correctly at one window.
See the two follow-ups below for what the label form costs in the second render site.

### US-3 · minor · **partially closed** · `sign_in_expired`

> `{value} caducadas` (was `{value} con inicio de sesión caducado`)

**Width: closed.** The mixed row drops from 46 to 27 cells and now fits the 31-cell floor body —
re-measured, not inferred.

**Agreement: not closed.** The finding that this fix was meant to answer (US-2) was precisely that
`{value}` is a bare count of *windows*, so a feminine plural participle is wrong at one:
`1 caducadas` renders today. The same defect now sits alone in a row whose two neighbours were just
made count-neutral, and the row mixes shapes — `2 caducadas  ·  obsoletas: 1` — one count-first
item beside one label:count item.

- **F-U1 · minor · `tui.usage.sign_in_expired`**
  Suggested: `caducadas: {value}` — count-neutral, the same shape as its two siblings, and
  **shorter than what it replaces on the row**: mixed row `caducadas: 2  ·  obsoletas: 1` = **29**
  cells against the 31-cell floor body (current 27 + the agreement error; the suggested 29 fits).
  It costs the words `inicio de sesión`, which round 1 had already accepted as the width trade —
  the body's own blocks carry `inicio de sesión caducado`.

### US-4 · minor · DISPOSITION ACCEPTED (not applied) · `paid.family_caps` / `paid.eligibility_unknown`

Left relay-matched and to be recorded as a width exception. I accept the disposition: the wording
is the shipped relay copy for the same two facts, the meaning is exact, and re-wording to shave
cells would give one concept two names across surfaces — the very defect US-1 was about.

Two things the manager should know:

1. The exception is **not yet recorded**. `i18n/style-guides/es.md` still carries only the
   `cli.services_status` width exception (line 118); no paid-annotation row exists. Until it is
   written, `family_caps` (55) and `eligibility_unknown` (37) are still *unrecorded* overruns of
   the 60-column body (51) — the precedent the guide sets for `cli.services_status` is what makes
   an accepted overrun legitimate.
2. Round 1 said "at 60+ columns both fit" — that was imprecise: `family_caps` composes to 55
   cells, so it needs a **64**-column terminal to fit (body = columns − 9, i.e. body ≥ 55).
   Corrected here; the accepted disposition is unaffected.

## New findings from this remediation

### F-U2 · minor · `tui.usage.paid.continue` (width regression)

> `Asignación incluida agotada; continuará con créditos`

Aligning to relay cost 5 cells: the note was 49 composed (47 + the 2-cell `TIER_INDENT`), it
is now **54**, while en is 44. With `body = columns − 9` (for `columns ≥ 36`):

- before: es fit from a **58**-column terminal;
- after: es needs a **63**-column terminal;
- en fits from **53**.

In the **58–62** column band the es row now clips where it previously fit, and at 60 columns the
body is 51, so the row truncates to `Asignación incluida agotada; continuará con crédit…` — the
object of the money claim is cut, on a row where en shows it whole. At 80 columns (body 71)
nothing clips.

Two acceptable dispositions; the first is what the sibling keys already rely on:

1. **Extend the recorded width exception** (US-4) to cover this key too — same row, same surface,
   same class, and it keeps one name for one money object. This is my recommendation.
2. Shorten the tail if the wave wants no exception here: `Asignación incluida agotada; sigue con
   créditos` = **49** composed (fits from 58 columns) or `…; continúa con créditos` = **52** (fits
   from 61). Neither reaches en's 53-column threshold while keeping the shipped term.

### F-U3 · nit · `tui.usage.exhausted` in the footer tally

> Footer: `5 ventanas · agotadas: 1 · 1 cerca del límite · 2 saldos informados`

`exhausted` has a **second render site**: `UsageStats.describe()` joins its counters with ` · `,
and every sibling there is count-first (`{value} cerca del límite`, `{value} saldos informados`,
`{value} sin informar`). The new label form sits among them as the odd one out (1 cell longer, and
no layout risk: the footer drops the whole tally segment rather than truncating it, and it only
appears on wide panels — at 80 columns the stats segment is dropped once the hint row is present).

The label form is still the right fix for the agreement defect, and making the tally uniform would
mean reshaping four more keys. Recommendation: **accept and leave it**; recorded so it is not
rediscovered as a defect.

## Neighbours checked because the fix could have touched them

- `near_limit`, `not_reported`, `reported_balance`: unchanged, still count-first, still correct at
  one and many.
- `windows_count`: unchanged plural, `1 ventana` / `2 ventanas` / `7 de ventanas` all render.
- `blocked`, `unlimited`, `eligibility_unknown`, `family_caps`: re-read; the money claims are
  unchanged (`Gasto no disponible` still claims no zero; `Asignación de pago ilimitada` still
  invents no denominator).
- `refresh_failed`, `fetching…`, `refreshing…`, `title` = `Uso`: unchanged and clean.

## Limitations

LLM judgement, no native-speaker attestation, no rendered frame captured: the row widths are
computed with `rich.cells.cell_len` against the geometry constants in `usage_panel.py`, and the
truncation points are computed from `Text.truncate(…, overflow="ellipsis")`, not photographed. No
catalogue, sidecar, ledger or allowance was edited.

## Ledger row — prepared, NOT written

This round forbids ledger writes, so `i18n/ledger/es.json` was not touched. Row for the owner:

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
    "result": "passed",
    "findings": 3
  },
  "status": "passed",
  "note": "r2 passed (0 blocker/major; 2 minor, 1 nit). Evidence i18n/audits/evidence/es/tui.usage/w6a2/record.json; es sha256 e2ab2abd…, en 5aeaec25…; independent, radient/auto, 2026-10-30. US-1/US-2 closed; sign_in_expired still `1 caducadas`; paid.continue clips at 58-62 cols."
}
```

Note-length check: the `note` above is exactly 269 characters, inside the 280-character budget that binds a ledger row.
