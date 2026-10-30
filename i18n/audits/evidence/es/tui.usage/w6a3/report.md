# Audit — es / tui.usage / w6a3 (round 3, narrow convergence)

- **Verdict: passed** — 0 blocker, 0 major. Round 2's open minor **F-U1
  closes**. One minor and two nits remain in the *style-guide width record*
  (SG-1/2/3), not in the catalogue copy; no new catalogue finding.
- Round 2: `i18n/audits/evidence/es/tui.usage/w6a2/` (`passed`; F-U1 left
  `{value} caducadas` rendering `1 caducadas`). Round 1: `w6a1/`.
- Revision `b925c2e11be36653dfe88ac78f627bc2531427df`; branch
  `feat/i18n-es-tui`. The es target is **UNTRACKED** at this revision, so the
  on-disk sha256 binds it: es `52166a416bf6d49a…`, en `5aeaec2543528961…`,
  context `bf599f5f161087b8…` (full digests in `record.json`).
- **Delta: exactly one value changed.** Restoring round 2's `{value}
  caducadas` reproduces the round-2 file sha256 `e2ab2abd3d581eb5…`
  byte-for-byte; the current file is `52166a416bf6d49a…`. No other value in
  this file changed (probe F1/F2; `audit.log` §3).
- Auditor: independent translation auditor (lopdev team), session
  `c0b953a7c12a`; model **radient/auto** as actually run. The header named
  `anthropic/claude-sonnet-5-5`; provider quota failures moved the session
  through `openai/gpt-6-astra` and `kimi/k3.1-pro` to `radient/auto` before any
  audit work (`active_model_route` records in this run's transcript; chain in
  `record.json`). I authored none of this copy and repaired none of it.

## What changed

`tui.usage.sign_in_expired`

| | |
|---|---|
| en | `{value} sign-in expired` |
| es, round 2 | `{value} caducadas` |
| es, now | `caducadas: {value}` |

Sidecar: “Windows whose credential has expired; raw digit.” The count is a
bare count of windows, so a feminine plural participle after it was wrong at
one (`1 caducadas`). The label form is count-neutral and is the shape both
siblings already use: `agotadas: {value}` and `obsoletas: {value}`.

## F-U1 closes

- **Shape:** all three counters are now `label: {value}`; no count-first
  counter remains in this title row.
- **Renders** through the shipped runtime: `caducadas: 1`, `caducadas: 2`,
  `caducadas: 12`, `caducadas: 100`, `caducadas: 1000`, `caducadas: 1000000`.
  There is no ICU plural in this key; `{value}` is a simple argument, exactly
  as in en (`message_arguments` both `(('value', 'simple'),)`).
- **At one** the string reads as a status label, not as a plural agreement
  error — the same way `agotadas: 1` and `obsoletas: 1` read. The noun is the
  trade round 1 accepted; the block-level note is where the full phrase lives.
  On current bytes that note is a hard-coded English string (`sign-in expired
  — /login {provider}`, `usage_panel.py:635`), not an extracted es key — see
  "Not addressed" below. The counter itself is still the count-neutral form
  round 2 asked for.

## Fit — re-measured with `rich.cells.cell_len`

Geometry from `usage_panel.py`: `PANEL_MIN_WIDTH 32`, `PANEL_PADDING_CELLS 4`,
`SCROLLBAR_GUTTER_CELLS 1`, `PANEL_WIDTH_MARGIN 4`. At a 40-column terminal
the card is 36, content 32, **body 31**; at 60 columns content 52, body 51.

| count | counter | counters segment (`caducadas: N  ·  obsoletas: 1`) | full title row | en full row |
|---|---|---|---|---|
| 1 | 12 | **28** | 43 | 45 |
| 2 | 12 | **28** | 43 | 45 |
| 12 | 13 | **29** | 44 | 46 |
| 100 | 14 | **30** | 45 | 47 |
| 1000 | 15 | **31** | 46 | 48 |
| 1,000,000 | 18 | 34 | 49 | 51 |

The **counters segment** — the mixed-state block round 2 measured — fits the
31-cell floor body through 1,000 expired windows (28-31 cells). At 1,000,000
it is 34, but en is 34 too; that count is outside any realistic account set
and is reported rather than hidden.

The **full title row** is 43 cells, not 31: it is chrome (title + age +
counters), and at the 40-column floor the chrome is 32 cells. It therefore
truncates at 40 columns in **both** locales — 43 es / 45 en against 32 — but
it fits the 52-cell content at 60 columns. es is 2 cells shorter than en at
every tested count, and at 40 columns the truncation preserves the actionable
half in es (`Uso  2m ago  · caducadas: 2  · …`) while en loses the tail of
`sign-in expired` itself (`Usage  2m ago  · 2 sign-in expi…`). This is the
design's existing limit, not a translation regression.

Round 2 projected 29/44 for this fix; the measured values are **28/43**. The
difference is one cell: the renderer's separator is `"  · "` (two spaces
before the dot, one after), not `"  ·  "`. Re-measured from the source, not
carried over.

## Mechanical result — clean

- `python3 scripts/i18n/check.py --catalogues-only` → `0 problem(s)`, exit 0.
- Full `python3 scripts/i18n/check.py` → exit 0: 191 files, 2,348 literals at
  or below baseline, catalogues consistent, 51 advisories (report-only).
- Probe: parity 21 en / 21 es / 21 context; every value parses in the shipped
  ICU subset; placeholder tuples identical on all 21 keys; both plurals
  (`windows_count`, `reported_balance`) keep one/many/other with no empty
  branch and still render `1 ventana`, `2 ventanas`, `1000000 de ventanas`,
  `1 saldo informado`, `2 saldos informados`; the old `{value} caducadas`
  string appears nowhere in the file.

## Style-guide record — measured, not trusted

`i18n/style-guides/es.md:123-139` now carries the two wave exceptions. The
decision it records is the one actually taken: keep the paid annotations long
and relay-matched (the strings are the shipped relay wording, adapted to the
TUI's sentence case/period), and accept that `keys_bare` is 14 cells against a
12-cell cap because English itself is 13 against that cap.

The numbers:

| string | es raw | en raw | es composed (`TIER_INDENT` + note) | en composed | body needed at 60 cols (51) |
|---|---|---|---|---|---|
| `paid.family_caps` | 53 | 29 | **55** | 31 | clips; needs 64 columns |
| `paid.eligibility_unknown` | 35 | **25** | 37 | 27 | fits |
| `paid.continue` | 52 | 42 | **54** | 44 | clips; needs 63 columns |
| `tui.analytics.hint.expand.keys_bare` | **14** | **13** | — | — | cap 12; footer crops |

- **SG-1 · minor · style guide says “at 60+ columns all three fit”; they do
  not.** At 60 columns the body is 51: `family_caps` is 55 and `paid.continue`
  54, so both still clip; only `eligibility_unknown` (37) fits. At 63 columns
  `paid.continue` fits, and at **64** all three fit. The same paragraph
  already says `paid.continue` needs 63 and the exception itself is the reason
  the clipping is accepted, so the **decision stands** — the threshold
  sentence is wrong and contradicts its own numbers. This is a record error,
  not a catalogue error, and it is minor for that reason.
- **SG-2 · nit · en comparator for `eligibility_unknown` is 25, not 23.**
  Measured with both `rich.cells.cell_len` and `len`: 25. es 35 is correct;
  the exception is unaffected, but the guide understates en by two cells.
- **SG-3 · nit · “both strings are byte-identical” is true only for
  `eligibility_unknown`.** relay: `Elegibilidad para gasto desconocida`
  (identical). `family_caps`: relay `Siguen aplicándose los límites por
  familia de modelos.` vs TUI `siguen aplicándose los límites por familia de
  modelos` — same wording, but initial capital and final period differ. The
  relay-matching decision is sound; “byte-identical” overstates it.
- **`keys_bare` verified:** es 14 / en 13 / `maxCells` 12, exactly as the
  guide records; the footer selects the last rung and paints it with
  `Text(no_wrap=True, overflow="crop")` (`analytics_panel.py:3091-3093`), so
  it crops rather than wraps. No finding.

## Negative controls

- **E1** restores `{value} agotadas` in `exhausted` → C2 (all three counters
  label-first) rejects it. Caught.
- **E2** restores `{value} caducadas` in `sign_in_expired` → C1/C2 reject it.
  Caught.
- **E3** restores `caducadas: {count}` → the placeholder signature differs
  from en. Caught.

## Findings

- **F-U1 · minor · CLOSED** — `caducadas: {value}` is count-neutral,
  identical in shape to its siblings, renders correctly at 1/2/multi-digit,
  and the counters fit the 31-cell floor body through 1,000.
- **SG-1 · minor** — the style-guide threshold sentence (“at 60+ columns all
  three fit”) is false; measured 63/64. The accepted exception stands.
- **SG-2 · nit** — en `eligibility_unknown` is 25, not 23.
- **SG-3 · nit** — “both strings are byte-identical” is exact for
  `eligibility_unknown`, not for `family_caps` (case + final period).
- No new catalogue finding. The other 20 values were not re-audited: round 2
  passed them, and this round re-checks the file's mechanical integrity
  around the one edit.

## Not addressed (pre-existing, out of scope)

- **The block-level sign-in note is English-only.** Round 2's F-U1 rationale
  said the body blocks carry `inicio de sesión caducado`; the es catalogue has
  no such string (`grep -r caducad` over `catalogues/es` finds only this key
  and an unrelated `relay.asks` line). `_account_status_note` returns
  `f"sign-in expired — /login {report.provider}"` directly from source
  (`usage_panel.py:635`). This is a pre-existing extraction gap in the panel,
  not a w6a3 delta; the Spanish user sees the English phrase in the block. It
  does not change the verdict on `caducadas: {value}` — the label is
  count-neutral and the block note still names the sign-in — but it should be
  extracted in the panel's own slice rather than claimed as already
  translated.

## Limitations

LLM judgement; no native-speaker attestation is claimed. No rendered TUI
frame was captured: widths are computed with `rich.cells.cell_len` against
the geometry the source documents, and truncation is computed with
`rich.text.Text.truncate(..., overflow="ellipsis")`, not photographed. No
catalogue, sidecar, ledger, allowance, style guide or generated file was
written; only this evidence directory.

## Ledger row — prepared, NOT written

This round forbids ledger writes (no git writes; no catalogue/sidecar/ledger/
allowance/style-guide edits), so `i18n/ledger/es.json` was not touched. Row
for the ledger owner:

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
  "note": "w6a3 r3 passed (F-U1 closed; 0 blocker/major; SG-1 minor, SG-2/3 nits, style guide). Evidence i18n/audits/evidence/es/tui.usage/w6a3/record.json; es 52166a41…, en 5aeaec25…; independent, radient/auto, 2026-10-30. `caducadas: {value}` renders 1/2/12; counters 28–31 (1–1000)."
}
```

Note-length check: the `note` is 274 characters, inside the 280-character
budget.
