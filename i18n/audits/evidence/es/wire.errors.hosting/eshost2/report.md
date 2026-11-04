# Audit report — es · `wire.errors.hosting` (audit `es-wh2-f1`)

- **Verdict: PASSED** — 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT.
- Key: `wire.errors.hosting.not_configured` (1 of 1; the whole file group)
- EN: `Hosting platform is not configured.`
- es: `La plataforma de alojamiento no está configurada.`
- Auditor: `i18n-auditor`, ES reader2, session `425221b9b2c6`, fresh context.
  Model requested `radient/auto`; served model **UNKNOWN** (not exposed in-session).
  Not the translator (804764f54d82) and not the code author (94dcef624dec).
  No prior auditor conclusions were read; the earlier ES reader2 record is
  preserved untouched and is not relied on here.
- Date: 2026-11-03.

## Inputs (hashes recomputed here, not copied)

| Input | SHA-256 | Match |
|---|---|---|
| en `wire.errors.hosting.json` | `458d8f8b…de964b63` | expected ✓ |
| context sidecar | `ed0ae50f…0d7063cf9` | expected ✓ |
| es target | `8de6f07a…895ff3c5cf` | expected ✓ |

The English bundle and the sidecar are byte-identical to
`local_operator/i18n/catalogues/en/wire.errors.hosting{,.context}.json` in the
core worktree, so the audited English is the repository's source of record.
No `i18n/catalogues/es/wire.errors.hosting.json` exists in that checkout —
consistent with the brief's display-key-only, uncommitted, unshipped status.

## 1. Mechanical — PASS

Checked by hand and with stdlib `json`/`hashlib`/`unicodedata` (no processes:
the source-only exception for this pass bars the repo validator; that is stated
plainly rather than implied).

- Valid JSON in all three files; key sets identical (1 = 1), none added or dropped.
- **No placeholders** and **no ICU structure**: the key takes no arguments, so
  CLDR categories (`one`/`many`/`other` for es) are **N/A by construction** —
  there is no `{` selector to validate.
- No markup, Rich tags, backticks, code spans, commands or flags on either side.
- Both strings NFC; no `Cf`/`Cc`/`Co`/`Cs` format or control characters; no
  trailing whitespace, no double spaces, no BOM. `á` is precomposed U+00E1.
- Terminal period present; initial capital present; identical sentence shape.

## 2. Accuracy — PASS

Meaning is preserved exactly: the hosting platform is in an unconfigured state.
No omission, no addition, no polarity or scope change. `no está configurada`
renders the *state* (a preflight condition), not an event (`no se ha
configurado`), which is the right reading for a diagnostic cause clause.
Nothing here is destructive, permission-granting or safety-critical, and no
remedy is introduced — the surrounding refusal still supplies it, so the
"never restates the remedy" rule holds.

## 3. Idiom and register — PASS

A complete, natural Spanish sentence with correct feminine agreement
(`plataforma` → `configurada`). Neutral Spanish: no region-marked vocabulary.
No address form, so the `tú` decision is untouched. No `!`, no Title Case, no
calque. The colón-plus-capital construction is exactly what the sidecar asks for
(`keep its terminal period and its capitalization`).

## 4. Terminology — PASS

`plataforma de alojamiento` is the shipped es rendering of *hosting platform* in
the core catalogue — `cli.global_help.hosting`,
`cli.global_help.propagated_hosting`, `cli.exec_worker_help.hosting`,
`cli.credential_help.credential` — and `alojamiento` is the only word es uses
for the concept (no kept-English `hosting` variant exists in any es file).
The glossary has no *hosting* row and the es style guide records no call, so
catalogue precedent governs, and the target follows it word for word. No product
name, flag, path or identifier appears, so nothing there could drift.

## 5. Fit — NOT RUN

The sidecar declares no `maxCells`/`maxChars` (its fields are `note`, `role`,
`tone`), so there is no cap to test against. Measured anyway: 49 cells vs EN 35
(+40% in characters, above the guide's typical 15–25% band), all narrow glyphs
(the only non-ASCII character is `á`, width 1). No overflow is claimed and none
is predicted — the consuming refusal frame owns that budget.

## 5b. Native render / pseudo-localization — NOT RUN

Neither was dispatched for this pass; no coverage is implied.

## 6. RTL — N/A

`es` is LTR.

## Observations (deliberately not scored as findings)

1. Spanish typography would normally lowercase after a colon (RAE). The source
   contract overrides it explicitly. Recorded so nobody "fixes" the capital
   later; scoring it would be inventing a defect against the contract.
2. The +40% growth is worth the frame's owner knowing, but with no declared cap
   and no executed fit check it is not a MAJOR — asserting overflow without a
   code path would be exactly the padding this audit is meant to avoid.

## Provenance and limitations

- Source-only exception honoured: no processes, no project imports, no tests, no
  repo validator, no native/browser run, no install, no repo edit, no ledger
  write. The ledger row is therefore **not** recorded by me; this scratch
   record + report are the artifact of record pending attachment.
- `audit.log` reproduces the values printed by the actual check run; `report.md`
   and `record.json` are analysis notes, not an execution log.
- This is a single-key replacement audit. It does not claim the full programme
  (all seven locales) is complete — that remains open.
