# Spanish audit — wire.settings.account_policy — round 3 (w1a4, delta only)

- **Verdict: passed** — 0 blocker, 0 major, 0 minor, 2 nit
- Audit id `w1a4`, round 3, 2026-10-29; supersedes `w1a3` for the **delta only**
- Delta keys: `wire.settings.account_policy.label`, `wire.settings.account_policy.legacy.help`
- Auditor: translation-auditor (lopdev team delegation es-w1-auditor-r2), session 3da6fd7a1b13, model **radient/auto** (self-reported; fallback disclosed)
- Bound bytes: `f5824ce686fde85dea395fbca9d91d7354bea0cd8a4a4bbe545dc9b1ed24d04c` (untracked; verified by this round)
- Keys: 14 en / 14 es

## Prior rounds

- round 1 w1a1 — first independent audit of this namespace
- round 2 w1a3 — final-byte audit; raised the findings this delta judges
- round 3 w1a4 — this record; delta only, on bytes that moved after round 2

Wave context: wire.console w1a3 amendment A1 (NOT this namespace): that amendment established the same-code-control rule this delta applies — a width finding must first be run through the identical budget on the English source before it is called a translation defect.

## What this round did and did not do

It re-judged the four changed values and diffed everything else against what w1a3 read. It did not re-read or re-judge the unchanged keys.

## Delta judgement

### `label` — PASS

PASS. 'Política de cuentas' (19 cells) keeps both nouns the English has - policy and accounts - and drops only 'selection', which the row's own enum values carry. This was es-w1-C06's proposal; the row sits in the 'failover' section with no account cue in the section title, so naming accounts on the label matters.

### `legacy.help` — PASS

PASS, and this is the part that makes the label change safe. EN names the setting three times inside this help; the Spanish now says 'política de cuentas' in all three, which is exactly the label, so a reader following the instruction can find the control. The three policy names quoted in the help still match their own labels byte-for-byte. true/false remain literal. 339 cells, no declared budget; the ratio to English is 1.23, inside the guide's 15-25% band.

## Findings closed by measurement

- **es-w1-C06** — label 'Política de selección' -> 'Política de cuentas' (19 cells): keeps policy and accounts, closes round 1's es-w1-B01 and round 2's es-w1-C06.

## Findings still open

- **es-w1-C13** (nit) — open — help string unchanged by the delta
  - 'de una petición con éxito' is a calque of 'a successful request'; 'una petición correcta' is the idiomatic form.
  - Proposed: 'Una concesión compatible y sin vencer de una petición correcta tiene prioridad…'
- **es-w1-C14** (nit) — open — help string unchanged by the delta
  - 'concesión' renders 'lease' and is an unrecorded term decision: it appears once and nowhere else in the locale.
  - Proposed: Record it in the es guide when the account-routing copy is next touched.

## Accepted trades (considered, not findings)

None.

## Source-side notes

- (width) The two long help strings (423 and 339 cells) are not whole at 80 or 100 columns, but neither is their English source (376 and 275): the Spanish does not change the class of the defect, so by the same rule this audit applied to console C01/C02 it is recorded as a source-side budget gap, not a Spanish finding. The sidecar pins no maxCells here.

## Limitations

- LLM semantic judgement, not native-speaker attestation.
- Delta audit: only the four changed values were re-judged; the rest were diffed, not re-read.
- No rendered frame was captured; width verdicts come from the renderer's own truncate_cells.
- The catalogue is untracked, so this binding is to on-disk bytes and can be invalidated by a further edit. No copy of the superseded bytes is retained anywhere.
- The audit model is an auto-routing alias; the concrete backend is not identifiable.

## Model

SELF-REPORTED from this session's model-switch notices; not independently observable. The request opened on anthropic/claude-sonnet-5-5; three provider-failure switches fired before any audit work and the whole audit ran under radient/auto, an auto-routing alias whose concrete backend the header does not identify. No pinned model claimed.

Switch chain: anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto; each transition a session model-switch notice reading "provider failure: quota".
