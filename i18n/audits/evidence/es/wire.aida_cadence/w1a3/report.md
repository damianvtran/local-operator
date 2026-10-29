# Spanish audit — wire.aida_cadence — round 2 (w1a3)

- **Verdict: passed** — 0 blocker, 0 major, 0 minor, 0 nit
- Audit id `w1a3`, round 2, 2026-10-29; supersedes `w1a2` (round 1, which bound superseded bytes sha256 `69f2bdb08d2175c6…`)
- Auditor: translation-auditor (lopdev team delegation es-w1-auditor-r2), session 3da6fd7a1b13, model **radient/auto** (self-reported; disclosed fallback — see Model)
- Revision `efcfbe812609cbf228c01f11a1fb622f3555b487`, branch feat/i18n-es-wire-runtime
- **The es catalogue is untracked at this revision** (`git status --porcelain` → `??`). The audited bytes are pinned by sha256, not by HEAD: `119a09f199b86afe94bfac1614def110ac0c5d9b6949e3074848af4978dd52b8`
- Keys: 2 in `en`, 2 in `es`, 2 in the en context sidecar
- Independence: I authored none of this copy, audited none of it before, and edited nothing. My only writes are this directory.

## Verdict reason

Both keys read. The whole-elapsed-hours distinction (horas transcurridas), the 0-keeps-the-daily-time rule, the inclusive 1-168 bound, the '4' example and the already-owed-check-in promise all survive, and the en dash in 1–168 is byte-identical. Round-1 nit (the word order of the first sentence) was applied and reads correctly now. The label sits exactly at the 29-cell limit: no-op for truncate_cells, but zero slack.

## What was verified

1. label 'Intervalo de contacto (horas)' is 29 cells; truncate_cells(x, 29) returns it unchanged (no-op). It is exactly at the limit, so any later lengthening clips.
2. 'contacto' is the locale's established rendering of an aida check-in, not drift: wire.settings.aida_cadence_at.label is 'Hora de contacto diario', aida_cadence_max_extra_per_day.label is 'Contactos extra / día'.
3. 'horas transcurridas' keeps the sidecar's whole-elapsed-hours distinction; 'la hora local del contacto diario' attaches 'local' to the clock, which is what round 1's nit asked for.
4. 'un contacto ya pendiente' keeps the preservation of an already-owed check-in without promising delivery during a pause.
5. Round-1 nit es-w1-B01 (word order of the first sentence) is APPLIED: the round-1 text was '0 mantiene la hora de contacto diaria local.' and the final bytes read '0 mantiene la hora local del contacto diario.'

## Findings

None. No blocker, major, minor or nit finding in this namespace.

## Width note on the two over-budget help strings

**Status: not-applicable.** The two over-budget help strings are not in this namespace; no per-key maxCells is declared here either.

## Source-side and display-side risk

- (sidecar) The sidecar declares no maxCells for either key, although the label is rendered through the 29-cell settings truncation. Recorded as a source-side risk: a translator has no width signal for this label.
- (display) No value in this namespace displayed as a redaction marker, so no sanctioned read_public_copy re-read was needed to judge any of the copy. Verified source-side instead: the reader serves the HEAD blob and REFUSES a working-tree difference or an untracked file (local_operator/tools/public_copy.py:487-496), so it cannot read these untracked es bytes at all; and its catalogue reader rejects any value that is not a string (public_copy.py:343-345), so *.context.json sidecars are unreadable through it by construction. Both are recorded as source-side risks, never worked around.

## Limitations

- LLM semantic judgement, not native-speaker attestation.
- Static copy audit only: no rendered-frame, runtime-service, eligibility or release claim. Width verdicts are derived from the renderer source at the audited revision and from the repository's own truncate_cells, not from screenshots.
- The eight es catalogues are UNTRACKED at efcfbe812609; every record binds on-disk sha256 values rather than a committed blob.
- The audit model is an auto-routing alias; the concrete backend is not identifiable from the session header.
- Evidence files were written but never committed: this role performs no git writes.

## Model

SELF-REPORTED from this session's own model-switch notices; not independently observable. The request opened on anthropic/claude-sonnet-5-5; three provider-failure switches fired before any audit work began and the entire audit ran under radient/auto, an auto-routing alias whose concrete backend the session header does not identify. No pinned model is claimed.

Switch chain: anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto; each transition was a session model-switch notice reading "provider failure: quota".
