# Spanish audit — wire.settings.account_policy — round 2 (w1a3)

- **Verdict: passed** — 0 blocker, 0 major, 1 minor, 2 nit
- Audit id `w1a3`, round 2, 2026-10-29; supersedes `w1a1` (round 1, which bound superseded bytes sha256 `ce1a2c0185b94c91…`)
- Auditor: translation-auditor (lopdev team delegation es-w1-auditor-r2), session 3da6fd7a1b13, model **radient/auto** (self-reported; disclosed fallback — see Model)
- Revision `efcfbe812609cbf228c01f11a1fb622f3555b487`, branch feat/i18n-es-wire-runtime
- **The es catalogue is untracked at this revision** (`git status --porcelain` → `??`). The audited bytes are pinned by sha256, not by HEAD: `88c7c68e05d9a173dc7251c68df3de3f1436a09b1605dba2e34c58ddc2b5358e`
- Keys: 14 in `en`, 14 in `es`, 14 in the en context sidecar
- Independence: I authored none of this copy, audited none of it before, and edited nothing. My only writes are this directory.

- **Post-audit drift:** this catalogue changed after the audit bound it. Bound `88c7c68e05d9a173dc7251c68df3de3f1436a09b1605dba2e34c58ddc2b5358e`; current `f5824ce686fde85dea395fbca9d91d7354bea0cd8a4a4bbe545dc9b1ed24d04c`. This record binds the audited bytes, not the current file. The catalogue is being edited while this amendment was written: it moved twice, so the hash above is what was on disk at the moment of this note and nothing more. This record binds the AUDITED bytes only. Re-measured only (not re-audited): the label now reads 'Política de cuentas' at 19 cells (the delegation quoted 18), which keeps both 'policy' and 'accounts' and closes es-w1-C06 by measurement on those bytes. A fresh binding must be taken once the wave settles.

## Verdict reason

All 14 keys read. Round-1's minor ('the label drops policy') was applied: the label is now 'Política de selección', which keeps 'policy' and fits its declared 29 cells. All five labels fit their declared budgets. The lease sentence, the does-not-guarantee-a-cache-hit caveat, the tie rule, the override precedence and the reset-restores-Reset-soonest-and-clears-legacy-overrides instruction all survive, with true/false literal and the enum ids untranslated. One minor and two nits remain: the applied label now drops 'accounts' instead, and the two round-1 nits were not adopted.

## What was verified

1. All five labels meet their declared maxCells of 29: label 21, reset_soonest 20, headroom 10, fair 18, legacy 27. No label clips.
2. Round-1 MINOR es-w1-B01 is APPLIED: the round-1 text was 'Selección de cuentas'; the final bytes read 'Política de selección'.
3. legacy.help names the same labels the Spanish labels use: 'Más margen' and 'Reparto equitativo' appear verbatim, and 'Reinicio más próximo' is the restore target - no dangling reference to a control with a different name.
4. true/false stay literal in legacy.help, and the enum ids live in key suffixes, which match the English key set exactly.
5. 'Reinicio más próximo' is used consistently for the 'Reset soonest' policy in the label, the help and legacy.help.
6. Every range/default-free refusal keeps its meaning: 'El enrutamiento de cuentas no está disponible', 'Solo se esperaba un campo de valor', 'Se esperaba un objeto JSON', and the write refusal.

## Findings

### es-w1-C06 — minor — new

Keys: `wire.settings.account_policy.label`

'account selection policy' is rendered 'Política de selección' (21 cells). Round-1's finding was that 'policy' was missing; the applied fix keeps 'policy' and now drops 'accounts'. The row is registered in section 'failover' (settings_io.py:2276-2282, key retry.accountSelectionPolicy), so nothing on screen names accounts next to the label; the help does, in its first sentence. 'Política de selección de cuentas' is 32 cells and would clip, so the trade is forced, but this is the worse half of it: 'selection policy' is generic in a failover section.

**Proposed:** 'Política de cuentas' (19 cells) - keeps both the domain noun and 'policy', dropping only 'selection', which the row's three value labels already show.

### es-w1-C13 — nit — carried-unresolved

Keys: `wire.settings.account_policy.help`

'de una petición con éxito' is a literal rendering of 'from a successful request'; Spanish UI normally writes 'una petición correcta'. Round-1 nit es-w1-B02, not adopted.

**Proposed:** 'de una petición correcta'.

### es-w1-C14 — nit — carried-unresolved

Keys: `wire.settings.account_policy.help`

'Una concesión compatible y sin vencer' renders 'lease'. The choice is defensible and used consistently, but it is in neither i18n/glossary.md nor the es guide, so a later wave can silently pick another word. Round-1 nit es-w1-B03, not adopted.

**Proposed:** Record 'lease -> concesión' in i18n/style-guides/es.md the next time account-routing copy is touched.

## Width note on the two over-budget help strings

**Status: not-applicable.** The two over-budget help strings are not in this namespace; every declared label budget here is met.

## Source-side and display-side risk

- (sidecar) This sidecar DOES pin maxCells 29 on all five labels, which is why round 1 could reason about the trade. The descriptions and help carry no budget although they render on the detail band (99-423 cells against 98): inherited from the English, which is also over.
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
