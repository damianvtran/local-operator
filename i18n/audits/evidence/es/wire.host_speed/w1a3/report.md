# Spanish audit — wire.host_speed — round 2 (w1a3)

- **Verdict: passed** — 0 blocker, 0 major, 0 minor, 1 nit
- Audit id `w1a3`, round 2, 2026-10-29; supersedes `w1a2` (round 1, which bound superseded bytes sha256 `1c2281ae22b7ed44…`)
- Auditor: translation-auditor (lopdev team delegation es-w1-auditor-r2), session 3da6fd7a1b13, model **radient/auto** (self-reported; disclosed fallback — see Model)
- Revision `efcfbe812609cbf228c01f11a1fb622f3555b487`, branch feat/i18n-es-wire-runtime
- **The es catalogue is untracked at this revision** (`git status --porcelain` → `??`). The audited bytes are pinned by sha256, not by HEAD: `b572842749c73bbb1e718bdef81b40317620bbcf5c3e20141e9b36cf8b7e9942`
- Keys: 2 in `en`, 2 in `es`, 2 in the en context sidecar
- Independence: I authored none of this copy, audited none of it before, and edited nothing. My only writes are this directory.

## Verdict reason

Both keys read. Round-1's minor finding was applied: the help now names the local consumer ('el tiempo que tarda el consumidor local') rather than an abstract consumption. The label is 26 cells against its declared 29-cell budget and 'tok/s' is byte-identical. One round-1 nit survives unchanged where the bytes are unchanged; it is a preference and does not gate.

## What was verified

1. Round-1 MINOR es-w1-B02 is APPLIED: the round-1 text was 'sin contar el tiempo de consumo local'; the final bytes read 'sin contar el tiempo que tarda el consumidor local', which names the actor the English names.
2. The experimental / off-by-default disclaimer, the one-model one-run evidence caveat, the three-long-replies trigger, the positive-floor semantics, '0 clears speed exclusions while cache-miss exclusions remain' and the cold-cache warning all survive.
3. Label 'retirar host lento (tok/s)' is 26 cells against the declared maxCells 29: no-op for truncate_cells, and 'tok/s' is byte-identical.
4. The help is 451 cells (en 310), which cannot fit the detail band and does not fit the English either; the sidecar pins no maxCells for it. Recorded so the number is not mistaken for a regression: the source itself is longer than any detail line.

## Findings

### es-w1-C12 — nit — carried-unresolved

Keys: `wire.host_speed.retire_below_tps.help`

'Cambiar de host puede iniciar una caché de prompts en frío' calques 'Moving hosts may start a cold prompt cache': a cache does not get started cold, it starts cold. The catalogue already has the parallel 'costarte una caché de prompts en caliente'. Round-1 nit es-w1-B03, not adopted; the sentence is otherwise a strong translation.

**Proposed:** 'Cambiar de host puede hacer que la caché de prompts arranque en frío.'

## Width note on the two over-budget help strings

**Status: not-applicable.** The two over-budget help strings are not in this namespace; the one declared budget here (label, 29) is met.

## Source-side and display-side risk

- (sidecar) The 451-cell help carries no declared maxCells, and its English source (310 cells) already exceeds the detail band, so truncation is inherited from the source rather than introduced by the translation.
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
