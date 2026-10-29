# Spanish audit — wire.slash.library — round 2 (w1a3)

- **Verdict: passed** — 0 blocker, 0 major, 0 minor, 0 nit
- Audit id `w1a3`, round 2, 2026-10-29; supersedes `w1a2` (round 1, which bound superseded bytes sha256 `f6cc726978641736…`)
- Auditor: translation-auditor (lopdev team delegation es-w1-auditor-r2), session 3da6fd7a1b13, model **radient/auto** (self-reported; disclosed fallback — see Model)
- Revision `efcfbe812609cbf228c01f11a1fb622f3555b487`, branch feat/i18n-es-wire-runtime
- **The es catalogue is untracked at this revision** (`git status --porcelain` → `??`). The audited bytes are pinned by sha256, not by HEAD: `f6cc726978641736096351eb2f4bc7d754ce848cc6b3b93d8f5523cecaf47875`
- Keys: 2 in `en`, 2 in `es`, no en context sidecar (absent by construction)
- Independence: I authored none of this copy, audited none of it before, and edited nothing. My only writes are this directory.

## Verdict reason

Both keys read. The two descriptions are accurate, use the locale's established tú-imperative register for slash-command help ('Lista', 'Abre', 'Muestra', 'Compacta' elsewhere in wire.slash), keep all three verbs of 'Open, pause or run', and use the renderings the catalogue already established for Library, Librarian and assets. No findings.

## What was verified

1. 'Explora la Biblioteca global de recursos' renders 'Browse the global asset library'; 'recursos' for 'assets' is this locale's word, and 'Biblioteca' / 'Bibliotecario' capitalisation matches the catalogue's existing renderings (wire.settings.librarian_name.*, wire.errors.desktop).
2. 'Abre, pausa o ejecuta el Bibliotecario' keeps all three verbs of 'Open, pause or run the Librarian'.
3. The register matches the neighbouring slash descriptions rather than the infinitive action-label rule, which is correct here: these are command descriptions, not buttons.
4. The two values are byte-identical to the round-1 bound bytes (sha256 f6cc7269...), so no remediation happened and none was needed.
5. 400 cells is not at issue: the longer value is 40 cells, far inside any budget.

## Findings

None. No blocker, major, minor or nit finding in this namespace.

## Width note on the two over-budget help strings

**Status: not-applicable.** No help string in this namespace carries a declared budget, and none is near one.

## Source-side and display-side risk

- (sidecar) There is no en/wire.slash.library.context.json, so role, tone and any maxCells are unknown by construction; both keys were judged from the English value and the key name, and no context was invented.
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
