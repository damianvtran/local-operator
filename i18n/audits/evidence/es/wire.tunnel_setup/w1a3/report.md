# Spanish audit — wire.tunnel_setup — round 2 (w1a3)

- **Verdict: passed** — 0 blocker, 0 major, 0 minor, 0 nit
- Audit id `w1a3`, round 2, 2026-10-29; supersedes `w1a2` (round 1, which bound superseded bytes sha256 `f7d4114e7083bf26…`)
- Auditor: translation-auditor (lopdev team delegation es-w1-auditor-r2), session 3da6fd7a1b13, model **radient/auto** (self-reported; disclosed fallback — see Model)
- Revision `efcfbe812609cbf228c01f11a1fb622f3555b487`, branch feat/i18n-es-wire-runtime
- **The es catalogue is untracked at this revision** (`git status --porcelain` → `??`). The audited bytes are pinned by sha256, not by HEAD: `f146d6c709f86ecf311eddfbdfb541ab80bbe30dd93b719d0cb76ad57d496f91`
- Keys: 5 in `en`, 5 in `es`, 5 in the en context sidecar
- Independence: I authored none of this copy, audited none of it before, and edited nothing. My only writes are this directory.

## Verdict reason

All five keys read. Both round-1 minors were applied: 'drenado' is now 'vaciado' and 'del teléfono' is now 'del acceso telefónico', so the namespace now names the feature the same way in all three refusals. Every refusal keeps an actionable remedy, the do-not-assume-rollback warning survives, the deadline-after-the-work-was-drained ordering survives, and 'checking' still makes no claim that billing or installation has started. No findings.

## What was verified

1. Round-1 MINOR es-w1-B04 is APPLIED: 'una vez drenado el trabajo' -> 'una vez vaciado el trabajo que tenía a su cargo'.
2. Round-1 MINOR es-w1-B05 is APPLIED: 'La solicitud guardada de configuración del teléfono' -> 'La solicitud guardada de configuración del acceso telefónico', which now matches in_progress and unavailable ('la configuración del acceso telefónico').
3. create.json is byte-identical and the instruction not to touch it survives; no raw provider error, account fallback or private JSON detail is introduced, which the sidecar's 'cautious, actionable' tone and 'no raw private JSON/error detail' note require.
4. 'no des por hecho que se haya revertido' keeps 'do not assume rollback' without claiming a rollback happened.
5. The addressing is tú throughout ('Comprueba', 'Deja', 'pide'), matching the style guide.
6. No placeholders, digits, en dashes or machine literals in this namespace, so nothing of that class could be lost.

## Findings

None. No blocker, major, minor or nit finding in this namespace.

## Width note on the two over-budget help strings

**Status: not-applicable.** No key in this namespace carries a declared maxCells and none is a settings row label; the longest value is 222 cells and renders as a desktop refusal, not in the settings detail band.

## Source-side and display-side risk

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
