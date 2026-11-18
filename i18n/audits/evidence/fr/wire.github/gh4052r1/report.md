# PR #4052 independent translation audit — round 1

Locale: fr; namespace: wire.github. Auditor 91d1e05d2d82, actual openai/gpt-6.1-sol after Opus quota fallback. Source head 0b754ce89567ae158fe91db3ec0ed0b4a621f312.

Scope: all 68 new wire.github values, or only the one settings help value. The other 866 settings target values and English source values are unchanged from the named base; prior passed receipts are retained in summary.json. No new audit of those unchanged settings values is claimed.

Mechanical independent checks and repository validator pass; all target/source receipt hashes match current bytes. This audit executed 6,720 ICU renders across English and seven target locales (840 per locale), both select arms, all CLDR categories, 12 numeric probes; 68 protected-token pseudo expansion/wrapping controls passed. Static Rich 100-column CLI-copy SVGs were converted with rsvg-convert and all seven target PNGs actually viewed. Context gives no fixed width cap. These are raw copy/layout evidence, not screenshots of the live app. Urdu frame visibly exhibits the known unshaped/degraded path; native terminal bidi/shaping is NOT certified. Production messages.render_record remains English because all target locales are currently ineligible; direct ICU candidate formatting is not activation proof. Explicit balanced-isolate adapter probe is not proof that plain CLI invokes that adapter.

Safety: exact owner-only remedies, no access lent on refusal, last-good uncertainty, unknown versus earlier login, retrieval-time opt-out, and never-set versus cleared previously shared list all reviewed. No authority or remote deletion promise added. No translation service used.

Execution note: render attempt 1 used a negative fixture missing other that this parser accepts; its assertion failed. Corrected fixture has an unmatched closing structure and failed as expected in attempts 2/3. Attempt 3 also protects command/literal tokens during AST-based pseudo expansion. The retained audit.log includes the failed setup, not a fabricated green first run.

## Findings

### I-FR-01 — MINOR

Échec de la liste names a failed list rather than failed retrieval; surrounding failure copy uses an operation. Meaning remains recoverable.

- `wire.github.failure_unknown`
  - Current: échec de la liste des dépôts GitHub
  - Suggested: échec de la récupération de la liste des dépôts GitHub


## Eligibility disposition

Only failed es/ru/ur wire.github receipts are appended now. Eleven clean locale/namespace receipts are intentionally NOT appended yet: manager requested no eligibility restoration before the unified translation remediation. Existing translator-only rows remain current there. Generator --check currently fails on seven orphaned locale mobile bundles; none were deleted or changed. Second independent pass is separately preserved under gh4052s1, with its own actual identity, original artifacts and minor/major ratings.
