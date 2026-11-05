# French account-policy translation staging packet

Status: 14/14 translated, staged only; not audited, activated, committed or released.

English and old context are immutable commit 0e7cde33ca8008d561e74f4aed774aa5f76ff7d5. English SHA-256 is verified against the supplied hash. The original context contains stale English-only notes; their exact bytes are retained in inputs/en.context.json. The manager authorized human-copy translation and reports a context-only correction is being prepared. Final auditors must use and hash the corrected actual context; this packet does not pretend it already exists.

## Validation actually executed

See validation.log (real Node ICU execution, 256 MB heap) and widths.log (real rich.cells.cell_len execution). 14 exact keys in English order; no arguments added; true/false/value/JSON retained; three intentionally bad fixture controls rejected. All five labels fit 29 cells: 28, 26, 24, 21, 26. No semantic audit or rendered/pseudo/RTL proof is claimed.

## Native glossary proposals

- account selection policy: politique de choix du compte (compact label), politique de sélection des comptes (unbounded prose).
- credential tier: catégorie d’identifiants, not a claim about privilege escalation.
- compatible, unexpired lease: bail compatible et non expiré; applies to all policies and guarantees no cache hit.
- Reset soonest: Prochaine réinitialisation; weekly refill precedes five-hour tie-breaking within one minute.
- Most headroom: Quota disponible maximal; spread uses ten percentage POINTS.
- Fair spread: Répartition équitable; stable session-based, not quota/reset ranking.
- legacy overrides: anciennes dérogations; explicit policy wins, reset restores default and clears overrides.

No style/glossary files were edited. Vous/imperative and French narrow no-break spaces follow the read style guide. Identifiers, enum values, HTTP status and code fields remain separate from human copy; caller-proof.json is explicitly source reading, not execution evidence.

## Provenance

Actual served model openai/gpt-6.1-sol verified from persisted usage on the translation-writing tool call b2aa25b6c8b443dc91bbf52ceb567528 and label-edit call 55a908ca38bd47d98ffc43fd9bde6101. Route events record quota fallbacks from configured anthropic/claude-sonnet-5-5 through anthropic/claude-opus-5-5 to openai/gpt-6.1-sol. See provenance.json. Direct LLM translation, no MT service.

## Gaps and handoff

Immutable French namespace is absent: 14 missing keys. Packet: zero missing keys. The repository gap remains 14 until integration; no global French/other-language completeness claim. First and second independent auditors are required. SDK namespaces and other locale packets are outside scope.
