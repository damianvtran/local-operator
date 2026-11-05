# French wire.console staged translation

8/8 keys translated. Not audited, integrated or activated.

Source commit: 0e7cde33ca8008d561e74f4aed774aa5f76ff7d5. Exact English/context snapshots, style/glossary snapshots and hashes are retained. caller-proof.json is source inspection, NOT runtime execution evidence. It includes complete sibling-locale key/non-English-value counts and caller excerpts with immutable file hashes.

## Native choices

- live terminal: terminal actif
- process slot: emplacement de processus
- terminal cell: cellule de terminal
- managed memory: mémoire gérée, not arbitrary child-process RSS
- admission: admission; controls future starts, never stops running work
- bounds: 32–256 / 128 default; 1–32 / 8 default; 4–32 / 16 default; user reservation at most one quarter of live ceiling
- fairness: Each conversation shares global live allowance; finished terminals do not consume it; agents cannot consume user-reserved slots.

## Caller/context caveat

English fallback until independently translated is a transitional note, not authorization to activate this scratch translation; all six sibling locales have complete non-English-value copies. No English-only parser contract was found in these metadata consumers.

## Actual validation

validation.log captures the actual existing ICU parser execution at 256 MB heap: 8 exact keys in English order, no added arguments, no bidi controls, numeric literal parity, two bad controls rejected. widths.log captures rich.cells.cell_len results; rendered fit was NOT RUN. No full validator/scanner/native/build/pseudo/RTL/semantic-audit claims.

## Provenance

Served model(s): openai/gpt-6.1-sol, derived from this packet's translation write/edit usage entries, not model header. Session 7b2ba4ed501d, i18n-translator. provenance.json names entries and quota-fallback routes. No MT service used.

## Handoff

Immutable French gap: 8; staged gap: zero; repository gap remains 8 until integration. Independent audit required. These are translator staging records, not audit records; do not mark passed or activate from them. No source-English/context edits, generated edits, ledger edits, commits, PRs or unrelated namespace changes.
