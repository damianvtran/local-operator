# Attestation — zh-CN / cli.sessions_help / zhca1

- **Verdict: not_carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 47 (en) / 47 (zh)

## Hash equality proof

- Source: prior row `81b18d3d6b7e10419262b2ea97a4dfc423779be6d82b872dbf04209704c2bfe3` = held `81b18d3d6b7e10419262b2ea97a4dfc423779be6d82b872dbf04209704c2bfe3` = current `a250a185a780ab330187560c23dfe5e8d95956da24d417b22e1fc8bb121c2a6a` — **DIFFERS**
- Target: held `3b859a0e1f9f5a244b41e14aab9ea4be5319dabed46caaa0b825be5fca410cac` = current `d8e41d97fdbfdf890e14088ebdd44f87aeb3c2efb49a6401b430f45e51914bf8` — **DIFFERS** (the prior ledger row does not record a target hash)
- Context sidecar: DIFFERS (`0b1f7b4fc04b7d033d9b2a495415b115bca204ce4b38f641aa42ead4bf7151bb`)
- Whole target file byte-identical to held candidate: **false**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 47 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 4 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

Not performed: the namespace is not attestable.

## Change since the held candidate

- en keys: 46 -> 47; zh keys: 46 -> 47
- added in en: cli.sessions_help.cleanup_scratchpads_only
- added in zh: cli.sessions_help.cleanup_scratchpads_only
- removed in zh: none
- changed in zh: none

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. No attestation or semantic read is made: the hashes differ and a fresh audit is required.
