# Attestation — zh-CN / tui.schedule / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `anthropic/claude-sonnet-5-5`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 61 (en) / 61 (zh)

## Hash equality proof

- Source: prior row `6caa74d9248ba9ccfdd6e61ef97976a8a7fcb76ed47cb7cca30911c980f601d7` = held `6caa74d9248ba9ccfdd6e61ef97976a8a7fcb76ed47cb7cca30911c980f601d7` = current `6caa74d9248ba9ccfdd6e61ef97976a8a7fcb76ed47cb7cca30911c980f601d7` — **equal**
- Target: held `d53fa75a48cd16bb20d92aed73632f928687cbe5551a5bda4df76176342debfb` = current `d53fa75a48cd16bb20d92aed73632f928687cbe5551a5bda4df76176342debfb` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`829e18450b21ca836cc047813b9d18770414f1fd23f14c18b794c1f45648afb4`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 61 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

13/61 keys read (21.3%), including all 0 messages over 120 characters and all 4 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
