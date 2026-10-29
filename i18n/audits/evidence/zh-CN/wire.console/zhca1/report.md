# Attestation — zh-CN / wire.console / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 8 (en) / 8 (zh)

## Hash equality proof

- Source: prior row `1a591fbc1f6a939651cfe9bc5c4ff619a5a4ade2099cc9cfb348534d09a0011b` = held `1a591fbc1f6a939651cfe9bc5c4ff619a5a4ade2099cc9cfb348534d09a0011b` = current `1a591fbc1f6a939651cfe9bc5c4ff619a5a4ade2099cc9cfb348534d09a0011b` — **equal**
- Target: held `3ce0028e16106b17b53750ead26a1d32845cb2218ec1ae46a52ad66d1586cb25` = current `3ce0028e16106b17b53750ead26a1d32845cb2218ec1ae46a52ad66d1586cb25` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`453aadc8095e93122719cef9b8da2ed9f8b57170449804e6af716fe684ef2dd9`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 8 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

5/8 keys read (62.5%), including all 3 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
