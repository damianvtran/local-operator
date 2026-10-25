# Attestation — zh-CN / cli.exec_runtime / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 11 (en) / 11 (zh)

## Hash equality proof

- Source: prior row `3eee7ca31076e48ccf7e1c8854c5647d0859e5341b26b10c8043e53825de72fd` = held `3eee7ca31076e48ccf7e1c8854c5647d0859e5341b26b10c8043e53825de72fd` = current `3eee7ca31076e48ccf7e1c8854c5647d0859e5341b26b10c8043e53825de72fd` — **equal**
- Target: held `819ddfc84fc2797279c9196ac72696b4ed2ff376c6239c42ccc602d5d3fa5211` = current `819ddfc84fc2797279c9196ac72696b4ed2ff376c6239c42ccc602d5d3fa5211` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`8a1e4417475daf6d0de3398f511eb46fb591cbc586cd3a039e7813a139d84e33`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 11 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

3/11 keys read (27.3%), including all 0 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
