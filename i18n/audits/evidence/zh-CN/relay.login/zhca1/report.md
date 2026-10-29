# Attestation — zh-CN / relay.login / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 4 (en) / 4 (zh)

## Hash equality proof

- Source: prior row `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5` = held `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5` = current `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5` — **equal**
- Target: held `6f32c4567a03dab4db4abf5ee0899b9fe53a0b0550d67b8f8a476feb35a6a02e` = current `6f32c4567a03dab4db4abf5ee0899b9fe53a0b0550d67b8f8a476feb35a6a02e` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`9fcd5a1669e6831b10b7041d82f77c69e858183c001ba9faa949532b4d71980a`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 4 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

1/4 keys read (25.0%), including all 0 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
