# Attestation — zh-CN / cli.browser / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 64 (en) / 64 (zh)

## Hash equality proof

- Source: prior row `a9841deed81927762716c7592a592bb4e3f069d16f6ba0def6becb755b728321` = held `a9841deed81927762716c7592a592bb4e3f069d16f6ba0def6becb755b728321` = current `a9841deed81927762716c7592a592bb4e3f069d16f6ba0def6becb755b728321` — **equal**
- Target: held `ee537da4b8602e18728fe5a92f11e27d18b4c0714709fb356ea9577569da7bda` = current `ee537da4b8602e18728fe5a92f11e27d18b4c0714709fb356ea9577569da7bda` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`1e5f1dfeaa5d828302831cba4134605a1a7ff46b5c8ac7af7fcd913da95cbaee`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 64 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 2 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

16/64 keys read (25.0%), including all 6 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
