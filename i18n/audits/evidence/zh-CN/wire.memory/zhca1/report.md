# Attestation — zh-CN / wire.memory / zhca1

- **Verdict: not_carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 17 (en) / 17 (zh)

## Hash equality proof

- Source: prior row `8732c8a8ab1572a39bf81ff94d80c8ea02f146d7da933db6eb41ee3c73062a00` = held `8732c8a8ab1572a39bf81ff94d80c8ea02f146d7da933db6eb41ee3c73062a00` = current `319e23e6e4a6abee4e9549655142bebb5402563902c6f0497f3ee761fb53aec7` — **DIFFERS**
- Target: held `304e44b834272be678bdcacc57b48b18810690e9a2d283156d419f5c58098fc1` = current `4e39d2bacb0ea88a16f7de720a5ae2208a580ea789ecc135a1c4368d2f3d1e28` — **DIFFERS** (the prior ledger row does not record a target hash)
- Context sidecar: DIFFERS (`852518b5d2bd41f4b8b72d6d89d961d35b1bafaf341726e0406973d12b1c5e7d`)
- Whole target file byte-identical to held candidate: **false**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 17 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

Not performed: the namespace is not attestable.

## Change since the held candidate

- en keys: 15 -> 17; zh keys: 15 -> 17
- added in en: wire.memory.capture.locators.label, wire.memory.capture.locators.help
- added in zh: wire.memory.capture.locators.label, wire.memory.capture.locators.help
- removed in zh: none
- changed in zh: wire.memory.capture.help

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. No attestation or semantic read is made: the hashes differ and a fresh audit is required.
