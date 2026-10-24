# Attestation — zh-CN / cli.config_help / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 9 (en) / 9 (zh)

## Hash equality proof

- Source: prior row `97f625f6c6e26f8723e471569a0227f8e4da18518f2ab50e444fe93345ab2c8c` = held `97f625f6c6e26f8723e471569a0227f8e4da18518f2ab50e444fe93345ab2c8c` = current `97f625f6c6e26f8723e471569a0227f8e4da18518f2ab50e444fe93345ab2c8c` — **equal**
- Target: held `384100fa02cc33629a498cee3e675db5dad6bc2a5c8d30254add3b212452d92b` = current `384100fa02cc33629a498cee3e675db5dad6bc2a5c8d30254add3b212452d92b` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`f1bc05e555bc66b4690ffc87eac141afed72d5e211ca634a94aa2c846eb71c3b`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 9 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

5/9 keys read (55.6%), including all 2 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
