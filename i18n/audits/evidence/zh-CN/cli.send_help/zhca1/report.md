# Attestation — zh-CN / cli.send_help / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 7 (en) / 7 (zh)

## Hash equality proof

- Source: prior row `ab7d3c4a3ac0696b935d613c2b7596c56dbd3b32a8fdabe29ff2bc3a93a1bfb9` = held `ab7d3c4a3ac0696b935d613c2b7596c56dbd3b32a8fdabe29ff2bc3a93a1bfb9` = current `ab7d3c4a3ac0696b935d613c2b7596c56dbd3b32a8fdabe29ff2bc3a93a1bfb9` — **equal**
- Target: held `7a3a05cf2bb1519507764a1b6de4981c76ddc327e074125582a25e2b462f9d6f` = current `7a3a05cf2bb1519507764a1b6de4981c76ddc327e074125582a25e2b462f9d6f` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`cc8f29b50b4a841a2d88200c8c033b6da90e31f6d934c94e49cb92c7ac3ad193`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 7 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 1 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

4/7 keys read (57.1%), including all 2 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
