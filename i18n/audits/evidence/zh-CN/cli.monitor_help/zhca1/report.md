# Attestation — zh-CN / cli.monitor_help / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `anthropic/claude-sonnet-5-5`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 6 (en) / 6 (zh)

## Hash equality proof

- Source: prior row `766f6162643fec0c473dd4ffd822e5f1f9590e32caa2bc98793e9d65061161ab` = held `766f6162643fec0c473dd4ffd822e5f1f9590e32caa2bc98793e9d65061161ab` = current `766f6162643fec0c473dd4ffd822e5f1f9590e32caa2bc98793e9d65061161ab` — **equal**
- Target: held `e2ebeb0c826a483a20c8dc2307fa0d425fe717088204bdcb575e884acef99396` = current `e2ebeb0c826a483a20c8dc2307fa0d425fe717088204bdcb575e884acef99396` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`35396c652554d037e336be0d49098a4d1c4cc8f2b09df1a60726f1d5c8af250c`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 6 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

3/6 keys read (50.0%), including all 0 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
