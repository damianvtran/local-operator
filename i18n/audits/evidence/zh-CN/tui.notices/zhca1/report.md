# Attestation — zh-CN / tui.notices / zhca1

- **Verdict: not_carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 30 (en) / 30 (zh)

## Hash equality proof

- Source: prior row `2eaf4097197ee0257cfcf7d9d0f450e06fb496db67a92edb21e6f6b1122aaef1` = held `2eaf4097197ee0257cfcf7d9d0f450e06fb496db67a92edb21e6f6b1122aaef1` = current `4a159689628113c32a2df75a9e3d98d74aab790c9946848307bb267318731510` — **DIFFERS**
- Target: held `035b9df1c4d6c2e124c7ffab6f2e49eb310f6508f08a6957ba042d3964e90ebe` = current `40af781ff575e4571fbb4caa7f2ae55443f1bdb2c344c2e04137d59198c77c61` — **DIFFERS** (the prior ledger row does not record a target hash)
- Context sidecar: DIFFERS (`3f1298de1bdb84a67bde66a5bb937f05376c81e06d866e94180f21b1525c8370`)
- Whole target file byte-identical to held candidate: **false**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 30 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

Not performed: the namespace is not attestable.

## Change since the held candidate

- en keys: 29 -> 30; zh keys: 29 -> 30
- added in en: tui.notices.fast_mode_refused
- added in zh: tui.notices.fast_mode_refused
- removed in zh: none
- changed in zh: none

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. No attestation or semantic read is made: the hashes differ and a fresh audit is required.
