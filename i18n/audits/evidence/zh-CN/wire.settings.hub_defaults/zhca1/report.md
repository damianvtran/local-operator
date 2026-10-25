# Attestation — zh-CN / wire.settings.hub_defaults / zhca1

- **Verdict: not_carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 9 (en) / 9 (zh)

## Hash equality proof

- Source: prior row `9e785003bab2f7c01f32bae9b2cb114a240a82bac509c05bc55266585a41241d` = held `9e785003bab2f7c01f32bae9b2cb114a240a82bac509c05bc55266585a41241d` = current `8720b5719479b638eab24b081f76f97c44ce8d8554cb1b8280f6980182b6740c` — **DIFFERS**
- Target: held `2ba72847cff17089eaf641666b2ea7602a4e27bbc585a97da01d1e400ef25197` = current `b547b9730d78a81e840746de723e8a44597a3dcb175cd7e0200a143bb4972c13` — **DIFFERS** (the prior ledger row does not record a target hash)
- Context sidecar: DIFFERS (`a337e7394f3856476044f9accdb8daeb7a864a9a8ecfe0c579ec97ec646712ba`)
- Whole target file byte-identical to held candidate: **false**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 9 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 2 declared `maxCells` key(s), 0 over budget

## Stratified sample

Not performed: the namespace is not attestable.

## Change since the held candidate

- en keys: 9 -> 9; zh keys: 9 -> 9
- added in en: none
- added in zh: none
- removed in zh: none
- changed in zh: wire.settings.hub_defaults.agents.help, wire.settings.hub_defaults.agents.label, wire.settings.hub_defaults.agents.off.description, wire.settings.hub_defaults.agents.on.description

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. No attestation or semantic read is made: the hashes differ and a fresh audit is required.
