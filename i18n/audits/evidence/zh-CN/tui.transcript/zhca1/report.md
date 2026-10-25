# Attestation — zh-CN / tui.transcript / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 17 (en) / 17 (zh)

## Hash equality proof

- Source: prior row `13915d6c6dcec0da9fd892c1cb74c23546045717c1d3400d264e8608e75876ba` = held `13915d6c6dcec0da9fd892c1cb74c23546045717c1d3400d264e8608e75876ba` = current `13915d6c6dcec0da9fd892c1cb74c23546045717c1d3400d264e8608e75876ba` — **equal**
- Target: held `a614afc86a13226ca2002e64b25f940bf0ae28a3bbccd80a1fc7bc61fda94b41` = current `a614afc86a13226ca2002e64b25f940bf0ae28a3bbccd80a1fc7bc61fda94b41` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`3c2252debbbc33c2fab2061a5dcdc9dfa036e07ab5793ba4dabdf889ac7113fc`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 17 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

7/17 keys read (41.2%), including all 0 messages over 120 characters and all 5 plural messages.

## Findings

- **ZHC-08 (nit)** `tui.transcript.attach.receipt` — Four carried strings put a space between # and the classifier while 26 do not. The style guide's plural example shows a space, but its punctuation rule forbids a space between Chinese and digit runs; either way the corpus should not be split. Proposed: follow the majority and the punctuation rule.
  - current: `↑ 已附加 {count, plural, other {# 张图像}}`
  - proposed: `↑ 已附加 {count, plural, other {#张图像}}`
- **ZHC-08 (nit)** `tui.transcript.catchup.missed` — Four carried strings put a space between # and the classifier while 26 do not. The style guide's plural example shows a space, but its punctuation rule forbids a space between Chinese and digit runs; either way the corpus should not be split. Proposed: follow the majority and the punctuation rule.
  - current: `{count, plural, other {# 次错过的唤醒}}`
  - proposed: `{count, plural, other {#次错过的唤醒}}`

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
