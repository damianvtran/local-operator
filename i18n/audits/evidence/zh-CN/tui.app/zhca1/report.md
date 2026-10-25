# Attestation — zh-CN / tui.app / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 46 (en) / 46 (zh)

## Hash equality proof

- Source: prior row `7e5f39df72438663a66afca0e5338e90b7b984bdca6b562195ce976945bd7875` = held `7e5f39df72438663a66afca0e5338e90b7b984bdca6b562195ce976945bd7875` = current `7e5f39df72438663a66afca0e5338e90b7b984bdca6b562195ce976945bd7875` — **equal**
- Target: held `e1f79b11108e0179e4b0a9e6b8a7c273931fd322ddc9e483d54042f45567b490` = current `e1f79b11108e0179e4b0a9e6b8a7c273931fd322ddc9e483d54042f45567b490` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`a840ac61e3c1e1178825257c1b9ebaa6ebfef500df4f3a29e8c463b50eef42fa`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 46 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

15/46 keys read (32.6%), including all 3 messages over 120 characters and all 6 plural messages.

## Findings

- **ZHC-03 (minor)** `tui.app.aida.next_checkin` — The carried cadence namespace says 问候 (wire.aida_cadence), while the TUI status surface says 回访 (tui.app.aida). The settings namespace that is not carried (wire.settings) says 检查. One feature term should have one rendering; 问候 matches the cadence labels that name the setting.
  - current: `下次回访{when}`
  - proposed: `下次问候{when}`
- **ZHC-03 (minor)** `tui.app.aida.extras_used` — The carried cadence namespace says 问候 (wire.aida_cadence), while the TUI status surface says 回访 (tui.app.aida). The settings namespace that is not carried (wire.settings) says 检查. One feature term should have one rendering; 问候 matches the cadence labels that name the setting.
  - current: `今日额外回访已用：{used} / {maximum}`
  - proposed: `今日额外问候已用：{used} / {maximum}`

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
