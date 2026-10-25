# Attestation — zh-CN / wire.aida_cadence / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 2 (en) / 2 (zh)

## Hash equality proof

- Source: prior row `3bb421a58f2ffa4fec089421d9cb91257fa626d66b2e215b3e120c068bb8e5dd` = held `3bb421a58f2ffa4fec089421d9cb91257fa626d66b2e215b3e120c068bb8e5dd` = current `3bb421a58f2ffa4fec089421d9cb91257fa626d66b2e215b3e120c068bb8e5dd` — **equal**
- Target: held `7ef9cecdc2d0e4906f3ab603fbe92305e784d3c7df0992cab83b36956b8fa3b3` = current `7ef9cecdc2d0e4906f3ab603fbe92305e784d3c7df0992cab83b36956b8fa3b3` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`6efad501b08345b72e5d4d19cfee3dd27a02a48c2eb721780cf9d7f8c7694d0e`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 2 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

2/2 keys read (100.0%), including all 1 messages over 120 characters and all 0 plural messages.

## Findings

- **ZHC-03 (minor)** `wire.aida_cadence.interval_hours.label` — The carried cadence namespace says 问候 (wire.aida_cadence), while the TUI status surface says 回访 (tui.app.aida). The settings namespace that is not carried (wire.settings) says 检查. One feature term should have one rendering; 问候 matches the cadence labels that name the setting.
  - current: `问候间隔（小时）`
  - proposed: no change (this rendering is the reference side)
- **ZHC-03 (minor)** `wire.aida_cadence.interval_hours.help` — The carried cadence namespace says 问候 (wire.aida_cadence), while the TUI status surface says 回访 (tui.app.aida). The settings namespace that is not carried (wire.settings) says 检查. One feature term should have one rendering; 问候 matches the cadence labels that name the setting.
  - current: `0保留每日本地问候时间。1–168按经过的小时数计时；4表示每四小时问候一次。更改会保留已到期应执行的问候。`
  - proposed: no change (this rendering is the reference side)

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
