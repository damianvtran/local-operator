# Attestation — zh-CN / tui.status / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-11, model `deepseek/deepseek-flash`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 7 (en) / 7 (zh)

## Hash equality proof

- Source: prior row `2403f9c4f37b40264f575929627b99a0bcabb0baf1bcb6a0b951c6c31ec6843a` = held `2403f9c4f37b40264f575929627b99a0bcabb0baf1bcb6a0b951c6c31ec6843a` = current `2403f9c4f37b40264f575929627b99a0bcabb0baf1bcb6a0b951c6c31ec6843a` — **equal**
- Target: held `6016e15734c3c7dd13b87fdb0d7aa104d36084d23b8fa362228699f278af2232` = current `6016e15734c3c7dd13b87fdb0d7aa104d36084d23b8fa362228699f278af2232` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`3bab032efffea4adb669aa1f3c8d7a489f7b3b630b19b52636d70bd2b09bdfbc`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 7 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

3/7 keys read (42.9%), including all 0 messages over 120 characters and all 2 plural messages.

## Findings

- **ZHC-08 (nit)** `tui.status.agents` — Four carried strings put a space between # and the classifier while 26 do not. The style guide's plural example shows a space, but its punctuation rule forbids a space between Chinese and digit runs; either way the corpus should not be split. Proposed: follow the majority and the punctuation rule.
  - current: `{count, plural, other {# 个智能体}}`
  - proposed: `{count, plural, other {#个智能体}}`
- **ZHC-08 (nit)** `tui.status.jobs` — Four carried strings put a space between # and the classifier while 26 do not. The style guide's plural example shows a space, but its punctuation rule forbids a space between Chinese and digit runs; either way the corpus should not be split. Proposed: follow the majority and the punctuation rule.
  - current: `{count, plural, other {# 个任务}}`
  - proposed: `{count, plural, other {#个任务}}`

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
