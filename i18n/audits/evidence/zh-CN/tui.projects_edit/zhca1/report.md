# Attestation — zh-CN / tui.projects_edit / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 35 (en) / 35 (zh)

## Hash equality proof

- Source: prior row `dd911df0afbd69461e9831a8a5b7b6218795b6c2aee7c3f1be12a0a9aaaa3650` = held `dd911df0afbd69461e9831a8a5b7b6218795b6c2aee7c3f1be12a0a9aaaa3650` = current `dd911df0afbd69461e9831a8a5b7b6218795b6c2aee7c3f1be12a0a9aaaa3650` — **equal**
- Target: held `15908975fd6c0eb0f9a5e7fa5ee213e5cdcb5ddd6dc9ef982467a9b731ba83e1` = current `15908975fd6c0eb0f9a5e7fa5ee213e5cdcb5ddd6dc9ef982467a9b731ba83e1` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`db995023c6489ee9313403c2f463c36ea65162ceea52a572c3da26db02fbd01c`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 35 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

6/35 keys read (17.1%), including all 0 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
