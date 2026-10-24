# Attestation — zh-CN / tui.analytics / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 98 (en) / 98 (zh)

## Hash equality proof

- Source: prior row `aa5221066aa84eedf0f924628d51bc8f08ccf5f26b5075638b9d40062f7b617e` = held `aa5221066aa84eedf0f924628d51bc8f08ccf5f26b5075638b9d40062f7b617e` = current `aa5221066aa84eedf0f924628d51bc8f08ccf5f26b5075638b9d40062f7b617e` — **equal**
- Target: held `725d9653bcccf233f708719ec8b9180b53947a9118a73627382fe4f2bf5cfaea` = current `725d9653bcccf233f708719ec8b9180b53947a9118a73627382fe4f2bf5cfaea` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`04356facdf1802a34a4be9ab6d3255cc9e92826d2b4e9ae917c43f687a9325eb`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 98 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 1 key(s) checked, 0 failure(s)
- Width: 5 declared `maxCells` key(s), 0 over budget

## Stratified sample

28/98 keys read (28.6%), including all 1 messages over 120 characters and all 14 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
