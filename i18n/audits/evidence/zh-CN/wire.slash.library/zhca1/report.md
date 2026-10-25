# Attestation — zh-CN / wire.slash.library / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 2 (en) / 2 (zh)

## Hash equality proof

- Source: prior row `0f8615f2ed9ffaf3aeb0288a4ff4b8fc8becd46017a931db9d4b3d9ec32fce46` = held `0f8615f2ed9ffaf3aeb0288a4ff4b8fc8becd46017a931db9d4b3d9ec32fce46` = current `0f8615f2ed9ffaf3aeb0288a4ff4b8fc8becd46017a931db9d4b3d9ec32fce46` — **equal**
- Target: held `f9834d2df36bbd0f74f27f1053b775a4a3ffef5d77e95bed88a1b527383c55a9` = current `f9834d2df36bbd0f74f27f1053b775a4a3ffef5d77e95bed88a1b527383c55a9` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`<absent>`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 2 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

1/2 keys read (50.0%), including all 0 messages over 120 characters and all 0 plural messages.

## Findings

- **ZHC-04 (minor)** `wire.slash.library.help` — tui.library consistently calls the feature 资料库; the slash help calls the same Library 资产库, and the (not-carried) desktop errors call it 资源库. One feature name is needed; 资料库 is the rendering the TUI title and the Librarian settings both use.
  - current: `浏览全局资产库`
  - proposed: `浏览全局资料库`

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
