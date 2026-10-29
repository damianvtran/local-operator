# Attestation — zh-CN / tui.library / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `anthropic/claude-sonnet-5-5`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 108 (en) / 108 (zh)

## Hash equality proof

- Source: prior row `5a389b6a7d8de06621462f2685eff6dc95cd4cd6584d35dd6c26a0a19ad50b11` = held `5a389b6a7d8de06621462f2685eff6dc95cd4cd6584d35dd6c26a0a19ad50b11` = current `5a389b6a7d8de06621462f2685eff6dc95cd4cd6584d35dd6c26a0a19ad50b11` — **equal**
- Target: held `c8db59a6cdc011050c9dd8944eafad703a4f4723095a5e069a72cc8aac0af813` = current `c8db59a6cdc011050c9dd8944eafad703a4f4723095a5e069a72cc8aac0af813` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`<absent>`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 108 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

20/108 keys read (18.5%), including all 0 messages over 120 characters and all 2 plural messages.

## Findings

- **ZHC-04 (minor)** `tui.library.title` — tui.library consistently calls the feature 资料库; the slash help calls the same Library 资产库, and the (not-carried) desktop errors call it 资源库. One feature name is needed; 资料库 is the rendering the TUI title and the Librarian settings both use.
  - current: `资料库`
  - proposed: no change (this rendering is the reference side)
- **ZHC-04 (minor)** `tui.library.loading` — tui.library consistently calls the feature 资料库; the slash help calls the same Library 资产库, and the (not-carried) desktop errors call it 资源库. One feature name is needed; 资料库 is the rendering the TUI title and the Librarian settings both use.
  - current: `正在读取资料库……`
  - proposed: no change (this rendering is the reference side)

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
