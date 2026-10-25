# Attestation — zh-CN / tui.projects / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `anthropic/claude-sonnet-5-5`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 97 (en) / 97 (zh)

## Hash equality proof

- Source: prior row `aeb473eb7de44143822962b44b6be239d63f930710e754ce5205781ca4b4ae91` = held `aeb473eb7de44143822962b44b6be239d63f930710e754ce5205781ca4b4ae91` = current `aeb473eb7de44143822962b44b6be239d63f930710e754ce5205781ca4b4ae91` — **equal**
- Target: held `e24bd6988479039fd71cdeb82aef4c0cf4223f244c0da9f48e8f5f76ef99bf74` = current `e24bd6988479039fd71cdeb82aef4c0cf4223f244c0da9f48e8f5f76ef99bf74` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`fa39ad8f21674a42e5218698d11396a08d47e2f82297467ad17bb1ab292bb1fe`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 97 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

18/97 keys read (18.6%), including all 0 messages over 120 characters and all 1 plural messages.

## Findings

- **ZHC-07 (nit)** `tui.projects.form.readback_hint` — The zh-CN style guide requires full-width punctuation in prose. These are the only two Chinese-prose strings in the carried corpus that use an ASCII colon and a following space.
  - current: `{label}提示: {hint}`
  - proposed: `{label}提示：{hint}`
- **ZHC-07 (nit)** `tui.projects.form.readback_error` — The zh-CN style guide requires full-width punctuation in prose. These are the only two Chinese-prose strings in the carried corpus that use an ASCII colon and a following space.
  - current: `{label}错误: {error}`
  - proposed: `{label}错误：{error}`
- **ZHC-10 (nit)** `tui.projects.render.open_over_total` — {open}项未完成 / {total}项 is 26 cells against the English 21, in a narrow summary row with no declared maxCells. The shorter form keeps the open/total meaning and fits; flag for a rendered check rather than assert a break.
  - current: `{open}项未完成 / {total}项`
  - proposed: `{open}/{total}项未完成`

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
