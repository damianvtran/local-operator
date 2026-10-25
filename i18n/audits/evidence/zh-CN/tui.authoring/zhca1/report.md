# Attestation — zh-CN / tui.authoring / zhca1

- **Verdict: not_carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 49 (en) / 49 (zh)

## Hash equality proof

- Source: prior row `d1a49c668d1049700396c33b7af3c905d0f02b30a2cbf55aab7ea91b40c36cb4` = held `d1a49c668d1049700396c33b7af3c905d0f02b30a2cbf55aab7ea91b40c36cb4` = current `7431acf3288cc3766cf37135badb29511737120ab52fc518bf514311ee488d13` — **DIFFERS**
- Target: held `d233ce0bfe19bfbc60495f39a87ea9a6165578bf3efda85b9ca94c485320d3e9` = current `b26f3ab6cddbf93beff849976a2f453feaa4ee4a828474a2f30e37e9410c7470` — **DIFFERS** (the prior ledger row does not record a target hash)
- Context sidecar: DIFFERS (`df98bba4457c1080b3941e9a24a04a52e360d11d2a07bdda992861c6cb54928c`)
- Whole target file byte-identical to held candidate: **false**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 49 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

Not performed: the namespace is not attestable.

## Change since the held candidate

- en keys: 40 -> 49; zh keys: 40 -> 49
- added in en: tui.authoring.promote_failed, tui.authoring.promote_mixed_plan, tui.authoring.promote_plan, tui.authoring.promote_receipt_created, tui.authoring.promote_receipt_keeps, tui.authoring.promote_receipt_replaced, tui.authoring.promote_replace_plan, tui.authoring.promote_usage, tui.authoring.scope_conversation
- added in zh: tui.authoring.promote_failed, tui.authoring.promote_mixed_plan, tui.authoring.promote_plan, tui.authoring.promote_receipt_created, tui.authoring.promote_receipt_keeps, tui.authoring.promote_receipt_replaced, tui.authoring.promote_replace_plan, tui.authoring.promote_usage, tui.authoring.scope_conversation
- removed in zh: none
- changed in zh: none

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. No attestation or semantic read is made: the hashes differ and a fresh audit is required.
