# Attestation — zh-CN / wire.errors.desktop / zhca1

- **Verdict: not_carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 84 (en) / 84 (zh)

## Hash equality proof

- Source: prior row `30312c2052a41da6f580f00dce8daf9babbc0b25c0462c1f2658817cd51f391c` = held `30312c2052a41da6f580f00dce8daf9babbc0b25c0462c1f2658817cd51f391c` = current `9a164be42a80d9bf0f60ad2a9f1d76fd1601855a4d19ffac3bf35d6bef11f76d` — **DIFFERS**
- Target: held `cf2fbad491db3452ce9ef80f9764e1e4937a352ed66eb2f99cc1d151e2f2daf1` = current `a8cb0db6bbc8a33ef041c082d68b9c06dd0b3168a303e447e0f40e0334fc89f4` — **DIFFERS** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`<absent>`)
- Whole target file byte-identical to held candidate: **false**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 84 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 1 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

Not performed: the namespace is not attestable.

## Change since the held candidate

- en keys: 83 -> 84; zh keys: 83 -> 84
- added in en: wire.errors.desktop.fast_mode_unsupported
- added in zh: wire.errors.desktop.fast_mode_unsupported
- removed in zh: none
- changed in zh: none

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. No attestation or semantic read is made: the hashes differ and a fresh audit is required.
