# Attestation — zh-CN / tui.review / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `anthropic/claude-sonnet-5-5`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 73 (en) / 73 (zh)

## Hash equality proof

- Source: prior row `39c168340e3bac3f47e3a94358026e10b99806ce4147427e6fd7c70334e00a61` = held `39c168340e3bac3f47e3a94358026e10b99806ce4147427e6fd7c70334e00a61` = current `39c168340e3bac3f47e3a94358026e10b99806ce4147427e6fd7c70334e00a61` — **equal**
- Target: held `cfe7527cc83728f8177d4ac3e47e76f71d467c5593e0b53451e5e04828028ae6` = current `cfe7527cc83728f8177d4ac3e47e76f71d467c5593e0b53451e5e04828028ae6` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`b879498a3ba2debfac6ec65a640a75b8d53e71073a1d32676dc41f7325dbc202`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 73 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

16/73 keys read (21.9%), including all 0 messages over 120 characters and all 5 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
