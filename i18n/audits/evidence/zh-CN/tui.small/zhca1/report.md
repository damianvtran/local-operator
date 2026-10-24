# Attestation — zh-CN / tui.small / zhca1

- **Verdict: not_carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 137 (en) / 137 (zh)

## Hash equality proof

- Source: prior row `7cd74d45ada43efceba846a9c5d29a0aa90e0567e1f137778300e1742a6adfec` = held `7cd74d45ada43efceba846a9c5d29a0aa90e0567e1f137778300e1742a6adfec` = current `16d49c4ba2a514bbde7da494443abade26a3b28c0389e6bca124d00aa7d0ed75` — **DIFFERS**
- Target: held `a05ee9e08e560ddb5bbaf999875e20e53a9eb7e58d73ffbc1712c119ac8a92d7` = current `5b7e05905402a7713dee3dd0ffa3df8048bae23e6b18108cfcec1c6cd4567ca6` — **DIFFERS** (the prior ledger row does not record a target hash)
- Context sidecar: DIFFERS (`1e3d6785f78126b8389b2acbcc081d5bf782fbaf744114e47611fbff3c45dde1`)
- Whole target file byte-identical to held candidate: **false**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 137 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 16 declared `maxCells` key(s), 0 over budget

## Stratified sample

Not performed: the namespace is not attestable.

## Change since the held candidate

- en keys: 121 -> 137; zh keys: 121 -> 137
- added in en: tui.small.spend.no_charge, tui.small.spend.unpriced_failed_wide, tui.small.spend.unpriced_failed, tui.small.spend.unpriced_failed_short, tui.small.spend.unpriced_failed_mixed, tui.small.spend.unpriced_failed_mixed_mid, tui.small.spend.unpriced_failed_mixed_short, tui.small.spend.unpriced_cancelled_wide, tui.small.spend.unpriced_cancelled, tui.small.spend.unpriced_cancelled_short, tui.small.spend.unpriced_cancelled_shortest, tui.small.spend.unpriced_ok, tui.small.spend.unpriced_ok_short, tui.small.spend.unpriced_partial, tui.small.spend.unpriced_partial_short, tui.small.spend.unpriced_partial_shortest
- added in zh: tui.small.spend.no_charge, tui.small.spend.unpriced_failed_wide, tui.small.spend.unpriced_failed, tui.small.spend.unpriced_failed_short, tui.small.spend.unpriced_failed_mixed, tui.small.spend.unpriced_failed_mixed_mid, tui.small.spend.unpriced_failed_mixed_short, tui.small.spend.unpriced_cancelled_wide, tui.small.spend.unpriced_cancelled, tui.small.spend.unpriced_cancelled_short, tui.small.spend.unpriced_cancelled_shortest, tui.small.spend.unpriced_ok, tui.small.spend.unpriced_ok_short, tui.small.spend.unpriced_partial, tui.small.spend.unpriced_partial_short, tui.small.spend.unpriced_partial_shortest
- removed in zh: none
- changed in zh: none

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. No attestation or semantic read is made: the hashes differ and a fresh audit is required.
