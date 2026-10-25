# Attestation — zh-CN / tui.import / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `anthropic/claude-sonnet-5-5`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 87 (en) / 87 (zh)

## Hash equality proof

- Source: prior row `932e0f87210ff80950346c684afb000907610c9ff35388125977182120b136c2` = held `932e0f87210ff80950346c684afb000907610c9ff35388125977182120b136c2` = current `932e0f87210ff80950346c684afb000907610c9ff35388125977182120b136c2` — **equal**
- Target: held `1020564406e7807119d0fb10c7a5bb6e626ad451e8740774fbd4f77f412bf8e6` = current `1020564406e7807119d0fb10c7a5bb6e626ad451e8740774fbd4f77f412bf8e6` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`<absent>`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 87 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

24/87 keys read (27.6%), including all 4 messages over 120 characters and all 11 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
