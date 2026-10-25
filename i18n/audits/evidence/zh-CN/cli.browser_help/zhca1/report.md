# Attestation — zh-CN / cli.browser_help / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 20 (en) / 20 (zh)

## Hash equality proof

- Source: prior row `7fd47199bb8d443a0fbf7469d635453cd6f57bb7c14efec91862c6b5b0c1b1e6` = held `7fd47199bb8d443a0fbf7469d635453cd6f57bb7c14efec91862c6b5b0c1b1e6` = current `7fd47199bb8d443a0fbf7469d635453cd6f57bb7c14efec91862c6b5b0c1b1e6` — **equal**
- Target: held `7198093cde4f9758a1d9b4fc424c2fa05cf5845f752a697b7dbb8cae30b5f162` = current `7198093cde4f9758a1d9b4fc424c2fa05cf5845f752a697b7dbb8cae30b5f162` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`05160cd292e9c18125f62b7112e06cbb474556847b449ad63d3fcbac5d8c0908`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 20 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 1 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

6/20 keys read (30.0%), including all 1 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
