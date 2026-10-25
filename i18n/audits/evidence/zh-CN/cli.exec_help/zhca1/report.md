# Attestation — zh-CN / cli.exec_help / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 20 (en) / 20 (zh)

## Hash equality proof

- Source: prior row `1a44c9ea283ce8c4a9bb8ffab403e9cb36cf4a73fce832d91e79f355448b02eb` = held `1a44c9ea283ce8c4a9bb8ffab403e9cb36cf4a73fce832d91e79f355448b02eb` = current `1a44c9ea283ce8c4a9bb8ffab403e9cb36cf4a73fce832d91e79f355448b02eb` — **equal**
- Target: held `7540f682dfc96fbd6bcc7cc5f4e946db7e148c2b9fab0eac731456dfb7903908` = current `7540f682dfc96fbd6bcc7cc5f4e946db7e148c2b9fab0eac731456dfb7903908` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`c195f3c67a0669ce114d6498ccde81886964653ff3b5e1b1cd8051968080fcfd`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 20 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 6 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

11/20 keys read (55.0%), including all 7 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
