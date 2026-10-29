# Attestation — zh-CN / cli.serve_help / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 5 (en) / 5 (zh)

## Hash equality proof

- Source: prior row `bd0b51271edc67c23978ca23688a9af1ef94a5623b0e86a5e3bbb418a5b243f0` = held `bd0b51271edc67c23978ca23688a9af1ef94a5623b0e86a5e3bbb418a5b243f0` = current `bd0b51271edc67c23978ca23688a9af1ef94a5623b0e86a5e3bbb418a5b243f0` — **equal**
- Target: held `c5079f99a6180ae4edb05b4dafd89fc059cc498dde31505fa45158a7e262c36e` = current `c5079f99a6180ae4edb05b4dafd89fc059cc498dde31505fa45158a7e262c36e` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`d569b1252c7965cc07456c6dba086187237c609a6d9f3021a206a07bf731c770`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 5 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

3/5 keys read (60.0%), including all 2 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
