# Attestation — zh-CN / cli.services_help / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 9 (en) / 9 (zh)

## Hash equality proof

- Source: prior row `76aca39fb3bfe46be1d31fbfde2176ee525b7ac650961104894a894f71751935` = held `76aca39fb3bfe46be1d31fbfde2176ee525b7ac650961104894a894f71751935` = current `76aca39fb3bfe46be1d31fbfde2176ee525b7ac650961104894a894f71751935` — **equal**
- Target: held `eb5136dd88d32d513168cad32d8a1eb0bdd53db079473df34a07a994a9196766` = current `eb5136dd88d32d513168cad32d8a1eb0bdd53db079473df34a07a994a9196766` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`cc6104f97fbb0faac19232252063cdb7effaa2ffb4bbe2d4b65c2c749a8fe1e1`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 9 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 2 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

7/9 keys read (77.8%), including all 5 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
