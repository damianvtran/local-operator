# Attestation — zh-CN / cli.aida_help / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 4 (en) / 4 (zh)

## Hash equality proof

- Source: prior row `31e8445b6d8a9540ef524988b0fca3efca111362188f2ba41d6f857499828423` = held `31e8445b6d8a9540ef524988b0fca3efca111362188f2ba41d6f857499828423` = current `31e8445b6d8a9540ef524988b0fca3efca111362188f2ba41d6f857499828423` — **equal**
- Target: held `15b992210c3fe05c8265df5ba74f4c7d35ca155a3443b3a1e04917f69e0fda70` = current `15b992210c3fe05c8265df5ba74f4c7d35ca155a3443b3a1e04917f69e0fda70` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`9aea5c2fd1bc65c766b23c1f45116042ba0968b8242fae988695e922ae4083e9`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 4 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

2/4 keys read (50.0%), including all 1 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
