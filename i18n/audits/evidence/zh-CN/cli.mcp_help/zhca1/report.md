# Attestation — zh-CN / cli.mcp_help / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `anthropic/claude-sonnet-5-5`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 19 (en) / 19 (zh)

## Hash equality proof

- Source: prior row `3f5e0f11b186501e6a708ffaba4af294ebdb663175722b58c24143dbf0173417` = held `3f5e0f11b186501e6a708ffaba4af294ebdb663175722b58c24143dbf0173417` = current `3f5e0f11b186501e6a708ffaba4af294ebdb663175722b58c24143dbf0173417` — **equal**
- Target: held `6fd7827e5f1be31310b19cf1ebcacece8cfec1e9b488392e9220c17ebcdcf927` = current `6fd7827e5f1be31310b19cf1ebcacece8cfec1e9b488392e9220c17ebcdcf927` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`a8a42a82ada1777293a840e6d9280e855671394c0ee9cd5bd19cd8957d0f9bc0`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 19 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

4/19 keys read (21.1%), including all 0 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
