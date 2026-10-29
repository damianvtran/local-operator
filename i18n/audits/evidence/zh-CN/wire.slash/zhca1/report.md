# Attestation — zh-CN / wire.slash / zhca1

- **Verdict: withheld**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `failed`, 0 finding(s), from the held-candidate ledger row
- Key count: 52 (en) / 52 (zh)

## Hash equality proof

- Source: prior row `b20f4be0dae6446424862ba227313b6c3384bd115a033edc57c0a289bfe74202` = held `b20f4be0dae6446424862ba227313b6c3384bd115a033edc57c0a289bfe74202` = current `b20f4be0dae6446424862ba227313b6c3384bd115a033edc57c0a289bfe74202` — **equal**
- Target: held `a3de10768a538e054afc35ca0aff44c29bbaa042e10e10a8913781d19de87b11` = current `a3de10768a538e054afc35ca0aff44c29bbaa042e10e10a8913781d19de87b11` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`32b3475a5808048a6eeaae462fee0feb55405fca77026e41b9609a516d74a215`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 52 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 1 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

Not performed: the namespace is not attestable.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. No attestation is made: the bytes match the held candidate but the prior audit row failed.
