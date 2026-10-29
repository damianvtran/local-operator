# Attestation — zh-CN / wire.host_speed / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 2 (en) / 2 (zh)

## Hash equality proof

- Source: prior row `e193473d58c0de1779bcefab06d0e9b1734ee493aea30ea827e4499510282b8c` = held `e193473d58c0de1779bcefab06d0e9b1734ee493aea30ea827e4499510282b8c` = current `e193473d58c0de1779bcefab06d0e9b1734ee493aea30ea827e4499510282b8c` — **equal**
- Target: held `ce807f4dce1514fea22cc05a38756663721ab0866192b9fff0de4dc948763214` = current `ce807f4dce1514fea22cc05a38756663721ab0866192b9fff0de4dc948763214` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`d5316700be4f79de5280b95f8ae96e48dfa66bef378dec38074fd3311969098d`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 2 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 1 declared `maxCells` key(s), 0 over budget

## Stratified sample

2/2 keys read (100.0%), including all 1 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
