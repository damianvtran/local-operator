# Attestation — zh-CN / cli.global_help / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 17 (en) / 17 (zh)

## Hash equality proof

- Source: prior row `c6e91dda89a9b792587f4ac9c594e188b78b8cd529fa1acf51ca783422c3152f` = held `c6e91dda89a9b792587f4ac9c594e188b78b8cd529fa1acf51ca783422c3152f` = current `c6e91dda89a9b792587f4ac9c594e188b78b8cd529fa1acf51ca783422c3152f` — **equal**
- Target: held `432b0a321f314fff0b66799cc01f0a4e78c013937405a53d694a618bb1e014ea` = current `432b0a321f314fff0b66799cc01f0a4e78c013937405a53d694a618bb1e014ea` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`27388bf24447053aa71a26a5457827a419daf7046d3169101d28eda893b358c7`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 17 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

10/17 keys read (58.8%), including all 7 messages over 120 characters and all 0 plural messages.

## Findings

- **ZHC-06 (nit)** `cli.global_help.run_in` — Everywhere else the person is 操作员 and the product is Local Operator. '运行operator的工作目录' reads as an untranslated token; name the product or the role.
  - current: `运行operator的工作目录。  必须是有效目录。`
  - proposed: `运行Local Operator的工作目录。  必须是有效目录。`

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
