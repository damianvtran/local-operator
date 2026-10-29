# Attestation — zh-CN / cli.login_status_help / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 2 (en) / 2 (zh)

## Hash equality proof

- Source: prior row `17ef2629c0ad8a20c7e0a939d8fb38e594936e0c5dd47a3a1600300c1bf3f0f3` = held `17ef2629c0ad8a20c7e0a939d8fb38e594936e0c5dd47a3a1600300c1bf3f0f3` = current `17ef2629c0ad8a20c7e0a939d8fb38e594936e0c5dd47a3a1600300c1bf3f0f3` — **equal**
- Target: held `c71290053ecda9f0dbb714c6597c7abb1192786647a77fa2067862deb57aca8e` = current `c71290053ecda9f0dbb714c6597c7abb1192786647a77fa2067862deb57aca8e` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`d64a6e4f6338cf33a83cde2651694babaacab4ef3cd4a6456faddeb5aecb3963`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 2 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

1/2 keys read (50.0%), including all 0 messages over 120 characters and all 0 plural messages.

## Findings

None in this namespace.

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
