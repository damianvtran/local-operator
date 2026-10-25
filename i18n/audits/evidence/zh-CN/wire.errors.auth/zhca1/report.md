# Attestation — zh-CN / wire.errors.auth / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 12 (en) / 12 (zh)

## Hash equality proof

- Source: prior row `9d48edaa61269239e132201224c1c7d6cf89ef198598f0ec6c99c1f3ade542f6` = held `9d48edaa61269239e132201224c1c7d6cf89ef198598f0ec6c99c1f3ade542f6` = current `9d48edaa61269239e132201224c1c7d6cf89ef198598f0ec6c99c1f3ade542f6` — **equal**
- Target: held `221b79ace66ae93293017769111a5a6a98c2075bbc2cb9c6f11a72faa72e2f98` = current `221b79ace66ae93293017769111a5a6a98c2075bbc2cb9c6f11a72faa72e2f98` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`<absent>`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 12 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

3/12 keys read (25.0%), including all 0 messages over 120 characters and all 0 plural messages.

## Findings

- **ZHC-09 (nit)** `wire.errors.auth.store_unreadable` — Here 'census' is the enumeration of configured providers that the credential store is needed for; 普查 (a statistical census) is not the idiomatic software term. 清点 keeps the counting sense without the demographic register.
  - current: `无法读取凭据存储——提供商普查需要读取它。`
  - proposed: `无法读取凭据存储——提供商清点需要读取它。`

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
