# Attestation — zh-CN / relay.shell / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 44 (en) / 44 (zh)

## Hash equality proof

- Source: prior row `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560` = held `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560` = current `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560` — **equal**
- Target: held `1a9c681b9175ce6c7189c1fac61fe887dc87084a44dcbfb0917ed053db92e6c2` = current `1a9c681b9175ce6c7189c1fac61fe887dc87084a44dcbfb0917ed053db92e6c2` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`16746ecd1537006992935bb2aa88fe39c46700d9e0341e7aacd565e2366349ea`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 44 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 9 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

10/44 keys read (22.7%), including all 0 messages over 120 characters and all 3 plural messages.

## Findings

- **ZHC-01 (minor)** `relay.shell.usage.unit_credits` — The same English billing term is rendered three ways: 点数 in mobile.setup and relay.shell.usage.unit_credits, 积分 in tui.usage.paid.continue, and bare 余额 (dropping the credit noun) in the two speech errors. 点数 is the dominant rendering and matches 'Add credit'/'account credit'; standardise on 点数, keeping 余额 only as the compound 点数余额.
  - current: `{count, plural, other {点数}}`
  - proposed: no change (this rendering is the reference side)
- **ZHC-02 (minor)** `relay.shell.usage.continue` — The same English sentence family ('Included quota used; credits will continue.') is 包含的配额已用尽；将继续使用积分 in tui.usage and 已用完内含额度；将继续使用点数 in relay.shell. 配额 is the dominant rendering (wire.settings, wire.slash.usage), so relay.shell should use 配额 and 包含的.
  - current: `已用完内含额度；将继续使用点数。`
  - proposed: `包含的配额已用尽；将继续使用点数。`

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
