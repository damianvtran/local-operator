# Attestation — zh-CN / tui.usage / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 21 (en) / 21 (zh)

## Hash equality proof

- Source: prior row `5aeaec254352896101bf2825ac03d06f74c95e3598cda23536c5fe28507a2f8b` = held `5aeaec254352896101bf2825ac03d06f74c95e3598cda23536c5fe28507a2f8b` = current `5aeaec254352896101bf2825ac03d06f74c95e3598cda23536c5fe28507a2f8b` — **equal**
- Target: held `dc96189139b3ef4bb58dfb29dc8fb2c7e6cd965553ed8c5784bfa03a5312261a` = current `dc96189139b3ef4bb58dfb29dc8fb2c7e6cd965553ed8c5784bfa03a5312261a` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`bf599f5f161087b890e1e24434a03209fee286b865173af6b578da3133b6ef19`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 21 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

6/21 keys read (28.6%), including all 0 messages over 120 characters and all 2 plural messages.

## Findings

- **ZHC-01 (minor)** `tui.usage.paid.continue` — The same English billing term is rendered three ways: 点数 in mobile.setup and relay.shell.usage.unit_credits, 积分 in tui.usage.paid.continue, and bare 余额 (dropping the credit noun) in the two speech errors. 点数 is the dominant rendering and matches 'Add credit'/'account credit'; standardise on 点数, keeping 余额 only as the compound 点数余额.
  - current: `包含的配额已用尽；将继续使用积分`
  - proposed: `包含的配额已用尽；将继续使用点数`
- **ZHC-02 (minor)** `tui.usage.paid.continue` — The same English sentence family ('Included quota used; credits will continue.') is 包含的配额已用尽；将继续使用积分 in tui.usage and 已用完内含额度；将继续使用点数 in relay.shell. 配额 is the dominant rendering (wire.settings, wire.slash.usage), so relay.shell should use 配额 and 包含的.
  - current: `包含的配额已用尽；将继续使用积分`
  - proposed: `包含的配额已用尽；将继续使用点数`

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
