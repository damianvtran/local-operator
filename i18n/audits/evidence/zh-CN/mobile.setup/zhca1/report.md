# Attestation — zh-CN / mobile.setup / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `anthropic/claude-sonnet-5-5`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 76 (en) / 76 (zh)

## Hash equality proof

- Source: prior row `6249b10379beb9b2433746416fa7baff061265a1b9315d47465140cd5fcf61e9` = held `6249b10379beb9b2433746416fa7baff061265a1b9315d47465140cd5fcf61e9` = current `6249b10379beb9b2433746416fa7baff061265a1b9315d47465140cd5fcf61e9` — **equal**
- Target: held `15333fe4a0bc853904649906e60d2335036e898d15b361845b97d67b99bea47a` = current `15333fe4a0bc853904649906e60d2335036e898d15b361845b97d67b99bea47a` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`be33c74a329196fdbe78f11f015848ff4f9593add57eba414163d4042807c811`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 76 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

22/76 keys read (28.9%), including all 11 messages over 120 characters and all 0 plural messages.

## Findings

- **ZHC-01 (minor)** `mobile.setup.balance` — The same English billing term is rendered three ways: 点数 in mobile.setup and relay.shell.usage.unit_credits, 积分 in tui.usage.paid.continue, and bare 余额 (dropping the credit noun) in the two speech errors. 点数 is the dominant rendering and matches 'Add credit'/'account credit'; standardise on 点数, keeping 余额 only as the compound 点数余额.
  - current: `自行前往https://console.radienthq.com/dashboard/tunnels充值；设置绝不会购买点数。之后重试。`
  - proposed: no change (this rendering is the reference side)
- **ZHC-01 (minor)** `mobile.setup.consent` — The same English billing term is rendered three ways: 点数 in mobile.setup and relay.shell.usage.unit_credits, 积分 in tui.usage.paid.continue, and bare 余额 (dropping the credit noun) in the two speech errors. 点数 is the dominant rendering and matches 'Add credit'/'account credit'; standardise on 点数, keeping 余额 only as the compound 点数余额.
  - current: `激活将为此Radient账户开通月费为USD {amount}的隧道订阅，预留手机地址，并安装和启动后台连接器。此操作不会购买点数。`
  - proposed: no change (this rendering is the reference side)
- **ZHC-01 (minor)** `mobile.setup.title_balance` — The same English billing term is rendered three ways: 点数 in mobile.setup and relay.shell.usage.unit_credits, 积分 in tui.usage.paid.continue, and bare 余额 (dropping the credit noun) in the two speech errors. 点数 is the dominant rendering and matches 'Add credit'/'account credit'; standardise on 点数, keeping 余额 only as the compound 点数余额.
  - current: `手机访问需要账户点数`
  - proposed: no change (this rendering is the reference side)

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
