# Attestation — zh-CN / wire.errors.media / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `openai/gpt-6.1-sol`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 20 (en) / 20 (zh)

## Hash equality proof

- Source: prior row `1c109f63b8d3ce00657a47836199ce8ff924bb8e4bd999fda9f1d64f760a823b` = held `1c109f63b8d3ce00657a47836199ce8ff924bb8e4bd999fda9f1d64f760a823b` = current `1c109f63b8d3ce00657a47836199ce8ff924bb8e4bd999fda9f1d64f760a823b` — **equal**
- Target: held `1c51112e65771daa0fceacfd53e571ba6d5f8fad7f4f31903b97b2df87791b64` = current `1c51112e65771daa0fceacfd53e571ba6d5f8fad7f4f31903b97b2df87791b64` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`<absent>`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 20 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 0 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

5/20 keys read (25.0%), including all 0 messages over 120 characters and all 0 plural messages.

## Findings

- **ZHC-01 (minor)** `wire.errors.media.speech_radient_credit_low` — The same English billing term is rendered three ways: 点数 in mobile.setup and relay.shell.usage.unit_credits, 积分 in tui.usage.paid.continue, and bare 余额 (dropping the credit noun) in the two speech errors. 点数 is the dominant rendering and matches 'Add credit'/'account credit'; standardise on 点数, keeping 余额 only as the compound 点数余额.
  - current: `Radient余额不足以生成语音。在Radient Console充值后继续。`
  - proposed: `Radient点数余额不足，无法生成语音。请在Radient Console充值后继续。`
- **ZHC-01 (minor)** `wire.errors.media.speech_vendor_credit_low` — The same English billing term is rendered three ways: 点数 in mobile.setup and relay.shell.usage.unit_credits, 积分 in tui.usage.paid.continue, and bare 余额 (dropping the credit noun) in the two speech errors. 点数 is the dominant rendering and matches 'Add credit'/'account credit'; standardise on 点数, keeping 余额 only as the compound 点数余额.
  - current: `{vendor}余额不足以生成语音。在{vendor}充值后继续。`
  - proposed: `{vendor}点数余额不足，无法生成语音。请在{vendor}充值后继续。`

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
