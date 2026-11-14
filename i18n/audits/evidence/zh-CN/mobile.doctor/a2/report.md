# zh-CN / mobile.doctor — independent translation audit a2 (remediation delta)

- Verdict: **passed** — blocker 0 · major 0 · minor 0 · nit 0 new (unresolved blocker/major: 0)
- Scope: round 2, remediation delta only — 3 changed key(s): `mobile.doctor.daemon.slow`, `mobile.doctor.daemon.not_answering`, `mobile.doctor.daemon.not_running`; the other keys were confirmed byte-identical to the a1-audited values and not re-audited
- Revision: a3722be7dadacaf140a2587a0100bd3b26c2ab50 + uncommitted working-tree remediation (branch feat/mobile-doctor, worktree .worktrees/mobile-doctor, node lop-remote-node-2)
- Source sha256 (en): `0ca00b276fdea066228394e3fee336042b1e149c9e7df3ea31be11a24f0beb72`; context `c6baa99bc539ef2d2e03bf7b6b11e83edac37139eede8db82034f5fad98d4614`
- Target sha256 (zh-CN): `192643761971f366519342cfacb4261deee88f55e6a20f4d4ea5c5e8cf4a4024` (a1 audited `203ed8d5aae57db257226d7085a3b9787c55c6de188765571781cf0a81f78f87`)
- Prior audit: a1 `i18n/audits/evidence/zh-CN/mobile.doctor/a1/record.json` — passed (auditor session 2c1a73eaa6cd)
- Translator: i18n-translator subagent, session 8dedc962f71a (remediation of a1 findings); original values by session 398cdb26808f (English source drafted by coder session b0300e2c2c7f), anthropic/claude-opus-5-5, 2026-11-14
- Auditor: session 24933ff37418, anthropic/claude-opus-5-5, 2026-11-14 — independent of the translators and the round-1 auditor; edited no catalogue value. Same model as the translator (disclosed).

## Prior findings

| id | severity | round-2 status |
|---|---|---|
| ZH-D1 | minor | resolved — daemon.slow, daemon.not_answering now «未响应» |
| ZH-D2 | minor | resolved — daemon.not_running now «睡眠状态» |
| ZH-D3 | nit | carried (accepted in a1; keys unchanged) |
| ZH-D4 | nit | carried (accepted in a1; «响应用时» unchanged in daemon.ok/daemon.slow) |

## New findings

None.

## Changed keys

| key | English | a1 value | a2 value | judgement |
|---|---|---|---|---|
| mobile.doctor.daemon.slow | The phone service is running but took {ms} ms to answer, which the phone app may show as not answering. | 手机访问服务正在运行，但响应用时{ms}毫秒，手机应用可能会将此电脑显示为无响应。 | 手机访问服务正在运行，但响应用时{ms}毫秒，手机应用可能会将此电脑显示为未响应。 | ok — Resolves ZH-D1. «显示为未响应» now uses the phone app's own label «未响应» (relay.chrome.session.subtitle.not_answering, also relay.chrome.session.degraded). The unchanged «响应用时» is the accepted nit ZH-D4; not re-raised. |
| mobile.doctor.daemon.not_answering | The phone service is running but not answering; the phone app shows this computer as not answering. It may have run out of files. | 手机访问服务正在运行但无响应；手机应用会将此电脑显示为无响应。该服务可能已达到打开文件数上限。 | 手机访问服务正在运行但未响应；手机应用会将此电脑显示为未响应。该服务可能已达到打开文件数上限。 | ok — Resolves ZH-D1. «正在运行但未响应；…显示为未响应» — both occurrences switched, mirroring English “not answering … not answering”; grammatical and matches the app label. |
| mobile.doctor.daemon.not_running | The phone service is set up but not running; the phone app shows this computer as asleep until it starts. | 手机访问服务已设置但未运行；在服务启动前，手机应用会将此电脑显示为休眠。 | 手机访问服务已设置但未运行；在服务启动前，手机应用会将此电脑显示为睡眠状态。 | ok — Resolves ZH-D2. «显示为睡眠状态» uses 睡眠 (OS term for Sleep) instead of 休眠 (Hibernate; used for “dormant” in tui.app.parked.dormant). «状态» makes «显示为…» idiomatic. The phone app's own zh-CN 'asleep' label lives outside this repo and was not compared (same limit as a1). |

## Verified

- placeholder sets identical to English for every changed key; NFC; key set and order identical (audit.log)
- only the 3 remediation key(s) differ from the a1-audited file; every other key byte-identical
- each changed value matches the a1 suggested fix in meaning; no command named; calm factual tone kept
- «未响应» = zh-CN relay.chrome.session.subtitle.not_answering (phone app label in this repo); full-width punctuation per style guide

## Ledger

- `i18n/audit-history/zh-CN/mobile.doctor.json`: history sha `d33efe9b8b788d0af8e799deae2778eb6ded2c4f0afac3112d914f8084d1936f` → `29061148b4d724e3404b7ef17ee00ecb61deab0e8f5c30eb4171c7c98585df71` (append --dry-run then --apply, exit 0; dryrun.log)

## Method and limits

- Independent LLM review on anthropic/claude-opus-5-5, not native-speaker certification; translator and auditor ran the same model (disclosed).
- Text renders via the product renderer only (audit.log); no terminal, desktop or device frame.
- Round-2 scope: only the remediation delta was judged; unchanged keys were confirmed byte-identical to the a1-audited values, not re-audited.
- generate.py / check.py not run by this lane; nothing committed or pushed.
- The phone app's own zh-CN 'asleep' label is in a separate repository and was not compared.
