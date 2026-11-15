# zh-CN / mobile.doctor — independent translation audit a1

- Verdict: **passed** — blocker 0 · major 0 · minor 2 · nit 2 (unresolved blocker/major: 0)
- Scope: complete NEW namespace, all 23 keys judged
- Revision: a3722be7dadacaf140a2587a0100bd3b26c2ab50 (branch feat/mobile-doctor, worktree .worktrees/mobile-doctor)
- Source sha256 (en): `0ca00b276fdea066228394e3fee336042b1e149c9e7df3ea31be11a24f0beb72`
- Target sha256 (zh-CN): `203ed8d5aae57db257226d7085a3b9787c55c6de188765571781cf0a81f78f87`
- Translator: i18n-translator subagent, session 398cdb26808f (English source drafted by coder session b0300e2c2c7f), anthropic/claude-opus-5-5, 2026-11-14
- Auditor: session 2c1a73eaa6cd, anthropic/claude-opus-5-5, 2026-11-14 — independent; edited no catalogue value

## Findings

### ZH-D1 — minor (open)
Keys: `mobile.doctor.daemon.slow`, `mobile.doctor.daemon.not_answering`

The sentence tells the reader what the phone app shows, but uses «无响应» while the phone app's own zh-CN label for this state is «未响应» (relay.chrome.session.subtitle.not_answering). A user comparing the report with the phone screen sees two different words.

Suggested fix: Use «未响应»: «…手机应用可能会将此电脑显示为未响应。» / «手机访问服务正在运行但未响应；手机应用会将此电脑显示为未响应。…»

### ZH-D2 — minor (open)
Keys: `mobile.doctor.daemon.not_running`

Translator question 6: «休眠» is the zh-CN OS term for Hibernate (Sleep is «睡眠» on both Windows and macOS) and this catalogue set already uses «休眠» for “dormant” (tui.app.parked.dormant). «显示为休眠» is understood, but «睡眠» is the precise everyday word for a computer that is asleep.

Suggested fix: «…在服务启动前，手机应用会将此电脑显示为睡眠状态。» — or whatever exact word the phone app's zh-CN UI shows for this state (that catalogue is outside this repo and was not checked).

### ZH-D3 — nit (accepted)
Keys: `mobile.doctor.login.ok`, `mobile.doctor.login.missing`

Translator question 3: no final 。. The style guide permits omitting 。 on short statuses, so this is acceptable; but doctor.text() prints one summary per line and every other check summary in this namespace ends in 。, so the report mixes both.

Suggested fix: Optional: «已登录Radient。» / «未登录Radient。» for consistency within the report.

### ZH-D4 — nit (accepted)
Keys: `mobile.doctor.daemon.ok`, `mobile.doctor.daemon.slow`

«响应用时{ms}毫秒» briefly parses as «响/应用/时» (应用 = app) in sentences that also say «手机应用».

Suggested fix: Optional: «响应耗时{ms}毫秒».


## Verified

- “phone service” consistently «手机访问服务»; «手机访问» matches mobile.setup / cli.mobile_help (24+2 shipped uses)
- «隧道» / «连接器» match mobile.setup; «此电脑» matches mobile.setup and cli.mobile_help.devices
- full-width punctuation, no space between CJK and Latin/digits, per style guide; {state}/{reason} left Latin
- counts take 个 classifier (files, tunnels); zh-CN plural category other only — tunnel.duplicates fine
- «打开文件过多» is a faithful plain rendering of EMFILE; gate_open calm and factual
- no command named; NFC verified

## Translator's open questions

 Q3 (no final 。 on login.ok/missing): acceptable per style guide, nit ZH-D3 for consistency. Q6 (asleep): «休眠» = Hibernate in zh-CN OS vocabulary — minor ZH-D2, suggest «睡眠». Q1/Q2/Q4/Q5 concern other locales.

## Method and limits

- Every key read against the English value and its context note; glossary `i18n/glossary.md` and `i18n/style-guides/zh-CN.md` applied.
- `audit.log`: placeholder-set parity, NFC and product-renderer output (`local_operator.i18n.runtime.render_message`) for every key with sample arguments, run with `.venv/bin/python`; 0 placeholder/NFC defects.
- LLM review, not native-speaker certification. Text renders only: no terminal, desktop or device frame captured. The phone app's own catalogue (separate repository) was not available, so its exact wording for the “asleep” state was not compared.
- Did not run generate.py or check.py (lane instruction).
