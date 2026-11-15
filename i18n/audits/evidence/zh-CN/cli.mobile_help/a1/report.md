# zh-CN / cli.mobile_help — independent translation audit a1

- Verdict: **passed** — blocker 0 · major 0 · minor 0 · nit 0 (unresolved blocker/major: 0)
- Scope: complete 17-key namespace at the new English source hash: 2 NEW keys (cli.mobile_help.doctor, cli.mobile_help.doctor_json) judged in full; the 15 previously audited keys re-read for consistency with the new keys and with mobile.doctor
- Revision: a3722be7dadacaf140a2587a0100bd3b26c2ab50 (branch feat/mobile-doctor, worktree .worktrees/mobile-doctor)
- Source sha256 (en): `3e1aea77751a6a1286cd0bc525163ac6e90da2f81fd19fff1fc23ae247133267`
- Target sha256 (zh-CN): `b07a34c8e22bceacb9335c2941d08bbdeb995a9c8bde4a30b47e4a07ee56752f`
- Translator: i18n-translator subagent, session 398cdb26808f (English source drafted by coder session b0300e2c2c7f), anthropic/claude-opus-5-5, 2026-11-14
- Auditor: session 2c1a73eaa6cd, anthropic/claude-opus-5-5, 2026-11-14 — independent; edited no catalogue value

## Findings

None.

## Verified

- doctor: «手机访问的只读诊断（可放心分享）» keeps “read-only” («只读», shipped term) and “safe to share” («可放心分享»); names no command
- doctor_json: «以JSON格式输出诊断信息», JSON verbatim
- new keys consistent with namespace terms (手机访问, 守护进程); existing 15 keys unchanged and consistent

## Translator's open questions

 None specific to this namespace for this locale.

## Method and limits

- Every key read against the English value and its context note; glossary `i18n/glossary.md` and `i18n/style-guides/zh-CN.md` applied.
- `audit.log`: placeholder-set parity, NFC and product-renderer output (`local_operator.i18n.runtime.render_message`) for every key with sample arguments, run with `.venv/bin/python`; 0 placeholder/NFC defects.
- LLM review, not native-speaker certification. Text renders only: no terminal, desktop or device frame captured. The phone app's own catalogue (separate repository) was not available, so its exact wording for the “asleep” state was not compared.
- Did not run generate.py or check.py (lane instruction).
