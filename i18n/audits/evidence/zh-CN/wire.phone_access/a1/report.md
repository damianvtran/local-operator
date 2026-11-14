# i18n audit a1 — zh-CN / wire.phone_access (branch fix/tunnel-remedies-auto-up)

Head audited: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (worktree .worktrees/tunnel-remedies-auto-up). Scope: 1fce7ecb1f5f~1..1fce7ecb1f5f (values introduced in 1fce7ecb1f5f). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session ede889388023 (node lop-remote-node-2); not the translating i18n-translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session ede889388023: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (values introduced in commit 1fce7ecb1f5f; English drafted by coder session b0300e2c2c7f); model anthropic/claude-opus-5-5 (as stated by the delegation brief; not independently verified); 2026-11-14.
Prior audit: none — new namespace; all 4 keys audited.

**Verdict: passed** — 0 unresolved blocker/major; 2 finding(s) total (ZH-04 minor, ZH-05 minor).

Source en sha256 2979b226db50db768fcad68f5f7376854f2185b08e6b0af08eb8e72bdb470f58; target sha256 a5ca3ff66513d0d487bd4f6084d23d63fbcf7a5db3ca5a0d096a291ab087feed; keys 4 (4 new, all judged).

Read: docs/I18N.md L260-310, i18n/glossary.md (no tunnel/billing/credit row; §1 keeps Radient, Local Operator, URLs, commands as-is), i18n/style-guides/zh-CN.md, en + context sidecar for every new key, corpus usage of the key terms (audit.log), and every new value in full.

Safety meanings checked on every new key: no terminal command named (only `/mobile enable`, verbatim, in mobile.setup.login_offer); nothing-is-charged / never activates billing or spends credit / setup never purchases credit / price shown before billing all survive; {url} and the console URL are byte-identical.

## Per-key judgement

| key | value | judgement |
|---|---|---|
| wire.phone_access.title | 手机访问 | OK; 37 corpus uses. |
| wire.phone_access.description | … | ZH-04 (minor). |
| wire.phone_access.auto_setup.label | 登录后重新连接手机访问 | OK; short, no 。. |
| wire.phone_access.auto_setup.help | … | Safety meaning intact ('绝不会激活计费或消耗点数'); 点数 matches mobile.setup credit. ZH-05 (minor). |

## Findings

| ID | severity | key | issue | suggestion |
|---|---|---|---|---|
| ZH-04 | minor | wire.phone_access.description | Calque: '这台电脑如何为手机应用保持其Radient隧道正常工作。' — a bare 如何-clause closed with 。 reads as an indirect-question fragment, not a settings description. | '设置这台电脑如何为手机应用保持Radient隧道正常工作。更改将在下次登录时生效。' |
| ZH-05 | minor | wire.phone_access.auto_setup.help | '如有任何价格或点数不足，都会改为向你展示' — the coordination parses as 'any price or credit is insufficient'; meaning is recoverable and the never-activates/never-spends clause is intact. | '…此操作绝不会激活计费或消耗点数；如需付费或点数不足，会改为向你说明。' |

Minor/nit findings do not block; they are offered to the translator for an optional follow-up. The auditor did not edit any value.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used.

## Ledger note

history-sha before append: 4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945. Receipt appended with `ledger.py append --dry-run` then `--apply` (dryrun.log). generate.py NOT run (coordinator regenerates).
