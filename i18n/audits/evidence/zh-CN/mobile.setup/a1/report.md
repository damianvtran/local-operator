# i18n audit a1 — zh-CN / mobile.setup (branch fix/tunnel-remedies-auto-up)

Head audited: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (worktree .worktrees/tunnel-remedies-auto-up). Scope: 1fce7ecb1f5f~1..1fce7ecb1f5f (values introduced in 1fce7ecb1f5f). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session ede889388023 (node lop-remote-node-2); not the translating i18n-translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session ede889388023: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (values introduced in commit 1fce7ecb1f5f; English drafted by coder session b0300e2c2c7f); model anthropic/claude-opus-5-5 (as stated by the delegation brief; not independently verified); 2026-11-14.
Prior audit: zh-CN mobile.setup last row r1 (passed, source f38364c1…); this round covers the 7 new keys and re-reads the unchanged keys for consistency only (git diff shows the only removed lines are trailing-comma rewrites).

**Verdict: passed** — 0 unresolved blocker/major; 0 finding(s) total (none).

Source en sha256 6883f92e94dfcec4be0af6041a0531c974fbf18294f7094a132cc555a2ad2065; target sha256 c696f2ea4828c70b8f8e542aa8beba0e7a41de1169f0138627500a04f0f076fc; keys 86 (7 new, all judged).

Read: docs/I18N.md L260-310, i18n/glossary.md (no tunnel/billing/credit row; §1 keeps Radient, Local Operator, URLs, commands as-is), i18n/style-guides/zh-CN.md, en + context sidecar for every new key, corpus usage of the key terms (audit.log), and every new value in full.

Safety meanings checked on every new key: no terminal command named (only `/mobile enable`, verbatim, in mobile.setup.login_offer); nothing-is-charged / never activates billing or spends credit / setup never purchases credit / price shown before billing all survive; {url} and the console URL are byte-identical.

## Per-key judgement

| key | value | judgement |
|---|---|---|
| mobile.setup.disabled | … | OK; URL verbatim; '设置不会覆盖这一选择' keeps no-override meaning. |
| mobile.setup.billing_lapsed | … | OK; '设置绝不会购买点数' identical to the audited mobile.setup.balance clause. |
| mobile.setup.reconnect | … | OK; '不会产生任何费用'. |
| mobile.setup.network | … | OK; '无需任何操作' — asks nothing. |
| mobile.setup.radient_unavailable | … | OK. |
| mobile.setup.reconnecting | … | OK; '不会产生任何费用' kept; '一两分钟内会自动完成' is a natural rendering of 'clears by itself'. |
| mobile.setup.login_offer | 想用手机应用访问这台电脑？运行/mobile enable。开始计费前，任何月费都会先展示给你。 | Open question answered: accepted. A question opener is idiomatic, inviting and not pushy in zh-CN product copy; /mobile enable verbatim; price-before-billing clause intact; 你 per guide. |

## Findings

None.

Minor/nit findings do not block; they are offered to the translator for an optional follow-up. The auditor did not edit any value.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used.

## Ledger note

history-sha before append: 529a6c444a46df06ffd1668bec5182baea855d08a05c53a9f9dcdcce8a887663. Receipt appended with `ledger.py append --dry-run` then `--apply` (dryrun.log). generate.py NOT run (coordinator regenerates).
