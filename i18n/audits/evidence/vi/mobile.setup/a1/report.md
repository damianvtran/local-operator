# i18n audit a1 — vi / mobile.setup (branch fix/tunnel-remedies-auto-up)

Head audited: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (worktree .worktrees/tunnel-remedies-auto-up). Scope: 1fce7ecb1f5f~1..1fce7ecb1f5f (values introduced in 1fce7ecb1f5f). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session ede889388023 (node lop-remote-node-2); not the translating i18n-translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session ede889388023: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (values introduced in commit 1fce7ecb1f5f; English drafted by coder session b0300e2c2c7f); model anthropic/claude-opus-5-5 (as stated by the delegation brief; not independently verified); 2026-11-14.
Prior audit: vi mobile.setup last row r1 (passed, source f38364c1…); this round covers the 7 new keys and re-reads the unchanged keys for consistency only (git diff shows the only removed lines are trailing-comma rewrites).

**Verdict: passed** — 0 unresolved blocker/major; 0 finding(s) total (none).

Source en sha256 6883f92e94dfcec4be0af6041a0531c974fbf18294f7094a132cc555a2ad2065; target sha256 2fae6ac3321a41639515cfbb6cd451ee724c3c567d94472ab983cf080c1b44ca; keys 86 (7 new, all judged).

Read: docs/I18N.md L260-310, i18n/glossary.md (no tunnel/billing/credit row; §1 keeps Radient, Local Operator, URLs, commands as-is), i18n/style-guides/vi.md, en + context sidecar for every new key, corpus usage of the key terms (audit.log), and every new value in full.

Safety meanings checked on every new key: no terminal command named (only `/mobile enable`, verbatim, in mobile.setup.login_offer); nothing-is-charged / never activates billing or spends credit / setup never purchases credit / price shown before billing all survive; {url} and the console URL are byte-identical.

## Per-key judgement

| key | value | judgement |
|---|---|---|
| mobile.setup.disabled | … | OK; URL verbatim; 'thiết lập sẽ không ghi đè lựa chọn đó' keeps the no-override safety meaning. |
| mobile.setup.billing_lapsed | … | OK; 'thiết lập không bao giờ tự mua tín dụng' identical to the audited mobile.setup.balance clause. |
| mobile.setup.reconnect | … | OK; 'không mất phí' = nothing is charged; idiomatic. |
| mobile.setup.network | … | OK; asks for no action ('bạn không cần làm gì'); one 'bạn' is natural here. |
| mobile.setup.radient_unavailable | … | OK; reassuring, no action requested. |
| mobile.setup.reconnecting | … | OK; 'Không mất phí' kept; 'thông báo này sẽ tự biến mất sau một hai phút' renders 'this clears by itself in a minute or two' naturally. |
| mobile.setup.login_offer | … | OK; /mobile enable verbatim, no other command; price-before-billing clause intact ('Mọi mức giá hằng tháng đều được hiển thị cho bạn trước khi bắt đầu tính phí'); inviting, not pushy. |

## Findings

None.

Minor/nit findings do not block; they are offered to the translator for an optional follow-up. The auditor did not edit any value.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used.

## Ledger note

history-sha before append: 83dcbad3fd53e560443aaba7c34cb89cf9666a43bd717ba3efc8ebf6ba5250c2. Receipt appended with `ledger.py append --dry-run` then `--apply` (dryrun.log). generate.py NOT run (coordinator regenerates).
