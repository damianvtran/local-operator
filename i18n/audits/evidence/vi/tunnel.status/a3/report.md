# i18n audit a3 — vi / tunnel.status (branch fix/tunnel-remedies-auto-up)

Head audited: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (worktree .worktrees/tunnel-remedies-auto-up). Scope: 1fce7ecb1f5f~1..1fce7ecb1f5f (values introduced in 1fce7ecb1f5f). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session ede889388023 (node lop-remote-node-2); not the translating i18n-translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session ede889388023: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (values introduced in commit 1fce7ecb1f5f; English drafted by coder session b0300e2c2c7f); model anthropic/claude-opus-5-5 (as stated by the delegation brief; not independently verified); 2026-11-14.
Prior audit: vi tunnel.status a2 (passed, source e5e82a9a…, PR #3931); this round covers the 8 new keys and re-reads the 13 a2-passed keys for consistency only.

**Verdict: passed** — 0 unresolved blocker/major; 2 finding(s) total (VI-01 minor, VI-02 nit).

Source en sha256 b368acfb0cdee89c8de5a210fc9956251701a9f3f73c88b1096611d0c1f9063f; target sha256 df6e1132957448034b7470e95792783dd9e892ad84720b71c7e1fa6d5a620a1a; keys 21 (8 new, all judged).

Read: docs/I18N.md L260-310, i18n/glossary.md (no tunnel/billing/credit row; §1 keeps Radient, Local Operator, URLs, commands as-is), i18n/style-guides/vi.md, en + context sidecar for every new key, corpus usage of the key terms (audit.log), and every new value in full.

Safety meanings checked on every new key: no terminal command named (only `/mobile enable`, verbatim, in mobile.setup.login_offer); nothing-is-charged / never activates billing or spends credit / setup never purchases credit / price shown before billing all survive; {url} and the console URL are byte-identical.

## Per-key judgement

| key | value | judgement |
|---|---|---|
| tunnel.status.tunnel_inactive.label | tunnel không hoạt động | OK — lowercase fragment, 'tunnel' kept English per vi guide L104-107. 'tunnel bị tạm dừng' would echo the detail more closely, but 'không hoạt động' is the faithful rendering of 'not active'; accepted. |
| tunnel.status.tunnel_inactive.detail | … | Meaning, calm tone, Reconnect-in-Local-Operator remedy, no command, {url} intact. 'dù việc thanh toán vẫn ổn' is idiomatic. VI-01 (minor). |
| tunnel.status.tunnel_disabled.label | tunnel bị tắt trong bảng điều khiển | OK; 'bảng điều khiển' matches the 7 corpus uses for console. |
| tunnel.status.tunnel_disabled.detail | … | Meaning and {url} intact; console is the only fix; no command. VI-02 (nit). |
| tunnel.status.billing_inactive.label | chưa kích hoạt thanh toán | OK; matches mobile.setup.billing_activated 'Đã kích hoạt thanh toán'. |
| tunnel.status.billing_inactive.detail | … | OK; 'chấp nhận mức giá hằng tháng' matches corpus price wording; remedy is phone access setup, no command. |
| tunnel.status.billing_lapsed.label | thanh toán đã hết hiệu lực | Open question answered: accepted. 'hết hiệu lực' = no longer in effect, correctly distinct from 'chưa kích hoạt'; identical wording in mobile.setup.billing_lapsed keeps the pair consistent. |
| tunnel.status.billing_lapsed.detail | … | OK; 'nạp tín dụng' matches mobile.setup.balance; auto-reconnect clause kept; {url} intact. |

## Findings

| ID | severity | key | issue | suggestion |
|---|---|---|---|---|
| VI-01 | minor | tunnel.status.tunnel_inactive.detail | Two 'nó' in one sentence with shifting referents ('trên máy tính đang chạy nó; nó cũng tự thử lại liên tục'): the second 'nó' can read as the computer rather than the tunnel/connector. | '… trên máy tính đang chạy tunnel; tunnel cũng tự động thử lại liên tục.' (translator's call) |
| VI-02 | nit | tunnel.status.tunnel_disabled.detail | 'truy cập từ xa sẽ tắt' (will switch off) for EN 'stays off'. | 'truy cập từ xa sẽ vẫn tắt cho đến khi …' |

Minor/nit findings do not block; they are offered to the translator for an optional follow-up. The auditor did not edit any value.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used.

## Ledger note

history-sha before append: 7fa2ca421925c44f5143dba82dd64afe72c861cb4ded6cd11dfd04ef626fe9ea. Receipt appended with `ledger.py append --dry-run` then `--apply` (dryrun.log). generate.py NOT run (coordinator regenerates).
