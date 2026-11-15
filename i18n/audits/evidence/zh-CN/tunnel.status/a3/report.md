# i18n audit a3 — zh-CN / tunnel.status (branch fix/tunnel-remedies-auto-up)

Head audited: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (worktree .worktrees/tunnel-remedies-auto-up). Scope: 1fce7ecb1f5f~1..1fce7ecb1f5f (values introduced in 1fce7ecb1f5f). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session ede889388023 (node lop-remote-node-2); not the translating i18n-translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session ede889388023: `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (values introduced in commit 1fce7ecb1f5f; English drafted by coder session b0300e2c2c7f); model anthropic/claude-opus-5-5 (as stated by the delegation brief; not independently verified); 2026-11-14.
Prior audit: zh-CN tunnel.status a2 (passed, source e5e82a9a…, PR #3931); this round covers the 8 new keys and re-reads the 13 a2-passed keys for consistency only.

**Verdict: passed** — 0 unresolved blocker/major; 3 finding(s) total (ZH-01 minor, ZH-02 nit, ZH-03 nit).

Source en sha256 b368acfb0cdee89c8de5a210fc9956251701a9f3f73c88b1096611d0c1f9063f; target sha256 b5ae3b1046bf0dbc02ce8e8f7e83f51bbd18bc9cc56b1b413f79d9b21f41768c; keys 21 (8 new, all judged).

Read: docs/I18N.md L260-310, i18n/glossary.md (no tunnel/billing/credit row; §1 keeps Radient, Local Operator, URLs, commands as-is), i18n/style-guides/zh-CN.md, en + context sidecar for every new key, corpus usage of the key terms (audit.log), and every new value in full.

Safety meanings checked on every new key: no terminal command named (only `/mobile enable`, verbatim, in mobile.setup.login_offer); nothing-is-charged / never activates billing or spends credit / setup never purchases credit / price shown before billing all survive; {url} and the console URL are byte-identical.

## Per-key judgement

| key | value | judgement |
|---|---|---|
| tunnel.status.tunnel_inactive.label | 隧道已暂停 | Open question answered: accepted. Context says the tunnel is suspended by Radient; 暂停 matches the detail ('Radient已暂停此隧道') and mobile.setup.reconnect, and stays distinct from user-initiated 停止 (mobile.setup.stopped). |
| tunnel.status.tunnel_inactive.detail | … | Meaning, Reconnect-in-Local-Operator remedy, no command, {url} intact. ZH-02, ZH-03 (nits). |
| tunnel.status.tunnel_disabled.label | 隧道已在控制台中关闭 | OK; no 。 on label; 控制台 matches corpus. |
| tunnel.status.tunnel_disabled.detail | … | ZH-01 (minor). |
| tunnel.status.billing_inactive.label | 计费未激活 | OK; 计费 matches 30 corpus uses. |
| tunnel.status.billing_inactive.detail | … | OK; 接受月费 matches mobile.setup.price/billing_activated. |
| tunnel.status.billing_lapsed.label | 计费已失效 | Open question answered: accepted. 失效 conveys 'was valid, no longer is', distinct from 未激活; same wording in mobile.setup.billing_lapsed. |
| tunnel.status.billing_lapsed.detail | … | OK; 充值 / 余额 consistent with mobile.setup; auto-reconnect kept; {url} intact. |

## Findings

| ID | severity | key | issue | suggestion |
|---|---|---|---|---|
| ZH-01 | minor | tunnel.status.tunnel_disabled.detail | The URL is attached after '远程访问将保持关闭：{url}', so it reads as elaborating 'stays off' rather than as where to turn the tunnel back on (EN: 'turned back on there: {url}'). | '此隧道已在Radient控制台中关闭，重新开启前远程访问将保持关闭。请在{url}重新开启；开启后会自动重新连接。' (mirrors mobile.setup.disabled) |
| ZH-02 | nit | tunnel.status.tunnel_inactive.detail | '请在运行它的电脑上，于Local Operator中重新连接手机访问' stacks 在/于 and reads written-formal. | '请在运行它的电脑上打开Local Operator，重新连接手机访问；…' |
| ZH-03 | nit | tunnel.status.{tunnel_inactive,billing_inactive,billing_lapsed}.detail | zh-CN guide: '请' only where the English says please (blocking dialogs); 4 new uses here (corpus has 162; the a2 auditor accepted 3 in this namespace, and the sibling mobile.setup.balance drops it). | Optional: drop 请 for terse imperative ('在{url}充值；…') if the coordinator wants guide-strict consistency. |

Minor/nit findings do not block; they are offered to the translator for an optional follow-up. The auditor did not edit any value.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used.

## Ledger note

history-sha before append: f7684438a29ab31a65b204654c4242ee05473a6b678ddf62d11127c879ae3fa5. Receipt appended with `ledger.py append --dry-run` then `--apply` (dryrun.log). generate.py NOT run (coordinator regenerates).
