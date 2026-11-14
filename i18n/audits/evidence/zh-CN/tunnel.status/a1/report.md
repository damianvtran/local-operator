# i18n audit a1 — zh-CN / tunnel.status (PR #3931)

Head audited: e95a77a5f3ac5266d24c2bc317a8710d26817d60 (worktree .worktrees/mobile-tunnel-resilience, round 2). Date 2026-11-13.
Auditor: i18n-auditor (lopdev team), independent `task` subagent, session 5469a14c00f5; did not translate or edit any value it judged; model anthropic/claude-opus-5-5 (harness system prompt of session 5469a14c00f5 (`Model: anthropic/claude-opus-5-5`); no provider fallback was disclosed during the audit).
Translator: coder subagent (lopdev team) remediating PR #3931 round 1 — not an i18n-translator lane; its session id is named in neither the manager brief nor commit e95a77a5f3ac; anthropic/claude-opus-5-5 (as stated by the manager brief; not independently verified); 2026-11-13. Not an i18n-translator lane — recorded as found.

**Verdict: changes_requested** — 1 unresolved blocker/major.

Meaning, terms (隧道/连接器/中继/登录) and full-width punctuation correct. Fails on Han↔Latin spacing against the guide and the shipped corpus. Security hold: SECURITY_HELD_LOCALES is empty (scripts/i18n/generate.py:143; docs/I18N.md L55-63, released 2026-10-24); the ~L415 table is the superseded 0.70.4 snapshot, so zh-CN is audited normally.

Read: docs/I18N.md L262-310, i18n/glossary.md, i18n/style-guides/zh-CN.md, en + context sidecar, all 13 target values; corpus cross-checks in audit.log.

## Findings

### ZH-01 — MAJOR
- Keys: `create.conflict_unmatched, gateway.listening_http_status, gateway.listening_timed_out, gateway.listening_unreadable, login.unchecked, requests.heading, server_unavailable.label, server_unavailable.relay, server_unavailable.terminal`
- Current: spaces between Han text and Latin/digit runs, e.g. «Radient 服务器错误», «最近 {window} 条», «每 10 秒», «用 lop tunnel connect … 连接»
- Problem: zh-CN guide (Punctuation) forbids a space between Chinese and Latin/digit runs; shipped corpus has 1915 tight vs 107 spaced boundaries and mobile.setup — the same tunnel flow — has 0 of 79 values spaced («运行lop login-status，»).
- Suggested: Remove only the Han↔Latin/digit spaces, keep spaces inside Latin runs: «Radient服务器错误», «请求（最近{window}条）：{tally}», «…返回了HTTP {status}：…», «本地中继网关正在127.0.0.1:{port}上监听…», «…（刷新请求无法连接到Radient）。», «它每10秒重试一次，Radient恢复后会自动重新授权；无需执行任何本地命令。»

### ZH-02 — MINOR
- Keys: `create.conflict_unmatched`
- Current: 此登录的隧道列表
- Problem: "this login's" calque; the corpus names the owning account (所属账户/账户).
- Suggested: «当前登录账户的隧道列表»

### ZH-03 — NIT
- Keys: `create.conflict_unmatched`
- Current: 请先查看
- Problem: Guide: 请 only where English says "please"; it does not.
- Suggested: «先查看lop tunnel list，然后用lop tunnel connect --tunnel-id <id>连接。»

### ZH-04 — NIT
- Keys: `login.deferred_retried`
- Current: 刷新已推迟 — 将自动重试
- Problem: Half-width spaced dash in Chinese prose (corpus mixed).
- Suggested: «登录：刷新已推迟，将自动重试。»

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame, no RTL render (ur). Judgement by reasoning only; no MT service used.

This bundle is STAGED in session scratch; it is not on any branch until the implementer commits it at i18n/audits/evidence/zh-CN/tunnel.status/a1/.
