# i18n audit a2 — zh-CN / tunnel.status (PR #3931)

Head audited: cf5fa645a0d44abd77a77109b21cff6f0f50aca4 (worktree .worktrees/mobile-tunnel-resilience). Scope: e95a77a5f3ac..cf5fa645a0d4 (revision of a1 findings). Date 2026-11-13.
Auditor: i18n-auditor (lopdev team), independent `task` subagent, session c7c8dd95452d; neither the a1 auditor (5469a14c00f5) nor the drafting coder nor the revising i18n-translator (job ec010cb7b6b8); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session c7c8dd95452d (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: original drafts: coder subagent (PR #3931 round 1); revision: i18n-translator subagent job ec010cb7b6b8 (commit cf5fa645a0d4); model anthropic/claude-opus-5-5 (as stated by the manager brief; not independently verified); 2026-11-13.
Prior audit: a1 (session 5469a14c00f5) at target sha256 ecfa838240466f61427faf9a604ccac22a6e8ae4938354320befcc89a7d7fd3f.

**Verdict: passed** — 0 unresolved blocker/major.

Revised values are spacing-only plus three wording fixes; meaning unchanged and correct.

Read: docs/I18N.md L262-310, a1 report, i18n/glossary.md (no gateway/tunnel entry), i18n/style-guides/zh-CN.md, en + context sidecar, the e95a77a5f3ac..cf5fa645a0d4 diff and every revised value in full. Unchanged, a1-accepted values were not re-litigated.

## a1 findings — verification

| ID | a1 severity | a2 status | evidence |
|---|---|---|---|
| ZH-01 | MAJOR | fixed | 0 spaced Han↔Latin/digit boundaries remain (audit.log); spaces kept only inside Latin runs (HTTP {status}, command lines) |
| ZH-02 | MINOR | fixed | 「当前登录账户的隧道列表」 |
| ZH-03 | NIT | fixed | 「先查看lop tunnel list，…」 (remaining 请 occurrences are 请求) |
| ZH-04 | NIT | fixed | 「登录：刷新已推迟，将自动重试。」 |

## New findings in revised strings

None. Mechanical re-check (check.py --catalogues-only: 0 problems; key parity, placeholders, literal command/flag/address tokens identical to en) in audit.log.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used.

## Ledger note

history-sha 4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945 is the empty-history value: the a1 receipts were not applied. If a1 rows are appended first, re-read history-sha before applying this a2 receipt (last row wins).

This bundle is STAGED in session scratch; it is not on any branch until the implementer commits it at i18n/audits/evidence/zh-CN/tunnel.status/a2/.
