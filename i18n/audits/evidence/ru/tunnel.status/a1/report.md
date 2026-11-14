# i18n audit a1 — ru / tunnel.status (PR #3931)

Head audited: e95a77a5f3ac5266d24c2bc317a8710d26817d60 (worktree .worktrees/mobile-tunnel-resilience, round 2). Date 2026-11-13.
Auditor: i18n-auditor (lopdev team), independent `task` subagent, session 5469a14c00f5; did not translate or edit any value it judged; model anthropic/claude-opus-5-5 (harness system prompt of session 5469a14c00f5 (`Model: anthropic/claude-opus-5-5`); no provider fallback was disclosed during the audit).
Translator: coder subagent (lopdev team) remediating PR #3931 round 1 — not an i18n-translator lane; its session id is named in neither the manager brief nor commit e95a77a5f3ac; anthropic/claude-opus-5-5 (as stated by the manager brief; not independently verified); 2026-11-13. Not an i18n-translator lane — recorded as found.

**Verdict: passed** — 0 unresolved blocker/major.

вы/impersonal register, ретранслятор per glossary, коннектор/вход match corpus, «ёлочки» not needed, em dashes spaced. No blocker/major.

Read: docs/I18N.md L262-310, i18n/glossary.md, i18n/style-guides/ru.md, en + context sidecar, all 13 target values; corpus cross-checks in audit.log.

## Findings

### RU-01 — MINOR
- Keys: `create.conflict_unmatched`
- Current: в списке туннелей этого входа
- Problem: «вход» is the act of signing in; owning a list reads odd.
- Suggested: «в списке туннелей этой учётной записи»

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame, no RTL render (ur). Judgement by reasoning only; no MT service used.

This bundle is STAGED in session scratch; it is not on any branch until the implementer commits it at i18n/audits/evidence/ru/tunnel.status/a1/.
