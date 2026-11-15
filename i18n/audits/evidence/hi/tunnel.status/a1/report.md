# i18n audit a1 — hi / tunnel.status (PR #3931)

Head audited: e95a77a5f3ac5266d24c2bc317a8710d26817d60 (worktree .worktrees/mobile-tunnel-resilience, round 2). Date 2026-11-13.
Auditor: i18n-auditor (lopdev team), independent `task` subagent, session 5469a14c00f5; did not translate or edit any value it judged; model anthropic/claude-opus-5-5 (harness system prompt of session 5469a14c00f5 (`Model: anthropic/claude-opus-5-5`); no provider fallback was disclosed during the audit).
Translator: coder subagent (lopdev team) remediating PR #3931 round 1 — not an i18n-translator lane; its session id is named in neither the manager brief nor commit e95a77a5f3ac; anthropic/claude-opus-5-5 (as stated by the manager brief; not independently verified); 2026-11-13. Not an i18n-translator lane — recorded as found.

**Verdict: changes_requested** — 1 unresolved blocker/major.

आप register, danda, Latin digits, terms टनल/कनेक्टर/रिले/लॉगिन match corpus. Fails on निकास.

Read: docs/I18N.md L262-310, i18n/glossary.md, i18n/style-guides/hi.md, en + context sidecar, all 13 target values; corpus cross-checks in audit.log.

## Findings

### HI-01 — MAJOR
- Keys: `last_exit.at, last_exit.plain`
- Current: कनेक्टर का पिछला निकास
- Problem: निकास is an outlet/way out (जल निकास = drainage), not a process stopping; as a terse label it does not read as "the connector last exited". No corpus precedent.
- Suggested: at: «कनेक्टर पिछली बार बंद हुआ: {label} ({at} पर)» · plain: «कनेक्टर पिछली बार बंद हुआ: {label}»

### HI-02 — MINOR
- Keys: `gateway.listening_http_status, gateway.listening_timed_out, gateway.listening_unreadable`
- Current: हेल्थ जाँच / हेल्थ जवाब
- Problem: Hybrid; «हेल्थ जवाब» is not Hindi copy.
- Suggested: «स्वास्थ्य जाँच» / timed_out: «…लेकिन उसकी स्वास्थ्य जाँच का जवाब समय पर नहीं आया; …» · unreadable: «…लेकिन उसकी स्वास्थ्य जाँच का जवाब पढ़ा नहीं जा सका; …»

### HI-03 — NIT
- Keys: `server_unavailable.terminal`
- Current: गलती
- Problem: hi guide: keep nukta consistently.
- Suggested: «ग़लती»

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame, no RTL render (ur). Judgement by reasoning only; no MT service used.

This bundle is STAGED in session scratch; it is not on any branch until the implementer commits it at i18n/audits/evidence/hi/tunnel.status/a1/.
