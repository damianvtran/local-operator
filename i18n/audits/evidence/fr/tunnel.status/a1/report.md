# i18n audit a1 — fr / tunnel.status (PR #3931)

Head audited: e95a77a5f3ac5266d24c2bc317a8710d26817d60 (worktree .worktrees/mobile-tunnel-resilience, round 2). Date 2026-11-13.
Auditor: i18n-auditor (lopdev team), independent `task` subagent, session 5469a14c00f5; did not translate or edit any value it judged; model anthropic/claude-opus-5-5 (harness system prompt of session 5469a14c00f5 (`Model: anthropic/claude-opus-5-5`); no provider fallback was disclosed during the audit).
Translator: coder subagent (lopdev team) remediating PR #3931 round 1 — not an i18n-translator lane; its session id is named in neither the manager brief nor commit e95a77a5f3ac; anthropic/claude-opus-5-5 (as stated by the manager brief; not independently verified); 2026-11-13. Not an i18n-translator lane — recorded as found.

**Verdict: changes_requested** — 2 unresolved blocker/major.

Meaning accurate in all 13; vous register, tokens intact. Fails on typography (U+202F) and the health-check calque.

Read: docs/I18N.md L262-310, i18n/glossary.md, i18n/style-guides/fr.md, en + context sidecar, all 13 target values; corpus cross-checks in audit.log.

## Findings

### FR-01 — MAJOR
- Keys: `gateway.listening_http_status, gateway.listening_timed_out, gateway.listening_unreadable, last_exit.at, last_exit.plain, login.deferred_retried, login.unchecked, requests.heading, server_unavailable.relay, server_unavailable.terminal`
- Current: U+0020 before « : » / « ; » (11 occurrences in 10 keys)
- Problem: fr guide L93 requires espace fine insécable U+202F before : ; ! ?; the shipped fr corpus uses U+202F 976 times vs U+0020 53. A plain space also lets the colon wrap to the next line in the status output.
- Suggested: Replace every " :" / " ;" with "\u202f:" / "\u202f;"; no other byte changes.

### FR-02 — MAJOR
- Keys: `gateway.listening_http_status, gateway.listening_timed_out, gateway.listening_unreadable`
- Current: « contrôle de santé » / « sa réponse de santé a expiré » / « sa réponse de santé est illisible »
- Problem: Calque of "health check/answer"; "réponse de santé" is not French product copy and "a expiré" says the answer expired, not that it never arrived. Settled fr: « vérification d’état ».
- Suggested: http_status: « …mais il a répondu à la vérification d’état par HTTP {status}\u202f: un autre programme occupe peut-être le port, ou la passerelle est défaillante. Une réinstallation n’y changera rien. » · timed_out: « La passerelle de relais locale écoute sur 127.0.0.1:{port}, mais elle n’a pas répondu à temps à la vérification d’état\u202f; une réinstallation n’y changera rien. » · unreadable: « …mais sa réponse à la vérification d’état est illisible\u202f; une réinstallation n’y changera rien. »

### FR-03 — NIT
- Keys: `last_exit.at`
- Current: (à {at})
- Problem: {at} is "YYYY-MM-DD HH:MM" for older exits (cli.py _stamp); « à 2026-11-12 14:32 » reads off.
- Suggested: « ({at}) »

### FR-04 — NIT
- Keys: `create.conflict_unmatched`
- Current: Vérifiez lop tunnel list
- Problem: One reads a list rather than verifies it.
- Suggested: « Consultez lop tunnel list, puis… »

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame, no RTL render (ur). Judgement by reasoning only; no MT service used.

This bundle is STAGED in session scratch; it is not on any branch until the implementer commits it at i18n/audits/evidence/fr/tunnel.status/a1/.
