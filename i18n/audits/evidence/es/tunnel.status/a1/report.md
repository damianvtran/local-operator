# i18n audit a1 — es / tunnel.status (PR #3931)

Head audited: e95a77a5f3ac5266d24c2bc317a8710d26817d60 (worktree .worktrees/mobile-tunnel-resilience, round 2). Date 2026-11-13.
Auditor: i18n-auditor (lopdev team), independent `task` subagent, session 5469a14c00f5; did not translate or edit any value it judged; model anthropic/claude-opus-5-5 (harness system prompt of session 5469a14c00f5 (`Model: anthropic/claude-opus-5-5`); no provider fallback was disclosed during the audit).
Translator: coder subagent (lopdev team) remediating PR #3931 round 1 — not an i18n-translator lane; its session id is named in neither the manager brief nor commit e95a77a5f3ac; anthropic/claude-opus-5-5 (as stated by the manager brief; not independently verified); 2026-11-13. Not an i18n-translator lane — recorded as found.

**Verdict: changes_requested** — 1 unresolved blocker/major.

tú register, neutral vocabulary, glossary relé/inicio de sesión consistent. Fails on the «solo cuando» ambiguity in both self-healing sentences.

Read: docs/I18N.md L262-310, i18n/glossary.md, i18n/style-guides/es.md, en + context sidecar, all 13 target values; corpus cross-checks in audit.log.

## Findings

### ES-01 — MAJOR
- Keys: `server_unavailable.relay, server_unavailable.terminal`
- Current: se vuelve a autorizar solo cuando Radient se recupere
- Problem: "solo cuando" reads as "only when"; the source says it reauthorizes BY ITSELF once Radient recovers. On the phone (relay) this is the only self-healing cue, so the reassurance the sidecar asks for is lost/ambiguous.
- Suggested: relay: «…No hay ningún problema con el inicio de sesión ni con este equipo; volverá a autorizarse automáticamente en cuanto Radient se recupere.» · terminal: «…Reintenta cada 10 segundos y volverá a autorizarse automáticamente en cuanto Radient se recupere; no hace falta ningún comando local.»

### ES-02 — MINOR
- Keys: `last_exit.at`
- Current: (a las {at})
- Problem: {at} may be "2026-11-12 14:32" (cli.py _stamp; sidecar says so); «a las 2026-11-12 14:32» is ungrammatical.
- Suggested: «({at})»

### ES-03 — MINOR
- Keys: `last_exit.at, last_exit.plain`
- Current: Última salida del conector
- Problem: In a terse label «salida» first reads as "output" («salida del conector: Error del servidor…»).
- Suggested: «Última detención del conector: {label}» (and «… ({at})»)

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame, no RTL render (ur). Judgement by reasoning only; no MT service used.

This bundle is STAGED in session scratch; it is not on any branch until the implementer commits it at i18n/audits/evidence/es/tunnel.status/a1/.
