# i18n audit a1 — ur / tunnel.status (PR #3931)

Head audited: e95a77a5f3ac5266d24c2bc317a8710d26817d60 (worktree .worktrees/mobile-tunnel-resilience, round 2). Date 2026-11-13.
Auditor: i18n-auditor (lopdev team), independent `task` subagent, session 5469a14c00f5; did not translate or edit any value it judged; model anthropic/claude-opus-5-5 (harness system prompt of session 5469a14c00f5 (`Model: anthropic/claude-opus-5-5`); no provider fallback was disclosed during the audit).
Translator: coder subagent (lopdev team) remediating PR #3931 round 1 — not an i18n-translator lane; its session id is named in neither the manager brief nor commit e95a77a5f3ac; anthropic/claude-opus-5-5 (as stated by the manager brief; not independently verified); 2026-11-13. Not an i18n-translator lane — recorded as found.

**Verdict: changes_requested** — 1 unresolved blocker/major.

آپ register, Urdu punctuation ۔ ، ؛, Latin runs byte-identical, no manual bidi controls, Latin digits. Fails on اخراج. RTL: Latin runs (lop tunnel list, 127.0.0.1:{port}, HTTP {status}, <id>) rely on the render layer's FSI/PDI; no copy-level bidi defect.

Read: docs/I18N.md L262-310, i18n/glossary.md, i18n/style-guides/ur.md, en + context sidecar, all 13 target values; corpus cross-checks in audit.log.

## Findings

### UR-01 — MAJOR
- Keys: `last_exit.at, last_exit.plain`
- Current: کنیکٹر کا آخری اخراج
- Problem: اخراج means expulsion/emission/exclusion (its only other corpus use, wire.host_speed, is "excluding"), not a process stopping; the row reads "the connector's last expulsion".
- Suggested: at: «کنیکٹر آخری بار بند ہوا: {label} ({at} پر)» · plain: «کنیکٹر آخری بار بند ہوا: {label}»

### UR-02 — MINOR
- Keys: `gateway.listening_http_status, gateway.listening_timed_out, gateway.listening_unreadable`
- Current: ہیلتھ چیک / ہیلتھ جواب
- Problem: Hybrid transliteration; «ہیلتھ جواب» is not Urdu. Corpus renders health as صحت («ڈیمن کی صحت»).
- Suggested: «صحت کی جانچ» / timed_out: «…لیکن صحت کی جانچ کا جواب وقت پر نہیں آیا؛ …» · unreadable: «…لیکن صحت کی جانچ کا جواب پڑھا نہیں جا سکا؛ …»

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame, no RTL render (ur). Judgement by reasoning only; no MT service used.

This bundle is STAGED in session scratch; it is not on any branch until the implementer commits it at i18n/audits/evidence/ur/tunnel.status/a1/.
