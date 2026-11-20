# i18n audit B — second pass, approval-card copy — community (fr)

Auditor: reviewer / i18n-auditor, session c739715f0519, model **anthropic/claude-opus-5-5** (served model from the session header; no provider fallback). Translator: Local Operator coder subagent, session 06c3529d8318, anthropic/claude-opus-5-5, 2026-11-18. The auditor did not author, translate or remediate any audited byte.

PR: https://github.com/damianvtran/local-operator/pull/4221. Tree: `bd219f20b928f570b0230bac1197173267f04a74` + held patch `pr4221-i18n.patch` (sha256 `fcf8ed47f7d7fa28a4ab4ab4e728ac4efdc0da489d71911ed82ccdb68762a7c5`).
English: tui.community `cf2cc44838daed3feee13561e45dac11ddce031ac3fa702854a3215536671ac1`, wire.community `921c0e82256dc942800790bd098bca140fa075acd14d05dbe035c9ced04825ff`.
Targets: tui `a8cb3e6fbe01704b1178c74ab8849aa68e18082083374dfd77c0ab6cf14e376f`, wire `fae760df36aedb49b847881eaf1957d80fe9834ce4905eea31414ef2cdb1c342`.

Scope: 8 approval-card keys: wire.community.post_card_{title,audience,label,each_time,tainted} + tui.community.{post_card_more,post_view_keys,post_view_keys_read}. Guides: i18n/glossary.md, i18n/style-guides/fr.md; terminology compared with local-operator-ui origin/feat/community-ui @ 1b1c4d31875dcbc44cc76f82b9a339278d93419d src/i18n/catalogues/<loc>/ui.community.json.

Independence: this is a SEPARATE, key-by-key pass with its own receipt, as briefed. It is independent of the translator (different agent/session) but NOT independent of audit A (same auditor session, same model) and not model-independent of the translator. The `wire.bridge_taint` precedent used a different agent AND model for its second safety audit; if that standard is required here, a further pass by another session/model is still needed before un-darking.

Safety reading checked per key (English meaning first, then back-reading): who will see the post (all members of the channel, moderated first); that it appears as the owner's own post with no agent label; why it asks (no standing permission; tainted session ⇒ none until a new session; only the owner's answer posts); the read-in-full rule before Allow; Allow/Deny kept as the literal English TUI button labels (approval.py:325-327, 961-963).

Mechanical: PASS (0 errors) — see audit.log.

Findings:
- **B-fr-M1** (MINOR) `wire.community.post_card_each_time` — Concurs with A-fr-M1 (no safety inversion: 'il n’y a pas encore d’autorisation permanente' carries the gate meaning).
- **B-fr-N1** (NIT) `wire.community.post_card_audience, tui.community.post_view_keys_read` — Concurs with A-fr-N1.

Verdict: PASS at the audited bytes; 0 blocker/major; 1 minor, 1 nit, all non-gating and deferred in the PR thread; no wording changed. Linguistic/copy audit only — no rendered native flow or RTL paint proof is claimed.
