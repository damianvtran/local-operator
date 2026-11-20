# i18n audit B — second pass, approval-card copy — community (ur)

Auditor: reviewer / i18n-auditor, session c739715f0519, model **anthropic/claude-opus-5-5** (served model from the session header; no provider fallback). Translator: Local Operator coder subagent, session 06c3529d8318, anthropic/claude-opus-5-5, 2026-11-18. The auditor did not author, translate or remediate any audited byte.

PR: https://github.com/damianvtran/local-operator/pull/4221. Tree: `bd219f20b928f570b0230bac1197173267f04a74` + held patch `pr4221-i18n.patch` (sha256 `fcf8ed47f7d7fa28a4ab4ab4e728ac4efdc0da489d71911ed82ccdb68762a7c5`).
English: tui.community `cf2cc44838daed3feee13561e45dac11ddce031ac3fa702854a3215536671ac1`, wire.community `921c0e82256dc942800790bd098bca140fa075acd14d05dbe035c9ced04825ff`.
Targets: tui `9435e686cebdbc3d765fbeb5fa262a5c45af018e12d1322dc3b197be2558f068`, wire `fde0737e968f461116ccf1255f5a8b4073dc265550ee030cf81484189b3c71ea`.

Scope: 8 approval-card keys: wire.community.post_card_{title,audience,label,each_time,tainted} + tui.community.{post_card_more,post_view_keys,post_view_keys_read}. Guides: i18n/glossary.md, i18n/style-guides/ur.md; terminology compared with local-operator-ui origin/feat/community-ui @ 1b1c4d31875dcbc44cc76f82b9a339278d93419d src/i18n/catalogues/<loc>/ui.community.json.

Independence: this is a SEPARATE, key-by-key pass with its own receipt, as briefed. It is independent of the translator (different agent/session) but NOT independent of audit A (same auditor session, same model) and not model-independent of the translator. The `wire.bridge_taint` precedent used a different agent AND model for its second safety audit; if that standard is required here, a further pass by another session/model is still needed before un-darking.

Safety reading checked per key (English meaning first, then back-reading): who will see the post (all members of the channel, moderated first); that it appears as the owner's own post with no agent label; why it asks (no standing permission; tainted session ⇒ none until a new session; only the owner's answer posts); the read-in-full rule before Allow; Allow/Deny kept as the literal English TUI button labels (approval.py:325-327, 961-963).

Mechanical: PASS (0 errors) — see audit.log.

Findings: none.

Verdict: PASS at the audited bytes; 0 blocker/major; 0 minor, 0 nit, all non-gating and deferred in the PR thread; no wording changed. Linguistic/copy audit only — no rendered native flow or RTL paint proof is claimed.
