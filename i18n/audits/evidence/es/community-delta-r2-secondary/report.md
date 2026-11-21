# i18n delta audit round 2, pass B — approval-card keys — tui.community (es)

Auditor: reviewer / i18n-auditor, session 837443aac8da, model **anthropic/claude-opus-5-5** (served model from the session header; no provider fallback). Translator: Local Operator coder subagent (community lane, session 06c3529d8318), anthropic/claude-opus-5-5, 2026-11-18. The auditor did not author, translate or remediate any audited byte.

PR: https://github.com/damianvtran/local-operator/pull/4221. Tree: `37aa1d6ebb275c0928576366d48b52809d9ce02b` + held patch `pr4221-d-r1-i18n.patch (held, b9ae8b63, applied on head)` (sha256 `9b00194c5f960523fc2e0c0092050f725b93b52b7e84755a320c36cd7093cd36`).
English tui.community `67cf18960cb91fb3a557d5a725fd11fc90fc68714f93122dfee58478e29e7616`; context `d3030886a774ad2ed4cdb2f14e9969e8eb2c3afb2b9cd416c116d20cfa9abd88`. Target `e2e3de1bd2e9cc5239271b1fa2557c780f739235013a9dd9d2f7dc2a8f9a4219`.

Scope: `tui.community.post_allow_unread`, `tui.community.post_card_cut`. Safety reading: the label must not promise an approval the key does not give (y opens the whole post first; Allow needs reading it to the end); the cut cue must say the post is hidden and that v shows it; `v` untranslated, leading `…` kept; Allow/Deny remain the literal English labels elsewhere.
Guides: i18n/glossary.md, i18n/style-guides/es.md; context notes in en/tui.community.context.json. Terminology (publicación / permitir) matches this locale's audited `post_card_more`, `post_view_keys*` and the existing approval verbs.

Independence: Separate key-by-key pass with its own receipt, but the SAME session and model as pass A, and the same model as the translator. The wire.bridge_taint precedent (i18n/audits/evidence/fr/bridge-taint-secondary) used a different agent AND model; if that standard applies, one more pass on these 2 keys by another session/model is still needed before un-darking.

Mechanical: PASS (0 errors) — see audit.log.

Per key (back-reading):
- `tui.community.post_allow_unread`: “Leer la publicación y luego permitir” ≈ “Read the post and then allow” (en “Read the post, then allow”) — PASS.
- `tui.community.post_card_cut`: “… v muestra la publicación” ≈ “… v shows the post” (en “… v reads the post”) — PASS.

Findings:
- **B2-es-M1** (MINOR, cross-string, source-level (all locales); concurs with pass A) — On a cut card with the more-hint, `post_card_more` names the literal English “Allow” label while `post_allow_unread` now replaces that label with a translated verb until the post is read (approval.py:1200 sets `_post_cut` before both branches; 1215 renders the hint; 2318 swaps the label). In this locale the quoted label is therefore not on screen when the hint names it. Not a safety inversion: both strings say the post is read first, and the literal label returns once it is. English has the same loose coupling (“Allow needs that first” beside “Read the post, then allow”). Suggested fix (source, deferred): have en `post_card_more` stop naming the label (e.g. “v reads it all first”), then re-translate that one key.

Verdict: **passed** — 0 blocker, 0 major, 1 minor, 0 nit. No wording changed.
