# i18n delta audit round 2, pass A — changed/new keys — tui.community (ur)

Auditor: reviewer / i18n-auditor, session 837443aac8da, model **anthropic/claude-opus-5-5** (served model from the session header; no provider fallback). Translator: Local Operator coder subagent (community lane, session 06c3529d8318), anthropic/claude-opus-5-5, 2026-11-18. The auditor did not author, translate or remediate any audited byte.

PR: https://github.com/damianvtran/local-operator/pull/4221. Tree: `37aa1d6ebb275c0928576366d48b52809d9ce02b` + held patch `pr4221-d-r1-i18n.patch (held, b9ae8b63, applied on head)` (sha256 `9b00194c5f960523fc2e0c0092050f725b93b52b7e84755a320c36cd7093cd36`).
English tui.community `67cf18960cb91fb3a557d5a725fd11fc90fc68714f93122dfee58478e29e7616`; context `d3030886a774ad2ed4cdb2f14e9969e8eb2c3afb2b9cd416c116d20cfa9abd88`. Target `52879691abde0db4649268bb9b02350cfdd10a5eaa75924c9011a721eef0297f`.

Scope: `tui.community.page_error`, `tui.community.post_allow_unread`, `tui.community.post_card_cut`. The other 147 keys are byte-identical to the bytes the round-1 audit passed at en `cf2cc448…`, and their English is unchanged; this pass does not re-read them.
Guides: i18n/glossary.md, i18n/style-guides/ur.md; context notes in en/tui.community.context.json. Terminology (پوسٹ / اجازت دیں) matches this locale's audited `post_card_more`, `post_view_keys*` and the existing approval verbs.

Independence: Different session from the translator (06c3529d8318) and from the round-1 auditor (c739715f0519). Same model as the translator: agent-independent, not model-independent.

Mechanical: PASS (0 errors) — see audit.log.

Per key (back-reading):
- `tui.community.page_error`: “✕ کمیونٹی تک رسائی نہیں ہو سکی۔” ≈ “✕ Could not reach the community.” (en “✕ Couldn't reach the community.”) — PASS.
- `tui.community.post_allow_unread`: “پوسٹ پڑھیں، پھر اجازت دیں” ≈ “Read the post, then give permission” (en “Read the post, then allow”) — PASS.
- `tui.community.post_card_cut`: “… v پوسٹ پڑھیں” ≈ “… v read the post” (en “… v reads the post”) — PASS.

Findings:
- **A2-ur-M1** (MINOR, cross-string, source-level (all locales)) — On a cut card with the more-hint, `post_card_more` names the literal English “Allow” label while `post_allow_unread` now replaces that label with a translated verb until the post is read (approval.py:1200 sets `_post_cut` before both branches; 1215 renders the hint; 2318 swaps the label). In this locale the quoted label is therefore not on screen when the hint names it. Not a safety inversion: both strings say the post is read first, and the literal label returns once it is. English has the same loose coupling (“Allow needs that first” beside “Read the post, then allow”). Suggested fix (source, deferred): have en `post_card_more` stop naming the label (e.g. “v reads it all first”), then re-translate that one key.

Verdict: **passed** — 0 blocker, 0 major, 1 minor, 0 nit. No wording changed.
