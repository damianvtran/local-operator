# i18n audit A — first independent audit — community (hi)

Auditor: reviewer / i18n-auditor, session c739715f0519, model **anthropic/claude-opus-5-5** (served model from the session header; no provider fallback). Translator: Local Operator coder subagent, session 06c3529d8318, anthropic/claude-opus-5-5, 2026-11-18. The auditor did not author, translate or remediate any audited byte.

PR: https://github.com/damianvtran/local-operator/pull/4221. Tree: `bd219f20b928f570b0230bac1197173267f04a74` + held patch `pr4221-i18n.patch` (sha256 `fcf8ed47f7d7fa28a4ab4ab4e728ac4efdc0da489d71911ed82ccdb68762a7c5`).
English: tui.community `cf2cc44838daed3feee13561e45dac11ddce031ac3fa702854a3215536671ac1`, wire.community `921c0e82256dc942800790bd098bca140fa075acd14d05dbe035c9ced04825ff`.
Targets: tui `eb715523250ef6823de80217c0b2aff04f2a08913fc7343e62f99d38189a77b5`, wire `04333dc1787c47a3e2840471a588b9a4417799a0d7062e503e7b1e96ca3d3250`.

Scope: all 153 keys (tui.community 148 + wire.community 5). Guides: i18n/glossary.md, i18n/style-guides/hi.md; terminology compared with local-operator-ui origin/feat/community-ui @ 1b1c4d31875dcbc44cc76f82b9a339278d93419d src/i18n/catalogues/<loc>/ui.community.json.

Mechanical: PASS (0 errors) — see audit.log.

Findings:
- **A-hi-M2** (MINOR) `tui.community.category_unsafe_or_illegal, category_spam_or_promotion` — 'अवैध' / 'बिना बताया गया प्रचार' vs desktop UI audited 'गैरकानूनी' / 'बिना बताए प्रचार' for the same refusal categories.

Verdict: PASS at the audited bytes; 0 blocker/major; 1 minor, 0 nit, all non-gating and deferred in the PR thread; no wording changed. Linguistic/copy audit only — no rendered native flow or RTL paint proof is claimed.
