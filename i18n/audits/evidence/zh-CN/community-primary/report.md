# i18n audit A — first independent audit — community (zh-CN)

Auditor: reviewer / i18n-auditor, session c739715f0519, model **anthropic/claude-opus-5-5** (served model from the session header; no provider fallback). Translator: Local Operator coder subagent, session 06c3529d8318, anthropic/claude-opus-5-5, 2026-11-18. The auditor did not author, translate or remediate any audited byte.

PR: https://github.com/damianvtran/local-operator/pull/4221. Tree: `bd219f20b928f570b0230bac1197173267f04a74` + held patch `pr4221-i18n.patch` (sha256 `fcf8ed47f7d7fa28a4ab4ab4e728ac4efdc0da489d71911ed82ccdb68762a7c5`).
English: tui.community `cf2cc44838daed3feee13561e45dac11ddce031ac3fa702854a3215536671ac1`, wire.community `921c0e82256dc942800790bd098bca140fa075acd14d05dbe035c9ced04825ff`.
Targets: tui `afcdb8b7157c8f5e343d2da055e2c65ceb4966b01d4e226b4152a71f63c3a2b1`, wire `252c0e226362893b015de0a88835991ed05f22b2637775f54fcba0efd8beb351`.

Scope: all 153 keys (tui.community 148 + wire.community 5). Guides: i18n/glossary.md, i18n/style-guides/zh-CN.md; terminology compared with local-operator-ui origin/feat/community-ui @ 1b1c4d31875dcbc44cc76f82b9a339278d93419d src/i18n/catalogues/<loc>/ui.community.json.

Mechanical: PASS (0 errors) — see audit.log.

Findings:
- **A-zh-N1** (NIT) `tui.community.keys_body, browse_tab_channels, category_unsafe_or_illegal` — Spacing/wording drift: 'tab 频道列表' (space) vs 'tab频道' elsewhere and '[频道 {count}]'; desktop UI audited label is '不安全或违法的内容' (with 的).

Verdict: PASS at the audited bytes; 0 blocker/major; 0 minor, 1 nit, all non-gating and deferred in the PR thread; no wording changed. Linguistic/copy audit only — no rendered native flow or RTL paint proof is claimed.
