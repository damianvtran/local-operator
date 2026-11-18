# i18n audit A — first independent audit — community (vi)

Auditor: reviewer / i18n-auditor, session c739715f0519, model **anthropic/claude-opus-5-5** (served model from the session header; no provider fallback). Translator: Local Operator coder subagent, session 06c3529d8318, anthropic/claude-opus-5-5, 2026-11-18. The auditor did not author, translate or remediate any audited byte.

PR: https://github.com/damianvtran/local-operator/pull/4221. Tree: `bd219f20b928f570b0230bac1197173267f04a74` + held patch `pr4221-i18n.patch` (sha256 `fcf8ed47f7d7fa28a4ab4ab4e728ac4efdc0da489d71911ed82ccdb68762a7c5`).
English: tui.community `cf2cc44838daed3feee13561e45dac11ddce031ac3fa702854a3215536671ac1`, wire.community `921c0e82256dc942800790bd098bca140fa075acd14d05dbe035c9ced04825ff`.
Targets: tui `bac6e6b8ee32ef9c70972d655a9eeb21bbed21a02ac9bcb81bf5db1b81f0df44`, wire `70cd8f2d3a6c0e98c4a787178c829e92000ccc1153788c6d1e15618e4cb48ff5`.

Scope: all 153 keys (tui.community 148 + wire.community 5). Guides: i18n/glossary.md, i18n/style-guides/vi.md; terminology compared with local-operator-ui origin/feat/community-ui @ 1b1c4d31875dcbc44cc76f82b9a339278d93419d src/i18n/catalogues/<loc>/ui.community.json.

Mechanical: PASS (0 errors) — see audit.log.

Findings:
- **A-vi-M2** (MINOR) `tui.community.category_unsafe_or_illegal` — 'Nội dung nguy hiểm hoặc bất hợp pháp' vs desktop UI audited 'Nội dung không an toàn hoặc bất hợp pháp' for the same refusal category.
- **A-vi-M1** (MINOR) `status_local_command, status_rate_probation, status_read_only` — Fit: at 80 columns the single-row status line (.community-view-status height: 1; column = 80 − RAIL_WIDTH 22 = 58 cells) cuts these strings (cell_len with sample values, en→vi): status_local_command 53→80; status_rate_probation 48→63; status_read_only 41→62. Meaning intact at ≥100 columns; en already overflows status_restricted (60). Renderer/layout lead plus optional shorter wording; not a wording defect.

Verdict: PASS at the audited bytes; 0 blocker/major; 2 minor, 0 nit, all non-gating and deferred in the PR thread; no wording changed. Linguistic/copy audit only — no rendered native flow or RTL paint proof is claimed.
