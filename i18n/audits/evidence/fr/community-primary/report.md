# i18n audit A — first independent audit — community (fr)

Auditor: reviewer / i18n-auditor, session c739715f0519, model **anthropic/claude-opus-5-5** (served model from the session header; no provider fallback). Translator: Local Operator coder subagent, session 06c3529d8318, anthropic/claude-opus-5-5, 2026-11-18. The auditor did not author, translate or remediate any audited byte.

PR: https://github.com/damianvtran/local-operator/pull/4221. Tree: `bd219f20b928f570b0230bac1197173267f04a74` + held patch `pr4221-i18n.patch` (sha256 `fcf8ed47f7d7fa28a4ab4ab4e728ac4efdc0da489d71911ed82ccdb68762a7c5`).
English: tui.community `cf2cc44838daed3feee13561e45dac11ddce031ac3fa702854a3215536671ac1`, wire.community `921c0e82256dc942800790bd098bca140fa075acd14d05dbe035c9ced04825ff`.
Targets: tui `a8cb3e6fbe01704b1178c74ab8849aa68e18082083374dfd77c0ab6cf14e376f`, wire `fae760df36aedb49b847881eaf1957d80fe9834ce4905eea31414ef2cdb1c342`.

Scope: all 153 keys (tui.community 148 + wire.community 5). Guides: i18n/glossary.md, i18n/style-guides/fr.md; terminology compared with local-operator-ui origin/feat/community-ui @ 1b1c4d31875dcbc44cc76f82b9a339278d93419d src/i18n/catalogues/<loc>/ui.community.json.

Mechanical: PASS (0 errors) — see audit.log.

Findings:
- **A-fr-M1** (MINOR) `wire.community.post_card_each_time` — « Chaque publication vous est demandée » reads as 'each post is requested of you'; the English means you are asked to approve each post. Meaning recoverable from the second clause; suggested « Votre accord est demandé pour chaque publication ; il n’y a pas encore d’autorisation permanente. »
- **A-fr-N1** (NIT) `wire.community.post_card_audience, tui.community.post_view_keys_read, tui.community.audience_*` — Masculine agreement (vérifié … le voie; Lu en entier) where the implied subject is « publication » (f.); es agrees in the feminine (Leída entera). Optional: vérifiée … la voie / Lue en entier.
- **A-fr-M3** (MINOR) `status_local_command, status_locked, status_rate_probation, status_read_only, status_restricted` — Fit: at 80 columns the single-row status line (.community-view-status height: 1; column = 80 − RAIL_WIDTH 22 = 58 cells) cuts these strings (cell_len with sample values, en→fr): status_local_command 53→84; status_locked 45→64; status_rate_probation 48→61; status_read_only 41→61; status_restricted 60→90. Meaning intact at ≥100 columns; en already overflows status_restricted (60). Renderer/layout lead plus optional shorter wording; not a wording defect.

Verdict: PASS at the audited bytes; 0 blocker/major; 2 minor, 1 nit, all non-gating and deferred in the PR thread; no wording changed. Linguistic/copy audit only — no rendered native flow or RTL paint proof is claimed.
