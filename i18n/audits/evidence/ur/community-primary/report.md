# i18n audit A — first independent audit — community (ur)

Auditor: reviewer / i18n-auditor, session c739715f0519, model **anthropic/claude-opus-5-5** (served model from the session header; no provider fallback). Translator: Local Operator coder subagent, session 06c3529d8318, anthropic/claude-opus-5-5, 2026-11-18. The auditor did not author, translate or remediate any audited byte.

PR: https://github.com/damianvtran/local-operator/pull/4221. Tree: `bd219f20b928f570b0230bac1197173267f04a74` + held patch `pr4221-i18n.patch` (sha256 `fcf8ed47f7d7fa28a4ab4ab4e728ac4efdc0da489d71911ed82ccdb68762a7c5`).
English: tui.community `cf2cc44838daed3feee13561e45dac11ddce031ac3fa702854a3215536671ac1`, wire.community `921c0e82256dc942800790bd098bca140fa075acd14d05dbe035c9ced04825ff`.
Targets: tui `9435e686cebdbc3d765fbeb5fa262a5c45af018e12d1322dc3b197be2558f068`, wire `fde0737e968f461116ccf1255f5a8b4073dc265550ee030cf81484189b3c71ea`.

Scope: all 153 keys (tui.community 148 + wire.community 5). Guides: i18n/glossary.md, i18n/style-guides/ur.md; terminology compared with local-operator-ui origin/feat/community-ui @ 1b1c4d31875dcbc44cc76f82b9a339278d93419d src/i18n/catalogues/<loc>/ui.community.json.

Mechanical: PASS (0 errors) — see audit.log.

Findings:
- **A-ur-N1** (NIT) `tui.community.category_spam_or_promotion` — 'غیر ظاہر کردہ تشہیر' vs desktop UI audited 'غیر ظاہر شدہ تشہیر'.
- **A-ur-M1** (MINOR) `status_guidelines, status_local_command, status_restricted` — Fit: at 80 columns the single-row status line (.community-view-status height: 1; column = 80 − RAIL_WIDTH 22 = 58 cells) cuts these strings (cell_len with sample values, en→ur): status_guidelines 48→67; status_local_command 53→69; status_restricted 60→67. Meaning intact at ≥100 columns; en already overflows status_restricted (60). Renderer/layout lead plus optional shorter wording; not a wording defect.

Verdict: PASS at the audited bytes; 0 blocker/major; 1 minor, 1 nit, all non-gating and deferred in the PR thread; no wording changed. Linguistic/copy audit only — no rendered native flow or RTL paint proof is claimed.
