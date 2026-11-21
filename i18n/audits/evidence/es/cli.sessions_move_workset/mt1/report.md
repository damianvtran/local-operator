# es — move working-set TRANSFER copy (PR #4245) — independent i18n audit (mt1)

Auditor: i18n auditor, lopdev task session 482db48afe47, served model anthropic/claude-opus-5-5 (no fallback), 2026-11-18.
Translators (lopdev manager brief): coder jobs 1768f308a175 (B2), 04a689e5090b / b57675c8318a (B3), b7523be856f2 (B4),
b404dd5868ee (B5), all anthropic/claude-opus-5-5. Independence: different agent and session from every translator; same model (disclosed).
Audited head: a69e0e74bd4980cd9e598aec61597c09b256ad7d (branch feat/move-working-set-transfer). Scope: every key 696a18338..a69e0e74b adds in
cli.sessions_move_workset (22 keys) and wire.settings (mesh_moves.*, mesh_move_working_set_max_bytes.*,
mesh_move_working_set_roots.*: 6 keys). No existing value changed; every other key is byte-identical to the copy whose
last receipt at the 696a18338 English digest is `passed` (verified per locale).

### cli.sessions_move_workset: PASS
- minor — too_many_worktrees says 'árboles de trabajo' while every other key keeps 'worktree'

### wire.settings: PASS
- no findings

Checked: idiom/register per i18n/style-guides/es.md; glossary §1/§2 (session, device, credentials, kept-English git verbs,
setting keys such as mesh.move.working_set_roots verbatim); ICU plural arms; placeholders; safety meaning (what was sent,
restored, left on {source}, refused by policy, not granted).

## Limits
- No native-speaker review; no rendered frame (evidence: product ICU runtime, audit.log).
- Safety-confirmation copy: docs/I18N.md step 3 asks for a second independent audit. This is one pass.
