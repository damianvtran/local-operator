# i18n audit — ur · cli.daemon_build · r1-6236

Round 1 · 2026-11-18 · PR https://github.com/damianvtran/local-operator/pull/4215 · worktree head `2cc53faf08f6e5a7295b5e3b6a512f3b5a6ecbb3` · 5/5 keys.
Auditor: i18n-auditor (lopdev), session 623609b4ac65, anthropic/claude-opus-5-5. Translator: PR coder subagent
(anthropic/claude-opus-5-5); independent agent instance. No MT/translation service used.

**Verdict: passed** — 0 BLOCKER, 0 MAJOR, 1 MINOR, 1 NIT.

Mechanical: valid JSON, 5 keys = en, no added/dropped keys, `{generation}`/`{command}` and backtick
spans identical to en (audit.log `[parity]`). No plurals in this namespace (no ICU). `check.py --audit-base origin/main` ok.

Semantics checked for every key: install_restarted_unconfirmed must not claim the old build was stale;
install_unconfirmed must stay a hedge. status_stale is printed bare (no label/subject) by
`services.stale_daemon_status_line` → `cli.py` / `tunnels/cli.py`.

## Findings

- **UR-1 — MINOR** · `all 5 keys`
  - current: "بِلڈ" (with zer U+0650)
  - problem: Established catalogue spelling is unvocalised "بلڈ" (42 uses, 0 with zer); Urdu UI copy does not write aerab, and the combining mark is an extra terminal width/shaping risk for no gain.
  - suggested: "بلڈ".
- **UR-2 — NIT** · `install_restart_failed`
  - current: "اسے وہاں منتقل کرنے کے لیے"
  - problem: "وہاں" (there) is vague.
  - suggested: "اسے نئے بلڈ پر لانے کے لیے".

## Source finding (all locales, for the coder)

- **S-1 — MINOR (source context, all locales)** · `cli.daemon_build.status_stale context note`
  - current: "Shown after a 'build:' label."
  - problem: The code prints the line bare (cli.py mobile status and tunnels/cli.py: "printed bare … adds no untranslated word"). The note led translators to subject-less fragments (FR-3, ES-2).
  - suggested: Fix the context note: "Printed on its own line, no label or subject; the subject is the service."

Not covered: no rendered frame (TUI/RTL) and no human native review; the runtime FSI/PDI isolation of placeholders was not exercised here.
