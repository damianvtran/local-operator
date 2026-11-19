# i18n audit — ru · cli.daemon_build · r1-6236

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

- **RU-1 — MINOR** · `install_restart_failed`
  - current: "перезапустить работающую службу на новой сборке; … чтобы перевести её"
  - problem: "службу на новой сборке" can attach to the noun ("the service on the new build"); "перевести её" lacks its goal.
  - suggested: "не удалось перевести работающую службу на новую сборку; выполните `{command}`, чтобы сделать это".
- **RU-2 — NIT** · `status_stale`
  - current: "чтобы перезапустить её на новой сборке"
  - problem: Bare line has no subject; nearest antecedent of "её" is "сборке".
  - suggested: "…, чтобы перезапустить службу на новой сборке".

## Source finding (all locales, for the coder)

- **S-1 — MINOR (source context, all locales)** · `cli.daemon_build.status_stale context note`
  - current: "Shown after a 'build:' label."
  - problem: The code prints the line bare (cli.py mobile status and tunnels/cli.py: "printed bare … adds no untranslated word"). The note led translators to subject-less fragments (FR-3, ES-2).
  - suggested: Fix the context note: "Printed on its own line, no label or subject; the subject is the service."

Not covered: no rendered frame (TUI/RTL) and no human native review; the runtime FSI/PDI isolation of placeholders was not exercised here.
