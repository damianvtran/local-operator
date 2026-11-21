# i18n audit — es · cli.daemon_build · r1-6236

Round 1 · 2026-11-18 · PR https://github.com/damianvtran/local-operator/pull/4215 · worktree head `2cc53faf08f6e5a7295b5e3b6a512f3b5a6ecbb3` · 5/5 keys.
Auditor: i18n-auditor (lopdev), session 623609b4ac65, anthropic/claude-opus-5-5. Translator: PR coder subagent
(anthropic/claude-opus-5-5); independent agent instance. No MT/translation service used.

**Verdict: changes_requested** — 0 BLOCKER, 2 MAJOR, 1 MINOR, 0 NIT.

Mechanical: valid JSON, 5 keys = en, no added/dropped keys, `{generation}`/`{command}` and backtick
spans identical to en (audit.log `[parity]`). No plurals in this namespace (no ICU). `check.py --audit-base origin/main` ok.

Semantics checked for every key: install_restarted_unconfirmed must not claim the old build was stale;
install_unconfirmed must stay a hedge. status_stale is printed bare (no label/subject) by
`services.stale_daemon_status_line` → `cli.py` / `tunnels/cli.py`.

## Findings

- **ES-1 — MAJOR** · `all 5 keys`
  - current: "versión" / "nueva versión" / "versión anterior"
  - problem: Terminology + accuracy: established es term for build in the same services/update area is "compilación" (cli.services_help.*, cli.update_help.no_services; 26 of 33 build-keys). A same-version rebuild makes "versión anterior" potentially false.
  - suggested: "compilación": "la nueva compilación", "una compilación anterior", "su compilación".
- **ES-2 — MAJOR** · `status_stale`
  - current: "ejecuta una versión anterior ({generation}); ejecuta `{command}` …"
  - problem: Printed bare with no subject, so the first "ejecuta" reads as the tú imperative of the second: "run an older version (…); run `lop mobile restart`" — an instruction to run the old build.
  - suggested: "en ejecución con una compilación anterior ({generation}); ejecuta `{command}` para reiniciarlo con la nueva compilación".
- **ES-3 — MINOR** · `install_restart_failed`
  - current: "reiniciar el servicio en ejecución con la nueva versión"
  - problem: "el servicio en ejecución con la nueva versión" can parse as "the service running with the new version".
  - suggested: "no se pudo pasar el servicio en ejecución a la nueva compilación; ejecuta `{command}` para hacerlo".

## Source finding (all locales, for the coder)

- **S-1 — MINOR (source context, all locales)** · `cli.daemon_build.status_stale context note`
  - current: "Shown after a 'build:' label."
  - problem: The code prints the line bare (cli.py mobile status and tunnels/cli.py: "printed bare … adds no untranslated word"). The note led translators to subject-less fragments (FR-3, ES-2).
  - suggested: Fix the context note: "Printed on its own line, no label or subject; the subject is the service."

Not covered: no rendered frame (TUI/RTL) and no human native review; the runtime FSI/PDI isolation of placeholders was not exercised here.
