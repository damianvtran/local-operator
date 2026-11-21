# i18n audit — fr · cli.daemon_build · r1-6236

Round 1 · 2026-11-18 · PR https://github.com/damianvtran/local-operator/pull/4215 · worktree head `2cc53faf08f6e5a7295b5e3b6a512f3b5a6ecbb3` · 5/5 keys.
Auditor: i18n-auditor (lopdev), session 623609b4ac65, anthropic/claude-opus-5-5. Translator: PR coder subagent
(anthropic/claude-opus-5-5); independent agent instance. No MT/translation service used.

**Verdict: changes_requested** — 0 BLOCKER, 3 MAJOR, 2 MINOR, 0 NIT.

Mechanical: valid JSON, 5 keys = en, no added/dropped keys, `{generation}`/`{command}` and backtick
spans identical to en (audit.log `[parity]`). No plurals in this namespace (no ICU). `check.py --audit-base origin/main` ok.

Semantics checked for every key: install_restarted_unconfirmed must not claim the old build was stale;
install_unconfirmed must stay a hedge. status_stale is printed bare (no label/subject) by
`services.stale_daemon_status_line` → `cli.py` / `tunnels/cli.py`.

## Findings

- **FR-1 — MAJOR** · `all 5 keys`
  - current: "version" / "nouvelle version" / "version plus ancienne"
  - problem: Terminology + accuracy: "build" is rendered "version", but the established fr term in the same daemon/services area is "build" (cli.services_help.*, cli.update_help.no_services: "le nouveau build"; 23 of 33 build-keys). The distinction is load-bearing here: a same-version rebuild (lop-update) produces a new generation with an unchanged version number, so "exécute une version plus ancienne" can be false when `lop --version` shows the same version.
  - suggested: Use "build" (masc.): "nouveau build", "build plus ancien", "son build".
- **FR-2 — MAJOR** · `install_restarted, install_restarted_unconfirmed, install_restart_failed, install_unconfirmed`
  - current: "service en cours"
  - problem: "en cours" alone means "in progress/current", not "running"; "le service en cours" reads as "the current service". Calque of "running service".
  - suggested: "service en cours d’exécution" (or "service actif"). E.g. install_restarted: "service en cours d’exécution redémarré sur le nouveau build (il tournait encore sur {generation})".
- **FR-3 — MAJOR** · `status_stale`
  - current: "exécute une version plus ancienne ({generation}) ; …"
  - problem: Printed bare (services.stale_daemon_status_line → cli.py / tunnels/cli.py print it with no label or subject), so a subject-less finite verb reads as a broken sentence in French.
  - suggested: "build plus ancien en cours d’exécution ({generation}) ; lancez `{command}` pour le redémarrer sur le nouveau build".
- **FR-4 — MINOR** · `install_restart_failed`
  - current: "pour le faire passer dessus"
  - problem: Colloquial and vague ("dessus" = on top of it).
  - suggested: "pour l’amener sur le nouveau build" (matches cli.services_help "Amener … sur le build courant").
- **FR-5 — MINOR** · `install_restart_failed, install_unconfirmed, status_stale`
  - current: " ;" (U+0020 before ;)
  - problem: fr style guide §Punctuation: espace fine insécable U+202F before ; (410 catalogue uses vs 42 plain).
  - suggested: Replace the U+0020 before ";" with U+202F.

## Source finding (all locales, for the coder)

- **S-1 — MINOR (source context, all locales)** · `cli.daemon_build.status_stale context note`
  - current: "Shown after a 'build:' label."
  - problem: The code prints the line bare (cli.py mobile status and tunnels/cli.py: "printed bare … adds no untranslated word"). The note led translators to subject-less fragments (FR-3, ES-2).
  - suggested: Fix the context note: "Printed on its own line, no label or subject; the subject is the service."

Not covered: no rendered frame (TUI/RTL) and no human native review; the runtime FSI/PDI isolation of placeholders was not exercised here.
