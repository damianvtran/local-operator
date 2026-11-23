# i18n audit — zh-CN · cli.daemon_build · r1-6236

Round 1 · 2026-11-18 · PR https://github.com/damianvtran/local-operator/pull/4215 · worktree head `2cc53faf08f6e5a7295b5e3b6a512f3b5a6ecbb3` · 5/5 keys.
Auditor: i18n-auditor (lopdev), session 623609b4ac65, anthropic/claude-opus-5-5. Translator: PR coder subagent
(anthropic/claude-opus-5-5); independent agent instance. No MT/translation service used.

**Verdict: changes_requested** — 0 BLOCKER, 1 MAJOR, 1 MINOR, 0 NIT.

Mechanical: valid JSON, 5 keys = en, no added/dropped keys, `{generation}`/`{command}` and backtick
spans identical to en (audit.log `[parity]`). No plurals in this namespace (no ICU). `check.py --audit-base origin/main` ok.

Semantics checked for every key: install_restarted_unconfirmed must not claim the old build was stale;
install_unconfirmed must stay a hedge. status_stale is printed bare (no label/subject) by
`services.stale_daemon_status_line` → `cli.py` / `tunnels/cli.py`.

## Findings

- **ZH-1 — MAJOR** · `all 5 keys`
  - current: "新版本" / "旧版本" / "当前版本"
  - problem: Terminology + accuracy: established zh-CN term is 构建 (cli.services_help.* 切换至当前构建; 28 of 33 build-keys). cli.services_help.status_description itself says the common drift here is a same-version rebuild invisible to the version number — 版本 names exactly the wrong thing.
  - suggested: 构建: "已将正在运行的服务重启到新构建（它此前仍在运行{generation}）", "正在运行旧构建（{generation}）；请运行`{command}`将其重启到新构建".
- **ZH-2 — MINOR** · `install_restarted, install_restart_failed, install_unconfirmed, status_stale`
  - current: "运行 {generation}", "运行 `{command}` 将"
  - problem: zh-CN style guide: no space between Chinese and Latin/placeholder runs (catalogue: 459 unspaced vs 25 spaced before {; 103 vs 5 before `).
  - suggested: Drop the spaces around {generation} and `{command}`.

## Source finding (all locales, for the coder)

- **S-1 — MINOR (source context, all locales)** · `cli.daemon_build.status_stale context note`
  - current: "Shown after a 'build:' label."
  - problem: The code prints the line bare (cli.py mobile status and tunnels/cli.py: "printed bare … adds no untranslated word"). The note led translators to subject-less fragments (FR-3, ES-2).
  - suggested: Fix the context note: "Printed on its own line, no label or subject; the subject is the service."

Not covered: no rendered frame (TUI/RTL) and no human native review; the runtime FSI/PDI isolation of placeholders was not exercised here.
