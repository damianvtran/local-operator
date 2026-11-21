# i18n audit — vi · cli.daemon_build · r1-6236

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

- **VI-1 — MAJOR** · `install_restart_failed`
  - current: "không thể khởi động lại dịch vụ đang chạy trên bản dựng mới"
  - problem: Accuracy in an error line: "dịch vụ đang chạy trên bản dựng mới" is the default parse ("the service running on the new build"), i.e. it states the service is ALREADY on the new build, the opposite of the fault.
  - suggested: "không thể đưa dịch vụ đang chạy lên bản dựng mới; hãy chạy `{command}` để chuyển nó sang" (matches cli.services_help "Đưa … lên bản dựng").
- **VI-2 — MINOR** · `install_restarted, status_stale`
  - current: "khởi động lại dịch vụ đang chạy trên bản dựng mới (…)" / "khởi động lại nó trên bản dựng mới"
  - problem: Same attachment ambiguity; the parenthetical rescues key 1 but it still reads backwards first.
  - suggested: "đã khởi động lại dịch vụ đang chạy sang bản dựng mới (trước đó nó vẫn chạy {generation})"; status_stale: "…để khởi động lại nó sang bản dựng mới".

## Source finding (all locales, for the coder)

- **S-1 — MINOR (source context, all locales)** · `cli.daemon_build.status_stale context note`
  - current: "Shown after a 'build:' label."
  - problem: The code prints the line bare (cli.py mobile status and tunnels/cli.py: "printed bare … adds no untranslated word"). The note led translators to subject-less fragments (FR-3, ES-2).
  - suggested: Fix the context note: "Printed on its own line, no label or subject; the subject is the service."

Not covered: no rendered frame (TUI/RTL) and no human native review; the runtime FSI/PDI isolation of placeholders was not exercised here.
