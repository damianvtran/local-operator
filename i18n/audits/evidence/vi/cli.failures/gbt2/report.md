# vi / cli.failures — independent audit gbt2 (2026-11-16)

Auditor: i18n-auditor `audit2-vi`, session `99e712d49b8f`. Model: started on
`openai/gpt-6.1-sol`; that provider returned HTTP 429 after the first (blocked,
no-output) tool call and the session fell back to `anthropic/claude-opus-5-5`,
which performed all substantive audit work. Did not translate these strings.
Translator (per parent): `i18n-translator (timeout-vi)`, `openai/gpt-6.1-sol`
with fallback `anthropic/claude-opus-5-5` — same model families as this auditor;
independence is by session, not by model.

Revision: worktree `.worktrees/github-borrow-timeout-note` @ `80a55b208`
(base `f0689a27`). Inputs read: `i18n/style-guides/vi.md`, `i18n/glossary.md`,
`docs/I18N.md`, en `cli.failures.json` + `.context.json`, vi `cli.failures.json`.
No MT/translation service used; no earlier audit records for this key consulted.

## Verdict: passed — 0 BLOCKER, 0 MAJOR, 1 MINOR, 3 NIT (5/5 keys)

## Mechanical (check.py rc=0; own parity script — see audit.log)
- 5/5 keys, no add/drop; placeholders identical per key; no ICU plurals in
  group (vi `other`-only rule n/a); no backticks/markup; valid JSON; all NFC.
- `github_borrow_timeout` 91 cells vs en 83; no `maxCells`; one stderr line
  inside a tool result — acceptable.

## Per key
- `error_line` "{prog}: lỗi: {message}" — shape kept, correct. OK.
- `required` "thiếu đối số bắt buộc: {names}" — accurate, terse. OK.
- `invalid_choice` — accurate; gbt2-n2 phrasing nit.
- `invalid_command` "Lệnh {group} không hợp lệ: {command}" — OK.
- `github_borrow_timeout` "Chưa nhận được thông tin xác thực GitHub từ {owner}
  trong {seconds} giây; hãy chạy lại lệnh":
  - Glossary: credentials → «thông tin xác thực» ✓; GitHub kept ✓; unit
    `s` → «giây» natural ✓.
  - Constraints: no denial/revoke/login/token-lease wording ✓; single retry
    instruction matches source "retry the command" (rerun, not an extra
    automatic retry) ✓; «hãy chạy lại lệnh» matches the vi corpus («chạy lại
    lệnh này»). Register: no «bạn», no «vui lòng» ✓.
  - gbt2-m1 (MINOR): «Chưa» = "not yet" leans "still pending"; prefer
    «Không nhận được … trong vòng {seconds} giây». Not unsafe given the
    retry clause.
  - gbt2-n1 (NIT): optional «từ thiết bị {owner}».
  - gbt2-n3 (NIT): 1-space indent drift on the new line (all locales).

## Limits
No rendered frame, no human native-speaker review, no runtime exercise.
