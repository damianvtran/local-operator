# Vietnamese core literals — translation note

- Locale: vi. Translator: i18n-translator; session 715297bf2b3b; actual translation model openai/gpt-6.1-sol (provider-quota fallback from anthropic/claude-sonnet-5-5, with an intervening claude-opus-5-5 fallback notice). Translation was reasoned directly; no translation service used.
- Files: local_operator/i18n/catalogues/vi/cli.failures.json (4 keys), cli.group_help.json (10 keys), tui.setup_cues.json (2 keys): 16 keys total.
- Inputs: English catalogues and all three context sidecars; shared glossary; vi style guide; RFC catalogue/audit sections; AGENTS.md internationalization section. All three target files were absent before writing.

## Terminology and interpretation

- Error wrapper: `{prog}: lỗi: {message}` retains the required structure. Missing required arguments uses terse `thiếu đối số bắt buộc`; invalid choices uses `lựa chọn không hợp lệ`. All placeholders remain verbatim, so quoted choice values, command names and the literal `None` are supplied unchanged.
- Followed glossary/style: tác nhân, phiên, kỹ năng, phê duyệt, mã hóa; kept runtime, mesh, Radient and Agent Hub. `pull` remains English per the glossary's kept-English git/hub operations rule.
- Bundle rendered `gói` / `gói kỹ năng`; Unix execute bit rendered `bit thực thi`. Explicit refusal of any execute bit and the guarantee that installation never runs code are both retained.
- Operator means `người vận hành`, not a telephone operator. The key may `nới lỏng yêu cầu phê duyệt đang áp dụng`, describing the active approval restriction rather than disabling all safety. Phone pairing retains the one-time signature authorization.
- `without a prompt` means `không cần hỏi lại`, not absence of an AI prompt. Headless fetching is described as `không hiển thị cửa sổ` to make the user-visible behavior clear.
- Glossary proposals (not edits): bundle → gói kỹ năng; execute bit → bit thực thi; operator → người vận hành; headless → không hiển thị cửa sổ. These were needed here and are not new glossary entries.

## Actual verification

Executed a read-only Python probe using /Users/damian/local-operator/.venv/bin/python. Actual output:

```text
cli.failures: 4 keys; key order, placeholders, canonical JSON, NFC OK
cli.group_help: 10 keys; key order, placeholders, canonical JSON, NFC OK
tui.setup_cues: 2 keys; key order, placeholders, canonical JSON, NFC OK
no_provider_cue: Kết nối tài khoản AI trước: nhập /login radient
rich.cells.cell_len: 47 / 74
Total: 16 keys
```

The cue fits with 27 cells spare; no width exception or truncation needed. All file bytes were read back by the probe and match ensure_ascii=False, indent=2, one trailing newline. No generate.py, test suite, git commands, ledger changes or generated-file changes were made. Full catalogue validator was not run under this task's narrow write/stop scope; coordinator must run it after integration. This is translation plus structural verification, not an independent locale audit.
