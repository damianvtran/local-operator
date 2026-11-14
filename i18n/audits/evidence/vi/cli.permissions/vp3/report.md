# Audit report — vi / cli.permissions — vp3 (round 3, independent re-audit)

Auditor: i18n-auditor (lopdev team), session `b03c86526f84`, actual model
`anthropic/claude-opus-5-5` (session system-prompt `Model:` line; no fallback
observed). Independent of the translators (commits 788cc399410, 85fb35811f9)
and of prior auditors (vp1 `fe66a4c43d27`, vp2 `bac062ad618b`). No MT service.
Date 2026-11-09. Worktree HEAD `85fb35811f9d17fb6dfadfc2d5df3269db04bd12`.

Scope (round 2+ rule): vp2 findings v2-m1, v2-n1; keys changed/added in
`788cc399..85fb3581` (en + vi): `setup.already_signed`, `setup.always_allow_kept`
(new), `status.scope`, `setup.explain` (vi-only remediation). Unchanged keys
retained from vp2 (passed). Call site read: `permissions_cli._setup()` l.301-308
(`always_allow_kept` prints only when `kept`).

## Verdict: passed — 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT

### Round-2 findings
| id | key | status | current vi |
|---|---|---|---|
| v2-m1 (MINOR) | setup.explain | fixed | “Luôn Cho phép” giúp macOS không hỏi lại nữa, … |
| v2-n1 (NIT) | setup.explain | fixed | … chọn khóa cá nhân tương ứng, rồi xóa codesign … |
No other byte of setup.explain changed (diffed).

### Changed / new keys
- `setup.already_signed` — “Runtime của bản build Python này đã được ký từ trước,
  nên lần này không ký gì và macOS không có gì để hỏi.” Faithful; the
  “Luôn Cho phép” sentence is removed as the context note requires.
- `setup.always_allow_kept` (new) — “Câu trả lời “Luôn Cho phép” trước đó của bạn
  cho khóa này vẫn được giữ.” Accurate (earlier/confirmed answer, for this key,
  kept); “khóa này” correctly refers to the Developer ID signing key; register
  (bạn) and “ ” quotes per style guide; Apple label exact.
- `status.scope` — “Chỉ kiểm tra Ghi Màn hình. Tệp & Thư mục và các dịch vụ quyền
  riêng tư khác cũng dùng chung mục này.” Accessibility/Trợ năng removed as the
  source and context note require; no stale pane name remains.

### Mechanical (executed — audit.log)
check.py exit 0 (“catalogues consistent”); parity script (vp2's, reused
unchanged): 39/39 keys, 0 failures (placeholders, backticks, newlines, [y/N]
trailing space, NFC, quotes, verbatim tokens). No ICU in source.

### macOS labels in changed keys — Apple's installed vi tables (macOS 27.2, 26B123)
- Ghi Màn hình — SecurityPrivacyExtension Localizable.loctable :: SCREEN_CAPTURE
- Tệp & Thư mục — same :: FILE_ACCESS_COMBINED
- Luôn Cho phép — SecurityAgent ConfirmAccess.loctable :: 100267.title (en_GB “Always Allow”)
- Truy cập chuỗi khóa / khóa cá nhân / Kiểm soát truy cập (setup.explain, unchanged labels) — Keychain Access.app tables, as in vp2.

Limits: no rendered frame, no human native-speaker review.
