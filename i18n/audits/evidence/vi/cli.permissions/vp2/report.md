# Audit report — vi / cli.permissions — vp2 (round 2, independent re-audit)

Auditor: i18n-auditor (lopdev team), session `bac062ad618b`, actual model
`anthropic/claude-opus-5-5` (session system-prompt `Model:` line; no fallback
observed). Independent of the round-2 translator (one i18n-translator per
locale, commit 788cc399410; session/model not recorded in the repo) and of the
round-1 auditor (session `fe66a4c43d27`, bundle `vp1`). No MT/translation
service used. Date 2026-11-09. Worktree HEAD
`788cc399410196b080cfc9f7e7bd558ccb713140`.

Scope: all 38 keys (28 round-1 + 10 new). Inputs: i18n/style-guides/vi.md,
i18n/glossary.md, en catalogue (d5eed404…) + context sidecar (f2d9ce73…), vi
catalogue (afbd41b9…), `local_operator/permissions_cli.py` `_setup()` (lines
~255-310) for what each prompt gates.

## Verdict: passed — 0 BLOCKER, 0 MAJOR, 1 MINOR, 1 NIT

### Mechanical (executed — audit.log)
`scripts/i18n/check.py` exit 0 ("catalogues consistent"). Independent parity
(`parity.py.txt`): 38/38 keys, none missing/extra; placeholders
`{identifier} {path} {identity} {team} {reason} {row} {name} {detail}`
identical; backtick spans identical and in order; `setup.explain` keeps both
`\n` bullet breaks; `[y/N] ` with trailing space on both prompts; NFC; no ASCII
`"` left; `Local Operator`, `Local Operator Runtime`, `Developer ID
Application`, `codesign`, `/update`, `guide://permissions`, env names and
`~/.local-operator` verbatim. No ICU plurals in source.

### macOS labels — verified against Apple's installed vi tables (macOS 27.2, 26B123)
| en label | vi in catalogue | Apple table :: key |
|---|---|---|
| System Settings | Cài đặt hệ thống | System Settings.app InfoPlist.loctable :: CFBundleDisplayName (vi, vi~mac) |
| Privacy & Security | Quyền riêng tư & Bảo mật | SecurityPrivacyExtension.appex InfoPlist.loctable :: CFBundleDisplayName |
| Screen & System Audio Recording | Bản ghi âm thanh hệ thống & màn hình | SecurityPrivacyExtension.appex Localizable.loctable :: SCREEN_CAPTURE_HEADER / SCREENANDAUDIOCAPTURE |
| Screen Recording (service) | Ghi Màn hình | same :: SCREEN_CAPTURE |
| Files & Folders | Tệp & Thư mục | same :: FILE_ACCESS_COMBINED |
| Accessibility | Trợ năng | AccessibilitySettingsExtension.appex InfoPlist.loctable :: CFBundleDisplayName |
| Allow / Always Allow | Cho phép / Luôn Cho phép | Security.framework SecurityAgent.bundle ConfirmAccess.loctable :: 100268.title / 100267.title (en_GB pair) |
| Keychain Access | Truy cập chuỗi khóa | Keychain Access.app InfoPlist.loctable :: CFBundleDisplayName |
| Access Control | Kiểm soát truy cập | Keychain Access.app KeyEditor.loctable :: 302.label |
| private key / keychain | khóa cá nhân / chuỗi khóa | Keychain Access.app Localizable.loctable :: "private key"; InfoPlist :: keychain |

All exact, including Apple's capital "C" in "Luôn Cho phép" and "M" in "Ghi
Màn hình" (Apple's casing overrides the sentence-case rule per the context
note). Round-1's note that the old "Ghi màn hình" pane name was stale is now
fixed in source and target.

### Safety copy (strict read)
- `setup.explain`: "Cho phép" = this signing only; terminal `lop update` asks
  again; terminal-less updates skip team signing → new builds use the
  "Local Operator Runtime" row. "Luôn Cho phép" disclosure is complete and not
  softened: *any program running as your macOS user* (bất kỳ chương trình nào
  chạy dưới tài khoản người dùng macOS của bạn) *can use codesign to sign with
  this Developer ID key without asking*. Undo path uses the exact Apple labels.
- `setup.always_allow` y/N: "Bạn đã chọn “Luôn Cho phép” hoặc macOS không hiện
  hộp thoại nào, đúng không?" — "có" ⇒ `always_allow_confirmed=True`, which is
  what the code does (line ~296). No polarity flip.
- `setup.failed` "không hoàn tất … mọi thiết lập trước đó không thay đổi"
  matches `_restore_optin(prior)`. `setup.cancelled` "Không có gì thay đổi"
  matches the cancel-before-write path.
- `already_signed`, `done_allow_once` faithful.

### Round-1 findings (vp1)
m1 resolved (wording as suggested); m2 resolved ("không hoàn tất"); n1 resolved
(all “ ”); n2 resolved; n3 resolved ("mục" throughout); n4 applied ("Ký bằng
nhóm nhà phát triển"; longer, but no maxCells budget in the sidecar).

### Terminology
runtime kept English (style guide wave-2); setup → thiết lập; session → phiên;
agent → tác nhân; team → nhóm; "bạn" only on the decision prompt/disclosure.
"image", "bản build", "terminal", "shell", "thư mục home" kept/precedented in
existing vi catalogues; consistent within the file.

### Findings
| ID | Sev | Key | Current | Problem | Suggested |
|---|---|---|---|---|---|
| v2-m1 | MINOR | setup.explain (bullet 2) | “Luôn Cho phép” sẽ không hỏi lại nữa, nhưng khi đó … | The button is the subject of "will not ask again" — illogical agent (macOS asks, not the button). en "stops the questions". Meaning recoverable. | “Luôn Cho phép” giúp macOS không hỏi lại nữa, nhưng khi đó … |
| v2-n1 | NIT | setup.explain (bullet 2) | chọn khóa cá nhân của khóa đó | "khóa … của khóa" repetition (mirrors en "the key's private key"). | chọn khóa cá nhân tương ứng |

Not covered: no rendered terminal frame; no human native-speaker review.
