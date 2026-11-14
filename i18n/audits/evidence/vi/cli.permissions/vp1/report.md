# Audit report — vi / cli.permissions — vp1

Independent auditor: i18n-auditor (lopdev team), session `fe66a4c43d27`,
actual model `anthropic/claude-opus-5-5` (the session's own model header; no
fallback observed). Translator: PR #3691 coder agent, session `ad99821ab9e1`,
`anthropic/claude-opus-5-5` — a different session; same model family (disclosed).
No MT/translation service was used. Date 2026-11-09. Worktree HEAD
`50badda034d755ebb049c4f542f3073439bf823e`.

Scope: all 28 keys of `cli.permissions` (`lop permissions status|setup`).
Inputs read: docs/I18N.md "Add or extend a locale" 2-6, i18n/glossary.md,
i18n/style-guides/vi.md, en catalogue + context sidecar, vi catalogue, and
`local_operator/permissions_cli.py` `_setup()` to check what each prompt gates.

## Verdict: passed — 0 BLOCKER, 0 MAJOR, 2 MINOR, 4 NIT

### Mechanical (see audit.log — executed)
28/28 keys, no extra/missing; `{identifier} {path} {identity} {team} {reason}
{row} {name}` identical; backtick commands, `guide://permissions`, `codesign`,
"Local Operator", "Local Operator Runtime", "Developer ID Application" verbatim;
`[y/N] ` with trailing space on both prompts; NFC; no ICU plurals in source.
`scripts/i18n/check.py` exit 0.

### Accuracy / safety
- macOS vi labels match Apple's Vietnamese UI as I know it: Cài đặt hệ thống >
  Quyền riêng tư & Bảo mật > Ghi màn hình; keychain button "Luôn cho phép";
  keychain → "chuỗi khóa". (The en source's "Screen Recording" is now
  "Screen & System Audio Recording" on macOS 15; source issue, not gated here.)
- setup.explain tells the user exactly what en says: macOS may ask once whether
  codesign may use the key; choose "Luôn cho phép" so later `lop update` runs
  sign without asking. No added/omitted claims.
- setup.cancelled "Không có gì thay đổi" correct (code cancels before writing).
- Glossary: runtime kept English, setup → thiết lập, session → phiên, team →
  nhóm, bạn register sparing. "ứng dụng máy tính" for desktop app follows the
  existing vi catalogue precedent (several namespaces).

### Findings
| ID | Sev | Key | Current | Problem | Suggested |
|---|---|---|---|---|---|
| m1 | MINOR | setup.always_allow | Bạn đã chọn "Luôn cho phép" (hoặc không có hộp thoại nào hiện ra) chưa? [y/N] | The "đã…chưa?" frame wraps the parenthetical, so it reads "…(or has no dialog appeared) yet?" — clumsy, and "chưa" invites a "not yet" reading although this prompt runs after signing finished. The answer gates `always_allow_confirmed` (agent-run updates signing without a dialog), so clarity matters. Meaning is still recoverable → MINOR. | Bạn đã chọn "Luôn cho phép" hoặc không thấy hộp thoại nào hiện ra, đúng không? [y/N] |
| m2 | MINOR | setup.failed | Việc ký chưa hoàn tất. … | "chưa hoàn tất" = "not complete yet", implies still in progress; this is a terminal failure (opt-in record is removed). | Việc ký không hoàn tất. Runtime vẫn giữ dòng "Local Operator Runtime". Chạy `lop permissions status` để xem chi tiết. |
| n1 | NIT | 7 keys with quoted labels | ASCII "…" | Style guide prefers “ ”; ASCII is tolerable here since they quote literal UI labels, and en uses ASCII. | “Local Operator Runtime”, “Luôn cho phép”, “{row}” … |
| n2 | NIT | status.identity.none | …; nó sẽ được tạo vào lần tới một runtime khởi động | "nó" + "vào lần tới một runtime" is a calque. | Danh tính runtime: chưa được tạo; sẽ được tạo khi runtime khởi động lần tới |
| n3 | NIT | identity.team/adhoc, optin.off, no_identity, failed, done | dòng | A System Settings list entry reads more natively as "mục"; "dòng" is literal for "row". Consistent within the file, so not gating. | mục "Local Operator" / mục riêng "Local Operator Runtime" |
| n4 | NIT | status.optin.* , setup.refused.isolated | Ký theo nhóm | "ký theo nhóm" can read as batch/group signing; the concept is signing with the Apple developer team identity. | Ký bằng nhóm nhà phát triển (or keep, given "(nhóm {team})" beside it) |

Not covered: no rendered terminal frame; no native-speaker human review; no
second independent pass (manager-requested separately if needed).
