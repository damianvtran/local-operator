# zh-CN cli.permissions — independent re-audit zhp2 (round 2)

PR #3691 · worktree @788cc399410196b080cfc9f7e7bd558ccb713140 · 2026-11-09 · auditor session bb7ced26a175 (anthropic/claude-opus-5-5), independent of the round-2 translator (one i18n-translator per locale; session id not recorded in the repo or brief), of translator job ad99821ab9e1 and of round-1 auditor 6b7e2046be3d.

Inputs: en d5eed404…, context f2d9ce73…, zh-CN 2b018162… (full hashes in record.json / audit.log).

Verdict: **changes_requested** — 1 MAJOR (Z8), 1 MINOR (Z9), 2 NIT (Z10, Z11). No BLOCKER: all 38 keys keep meaning, polarity, placeholders and code. The safety copy is accurate: the “始终允许” bullet in setup.explain states that afterwards any program running as the user can sign with the key via codesign without asking, and gives the undo path (钥匙串访问 → 私钥 → 访问控制 → 移除codesign); “允许” = this time only; y/N defaults kept.

## Mechanical

check.py rc=0 (catalogues consistent). Own parity script (audit.log): 38/38 keys, no added or dropped keys, placeholders/backtick code/kept tokens identical, trailing spaces and newline counts match (explain keeps its two "- " bullets), no CJK–Latin spaces, no "...". No ICU plural/select in the namespace. No maxCells/maxChars hints. The only half-width punctuation in prose is the ">" breadcrumb, which copies the source.

## Round-1 findings

- Z1 spacing: resolved (no CJK–Latin junction spaces left).
- Z2 row → 条目: resolved in all 7 keys and used the same way in the 10 new keys.
- Z3 请: resolved (no 请 left).
- Z4 pane name: resolved. 录屏与系统录音 matches Apple (below).
- Z5, Z6, Z7: resolved as suggested.

## Apple label verification (macOS 27.2 build 26B123, installed .loctable, en → zh_CN)

| Label | Table / key | zh_CN | Catalogue |
|---|---|---|---|
| System Settings | System Settings.app InfoPlist.loctable CFBundleDisplayName | 系统设置 | ✓ |
| Privacy & Security | System Settings.app Localizable.loctable "Privacy & Security"; SecurityPrivacyExtension.appex InfoPlist.loctable CFBundleDisplayName | 隐私与安全 | ✗ 隐私与安全性 (Z8) |
| Screen & System Audio Recording | SecurityPrivacyExtension Localizable.loctable SCREEN_CAPTURE_HEADER / SCREENANDAUDIOCAPTURE | 录屏与系统录音 | ✓ |
| Files & Folders | SecurityPrivacyExtension Localizable.loctable FILE_ACCESS_COMBINED | 文件与文件夹 | ✓ |
| Accessibility | System Settings.app Localizable.loctable "Accessibility" | 无障碍 | ✗ 辅助功能 (Z9) |
| Keychain Access | Keychain Access.app InfoPlist.loctable CFBundleDisplayName | 钥匙串访问 | ✓ |
| Access Control | Keychain Access.app KeyEditor.loctable 302.label | 访问控制 | ✓ |
| Always Allow / Allow | SecurityAgent.bundle ConfirmAccess.loctable 100267.title / 100268.title | 始终允许 / 允许 | ✓ |

## Findings

### Z8 — MAJOR
Key: cli.permissions.status.screen.denied

The breadcrumb tells the user where to click, and the context says to use the label exactly as localized macOS shows it. On the installed macOS 27.2, both the System Settings sidebar and the pane's own bundle name are 隐私与安全, not 隐私与安全性. 隐私与安全性 matches what earlier macOS releases showed, from recollection only. That could not be verified on this Mac, and it does not match the current label. The rest of the breadcrumb is correct.

Current: 屏幕录制（运行时预检）：未允许。在“系统设置”>“隐私与安全性”>“录屏与系统录音”中打开“{row}”。

Suggested: 屏幕录制（运行时预检）：未允许。在“系统设置”>“隐私与安全”>“录屏与系统录音”中打开“{row}”。

### Z9 — MINOR
Key: cli.permissions.status.scope

辅助功能 is the pre-macOS-26 zh-CN name. On this Mac, System Settings' "Accessibility" is 无障碍. This line is information, not a click instruction, so MINOR.

Source-level note for the en owner (does not gate zh-CN): in SecurityPrivacyExtension Localizable.loctable, key ACCESSIBILITY reads "Device Control and Data Access" (zh_CN 设备控制和数据访问) on 27.2. That suggests Apple renamed the privacy-list row, which would make the English "Accessibility" stale as well. Confirm it visually before changing en.

Current: 仅检查屏幕录制。“文件与文件夹”“辅助功能”及其他隐私服务使用同一条目。

Suggested: 仅检查屏幕录制。“文件与文件夹”“无障碍”及其他隐私服务使用同一条目。 (or “设备控制和数据访问” if en adopts the macOS 27 row name)

### Z10 — NIT
Keys: cli.permissions.help.status, cli.permissions.status.screen.granted/denied/unknown/no_image, cli.permissions.status.scope

The service name is 屏幕录制, but Apple's own zh_CN term for the Screen Recording service is 录屏 (SecurityPrivacyExtension SCREEN_CAPTURE). The current wording is clear; switching to 录屏 would match the pane name the user sees.

Suggested: 录屏（运行时预检）：已允许 … / 是否已允许录屏（只读）/ 仅检查录屏。

### Z11 — NIT
Key: cli.permissions.setup.done_allow_once

在你于终端中运行 — 于 is literary in a terse CLI line.

Suggested: …；在终端中运行`lop update`或`lop permissions setup`之前，这些构建的运行时将使用“Local Operator Runtime”条目。
