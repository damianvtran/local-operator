# zh-CN cli.permissions — independent re-audit zhp3 (round 3)

PR #3691 · worktree @85fb35811f9d17fb6dfadfc2d5df3269db04bd12 · 2026-11-09 · auditor session e1cf8e96b473 (anthropic/claude-opus-5-5), independent of the translator commit 85fb35811f9 and of prior auditors 6b7e2046be3d (zhp1) and bb7ced26a175 (zhp2).

Inputs: en 58e58c3b…, context 6f8427bf…, zh-CN 7388180e… (full hashes in record.json / audit.log).

Scope (round 2+): the zhp2 findings Z8–Z11, plus the keys changed in en/zh-CN by 788cc399..85fb3581: status.scope, setup.already_signed, the new setup.always_allow_kept, and the 6 keys where the translator renamed the Screen Recording service per Z10. Unchanged keys that earlier rounds approved were not re-audited.

Verdict: **passed** — 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT.

## Mechanical

check.py rc=0 (catalogues consistent). My own parity script (audit.log): 39/39 keys, nothing added or dropped. Placeholders, backtick code and kept tokens match the source. Trailing spaces and newline counts match. No spaces at CJK–Latin junctions and no "...". The namespace has no ICU plural/select and no maxCells/maxChars hints.

## Round-2 findings

- Z8 (MAJOR): fixed. status.screen.denied now reads “系统设置”>“隐私与安全”>“录屏与系统录音”. All three labels match the installed macOS 27.2 tables (below).
- Z9 (MINOR): resolved by the source change. en status.scope no longer names Accessibility, and zh-CN dropped “辅助功能” to match: 仅检查录屏。“文件与文件夹”及其他隐私服务使用同一条目。 No “辅助功能” or “无障碍” remains in the namespace.
- Z10 (NIT): applied in all 6 keys (help.status, status.screen.granted/denied/unknown/no_image, status.scope). 屏幕录制 now appears 0 times. 录屏 matches SecurityPrivacyExtension SCREEN_CAPTURE.
- Z11 (NIT): applied. done_allow_once now reads 在终端中运行…之前，这些构建的运行时将使用…. 于 appears 0 times, and 将 correctly marks the future.

## Changed/new keys

- setup.already_signed: 此Python构建的运行时之前已签名，因此本次未进行签名，macOS也无需询问。 This is faithful to the shortened en. As the context requires, it no longer mentions the Always Allow answer.
- setup.always_allow_kept (new): 之前为此密钥选择的“始终允许”仍然保留。 The meaning and polarity are right: an earlier answer is kept, nothing is newly granted. “始终允许” is Apple's exact ConfirmAccess label. 密钥 matches the catalogue's existing term for the signing key (optin.unavailable, setup.explain) and Keychain Access's 'Keys' = 密钥. 选择 echoes the setup.always_allow prompt (是否已选择“始终允许”), so the follow-up reads as an answer to that prompt. The line drops the pronoun, as the style guide prefers. It reads as native product copy.
- status.scope: covered under Z9.

## Apple label verification (macOS 27.2 build 26B123, installed .loctable, zh_CN)

| Label | Table / key | zh_CN | Catalogue |
|---|---|---|---|
| System Settings | System Settings.app InfoPlist CFBundleDisplayName | 系统设置 | ✓ |
| Privacy & Security | System Settings.app Localizable "Privacy & Security"; SecurityPrivacyExtension InfoPlist CFBundleDisplayName | 隐私与安全 | ✓ |
| Screen & System Audio Recording | SecurityPrivacyExtension Localizable SCREEN_CAPTURE_HEADER | 录屏与系统录音 | ✓ |
| Screen Recording | SecurityPrivacyExtension Localizable SCREEN_CAPTURE | 录屏 | ✓ |
| Files & Folders | SecurityPrivacyExtension Localizable FILE_ACCESS_COMBINED | 文件与文件夹 | ✓ |
| Always Allow / Allow | SecurityAgent ConfirmAccess 100267.title / 100268.title | 始终允许 / 允许 | ✓ |
| Keys | Keychain Access Localizable "Keys" | 密钥 | ✓ |

## Out-of-scope observation (not a finding, does not gate)

setup.explain is unchanged since zhp2 and earlier rounds approved it. It says 选中该密钥的私钥, while Keychain Access's Kind column shows “专用密钥” for 'private key'. This is descriptive text, not a quoted label, and the user can still follow it, so it does not meet the BLOCKER bar for re-opening an unchanged key. It is noted here for whoever next touches that key.
