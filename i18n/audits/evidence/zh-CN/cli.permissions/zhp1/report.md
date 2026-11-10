# zh-CN cli.permissions — independent audit zhp1

PR #3691 · worktree @50badda034d755ebb049c4f542f3073439bf823e · 2026-11-09 · auditor session 6b7e2046be3d (anthropic/claude-opus-5-5), independent of translator job ad99821ab9e1.

Verdict: **changes_requested** — 2 MAJOR (Z1, Z2), 2 MINOR, 3 NIT. No BLOCKER: every key preserves meaning, polarity and placeholders; the safety copy (what to click: 系统设置>隐私与安全性, 始终允许 matches the macOS zh-CN keychain button; y/N defaults; 已取消，未做任何更改) is accurate.

Mechanical: check.py rc=0 (catalogues consistent); own parity check clean (28/28, placeholders, code spans, kept tokens, trailing spaces); no plurals in namespace. Glossary: 会话 used correctly; 钥匙串, 签名身份, 映像 (matches cli.serve_help 进程映像) consistent.

## Findings

### Z1 — MAJOR
Keys: cli.permissions.help.permissions, cli.permissions.help.setup, cli.permissions.status.not_macos, cli.permissions.status.identity.legacy, cli.permissions.status.optin.on, cli.permissions.status.optin.off, cli.permissions.status.app.found, cli.permissions.setup.refused.tty, cli.permissions.setup.refused.isolated, cli.permissions.setup.no_app, cli.permissions.setup.no_identity, cli.permissions.setup.explain, cli.permissions.setup.failed, cli.permissions.setup.done

Space inserted between Chinese and Latin/code/placeholder runs throughout, contrary to the zh-CN style guide (“No space inserted between Chinese text and Latin/digit runs”) and to the rest of the zh-CN catalogue (2480 unspaced vs 116 spaced junctions in other namespaces).

Current: 查看或设置 Local Operator 运行时使用的 macOS 隐私身份

Suggested: 查看或设置Local Operator运行时使用的macOS隐私身份 — apply to every listed key; keep spaces only inside Latin/code runs (e.g. `lop permissions setup`, Developer ID Application) and the fixed “[y/N] ” tail.

### Z2 — MAJOR
Keys: cli.permissions.status.identity.team, cli.permissions.status.identity.adhoc, cli.permissions.status.identity.legacy, cli.permissions.status.optin.off, cli.permissions.setup.no_identity, cli.permissions.setup.failed, cli.permissions.setup.done

System Settings list "row" calqued as 这一行/一行. In Chinese UI copy a toggle in the Privacy list is a 项/条目; 共用一行 standing alone (optin.off, done) reads as "share a line (of text)" and does not tell the user which System Settings entry is meant — this is the copy that tells them what to enable.

Current: 运行 `lop permissions setup` 即可与桌面应用共用一行。 / 与桌面应用共用“Local Operator”这一行 / 每个 Python 构建各占一行

Suggested: identity.team: 运行时身份：{identifier}，与桌面应用共用“Local Operator”条目 · identity.adhoc: 运行时身份：{identifier}，使用单独的“Local Operator Runtime”条目 · identity.legacy: 运行时身份：按路径识别的映像（此处无法创建bundle），因此每个Python构建各有单独条目 · optin.off: 团队签名：未开启。运行`lop permissions setup`即可与桌面应用共用同一隐私权限条目。 · no_identity/failed: …运行时继续使用“Local Operator Runtime”条目。 · done: 完成。新的运行时将使用{identifier}，并与桌面应用共用同一条目。正在运行的会话在重新启动前继续使用当前映像。

### Z3 — MINOR
Keys: cli.permissions.status.screen.denied, cli.permissions.setup.refused.tty, cli.permissions.setup.refused.isolated, cli.permissions.setup.explain

请 added where the English has no "please"; style guide: 请 only where English says please (blocking dialogs).

Current: 请在“系统设置”>…中打开“{row}”。 / …需要交互，请在终端中运行。 / 请在你自己的登录 shell 中运行。 / 请选择“始终允许”，这样…

Suggested: 在“系统设置”>“隐私与安全性”>“屏幕录制”中打开“{row}”。 · `lop permissions setup`为交互式命令，需在终端中运行。 · 改为在自己的登录shell中运行。 · 选择“始终允许”，之后运行`lop update`时即可直接签名，不再询问。

### Z4 — MINOR
Keys: cli.permissions.help.status, cli.permissions.status.screen.granted, cli.permissions.status.screen.denied, cli.permissions.status.screen.unknown

SOURCE-LEVEL, not a translation defect: context asks for the localized macOS setting name, but since macOS Sequoia 15 the pane is "Screen & System Audio Recording" (zh-CN 录屏与系统录音; Apple zh-cn Mac User Guide title "在 Mac 上控制对录屏和系统录音的访问"). 屏幕录制 faithfully matches the stale English/Sonoma name. Fix en + all locales together; does not gate zh-CN.

Current: “系统设置”>“隐私与安全性”>“屏幕录制”

Suggested: When en is updated: “系统设置”>“隐私与安全性”>“录屏与系统录音” (confirm exact label on a zh-CN macOS 15+/26 UI).

### Z5 — NIT
Keys: cli.permissions.setup.always_allow

根本 adds emphasis absent from source; 你 is droppable per guide.

Current: 你是否选择了“始终允许”（或者根本没有出现对话框）？[y/N] 

Suggested: 是否已选择“始终允许”（或未出现对话框）？[y/N] 

### Z6 — NIT
Keys: cli.permissions.status.identity.none

下次启动运行时 can parse as "when next started and running" (运行时 = runtime or "while running").

Current: 运行时身份：尚未创建；下次启动运行时会自动创建

Suggested: 运行时身份：尚未创建；将在下次有运行时启动时创建

### Z7 — NIT
Keys: cli.permissions.setup.confirm

Terser per guide.

Current: 继续吗？[y/N] 

Suggested: 继续？[y/N] 
