# Independent translation audit — zh-CN / wire.phone_access / a2

Commit: 0e4ad0a9171c59d7833d222761cbaa1f6d09bf51 (PR #4145, branch of worktree .worktrees/desk-presence-core). Date: 2026-11-17.
Auditor: i18n-auditor role, independent `task` subagent of team lopdev, Local Operator session 43bcb19f888e; not the translator (the coder that authored 9134015c with the draft values) nor the English-drafting coder; did not edit any catalogue value; model anthropic/claude-opus-5-5 (harness system prompt of session 43bcb19f888e (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: coder subagent of team lopdev that authored commit 9134015c (draft values and English source in the same commit); session id not disclosed in the brief or PR #4145; model not stated in the brief — unknown, not independently verified.
Source: en sha256 1d6935b1259cab3b9d4975c8b8149de76ceb61be177f22c6b4e01a289b6e762a (10 keys); context sidecar sha256 95c0c4b8d5f82cef978c751326ba6c214c1c8bf8152717f608eaeee943071c07; target sha256 f068d44a3df4b85a0d34beff18badbc23b2d1fdd545a9e1d99c909f698e42738.
Prior audit on disk: a1 (passed at en sha256 2979b226…, 4 keys); stale because the English source gained 6 keys.

**Verdict: passed** — 0 unresolved blocker/major; 2 minor/nit finding(s).

Scope: the 6 keys new in this PR (ADR 0008 `notifications.phone_away_after_s` registry row: label, help, 1/2/5/10-minute choices) judged in full — accuracy, register, placeholders, length/fit, glossary/corpus consistency. The 4 keys passed in a1 are byte-unchanged in both en and zh-CN (diff origin/main...HEAD touches only the appended lines), so their a1 judgement is retained; the receipt covers the whole namespace (10 keys) at the new English source hash. No other i18n keys are added or changed by the PR (checked: `git diff origin/main...HEAD -- local_operator/i18n` touches only wire.phone_access and its generated key modules).

Read: docs/I18N.md 'Add or extend a locale', i18n/glossary.md (no row for away/idle/phone; corpus usage governs), i18n/style-guides/zh-CN.md, en + context sidecar, settings_io.py registry row (ENUM: label rendered with the chosen value after it; choices also listed in the picker), corpus precedent wire.settings.session_cleanup_temporary_idle_minutes.* and existing 'desktop app' renderings.

## Per-key judgement (new keys)

| key | en | zh-CN | judgement |
|---|---|---|---|
| `wire.phone_access.away_after.label` | Send to phone when I'm away from my computer for | 离开电脑多久后改为发送到手机 | OK — «离开电脑多久后改为发送到手机» is a natural settings-label question answered by the value; no pronoun. |
| `wire.phone_access.away_after.help` | How long this computer can go without keyboard or mouse input before a finished task it would show you here goes to your phone instead. Locking the screen counts as away right away. Needs the desktop app. | 这台电脑在多长时间内没有键盘或鼠标操作后，原本会在这里提示你的已完成任务改为发送到你的手机。锁定屏幕会立即视为离开。需要桌面应用。 | OK in meaning; ZH-N2 nit on phrasing. |
| `wire.phone_access.away_after.choice.1m.label` | 1 minute | 1 分钟 | Meaning OK; ZH-M1 minor: digit–CJK space violates the style guide. |
| `wire.phone_access.away_after.choice.2m.label` | 2 minutes | 2 分钟 | Meaning OK; ZH-M1 minor: digit–CJK space violates the style guide. |
| `wire.phone_access.away_after.choice.5m.label` | 5 minutes | 5 分钟 | Meaning OK; ZH-M1 minor: digit–CJK space violates the style guide. |
| `wire.phone_access.away_after.choice.10m.label` | 10 minutes | 10 分钟 | Meaning OK; ZH-M1 minor: digit–CJK space violates the style guide. |

## Safety / meaning

No placeholders, URLs or commands in any new key (mechanical check in audit.log). The three load-bearing facts of the help text — the threshold is keyboard/mouse idle time, locking the screen counts as away immediately, and it needs the desktop app — are present in the zh-CN value.

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| ZH-M1 | minor | `wire.phone_access.away_after.choice.1m.label`, `wire.phone_access.away_after.choice.2m.label`, `wire.phone_access.away_after.choice.5m.label`, `wire.phone_access.away_after.choice.10m.label` | «1 分钟»/«2 分钟»/«5 分钟»/«10 分钟» insert a space between the digit and 分钟, against style guide zh-CN §Punctuation & typography ('No space inserted between Chinese text and Latin/digit runs') and the shipped precedent «15分钟»/«30分钟» (wire.settings.session_cleanup_temporary_idle_minutes.choice.*). Meaning unaffected. | «1分钟» «2分钟» «5分钟» «10分钟» |
| ZH-N2 | nit | `wire.phone_access.away_after.help` | «在多长时间内没有…操作后» stacks 在…内 and …后 (slightly awkward), and «提示你的…发送到你的手机» repeats 你的, which the style guide's pronoun-dropping rule discourages. | «这台电脑多长时间没有键盘或鼠标操作后，原本会在这里提示你的已完成任务改为发送到手机。…» |

Minor/nit findings are advisory and do not block; the coordinator may batch them into a translator revision (a value change would need a fresh receipt at the new target bytes).

## Not verified

No native-speaker review and no rendered desktop/TUI/phone frame of this settings row. Judgement by reasoning only; no MT service used. Translator identity/model as recorded above, not independently verified.
