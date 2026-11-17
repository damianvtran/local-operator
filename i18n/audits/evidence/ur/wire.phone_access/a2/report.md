# Independent translation audit — ur / wire.phone_access / a2

Commit: 0e4ad0a9171c59d7833d222761cbaa1f6d09bf51 (PR #4145, branch of worktree .worktrees/desk-presence-core). Date: 2026-11-17.
Auditor: i18n-auditor role, independent `task` subagent of team lopdev, Local Operator session 43bcb19f888e; not the translator (the coder that authored 9134015c with the draft values) nor the English-drafting coder; did not edit any catalogue value; model anthropic/claude-opus-5-5 (harness system prompt of session 43bcb19f888e (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: coder subagent of team lopdev that authored commit 9134015c (draft values and English source in the same commit); session id not disclosed in the brief or PR #4145; model not stated in the brief — unknown, not independently verified.
Source: en sha256 1d6935b1259cab3b9d4975c8b8149de76ceb61be177f22c6b4e01a289b6e762a (10 keys); context sidecar sha256 95c0c4b8d5f82cef978c751326ba6c214c1c8bf8152717f608eaeee943071c07; target sha256 ed03ca00cc52c95ce86e47440a4a206aab6e192a1252691dc0c2ad439c3ecd2a.
Prior audit on disk: a1 (passed at en sha256 2979b226…, 4 keys); stale because the English source gained 6 keys.

**Verdict: passed** — 0 unresolved blocker/major; 1 minor/nit finding(s).

Scope: the 6 keys new in this PR (ADR 0008 `notifications.phone_away_after_s` registry row: label, help, 1/2/5/10-minute choices) judged in full — accuracy, register, placeholders, length/fit, glossary/corpus consistency, RTL. The 4 keys passed in a1 are byte-unchanged in both en and ur (diff origin/main...HEAD touches only the appended lines), so their a1 judgement is retained; the receipt covers the whole namespace (10 keys) at the new English source hash. No other i18n keys are added or changed by the PR (checked: `git diff origin/main...HEAD -- local_operator/i18n` touches only wire.phone_access and its generated key modules).

Read: docs/I18N.md 'Add or extend a locale', i18n/glossary.md (no row for away/idle/phone; corpus usage governs), i18n/style-guides/ur.md, en + context sidecar, settings_io.py registry row (ENUM: label rendered with the chosen value after it; choices also listed in the picker), corpus precedent wire.settings.session_cleanup_temporary_idle_minutes.* and existing 'desktop app' renderings.

## Per-key judgement (new keys)

| key | en | ur | judgement |
|---|---|---|---|
| `wire.phone_access.away_after.label` | Send to phone when I'm away from my computer for | جب میں اتنی دیر کمپیوٹر سے دور رہوں تو فون پر بھیجیں | OK — «جب میں اتنی دیر کمپیوٹر سے دور رہوں تو فون پر بھیجیں»: «اتنی دیر» points at the value; person mix mirrors English ('Send… when I'm'). |
| `wire.phone_access.away_after.help` | How long this computer can go without keyboard or mouse input before a finished task it would show you here goes to your phone instead. Locking the screen counts as away right away. Needs the desktop app. | یہ کمپیوٹر کتنی دیر کی‌بورڈ یا ماؤس کے استعمال کے بغیر رہ سکتا ہے، اس کے بعد مکمل ہونے والا کام جو آپ کو یہاں دکھایا جاتا، آپ کے فون پر بھیج دیا جائے گا۔ اسکرین لاک کرنا فوراً غیر موجودگی شمار ہوتا ہے۔ ڈیسک ٹاپ ایپ درکار ہے۔ | OK — آپ register; lock = «فوراً غیر موجودگی»; «ڈیسک ٹاپ ایپ» matches corpus; UR-N1 nit. |
| `wire.phone_access.away_after.choice.1m.label` | 1 minute | 1 منٹ | OK — «N منٹ»; Latin digits per style guide §Numbers (Urdu digits deferred in v1). |
| `wire.phone_access.away_after.choice.2m.label` | 2 minutes | 2 منٹ | OK — «N منٹ»; Latin digits per style guide §Numbers (Urdu digits deferred in v1). |
| `wire.phone_access.away_after.choice.5m.label` | 5 minutes | 5 منٹ | OK — «N منٹ»; Latin digits per style guide §Numbers (Urdu digits deferred in v1). |
| `wire.phone_access.away_after.choice.10m.label` | 10 minutes | 10 منٹ | OK — «N منٹ»; Latin digits per style guide §Numbers (Urdu digits deferred in v1). |

## Safety / meaning

No placeholders, URLs or commands in any new key (mechanical check in audit.log). The three load-bearing facts of the help text — the threshold is keyboard/mouse idle time, locking the screen counts as away immediately, and it needs the desktop app — are present in the ur value.

## RTL

No Latin runs or placeholders in the six new values, so no bidi isolation is needed; Latin digits (style guide §Numbers, v1) are weak-LTR and resolve with the number on the right of «منٹ» in an RTL paragraph, the correct Urdu reading order. The label is a complete RTL clause followed by the value; nothing depends on string-internal direction marks. Not verified in a rendered frame.

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| UR-N1 | nit | `wire.phone_access.away_after.help` | «کی‌بورڈ» joins with ZWNJ (U+200C); the one corpus occurrence spells «کی بورڈ» with a space. Both render identically; ZWNJ is arguably better (keeps one word for line breaking). Consistency only. | none required |

Minor/nit findings are advisory and do not block; the coordinator may batch them into a translator revision (a value change would need a fresh receipt at the new target bytes).

## Not verified

No native-speaker review and no rendered desktop/TUI/phone frame of this settings row. Judgement by reasoning only; no MT service used. Translator identity/model as recorded above, not independently verified.
