# Independent translation audit — hi / wire.phone_access / a2

Commit: 0e4ad0a9171c59d7833d222761cbaa1f6d09bf51 (PR #4145, branch of worktree .worktrees/desk-presence-core). Date: 2026-11-17.
Auditor: i18n-auditor role, independent `task` subagent of team lopdev, Local Operator session 43bcb19f888e; not the translator (the coder that authored 9134015c with the draft values) nor the English-drafting coder; did not edit any catalogue value; model anthropic/claude-opus-5-5 (harness system prompt of session 43bcb19f888e (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: coder subagent of team lopdev that authored commit 9134015c (draft values and English source in the same commit); session id not disclosed in the brief or PR #4145; model not stated in the brief — unknown, not independently verified.
Source: en sha256 1d6935b1259cab3b9d4975c8b8149de76ceb61be177f22c6b4e01a289b6e762a (10 keys); context sidecar sha256 95c0c4b8d5f82cef978c751326ba6c214c1c8bf8152717f608eaeee943071c07; target sha256 c432f92a645d7bd87dab0ee66081272f2ede412a098fbcc8798eba4e2defe6db.
Prior audit on disk: a1 (passed at en sha256 2979b226…, 4 keys); stale because the English source gained 6 keys.

**Verdict: passed** — 0 unresolved blocker/major; 0 minor/nit finding(s).

Scope: the 6 keys new in this PR (ADR 0008 `notifications.phone_away_after_s` registry row: label, help, 1/2/5/10-minute choices) judged in full — accuracy, register, placeholders, length/fit, glossary/corpus consistency. The 4 keys passed in a1 are byte-unchanged in both en and hi (diff origin/main...HEAD touches only the appended lines), so their a1 judgement is retained; the receipt covers the whole namespace (10 keys) at the new English source hash. No other i18n keys are added or changed by the PR (checked: `git diff origin/main...HEAD -- local_operator/i18n` touches only wire.phone_access and its generated key modules).

Read: docs/I18N.md 'Add or extend a locale', i18n/glossary.md (no row for away/idle/phone; corpus usage governs), i18n/style-guides/hi.md, en + context sidecar, settings_io.py registry row (ENUM: label rendered with the chosen value after it; choices also listed in the picker), corpus precedent wire.settings.session_cleanup_temporary_idle_minutes.* and existing 'desktop app' renderings.

## Per-key judgement (new keys)

| key | en | hi | judgement |
|---|---|---|---|
| `wire.phone_access.away_after.label` | Send to phone when I'm away from my computer for | कंप्यूटर से इतनी देर दूर रहने पर फ़ोन पर भेजें | OK — «कंप्यूटर से इतनी देर दूर रहने पर फ़ोन पर भेजें» natural; value supplies the duration. |
| `wire.phone_access.away_after.help` | How long this computer can go without keyboard or mouse input before a finished task it would show you here goes to your phone instead. Locking the screen counts as away right away. Needs the desktop app. | यह कंप्यूटर कीबोर्ड या माउस इनपुट के बिना कितनी देर रह सकता है, उसके बाद पूरा हुआ कार्य, जो आपको यहाँ दिखाया जाता, आपके फ़ोन पर भेजा जाएगा। स्क्रीन लॉक करना तुरंत दूर होना माना जाता है। डेस्कटॉप ऐप आवश्यक है। | OK — आप register; «डेस्कटॉप ऐप» and nukta «फ़ोन» match corpus; meaning complete. |
| `wire.phone_access.away_after.choice.1m.label` | 1 minute | 1 मिनट | OK — «N मिनट»; Latin digits per style guide. |
| `wire.phone_access.away_after.choice.2m.label` | 2 minutes | 2 मिनट | OK — «N मिनट»; Latin digits per style guide. |
| `wire.phone_access.away_after.choice.5m.label` | 5 minutes | 5 मिनट | OK — «N मिनट»; Latin digits per style guide. |
| `wire.phone_access.away_after.choice.10m.label` | 10 minutes | 10 मिनट | OK — «N मिनट»; Latin digits per style guide. |

## Safety / meaning

No placeholders, URLs or commands in any new key (mechanical check in audit.log). The three load-bearing facts of the help text — the threshold is keyboard/mouse idle time, locking the screen counts as away immediately, and it needs the desktop app — are present in the hi value.

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| — | — | — | none | — |

Minor/nit findings are advisory and do not block; the coordinator may batch them into a translator revision (a value change would need a fresh receipt at the new target bytes).

## Not verified

No native-speaker review and no rendered desktop/TUI/phone frame of this settings row. Judgement by reasoning only; no MT service used. Translator identity/model as recorded above, not independently verified.
