# Independent translation audit — zh-CN / wire.phone_access / a3

Commit: d30d1fb1ee99567900586e510c99162b13c9856d (PR #4145, branch of worktree .worktrees/desk-presence-core). Date: 2026-11-17.
Auditor: i18n-auditor, independent `task` subagent of team lopdev, Local Operator session cee2e3bacd71; not the translator (the remediation coder that authored d30d1fb1 with the ZH-M1 value change, nor the coder that authored 9134015c) nor the English-drafting coder; did not edit any catalogue value; model anthropic/claude-opus-5-5 (harness system prompt of session cee2e3bacd71 (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: remediation coder subagent of team lopdev that authored commit d30d1fb1 (ZH-M1 fix to the four minute labels, applying the a2 auditor's suggested values verbatim); other six values by the coder of 9134015c; session ids not disclosed in the brief or PR #4145; model not stated in the brief — unknown, not independently verified.
Source: en sha256 1d6935b1259cab3b9d4975c8b8149de76ceb61be177f22c6b4e01a289b6e762a (10 keys, unchanged since a2); context sidecar sha256 95c0c4b8d5f82cef978c751326ba6c214c1c8bf8152717f608eaeee943071c07; target sha256 91e8bf24126ab3f36601f25cbfa526d7626f810a27aa17bc955511fdcfd45b9d.
Prior audits on disk: a1 (4 keys), a2 (passed at the same en sha256 1d6935b1…, target sha256 f068d44a…); a2's target record no longer matches because d30d1fb1 changed four values (ZH-M1 remediation).

**Verdict: passed** — 0 unresolved blocker/major; 1 nit finding carried (ZH-N2).

Scope: the four values changed since a2 (`away_after.choice.{1m,2m,5m,10m}.label`) judged in full; the other six zh-CN values are byte-unchanged since a2 (audit.log: `git diff 294e5923 HEAD` on the zh-CN catalogue touches only those four lines) and the en source, context sidecar, glossary and zh-CN style guide are byte-unchanged since a2 (empty `git diff --stat`), so their a1/a2 judgements are retained; the receipt covers the whole namespace (10 keys) at the current en source hash and the new target hash.

Read: i18n/style-guides/zh-CN.md §Punctuation & typography, the a2 report and record, corpus precedent wire.settings.session_cleanup_temporary_idle_minutes.choice.* («15分钟», «30分钟», «2小时», «24小时»).

## Per-key judgement (changed keys)

| key | en | zh-CN | judgement |
|---|---|---|---|
| `wire.phone_access.away_after.choice.1m.label` | 1 minute | 1分钟 | OK — ZH-M1 resolved: no digit–CJK space, per style guide §Punctuation & typography and corpus precedent «15分钟»/«30分钟». |
| `wire.phone_access.away_after.choice.2m.label` | 2 minutes | 2分钟 | OK — ZH-M1 resolved: no digit–CJK space, per style guide §Punctuation & typography and corpus precedent «15分钟»/«30分钟». |
| `wire.phone_access.away_after.choice.5m.label` | 5 minutes | 5分钟 | OK — ZH-M1 resolved: no digit–CJK space, per style guide §Punctuation & typography and corpus precedent «15分钟»/«30分钟». |
| `wire.phone_access.away_after.choice.10m.label` | 10 minutes | 10分钟 | OK — ZH-M1 resolved: no digit–CJK space, per style guide §Punctuation & typography and corpus precedent «15分钟»/«30分钟». |

Chinese has no plural inflection, so «2分钟»/«5分钟»/«10分钟» correctly render the English plurals; the values are the a2 auditor's suggested fix verbatim. Mechanical check (audit.log): key parity OK, no placeholders, no format characters, digit_space_cjk=False for all 10 keys.

## Findings

| ID | severity | keys | status |
|---|---|---|---|
| ZH-M1 | minor | `wire.phone_access.away_after.choice.{1m,2m,5m,10m}.label` | resolved in d30d1fb1 (verified above) |
| ZH-N2 | nit | `wire.phone_access.away_after.help` | carried from a2, unchanged value; advisory, does not block |

## Not verified

No native-speaker review and no rendered desktop/TUI/phone frame of this settings row. Judgement by reasoning only; no MT service used. Translator identity/model as recorded above, not independently verified.
