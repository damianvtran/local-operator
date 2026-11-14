# i18n audit a1 — ru / wire.phone_access (branch fix/tunnel-remedies-auto-up @ 1fce7ecb1f5f)

Head audited: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (worktree .worktrees/tunnel-remedies-auto-up, branch fix/tunnel-remedies-auto-up). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent on node lop-remote-node-2, session 110287e9c4f3; not the translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 110287e9c4f3 (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (values in commit 1fce7ecb1f5f; English source drafted by coder session b0300e2c2c7f); model anthropic/claude-opus-5-5 (as stated by the coordinator brief; not independently verified); 2026-11-14.
Source: en sha256 2979b226db50db768fcad68f5f7376854f2185b08e6b0af08eb8e72bdb470f58 (4 keys); target sha256 3ec4a68b6c7e8368874df3aeff123af3995161f803f05ff45a842f061e67ecf4. Prior rounds on disk: none.

**Verdict: passed** — 0 unresolved blocker/major; 0 minor/nit finding(s).

Scope: the whole namespace at the new English source hash. The 4 keys new in 1fce7ecb1f5f were judged in full; the namespace is new; every key is new.

Read: docs/I18N.md 'Add or extend a locale', i18n/glossary.md (no row for phone access/tunnel/billing/credit), i18n/style-guides/ru.md, en + context sidecar, corpus precedents (mobile.setup.balance/consent/billing_activated, tunnel.status.server_unavailable.*).

## Per-key judgement (new keys)

| key | en | ru | judgement |
|---|---|---|---|
| `wire.phone_access.title` | Phone access | Доступ с телефона | OK — «Доступ с телефона» matches corpus. |
| `wire.phone_access.description` | How this computer keeps its Radient tunnel for the phone app working. Changes apply to the next sign-in. | Как этот компьютер поддерживает работу своего туннеля Radient для приложения на телефоне. Изменения применяются при следующем входе. | OK. |
| `wire.phone_access.auto_setup.label` | Reconnect phone access after signing in | Переподключать доступ с телефона после входа | OK — imperfective infinitive «Переподключать», idiomatic for a setting. |
| `wire.phone_access.auto_setup.help` | When you sign in to Radient on a computer that already has phone access, reconnect or repair its tunnel in the background. It never activates billing or spends credit; any price or missing credit is shown to you instead. | Когда вы входите в Radient на компьютере, где доступ с телефона уже настроен, его туннель переподключается или восстанавливается в фоне. Это никогда не активирует оплату и не тратит кредиты; вместо этого вам показывается цена или нехватка кредитов. | OK — «никогда не активирует оплату и не тратит кредиты» intact. |

## Safety meanings

No terminal command is named in any new key (only `/mobile enable`, verbatim, in mobile.setup.login_offer). Money clauses survive where the English has them (nothing charged; setup never purchases credit; never activates billing or spends credit; price shown before billing) — regex presence logged per key in audit.log, meaning judged above. {url} and the console URL are byte-identical to en.

## Findings

None.

Minor/nit findings are advisory and do not block; the coordinator may batch them into a translator revision.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used. Translator model identity is as stated by the coordinator, not independently verified.
