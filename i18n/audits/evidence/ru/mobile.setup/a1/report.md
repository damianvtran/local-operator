# i18n audit a1 — ru / mobile.setup (branch fix/tunnel-remedies-auto-up @ 1fce7ecb1f5f)

Head audited: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (worktree .worktrees/tunnel-remedies-auto-up, branch fix/tunnel-remedies-auto-up). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent on node lop-remote-node-2, session 110287e9c4f3; not the translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 110287e9c4f3 (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (values in commit 1fce7ecb1f5f; English source drafted by coder session b0300e2c2c7f); model anthropic/claude-opus-5-5 (as stated by the coordinator brief; not independently verified); 2026-11-14.
Source: en sha256 6883f92e94dfcec4be0af6041a0531c974fbf18294f7094a132cc555a2ad2065 (86 keys); target sha256 d1d26ba807b11ebd26bd120c3cfc72d42789eae2ca10ba3e18d4f30d4d931da1. Prior rounds on disk: cg1, r1.

**Verdict: passed** — 0 unresolved blocker/major; 3 minor/nit finding(s).

Scope: the whole namespace at the new English source hash. The 7 keys new in 1fce7ecb1f5f were judged in full; the other keys were re-read for consistency only (already passed in earlier rounds; unchanged in this commit).

Read: docs/I18N.md 'Add or extend a locale', i18n/glossary.md (no row for phone access/tunnel/billing/credit), i18n/style-guides/ru.md, en + context sidecar, corpus precedents (mobile.setup.balance/consent/billing_activated, tunnel.status.server_unavailable.*).

## Per-key judgement (new keys)

| key | en | ru | judgement |
|---|---|---|---|
| `mobile.setup.disabled` | This tunnel is turned off in the Radient console. Turn it back on at https://console.radienthq.com/dashboard/tunnels; it reconnects by itself, and setup will not override that choice. | Этот туннель отключён в консоли Radient. Включите его снова на https://console.radienthq.com/dashboard/tunnels; он переподключится сам, а настройка не будет отменять это решение. | OK — URL verbatim; non-override kept. |
| `mobile.setup.billing_lapsed` | This tunnel's billing lapsed because the Radient balance ran out. Add credit at https://console.radienthq.com/dashboard/tunnels; setup never purchases credit, and the tunnel reconnects by itself once the balance is restored. | Оплата этого туннеля истекла, потому что закончился баланс Radient. Пополните баланс на https://console.radienthq.com/dashboard/tunnels; настройка никогда не покупает кредиты, а туннель переподключится сам, как только баланс будет восстановлен. | OK on safety («настройка никогда не покупает кредиты» = mobile.setup.balance); RU-01 applies to «истекла». |
| `mobile.setup.reconnect` | Radient has paused this tunnel, but its billing is in order. Reconnect to bring phone access back; nothing is charged. | Radient приостановил этот туннель, хотя с оплатой всё в порядке. Переподключите его, чтобы вернуть доступ с телефона; ничего не списывается. | OK — «ничего не списывается». |
| `mobile.setup.network` | This computer could not reach Radient (its network may have changed or dropped). Phone access reconnects by itself once the connection is back; nothing needs doing. | Этому компьютеру не удалось связаться с Radient (возможно, его сеть изменилась или пропала). Доступ с телефона переподключится сам, когда соединение восстановится; ничего делать не нужно. | OK — «ничего делать не нужно». |
| `mobile.setup.radient_unavailable` | Radient answered with a temporary server error. Nothing is wrong with this computer or its sign-in; phone access reconnects by itself once Radient recovers. | Radient ответил временной ошибкой сервера. С этим компьютером и его входом всё в порядке; доступ с телефона переподключится сам, когда Radient заработает. | OK; RU-02 nit. |
| `mobile.setup.reconnecting` | Phone access is reconnecting in the background after your sign-in. Nothing is charged; this clears by itself in a minute or two. | Доступ с телефона переподключается в фоне после вашего входа. Ничего не списывается; через минуту-другую это сообщение исчезнет само. | OK — «Ничего не списывается»; «через минуту-другую» idiomatic. |
| `mobile.setup.login_offer` | Reach this computer from the phone app: run /mobile enable. Any monthly price is shown to you before billing starts. | Подключайтесь к этому компьютеру из приложения на телефоне: выполните /mobile enable. Любая месячная цена будет показана вам до начала оплаты. | OK on safety (`/mobile enable` verbatim; price before billing); RU-03 nit (calque «Любая месячная цена»). |

## Safety meanings

No terminal command is named in any new key (only `/mobile enable`, verbatim, in mobile.setup.login_offer). Money clauses survive where the English has them (nothing charged; setup never purchases credit; never activates billing or spends credit; price shown before billing) — regex presence logged per key in audit.log, meaning judged above. {url} and the console URL are byte-identical to en.

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| RU-01 | minor | `mobile.setup.billing_lapsed` | «оплата истекла» is unidiomatic (a payment does not 'expire'); the detail also stacks «потому что … поэтому». Meaning (balance ran out → add credit → auto-reconnect) survives (answers translator open question). | label «оплата приостановлена» (or «баланс исчерпан»); detail «Оплата этого туннеля прекратилась: на балансе Radient закончились средства, поэтому Radient не будет его подключать. …»; mobile.setup «Оплата этого туннеля прекратилась, потому что на балансе Radient закончились средства. …». |
| RU-02 | nit | `mobile.setup.radient_unavailable` | «его входом» and «когда Radient заработает» read slightly clipped. | «С этим компьютером и входом в аккаунт всё в порядке; … когда Radient снова заработает». |
| RU-03 | nit | `mobile.setup.login_offer` | «Любая месячная цена будет показана вам» is a calque of 'Any monthly price is shown to you'. | «Месячную цену вам покажут до начала оплаты.» (keeps corpus term «месячная цена»). |

Minor/nit findings are advisory and do not block; the coordinator may batch them into a translator revision.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used. Translator model identity is as stated by the coordinator, not independently verified.
