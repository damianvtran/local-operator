# i18n audit a2 — ru / tunnel.status (branch fix/tunnel-remedies-auto-up @ 1fce7ecb1f5f)

Head audited: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (worktree .worktrees/tunnel-remedies-auto-up, branch fix/tunnel-remedies-auto-up). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent on node lop-remote-node-2, session 110287e9c4f3; not the translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 110287e9c4f3 (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (values in commit 1fce7ecb1f5f; English source drafted by coder session b0300e2c2c7f); model anthropic/claude-opus-5-5 (as stated by the coordinator brief; not independently verified); 2026-11-14.
Source: en sha256 b368acfb0cdee89c8de5a210fc9956251701a9f3f73c88b1096611d0c1f9063f (21 keys); target sha256 f302c30cebd43ee0e43263b7ec4f22f28d3dd55de0275790cc282be3cf48c9c0. Prior rounds on disk: a1.

**Verdict: passed** — 0 unresolved blocker/major; 1 minor/nit finding(s).

Scope: the whole namespace at the new English source hash. The 8 keys new in 1fce7ecb1f5f were judged in full; the other keys were re-read for consistency only (already passed in earlier rounds; unchanged in this commit).

Read: docs/I18N.md 'Add or extend a locale', i18n/glossary.md (no row for phone access/tunnel/billing/credit), i18n/style-guides/ru.md, en + context sidecar, corpus precedents (mobile.setup.balance/consent/billing_activated, tunnel.status.server_unavailable.*).

## Per-key judgement (new keys)

| key | en | ru | judgement |
|---|---|---|---|
| `tunnel.status.tunnel_inactive.label` | tunnel not active | туннель неактивен | OK — «туннель неактивен». |
| `tunnel.status.tunnel_inactive.detail` | Radient has paused this tunnel, but its billing is in order. Reconnect phone access in Local Operator on the computer running it; it also keeps retrying by itself. If it stays paused, review the tunnel at {url}. | Radient приостановил этот туннель, хотя с оплатой всё в порядке. Переподключите доступ с телефона в Local Operator на компьютере, где он работает; кроме того, он сам продолжает повторять попытки. Если туннель так и остаётся приостановленным, проверьте его на {url}. | OK — «хотя с оплатой всё в порядке»; reconnect in Local Operator; no command; {url} verbatim; вы. |
| `tunnel.status.tunnel_disabled.label` | tunnel turned off in the console | туннель отключён в консоли | OK — ё kept. |
| `tunnel.status.tunnel_disabled.detail` | This tunnel is turned off in the Radient console, so remote access stays off until it is turned back on there: {url}. It reconnects by itself once it is on. | Этот туннель отключён в консоли Radient, поэтому удалённый доступ не заработает, пока туннель там снова не включат: {url}. Как только его включат, он переподключится сам. | OK — impersonal «пока туннель там снова не включат» mirrors the English passive. |
| `tunnel.status.billing_inactive.label` | billing not activated | оплата не активирована | OK. |
| `tunnel.status.billing_inactive.detail` | Billing for this tunnel has not been activated, so Radient will not connect it. Open phone access setup in Local Operator on the computer running it and accept the monthly price. | Оплата этого туннеля не активирована, поэтому Radient не будет его подключать. Откройте настройку доступа с телефона в Local Operator на компьютере, где он работает, и примите месячную цену. | OK — «примите месячную цену» matches corpus «по принятой месячной цене». |
| `tunnel.status.billing_lapsed.label` | billing lapsed | оплата истекла | RU-01 minor — «оплата истекла» unidiomatic (payment does not 'expire'). |
| `tunnel.status.billing_lapsed.detail` | This tunnel's billing lapsed because the Radient balance ran out, so Radient will not connect it. Add credit at {url}; it reconnects by itself once the balance is restored. | Оплата этого туннеля истекла, потому что закончился баланс Radient, поэтому Radient не будет его подключать. Пополните баланс на {url}; туннель переподключится сам, как только баланс будет восстановлен. | OK on meaning (cause, fix «Пополните баланс на {url}», auto-reconnect); RU-01 also covers «истекла … потому что … поэтому». |

## Safety meanings

No terminal command is named in any new key (only `/mobile enable`, verbatim, in mobile.setup.login_offer). Money clauses survive where the English has them (nothing charged; setup never purchases credit; never activates billing or spends credit; price shown before billing) — regex presence logged per key in audit.log, meaning judged above. {url} and the console URL are byte-identical to en.

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| RU-01 | minor | `tunnel.status.billing_lapsed.label`, `tunnel.status.billing_lapsed.detail` | «оплата истекла» is unidiomatic (a payment does not 'expire'); the detail also stacks «потому что … поэтому». Meaning (balance ran out → add credit → auto-reconnect) survives (answers translator open question). | label «оплата приостановлена» (or «баланс исчерпан»); detail «Оплата этого туннеля прекратилась: на балансе Radient закончились средства, поэтому Radient не будет его подключать. …»; mobile.setup «Оплата этого туннеля прекратилась, потому что на балансе Radient закончились средства. …». |

Minor/nit findings are advisory and do not block; the coordinator may batch them into a translator revision.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used. Translator model identity is as stated by the coordinator, not independently verified.
