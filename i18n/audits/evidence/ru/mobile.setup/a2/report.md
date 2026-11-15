# i18n audit a2 — ru / mobile.setup (PR #3945 round-1 remediation, branch fix/tunnel-remedies-auto-up)

Base audited: HEAD 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the UNCOMMITTED working-tree catalogue values (`git diff -- local_operator/i18n/catalogues/`), worktree .worktrees/tunnel-remedies-auto-up. Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent (audit-es-fr-ru) on host ip-172-31-18-0, session 117e988422ff; not the translator (c0c9e9803405) nor the English-drafting coder (889b9f67edf5); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 117e988422ff, `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (uncommitted working-tree values on 7c0fc7bf2; English source drafted by coder session 889b9f67edf5); model anthropic/claude-opus-5-5 (read from that session's transcript metadata, not otherwise verified).
Source: en sha256 0d245737d539f4d2fe54cf3a2c2174b0aa64820d2eb387e94ea16f65d1b487d7 (94 keys); context sha256 b1c0c11a269077507f7d22256f3874d7f5fd1a363f5c13f412014de70c7f038f; target sha256 b87bb035e4fc06d66c37e4396968b906cc0f1905a49ba599d5fa5f672ba0fc29. Prior rounds on disk: a1, cg1, r1 (previous ledger row audited in commit f0300179e).

**Verdict: passed** — 0 unresolved blocker/major; 0 finding(s): none.

Scope: the 12 keys changed or new in the round-1 remediation, judged in full against the en source, its context sidecar, i18n/glossary.md (no row for phone access/credit/balance/console; device → устройство not applicable to 'computer') and i18n/style-guides/ru.md. Unchanged keys were not re-audited (already passed in earlier rounds); key parity and the catalogue checker cover the whole namespace (audit.log).

Design intent checked: "phone access" on every surface (no tunnel / remote-access wording), {url}, `/mobile enable`, Radient and Local Operator verbatim, no command named in the console-only remedies. Card lines measured with `rich.cells.cell_len` including the 2-cell "! " prefix (card_paused after "! /mobile enable — "), budget ≤ 50 cells (tests/unit/test_tunnel_remedies.py).

## Per-key judgement

| key | en | ru | judgement |
|---|---|---|---|
| `mobile.setup.disabled` | Phone access for this computer is turned off in the Radient console. It reconnects by itself once it is back on; setup will not override that choice. | Доступ с телефона для этого компьютера отключён в консоли Radient. Он переподключится сам, как только его снова включат; настройка не будет отменять это решение. | OK — «настройка не будет отменять это решение». |
| `mobile.setup.billing_lapsed` | Your Radient balance has run out, so phone access is paused. Setup never purchases credit; phone access reconnects by itself once the balance is restored. | На вашем балансе Radient закончились средства, поэтому доступ с телефона приостановлен. Настройка никогда не покупает кредиты; доступ с телефона переподключится сам, как только баланс будет восстановлен. | OK — «Настройка никогда не покупает кредиты» matches corpus «кредиты» for Radient credit (wire.phone_access, wire.errors.media). |
| `mobile.setup.reconnect` | Radient has paused phone access for this computer, but billing is in order. Reconnecting brings it back; nothing is charged. | Radient приостановил доступ с телефона для этого компьютера, хотя с оплатой всё в порядке. Переподключение вернёт его; ничего не списывается. | OK — «Переподключение вернёт его; ничего не списывается.» |
| `mobile.setup.network` | This computer could not reach Radient (its network may have changed or dropped). Phone access reconnects by itself once the connection is back. You don't need to do anything. | Этому компьютеру не удалось связаться с Radient (возможно, его сеть изменилась или пропала). Доступ с телефона переподключится сам, когда соединение восстановится. Ничего делать не нужно. | OK — «Ничего делать не нужно.» asks nothing. |
| `mobile.setup.title_reconnect` | Phone access is paused by Radient | Radient приостановил доступ с телефона | OK — active voice, natural. |
| `mobile.setup.title_billing_lapsed` | Phone access needs more Radient credit | Для доступа с телефона нужно пополнить баланс Radient | OK — «Для доступа с телефона нужно пополнить баланс Radient». |
| `mobile.setup.title_disabled` | Phone access is turned off in the Radient console | Доступ с телефона отключён в консоли Radient | OK. |
| `mobile.setup.title_network` | Phone access is waiting for the network | Доступ с телефона ожидает сети | OK. |
| `mobile.setup.title_radient_unavailable` | Phone access is waiting for Radient | Доступ с телефона ожидает Radient | OK. |
| `mobile.setup.title_reconnecting` | Phone access is reconnecting | Доступ с телефона переподключается | OK. |
| `mobile.setup.action_add_credit` | Add credit in the Radient console: {url} | Пополните баланс в консоли Radient: {url} | OK. |
| `mobile.setup.action_open_console` | Turn phone access back on in the Radient console: {url} | Снова включите доступ с телефона в консоли Radient: {url} | OK. |

## Mechanical

Placeholders identical (only `{url}`), no ICU in scope, Radient / Local Operator verbatim, key parity 94/94, valid JSON, check.py 0 problems; no tunnel/remote-access term in any in-scope value. The only "phone access" omissions the term check reports are the card lines in the finding below (audit.log).

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| — | — | — | No findings. | — |

## Not verified

No native-speaker review and no rendered terminal/desktop/phone frame; widths are cell_len measurements, not screenshots. Judgement by reasoning only; no MT service used. This is one independent pass; a second independent pass for this high-visibility (error/status) group is the manager's to schedule.
