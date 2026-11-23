# i18n audit a3 — ru / tunnel.status (PR #3945 round-1 remediation, branch fix/tunnel-remedies-auto-up)

Base audited: HEAD 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the UNCOMMITTED working-tree catalogue values (`git diff -- local_operator/i18n/catalogues/`), worktree .worktrees/tunnel-remedies-auto-up. Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent (audit-es-fr-ru) on host ip-172-31-18-0, session 117e988422ff; not the translator (c0c9e9803405) nor the English-drafting coder (889b9f67edf5); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 117e988422ff, `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (uncommitted working-tree values on 7c0fc7bf2; English source drafted by coder session 889b9f67edf5); model anthropic/claude-opus-5-5 (read from that session's transcript metadata, not otherwise verified).
Source: en sha256 e2bc8c63d3124c363252511306fd3951917ab9f93a2398c49cac8c79302aa8f9 (24 keys); context sha256 78580b806d2acc2c0c84bcf2655b62ea5ae847bf206c43134762e288bba8695d; target sha256 b6b7cdba17644b965655dbf19c52cf0c42fd3b806a6026f3fce9e12e7e26d6e4. Prior rounds on disk: a1, a2 (previous ledger row audited in commit f0300179e).

**Verdict: changes_requested** — 1 unresolved major; 2 finding(s): 1 major, 1 nit.

Scope: the 8 keys changed or new in the round-1 remediation, judged in full against the en source, its context sidecar, i18n/glossary.md (no row for phone access/credit/balance/console; device → устройство not applicable to 'computer') and i18n/style-guides/ru.md. Unchanged keys were not re-audited (already passed in earlier rounds); key parity and the catalogue checker cover the whole namespace (audit.log).

Design intent checked: "phone access" on every surface (no tunnel / remote-access wording), {url}, `/mobile enable`, Radient and Local Operator verbatim, no command named in the console-only remedies. Card lines measured with `rich.cells.cell_len` including the 2-cell "! " prefix (card_paused after "! /mobile enable — "), budget ≤ 50 cells (tests/unit/test_tunnel_remedies.py).

## Per-key judgement

| key | en | ru | judgement |
|---|---|---|---|
| `tunnel.status.tunnel_inactive.label` | paused by Radient — reconnecting | приостановлен Radient — переподключение | OK — «приостановлен Radient — переподключение»: agentive reading of undeclined «Radient» is the normal one in this pattern (cf. «заблокирован Google»). |
| `tunnel.status.tunnel_inactive.detail` | Radient has paused phone access for this computer, but billing is in order. Reconnect phone access in Local Operator on the computer running it; it also keeps retrying by itself. If it stays paused, review it in the Radient console at {url}. | Radient приостановил доступ с телефона для этого компьютера, хотя с оплатой всё в порядке. Переподключите доступ с телефона в Local Operator на этом компьютере; кроме того, доступ сам продолжает повторять попытки. Если он так и остаётся приостановленным, проверьте его в консоли Radient: {url}. | OK — translator's «на этом компьютере» accepted: the first clause already says «для этого компьютера» (mirrors en «for this computer»), so the sentence is internally consistent on every surface. |
| `tunnel.status.tunnel_disabled.detail` | Phone access for this computer is turned off in the Radient console. Turn it back on at {url}; it reconnects by itself. | Доступ с телефона для этого компьютера отключён в консоли Radient. Включите его снова на {url}; он переподключится сам. | OK — «Включите его снова на {url}; он переподключится сам.» |
| `tunnel.status.billing_lapsed.detail` | Your Radient balance has run out, so phone access is paused. Add credit at {url}; it reconnects by itself once the balance is restored. | На вашем балансе Radient закончились средства, поэтому доступ с телефона приостановлен. Пополните баланс на {url}; доступ переподключится сам, как только баланс будет восстановлен. | OK — «На вашем балансе Radient закончились средства» idiomatic; «Пополните баланс» matches corpus (wire.errors.media «Пополните его в консоли Radient»). |
| `tunnel.status.billing_inactive.detail` | Phone access isn't turned on for this Radient account yet. Open phone access setup in Local Operator on that computer and accept the monthly price. | Для этой учётной записи Radient доступ с телефона ещё не включён. Откройте настройку доступа с телефона в Local Operator на том компьютере и примите месячную цену. | OK on meaning. RU-02 nit: «на том компьютере» has no antecedent (inherited from en «on that computer»). |
| `tunnel.status.card_paused` | paused by Radient | приостановлен Radient | OK — 40/50 cells after «/mobile enable — ». |
| `tunnel.status.card_add_credit` | Add Radient credit to resume phone access | Пополните баланс Radient, чтобы вернуть доступ | RU-01 MAJOR — «вернуть доступ» drops «с телефона»; 48/50 cells. |
| `tunnel.status.card_open_console` | Turn phone access on in the Radient console | Включите доступ с телефона в консоли Radient | OK — keeps «доступ с телефона», 46/50 cells. |

## Mechanical

Placeholders identical (only `{url}`), no ICU in scope, Radient / Local Operator verbatim, key parity 24/24, valid JSON, check.py 0 problems; no tunnel/remote-access term in any in-scope value. The only "phone access" omissions the term check reports are the card lines in the finding below (audit.log).

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| RU-01 | major | `tunnel.status.card_add_credit` | «Пополните баланс Radient, чтобы вернуть доступ» drops «с телефона». On a standalone one-line TUI card bare «доступ» names no feature, and the Radient balance also funds model usage, so it reads as restoring Radient access. Design intent requires «доступ с телефона» (card_open_console keeps it); a compliant wording fits 50 cells. | «Пополните баланс Radient для доступа с телефона» (49 cells). |
| RU-02 | nit | `tunnel.status.billing_inactive.detail` | «на том компьютере» has no antecedent; inherited from the en source's dangling «on that computer». | «на нужном компьютере». |

Source note for the coordinator (not a translation finding): en `tunnel.status.billing_inactive.detail` says «on that computer» with no computer named earlier in the string.

## Not verified

No native-speaker review and no rendered terminal/desktop/phone frame; widths are cell_len measurements, not screenshots. Judgement by reasoning only; no MT service used. This is one independent pass; a second independent pass for this high-visibility (error/status) group is the manager's to schedule.
