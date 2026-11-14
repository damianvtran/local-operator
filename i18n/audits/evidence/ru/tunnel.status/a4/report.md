# Independent translation re-audit — ru / tunnel.status / a4

Base: 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the uncommitted working-tree catalogue changes for local-operator PR #3945 (round-1 remediation, branch fix/tunnel-remedies-auto-up), after the translator's fixes to the a3 findings. Date: 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session 5ac9acc2ba69 (round-1 remediation re-audit); neither the i18n-translator (c0c9e9803405), the English-drafting coder (889b9f67edf5), nor the prior auditors (117e988422ff, dff33b90744d); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 5ac9acc2ba69 (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (#3945 round-1 remediation fixes, uncommitted on 7c0fc7bf2); English source drafted by coder session 889b9f67edf5; model anthropic/claude-opus-5-5 (as recorded in the prior bundle; not otherwise verified).
Prior audit: i18n/audits/evidence/ru/tunnel.status/a3/record.json (target sha256 b6b7cdba17644b965655dbf19c52cf0c42fd3b806a6026f3fce9e12e7e26d6e4).
Source: en sha256 e2bc8c63d3124c363252511306fd3951917ab9f93a2398c49cac8c79302aa8f9 (24 keys); context sha256 78580b806d2acc2c0c84bcf2655b62ea5ae847bf206c43134762e288bba8695d; target sha256 fb885e6ba5cf573e528a31972135af0b458c310905a24c0dc0b3bcd3b9895c4b (current bytes).

**Verdict: passed**: 0 unresolved blocker/major. Prior findings resolved: RU-01. Open carried nits: RU-02. New findings: none.

## Scope and method

Mechanical integrity was re-verified for the whole namespace (24 keys) at its current bytes. Exactly the values the translator reported changed since a3: `card_add_credit`. The other scope values match the a3 report verbatim, and the 16 non-scope keys are byte-identical to HEAD (audit.log: `delta.py` against the prior report table, since the prior target bytes are not retained, and `git diff HEAD -U0`). Because this group is high-visibility (status/error copy), I re-read all 8 remediation-scope keys independently instead of assuming the earlier pass was right. Judgement is by reasoning against the en source, its context sidecar, i18n/glossary.md (no row for phone access, credit, balance or console) and i18n/style-guides/ru.md. No MT service was used.

Card fit: each line is measured with `rich.cells.cell_len` including the 2-cell '! ' prefix (card_paused after '! /mobile enable — '); the budget is 50 cells (tests/unit/test_tunnel_remedies.py; tui/app.py `tunnel_park_card`). 'Phone access' must be explicit on every card line except card_paused, which follows the `/mobile enable` command. The real `tunnel_park_card` and `gateway.reason_label` render path was also run per locale (audit.log). Before the ledger row, es/fr/ru were unshipped (shipped.json had dropped them on the changes_requested rows), so the pre-ledger render fell back to English; the post-append `generate.py` run and re-render are recorded in audit.log under 'Post-append verification'.

## Changed keys (judged in full)

| key | en | current value | cells | judgement |
|---|---|---|---|---|
| `tunnel.status.card_add_credit` | Add Radient credit to resume phone access | Пополните баланс Radient для доступа с телефона | 49 | RESOLVED (RU-01). «Пополните баланс Radient для доступа с телефона» keeps «доступа с телефона» (correct genitive after «для»); «Пополните баланс» matches the corpus (billing_lapsed.detail, wire.errors.media); polite imperative; undeclined Radient per guide. 49/50 cells. |

## Remaining scope keys (re-read, unchanged since the prior round)

- `tunnel.status.tunnel_inactive.label`: «приостановлен Radient — переподключение» — OK: masculine agrees with the connector frame `Connector: retrying — {label}`.
- `tunnel.status.tunnel_inactive.detail`: «Radient приостановил доступ с телефона для этого компьютера, хотя с оплатой всё в порядке. Переподключите доступ с телефона в Local Operator на этом компьютере; кроме того, доступ сам продолжает повторять попытки. Если он так и остаётся приостановленным, проверьте его в консоли Radient: {url}.» — OK.
- `tunnel.status.tunnel_disabled.detail`: «Доступ с телефона для этого компьютера отключён в консоли Radient. Включите его снова на {url}; он переподключится сам.» — OK.
- `tunnel.status.billing_lapsed.detail`: «На вашем балансе Radient закончились средства, поэтому доступ с телефона приостановлен. Пополните баланс на {url}; доступ переподключится сам, как только баланс будет восстановлен.» — OK.
- `tunnel.status.billing_inactive.detail`: «Для этой учётной записи Radient доступ с телефона ещё не включён. Откройте настройку доступа с телефона в Local Operator на том компьютере и примите месячную цену.» — OK on meaning; RU-02 nit still open.
- `tunnel.status.card_paused`: «приостановлен Radient» — OK.
- `tunnel.status.card_open_console`: «Включите доступ с телефона в консоли Radient» — OK.

## Mechanical

Valid UTF-8 JSON; key parity 24/24 with en; placeholder sets identical on every key (only `{url}`, `{port}`, `{status}`, `{label}`, `{at}`, `{window}`, `{tally}` where the source has them); no ICU plural/select in this namespace; protected tokens (Radient, Local Operator, `lop …` commands, `/mobile`, HTTP, 127.0.0.1, `--tunnel-id`) unchanged; no tunnel/remote-access wording in the scope values; 'phone access' explicit on every scope value except the label and card_paused; no manual bidi controls; `check.py --catalogues-only`: 0 problems. A negative control (faults injected into a scratch copy) shows the checker reports placeholder, typography, bidi-control and missing-term faults (audit.log).

## Findings

| ID | severity | status | key | issue | suggested fix |
|---|---|---|---|---|---|
| RU-01 | major | resolved | card_add_credit | Raised in a3; fixed by the translator's new value (see the Changed keys table). | — |
| RU-02 | nit | open (carried, not re-counted as new) | `tunnel.status.billing_inactive.detail` | «на том компьютере» still has no antecedent (inherited from the en source); unchanged since a3. | «на нужном компьютере». |

No new findings this round.

## Considered, not raised

- Dropping 'to resume' (the ur pass raised the same omission as UR-07 minor). «…, чтобы вернуть доступ с телефона» is 59 cells; «Пополните Radient, чтобы вернуть доступ с телефона» is 52 and drops «баланс». No compliant wording keeps it. Accepted, not raised.
- card_open_console «Включите доступ с телефона в консоли Radient» (46 cells) was unchanged and stays OK.

## Not verified

No native-speaker review and no rendered screenshot of the terminal, desktop or phone frame; widths are `cell_len` measurements plus the real `tunnel_park_card` string. Judgement is by reasoning only; no MT service was used. This is the requested second independent pass over the remediation-scope keys; I did not re-litigate the 16 unchanged keys that earlier committed rounds passed.
