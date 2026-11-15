# Independent translation re-audit — fr / tunnel.status / a5

Base: 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the uncommitted working-tree catalogue changes for local-operator PR #3945 (round-1 remediation, branch fix/tunnel-remedies-auto-up), after the translator's fixes to the a4 findings. Date: 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session 5ac9acc2ba69 (round-1 remediation re-audit); neither the i18n-translator (c0c9e9803405), the English-drafting coder (889b9f67edf5), nor the prior auditors (117e988422ff, dff33b90744d); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 5ac9acc2ba69 (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (#3945 round-1 remediation fixes, uncommitted on 7c0fc7bf2); English source drafted by coder session 889b9f67edf5; model anthropic/claude-opus-5-5 (as recorded in the prior bundle; not otherwise verified).
Prior audit: i18n/audits/evidence/fr/tunnel.status/a4/record.json (target sha256 7d1f849749155980cccd2d9c219369cd18c280608b42e430cfeccae8b794a87c).
Source: en sha256 e2bc8c63d3124c363252511306fd3951917ab9f93a2398c49cac8c79302aa8f9 (24 keys); context sha256 78580b806d2acc2c0c84bcf2655b62ea5ae847bf206c43134762e288bba8695d; target sha256 573d83c55aa20ae7c52235599b4ccb4220ff66013e6aa4b9e7863447788dff36 (current bytes).

**Verdict: passed**: 0 unresolved blocker/major. Prior findings resolved: FR-01. Open carried nits: none. New findings: none.

## Scope and method

Mechanical integrity was re-verified for the whole namespace (24 keys) at its current bytes. Exactly the values the translator reported changed since a4: `card_add_credit`, `card_open_console`. The other scope values match the a4 report verbatim, and the 16 non-scope keys are byte-identical to HEAD (audit.log: `delta.py` against the prior report table, since the prior target bytes are not retained, and `git diff HEAD -U0`). Because this group is high-visibility (status/error copy), I re-read all 8 remediation-scope keys independently instead of assuming the earlier pass was right. Judgement is by reasoning against the en source, its context sidecar, i18n/glossary.md (no row for phone access, credit, balance or console) and i18n/style-guides/fr.md. No MT service was used.

Card fit: each line is measured with `rich.cells.cell_len` including the 2-cell '! ' prefix (card_paused after '! /mobile enable — '); the budget is 50 cells (tests/unit/test_tunnel_remedies.py; tui/app.py `tunnel_park_card`). 'Phone access' must be explicit on every card line except card_paused, which follows the `/mobile enable` command. The real `tunnel_park_card` and `gateway.reason_label` render path was also run per locale (audit.log). Before the ledger row, es/fr/ru were unshipped (shipped.json had dropped them on the changes_requested rows), so the pre-ledger render fell back to English; the post-append `generate.py` run and re-render are recorded in audit.log under 'Post-append verification'.

## Changed keys (judged in full)

| key | en | current value | cells | judgement |
|---|---|---|---|---|
| `tunnel.status.card_add_credit` | Add Radient credit to resume phone access | Ajoutez du crédit Radient pour l’accès téléphone | 50 | RESOLVED (FR-01). «Ajoutez du crédit Radient pour l’accès téléphone» names the feature; vous imperative per guide; typographic apostrophe U+2019; corpus term «accès téléphone». 50/50 cells. |
| `tunnel.status.card_open_console` | Turn phone access on in the Radient console | Activez l’accès téléphone sur la console Radient | 50 | RESOLVED (FR-01). «Activez l’accès téléphone sur la console Radient». «sur la console» (vs corpus «dans la console», 7 uses) is idiomatic for a web console and is what fits: «…dans la console Radient» is 51 cells. 50/50 cells. |

## Remaining scope keys (re-read, unchanged since the prior round)

- `tunnel.status.tunnel_inactive.label`: «suspendu par Radient — reconnexion en cours» — OK: masculine agrees with the connector frame `Connector: retrying — {label}`.
- `tunnel.status.tunnel_inactive.detail`: «Radient a suspendu l’accès téléphone de cet ordinateur, mais la facturation est en règle. Reconnectez l’accès téléphone dans Local Operator sur l’ordinateur qui l’héberge ; il continue aussi de réessayer de lui-même. S’il reste suspendu, vérifiez-le dans la console Radient : {url}.» — OK.
- `tunnel.status.tunnel_disabled.detail`: «L’accès téléphone de cet ordinateur est désactivé dans la console Radient. Réactivez-le sur {url} ; il se reconnecte de lui-même.» — OK.
- `tunnel.status.billing_lapsed.detail`: «Votre solde Radient est épuisé, donc l’accès téléphone est suspendu. Ajoutez du crédit sur {url} ; il se reconnecte de lui-même dès que le solde est rétabli.» — OK.
- `tunnel.status.billing_inactive.detail`: «L’accès téléphone n’est pas encore activé pour ce compte Radient. Ouvrez la configuration de l’accès téléphone dans Local Operator sur l’ordinateur concerné et acceptez le prix mensuel.» — OK.
- `tunnel.status.card_paused`: «suspendu par Radient» — OK.

## Mechanical

Valid UTF-8 JSON; key parity 24/24 with en; placeholder sets identical on every key (only `{url}`, `{port}`, `{status}`, `{label}`, `{at}`, `{window}`, `{tally}` where the source has them); no ICU plural/select in this namespace; protected tokens (Radient, Local Operator, `lop …` commands, `/mobile`, HTTP, 127.0.0.1, `--tunnel-id`) unchanged; no tunnel/remote-access wording in the scope values; 'phone access' explicit on every scope value except the label and card_paused; no manual bidi controls; `check.py --catalogues-only`: 0 problems. French: U+202F before every ; : ! ? in prose; no ASCII apostrophes. A negative control (faults injected into a scratch copy) shows the checker reports placeholder, typography, bidi-control and missing-term faults (audit.log).

## Findings

| ID | severity | status | key | issue | suggested fix |
|---|---|---|---|---|---|
| FR-01 | major | resolved | card_add_credit, card_open_console | Raised in a4; fixed by the translator's new value (see the Changed keys table). | — |

No new findings this round.

## Considered, not raised

- Dropping 'to resume' on card_add_credit (the ur pass raised the same omission as UR-07 minor). Resume-keeping wordings are over budget or lose 'add credit' («Rechargez Radient pour rétablir l’accès téléphone» 51; «Rechargez Radient et retrouvez l’accès téléphone» 50 but loses 'credit'). Accepted, not raised.
- «sur» vs «dans la console Radient»: budget-forced and idiomatic; not raised.

## Not verified

No native-speaker review and no rendered screenshot of the terminal, desktop or phone frame; widths are `cell_len` measurements plus the real `tunnel_park_card` string. Judgement is by reasoning only; no MT service was used. This is the requested second independent pass over the remediation-scope keys; I did not re-litigate the 16 unchanged keys that earlier committed rounds passed.
