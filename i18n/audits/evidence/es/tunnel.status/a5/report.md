# Independent translation re-audit — es / tunnel.status / a5

Base: 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the uncommitted working-tree catalogue changes for local-operator PR #3945 (round-1 remediation, branch fix/tunnel-remedies-auto-up), after the translator's fixes to the a4 findings. Date: 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent, session 5ac9acc2ba69 (round-1 remediation re-audit); neither the i18n-translator (c0c9e9803405), the English-drafting coder (889b9f67edf5), nor the prior auditors (117e988422ff, dff33b90744d); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 5ac9acc2ba69 (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (#3945 round-1 remediation fixes, uncommitted on 7c0fc7bf2); English source drafted by coder session 889b9f67edf5; model anthropic/claude-opus-5-5 (as recorded in the prior bundle; not otherwise verified).
Prior audit: i18n/audits/evidence/es/tunnel.status/a4/record.json (target sha256 3000565784e14521b953f6d2bdf5b86dc0b3d6e39805f68adda8c13b16c1ce3a).
Source: en sha256 e2bc8c63d3124c363252511306fd3951917ab9f93a2398c49cac8c79302aa8f9 (24 keys); context sha256 78580b806d2acc2c0c84bcf2655b62ea5ae847bf206c43134762e288bba8695d; target sha256 c39b00cd5218230e0a522fd7164ae19096f9bc934fd4a7866770e6f6bccc2abb (current bytes).

**Verdict: passed**: 0 unresolved blocker/major. Prior findings resolved: ES-01. Open carried nits: ES-02, ES-03. New findings: none.

## Scope and method

Mechanical integrity was re-verified for the whole namespace (24 keys) at its current bytes. Exactly the values the translator reported changed since a4: `card_add_credit`, `card_open_console`. The other scope values match the a4 report verbatim, and the 16 non-scope keys are byte-identical to HEAD (audit.log: `delta.py` against the prior report table, since the prior target bytes are not retained, and `git diff HEAD -U0`). Because this group is high-visibility (status/error copy), I re-read all 8 remediation-scope keys independently instead of assuming the earlier pass was right. Judgement is by reasoning against the en source, its context sidecar, i18n/glossary.md (no row for phone access, credit, balance or console) and i18n/style-guides/es.md. No MT service was used.

Card fit: each line is measured with `rich.cells.cell_len` including the 2-cell '! ' prefix (card_paused after '! /mobile enable — '); the budget is 50 cells (tests/unit/test_tunnel_remedies.py; tui/app.py `tunnel_park_card`). 'Phone access' must be explicit on every card line except card_paused, which follows the `/mobile enable` command. The real `tunnel_park_card` and `gateway.reason_label` render path was also run per locale (audit.log). Before the ledger row, es/fr/ru were unshipped (shipped.json had dropped them on the changes_requested rows), so the pre-ledger render fell back to English; the post-append `generate.py` run and re-render are recorded in audit.log under 'Post-append verification'.

## Changed keys (judged in full)

| key | en | current value | cells | judgement |
|---|---|---|---|---|
| `tunnel.status.card_add_credit` | Add Radient credit to resume phone access | Añade saldo en Radient para el acceso telefónico | 50 | RESOLVED (ES-01). «Añade saldo en Radient para el acceso telefónico» names the feature explicitly; «Añade saldo» is natural top-up copy, tú imperative per guide; Radient verbatim; no command/URL. 50/50 cells (zero headroom, within budget). The card no longer says 'resume' (see Considered, not raised). |
| `tunnel.status.card_open_console` | Turn phone access on in the Radient console | Activa el acceso telefónico en la web de Radient | 50 | RESOLVED (ES-01). «Activa el acceso telefónico en la web de Radient» keeps «acceso telefónico». «la web de Radient» departs from the corpus «consola de Radient» (7 uses), but the context sidecar calls it the 'Radient web console' and «…en la consola Radient» is 51 cells, so it is a budget-forced, accurate rendering. 50/50 cells. |

## Remaining scope keys (re-read, unchanged since the prior round)

- `tunnel.status.tunnel_inactive.label`: «pausado por Radient — reconectando» — OK: masculine agrees with the connector frame `Connector: retrying — {label}`.
- `tunnel.status.tunnel_inactive.detail`: «Radient ha pausado el acceso telefónico de este equipo, aunque la facturación está en regla. Reconecta el acceso telefónico en Local Operator, en el equipo donde se ejecuta; además, sigue reintentándolo por sí solo. Si continúa en pausa, revísalo en la consola de Radient, en {url}.» — OK on meaning; ES-03 nit still open.
- `tunnel.status.tunnel_disabled.detail`: «El acceso telefónico de este equipo está desactivado en la consola de Radient. Vuelve a activarlo en {url}; se reconectará automáticamente.» — OK.
- `tunnel.status.billing_lapsed.detail`: «Se agotó tu saldo de Radient, así que el acceso telefónico está en pausa. Añade crédito en {url}; se reconectará automáticamente en cuanto se restablezca el saldo.» — OK.
- `tunnel.status.billing_inactive.detail`: «El acceso telefónico aún no está activado para esta cuenta de Radient. Abre la configuración del acceso telefónico en Local Operator, en ese equipo, y acepta el precio mensual.» — OK on meaning; ES-02 nit still open.
- `tunnel.status.card_paused`: «pausado por Radient» — OK.

## Mechanical

Valid UTF-8 JSON; key parity 24/24 with en; placeholder sets identical on every key (only `{url}`, `{port}`, `{status}`, `{label}`, `{at}`, `{window}`, `{tally}` where the source has them); no ICU plural/select in this namespace; protected tokens (Radient, Local Operator, `lop …` commands, `/mobile`, HTTP, 127.0.0.1, `--tunnel-id`) unchanged; no tunnel/remote-access wording in the scope values; 'phone access' explicit on every scope value except the label and card_paused; no manual bidi controls; `check.py --catalogues-only`: 0 problems. A negative control (faults injected into a scratch copy) shows the checker reports placeholder, typography, bidi-control and missing-term faults (audit.log).

## Findings

| ID | severity | status | key | issue | suggested fix |
|---|---|---|---|---|---|
| ES-01 | major | resolved | card_add_credit, card_open_console | Raised in a4; fixed by the translator's new value (see the Changed keys table). | — |
| ES-02 | nit | open (carried, not re-counted as new) | `tunnel.status.billing_inactive.detail` | «en ese equipo» still has no antecedent (inherited from the en source's «on that computer»); unchanged since a4. | «en el equipo correspondiente». |
| ES-03 | nit | open (carried, not re-counted as new) | `tunnel.status.tunnel_inactive.detail` | «además, sigue reintentándolo por sí solo» still follows a clause whose subject is the user; unchanged since a4. | «además, se sigue reintentando automáticamente». |

No new findings this round.

## Considered, not raised

- Dropping 'to resume' on card_add_credit (the ur pass raised the same omission as UR-07 minor). Every resume-keeping wording I tried is over 50 cells or loses 'add credit' («Recarga saldo para recuperar el acceso telefónico» 51; «Recarga Radient y recupera el acceso telefónico» 49 but elliptical); the card's '!' glyph already marks a stopped state. Accepted as the better trade-off, not raised.
- «la web de Radient» vs «consola de Radient»: see the card_open_console row. A glossary row for 'Radient console' would settle it for every locale.
- «saldo» on the card vs «Añade crédito» in billing_lapsed.detail: both natural es for prepaid balance; the detail itself pairs «saldo»/«crédito». Not raised.
- Carried, not re-counted: a3 ES-03 («acceso telefónico» also reads as dial-up; «equipo» for 'computer' vs the guide's «equipo (team)»). The fixes use the corpus term, which is the right call until the glossary decides.

## Not verified

No native-speaker review and no rendered screenshot of the terminal, desktop or phone frame; widths are `cell_len` measurements plus the real `tunnel_park_card` string. Judgement is by reasoning only; no MT service was used. This is the requested second independent pass over the remediation-scope keys; I did not re-litigate the 16 unchanged keys that earlier committed rounds passed.
