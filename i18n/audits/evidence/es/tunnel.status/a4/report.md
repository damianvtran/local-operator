# i18n audit a4 — es / tunnel.status (PR #3945 round-1 remediation, branch fix/tunnel-remedies-auto-up)

Base audited: HEAD 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the UNCOMMITTED working-tree catalogue values (`git diff -- local_operator/i18n/catalogues/`), worktree .worktrees/tunnel-remedies-auto-up. Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent (audit-es-fr-ru) on host ip-172-31-18-0, session 117e988422ff; not the translator (c0c9e9803405) nor the English-drafting coder (889b9f67edf5); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 117e988422ff, `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (uncommitted working-tree values on 7c0fc7bf2; English source drafted by coder session 889b9f67edf5); model anthropic/claude-opus-5-5 (read from that session's transcript metadata, not otherwise verified).
Source: en sha256 e2bc8c63d3124c363252511306fd3951917ab9f93a2398c49cac8c79302aa8f9 (24 keys); context sha256 78580b806d2acc2c0c84bcf2655b62ea5ae847bf206c43134762e288bba8695d; target sha256 3000565784e14521b953f6d2bdf5b86dc0b3d6e39805f68adda8c13b16c1ce3a. Prior rounds on disk: a1, a2, a3 (previous ledger row audited in commit f0300179e).

**Verdict: changes_requested** — 1 unresolved major; 3 finding(s): 1 major, 2 nit.

Scope: the 8 keys changed or new in the round-1 remediation, judged in full against the en source, its context sidecar, i18n/glossary.md (no row for phone access/credit/balance/console; device → dispositivo not applicable to 'computer') and i18n/style-guides/es.md. Unchanged keys were not re-audited (already passed in earlier rounds); key parity and the catalogue checker cover the whole namespace (audit.log).

Design intent checked: "phone access" on every surface (no tunnel / remote-access wording), {url}, `/mobile enable`, Radient and Local Operator verbatim, no command named in the console-only remedies. Card lines measured with `rich.cells.cell_len` including the 2-cell "! " prefix (card_paused after "! /mobile enable — "), budget ≤ 50 cells (tests/unit/test_tunnel_remedies.py).

## Per-key judgement

| key | en | es | judgement |
|---|---|---|---|
| `tunnel.status.tunnel_inactive.label` | paused by Radient — reconnecting | pausado por Radient — reconectando | OK — «pausado por Radient — reconectando»: gerund status fragment, lowercase per guide; agrees with implied «acceso». |
| `tunnel.status.tunnel_inactive.detail` | Radient has paused phone access for this computer, but billing is in order. Reconnect phone access in Local Operator on the computer running it; it also keeps retrying by itself. If it stays paused, review it in the Radient console at {url}. | Radient ha pausado el acceso telefónico de este equipo, aunque la facturación está en regla. Reconecta el acceso telefónico en Local Operator, en el equipo donde se ejecuta; además, sigue reintentándolo por sí solo. Si continúa en pausa, revísalo en la consola de Radient, en {url}. | OK on meaning (paused by Radient, billing fine, Reconnect in Local Operator, auto-retry, console at {url}). ES-03 nit: «además, sigue reintentándolo por sí solo» has no clear subject. |
| `tunnel.status.tunnel_disabled.detail` | Phone access for this computer is turned off in the Radient console. Turn it back on at {url}; it reconnects by itself. | El acceso telefónico de este equipo está desactivado en la consola de Radient. Vuelve a activarlo en {url}; se reconectará automáticamente. | OK — console-only fix, auto-reconnect; tú register; {url} verbatim. |
| `tunnel.status.billing_lapsed.detail` | Your Radient balance has run out, so phone access is paused. Add credit at {url}; it reconnects by itself once the balance is restored. | Se agotó tu saldo de Radient, así que el acceso telefónico está en pausa. Añade crédito en {url}; se reconectará automáticamente en cuanto se restablezca el saldo. | OK — «Se agotó tu saldo de Radient» natural; fix «Añade crédito en {url}»; auto-reconnect. Old «caducó» (a3 ES-02) gone from this key. |
| `tunnel.status.billing_inactive.detail` | Phone access isn't turned on for this Radient account yet. Open phone access setup in Local Operator on that computer and accept the monthly price. | El acceso telefónico aún no está activado para esta cuenta de Radient. Abre la configuración del acceso telefónico en Local Operator, en ese equipo, y acepta el precio mensual. | OK on meaning (not turned on for this account; accept the monthly price in setup). ES-02 nit: «en ese equipo» has no antecedent (inherited from en «on that computer»). |
| `tunnel.status.card_paused` | paused by Radient | pausado por Radient | OK — 38/50 cells after «/mobile enable — ». |
| `tunnel.status.card_add_credit` | Add Radient credit to resume phone access | Añade saldo en Radient para reanudar el acceso | ES-01 MAJOR — drops «telefónico»; 48/50 cells. |
| `tunnel.status.card_open_console` | Turn phone access on in the Radient console | Activa el acceso en la consola de Radient | ES-01 MAJOR — drops «telefónico»; 43/50 cells. |

## Mechanical

Placeholders identical (only `{url}`), no ICU in scope, Radient / Local Operator verbatim, key parity 24/24, valid JSON, check.py 0 problems; no tunnel/remote-access term in any in-scope value. The only "phone access" omissions the term check reports are the card lines in the finding below (audit.log).

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| ES-01 | major | `tunnel.status.card_add_credit`, `tunnel.status.card_open_console` | Both card lines drop «telefónico»: «Añade saldo en Radient para reanudar el acceso», «Activa el acceso en la consola de Radient». The card is one standalone TUI line; bare «el acceso» names no feature, and because Radient credit also funds model usage in this product, the add-credit line reads as restoring Radient/model access. Design intent requires 'phone access' on these surfaces, and compliant wordings fit the 50-cell budget (measured with rich cell_len incl. the '! ' prefix). «saldo» vs sibling «crédito» is acceptable es idiom, not separately counted. | card_add_credit «Añade saldo en Radient para el acceso telefónico» (50 cells); card_open_console «Activa el acceso telefónico en la web de Radient» (50) or «Activa el acceso telefónico en Radient» (40). («…en la consola Radient» is 51, one over.) |
| ES-02 | nit | `tunnel.status.billing_inactive.detail` | «en ese equipo» has no antecedent; inherited from the en source's dangling «on that computer» (reported to the coordinator as a source note). | «en el equipo correspondiente». |
| ES-03 | nit | `tunnel.status.tunnel_inactive.detail` | Re-raise of a3 ES-04, still present in this changed key: «además, sigue reintentándolo por sí solo» follows a clause whose subject is the user. | «además, se sigue reintentando automáticamente». |

Carried, not re-counted: a3 ES-03 (corpus «equipo» for 'computer' vs the guide's «equipo (team)»; «acceso telefónico» also means dial-up) still applies to the new «este equipo» uses; a later glossary row should settle 'phone access' and 'computer'.

Source note for the coordinator (not a translation finding): en `tunnel.status.billing_inactive.detail` says «on that computer» with no computer named earlier in the string.

## Not verified

No native-speaker review and no rendered terminal/desktop/phone frame; widths are cell_len measurements, not screenshots. Judgement by reasoning only; no MT service used. This is one independent pass; a second independent pass for this high-visibility (error/status) group is the manager's to schedule.
