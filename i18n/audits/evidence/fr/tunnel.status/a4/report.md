# i18n audit a4 — fr / tunnel.status (PR #3945 round-1 remediation, branch fix/tunnel-remedies-auto-up)

Base audited: HEAD 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the UNCOMMITTED working-tree catalogue values (`git diff -- local_operator/i18n/catalogues/`), worktree .worktrees/tunnel-remedies-auto-up. Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent (audit-es-fr-ru) on host ip-172-31-18-0, session 117e988422ff; not the translator (c0c9e9803405) nor the English-drafting coder (889b9f67edf5); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 117e988422ff, `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (uncommitted working-tree values on 7c0fc7bf2; English source drafted by coder session 889b9f67edf5); model anthropic/claude-opus-5-5 (read from that session's transcript metadata, not otherwise verified).
Source: en sha256 e2bc8c63d3124c363252511306fd3951917ab9f93a2398c49cac8c79302aa8f9 (24 keys); context sha256 78580b806d2acc2c0c84bcf2655b62ea5ae847bf206c43134762e288bba8695d; target sha256 7d1f849749155980cccd2d9c219369cd18c280608b42e430cfeccae8b794a87c. Prior rounds on disk: a1, a2, a3 (previous ledger row audited in commit f0300179e).

**Verdict: changes_requested** — 1 unresolved major; 1 finding(s): 1 major.

Scope: the 8 keys changed or new in the round-1 remediation, judged in full against the en source, its context sidecar, i18n/glossary.md (no row for phone access/credit/balance/console; device → appareil not applicable to 'computer') and i18n/style-guides/fr.md. Unchanged keys were not re-audited (already passed in earlier rounds); key parity and the catalogue checker cover the whole namespace (audit.log).

Design intent checked: "phone access" on every surface (no tunnel / remote-access wording), {url}, `/mobile enable`, Radient and Local Operator verbatim, no command named in the console-only remedies. Card lines measured with `rich.cells.cell_len` including the 2-cell "! " prefix (card_paused after "! /mobile enable — "), budget ≤ 50 cells (tests/unit/test_tunnel_remedies.py).

## Per-key judgement

| key | en | fr | judgement |
|---|---|---|---|
| `tunnel.status.tunnel_inactive.label` | paused by Radient — reconnecting | suspendu par Radient — reconnexion en cours | OK — «suspendu par Radient — reconnexion en cours»; lowercase fragment. Longer than en (no budget on this status-line label). |
| `tunnel.status.tunnel_inactive.detail` | Radient has paused phone access for this computer, but billing is in order. Reconnect phone access in Local Operator on the computer running it; it also keeps retrying by itself. If it stays paused, review it in the Radient console at {url}. | Radient a suspendu l’accès téléphone de cet ordinateur, mais la facturation est en règle. Reconnectez l’accès téléphone dans Local Operator sur l’ordinateur qui l’héberge ; il continue aussi de réessayer de lui-même. S’il reste suspendu, vérifiez-le dans la console Radient : {url}. | OK — vous register, U+202F before ; and :, «vérifiez-le dans la console Radient : {url}». |
| `tunnel.status.tunnel_disabled.detail` | Phone access for this computer is turned off in the Radient console. Turn it back on at {url}; it reconnects by itself. | L’accès téléphone de cet ordinateur est désactivé dans la console Radient. Réactivez-le sur {url} ; il se reconnecte de lui-même. | OK — «Réactivez-le sur {url} ; il se reconnecte de lui-même.» |
| `tunnel.status.billing_lapsed.detail` | Your Radient balance has run out, so phone access is paused. Add credit at {url}; it reconnects by itself once the balance is restored. | Votre solde Radient est épuisé, donc l’accès téléphone est suspendu. Ajoutez du crédit sur {url} ; il se reconnecte de lui-même dès que le solde est rétabli. | OK — «Votre solde Radient est épuisé, donc l’accès téléphone est suspendu.» |
| `tunnel.status.billing_inactive.detail` | Phone access isn't turned on for this Radient account yet. Open phone access setup in Local Operator on that computer and accept the monthly price. | L’accès téléphone n’est pas encore activé pour ce compte Radient. Ouvrez la configuration de l’accès téléphone dans Local Operator sur l’ordinateur concerné et acceptez le prix mensuel. | OK — «sur l’ordinateur concerné» resolves the en source's dangling «that computer» gracefully. |
| `tunnel.status.card_paused` | paused by Radient | suspendu par Radient | OK — 39/50 cells after «/mobile enable — ». |
| `tunnel.status.card_add_credit` | Add Radient credit to resume phone access | Ajoutez du crédit Radient pour rétablir l’accès | FR-01 MAJOR — drops «téléphone»; 49/50 cells. |
| `tunnel.status.card_open_console` | Turn phone access on in the Radient console | Activez l’accès dans la console Radient | FR-01 MAJOR — drops «téléphone»; 41/50 cells. |

## Mechanical

Placeholders identical (only `{url}`), no ICU in scope, Radient / Local Operator verbatim, key parity 24/24, valid JSON, check.py 0 problems; no tunnel/remote-access term in any in-scope value. U+202F before every ; and :, no ASCII apostrophes. The only "phone access" omissions the term check reports are the card lines in the finding below (audit.log).

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| FR-01 | major | `tunnel.status.card_add_credit`, `tunnel.status.card_open_console` | Both card lines drop «téléphone»: «Ajoutez du crédit Radient pour rétablir l’accès», «Activez l’accès dans la console Radient». On a standalone one-line TUI card bare «l’accès» names no feature, and Radient credit also funds model usage, so the add-credit line reads as restoring Radient access. Design intent requires «accès téléphone»; compliant wordings fit 50 cells (rich cell_len incl. '! '). | card_add_credit «Ajoutez du crédit Radient pour l’accès téléphone» (50 cells); card_open_console «Activez l’accès téléphone sur la console Radient» (50). («…dans la console Radient» is 51.) |

## Not verified

No native-speaker review and no rendered terminal/desktop/phone frame; widths are cell_len measurements, not screenshots. Judgement by reasoning only; no MT service used. This is one independent pass; a second independent pass for this high-visibility (error/status) group is the manager's to schedule.
