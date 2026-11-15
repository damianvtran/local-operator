# i18n audit a2 — fr / mobile.setup (PR #3945 round-1 remediation, branch fix/tunnel-remedies-auto-up)

Base audited: HEAD 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the UNCOMMITTED working-tree catalogue values (`git diff -- local_operator/i18n/catalogues/`), worktree .worktrees/tunnel-remedies-auto-up. Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent (audit-es-fr-ru) on host ip-172-31-18-0, session 117e988422ff; not the translator (c0c9e9803405) nor the English-drafting coder (889b9f67edf5); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 117e988422ff, `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (uncommitted working-tree values on 7c0fc7bf2; English source drafted by coder session 889b9f67edf5); model anthropic/claude-opus-5-5 (read from that session's transcript metadata, not otherwise verified).
Source: en sha256 0d245737d539f4d2fe54cf3a2c2174b0aa64820d2eb387e94ea16f65d1b487d7 (94 keys); context sha256 b1c0c11a269077507f7d22256f3874d7f5fd1a363f5c13f412014de70c7f038f; target sha256 0be4a710581f092ab486d21920e02e80f6c4fba0ce5eda27c27a2a0074117117. Prior rounds on disk: a1, frbk1, r1 (previous ledger row audited in commit f0300179e).

**Verdict: passed** — 0 unresolved blocker/major; 1 finding(s): 1 nit.

Scope: the 12 keys changed or new in the round-1 remediation, judged in full against the en source, its context sidecar, i18n/glossary.md (no row for phone access/credit/balance/console; device → appareil not applicable to 'computer') and i18n/style-guides/fr.md. Unchanged keys were not re-audited (already passed in earlier rounds); key parity and the catalogue checker cover the whole namespace (audit.log).

Design intent checked: "phone access" on every surface (no tunnel / remote-access wording), {url}, `/mobile enable`, Radient and Local Operator verbatim, no command named in the console-only remedies. Card lines measured with `rich.cells.cell_len` including the 2-cell "! " prefix (card_paused after "! /mobile enable — "), budget ≤ 50 cells (tests/unit/test_tunnel_remedies.py).

## Per-key judgement

| key | en | fr | judgement |
|---|---|---|---|
| `mobile.setup.disabled` | Phone access for this computer is turned off in the Radient console. It reconnects by itself once it is back on; setup will not override that choice. | L’accès téléphone de cet ordinateur est désactivé dans la console Radient. Il se reconnecte de lui-même dès qu’il est réactivé ; la configuration ne passera pas outre ce choix. | OK — «la configuration ne passera pas outre ce choix». |
| `mobile.setup.billing_lapsed` | Your Radient balance has run out, so phone access is paused. Setup never purchases credit; phone access reconnects by itself once the balance is restored. | Votre solde Radient est épuisé, donc l’accès téléphone est suspendu. La configuration n’achète jamais de crédit ; l’accès téléphone se reconnecte de lui-même dès que le solde est rétabli. | OK — «La configuration n’achète jamais de crédit» kept. |
| `mobile.setup.reconnect` | Radient has paused phone access for this computer, but billing is in order. Reconnecting brings it back; nothing is charged. | Radient a suspendu l’accès téléphone de cet ordinateur, mais la facturation est en règle. La reconnexion le rétablit ; rien n’est facturé. | OK — «La reconnexion le rétablit ; rien n’est facturé.» |
| `mobile.setup.network` | This computer could not reach Radient (its network may have changed or dropped). Phone access reconnects by itself once the connection is back. You don't need to do anything. | Cet ordinateur n’a pas pu joindre Radient (son réseau a peut-être changé ou été coupé). L’accès téléphone se reconnecte de lui-même dès que la connexion revient. Vous n’avez rien à faire. | OK — «Vous n’avez rien à faire.» |
| `mobile.setup.title_reconnect` | Phone access is paused by Radient | L’accès téléphone est suspendu par Radient | OK. |
| `mobile.setup.title_billing_lapsed` | Phone access needs more Radient credit | L’accès téléphone nécessite plus de crédit Radient | FR-01 nit — «plus de crédit» ambiguity. |
| `mobile.setup.title_disabled` | Phone access is turned off in the Radient console | L’accès téléphone est désactivé dans la console Radient | OK. |
| `mobile.setup.title_network` | Phone access is waiting for the network | L’accès téléphone attend le réseau | OK. |
| `mobile.setup.title_radient_unavailable` | Phone access is waiting for Radient | L’accès téléphone attend Radient | OK. |
| `mobile.setup.title_reconnecting` | Phone access is reconnecting | L’accès téléphone se reconnecte | OK. |
| `mobile.setup.action_add_credit` | Add credit in the Radient console: {url} | Ajoutez du crédit dans la console Radient : {url} | OK — U+202F before the colon; {url} verbatim. |
| `mobile.setup.action_open_console` | Turn phone access back on in the Radient console: {url} | Réactivez l’accès téléphone dans la console Radient : {url} | OK. |

## Mechanical

Placeholders identical (only `{url}`), no ICU in scope, Radient / Local Operator verbatim, key parity 94/94, valid JSON, check.py 0 problems; no tunnel/remote-access term in any in-scope value. U+202F before every ; and :, no ASCII apostrophes. The only "phone access" omissions the term check reports are the card lines in the finding below (audit.log).

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| FR-01 | nit | `mobile.setup.title_billing_lapsed` | «nécessite plus de crédit Radient» — written «plus de» can also be read as the negative 'no more credit', which is exactly the situation (balance exhausted); the title is short and read at a glance. | «L’accès téléphone nécessite du crédit Radient supplémentaire». |

## Not verified

No native-speaker review and no rendered terminal/desktop/phone frame; widths are cell_len measurements, not screenshots. Judgement by reasoning only; no MT service used. This is one independent pass; a second independent pass for this high-visibility (error/status) group is the manager's to schedule.
