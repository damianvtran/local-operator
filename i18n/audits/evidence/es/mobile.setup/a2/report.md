# i18n audit a2 — es / mobile.setup (PR #3945 round-1 remediation, branch fix/tunnel-remedies-auto-up)

Base audited: HEAD 7c0fc7bf22aade04f78c367c906374b73a1914d7 plus the UNCOMMITTED working-tree catalogue values (`git diff -- local_operator/i18n/catalogues/`), worktree .worktrees/tunnel-remedies-auto-up. Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent (audit-es-fr-ru) on host ip-172-31-18-0, session 117e988422ff; not the translator (c0c9e9803405) nor the English-drafting coder (889b9f67edf5); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 117e988422ff, `Model: anthropic/claude-opus-5-5`; no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session c0c9e9803405 (uncommitted working-tree values on 7c0fc7bf2; English source drafted by coder session 889b9f67edf5); model anthropic/claude-opus-5-5 (read from that session's transcript metadata, not otherwise verified).
Source: en sha256 0d245737d539f4d2fe54cf3a2c2174b0aa64820d2eb387e94ea16f65d1b487d7 (94 keys); context sha256 b1c0c11a269077507f7d22256f3874d7f5fd1a363f5c13f412014de70c7f038f; target sha256 3bf4e9f634cd66f9655e59db241e28aa01db0a8c3e631ef3cdcb1a254c64644f. Prior rounds on disk: a1, r1, w5a1 (previous ledger row audited in commit f0300179e).

**Verdict: passed** — 0 unresolved blocker/major; 1 finding(s): 1 nit.

Scope: the 12 keys changed or new in the round-1 remediation, judged in full against the en source, its context sidecar, i18n/glossary.md (no row for phone access/credit/balance/console; device → dispositivo not applicable to 'computer') and i18n/style-guides/es.md. Unchanged keys were not re-audited (already passed in earlier rounds); key parity and the catalogue checker cover the whole namespace (audit.log).

Design intent checked: "phone access" on every surface (no tunnel / remote-access wording), {url}, `/mobile enable`, Radient and Local Operator verbatim, no command named in the console-only remedies. Card lines measured with `rich.cells.cell_len` including the 2-cell "! " prefix (card_paused after "! /mobile enable — "), budget ≤ 50 cells (tests/unit/test_tunnel_remedies.py).

## Per-key judgement

| key | en | es | judgement |
|---|---|---|---|
| `mobile.setup.disabled` | Phone access for this computer is turned off in the Radient console. It reconnects by itself once it is back on; setup will not override that choice. | El acceso telefónico de este equipo está desactivado en la consola de Radient. Se reconectará automáticamente en cuanto vuelva a estar activado; la configuración no anulará esa decisión. | OK — «la configuración no anulará esa decisión» keeps the deliberate non-override; no URL in en, none added. |
| `mobile.setup.billing_lapsed` | Your Radient balance has run out, so phone access is paused. Setup never purchases credit; phone access reconnects by itself once the balance is restored. | Se agotó tu saldo de Radient, así que el acceso telefónico está en pausa. La configuración nunca compra crédito; el acceso telefónico se reconectará automáticamente en cuanto se restablezca el saldo. | OK — safety clause «La configuración nunca compra crédito» kept; auto-reconnect. |
| `mobile.setup.reconnect` | Radient has paused phone access for this computer, but billing is in order. Reconnecting brings it back; nothing is charged. | Radient ha pausado el acceso telefónico de este equipo, aunque la facturación está en regla. Al reconectar, vuelve a funcionar; no se cobra nada. | OK — «Al reconectar, vuelve a funcionar; no se cobra nada.» keeps the no-charge clause. |
| `mobile.setup.network` | This computer could not reach Radient (its network may have changed or dropped). Phone access reconnects by itself once the connection is back. You don't need to do anything. | Este equipo no pudo conectar con Radient (puede que su red haya cambiado o se haya caído). El acceso telefónico se reconectará automáticamente en cuanto vuelva la conexión. No tienes que hacer nada. | OK — asks nothing of the reader; split sentence mirrors en. |
| `mobile.setup.title_reconnect` | Phone access is paused by Radient | El acceso telefónico está pausado por Radient | ES-01 nit — passive-with-agent calque. |
| `mobile.setup.title_billing_lapsed` | Phone access needs more Radient credit | El acceso telefónico necesita más crédito de Radient | OK. |
| `mobile.setup.title_disabled` | Phone access is turned off in the Radient console | El acceso telefónico está desactivado en la consola de Radient | OK. |
| `mobile.setup.title_network` | Phone access is waiting for the network | El acceso telefónico espera a que vuelva la red | OK — «espera a que vuelva la red» idiomatic. |
| `mobile.setup.title_radient_unavailable` | Phone access is waiting for Radient | El acceso telefónico espera a Radient | OK. |
| `mobile.setup.title_reconnecting` | Phone access is reconnecting | El acceso telefónico se está reconectando | OK. |
| `mobile.setup.action_add_credit` | Add credit in the Radient console: {url} | Añade crédito en la consola de Radient: {url} | OK — imperative tú, {url} verbatim; matches «Añade crédito» in tunnel.status.billing_lapsed.detail. |
| `mobile.setup.action_open_console` | Turn phone access back on in the Radient console: {url} | Vuelve a activar el acceso telefónico en la consola de Radient: {url} | OK. |

## Mechanical

Placeholders identical (only `{url}`), no ICU in scope, Radient / Local Operator verbatim, key parity 94/94, valid JSON, check.py 0 problems; no tunnel/remote-access term in any in-scope value. The only "phone access" omissions the term check reports are the card lines in the finding below (audit.log).

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| ES-01 | nit | `mobile.setup.title_reconnect` | «El acceso telefónico está pausado por Radient» — passive with explicit agent is an English calque; the body sentence (mobile.setup.reconnect) already uses the natural active form. | «Radient ha pausado el acceso telefónico». |

## Not verified

No native-speaker review and no rendered terminal/desktop/phone frame; widths are cell_len measurements, not screenshots. Judgement by reasoning only; no MT service used. This is one independent pass; a second independent pass for this high-visibility (error/status) group is the manager's to schedule.
