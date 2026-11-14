# i18n audit a3 — es / tunnel.status (branch fix/tunnel-remedies-auto-up @ 1fce7ecb1f5f)

Head audited: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (worktree .worktrees/tunnel-remedies-auto-up, branch fix/tunnel-remedies-auto-up). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent on node lop-remote-node-2, session 110287e9c4f3; not the translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 110287e9c4f3 (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (values in commit 1fce7ecb1f5f; English source drafted by coder session b0300e2c2c7f); model anthropic/claude-opus-5-5 (as stated by the coordinator brief; not independently verified); 2026-11-14.
Source: en sha256 b368acfb0cdee89c8de5a210fc9956251701a9f3f73c88b1096611d0c1f9063f (21 keys); target sha256 794abddea120435d962dfd93c1f6883232824f1777c97499fe0496da11a28700. Prior rounds on disk: a1, a2.

**Verdict: passed** — 0 unresolved blocker/major; 3 minor/nit finding(s).

Scope: the whole namespace at the new English source hash. The 8 keys new in 1fce7ecb1f5f were judged in full; the other keys were re-read for consistency only (already passed in earlier rounds; unchanged in this commit).

Read: docs/I18N.md 'Add or extend a locale', i18n/glossary.md (no row for phone access/tunnel/billing/credit), i18n/style-guides/es.md, en + context sidecar, corpus precedents (mobile.setup.balance/consent/billing_activated, tunnel.status.server_unavailable.*).

## Per-key judgement (new keys)

| key | en | es | judgement |
|---|---|---|---|
| `tunnel.status.tunnel_inactive.label` | tunnel not active | túnel no activo | OK — «túnel no activo», lowercase fragment; distinct from «en pausa» in detail, acceptable. |
| `tunnel.status.tunnel_inactive.detail` | Radient has paused this tunnel, but its billing is in order. Reconnect phone access in Local Operator on the computer running it; it also keeps retrying by itself. If it stays paused, review the tunnel at {url}. | Radient ha pausado este túnel, aunque su facturación está en regla. Reconecta el acceso telefónico en Local Operator, en el equipo donde se ejecuta; además, sigue reintentándolo por sí solo. Si continúa en pausa, revisa el túnel en {url}. | OK — paused-but-billing-fine meaning kept («aunque su facturación está en regla»); fix = reconnect in Local Operator; no command; {url} verbatim. ES-04 nit (implicit subject of «sigue reintentándolo»). |
| `tunnel.status.tunnel_disabled.label` | tunnel turned off in the console | túnel desactivado en la consola | OK. |
| `tunnel.status.tunnel_disabled.detail` | This tunnel is turned off in the Radient console, so remote access stays off until it is turned back on there: {url}. It reconnects by itself once it is on. | Este túnel está desactivado en la consola de Radient, así que el acceso remoto seguirá desactivado hasta que vuelvas a activarlo allí: {url}. Se reconectará automáticamente en cuanto esté activado. | OK — console-only fix, auto-reconnect once on; {url} verbatim; tú register. |
| `tunnel.status.billing_inactive.label` | billing not activated | facturación no activada | OK. |
| `tunnel.status.billing_inactive.detail` | Billing for this tunnel has not been activated, so Radient will not connect it. Open phone access setup in Local Operator on the computer running it and accept the monthly price. | La facturación de este túnel no se ha activado, así que Radient no lo conectará. Abre la configuración del acceso telefónico en Local Operator, en el equipo donde se ejecuta, y acepta el precio mensual. | OK — fix = accept monthly price in setup; matches corpus «precio mensual» (mobile.setup.billing_activated); no command. |
| `tunnel.status.billing_lapsed.label` | billing lapsed | facturación caducada | ES-02 minor — «caducada» reads as date expiry; cause is balance exhaustion. |
| `tunnel.status.billing_lapsed.detail` | This tunnel's billing lapsed because the Radient balance ran out, so Radient will not connect it. Add credit at {url}; it reconnects by itself once the balance is restored. | La facturación de este túnel caducó porque se agotó el saldo de Radient, así que Radient no lo conectará. Añade crédito en {url}; se reconectará automáticamente en cuanto se restablezca el saldo. | OK on meaning (cause «se agotó el saldo», fix «Añade crédito en {url}», auto-reconnect); ES-02 applies to «caducó». |

## Safety meanings

No terminal command is named in any new key (only `/mobile enable`, verbatim, in mobile.setup.login_offer). Money clauses survive where the English has them (nothing charged; setup never purchases credit; never activates billing or spends credit; price shown before billing) — regex presence logged per key in audit.log, meaning judged above. {url} and the console URL are byte-identical to en.

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| ES-02 | minor | `tunnel.status.billing_lapsed.label`, `tunnel.status.billing_lapsed.detail` | «facturación caducada/caducó» suggests a date-based expiry; the cause is the prepaid balance running out (answers translator open question). Detail sentences state the cause and fix correctly, so the risk is low. | label «facturación sin saldo» (or «facturación interrumpida»); detail «La facturación de este túnel se interrumpió porque se agotó el saldo de Radient…». |
| ES-03 | nit | `tunnel.status.tunnel_inactive.detail`, `tunnel.status.billing_inactive.detail` | Corpus-level, not introduced here: «acceso telefónico» is also the established es term for dial-up access, and «equipo» for computer collides with the es guide's «equipo (team)»; the corpus is mixed (máquina 16 / equipo 13 / ordenador 6 for 'computer'). New keys follow the already-audited tunnel.status/mobile.setup precedent, so no change is requested in this PR. | Record a glossary/style-guide row for 'phone access' and 'computer' in a later wave. |
| ES-04 | nit | `tunnel.status.tunnel_inactive.detail` | «además, sigue reintentándolo por sí solo» has no explicit subject after a clause whose subject is the user. | «además, el túnel sigue reintentándolo por sí solo». |

Minor/nit findings are advisory and do not block; the coordinator may batch them into a translator revision.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used. Translator model identity is as stated by the coordinator, not independently verified.
