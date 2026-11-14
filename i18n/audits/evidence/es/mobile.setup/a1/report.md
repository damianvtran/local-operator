# i18n audit a1 — es / mobile.setup (branch fix/tunnel-remedies-auto-up @ 1fce7ecb1f5f)

Head audited: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (worktree .worktrees/tunnel-remedies-auto-up, branch fix/tunnel-remedies-auto-up). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent on node lop-remote-node-2, session 110287e9c4f3; not the translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 110287e9c4f3 (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (values in commit 1fce7ecb1f5f; English source drafted by coder session b0300e2c2c7f); model anthropic/claude-opus-5-5 (as stated by the coordinator brief; not independently verified); 2026-11-14.
Source: en sha256 6883f92e94dfcec4be0af6041a0531c974fbf18294f7094a132cc555a2ad2065 (86 keys); target sha256 4519a62b46386d65f1fa5dd1d89e30e83f04ca7755ad83b00dba61d1ff7d80c7. Prior rounds on disk: r1, w5a1.

**Verdict: passed** — 0 unresolved blocker/major; 3 minor/nit finding(s).

Scope: the whole namespace at the new English source hash. The 7 keys new in 1fce7ecb1f5f were judged in full; the other keys were re-read for consistency only (already passed in earlier rounds; unchanged in this commit).

Read: docs/I18N.md 'Add or extend a locale', i18n/glossary.md (no row for phone access/tunnel/billing/credit), i18n/style-guides/es.md, en + context sidecar, corpus precedents (mobile.setup.balance/consent/billing_activated, tunnel.status.server_unavailable.*).

## Per-key judgement (new keys)

| key | en | es | judgement |
|---|---|---|---|
| `mobile.setup.disabled` | This tunnel is turned off in the Radient console. Turn it back on at https://console.radienthq.com/dashboard/tunnels; it reconnects by itself, and setup will not override that choice. | Este túnel está desactivado en la consola de Radient. Vuelve a activarlo en https://console.radienthq.com/dashboard/tunnels; se reconectará automáticamente, y la configuración no anulará esa decisión. | OK — URL verbatim; «la configuración no anulará esa decisión» keeps the deliberate non-override. |
| `mobile.setup.billing_lapsed` | This tunnel's billing lapsed because the Radient balance ran out. Add credit at https://console.radienthq.com/dashboard/tunnels; setup never purchases credit, and the tunnel reconnects by itself once the balance is restored. | La facturación de este túnel caducó porque se agotó el saldo de Radient. Añade crédito en https://console.radienthq.com/dashboard/tunnels; la configuración nunca compra crédito, y el túnel se reconectará automáticamente en cuanto se restablezca el saldo. | OK on safety («la configuración nunca compra crédito», consistent with mobile.setup.balance); ES-02 applies to «caducó». |
| `mobile.setup.reconnect` | Radient has paused this tunnel, but its billing is in order. Reconnect to bring phone access back; nothing is charged. | Radient ha pausado este túnel, aunque su facturación está en regla. Reconecta para recuperar el acceso telefónico; no se cobra nada. | OK — «no se cobra nada». |
| `mobile.setup.network` | This computer could not reach Radient (its network may have changed or dropped). Phone access reconnects by itself once the connection is back; nothing needs doing. | Este equipo no pudo conectar con Radient (puede que su red haya cambiado o se haya caído). El acceso telefónico se reconectará automáticamente en cuanto vuelva la conexión; no tienes que hacer nada. | OK — asks nothing of the reader («no tienes que hacer nada»). |
| `mobile.setup.radient_unavailable` | Radient answered with a temporary server error. Nothing is wrong with this computer or its sign-in; phone access reconnects by itself once Radient recovers. | Radient respondió con un error temporal del servidor. No hay ningún problema con este equipo ni con su inicio de sesión; el acceso telefónico se reconectará automáticamente en cuanto Radient se recupere. | OK — exonerates computer and sign-in; automatic recovery. |
| `mobile.setup.reconnecting` | Phone access is reconnecting in the background after your sign-in. Nothing is charged; this clears by itself in a minute or two. | El acceso telefónico se está reconectando en segundo plano tras tu inicio de sesión. No se cobra nada; esto se resolverá solo en uno o dos minutos. | OK — «No se cobra nada»; self-clearing in «uno o dos minutos». |
| `mobile.setup.login_offer` | Reach this computer from the phone app: run /mobile enable. Any monthly price is shown to you before billing starts. | Accede a este equipo desde la aplicación del teléfono: ejecuta /mobile enable. Cualquier precio mensual se te mostrará antes de que empiece la facturación. | OK on safety (`/mobile enable` verbatim; price shown before billing). ES-01 minor («aplicación del teléfono»). |

## Safety meanings

No terminal command is named in any new key (only `/mobile enable`, verbatim, in mobile.setup.login_offer). Money clauses survive where the English has them (nothing charged; setup never purchases credit; never activates billing or spends credit; price shown before billing) — regex presence logged per key in audit.log, meaning judged above. {url} and the console URL are byte-identical to en.

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| ES-01 | minor | `mobile.setup.login_offer` | «la aplicación del teléfono» can read as the built-in Phone (dialer) app, which is literally named «Teléfono» on iOS/Android in Spanish (answers translator open question). | «la aplicación móvil» (pan-regional; the es guide's region warning is about «móvil» as the noun for the handset) or «la app para el teléfono». |
| ES-02 | minor | `mobile.setup.billing_lapsed` | «facturación caducada/caducó» suggests a date-based expiry; the cause is the prepaid balance running out (answers translator open question). Detail sentences state the cause and fix correctly, so the risk is low. | label «facturación sin saldo» (or «facturación interrumpida»); detail «La facturación de este túnel se interrumpió porque se agotó el saldo de Radient…». |
| ES-03 | nit | `mobile.setup.network`, `mobile.setup.radient_unavailable`, `mobile.setup.login_offer` | Corpus-level, not introduced here: «acceso telefónico» is also the established es term for dial-up access, and «equipo» for computer collides with the es guide's «equipo (team)»; the corpus is mixed (máquina 16 / equipo 13 / ordenador 6 for 'computer'). New keys follow the already-audited tunnel.status/mobile.setup precedent, so no change is requested in this PR. | Record a glossary/style-guide row for 'phone access' and 'computer' in a later wave. |

Minor/nit findings are advisory and do not block; the coordinator may batch them into a translator revision.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used. Translator model identity is as stated by the coordinator, not independently verified.
