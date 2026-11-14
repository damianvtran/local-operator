# Independent translation audit — es / mobile.setup — a3

- PR: #3958 (branch fix/tunnel-remedy-no-commands, stacked on #3945), head 01113e5aa048884a507d9bb4868815f909f63a48; scope `git diff 374037856..01113e5aa -- local_operator/i18n/` (12 new `*_plain` keys; 0 existing keys changed).
- Translator: coder subagent of host Local Operator session 25fe8dfbf610 (values committed in 01113e5aa on fix/tunnel-remedy-no-commands; English source by the same coder); model not disclosed to the auditor (recorded null).
- Auditor: translation-auditor (lopdev team), independent Local Operator session 9054cd229d3f; not the translator/coder of host session 25fe8dfbf610; edited no catalogue value; model anthropic/claude-opus-5-5 (session system prompt; no provider fallback during the audit).
- Date: 2026-11-14. Result: **PASSED** — 0 blocker / 0 major; 0 minor/nit finding(s), none gating.

## Method
Read side by side with en source + context sidecar (12 new keys, 0 changed); glossary agent/approvals/service and the namespace's phone-access, phone-address, sign-in, account and setup terms match #3945's shipped copy; register per style guide; no placeholders in source or target; no command, flag, slash or code-span text (setup._SHELL_COMMAND + slash/flag/backtick scan); NFC; no bidi controls; no maxCells on these keys — the TUI's 50-cell park card renders tunnel.status card_* keys, not these; the *_plain keys reach the TUI only through setup.text(surface='tui') in a wrapping system notice and the desktop's remedy.text. Runtime: setup.plain_remedy({'step': s}) equals the catalogue value for all 12 steps.

## Locale notes
Clean. tú register; 'este equipo' matches the namespace's 8 prior uses; 'aplicación auxiliar', 'servicio en segundo plano', 'interfaz del teléfono', 'dirección de teléfono' consistent with titles/consent/repoint.

## Findings
- None.

## Keys read (12)
| key | en | es |
|---|---|---|
| `mobile.setup.stopped_plain` | Phone access is turned off on this computer. Turn it back on whenever you want to reach this computer from your phone. | El acceso telefónico está desactivado en este equipo. Vuelve a activarlo cuando quieras acceder a este equipo desde tu teléfono. |
| `mobile.setup.owner_missing_plain` | The Radient account that set up phone access is no longer signed in on this computer. Sign in with that account to bring phone access back. | La cuenta de Radient que configuró el acceso telefónico ya no tiene la sesión iniciada en este equipo. Inicia sesión con esa cuenta para recuperar el acceso telefónico. |
| `mobile.setup.login_plain` | Sign in to Radient with the account that owns phone access, then try again. | Inicia sesión en Radient con la cuenta propietaria del acceso telefónico y vuelve a intentarlo. |
| `mobile.setup.choose_account_plain` | Choose the Radient account phone access should use. Setup never moves an existing phone address to a different account. | Elige la cuenta de Radient que debe usar el acceso telefónico. La configuración nunca pasa una dirección de teléfono existente a otra cuenta. |
| `mobile.setup.account_plain` | Radient could not confirm the account and phone address. Check your internet connection and try again; nothing was changed. | Radient no pudo confirmar la cuenta y la dirección de teléfono. Comprueba tu conexión a internet y vuelve a intentarlo; no se cambió nada. |
| `mobile.setup.cloudflared_plain` | Phone access needs a helper app called cloudflared, which is not installed on this computer. Ask your agent to install it (it asks for your approval first), then try again. | El acceso telefónico necesita una aplicación auxiliar llamada cloudflared, que no está instalada en este equipo. Pide a tu agente que la instale (antes te pedirá tu aprobación) y vuelve a intentarlo. |
| `mobile.setup.mobile_bundle_plain` | The phone interface is not prepared on this computer yet. Ask your agent to prepare it (it asks for your approval first), then try again. Nothing was changed. | La interfaz del teléfono aún no está preparada en este equipo. Pide a tu agente que la prepare (antes te pedirá tu aprobación) y vuelve a intentarlo. No se cambió nada. |
| `mobile.setup.supervisor_plain` | This computer cannot keep phone access running in the background yet. Ask your agent to set up a background service for it, then try again. Nothing was changed. | Este equipo todavía no puede mantener el acceso telefónico funcionando en segundo plano. Pide a tu agente que configure un servicio en segundo plano para ello y vuelve a intentarlo. No se cambió nada. |
| `mobile.setup.apply_plain` | Everything is ready. Turn on phone access to finish. | Todo está listo. Activa el acceso telefónico para terminar. |
| `mobile.setup.verification_plain` | The connection is not confirmed yet. Check again in a moment; completed steps stay saved. | La conexión aún no está confirmada. Vuelve a comprobarlo en un momento; los pasos completados siguen guardados. |
| `mobile.setup.failed_plain` | Phone access setup did not finish. Try again; completed steps stay saved. | La configuración del acceso telefónico no terminó. Vuelve a intentarlo; los pasos completados siguen guardados. |
| `mobile.setup.complete_plain` | Open the address above on your phone and sign in with the same Radient account. | Abre la dirección de arriba en tu teléfono e inicia sesión con la misma cuenta de Radient. |
