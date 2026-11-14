# es/cli.mobile_help — independent i18n audit a1

**Verdict: PASSED.** blocker 0 · major 0 · minor 1 · nit 0; unresolved blocker/major: 0.

- Scope: whole namespace (17 keys) at the new English source hash: the 2 NEW keys (cli.mobile_help.doctor, cli.mobile_help.doctor_json) judged in full; the 15 previously audited keys re-read for consistency with the new keys and with mobile.doctor, and any defect seen is recorded.
- Revision: `a3722be7dadacaf140a2587a0100bd3b26c2ab50` (feat/mobile-doctor, worktree mobile-doctor, node lop-remote-node-2).
- English sha256 `3e1aea77751a6a1286cd0bc525163ac6e90da2f81fd19fff1fc23ae247133267`; context `dab85ceb9e4acf145e434179d866848912a0831bbf827ea196f82c28f855a869`; target `954d51e35b31d2a4ed89041da5a329e20035d3ae2db7f673189a2047751146ea`.
- Translator: i18n-translator subagent, session 398cdb26808f (English drafted by coder session b0300e2c2c7f), anthropic/claude-opus-5-5, 2026-11-14.
- Auditor: i18n-auditor, independent `task` subagent (C4 i18n-auditor es fr ru), session 5e5b8d070b27; did not translate or edit any value it judged; anthropic/claude-opus-5-5 (served model reported by the session; no provider fallback observed). Same model family as the translator (disclosed).

## Findings

### ES-H1 — minor
- Keys: cli.mobile_help.logs_follow
- Issue: Pre-existing value (outside the 2 new keys, unflagged by w2a3/w2a4): «(seguir el descriptor se pierde ambos)» is ungrammatical — the reflexive «se pierde» cannot take the plural direct object «ambos». Meaning recoverable.
- Suggested fix: «(si se sigue el descriptor, se pierden ambos)» or «(el seguimiento por descriptor pierde ambos)».


## Translator questions

- Q4 (es «acceso telefónico» vs «acceso desde el teléfono»): minor, recorded as ES-D1 (follow-up wave, not a gate blocker; mobile.setup already passed with the same term).
- Q5 (ru/es daemon.state_not_running fragment): nit ES-D3; acceptable.
- Q6 («asleep» renderings): «en reposo» natural for a sleeping device; not verified against the phone app's own label (separate repo).

## Per-key judgement

| key | English | es | judgement |
|---|---|---|---|
| cli.mobile_help.mobile | Phone access: control this machine's lop sessions | Acceso desde el teléfono: controlar las sesiones de lop de esta máquina | ok |
| cli.mobile_help.install | Install the daemon, password and LaunchAgent | Instalar el daemon, la contraseña y el LaunchAgent | ok |
| cli.mobile_help.status | Daemon health, gate and live sessions | Estado del daemon, control de acceso y sesiones activas | ok |
| cli.mobile_help.daemon_action | {action, select, start{Start} stop{Stop} restart{Restart} other{{action}}} the daemon | {action, select, start{Iniciar} stop{Detener} restart{Reiniciar} other{{action}}} el daemon | ok |
| cli.mobile_help.logs | Tail the daemon's and the runtimes' logs | Mostrar el final de los registros del daemon y de los runtimes | ok |
| cli.mobile_help.logs_lines | Lines per file (up to two are read) | Líneas por archivo (se leen hasta dos) | ok |
| cli.mobile_help.logs_follow | Follow by name, so a log created or rotated while you watch is seen (a descriptor-bound follow misses both) | Seguir por nombre para ver los registros creados o rotados mientras miras (seguir el descriptor se pierde ambos) | ES-H1 |
| cli.mobile_help.password | Show or rotate the portal password | Mostrar o rotar la contraseña del portal | ok |
| cli.mobile_help.devices | List this computer's push devices, or revoke/unrevoke one | Listar los dispositivos push de esta máquina o revocar/anular la revocación de uno | ok |
| cli.mobile_help.devices_command | list (default), revoke <device_id>, or unrevoke <device_id> | list (predeterminado), revoke <device_id> o unrevoke <device_id> | ok |
| cli.mobile_help.devices_device_id | the device_id `lop mobile devices` lists; required by revoke/unrevoke | El device_id que muestra `lop mobile devices`; obligatorio para revoke/unrevoke | ok |
| cli.mobile_help.devices_port | the mobile daemon's port (default: the daemon's own default) | Puerto del daemon móvil (por defecto, el del propio daemon) | ok |
| cli.mobile_help.uninstall | Remove the LaunchAgent | Eliminar el LaunchAgent | ok |
| cli.mobile_help.uninstall_purge | Also delete the password | Eliminar también la contraseña | ok |
| cli.mobile_help.serve | Run the daemon (foreground) | Ejecutar el daemon (en primer plano) | ok |
| cli.mobile_help.doctor | Read-only diagnostics for phone access (safe to share) | Diagnóstico de solo lectura del acceso desde el teléfono (seguro para compartir) | ok |
| cli.mobile_help.doctor_json | Print the diagnostics as JSON | Mostrar el diagnóstico en formato JSON | ok |

## Mechanical checks

- PASS — key parity (same set and order)
- PASS — ICU argument names identical
- PASS — select branch labels identical
- PASS — no command named that English does not name
- PASS — final period mirrors English
- PASS — rendered through Msg.envelope(locale=es) without error (17/17 keys present in render)
- PASS — no silent English fallback in render

## Not verified

- Native-speaker review; terminal/desktop frames (text render via the runtime only, see rendered.txt); the phone app's own labels (separate repo).
