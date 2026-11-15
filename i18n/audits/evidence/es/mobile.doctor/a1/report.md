# es/mobile.doctor — independent i18n audit a1

**Verdict: PASSED.** blocker 0 · major 0 · minor 1 · nit 2; unresolved blocker/major: 0.

- Scope: complete NEW namespace, all 23 keys judged semantically.
- Revision: `a3722be7dadacaf140a2587a0100bd3b26c2ab50` (feat/mobile-doctor, worktree mobile-doctor, node lop-remote-node-2).
- English sha256 `0ca00b276fdea066228394e3fee336042b1e149c9e7df3ea31be11a24f0beb72`; context `c6baa99bc539ef2d2e03bf7b6b11e83edac37139eede8db82034f5fad98d4614`; target `f641372869137bcb1fa22d4134707c185ca1cb2cbabfe284939a960a688f3c89`.
- Translator: i18n-translator subagent, session 398cdb26808f (English drafted by coder session b0300e2c2c7f), anthropic/claude-opus-5-5, 2026-11-14.
- Auditor: i18n-auditor, independent `task` subagent (C4 i18n-auditor es fr ru), session 5e5b8d070b27; did not translate or edit any value it judged; anthropic/claude-opus-5-5 (served model reported by the session; no provider fallback observed). Same model family as the translator (disclosed).

## Findings

### ES-D1 — minor
- Keys: mobile.doctor.daemon.not_installed, mobile.doctor.daemon.not_running, mobile.doctor.daemon.ok, mobile.doctor.daemon.slow, mobile.doctor.daemon.gate_open, mobile.doctor.files.ok, mobile.doctor.files.near, mobile.doctor.files.unknown, mobile.doctor.logs.errors, mobile.doctor.tunnel.not_configured, mobile.doctor.files.not_running, mobile.doctor.daemon.not_answering
- Issue: Translator Q4: «acceso telefónico» is also the established Spanish UI term for dial-up access («conexión de acceso telefónico»), so «servicio de acceso telefónico» can read as a dial-up service. Meaning survives in context (phone app, Radient tunnel) and the wording matches the already-passed es mobile.setup namespace (24 keys), so changing only this namespace would split the term; cli.mobile_help meanwhile uses «acceso desde el teléfono».
- Suggested fix: Optional follow-up wave across mobile.setup + mobile.doctor: «acceso desde el teléfono» (servicio → «el servicio para el teléfono» or «el servicio de acceso desde el teléfono»). Not required for this gate.

### ES-D2 — nit
- Keys: mobile.doctor.daemon.not_running
- Issue: «muestra esta máquina en reposo hasta que se inicie»: «se inicie» can attach to «máquina» (suggesting a reboot); «como» is more idiomatic after «mostrar». The English «until it starts» carries the same latent ambiguity.
- Suggested fix: «…la aplicación móvil muestra esta máquina como en reposo hasta que el servicio se inicie.»

### ES-D3 — nit
- Keys: mobile.doctor.daemon.state_not_running
- Issue: Translator Q5: rendered after the English label «auth gate:» in `lop mobile status`; «no está en ejecución» is a predicate rather than a state label. Grammatical and correct in meaning.
- Suggested fix: Optional: «detenido» or «sin ejecutar».


## Translator questions

- Q4 (es «acceso telefónico» vs «acceso desde el teléfono»): minor, recorded as ES-D1 (follow-up wave, not a gate blocker; mobile.setup already passed with the same term).
- Q5 (ru/es daemon.state_not_running fragment): nit ES-D3; acceptable.
- Q6 («asleep» renderings): «en reposo» natural for a sleeping device; not verified against the phone app's own label (separate repo).

## Per-key judgement

| key | English | es | judgement |
|---|---|---|---|
| mobile.doctor.versions.ok | Local Operator {lop} on Python {python} ({platform}). | Local Operator {lop} en Python {python} ({platform}). | ok |
| mobile.doctor.daemon.not_installed | The phone service is not set up on this computer, so it is not running. | El servicio de acceso telefónico no está configurado en esta máquina, por lo que no está en ejecución. | ES-D1 |
| mobile.doctor.daemon.not_running | The phone service is set up but not running; the phone app shows this computer as asleep until it starts. | El servicio de acceso telefónico está configurado, pero no está en ejecución; la aplicación móvil muestra esta máquina en reposo hasta que se inicie. | ES-D1,ES-D2 |
| mobile.doctor.daemon.ok | The phone service is running and answered in {ms} ms. | El servicio de acceso telefónico está en ejecución y respondió en {ms} ms. | ES-D1 |
| mobile.doctor.daemon.slow | The phone service is running but took {ms} ms to answer, which the phone app may show as not answering. | El servicio de acceso telefónico está en ejecución, pero tardó {ms} ms en responder; la aplicación móvil puede mostrar que esta máquina no responde. | ES-D1 |
| mobile.doctor.daemon.gate_open | The phone service answered without asking for its password: the phone portal is unprotected. | El servicio de acceso telefónico respondió sin pedir su contraseña: el portal del teléfono no está protegido. | ES-D1 |
| mobile.doctor.daemon.state_not_running | not running | no está en ejecución | ES-D3 |
| mobile.doctor.files.ok | The phone service has {open} of {limit} files open. | El servicio de acceso telefónico tiene {open} de {limit} archivos abiertos. | ES-D1 |
| mobile.doctor.files.near | The phone service has {open} of {limit} files open, close to its limit, so new phone connections and database opens can fail. | El servicio de acceso telefónico tiene {open} de {limit} archivos abiertos, cerca de su límite, así que pueden fallar nuevas conexiones del teléfono y aperturas de la base de datos. | ES-D1 |
| mobile.doctor.files.unknown | The phone service's open files could not be counted on this computer. | No se pudieron contar los archivos abiertos del servicio de acceso telefónico en esta máquina. | ES-D1 |
| mobile.doctor.files.status_line | {open} open of {limit} | {open} abiertos de {limit} | ok |
| mobile.doctor.logs.clean | No out-of-files or database-open errors in the last day of logs. | No hay errores de “demasiados archivos abiertos” ni de apertura de la base de datos en los registros del último día. | ok |
| mobile.doctor.logs.errors | The last day of logs shows out-of-files errors ({emfile}) and database-open errors ({sqlite}); the phone service may be running out of files. | Los registros del último día muestran errores de “demasiados archivos abiertos” ({emfile}) y de apertura de la base de datos ({sqlite}); puede que el servicio de acceso telefónico esté llegando a su límite de archivos abiertos. | ES-D1 |
| mobile.doctor.tunnel.not_configured | Phone access through Radient is not set up on this computer. | El acceso telefónico a través de Radient no está configurado en esta máquina. | ES-D1 |
| mobile.doctor.tunnel.ok | The tunnel is set up and its connector is {state}. | El túnel está configurado y su conector está en estado {state}. | ok |
| mobile.doctor.tunnel.problem | The tunnel is set up but its connector is {state} ({reason}). | El túnel está configurado, pero su conector está en estado {state} ({reason}). | ok |
| mobile.doctor.tunnel.single | This Radient account has one tunnel for this computer. | Esta cuenta de Radient tiene un solo túnel para esta máquina. | ok |
| mobile.doctor.tunnel.duplicates | This Radient account has {count} tunnels for this computer; the phone app may be connecting to an old one. | Esta cuenta de Radient tiene {count} túneles para esta máquina; puede que la aplicación móvil se esté conectando a uno antiguo. | ok |
| mobile.doctor.tunnel.list_unchecked | The account's tunnel list could not be read, so duplicate tunnels were not checked. | No se pudo leer la lista de túneles de la cuenta, así que no se comprobó si hay túneles duplicados. | ok |
| mobile.doctor.login.ok | Signed in to Radient. | Sesión iniciada en Radient. | ok |
| mobile.doctor.login.missing | Not signed in to Radient. | No has iniciado sesión en Radient. | ok |
| mobile.doctor.files.not_running | The phone service is not running, so it has no open files to count. | El servicio de acceso telefónico no está en ejecución, así que no tiene archivos abiertos que contar. | ES-D1 |
| mobile.doctor.daemon.not_answering | The phone service is running but not answering; the phone app shows this computer as not answering. It may have run out of files. | El servicio de acceso telefónico está en ejecución, pero no responde; la aplicación móvil muestra que esta máquina no responde. Puede que haya llegado a su límite de archivos abiertos. | ES-D1 |

## Mechanical checks

- PASS — key parity (same set and order)
- PASS — ICU argument names identical
- PASS — select branch labels identical
- PASS — no command named that English does not name
- PASS — final period mirrors English
- PASS — rendered through Msg.envelope(locale=es) without error (23/23 keys present in render)
- PASS — no silent English fallback in render

## Not verified

- Native-speaker review; terminal/desktop frames (text render via the runtime only, see rendered.txt); the phone app's own labels (separate repo).
