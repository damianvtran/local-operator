# Spanish W2 round-2 delta audit — cli.services_help — w2a4

**Verdict: PASSED.** 9 final keys mechanically checked; 7 remediated delta key(s) semantically reviewed. No blocker, major, minor or nit remains open.

- Supersedes: `w2a3`; scope: delta only: cli.services_help.pid, cli.services_help.reclaim, cli.services_help.reclaim_description, cli.services_help.services, cli.services_help.services_description, cli.services_help.status_description, cli.services_help.wait.
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d` (`feat/i18n-es-wire-runtime`).
- Final untracked target sha256: `244cfa578f571bd6e5ac5de66a04ecafd32f2f0597fe1d2eb07468940d735f4c`.
- English sha256: `76aca39fb3bfe46be1d31fbfde2176ee525b7ac650961104894a894f71751935`; context sha256: `cc6104f97fbb0faac19232252063cdb7effaa2ffb4bbe2d4b65c2c749a8fe1e1`.
- Auditor/model: kimi/k3.1-pro (self-reported from the current session model-switch notice); fallback disclosed in record.json.

## Round-1 disposition

### es-w2-C05 — CLOSED (round-1 major)

- Keys: cli.services_help.services, cli.services_help.services_description, cli.services_help.status_description, cli.services_help.wait, cli.services_help.reclaim, cli.services_help.reclaim_description, cli.services_help.pid.
- Final `cli.services_help.services`: Pasar a la compilación actual los servicios de esta máquina que no son runtimes (daemons serve y supervisados), sin detener ninguna conversación activa
- Final `cli.services_help.services_description`: Todo lo que local_operator ejecuta en esta máquina y no es una conversación. Un “servicio” es un daemon `lop serve` o un LaunchAgent supervisado: el relé móvil, el puente del navegador, el túnel y el agente de despertares. Los runtimes —los procesos que contienen tus conversaciones— nunca se detienen: “restart” recarga un daemon serve sin cambiar de proceso, conservando su pid y su socket.
- Final `cli.services_help.status_description`: Solo lectura. Indica la compilación de la instalación y, para cada daemon `lop serve`, la que está SIRVIENDO: ambos lados de cada comparación, porque la deriva habitual en esta máquina es una recompilación de la misma versión, y la versión por sí sola no revela que un daemon esté desactualizado.
- Final `cli.services_help.wait`: Cuánto esperar a que un daemon solicitado vuelva con la nueva compilación (predeterminado: {wait_seconds}s; debe ser positivo). Si no vuelve, se informa de ello, sigue sirviendo la compilación que cargó y el próximo `lop services restart` lo reintenta.
- Final `cli.services_help.reclaim`: Terminar un daemon `lop serve` registrado que no esté sirviendo su dirección (nunca uno que la esté sirviendo)
- Final `cli.services_help.reclaim_description`: Pedir a UN daemon `lop serve`, identificado por pid, que termine, escalando de SIGTERM a SIGKILL tras un tiempo límite, después de comprobar que el proceso es el daemon serve de este producto y que no es el que sirve la dirección indicada en su registro. `lop services status` lista los daemons a los que se aplica. Nunca es automático: el daemon puede supervisar runtimes de sesión y quien lee un registro no puede comprobar que un sucesor esté listo para ocupar su lugar.
- Final `cli.services_help.pid`: pid del daemon, exactamente como lo imprime `lop services status`
- Measurement: All seven cited service strings use daemon/daemons, with zero demonio/demonios occurrences. The safety restrictions on serving daemons, runtimes, pid, SIGTERM/SIGKILL and waits remain accurate.

## Mechanical and unchanged-file checks

- Final parser/token probe exited 0: ordered keys, ICU arguments and selectors, select labels, printf, flags, command/code tokens, angle tokens, markup and authored newlines passed.
- C05 closed: daemon/daemons is consistent and destructive-operation safety restrictions survive.
- VERIFIED by direct SHA-256 comparison against each round-1 record’s input_hashes.target.sha256: 21/21 match — cli.aida_help, cli.browser_help, cli.config_help, cli.credential_help, cli.exec_help, cli.exec_runtime, cli.exec_worker_help, cli.hub_shared_help, cli.login_help, cli.login_status_help, cli.logout_help, cli.mcp_help, cli.memory_help, cli.model_help, cli.monitor_help, cli.qwencloud_help, cli.refresh_help, cli.resume_click_help, cli.send_help, cli.stop_help, cli.wake_help. cli.exec_remote_help is not part of W2.
- No catalogue, ledger, allowance, generated file or git state was modified. No PR comment was posted.

The full hashes, exact parser outputs and limitations are in `record.json` and `audit.log`.
