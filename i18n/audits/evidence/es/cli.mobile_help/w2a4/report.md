# Spanish W2 round-2 delta audit — cli.mobile_help — w2a4

**Verdict: PASSED.** 15 final keys mechanically checked; 6 remediated delta key(s) semantically reviewed. No blocker, major, minor or nit remains open.

- Supersedes: `w2a3`; scope: delta only: cli.mobile_help.daemon_action, cli.mobile_help.devices_port, cli.mobile_help.install, cli.mobile_help.logs, cli.mobile_help.serve, cli.mobile_help.status.
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d` (`feat/i18n-es-wire-runtime`).
- Final untracked target sha256: `4613e77bb8a603e582d017869b94fb7f79fd85b21688dc6934ea8f33270350d1`.
- English sha256: `88681bae09adf93216e3e3b990652f7f57654a0438f6bc8c3cc5dd9ebb6eb947`; context sha256: `26054791e1413cfba7d777561c410c8de6f47a76a1d5c3a8081fc8b2077c84e1`.
- Auditor/model: kimi/k3.1-pro (self-reported from the current session model-switch notice); fallback disclosed in record.json.

## Round-1 disposition

### es-w2-C03 — CLOSED (round-1 major)

- Keys: cli.mobile_help.install, cli.mobile_help.status, cli.mobile_help.daemon_action, cli.mobile_help.logs, cli.mobile_help.devices_port, cli.mobile_help.serve.
- Final `cli.mobile_help.install`: Instalar el daemon, la contraseña y el LaunchAgent
- Final `cli.mobile_help.status`: Estado del daemon, control de acceso y sesiones activas
- Final `cli.mobile_help.daemon_action`: {action, select, start{Iniciar} stop{Detener} restart{Reiniciar} other{{action}}} el daemon
- Final `cli.mobile_help.logs`: Mostrar el final de los registros del daemon y de los runtimes
- Final `cli.mobile_help.devices_port`: Puerto del daemon móvil (por defecto, el del propio daemon)
- Final `cli.mobile_help.serve`: Ejecutar el daemon (en primer plano)
- Measurement: All six cited strings now use daemon; the namespace has zero demonio/demonios occurrences. The select labels stay start/stop/restart/other and real parser renders are Iniciar/Detener/Reiniciar el daemon; a synthetic translated-label control falls through to “start el daemon”.

## Mechanical and unchanged-file checks

- Final parser/token probe exited 0: ordered keys, ICU arguments and selectors, select labels, printf, flags, command/code tokens, angle tokens, markup and authored newlines passed.
- C03 closed: daemon terminology is consistent and ICU select machine labels are unchanged; parser negative control proves translated labels would fall through.
- VERIFIED by direct SHA-256 comparison against each round-1 record’s input_hashes.target.sha256: 21/21 match — cli.aida_help, cli.browser_help, cli.config_help, cli.credential_help, cli.exec_help, cli.exec_runtime, cli.exec_worker_help, cli.hub_shared_help, cli.login_help, cli.login_status_help, cli.logout_help, cli.mcp_help, cli.memory_help, cli.model_help, cli.monitor_help, cli.qwencloud_help, cli.refresh_help, cli.resume_click_help, cli.send_help, cli.stop_help, cli.wake_help. cli.exec_remote_help is not part of W2.
- No catalogue, ledger, allowance, generated file or git state was modified. No PR comment was posted.

The full hashes, exact parser outputs and limitations are in `record.json` and `audit.log`.
