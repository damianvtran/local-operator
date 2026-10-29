# Spanish W2 round-2 delta audit — cli.update_help — w2a4

**Verdict: PASSED.** 6 final keys mechanically checked; 2 remediated delta key(s) semantically reviewed. No blocker, major, minor or nit remains open.

- Supersedes: `w2a3`; scope: delta only: cli.update_help.from_snapshot, cli.update_help.no_services.
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d` (`feat/i18n-es-wire-runtime`).
- Final untracked target sha256: `6cfd182880c92fb3be703b3ad22139078e5fe857f09de7398358e14260d084d2`.
- English sha256: `78da39b393ee77900255ddfdf92a17220a6eedbecba340e7d925a2fbdf01abf9`; context sha256: `46dfc07c168a78805b437a4561dc0a8f47b005a48176b424b5aca47373cdaa5f`.
- Auditor/model: kimi/k3.1-pro (self-reported from the current session model-switch notice); fallback disclosed in record.json.

## Round-1 disposition

### es-w2-C06 — CLOSED (round-1 major)

- Keys: cli.update_help.no_services.
- Final `cli.update_help.no_services`: No pasar los daemons `lop serve` a la nueva compilación; los daemons supervisados sí se reparan. Usar solo cuando algo más vaya a iniciar esos daemons, p. ej., un script que controle su lanzamiento.
- Measurement: The no-services string now consistently uses daemons while preserving the distinction between moving `lop serve` daemons and still repairing supervised daemons.

### es-w2-C07 — CLOSED (round-1 minor)

- Keys: cli.update_help.from_snapshot.
- Final `cli.update_help.from_snapshot`: Instalar un árbol de código fuente local (un directorio o una ref de git del repositorio actual) en su propia generación de instalación. En macOS, una ref que coincide exactamente con una versión publicada instala el wheel publicado de PyPI — la única compilación que incluye el agente de claves de macOS —; cualquier otra ref se compila desde el código fuente.
- Measurement: Both em dashes now have surrounding spaces: “PyPI — la única … macOS —; cualquier”. The published-wheel/key-agent exception remains accurate.

## Mechanical and unchanged-file checks

- Final parser/token probe exited 0: ordered keys, ICU arguments and selectors, select labels, printf, flags, command/code tokens, angle tokens, markup and authored newlines passed.
- C06/C07 closed: daemon terminology and spaced-em-dash typography now satisfy the recorded decisions.
- VERIFIED by direct SHA-256 comparison against each round-1 record’s input_hashes.target.sha256: 21/21 match — cli.aida_help, cli.browser_help, cli.config_help, cli.credential_help, cli.exec_help, cli.exec_runtime, cli.exec_worker_help, cli.hub_shared_help, cli.login_help, cli.login_status_help, cli.logout_help, cli.mcp_help, cli.memory_help, cli.model_help, cli.monitor_help, cli.qwencloud_help, cli.refresh_help, cli.resume_click_help, cli.send_help, cli.stop_help, cli.wake_help. cli.exec_remote_help is not part of W2.
- No catalogue, ledger, allowance, generated file or git state was modified. No PR comment was posted.

The full hashes, exact parser outputs and limitations are in `record.json` and `audit.log`.
