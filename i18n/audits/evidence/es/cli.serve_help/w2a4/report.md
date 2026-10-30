# Spanish W2 round-2 delta audit — cli.serve_help — w2a4

**Verdict: PASSED.** 5 final keys mechanically checked; 1 remediated delta key(s) semantically reviewed. No blocker, major, minor or nit remains open.

- Supersedes: `w2a3`; scope: delta only: cli.serve_help.listener_fd.
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d` (`feat/i18n-es-wire-runtime`).
- Final untracked target sha256: `eb2006e610f131f78101d1a8a7ccebbd7543bd536012cc9acc900d3002bb0e75`.
- English sha256: `bd0b51271edc67c23978ca23688a9af1ef94a5623b0e86a5e3bbb418a5b243f0`; context sha256: `d569b1252c7965cc07456c6dba086187237c609a6d9f3021a206a07bf731c770`.
- Auditor/model: kimi/k3.1-pro (self-reported from the current session model-switch notice); fallback disclosed in record.json.

## Round-1 disposition

### es-w2-C04 — CLOSED (round-1 major)

- Keys: cli.serve_help.listener_fd.
- Final `cli.serve_help.listener_fd`: Adoptar un socket de escucha ya asociado en lugar de asociar uno. Lo establece un daemon que reemplaza su propia imagen de proceso con una nueva compilación (`server/reload`), para que el puerto nunca quede sin servicio. No es un flag para el usuario.
- Measurement: The listener description says “Lo establece un daemon”, retains `server/reload`, and preserves the internal, not-user-facing distinction.

## Mechanical and unchanged-file checks

- Final parser/token probe exited 0: ordered keys, ICU arguments and selectors, select labels, printf, flags, command/code tokens, angle tokens, markup and authored newlines passed.
- C04 closed: daemon is kept English and the listener/reload behaviour remains accurate.
- VERIFIED by direct SHA-256 comparison against each round-1 record’s input_hashes.target.sha256: 21/21 match — cli.aida_help, cli.browser_help, cli.config_help, cli.credential_help, cli.exec_help, cli.exec_runtime, cli.exec_worker_help, cli.hub_shared_help, cli.login_help, cli.login_status_help, cli.logout_help, cli.mcp_help, cli.memory_help, cli.model_help, cli.monitor_help, cli.qwencloud_help, cli.refresh_help, cli.resume_click_help, cli.send_help, cli.stop_help, cli.wake_help. cli.exec_remote_help is not part of W2.
- No catalogue, ledger, allowance, generated file or git state was modified. No PR comment was posted.

The full hashes, exact parser outputs and limitations are in `record.json` and `audit.log`.
