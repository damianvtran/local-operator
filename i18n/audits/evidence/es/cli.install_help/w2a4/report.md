# Spanish W2 round-2 delta audit — cli.install_help — w2a4

**Verdict: PASSED.** 5 final keys mechanically checked; 1 remediated delta key(s) semantically reviewed. No blocker, major, minor or nit remains open.

- Supersedes: `w2a3`; scope: delta only: cli.install_help.prune.
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d` (`feat/i18n-es-wire-runtime`).
- Final untracked target sha256: `02182f7c802873810cd3f3c48673783f53d00705dc279bdd0b23f7c830ea71d2`.
- English sha256: `f242bfdbf8e80bd7f956ae361985180d7eb797dd4223da8423dc678b6fec61e7`; context sha256: `9a9f1aaa7bda951f3b670d85e97b4fa07cbfc1f7b11ca6ee94a5c9b598588f6f`.
- Auditor/model: kimi/k3.1-pro (self-reported from the current session model-switch notice); fallback disclosed in record.json.

## Round-1 disposition

### es-w2-C02 — CLOSED (round-1 blocker)

- Keys: cli.install_help.prune.
- Final `cli.install_help.prune`: Eliminar generaciones de instalación desde las que no se ejecuta nada
- Measurement: “desde las que no se ejecuta nada” exactly restricts deletion to generations nothing is running from; it no longer broadens the rule to “not in use”.

## Mechanical and unchanged-file checks

- Final parser/token probe exited 0: ordered keys, ICU arguments and selectors, select labels, printf, flags, command/code tokens, angle tokens, markup and authored newlines passed.
- C02 closed: prune’s deletion restriction is exactly “generations nothing is running from”, and %(default)s remains byte-identical in keep.
- VERIFIED by direct SHA-256 comparison against each round-1 record’s input_hashes.target.sha256: 21/21 match — cli.aida_help, cli.browser_help, cli.config_help, cli.credential_help, cli.exec_help, cli.exec_runtime, cli.exec_worker_help, cli.hub_shared_help, cli.login_help, cli.login_status_help, cli.logout_help, cli.mcp_help, cli.memory_help, cli.model_help, cli.monitor_help, cli.qwencloud_help, cli.refresh_help, cli.resume_click_help, cli.send_help, cli.stop_help, cli.wake_help. cli.exec_remote_help is not part of W2.
- No catalogue, ledger, allowance, generated file or git state was modified. No PR comment was posted.

The full hashes, exact parser outputs and limitations are in `record.json` and `audit.log`.
