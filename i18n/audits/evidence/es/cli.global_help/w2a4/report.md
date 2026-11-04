# Spanish W2 round-2 delta audit — cli.global_help — w2a4

**Verdict: PASSED.** 17 final keys mechanically checked; 1 remediated delta key(s) semantically reviewed. No blocker, major, minor or nit remains open.

- Supersedes: `w2a1`; scope: delta only: cli.global_help.no_tui.
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d` (`feat/i18n-es-wire-runtime`).
- Final untracked target sha256: `62c0da2c8cb267d2177d20936661388f507f50fd4f4688adae90a6b8c9eee0cd`.
- English sha256: `c6e91dda89a9b792587f4ac9c594e188b78b8cd529fa1acf51ca783422c3152f`; context sha256: `27388bf24447053aa71a26a5457827a419daf7046d3169101d28eda893b358c7`.
- Auditor/model: kimi/k3.1-pro (self-reported from the current session model-switch notice); fallback disclosed in record.json.

## Round-1 disposition

### es-w2-A06 — CLOSED (round-1 major)

- Keys: cli.global_help.no_tui.
- Final `cli.global_help.no_tui`: Desactivar la TUI a pantalla completa; usar en su lugar la REPL básica en modo headless
- Measurement: Both surfaces now say “en modo headless”: “REPL básica en modo headless” and “Cada sesión se reabre en modo headless”. Neither says “sin interfaz”; this agrees with shipped es “arranques headless del REPL”.

## Mechanical and unchanged-file checks

- Final parser/token probe exited 0: ordered keys, ICU arguments and selectors, select labels, printf, flags, command/code tokens, angle tokens, markup and authored newlines passed.
- The global_help half of cross-namespace A06 is closed: no_tui keeps TUI/REPL and precisely says “en modo headless”, not “sin interfaz”.
- VERIFIED by direct SHA-256 comparison against each round-1 record’s input_hashes.target.sha256: 21/21 match — cli.aida_help, cli.browser_help, cli.config_help, cli.credential_help, cli.exec_help, cli.exec_runtime, cli.exec_worker_help, cli.hub_shared_help, cli.login_help, cli.login_status_help, cli.logout_help, cli.mcp_help, cli.memory_help, cli.model_help, cli.monitor_help, cli.qwencloud_help, cli.refresh_help, cli.resume_click_help, cli.send_help, cli.stop_help, cli.wake_help. cli.exec_remote_help is not part of W2.
- No catalogue, ledger, allowance, generated file or git state was modified. No PR comment was posted.

The full hashes, exact parser outputs and limitations are in `record.json` and `audit.log`.
