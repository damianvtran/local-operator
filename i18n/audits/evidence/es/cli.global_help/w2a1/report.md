# Audit — es / cli.global_help / w2a1

- **Verdict: failed** — 0 blocker, 1 major, 0 minor, 0 nit
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d`; target **untracked**; on-disk sha256 `4b13208f4837e6402de1509782ce7cf7ab8e02669742949a6c46aefac2f1abf4`
- Keys read: **17** en / 17 es / 17 context
- Auditor: independent translation-auditor, session `d2377f5f44ec`; model **kimi/k3.1-pro** (self-reported temporary fallback; primary anthropic/claude-sonnet-5-5).

## Verified, not assumed

All 17 keys were read against the English source and context. The real `local_operator.i18n.runtime` parser checked arguments, plural selectors and AST select-label shapes; flags, backtick code spans and printf tokens were compared exactly. No maxCells is declared. The complete output is in `audit.log`; the reproducible probe is `checks.py.txt`.

All 17 keys read. Provider/model tokens and root description whitespace survive; auto-approval, legacy storage, TUI and peer capability promises survive. Headless is mistranslated as no interface.

## Findings

### es-w2-A06 — major — cli.global_help.no_tui, cli.sessions_help.resume_description

Current text uses 'REPL básica sin interfaz' for 'plain headless REPL' and 'Cada sesión se reabre sin interfaz' for 'reopened headlessly'. Both help surfaces retain REPL/TUI and command-line execution, so 'sin interfaz' falsely suggests no interface at all; it also drifts from shipped es 'arranques headless del REPL'. The literal 'headless' is not itself a required token.

**Proposed remediation:** Prefer consistent 'REPL básica en modo headless' and 'Cada sesión se reabre en modo headless' (or an equivalent precise 'sin TUI' wording).

## Reader limitation and scope

No value displayed as a redaction marker. The sanctioned reader was attempted and refused because this delegated workspace is not recognized as a git checkout; it also refuses untracked working-tree files and context sidecars. No workaround was used; this limitation is recorded.

No catalogue or programme file was repaired. Only evidence was written. No native review or rendered visual evidence is claimed.
