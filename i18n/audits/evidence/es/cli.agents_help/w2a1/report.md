# Audit — es / cli.agents_help / w2a1

- **Verdict: failed** — 0 blocker, 1 major, 0 minor, 0 nit
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d`; target **untracked**; on-disk sha256 `ffe93eb7d0bc050572fcf3296213fc96cfe775f02ba8f8c2d1a7f590678fd122`
- Keys read: **21** en / 21 es / 21 context
- Auditor: independent translation-auditor, session `d2377f5f44ec`; model **kimi/k3.1-pro** (self-reported temporary fallback; primary anthropic/claude-sonnet-5-5).

## Verified, not assumed

All 21 keys were read against the English source and context. The real `local_operator.i18n.runtime` parser checked arguments, plural selectors and AST select-label shapes; flags, backtick code spans and printf tokens were compared exactly. No maxCells is declared. The complete output is in `audit.log`; the reproducible probe is `checks.py.txt`.

All 21 keys read. Push/pull exit codes, destructive scope, defaults and literal tokens survive, but starters -> perfiles iniciales conflicts with shipped plantilla incluida.

## Findings

### es-w2-A01 — major — cli.agents_help.sync, cli.agents_help.sync_all

Wrong current text: sync uses 'Actualizar los perfiles iniciales instalados...' and sync_all uses 'Sincronizar todos los perfiles iniciales instalados...'. English 'starters' is already translated as 'plantilla incluida' in shipped tui.authoring.seed_not_installed, not as 'perfil inicial'. 'Roster' -> 'plantilla' and kept-English 'briefs'/'scratchpad' were checked separately; 'perfiles iniciales' is a distinct cross-surface glossary drift.

**Proposed remediation:** Use 'plantillas iniciales instaladas' in both strings (or 'plantillas incluidas instaladas' for exact parity with tui.authoring); record the starter rendering in the es guide.

## Reader limitation and scope

No value displayed as a redaction marker. The sanctioned reader was attempted and refused because this delegated workspace is not recognized as a git checkout; it also refuses untracked working-tree files and context sidecars. No workaround was used; this limitation is recorded.

No catalogue or programme file was repaired. Only evidence was written. No native review or rendered visual evidence is claimed.
