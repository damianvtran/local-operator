# Audit — es / cli.sessions_help / w2a1

- **Verdict: failed** — 2 blocker, 1 major, 0 minor, 0 nit
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d`; target **untracked**; on-disk sha256 `27f21f00b13e15320ce3f989c7cca51738bf0951600a296dd5840a1e6376dd0b`
- Keys read: **47** en / 47 es / 47 context
- Auditor: independent translation-auditor, session `d2377f5f44ec`; model **kimi/k3.1-pro** (self-reported temporary fallback; primary anthropic/claude-sonnet-5-5).

## Verified, not assumed

All 47 keys were read against the English source and context. The real `local_operator.i18n.runtime` parser checked arguments, plural selectors and AST select-label shapes; flags, backtick code spans and printf tokens were compared exactly. No maxCells is declared. The complete output is in `audit.log`; the reproducible probe is `checks.py.txt`.

All 47 keys read. Two reclaim strings incorrectly promise recovery where the command previews termination or removes scratchpad content; pads also violates the explicit sidecar terminology instruction.

## Findings

### es-w2-A03 — major — cli.sessions_help.cleanup_scratchpads_only

Current text: 'Auditar pads con --dry-run; si no, recuperar solo el contenido de compilación y caché de scratchpad (nunca transcripciones ni tus conversaciones).' The sidecar explicitly says to use settled 'scratchpad', not 'pad'; 'pads' violates that instruction and the shipped es scratchpad terminology.

**Proposed remediation:** Replace 'Auditar pads' with 'Auditar scratchpads' and keep 'scratchpad' as the product noun throughout.

### es-w2-A04 — blocker — cli.sessions_help.cleanup_scratchpads_only

Current 'recuperar solo el contenido de compilación y caché de scratchpad' tells the operator content would be recovered/restored. The implementation removes scratchpad files/trees (session/cleanup.py:1622-1656); --dry-run audits without removing, otherwise this is destructive cleanup, with transcripts/conversations preserved.

**Proposed remediation:** Use 'eliminar solo el contenido de compilación y caché de scratchpad (nunca transcripciones ni tus conversaciones)'.

### es-w2-A05 — blocker — cli.sessions_help.reclaim_dry_run

Current text: 'listar qué se recuperaría, sin enviar señales'. English says 'list what would be reclaimed, without signalling anything'; the namespace's reclaim command ends unreachable runtimes with SIGTERM after draining, not recovering sessions. 'Recuperaría' reverses the dry-run's destructive preview meaning.

**Proposed remediation:** Use 'listar qué runtimes se finalizarían, sin enviar señales'.

## Reader limitation and scope

No value displayed as a redaction marker. The sanctioned reader was attempted and refused because this delegated workspace is not recognized as a git checkout; it also refuses untracked working-tree files and context sidecars. No workaround was used; this limitation is recorded.

No catalogue or programme file was repaired. Only evidence was written. No native review or rendered visual evidence is claimed.
