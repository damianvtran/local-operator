# Audit — es / cli.teams_help / w2a1

- **Verdict: passed** — 0 blocker, 0 major, 1 minor, 0 nit
- Revision: `088699bfea0fc33d6b2f0748037dd7e89d85627d`; target **untracked**; on-disk sha256 `412c8119912d7672b4b7cdc7c3c97161b4dbd5d5c59db92a07d64ce284d37408`
- Keys read: **34** en / 34 es / 34 context
- Auditor: independent translation-auditor, session `d2377f5f44ec`; model **kimi/k3.1-pro** (self-reported temporary fallback; primary anthropic/claude-sonnet-5-5).

## Verified, not assumed

All 34 keys were read against the English source and context. The real `local_operator.i18n.runtime` parser checked arguments, plural selectors and AST select-label shapes; flags, backtick code spans and printf tokens were compared exactly. No maxCells is declared. The complete output is in `audit.log`; the reproducible probe is `checks.py.txt`.

All 34 keys read. Roster -> plantilla and kept-English briefs preserve meaning; the three prose uses of tenant are an unrecorded borrowing. All flags, choice-like tokens, examples and exit codes match.

## Findings

### es-w2-A02 — minor — cli.teams_help.push_org, cli.teams_help.pull_org, cli.teams_help.link_org

Current text starts each value with English 'Tenant de la organización...'. This is not a product surface name or in the es guide's kept-English technical list; it reads as an unnecessary borrowing beside the already Spanish 'organización'. The examples' <tenant_id> tokens correctly remain unchanged.

**Proposed remediation:** Use 'Identificador de la organización...' (or record 'tenant' explicitly in the es style guide if the product intentionally keeps it). Do not change <tenant_id>.

## Reader limitation and scope

No value displayed as a redaction marker. The sanctioned reader was attempted and refused because this delegated workspace is not recognized as a git checkout; it also refuses untracked working-tree files and context sidecars. No workaround was used; this limitation is recorded.

No catalogue or programme file was repaired. Only evidence was written. No native review or rendered visual evidence is claimed.
