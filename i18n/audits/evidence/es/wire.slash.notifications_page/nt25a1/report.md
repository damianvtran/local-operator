# Spanish audit — wire.slash.notifications_page

Auditor: nt-es-37d9c9f0df44 / openai/gpt-6.1-sol; independent of L1-core-notify coder.

Verdict: clean-with-findings; 1/1 keys read; 0 blocker, 0 major, 1 minor.

All keys and context compared; ICU argument integrity, intended actions, neutral tú and literal command/page-name constraints preserved. One minor cross-surface glossary inconsistency remains.

## T-ES-1 — MINOR
Key: `wire.slash.notifications_page.help`

Current: `Preguntas y novedades; read marca todo como leído`

Proposed: `Consultas/actualizaciones; read marca todo leído`

The same Asks and Updates concepts are named Consultas and Actualizaciones in the UI filter, drawer tabs and existing asks rail; Preguntas/novedades introduces cross-surface term drift. Proposed compact wording retains the literal read command and all-read effect within the measured 50-cell budget.

Current 49/50 cells; repair 48/50 cells. Unresolved; auditor made no fix.

## Actual evidence
`node $LOCAL_OPERATOR_SCRATCHPAD/audit-es.mjs`; see audit.log for exact measured hashes, ICU parse/argument parity and negative-control outputs. English source SHA matches the delegation pin. Sidecar read in full. No rendered/native or release sign-off.

Actual model source: latest session model-switch notice after quota fallback; neither requested Opus nor intermediate Sonnet performed this audit.
