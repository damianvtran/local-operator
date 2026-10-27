# Spanish audit — wire.errors.notifications

Auditor: nt-es-37d9c9f0df44 / openai/gpt-6.1-sol; independent of L1-core-notify coder.

Verdict: clean; 4/4 keys read; 0 blocker, 0 major, 0 minor.

All keys and context compared; ICU argument integrity, intended actions, neutral tú and literal command/page-name constraints preserved. No findings.

All four error messages preserve actionable causes, no-change semantics, retry guidance and both exact {where} placeholders. Unknown-item destination remains Notificaciones, never Inbox. No plurals occur.

## Actual evidence
`node $LOCAL_OPERATOR_SCRATCHPAD/audit-es.mjs`; see audit.log for exact measured hashes, ICU parse/argument parity and negative-control outputs. English source SHA matches the delegation pin. Sidecar read in full. No rendered/native or release sign-off.

Actual model source: latest session model-switch notice after quota fallback; neither requested Opus nor intermediate Sonnet performed this audit.
