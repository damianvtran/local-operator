# tui.sound es delta audit — round 2

Reviewer: C4-i18n-delta / openai/gpt-6.1-sol / session 304e031dccf4 (no fallback).
Scope: 27695a49855f..bd9ad6f041ca.

Seven changed/added keys independently compared against exact English/context;
22 previous audited values retained byte-identically, with bound prior evidence.
`default` removed. Commands and placeholders preserved, including `[--select]`
and ASCII `0-100`. Heading/hint budgets pass; `id_help` intentionally longer than
96 cells, preview-on-arrow fact first per sidecar. Duration is 1.5 seconds, never
1.5 notifications per second. Changes and semantic corrections: record.json.

Prior translators/auditors: c4r1/record.json. Current independent translator:
session 1d36d618f249, openai/gpt-6.1-sol, no fallback. Full catalogue target hash
and current English/context/glossary/style hashes: record.json.

Runtime rendering: 29/29 messages; no plural messages, exact placeholder and
technical literal parity, NFC, no manually inserted bidi controls. Broken
placeholder negative control rejected; absent-key fallback returned English.
Synthetic real OperatorApp frames at 60 columns are linguistic paint evidence,
not live terminal shaping, settings-interaction review or native audio proof.
Urdu retains its documented degraded TUI stance. Full shared checks and bounds:
i18n/audits/tui-sound-c4r2/audit-evidence.json.

Verdict: passed / terminal for these corrected artifact bytes; no unresolved
BLOCKER/MAJOR. Actual branch integration and its checks remain required.
