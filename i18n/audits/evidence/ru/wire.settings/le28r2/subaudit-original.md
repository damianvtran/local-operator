# ru wire.settings audit
Verdict: **FINDINGS** (no blocker; 4 major glossary violations).
Counts: blocker 0, major 4, minor 10, nit 5.
keys_read: 820/820. Auditor model: anthropic/claude-sonnet-5-5 (fallback after deepseek/deepseek-flash quota failure).
Major: subagents_max_running.label, supplements_maxcostusd.help, supplements_timeouts.help (job→«задание», not «задача»); librarian_enabled.help («сеанс»→«сессия»).
Mechanical parity (placeholders, backticks, ICU braces, maxCells, quotes) clean; no plural branches in this namespace.
Could not verify: no native-speaker/UI render check; width only checked against sidecar maxCells by character count.
