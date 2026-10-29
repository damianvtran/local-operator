# vi wire.settings audit
Verdict: FINDINGS (2 distinct major issues, 3 findings; no blocker).
Counts: blocker 0, major 3 (F1; F2+F2b same root cause), minor 5 (F3-F7), nit 5 (N1-N5).
keys_read: 820/820. Model: anthropic/claude-sonnet-5-5 (fallback from deepseek-flash, quota 402).
Placeholders/ICU, numbers, units, defaults, negations, NFC, orthography, kept-English tokens and maxCells budgets all verified clean.
Not verifiable: native-speaker naturalness (F2, F4, N5 are low/medium confidence); whether the literal 'N more' in supplements_maxfeatured.help matches the rendered UI string (es/fr/ru translated it, vi/zh kept English).
