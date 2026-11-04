# es wire.settings audit
Verdict: **PASS** (no blocker, no major).
Counts: blocker 0, major 0, minor 16, nit 5.
Keys read: 820/820. Auditor: anthropic/claude-sonnet-5-5 (fallback from deepseek/deepseek-flash).
Checks: placeholders {name}, ICU quoting, numbers, backticks, slash/lop tokens, glossary terms (sesión, despertar, agente/subagente, herramienta, transcripción, aprobaciones, programación, coste, trabajo, habilidad, dispositivo, relé, compactación, monitorización) all conform; tú register consistent; no plural messages in this namespace; sidecar maxCells all met except remote_compute_enabled.choice.true.description (31 vs ~20) and session_compression_enabled.help (149 vs ~100).
Mechanical lead 'space before :' on network.description is a false positive (colon follows a backtick span with no space). Straight apostrophe leads are config-token quoting also present in en.
Not verifiable: actual rendered cell widths (used string length; accents single-cell, no CJK), whether speech voice enum labels are display-only or literal config values (finding F1 hinges on it).
Findings file: es.wire.settings.findings.json (top minor: F1 untranslated voice enum labels, F2 label overflow >29 cells, F3 remote_compute true-description length).
