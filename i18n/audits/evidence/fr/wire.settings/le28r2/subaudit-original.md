# fr wire.settings audit

Verdict: **FINDINGS** (1 major issue spanning 2 keys; no blocker).
Counts: blocker 0 · major 2 (F1/F2, same root cause) · minor 5 · nit 5.
Model: anthropic/claude-sonnet-5-5 (fallback; the job began on deepseek/deepseek-flash).
keys_read: 820/820.

Major: `local_providers.base_url.label` ("endpoint de {name}") and `local_providers.models.label` ("surcharges de modèle de {name}") — `{name}` is Ollama / OpenAI-compatible / LM Studio / vLLM / llama.cpp (providers/local.py LOCAL_PRESETS), so "de Ollama" and "de OpenAI-compatible" lack the required elision. Fix: "endpoint {name}" / "surcharges de modèle pour {name}" (no "de" before the dynamic name).

Minor: U+0020 instead of U+202F before ; : in 4 librarian keys; double space in remote_compute.description; session_compression_enabled.help stilted and 157 chars vs ~100-cell budget; Librarian gender inconsistent (masculine heading, "elle" in a help); scratchpad vs "zone de brouillon".

Not verified / notes: native-speaker idiom calls on nits are judgement; the "~100-cell help budget" has no maxCells in the sidecar so F5 is a risk, not a measured breach. Kept-English enum labels (female/male/high/warm/…) match es/ru/vi/zh-CN, so treated as cross-locale policy and not reported. Mechanical lead classes (straight quotes inside ICU-quoted examples, quotes around config tokens that en also has) were verified as false positives.
