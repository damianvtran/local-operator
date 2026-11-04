# Author-corrected provenance addendum — core #3349 French delta audit

This addendum supersedes only the model-provenance assertions in frca-delta-05c51; it does not replace or modify that immutable bundle, rerun checks, or change its semantic findings or verdict.

The prior author assertion “anthropic/claude-sonnet-5-5; no fallback observed” was based solely on the resumed runtime's requested/configured header. That header does not establish the model that actually served the completed delta audit. I withdraw the no-fallback assertion and the claim that Sonnet was the verified serving model.

Historical host evidence supplied by the manager in this delegation: “host wait completion record reported openai/gpt-6.1-sol”. Accordingly, the corrected audit model is recorded as openai/gpt-6.1-sol, HOST-REPORTED; requested/header model was anthropic/claude-sonnet-5-5. The actual historical per-response usage metadata and fallback sequence could not be independently recovered in this resumed child. The sessions.peek lookup exposed only the current resumed transcript window, not historical usage metadata. Thus GPT is not independently usage-verified here; exact routing/fallback cause is unknown. Do not equate the current header with historical usage. This distinction is part of the correction, not a claim that no fallback occurred.

All source/context/target hashes are copied byte-for-byte from the original verified receipt. All original companion hashes were verified in this provenance-only operation. Original auditor identity remains i18n-auditor / 05c51f9b1aba, independent of translator a58a48732351. Semantic verdict remains passed, 2 keys, zero findings at all severities. No translation reread, validator/native rerun, new translation, ledger/repo edit or push performed. Report is provenance-only and does not reset prior suites or confer rendered proof.

Retain this addendum alongside the prior immutable bundle, without rewriting it. Proposed tracked directory: i18n/audits/evidence/fr/wire.aida_cadence/frca-delta-prov-05c51/.
