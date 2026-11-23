# Spanish independent B audit — tui.provider_state

Verdict: **passed**, all five keys independently assessed; zero unresolved BLOCKER/MAJOR and no copy-change findings. This is native-fluent LLM linguistic reasoning, not a human-native-speaker attestation. No translation service, MT comparison or other independent auditor conclusions were used.

## Complete key assessment (audit notes, not execution log)

| Key suffix | Current Spanish | Independent judgment |
|---|---|---|
| logged_in | sesión iniciada | Natural established Spanish authentication status: completed login, not an imperative. Preserves stored-login status versus environment-key source without inventing a live connectivity assertion. |
| env_key | clave del entorno | Natural compact developer-facing credential-source label. In the provider-authentication context, entorno correctly names the environment supplying the key; this is not an actual variable identifier. No need to preserve English env in human prose. |
| needs_login | requiere iniciar sesión | Idiomatic impersonal requirement/status. Unlike iniciar sesión alone it does not collapse the requirement into an action label. No polarity loss. |
| local_ready | servidor configurado | Completed configuration, not servidor conectado/disponible/en funcionamiento. Thus does not overclaim reachability or health, and does not turn configuration into a login state. |
| local_unconfigured | configurar servidor | Standard neutral-Spanish infinitive action/instruction. Clearly distinct from configurado, no invented button affordance or successful-configuration claim. |

All five obey neutral Spanish, terse lowercase fragments, natural noun/participle order, no unnecessary final punctuation, and the glossary's sesión/servidor terminology. No pronoun or regional-register inconsistency. No destructive operation or permission wording is introduced. There are no preserved commands, flags, code spans, identifiers, numeric arguments, tags or plural/select messages inside these five values; keys remain byte-identical and ordered.

## Execution evidence

`audit.log` is actual output of `script.py.txt` executed with the primary checkout interpreter and PYTHONPATH pinned to the assigned worktree, isolated HOME/config, CMUX/LOP prefixes stripped, and XPC_FLAGS removed. It proves official check_catalogues narrowed solely to en/es and this namespace returns [], real render_record uses es (not English fallback), context/key order parity, zero arguments/plural/select, NFC, no protected tokens, and rejects missing-key and added-argument mutations. All five Spanish strings and English controls were passed through ICU rendering and Rich parsing/truncation; synthetic expanded pseudo strings were exercised too. RC 0.

Measured Rich cells in key order: Spanish 15/17/23/20/19; English 9/7/11/17/16. At synthetic segment width 17, three Spanish values truncate while two fit; at 23 all original Spanish values fit. This is a diagnostic, not a declared maxCells violation: the sidecar explicitly has no fixed maximum. The 17-cell example beside openrouter is not a global cap. No width-based copy repair requested; actual styled consumer integration measurement is retained for the integration lane, not claimed here.

## Boundaries and provenance

No application startup, native font rendering, screenshot, global validator/suite, eligibility clearance, code/source/context/translation/history change, staging, commit or push. UI producer integration remains outside this catalogue audit. Ledger append is dry-run only; ledger-receipt.json is the exact CLI input row and ledger-dry-run.log captures actual CLI output/return codes. artifact-manifest.json hashes all companions, including record.json; record.json hashes its nonrecursive companions.

Translator identity comes solely from es22a/record.json: i18n-translator session d37edc1334b7, openai/gpt-6.1-sol after quota fallback. Auditor is distinct session f6084ca8999b. Host initial header advertised anthropic/claude-opus-5-5; the subsequent explicit model-switch notice identifies openai/gpt-6.1-sol due to provider failure: quota. The linguistic judgment and produced artifacts follow that notice. No independent per-response served-model telemetry exists here; this qualification is not a claim that a configured header proves the backend model.
