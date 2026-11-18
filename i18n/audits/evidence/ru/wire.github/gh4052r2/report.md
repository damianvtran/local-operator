# PR4052 primary i18n audit — round 2

Locale: ru; namespace: wire.github. Actual auditor anthropic/claude-opus-5-5, session 91d1e05d2d82; no fallback observed in this resumed round. Historical round-1 Sol provenance remains unchanged.
Scope: 2afbb818b69074d9f11b0fc1a9a6360967218476..b06fbf62a7ccc9e4ae8aac59be70b046c7a5b4ee. Verdict: PASSED; 0 unresolved BLOCKER/MAJOR; no new findings.

Only the nine remediation values and direct neighbors were re-tested; all other scoped 474 values are retained by actual value equality/hashes from round 1. The other 866 settings values per target retain the previously passed baseline approval and byte-equality proof. This is not a full locale re-audit. Current source/context bytes match the original audited hashes. Both independent passes are now clean; second-pass current-byte records are retained in gh4052s2 and original source bytes are preserved in its shared French bundle.

Closed: I-RU-PUSH, I-RU-WRAPPER (deduplicated second-pass I2-RU-01), I-COMPOSE, I2-UR-01, I-FR-01, I-RU-02. Russian three timing messages now name push with next-fetch/TTL unchanged; ru denial names the sharing device with correct agreement and owner-only authority; es/ru wrappers accept all seven reason fragments naturally; Urdu retains ڈیوائس with coherent agreement and lender-only authority; French names list retrieval failure; ru help preserves never-set vs cleared-shared-list semantics and --use-default.

Executed primary delta evidence: 26 candidate/neighbor ICU renders (nine changed values, fourteen assembled es/ru reason/refusal cases, three French failure neighbors); nine AST-based protected-token pseudo wrapping cases; placeholder identity, NFC/control checks and two negative controls; all eight before/after static raw CLI-copy PNGs actually viewed. Original production preactivation control returned English as expected before ledger/generation. No native terminal bidi/shaping certification, browser/native app frame or full-suite claim. Urdu raw static export remains visibly degraded/unshaped; balanced paint-adapter isolates are only an adapter control, not proof that plain CLI adopts it. Subsequent production eligibility results are recorded separately after supported generation.

Retested keys: wire.github.exclude_added, wire.github.include_added, wire.github.opt_out_step, wire.github.share_empty, wire.github.wrapper_denied.
Retained full namespace values: 63; canonical unchanged-values SHA-256: 0be092e616f2c00312f0c63a1e8b60a0e51d8204a9766ab2eb96414d51d87ae9.

## wire.github.exclude_added

Current: {repo} исключён — устройства, которым вы предоставили доступ, потеряют право выполнять push в него при следующем получении учётных данных, {ttl}

Reasoning: matches the accepted round-1 remediation, source meaning, glossary/register and placeholders; no further finding.

## wire.github.include_added

Current: {repo} возвращён — устройства, которым вы предоставили доступ, снова смогут выполнять push в него при следующем получении учётных данных, {ttl}

Reasoning: matches the accepted round-1 remediation, source meaning, glossary/register and placeholders; no further finding.

## wire.github.opt_out_step

Current: исключить репозиторий: {command} — устройства, которым вы предоставили доступ, потеряют право выполнять push в него при следующем получении учётных данных, {ttl}

Reasoning: matches the accepted round-1 remediation, source meaning, glossary/register and placeholders; no further finding.

## wire.github.share_empty

Current: Репозитории GitHub: пока нет — общий доступ настроен, но все запросы заимствования отклоняются по следующей причине: {reason}; {remedy}

Reasoning: matches the accepted round-1 remediation, source meaning, glossary/register and placeholders; no further finding.

## wire.github.wrapper_denied

Current: {repo} отсутствует в списке репозиториев, предоставленном устройством, которое делится доступом (возможно, репозиторий исключён или с этого устройства в него нельзя выполнить push). Проверьте {command} на устройстве, которое делится доступом. Только это устройство может изменить предоставляемый доступ.

Reasoning: matches the accepted round-1 remediation, source meaning, glossary/register and placeholders; no further finding.

## Eligibility and production-path completion

After both independent passes were verified, all 14 current-byte receipts were appended through the supported CLI. Generator restored the exact eight locales (en/fr/es/zh-CN/ru/vi/ur/hi), with no manual shipped edits or orphan deletion. generate.py --check reports 172 byte-identical artifacts; full base-aware checker passes with 99 advisory findings. 51 unmodified production resolver executions passed: nine repaired values through messages.render_record plus active text.tr, 14 generated-Msg reason compositions, 28 restored-locale smokes; unsupported-locale and missing-key fallback controls passed. This proves resolver activation, not live credential minting or native terminal shaping. Shared production.json/logs record actual responses.
