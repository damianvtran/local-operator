# Independent B audit — zh-CN / tui.provider_state

Date: 2026-11-22. Audited revision: a54ec02de48a94b68d6c3dc92e1307c97ea776ad.
Auditor: i18n-auditor, session e800bd515a19, actual model openai/gpt-6.1-sol. Host model-switch message disclosed quota fallback from anthropic/claude-opus-5-5. Independent of translator 298f6fe9722a, English author 93b7b02899af, and auditor A 8aa95e755693. A's conclusions were not read. Translator identity was read from zh22author/record.json: i18n-translator, openai/gpt-6.1-sol, quota fallback, not the originally requested Claude model.

## Verdict: passed — all five keys, no findings

These are independent native-fluent LLM judgments, not an MT comparison or human-native certification. Read order: zh-CN guide and glossary, English/context, target. No translation service used.

| Key suffix | Current Chinese | Independent semantic/register/terminology decision | Cells (EN/ZH) |
|---|---|---|---|
| logged_in | 已登录 | Natural completed login status. Reports the stored-login state without adding live reachability or credential-validity guarantees. No pronoun or sentence punctuation needed. | 9/6 |
| env_key | 环境密钥 | Compact technical label for an environment-supplied key, contrasted with stored login. 环境 here is the environment credential source, not an environment-variable identifier to preserve verbatim. 密钥 is appropriate for API key; the glossary's auth-token 令牌 distinction does not turn a key into a token. In this terse provider-state context the compressed wording is acceptable, not a configuration or health claim. | 7/8 |
| needs_login | 需登录 | Idiomatic terse requirement status. 需 preserves the need for authentication, unlike 已登录; not an instruction claiming a login was attempted or failed. | 11/6 |
| local_ready | 服务器已配置 | Natural completed configuration status. 已配置 does not assert server connectivity or probe health. 服务器 conforms to the guide. Context supplies the local-provider scope, as in English. | 17/12 |
| local_unconfigured | 请配置服务器 | Action, not completed status. 请配置 is a natural next-step directive; not a bare state label or named clickable button in this context, so 请 is compatible with the guide's instruction rule. It contrasts unambiguously with 已配置. No added login/network assertion. | 16/12 |

No BLOCKER, MAJOR, MINOR or NIT finding; suggested replacement wording: none. Four-to-eight-character preference is met or improved by a naturally shorter status. No inappropriate 您, particles, exclamation, Latin-spacing, script or punctuation. Machine key IDs remain unchanged; no provider IDs occur in these values. No RTL requirement applies.

## Executed mechanical checks

Exact command and actual stdout/stderr are in audit.log; script.py.txt is the exact executed program. Primary checkout interpreter with PYTHONPATH pointing to this worktree; module paths asserted. Isolated scratch HOME/config and env -i strip CMUX variables; no Git command ran with redirected HOME.

The official scripts.i18n.check.check_catalogues function was called without modification, restricting only catalogue locale/namespace enumeration to en/zh-CN and tui.provider_state. Result []. Its actual missing-key and added-argument counterfactual controls were rejected with the expected diagnostics. Independently checked ordered JSON key/context parity 5/5, zero ICU arguments and zero plural/select structures, NFC, absence of combining/format controls, protected tokens/markup (zero occurrences), exact official runtime rendering for all ten English/Chinese values, and unchanged input hashes. CLDR Chinese other is available; no plural arm is required because these messages contain no plural structure.

20 PlainRich width samples at 17/31 cells include English controls. Ten explicitly synthetic pseudo-expansions exercised the official renderer; this is not an advertised official pseudo-localization workflow. The discovery command initially referred to nonexistent icu.py/pseudo.py and exited 2; this is preserved as a discovery warning, not concealed as a passing execution. The corrected actual scoped validator exited 0.

## Evidence boundaries

17-cell state and 31-cell pane examples are diagnostics, not fixed maxCells. Sidecar has no maxCells/maxChars cap. All Chinese samples are 6–12 cells; narrow consumers may still truncate. PlainRich is not real Settings geometry, a CJK-font rendering inspection, native/global UI validation, or whole-app fit proof. No app startup, network isolation or shipping claim. Original provider-before-offline claim remains **UNVERIFIED**: no app run was performed. Producer still emits English until the separate adapter is implemented.

Only this new B bundle was written. No catalogue/source/generated/eligibility/baseline/history/ledger edits, staging, commit or push. record.json is the provenance/verdict record. artifact-manifest.json is the companion hash manifest; ledger-receipt.json is the distinct append-ready receipt, NOT a ledger append or a claim this uncommitted bundle exists on main. The manager must commit the evidence with the allowed artifact scope and invoke the official optimistic ledger command separately if authorized.

This report is audit_notes_not_execution_log; only audit.log records executed validation output.
