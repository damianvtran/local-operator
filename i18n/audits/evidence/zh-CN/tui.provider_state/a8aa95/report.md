# Independent A — zh-CN / tui.provider_state

Date: 2026-11-22. Source revision: a54ec02de48a94b68d6c3dc92e1307c97ea776ad.
Auditor: i18n-auditor, session 8aa95e755693, actual openai/gpt-6.1-sol after quota fallback from anthropic/claude-opus-5-5 (host model-switch notice).
Independent of translator 298f6fe9722a and English author 93b7b02899af. Translator model is openai/gpt-6.1-sol, verified from committed zh22author/record.json, not guessed. Distinct session, not distinct model. No MT/service/API used.

Verdict: **passed**. All FIVE keys audited. 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT. Findings: none. Independent B remains a separate required actor.

## Native reading, in source order

| Key suffix | Current text | Reasoned judgment |
|---|---|---|
| logged_in | 已登录 | Natural completed authentication status, distinct from required login. Matches stored-login source meaning without adding a credential-validity/health promise. |
| env_key | 环境密钥 | Compact developer-register credential-source label. 环境 is the conventional terse abbreviation for environment in this explicitly contextual source label; 密钥 means a key, not an environment name or configuration command. The source likewise says “env key”, not a literal variable identifier. No glossary requires a different rendering here. |
| needs_login | 需登录 | Native terse requirement status; not mistaken for 已登录 or an imperative button. No 请 necessary for a bare state. |
| local_ready | 服务器已配置 | Completed configuration state, not “server online”, “connected” or “healthy”. 服务器 follows the style guide and 已配置 preserves the critical completed-state distinction. |
| local_unconfigured | 请配置服务器 | Natural next-step directive; 请 is appropriate because this is instruction copy, not a clickable button or bare state. The context explicitly marks it as an action. Distinct from completed 服务器已配置. |

Read style guide and glossary before English/context and then target. Reconsidered the two potentially ambiguous values against their full context: 环境密钥 is a human source label, not a protected identifier; 请配置服务器 is instruction rather than a button, so the guide’s directive 请 rule applies, not the no-button rule. Short labels correctly omit sentence-final punctuation. No product names, commands, identifiers, tags or paths occur in these five values.

## Mechanical execution

Executed the retained script.py.txt using primary checkout .venv with PYTHONPATH pointing to this worktree. Actual audit.log proves imported module/checker locations, isolated HOME/config, stripped CMUX_/LOP_ environment, before/after hashes, official scoped catalogue/context checks, exact ordered key parity, zero ICU arguments, zero plural/selects, NFC and protected-token checks, 10 English/Chinese official render/provenance cases, same-code Rich width controls, and expanded synthetic pseudo cases. Adapted the committed author’s mechanical rig, but ran it afresh independently and added in-memory missing-key and added-argument counterfactuals; both were rejected by the official checker. No target writes, generation, global checker, full suites or app startup.

## Fit boundary — diagnostics, not native UI approval

Chinese cell widths in source order: 6, 8, 6, 12, 12. No maxCells/maxChars metadata exists. All fit the observed 17-cell example (31-cell pane beside openrouter); that is NOT a global cap. Also tested a longer synthetic opaque ID with narrower segment, retaining the truncation diagnostics. English runs use exactly the same fitting code. Truncation under a sufficiently narrow segment is expected in this consumer formula and does not establish a translation defect. Rich is not native CJK font evidence or actual Settings geometry. The typed-row adapter is not yet implemented, so these are copy-only judgments; no localized live Settings, native, shipped, whole-UI, release-eligibility or certified network-offline claim.

## Evidence semantics and limits

This report is audit_notes_not_execution_log; audit.log contains actual command output. script.py.txt is the exact executed program retained inertly. The committed author receipt is a provenance source only, not borrowed execution proof. A preliminary read_public_copy request for that metadata was rejected as outside its copy allowlist; metadata was then read using ordinary read, while English, context and target product copy were all read with the approved public-copy reader. No target-reader refusal was bypassed. No CLI history apply, staging, commit or push is authorized. The supplied ledger receipt is append-ready and dry-run validated only.
