# Independent zh-CN audit — wire.settings.account_policy

## Verdict

**passed** — 14/14 keys reviewed; no BLOCKER, MAJOR, MINOR or NIT findings. This is reader 1 only, not a claim that both required independent settings audits are complete.

Auditor: i18n-auditor; session e55f54b6a26e; actual completed-usage model openai/gpt-6.1-sol. This is fallback relative to the initial Anthropic header, not an assumed requested model. Translator session 7fce46e29bee is distinct; this auditor did not implement or translate these values and did not read another auditor’s report. Other reader identity was not supplied.

## Scope and authority

All 14 keys in the proposed zh-CN namespace, against English at 0e7cde33ca8008d561e74f4aed774aa5f76ff7d5. The packet’s old translator-input context is retained and hashed separately from the proposed context actually audited. Independently compared both contexts: exactly 14 note fields changed; key sets and every non-note field (roles, tones, maxCells) were identical. The translation was NOT produced under the new context. The manager’s instruction authorizes this human-copy scope reconciliation; machine IDs/codes/config contracts remain out of scope.

## Native semantic audit (audit_notes_not_execution_log)

- `wire.settings.account_policy.label`: 账户选择策略 is the established, neutral noun phrase; 12 Rich cells within 29.
- `wire.settings.account_policy.help`: Preserves both same-provider and same-credential-tier eligibility. 对于所有策略 applies compatible/unexpired successful-request leases to ALL policies; 均优先 preserves priority and 并不保证缓存命中 explicitly denies cache-hit guarantees. Weekly refill is primary; five-hour refill is secondary only within the one-minute tie. Explicit policy overrides legacy selection.
- `wire.settings.account_policy.reset_soonest.label`: 最早重置 is concise and consistently quoted by both help strings; 8 cells.
- `wire.settings.account_policy.reset_soonest.description`: 套餐内额度 preserves included-plan scope. 确定性规则分散选择 preserves deterministic tie/unknown-report spread, and 新配置默认采用此策略 preserves default applicability.
- `wire.settings.account_policy.headroom.label`: 剩余额度最多 is idiomatic and names the criterion, not a machine enum; 12 cells.
- `wire.settings.account_policy.headroom.description`: 缓存的剩余额度 preserves cached reports; 十个百分点 unambiguously means ten percentage points rather than a ten-percent relative difference; spread stays within the best-account threshold.
- `wire.settings.account_policy.fair.label`: 均衡分配 is natural product copy and consistently referenced in legacy help; 8 cells.
- `wire.settings.account_policy.fair.description`: 按会话稳定地分散选择 preserves stable session-based distribution; 而非 preserves the exclusion of ranking cached resets or remaining quota.
- `wire.settings.account_policy.legacy.label`: 旧版按用量选账户 conveys legacy usage-aware selection concisely; 16 cells.
- `wire.settings.account_policy.legacy.help`: 仅在未设置...时 preserves absent-policy-only compatibility behavior. Literal true selects the localized headroom label and false selects fair. Explicit-policy precedence is retained. Resetting this row restores reset-soonest AND clears legacy overrides; no inversion or omitted side effect.
- `wire.settings.account_policy.error.unavailable`: 账户路由不可用 is concise and faithful; it does not invent a cause or recovery action.
- `wire.settings.account_policy.error.invalid_shape`: 仅应包含value字段 preserves the only-field constraint and keeps literal value intact.
- `wire.settings.account_policy.error.invalid_json`: 应为JSON对象 preserves object-type requirement and literal JSON.
- `wire.settings.account_policy.error.write_failed`: Accurately reports save failure and directs the user to local configuration permissions then retry; no credential paths or raw exceptions introduced.

## Mechanical execution

See audit.log, the actual output of the independent scratch validator. Existing core runtime parser, renderer, and check.py argument/plural helpers were exercised directly on all 14 values. Strict JSON duplicate rejection; exact key/context parity; no ICU arguments or plural branches; argument-free render reproduces every target value; Cf=0; stable true/false/value/JSON preserved. Rich label widths: 12, 8, 12, 8, 16 cells, all <=29. Negative controls rejected malformed ICU and an injected argument, and measured 15 CJK glyphs at 30 cells (>29).

The full scanner/check_catalogues suite was NOTRUN; the bounded, namespace-only existing validation helpers were used instead. No target, repo or ledger edit occurred. Source caller excerpts in the translator packet were inspected as contextual evidence only, not substituted for independent language/mechanical checks.

## Explicitly unverified / pending

- Pseudo-localization: NOTRUN.
- Native rendered fit / real app path / design evidence: NOTRUN; required separately after integration. Static Rich cell measurements do not prove wrapping or native layout.
- RTL rendering: NOTRUN / not applicable to this LTR locale. Cf=0 does not certify bidi behavior in another locale.
- Second independent high-visibility settings reader: outside this instance’s verdict; coordinator must collect it.
- Activation, integration, release, ledger registration: NOTRUN. Core docs describe ledger fields but no audit-ledger writer command was found in the consulted core HOWTO/RFC; the UI assembler schema was not used.

## Findings

[]
