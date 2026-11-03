# Independent staged zh-CN audit — wire.errors.sdk_spec

## Verdict and scope

**passed** for catalogue-only linguistic/mechanical contribution; 21/21 keys audited, no BLOCKER/MAJOR/MINOR/NIT findings. Findings: [] . This is first-reader approval only; second independent reader remains a separate gate.

Pinned English/context revision: 0e7cde33ca8008d561e74f4aed774aa5f76ff7d5. Target is the existing translator packet, not a repository change. Scope does NOT authorize modifying SDK producers, historical exception bytes, args/str semantics, ambient locale reads, or localized presentation. No account-policy bundle was touched.

## Actual provenance

Agent i18n-auditor; own session e55f54b6a26e, independent from translator 7fce46e29bee and coder eae0961e4e78. Not job196c3a94b346. Actual completed usage identifies openai/gpt-6.1-sol; fallback relative to initial Anthropic header is retained. No inference of exact upstream fallback chain. No other new reader report read; no translation service used.

## Historical contract (not waived)

All original context notes remain immutable, including English-source-only/no-width-cap and caller-contract-migration restrictions. Independently inspected pinned AST: all 21 wire_errors_sdk_spec envelope producers force locale=en. Also read pinned SessionSpecError(ValueError) documentation preserving one historical string argument and no new exception attributes. No actual SDK exception execution was performed here; generic catalogue renderer checks are not SDK real-path proof. The architect assessment at ~/Documents/LocalOperator/Reports/i18n-sdk-contract-assessment-2026-11-03-3ef6295a5545.md reports no current UI sink; that cross-client sink result is attributed to the architect, not re-derived by this audit. No localized presentation path is invented.

## Per-key native assessment — audit_notes_not_execution_log

- `wire.errors.sdk_spec.agent_identity_conflict`: 不能同时指定 faithfully expresses mutual exclusion, without incorrectly requiring either identity.
- `wire.errors.sdk_spec.approval_callback_gate_required`: 可调用的审批函数 is idiomatic for the callable approval gate; ApprovalPolicy.callback() is unchanged.
- `wire.errors.sdk_spec.approval_declared_tools_empty`: Preserves at-least-one-tool requirement and directs an undeclared bound session to ApprovalPolicy.refuse(), not auto/allow.
- `wire.errors.sdk_spec.approval_handler_wrong_mode`: Preserves callback-only applicability; mode is already repr-quoted opaque data and receives no new wrapper quotes.
- `wire.errors.sdk_spec.approval_tools_wrong_mode`: Preserves declared-only applicability, distinct from callback restriction; opaque mode is unchanged.
- `wire.errors.sdk_spec.attach_fields_unsupported`: Correctly describes viewer-only attach, lists all six excluded categories, assigns those to owner control, and preserves only-resume or own-mode remedy. Configuration profile rendering 配置档案 is intelligible native developer wording, not a machine field rename.
- `wire.errors.sdk_spec.attach_output_unsupported`: Preserves inability to set output enforcement on viewer attachment and the owning runtime authority; 输出约束 conveys enforcement without implying output itself is unsupported.
- `wire.errors.sdk_spec.attach_resume_required`: Preserves required spec.resume and identity-based viewer attachment; 会话ID clarifies the source id without altering behavior.
- `wire.errors.sdk_spec.mode_invalid`: Preserves the exact two allowed mode literals and rejection of the actual opaque mode.
- `wire.errors.sdk_spec.output_format_invalid`: 以下值之一 retains allowed-set semantics and leaves opaque formats untouched.
- `wire.errors.sdk_spec.output_retries_range`: Preserves the inclusive-style source 0-to-5 range, with Arabic digits and exact identifier.
- `wire.errors.sdk_spec.output_retries_requires_format`: 同时设置 makes the dependency clear without reversing it or implying output_format requires retries.
- `wire.errors.sdk_spec.output_schema_requires_format`: Preserves the schema-to-format dependency with the same idiomatic form.
- `wire.errors.sdk_spec.resume_empty`: Preserves session-ID-or-@latest choice and empty-string prohibition; @latest remains exact technical text.
- `wire.errors.sdk_spec.spawn_birth_effort_pair_required`: Creation-time parameter snapshot naturally conveys birth sample. Requires hosting/model alongside birth_effort, states all three or in-process open_session alternative, preserving ability to set initial reasoning effort without a model pair.
- `wire.errors.sdk_spec.spawn_fields_unsupported`: Preserves inability to deliver post-open state, opaque field list, and exact remedy order: in-process open, attach needed state, dispose, spawn same ID; resume restores attachment. Calls dispose explicitly, not a new deletion operation. The alternative is spawning without unsupported fields.
- `wire.errors.sdk_spec.spawn_model_pair_required`: Preserves both-or-neither hosting/model rule, new-runtime creation snapshot, and root-configuration resolution when absent.
- `wire.errors.sdk_spec.spawn_output_unsupported`: Preserves output-enforcement restriction and remedy: own-mode open_session first, then engage. 再让它开始工作 is idiomatic for engage without inventing an API identifier.
- `wire.errors.sdk_spec.tools_empty`: Preserves at-least-one tool and None-as-unrestricted semantics; does not confuse None with an empty collection.
- `wire.errors.sdk_spec.world_root_empty`: Preserves condition when set and requirement for a non-empty path; no invented existence/permission requirement.
- `wire.errors.sdk_spec.yolo_approval_unreachable`: Preserves gate-unreachability effect and alternatives ApprovalPolicy.auto() or dropping yolo. ApprovalPolicy.{approval_mode}() remains structurally intact with opaque argument.

## Mechanical execution

`audit.log` captures actual bounded execution using the existing core parser, render_message, arguments_match and plural helper, with isolated HOME, no inherited CMUX values, and 256 MiB process-group ceiling / 90-second tool limit. All 21 keys: valid JSON with duplicate rejection; exact key/argument-name/type/order parity; no plural/select branches required; no new markup; Cf=0/bidi controls absent. Technical methods, field tokens, mode literal quoting, retry digits, None, @latest and opaque parameters preserved. Opaque interpolation samples include repr-quoted braces, flags/commas/braces, and Chinese/emoji data. Negative controls: malformed ICU and an injected placeholder rejected. All 21 pinned English-forced producers checked by AST, not runtime launch.

## Style, glossary and fit

Native simplified Chinese uses terse developer diagnostics, normal CJK punctuation and no unnecessary 请. Approval prose uses 审批; session uses 会话; team 团队; model 模型; root configuration 根配置. Latin identifiers and code quoting stay technical, not untranslated product prose. Attach viewer wording 附接 versus adding resources 附加 distinguishes the operations rather than contradictory terminology. Source has no maxCells/maxChars caps; no invented cap or measured native layout claim. Style/glossary packet bytes were verified identical to those already read in this same auditor session.

Pseudo-localization: NOT RUN. RTL: not applicable to zh-CN; no native bidi execution performed. Native rendered fit/application flow: NOT RUN, pending separate integrated QA/design. No install/build/scanner/full-suite run. Passing does not mean activated, shipped, globally complete, legacy-compatible under producer changes, or UI no-English requirement fulfilled. Those future claims need actual integrated proof.

## Findings and remaining proof

No actionable translation findings. Preserve existing SDK English args/str/default semantics. Coordinator must collect the independently authored second reader, retain original bundle in canonical tracked evidence, and separately prove any future localized sink and rendered fit. No current sink is claimed by this contribution. No repository/ledger/generation/activation/PR changes were made.
